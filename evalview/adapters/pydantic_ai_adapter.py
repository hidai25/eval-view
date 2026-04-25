"""Native Pydantic AI adapter — runs agents in-process, no HTTP wrapper needed.

Uses the Pydantic AI SDK directly: calls ``agent.run()`` or ``agent.run_sync()``,
extracts tool calls from typed message history (``ToolCallPart``/``ToolReturnPart``),
and captures token usage from ``result.usage()``.

Usage::

    from pydantic_ai import Agent
    from evalview.adapters.pydantic_ai_adapter import PydanticAIAdapter

    agent = Agent("openai:gpt-4o-mini", tools=[my_tool])
    adapter = PydanticAIAdapter(agent=agent)

Or via config::

    # .evalview/config.yaml
    adapter: pydantic-ai
    agent_module: my_app.agent     # module.path:attribute
    agent_attribute: support_agent
"""
from __future__ import annotations

import importlib
import logging
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from evalview.adapters.base import AgentAdapter
from evalview.core.types import (
    ExecutionMetrics,
    ExecutionTrace,
    StepMetrics,
    StepTrace,
    TokenUsage,
)

logger = logging.getLogger(__name__)


class PydanticAIAdapter(AgentAdapter):
    """Native adapter for Pydantic AI agents.

    Runs the agent in-process — no HTTP server, no subprocess, no wrapping.
    Extracts tool calls from typed message history using Pydantic AI's
    ``ModelResponse`` / ``ToolCallPart`` / ``ToolReturnPart`` classes.

    Args:
        agent: A ``pydantic_ai.Agent`` instance. Either pass this directly
            or set ``agent_module`` + ``agent_attribute`` to load lazily.
        agent_module: Dotted Python module path (e.g. ``"my_app.agent"``).
        agent_attribute: Attribute name on the module (e.g. ``"support_agent"``).
        deps: Dependencies to pass to ``agent.run()``.
        model_override: Override the agent's model for this adapter
            (e.g. ``"openai:gpt-4o-mini"``).
        timeout: Per-query timeout in seconds.
    """

    def __init__(
        self,
        agent: Optional[Any] = None,
        agent_module: Optional[str] = None,
        agent_attribute: Optional[str] = None,
        deps: Optional[Any] = None,
        model_override: Optional[str] = None,
        timeout: float = 60.0,
        verbose: bool = False,
        **kwargs: Any,
    ):
        self._agent = agent
        self._agent_module = agent_module
        self._agent_attribute = agent_attribute or "agent"
        self._deps = deps
        self._model_override = model_override
        self._timeout = timeout
        self._verbose = verbose

    @property
    def name(self) -> str:
        return "pydantic-ai"

    def _resolve_agent(self) -> Any:
        """Resolve the agent instance (lazy import if needed)."""
        if self._agent is not None:
            return self._agent

        if self._agent_module:
            mod = importlib.import_module(self._agent_module)
            self._agent = getattr(mod, self._agent_attribute)
            return self._agent

        raise ValueError(
            "PydanticAIAdapter requires either an `agent` instance or "
            "`agent_module` + `agent_attribute` to import from."
        )

    async def execute(
        self, query: str, context: Optional[Dict[str, Any]] = None
    ) -> ExecutionTrace:
        """Execute the Pydantic AI agent and capture a full trace.

        Calls ``agent.run()`` (async), extracts tool calls from the typed
        message history, and returns a structured ``ExecutionTrace``.
        """
        from pydantic_ai.messages import ModelResponse, ModelRequest

        agent = self._resolve_agent()
        session_id = str(uuid.uuid4())[:12]
        start = time.time()
        start_dt = datetime.now(timezone.utc)

        # Build run kwargs
        run_kwargs: Dict[str, Any] = {}
        if self._deps is not None:
            run_kwargs["deps"] = self._deps
        if self._model_override:
            run_kwargs["model"] = self._model_override

        # Pass context as message_history if provided
        message_history: Optional[List[Any]] = None
        if context and "message_history" in context:
            message_history = context["message_history"]
            run_kwargs["message_history"] = message_history

        # Execute
        result = await agent.run(query, **run_kwargs)

        end = time.time()
        end_dt = datetime.now(timezone.utc)
        elapsed_ms = (end - start) * 1000

        # Extract tool calls and returns from typed messages
        steps: List[StepTrace] = []
        model_name: Optional[str] = None
        model_provider: Optional[str] = None
        step_idx = 0

        # Build a map of tool_call_id → return content
        tool_returns: Dict[str, str] = {}
        for msg in result.all_messages():
            if isinstance(msg, ModelRequest):
                for part in msg.parts:
                    if hasattr(part, "tool_call_id") and hasattr(part, "content"):
                        # ToolReturnPart
                        content = part.content
                        if isinstance(content, list):
                            content = str(content)
                        elif not isinstance(content, str):
                            content = str(content)
                        tool_returns[part.tool_call_id] = content

        # Extract tool calls from ModelResponse messages
        for msg in result.all_messages():
            if isinstance(msg, ModelResponse):
                if model_name is None and hasattr(msg, "model_name"):
                    model_name = msg.model_name
                if model_provider is None and hasattr(msg, "provider_name"):
                    model_provider = getattr(msg, "provider_name", None)

                for part in msg.parts:
                    if hasattr(part, "tool_name") and hasattr(part, "args"):
                        # ToolCallPart
                        tool_call_id = getattr(part, "tool_call_id", "")
                        args = part.args
                        if isinstance(args, str):
                            import json
                            try:
                                args = json.loads(args)
                            except (json.JSONDecodeError, ValueError):
                                args = {"raw": args}
                        elif not isinstance(args, dict):
                            args = {"value": str(args)}

                        tool_output = tool_returns.get(tool_call_id, "")

                        step_idx += 1
                        steps.append(StepTrace(
                            step_id=tool_call_id or f"step-{step_idx}",
                            step_name=f"tool_call_{step_idx}",
                            tool_name=part.tool_name,
                            parameters=args,
                            output=tool_output,
                            success=True,
                            metrics=StepMetrics(
                                latency=elapsed_ms / max(step_idx, 1),
                                cost=0.0,
                            ),
                        ))

        # Token usage
        usage = result.usage()
        input_tokens = getattr(usage, "input_tokens", 0) or 0
        output_tokens = getattr(usage, "output_tokens", 0) or 0

        # Cost estimation (rough, based on common pricing)
        from evalview.core.pricing import calculate_cost
        total_cost = calculate_cost(
            model_name or "unknown",
            input_tokens,
            output_tokens,
        )

        # Distribute cost across steps
        if steps and total_cost > 0:
            per_step = total_cost / len(steps)
            for step in steps:
                step.metrics.cost = per_step

        # Final output
        output = result.output
        if not isinstance(output, str):
            output = str(output)

        return ExecutionTrace(
            session_id=session_id,
            start_time=start_dt,
            end_time=end_dt,
            steps=steps,
            final_output=output,
            metrics=ExecutionMetrics(
                total_cost=total_cost,
                total_latency=elapsed_ms,
                total_tokens=TokenUsage(
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                ),
            ),
            model_id=model_name,
            model_provider=model_provider,
        )

    async def health_check(self) -> bool:
        """Check that the agent can be resolved."""
        try:
            self._resolve_agent()
            return True
        except Exception:
            return False
