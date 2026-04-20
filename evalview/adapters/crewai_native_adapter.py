"""Native CrewAI adapter — runs crews in-process, no HTTP server needed.

Uses the CrewAI SDK directly: calls ``crew.kickoff()``, captures tool calls
via the event bus (``ToolUsageFinishedEvent``), and extracts per-task results
from ``CrewOutput.tasks_output``.

Usage::

    from crewai import Crew
    from evalview.adapters.crewai_native_adapter import CrewAINativeAdapter

    crew = Crew(agents=[...], tasks=[...])
    adapter = CrewAINativeAdapter(crew=crew)

Or via config::

    # .evalview/config.yaml
    adapter: crewai-native
    crew_module: my_app.crew       # module.path:attribute
    crew_attribute: support_crew
"""
from __future__ import annotations

import importlib
import logging
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set

from evalview.adapters.base import AgentAdapter
from evalview.core.types import (
    ExecutionMetrics,
    ExecutionTrace,
    StepMetrics,
    StepTrace,
    TokenUsage,
)

logger = logging.getLogger(__name__)


class CrewAINativeAdapter(AgentAdapter):
    """Native adapter for CrewAI crews.

    Runs the crew in-process — no HTTP server, no subprocess, no wrapping.
    Captures tool calls via CrewAI's event bus and extracts per-task results
    from ``CrewOutput.tasks_output``.

    Args:
        crew: A ``crewai.Crew`` instance. Either pass this directly or
            set ``crew_module`` + ``crew_attribute`` to load lazily.
        crew_module: Dotted Python module path (e.g. ``"my_app.crew"``).
        crew_attribute: Attribute name on the module (e.g. ``"support_crew"``).
        timeout: Per-query timeout in seconds. CrewAI can be slow.
        verbose: Enable verbose logging.
    """

    def __init__(
        self,
        crew: Optional[Any] = None,
        crew_module: Optional[str] = None,
        crew_attribute: Optional[str] = None,
        timeout: float = 300.0,
        verbose: bool = False,
        **kwargs: Any,
    ):
        self._crew = crew
        self._crew_module = crew_module
        self._crew_attribute = crew_attribute or "crew"
        self._timeout = timeout
        self._verbose = verbose

    @property
    def name(self) -> str:
        return "crewai-native"

    def _resolve_crew(self) -> Any:
        """Resolve the crew instance (lazy import if needed)."""
        if self._crew is not None:
            return self._crew

        if self._crew_module:
            mod = importlib.import_module(self._crew_module)
            self._crew = getattr(mod, self._crew_attribute)
            return self._crew

        raise ValueError(
            "CrewAINativeAdapter requires either a `crew` instance or "
            "`crew_module` + `crew_attribute` to import from."
        )

    async def execute(
        self, query: str, context: Optional[Dict[str, Any]] = None
    ) -> ExecutionTrace:
        """Execute the CrewAI crew and capture a full trace.

        Calls ``crew.kickoff()`` with inputs, captures tool calls from the
        event bus, and extracts per-task results from the output.
        """
        crew = self._resolve_crew()
        session_id = str(uuid.uuid4())[:12]
        start = time.time()
        start_dt = datetime.now(timezone.utc)

        # Prepare inputs
        inputs: Dict[str, Any] = {"query": query}
        if context:
            inputs.update(context)

        # Set up tool call capture via event bus
        tool_calls: List[Dict[str, Any]] = []
        event_handler = None

        try:
            from crewai.events import crewai_event_bus
            from crewai.events import ToolUsageFinishedEvent

            def _on_tool_finished(source: Any, event: Any) -> None:
                tool_calls.append({
                    "tool_name": getattr(event, "tool_name", "unknown"),
                    "args": getattr(event, "tool_args", {}),
                    "output": str(getattr(event, "output", "")),
                    "agent": getattr(event, "agent_role", None),
                    "task": getattr(event, "task_name", None),
                    "started_at": getattr(event, "started_at", None),
                    "finished_at": getattr(event, "finished_at", None),
                    "from_cache": getattr(event, "from_cache", False),
                })

            event_handler = _on_tool_finished
            crewai_event_bus.on(ToolUsageFinishedEvent)(event_handler)
        except ImportError:
            logger.debug("CrewAI event bus not available — tool calls will be inferred from task outputs")

        # Execute crew
        import asyncio
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, lambda: crew.kickoff(inputs=inputs))

        end = time.time()
        end_dt = datetime.now(timezone.utc)
        elapsed_ms = (end - start) * 1000

        # Extract model name from first agent
        model_name: Optional[str] = None
        if hasattr(crew, "agents") and crew.agents:
            first_agent = crew.agents[0]
            if hasattr(first_agent, "llm"):
                llm = first_agent.llm
                if isinstance(llm, str):
                    model_name = llm
                elif hasattr(llm, "model"):
                    model_name = str(llm.model)
                elif hasattr(llm, "model_name"):
                    model_name = str(llm.model_name)

        # Build steps from event bus tool calls (preferred) or task outputs (fallback)
        steps: List[StepTrace] = []

        if tool_calls:
            # Event bus captured detailed tool calls
            for idx, tc in enumerate(tool_calls, 1):
                args = tc["args"]
                if isinstance(args, str):
                    import json
                    try:
                        args = json.loads(args)
                    except (json.JSONDecodeError, ValueError):
                        args = {"raw": args}
                elif not isinstance(args, dict):
                    args = {"value": str(args)}

                # Calculate step latency from event timestamps
                step_latency = 0.0
                if tc.get("started_at") and tc.get("finished_at"):
                    try:
                        delta = tc["finished_at"] - tc["started_at"]
                        step_latency = delta.total_seconds() * 1000
                    except (TypeError, AttributeError):
                        step_latency = elapsed_ms / len(tool_calls)
                else:
                    step_latency = elapsed_ms / len(tool_calls)

                agent_prefix = f"{tc['agent']}_" if tc.get("agent") else ""
                steps.append(StepTrace(
                    step_id=f"tool-{idx}",
                    step_name=f"{agent_prefix}{tc['tool_name']}",
                    tool_name=tc["tool_name"],
                    parameters=args,
                    output=tc["output"],
                    success=True,
                    metrics=StepMetrics(latency=step_latency, cost=0.0),
                ))
        else:
            # Fallback: build steps from task outputs
            task_outputs = getattr(result, "tasks_output", [])
            for idx, task_output in enumerate(task_outputs, 1):
                agent_name = getattr(task_output, "agent", "agent")
                task_name = getattr(task_output, "name", None) or getattr(task_output, "description", f"task_{idx}")
                raw_output = getattr(task_output, "raw", "") or ""

                steps.append(StepTrace(
                    step_id=f"task-{idx}",
                    step_name=f"{agent_name}_{task_name}"[:60],
                    tool_name=f"crew_task_{idx}",
                    parameters={"task": task_name, "agent": agent_name},
                    output=raw_output[:500],
                    success=True,
                    metrics=StepMetrics(
                        latency=elapsed_ms / max(len(task_outputs), 1),
                        cost=0.0,
                    ),
                ))

            # Also check task objects for tool usage counts
            if hasattr(crew, "tasks"):
                for task in crew.tasks:
                    used = getattr(task, "used_tools", 0)
                    errors = getattr(task, "tools_errors", 0)
                    if used > 0 and self._verbose:
                        logger.info(
                            f"Task '{getattr(task, 'description', '?')[:40]}': "
                            f"{used} tool calls, {errors} errors"
                        )

        # Token usage from CrewOutput
        usage = getattr(result, "token_usage", None)
        input_tokens = 0
        output_tokens = 0
        if usage:
            input_tokens = getattr(usage, "prompt_tokens", 0) or 0
            output_tokens = getattr(usage, "completion_tokens", 0) or 0

        total_tokens = input_tokens + output_tokens

        # Cost estimation
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
        final_output = getattr(result, "raw", "") or ""
        if not final_output and hasattr(result, "pydantic") and result.pydantic:
            final_output = str(result.pydantic)

        # Rationale capture. For CrewAI we emit two kinds of events:
        # - tool_choice for each tool call the event bus captured, with
        #   the prior tools as tool_state so cloud can group
        #   (query, prior-tools) decisions.
        # - branch whenever the agent running a step changes — that's
        #   the multi-agent handoff signal the April 2026 reports called
        #   out as "unsolved".
        from evalview.core.rationale import RationaleCollector

        rationale = RationaleCollector()
        last_agent: Optional[str] = None
        for i, (st, tc) in enumerate(zip(steps, tool_calls or [None] * len(steps))):
            agent_name = (tc or {}).get("agent") if tc else None
            if agent_name and agent_name != last_agent:
                rationale.capture_branch(
                    step_id=st.step_id,
                    chosen_branch=agent_name,
                    available_branches=[],
                    state_summary={"handoff_from": last_agent, "step_index": i},
                )
                last_agent = agent_name

            rationale.capture_tool_choice(
                step_id=st.step_id,
                chosen_tool=st.tool_name,
                available_tools=[],
                prompt=query if i == 0 else None,
                tool_state={
                    "agent": agent_name,
                    "prior_tools": [s.tool_name for s in steps[:i]],
                },
            )

        return ExecutionTrace(
            session_id=session_id,
            start_time=start_dt,
            end_time=end_dt,
            steps=steps,
            final_output=final_output,
            metrics=ExecutionMetrics(
                total_cost=total_cost,
                total_latency=elapsed_ms,
                total_tokens=TokenUsage(
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                ),
            ),
            model_id=model_name,
            rationale_events=rationale.events(),
        )

    async def health_check(self) -> bool:
        """Check that the crew can be resolved."""
        try:
            crew = self._resolve_crew()
            return hasattr(crew, "kickoff")
        except Exception:
            return False
