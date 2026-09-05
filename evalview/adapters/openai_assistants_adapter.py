"""OpenAI Responses API adapter for EvalView.

Formerly the OpenAI Assistants API adapter. OpenAI removed the Assistants
API (threads/runs/assistants) on August 26, 2026; its replacement is the
Responses API for execution plus isolated Conversations with replayed
user/assistant history for multi-turn tests. The adapter keeps its historical registry name
("openai-assistants"), but Assistant IDs require migration to explicit
configuration. Server-side Assistant objects no longer exist — model,
instructions, and tools are supplied per-request from adapter config, or
via a dashboard-managed Prompt object referenced by ``prompt_id``.
"""

import asyncio
import json
import os
from datetime import datetime
from typing import Any, Dict, List, Optional, Union
import logging

from evalview.adapters.base import AgentAdapter
from evalview.core.types import (
    ExecutionTrace,
    StepTrace,
    StepMetrics,
    ExecutionMetrics,
    TokenUsage,
    SpanKind,
)
from evalview.core.tracing import Tracer

logger = logging.getLogger(__name__)

# Output item types that are model narration rather than tool activity.
_NON_TOOL_ITEM_TYPES = {"message", "reasoning", "compaction"}


class OpenAIAssistantsAdapter(AgentAdapter):
    """Adapter for OpenAI agents via the Responses + Conversations APIs.

    Requires:
    - OPENAI_API_KEY environment variable

    Optional configuration:
    - model via model_config (defaults to gpt-4o)
    - instructions: system prompt for the agent
    - prompt_id: id of a dashboard-managed Prompt object (pmpt_...),
      also read from OPENAI_PROMPT_ID or test-case context
    - tools: list of built-in tool names ("code_interpreter",
      "web_search", "file_search") or raw Responses API tool dicts

    The legacy ``assistant_id`` setting requires an explicit replacement: OpenAI
    removed the Assistants API on 2026-08-26, so asst_... ids can no
    longer be resolved. Configure model/instructions/tools (or a
    prompt_id) instead.
    """

    def __init__(
        self,
        assistant_id: Optional[str] = None,
        timeout: float = 120.0,
        verbose: bool = False,
        model_config: Optional[Union[str, Dict[str, Any]]] = None,
        instructions: Optional[str] = None,
        prompt_id: Optional[str] = None,
        tools: Optional[List[Union[str, Dict[str, Any]]]] = None,
    ):
        self.assistant_id = assistant_id  # legacy, unused
        self.timeout = timeout
        self.verbose = verbose
        self.model_config = {"name": model_config} if isinstance(model_config, str) else (model_config or {})
        self.instructions = instructions
        self.prompt_id = prompt_id
        self.tools = tools

    @property
    def name(self) -> str:
        return "openai-assistants"

    async def execute(self, query: str, context: Optional[Dict[str, Any]] = None) -> ExecutionTrace:
        """Execute the agent via the Responses API and capture a trace."""
        try:
            from openai import AsyncOpenAI
        except ImportError:
            raise ImportError("OpenAI package required. Install with: pip install openai")

        from evalview.core.rationale import RationaleCollector

        context = context or {}

        prompt_id = context.get("prompt_id") or self.prompt_id or os.getenv("OPENAI_PROMPT_ID")
        model = self.model_config.get("name") or os.getenv("OPENAI_MODEL")
        legacy_assistant_id = (
            context.get("assistant_id") or self.assistant_id or os.getenv("OPENAI_ASSISTANT_ID")
        )
        if legacy_assistant_id:
            # A model setting existed before migration and does not replace
            # the instructions/tool definitions formerly held by an Assistant.
            if not (prompt_id or self.instructions is not None or self.tools is not None):
                raise ValueError(
                    "assistant_id / OPENAI_ASSISTANT_ID no longer identifies an executable agent: "
                    "the Assistants API shut down on 2026-08-26. Migrate its model, instructions, "
                    "and tools to adapter config or set prompt_id / OPENAI_PROMPT_ID, then "
                    "remove the legacy Assistant ID."
                )
            logger.warning(
                "assistant_id (%s) is ignored: OpenAI removed the Assistants API "
                "on 2026-08-26. Configure model/instructions/tools or prompt_id "
                "instead.",
                legacy_assistant_id,
            )

        if not model and not prompt_id:
            model = "gpt-4o"
        tools = self._build_tools(has_prompt=bool(prompt_id))

        start_time = datetime.now()

        # Initialize tracer for detailed span capture
        tracer = Tracer()

        if self.verbose:
            logger.info(f"🚀 Executing OpenAI agent: {query}...")
            logger.debug(f"Model: {model}")

        client = AsyncOpenAI(timeout=self.timeout, max_retries=0)

        # Close the HTTP client before this execution's event loop ends, even
        # when a request fails or a pending custom function aborts execution.
        async with client, tracer.start_span_async("Agent Execution", SpanKind.AGENT):
            # Each execution is isolated. Shared multi-turn runners supply the
            # prior user/assistant messages, which are replayed exactly once.
            try:
                conversation = await asyncio.wait_for(
                    client.conversations.create(), timeout=self.timeout
                )
            except asyncio.TimeoutError:
                raise TimeoutError(f"Conversation creation exceeded timeout of {self.timeout}s")

            request: Dict[str, Any] = {
                "input": query,
                "conversation": conversation.id,
            }
            history = context.get("conversation_history")
            if history:
                request["input"] = [*history, {"role": "user", "content": query}]
            if model:
                request["model"] = model
            if prompt_id:
                request["prompt"] = {"id": prompt_id}
            if self.instructions is not None:
                request["instructions"] = self.instructions
            if tools or self.tools is not None:
                request["tools"] = tools

            # Responses run synchronously — no run polling loop.
            run_start = datetime.now()
            try:
                response = await asyncio.wait_for(
                    client.responses.create(**request),
                    timeout=self.timeout,
                )
            except asyncio.TimeoutError:
                raise TimeoutError(f"Response exceeded timeout of {self.timeout}s")
            run_end = datetime.now()
            run_duration = (run_end - run_start).total_seconds() * 1000

            if response.status not in (None, "completed"):
                error_msg = f"Response failed with status: {response.status}"
                if getattr(response, "error", None):
                    error_msg += f" - {response.error.message}"
                raise RuntimeError(error_msg)

            # Record LLM call span
            model_name = getattr(response, "model", None) or model or "unknown"
            usage = getattr(response, "usage", None)
            input_tokens = getattr(usage, "input_tokens", 0) or 0
            output_tokens = getattr(usage, "output_tokens", 0) or 0

            # Calculate cost
            token_usage = TokenUsage(input_tokens=input_tokens, output_tokens=output_tokens)
            llm_cost = self._calculate_llm_cost(model_name, token_usage)

            tracer.record_llm_call(
                model=model_name,
                provider="openai",
                prompt=query,
                prompt_tokens=input_tokens,
                completion_tokens=output_tokens,
                finish_reason="completed",
                cost=llm_cost,
                duration_ms=run_duration,
            )

            # Extract steps and record tool spans
            steps = self._extract_steps_with_tracing(response, tracer)
            pending_functions = [
                item.name for item in (response.output or []) if item.type == "function_call"
            ]
            if pending_functions:
                raise RuntimeError(
                    "Custom function calls were requested but not executed: "
                    + ", ".join(pending_functions)
                    + ". This adapter does not execute custom functions or submit their outputs. "
                    "Use the HTTP adapter to evaluate your agent's complete tool-execution loop."
                )

            # Final output: aggregated text of the response's message items
            final_output = getattr(response, "output_text", "") or ""

        end_time = datetime.now()

        # Calculate metrics from response
        metrics = self._calculate_metrics(response, steps, start_time, end_time)

        # Build trace context
        trace_context = tracer.build_trace_context()

        if self.verbose:
            logger.info(f"✅ Agent completed in {metrics.total_latency:.0f}ms")

        # Rationale capture: emit one tool_choice event per tool call so
        # cloud can group identical (query, prior-tools) decisions across
        # runs. Reasoning summaries from the Responses API are not
        # captured here, so rationale_text stays None — the event itself
        # is the signal.
        rationale = RationaleCollector()
        for i, st in enumerate(steps):
            rationale.capture_tool_choice(
                step_id=st.step_id,
                chosen_tool=st.tool_name,
                available_tools=[],
                prompt=query if i == 0 else None,
                tool_state={"prior_tools": [s.tool_name for s in steps[:i]]},
            )

        return ExecutionTrace(
            session_id=conversation.id,
            start_time=start_time,
            end_time=end_time,
            steps=steps,
            final_output=final_output,
            metrics=metrics,
            trace_context=trace_context,
            rationale_events=rationale.events(),
            model_id=model_name,
            model_provider="openai",
        )

    def _build_tools(self, has_prompt: bool = False) -> List[Dict[str, Any]]:
        """Translate configured tools into Responses API tool params.

        Accepts shorthand names or raw tool dicts. Defaults to
        code_interpreter, matching the adapter's historical default —
        except when running a dashboard Prompt object, whose own tool
        configuration would be silently overridden by request-level
        tools, so no defaults are injected then.
        """
        if self.tools is not None:
            configured = self.tools
        elif has_prompt:
            configured = []
        else:
            configured = ["code_interpreter"]

        tools: List[Dict[str, Any]] = []
        for tool in configured:
            if isinstance(tool, dict):
                tools.append(tool)
                continue

            name = str(tool).lower()
            if name in ("code_interpreter", "code"):
                tools.append({"type": "code_interpreter", "container": {"type": "auto"}})
            elif name == "web_search":
                tools.append({"type": "web_search"})
            elif name in ("file_search", "retrieval"):
                vector_store_ids = [
                    vs.strip()
                    for vs in os.getenv("OPENAI_VECTOR_STORE_IDS", "").split(",")
                    if vs.strip()
                ]
                if vector_store_ids:
                    tools.append({"type": "file_search", "vector_store_ids": vector_store_ids})
                else:
                    raise ValueError(
                        "file_search requires vector store ids in the Responses API; "
                        "set OPENAI_VECTOR_STORE_IDS or pass a full tool dict."
                    )
            else:
                raise ValueError(f"Unknown OpenAI tool shorthand: '{tool}'. Use a supported built-in or a Responses API tool dict.")
        return tools

    def _extract_steps(self, response) -> List[StepTrace]:
        """Extract steps from a response's output items."""
        return self._extract_steps_with_tracing(response, None)

    def _extract_steps_with_tracing(self, response, tracer: Optional[Tracer]) -> List[StepTrace]:
        """Extract tool-call steps from response output items.

        The Responses API returns tool activity inline as typed output
        items (function_call, code_interpreter_call, file_search_call,
        ...) instead of the removed Run Steps API. Per-item timing is not
        exposed, so step latency is 0 and cost distribution falls back to
        an even split.
        """
        steps = []

        for item in getattr(response, "output", None) or []:
            item_type = getattr(item, "type", "")

            if item_type in _NON_TOOL_ITEM_TYPES:
                # Narration/message items aren't user-facing tool steps.
                continue

            step_id = getattr(item, "id", None) or getattr(item, "call_id", None) or ""
            success = getattr(item, "status", None) in (None, "completed")
            error = None if success else f"Tool call status: {item.status}"

            if item_type == "function_call":
                success = False
                error = "Custom function requested but not executed by this adapter"
                tool_name = item.name
                step_name = tool_name
                try:
                    arguments = json.loads(item.arguments) if item.arguments else {}
                    parameters = arguments if isinstance(arguments, dict) else {"raw": item.arguments}
                except (ValueError, TypeError):
                    parameters = {"raw": item.arguments}
                # Function outputs live in separate function_call_output
                # items supplied by the caller; a single eval turn has none.
                output = None

            elif item_type == "code_interpreter_call":
                tool_name = "code_interpreter"
                step_name = "Code Interpreter"
                parameters = {"input": item.code}
                output = "\n".join(
                    out.logs
                    for out in (item.outputs or [])
                    if getattr(out, "type", "") == "logs"
                )

            elif item_type == "file_search_call":
                tool_name = "file_search"
                step_name = "File Search"
                parameters = {"queries": list(getattr(item, "queries", []) or [])}
                output = None

            elif item_type == "web_search_call":
                tool_name = "web_search"
                step_name = "Web Search"
                parameters = {}
                output = None

            elif item_type.endswith("_call"):
                # Future/other built-in tools: record generically so the
                # trace stays complete.
                tool_name = item_type[: -len("_call")]
                step_name = tool_name
                parameters = {}
                output = None

            else:
                continue

            step_trace = StepTrace(
                step_id=step_id,
                step_name=step_name,
                tool_name=tool_name,
                parameters=parameters,
                output=output,
                success=success,
                error=error,
                metrics=StepMetrics(latency=0.0, cost=0.0),
            )
            steps.append(step_trace)

            # Record tool span
            if tracer:
                tracer.record_tool_call(
                    tool_name=tool_name,
                    parameters=parameters,
                    result=output,
                    error=error,
                    duration_ms=0.0,
                )

        return steps

    def _calculate_llm_cost(self, model: str, token_usage: TokenUsage) -> float:
        """Calculate cost for OpenAI LLM call."""
        input_tokens = token_usage.input_tokens
        output_tokens = token_usage.output_tokens

        # GPT-4o pricing (per 1M tokens)
        if "gpt-4o" in model:
            return (input_tokens / 1_000_000) * 2.50 + (output_tokens / 1_000_000) * 10.00
        # GPT-4 Turbo pricing
        elif "gpt-4-turbo" in model or "gpt-4-1106" in model:
            return (input_tokens / 1_000_000) * 10.00 + (output_tokens / 1_000_000) * 30.00
        # GPT-4 pricing
        elif "gpt-4" in model:
            return (input_tokens / 1_000_000) * 30.00 + (output_tokens / 1_000_000) * 60.00
        # GPT-3.5 Turbo pricing
        elif "gpt-3.5" in model:
            return (input_tokens / 1_000_000) * 0.50 + (output_tokens / 1_000_000) * 1.50
        return 0.0

    def _calculate_metrics(
        self, response, steps: List[StepTrace], start_time: datetime, end_time: datetime
    ) -> ExecutionMetrics:
        """Calculate execution metrics from the response."""
        total_latency = (end_time - start_time).total_seconds() * 1000

        # Responses report usage as input/output tokens
        token_usage = None
        usage = getattr(response, "usage", None)
        if usage:
            token_usage = TokenUsage(
                input_tokens=getattr(usage, "input_tokens", 0) or 0,
                output_tokens=getattr(usage, "output_tokens", 0) or 0,
                cached_tokens=0,
            )

        # Calculate cost based on model and tokens (2024-2025 pricing)
        total_cost = 0.0
        if token_usage and getattr(response, "model", None):
            model = response.model
            input_tokens = token_usage.input_tokens
            output_tokens = token_usage.output_tokens

            # GPT-4o pricing (per 1M tokens)
            if "gpt-4o" in model:
                total_cost = (input_tokens / 1_000_000) * 2.50 + (output_tokens / 1_000_000) * 10.00
            # GPT-4 Turbo pricing
            elif "gpt-4-turbo" in model or "gpt-4-1106" in model:
                total_cost = (input_tokens / 1_000_000) * 10.00 + (output_tokens / 1_000_000) * 30.00
            # GPT-4 pricing
            elif "gpt-4" in model:
                total_cost = (input_tokens / 1_000_000) * 30.00 + (output_tokens / 1_000_000) * 60.00
            # GPT-3.5 Turbo pricing
            elif "gpt-3.5" in model:
                total_cost = (input_tokens / 1_000_000) * 0.50 + (output_tokens / 1_000_000) * 1.50

        # Distribute cost proportionally based on each step's latency
        if steps and total_cost > 0:
            total_step_latency = sum(s.metrics.latency for s in steps)
            if total_step_latency > 0:
                for step in steps:
                    # Cost proportional to time spent
                    step.metrics.cost = total_cost * (step.metrics.latency / total_step_latency)
            else:
                # Fallback: distribute evenly if no timing data
                per_step_cost = total_cost / len(steps)
                for step in steps:
                    step.metrics.cost = per_step_cost

        return ExecutionMetrics(
            total_cost=total_cost,
            total_latency=total_latency,
            total_tokens=token_usage,
        )

    async def health_check(self) -> bool:
        """Check if OpenAI API is accessible."""
        try:
            from openai import AsyncOpenAI

            client = AsyncOpenAI()
            await client.models.list()
            return True
        except Exception:
            return False
