"""Anthropic Claude API adapter for EvalView.

Supports testing Anthropic Claude models with tool use capabilities.
"""

import json
from datetime import datetime
from typing import Any, Dict, List, Optional
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

# Pricing per 1M tokens (as of November 2025)
ANTHROPIC_PRICING = {
    # Claude 4.5 family (latest)
    "claude-opus-4-5-20251101": {"input": 5.00, "output": 25.00},
    "claude-sonnet-4-5-20250929": {"input": 3.00, "output": 15.00},
    "claude-haiku-4-5-20251001": {"input": 1.00, "output": 5.00},
    # Claude 4.1
    "claude-opus-4-1-20250805": {"input": 15.00, "output": 75.00},
    # Claude 4 family
    "claude-sonnet-4-20250514": {"input": 3.00, "output": 15.00},
    "claude-opus-4-20250514": {"input": 15.00, "output": 75.00},
    # Legacy Claude 3.5
    "claude-3-5-sonnet-20241022": {"input": 3.00, "output": 15.00},
    "claude-3-5-haiku-20241022": {"input": 1.00, "output": 5.00},
    # Legacy Claude 3
    "claude-3-opus-20240229": {"input": 15.00, "output": 75.00},
    "claude-3-sonnet-20240229": {"input": 3.00, "output": 15.00},
    "claude-3-haiku-20240307": {"input": 0.25, "output": 1.25},
}


class AnthropicAdapter(AgentAdapter):
    """Adapter for Anthropic Claude API with tool use.

    Requires:
    - ANTHROPIC_API_KEY environment variable
    - tools definition for tool use scenarios

    Example:
        >>> adapter = AnthropicAdapter(
        ...     model="claude-3-5-sonnet-20241022",
        ...     tools=[{
        ...         "name": "get_weather",
        ...         "description": "Get weather for a city",
        ...         "input_schema": {
        ...             "type": "object",
        ...             "properties": {"city": {"type": "string"}},
        ...             "required": ["city"]
        ...         }
        ...     }],
        ...     tool_executor=my_tool_executor,
        ... )
    """

    def __init__(
        self,
        model: str = "claude-sonnet-4-5-20250929",
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_executor: Optional[Any] = None,
        system_prompt: Optional[str] = None,
        max_tokens: int = 4096,
        timeout: float = 120.0,
        max_tool_rounds: int = 10,
        verbose: bool = False,
        model_config: Optional[Dict[str, Any]] = None,
    ):
        """Initialize Anthropic adapter.

        Args:
            model: Claude model to use
            tools: List of tool definitions (Anthropic format)
            tool_executor: Callable that executes tools: (name, input) -> result
            system_prompt: Optional system prompt
            max_tokens: Maximum tokens in response
            timeout: Request timeout in seconds
            max_tool_rounds: Maximum tool use iterations
            verbose: Enable verbose logging
            model_config: Additional model configuration
        """
        self.model = model
        self.tools = tools or []
        self.tool_executor = tool_executor
        self.system_prompt = system_prompt
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.max_tool_rounds = max_tool_rounds
        self.verbose = verbose
        self.model_config = model_config or {}

    @property
    def name(self) -> str:
        return "anthropic"

    async def execute(
        self, query: str, context: Optional[Dict[str, Any]] = None
    ) -> ExecutionTrace:
        """Execute Claude with tool use and capture trace."""
        try:
            from anthropic import AsyncAnthropic
        except ImportError:
            raise ImportError(
                "Anthropic package required. Install with: pip install anthropic"
            )

        context = context or {}

        # Allow context to override adapter settings
        model = context.get("model", self.model)
        tools = context.get("tools", self.tools)
        tool_executor = context.get("tool_executor", self.tool_executor)
        system_prompt = context.get("system_prompt", self.system_prompt)

        # Keep original tools with mock_response for internal lookups
        tools_with_mocks = tools

        # Strip mock_response from tools before sending to Anthropic API
        # (Anthropic API doesn't allow extra fields)
        api_tools = []
        for tool in tools:
            clean_tool = {k: v for k, v in tool.items() if k != "mock_response"}
            api_tools.append(clean_tool)

        start_time = datetime.now()
        steps: List[StepTrace] = []
        total_input_tokens = 0
        total_output_tokens = 0

        # Initialize tracer for detailed span capture
        tracer = Tracer()

        if self.verbose:
            logger.info(f"🚀 Executing Anthropic Claude: {query[:50]}...")
            logger.debug(f"Model: {model}, Tools: {len(api_tools)}")

        client = AsyncAnthropic()

        # Build messages
        messages: List[Dict[str, Any]] = [{"role": "user", "content": query}]

        # Agentic loop - keep going until no more tool calls
        final_output = ""
        round_count = 0

        # Start agent-level span
        async with tracer.start_span_async("Agent Execution", SpanKind.AGENT):
            while round_count < self.max_tool_rounds:
                round_count += 1

                # Make API call
                kwargs: Dict[str, Any] = {
                    "model": model,
                    "max_tokens": self.max_tokens,
                    "messages": messages,
                }

                if system_prompt:
                    kwargs["system"] = system_prompt

                if api_tools:
                    kwargs["tools"] = api_tools

                # Track API call timing with LLM span
                api_call_start = datetime.now()
                response = await client.messages.create(**kwargs)
                api_call_end = datetime.now()
                api_call_latency = (api_call_end - api_call_start).total_seconds() * 1000

                # Track token usage for this round
                round_input_tokens = 0
                round_output_tokens = 0
                if hasattr(response, "usage"):
                    round_input_tokens = response.usage.input_tokens
                    round_output_tokens = response.usage.output_tokens
                    total_input_tokens += round_input_tokens
                    total_output_tokens += round_output_tokens

                # Calculate cost for this round
                round_token_usage = TokenUsage(
                    input_tokens=round_input_tokens,
                    output_tokens=round_output_tokens,
                    cached_tokens=0,
                )
                round_cost = self._calculate_cost(model, round_token_usage)

                # Extract text content for LLM span
                llm_completion = ""
                for block in response.content:
                    if block.type == "text":
                        llm_completion += block.text

                # Determine finish reason
                finish_reason = response.stop_reason if hasattr(response, "stop_reason") else None

                # Record LLM call span
                tracer.record_llm_call(
                    model=model,
                    provider="anthropic",
                    prompt=query if round_count == 1 else None,  # Only include initial prompt
                    completion=llm_completion if llm_completion else None,
                    prompt_tokens=round_input_tokens,
                    completion_tokens=round_output_tokens,
                    finish_reason=finish_reason,
                    cost=round_cost,
                    duration_ms=api_call_latency,
                )

                # Check for tool use
                tool_use_blocks = [
                    block for block in response.content if block.type == "tool_use"
                ]

                if not tool_use_blocks:
                    # No tool calls - extract final text response
                    for block in response.content:
                        if block.type == "text":
                            final_output += block.text
                    break

                # Process tool calls - distribute round cost/latency across tools in this round
                num_tools_in_round = len(tool_use_blocks)
                per_tool_latency = api_call_latency / num_tools_in_round
                per_tool_cost = round_cost / num_tools_in_round

                tool_results = []

                for tool_block in tool_use_blocks:
                    tool_name = tool_block.name
                    tool_input = tool_block.input
                    tool_id = tool_block.id

                    if self.verbose:
                        logger.debug(f"🔧 Tool call: {tool_name}({tool_input})")

                    # Execute tool with timing
                    tool_start = datetime.now()
                    tool_result = None
                    tool_error = None

                    if tool_executor:
                        try:
                            # Support both sync and async executors
                            import asyncio
                            import inspect

                            if inspect.iscoroutinefunction(tool_executor):
                                tool_result = await tool_executor(tool_name, tool_input)
                            else:
                                tool_result = await asyncio.to_thread(
                                    tool_executor, tool_name, tool_input
                                )
                        except Exception as e:
                            tool_error = str(e)
                            tool_result = f"Error: {e}"
                    else:
                        # Check for mock_response in tool definition (use original tools with mocks)
                        mock_response = self._get_mock_response(tool_name, tools_with_mocks)
                        if mock_response is not None:
                            tool_result = mock_response
                            if self.verbose:
                                logger.debug(f"📦 Using mock response for {tool_name}")
                        else:
                            tool_result = f"Tool '{tool_name}' executed (no executor provided)"

                    tool_end = datetime.now()
                    tool_duration = (tool_end - tool_start).total_seconds() * 1000

                    # Record tool span
                    tracer.record_tool_call(
                        tool_name=tool_name,
                        parameters=tool_input,
                        result=tool_result if not tool_error else None,
                        error=tool_error,
                        duration_ms=tool_duration,
                    )

                    # Record step with actual API latency/cost for this round
                    step_trace = StepTrace(
                        step_id=tool_id,
                        step_name=tool_name,
                        tool_name=tool_name,
                        parameters=tool_input,
                        output=tool_result if not tool_error else None,
                        error=tool_error,
                        success=tool_error is None,
                        metrics=StepMetrics(latency=per_tool_latency, cost=per_tool_cost),
                    )
                    steps.append(step_trace)

                    # Prepare tool result for next message
                    tool_results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": tool_id,
                            "content": (
                                json.dumps(tool_result)
                                if not isinstance(tool_result, str)
                                else tool_result
                            ),
                        }
                    )

                # Add assistant message with tool use
                messages.append({"role": "assistant", "content": response.content})

                # Add tool results
                messages.append({"role": "user", "content": tool_results})

                # Also capture any text from this response
                for block in response.content:
                    if block.type == "text":
                        final_output += block.text

        end_time = datetime.now()

        # Calculate total metrics
        token_usage = TokenUsage(
            input_tokens=total_input_tokens,
            output_tokens=total_output_tokens,
            cached_tokens=0,
        )

        total_cost = self._calculate_cost(model, token_usage)
        total_latency = (end_time - start_time).total_seconds() * 1000

        # Build trace context from recorded spans
        trace_context = tracer.build_trace_context()

        if self.verbose:
            logger.info(
                f"✅ Claude completed in {total_latency:.0f}ms "
                f"({len(steps)} tool calls, ${total_cost:.4f})"
            )

        # Capture the resolved model ID from the last API response.
        # Anthropic always returns the exact model string in response.model,
        # which may differ from the requested model name if an alias was used.
        # Storing this in the trace allows DiffEngine to detect silent model
        # updates (e.g., when "claude-3-5-sonnet" resolves to a new version).
        resolved_model_id = response.model if hasattr(response, "model") else model

        return ExecutionTrace(
            session_id=f"anthropic-{start_time.timestamp()}",
            start_time=start_time,
            end_time=end_time,
            steps=steps,
            final_output=final_output,
            metrics=ExecutionMetrics(
                total_cost=total_cost,
                total_latency=total_latency,
                total_tokens=token_usage,
            ),
            trace_context=trace_context,
            model_id=resolved_model_id,
            model_provider="anthropic",
        )

    def _get_mock_response(self, tool_name: str, tools: List[Dict[str, Any]]) -> Any:
        """Get mock response for a tool from tool definitions."""
        for tool in tools:
            if tool.get("name") == tool_name:
                return tool.get("mock_response")
        return None

    def _calculate_cost(self, model: str, token_usage: TokenUsage) -> float:
        """Calculate cost based on model pricing."""
        pricing = ANTHROPIC_PRICING.get(model)

        if not pricing:
            # Default to sonnet pricing for unknown models
            pricing = {"input": 3.00, "output": 15.00}

        input_cost = (token_usage.input_tokens / 1_000_000) * pricing["input"]
        output_cost = (token_usage.output_tokens / 1_000_000) * pricing["output"]

        return input_cost + output_cost

    async def health_check(self) -> bool:
        """Check if Anthropic API is accessible."""
        try:
            from anthropic import AsyncAnthropic

            client = AsyncAnthropic()
            # Simple test message with cheapest model
            await client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=10,
                messages=[{"role": "user", "content": "hi"}],
            )
            return True
        except Exception:
            return False
