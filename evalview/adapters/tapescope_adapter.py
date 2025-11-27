"""Custom adapter for TapeScope streaming API and other streaming agents."""

from datetime import datetime
from typing import Any, Optional, Dict, Set, List
import httpx
import json
import logging
import os
import asyncio
from evalview.adapters.base import AgentAdapter
from evalview.core.types import (
    ExecutionTrace,
    StepTrace,
    StepMetrics,
    ExecutionMetrics,
    TokenUsage,
)
from evalview.core.pricing import calculate_cost

# Set up logging
logger = logging.getLogger(__name__)
# Enable verbose logging with DEBUG=1 environment variable
if os.getenv("DEBUG") == "1":
    logging.basicConfig(
        level=logging.DEBUG, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
else:
    logging.basicConfig(level=logging.WARNING)

# Tool name mappings: map agent-specific tool names to standardized names
# This allows test cases to use generic tool names while agents use their own
TOOL_NAME_MAPPINGS: Dict[str, List[str]] = {
    # TapeScope-specific mappings
    "analyzeStock": ["fetch_stock_data", "analyze_fundamentals", "get_stock_info"],
    "screenStocks": ["screen_stocks", "stock_screener", "filter_stocks"],
    "synthesizeOrchestratorResults": ["summarize_results", "aggregate_results", "synthesize"],
    "getMarketData": ["fetch_market_data", "market_data", "get_market_info"],
    "getNews": ["fetch_news", "get_news", "news_search"],
    # Add more mappings as needed for different agents
}

# Reverse mapping for lookup: standardized name -> agent tool name
_REVERSE_TOOL_MAPPINGS: Dict[str, str] = {}
for agent_tool, standard_names in TOOL_NAME_MAPPINGS.items():
    for standard_name in standard_names:
        _REVERSE_TOOL_MAPPINGS[standard_name.lower()] = agent_tool


def normalize_tool_name(tool_name: str, reverse: bool = False) -> str:
    """
    Normalize a tool name for comparison.

    Args:
        tool_name: The tool name to normalize
        reverse: If True, map from standardized name to agent-specific name.
                 If False, return the tool name as-is (for display).

    Returns:
        Normalized tool name
    """
    if reverse:
        # Map standardized name to agent tool name
        return _REVERSE_TOOL_MAPPINGS.get(tool_name.lower(), tool_name)
    return tool_name


def get_standardized_tool_names(agent_tool_name: str) -> List[str]:
    """
    Get all standardized names that map to an agent-specific tool name.

    Args:
        agent_tool_name: The agent's tool name (e.g., "analyzeStock")

    Returns:
        List of standardized names, or [agent_tool_name] if no mapping exists
    """
    return TOOL_NAME_MAPPINGS.get(agent_tool_name, [agent_tool_name])


class TapeScopeAdapter(AgentAdapter):
    """
    Universal adapter for streaming JSONL APIs.

    Works with any agent that streams responses in JSON Lines format.
    Supports multiple event types and gracefully falls back to plain text.

    Supported formats:
    - JSONL streaming responses ({"type": "...", "data": {...}})
    - Plain text streaming
    - Mixed JSON/text responses
    - Multiple event formats

    Compatible with:
    - TapeScope
    - LangServe streaming endpoints
    - Custom streaming agents
    - Any JSONL-based API

    Security Note:
        SSRF protection is enabled by default. URLs targeting private/internal
        networks will be rejected. Set `allow_private_urls=True` only in trusted
        development environments.
    """

    def __init__(
        self,
        endpoint: str,
        headers: Optional[Dict[str, str]] = None,
        timeout: float = 60.0,
        verbose: bool = False,
        model_config: Optional[Dict[str, Any]] = None,
        allow_private_urls: bool = False,
        allowed_hosts: Optional[Set[str]] = None,
    ):
        """
        Initialize streaming adapter.

        Args:
            endpoint: API endpoint URL (e.g., http://localhost:3000/api/unifiedchat)
            headers: Optional HTTP headers
            timeout: Request timeout in seconds
            verbose: Enable verbose logging (overrides DEBUG env var)
            model_config: Model configuration with name and optional custom pricing
            allow_private_urls: If True, allow requests to private/internal networks
                               (default: False for security)
            allowed_hosts: Optional set of explicitly allowed hostnames
        """
        # Set SSRF protection settings before validation
        self.allow_private_urls = allow_private_urls
        self.allowed_hosts = allowed_hosts

        # Validate endpoint URL for SSRF protection
        self.endpoint = self.validate_endpoint(endpoint)

        self.headers = headers or {}
        self.timeout = timeout
        self.verbose = verbose or os.getenv("DEBUG") == "1"
        self.model_config = model_config or {}

        if self.verbose:
            logger.setLevel(logging.DEBUG)

    @property
    def name(self) -> str:
        return "streaming"

    async def execute(self, query: str, context: Optional[Dict[str, Any]] = None) -> ExecutionTrace:
        """Execute agent via TapeScope API and capture trace."""
        try:
            # Wrap entire execution with timeout to prevent infinite waits
            return await asyncio.wait_for(
                self._execute_internal(query, context), timeout=self.timeout
            )
        except asyncio.TimeoutError:
            logger.error(f"❌ Request timed out after {self.timeout}s")
            # Return trace with timeout error
            return ExecutionTrace(
                session_id=f"timeout-{int(datetime.now().timestamp())}",
                start_time=datetime.now(),
                end_time=datetime.now(),
                steps=[],
                final_output=f"Error: Request timed out after {self.timeout} seconds. "
                f"Backend may be stuck in refinement loops or processing. "
                f"Check backend logs and consider reducing complexity.",
                metrics=ExecutionMetrics(
                    total_cost=0.0,
                    total_latency=self.timeout * 1000,
                    total_tokens=None,
                ),
            )

    async def _execute_internal(
        self, query: str, context: Optional[Dict[str, Any]] = None
    ) -> ExecutionTrace:
        """Internal execution logic (wrapped by timeout)."""
        start_time = datetime.now()

        # Default context if not provided
        if context is None:
            context = {}

        # Prepare request payload (TapeScope expects 'message' not 'query')
        payload = {
            "message": query,
            "prompt": query,
            "route": context.get("route", "conversational"),
            "userId": context.get("userId", "test-user"),
            **context,
        }

        events = []
        steps = []
        final_output = ""
        raw_text = ""  # Fallback for plain text
        total_usage = TokenUsage(input_tokens=0, output_tokens=0, cached_tokens=0)

        if self.verbose:
            logger.info(f"🚀 Executing request: {query[:100]}...")
            logger.debug(f"📤 Payload: {json.dumps(payload, indent=2)}")

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            async with client.stream(
                "POST",
                self.endpoint,
                json=payload,
                headers={
                    "Content-Type": "application/json",
                    **self.headers,
                },
            ) as response:
                response.raise_for_status()

                if self.verbose:
                    logger.info(f"✅ Response status: {response.status_code}")

                # Read JSONL stream line by line
                line_count = 0
                async for line in response.aiter_lines():
                    line_count += 1
                    if not line.strip():
                        continue

                    raw_text += line + "\n"  # Keep raw text as fallback

                    if self.verbose:
                        logger.debug(f"📥 Line {line_count}: {line[:200]}...")

                    # Try to parse as JSON (JSONL format)
                    try:
                        event = json.loads(line)
                        events.append(event)

                        # Parse different event types
                        event_type = event.get("type", "")

                        if self.verbose:
                            logger.debug(f"🔍 Event type: '{event_type}'")

                        if event_type == "tool_call":
                            # Tool execution step
                            tool_data = event.get("data", {})
                            step = StepTrace(
                                step_id=f"tool-{len(steps)}",
                                step_name=tool_data.get("name", "unknown"),
                                tool_name=tool_data.get("name", "unknown"),
                                parameters=tool_data.get("args", {}),
                                output=None,  # Will be filled by tool_result
                                success=True,
                                error=None,
                                metrics=StepMetrics(latency=0.0, cost=0.0, tokens=TokenUsage()),
                            )
                            steps.append(step)

                        elif event_type == "tool_result":
                            # Update last step with result
                            if steps:
                                result_data = event.get("data", {})
                                steps[-1].output = result_data.get("result", "")
                                steps[-1].success = result_data.get("success", True)
                                steps[-1].error = result_data.get("error")

                        elif event_type == "final_message":
                            # Final agent response
                            final_output = event.get("data", {}).get("text", "")

                        elif event_type == "message_complete":
                            # Complete message with clean text (preferred)
                            complete_data = event.get("data", {})
                            if "content" in complete_data:
                                final_output = complete_data["content"]
                            if self.verbose:
                                logger.info(f"✅ Got complete message, length: {len(final_output)}")

                        elif event_type == "token":
                            # Accumulate streaming tokens (fallback if no message_complete)
                            token = event.get("data", {}).get("token", "")
                            if token:
                                final_output += token

                        elif event_type == "error":
                            # Error occurred
                            error_msg = event.get("error", event.get("message", "Unknown error"))
                            if self.verbose:
                                logger.error(f"❌ Error event: {error_msg}")
                            final_output = f"Error: {error_msg}"

                        elif event_type == "step_narration":
                            # Step narration - capture as tool execution
                            narration_data = event.get("data", {})
                            step_name = narration_data.get("text", "").strip()

                            if step_name and not step_name.startswith("Step"):
                                # Create step for narration
                                step = StepTrace(
                                    step_id=f"step-{len(steps)}",
                                    step_name=step_name,
                                    tool_name=narration_data.get("toolName", "unknown"),
                                    parameters={},
                                    output=step_name,
                                    success=True,
                                    error=None,
                                    metrics=StepMetrics(latency=0.0, cost=0.0, tokens=TokenUsage()),
                                )
                                steps.append(step)

                                if self.verbose:
                                    logger.info(f"📝 Step: {step_name}")

                        elif event_type == "usage":
                            # Token usage event - calculate costs
                            usage_data = event.get("data", {})

                            input_tokens = usage_data.get("input_tokens", 0)
                            output_tokens = usage_data.get("output_tokens", 0)
                            cached_tokens = usage_data.get("cached_tokens", 0)

                            # Update total usage
                            total_usage.input_tokens += input_tokens
                            total_usage.output_tokens += output_tokens
                            total_usage.cached_tokens += cached_tokens

                            # Calculate cost using pricing module
                            model_name = self.model_config.get("name", "gpt-5-mini")
                            custom_pricing = self.model_config.get("pricing")

                            if custom_pricing:
                                # Use custom pricing
                                cost = (
                                    (input_tokens / 1_000_000)
                                    * custom_pricing.get("input_per_1m", 0.25)
                                    + (output_tokens / 1_000_000)
                                    * custom_pricing.get("output_per_1m", 2.0)
                                    + (cached_tokens / 1_000_000)
                                    * custom_pricing.get("cached_per_1m", 0.025)
                                )
                            else:
                                # Use standard pricing
                                cost = calculate_cost(
                                    model_name=model_name,
                                    input_tokens=input_tokens,
                                    output_tokens=output_tokens,
                                    cached_tokens=cached_tokens,
                                )

                            # Update last step with token usage and cost
                            if steps:
                                steps[-1].metrics.tokens = TokenUsage(
                                    input_tokens=input_tokens,
                                    output_tokens=output_tokens,
                                    cached_tokens=cached_tokens,
                                )
                                steps[-1].metrics.cost = cost

                            if self.verbose:
                                logger.info(
                                    f"💰 Usage: {input_tokens} in, {output_tokens} out, "
                                    f"{cached_tokens} cached → ${cost:.4f}"
                                )

                        elif event_type in [
                            "start",
                            "status",
                            "thinking",
                            "step_start",
                            "step_complete",
                        ]:
                            # Informational events - just log
                            if self.verbose:
                                logger.debug(f"ℹ️ Info event: {event_type}")

                        else:
                            # Unknown event type - try to extract any text/message
                            if self.verbose:
                                logger.warning(f"⚠️ Unhandled event type: '{event_type}'")

                            # Try to find text in common fields
                            for field in ["text", "message", "content", "data"]:
                                if field in event:
                                    text = event[field]
                                    if isinstance(text, str):
                                        final_output += text
                                    elif isinstance(text, dict) and "text" in text:
                                        final_output += text["text"]

                    except json.JSONDecodeError:
                        # Not JSON - might be plain text streaming
                        if self.verbose:
                            logger.debug(f"⚠️ Not JSON (plain text?): {line[:100]}...")
                        # Accumulate as plain text
                        final_output += line.strip() + " "
                        continue

        end_time = datetime.now()

        # Fallback: if no structured output captured, use raw text
        if not final_output.strip() and raw_text.strip():
            if self.verbose:
                logger.warning("⚠️ No structured output found, using raw text")
            final_output = raw_text.strip()

        if self.verbose:
            logger.info(f"✅ Stream complete: {line_count} lines received")
            logger.info(
                f"📊 Events: {len(events)}, Steps: {len(steps)}, Output length: {len(final_output)}"
            )
            logger.debug(f"📝 Final output preview: {final_output[:300]}...")

        # Calculate metrics
        total_latency = (end_time - start_time).total_seconds() * 1000
        total_cost = sum(step.metrics.cost for step in steps)

        if self.verbose:
            logger.info(f"💰 Total cost: ${total_cost:.4f}")
            logger.info(
                f"🎟️ Total tokens: {total_usage.total_tokens} (in: {total_usage.input_tokens}, out: {total_usage.output_tokens}, cached: {total_usage.cached_tokens})"
            )

        return ExecutionTrace(
            session_id=f"tapescope-{int(start_time.timestamp())}",
            start_time=start_time,
            end_time=end_time,
            steps=steps,
            final_output=final_output.strip() if final_output else "No response",
            metrics=ExecutionMetrics(
                total_cost=total_cost,
                total_latency=total_latency,
                total_tokens=total_usage if total_usage.total_tokens > 0 else None,
            ),
        )

    async def health_check(self) -> bool:
        """Check if the TapeScope endpoint is reachable."""
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(self.endpoint)
                return response.status_code == 200
        except Exception:
            return False
