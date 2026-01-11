"""Generic HTTP adapter for REST API agents."""

from datetime import datetime
from typing import Any, Optional, Dict, List, Set
import httpx
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
from evalview.core.pricing import calculate_cost
from evalview.core.tracing import Tracer

logger = logging.getLogger(__name__)


class HTTPAdapter(AgentAdapter):
    """Generic HTTP adapter for REST API agents.

    Security Note:
        SSRF protection is enabled by default. URLs targeting private/internal
        networks will be rejected. Set `allow_private_urls=True` only in trusted
        development environments.
    """

    def __init__(
        self,
        endpoint: str,
        headers: Optional[Dict[str, str]] = None,
        timeout: float = 30.0,
        model_config: Optional[Dict[str, Any]] = None,
        allow_private_urls: bool = False,
        allowed_hosts: Optional[Set[str]] = None,
    ):
        """
        Initialize HTTP adapter.

        Args:
            endpoint: API endpoint URL
            headers: Optional HTTP headers
            timeout: Request timeout in seconds
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
        self.model_config = model_config or {}
        self._last_raw_response = None  # For debug mode

    @property
    def name(self) -> str:
        return "http"

    async def execute(self, query: str, context: Optional[Dict[str, Any]] = None) -> ExecutionTrace:
        """Execute agent via HTTP and capture trace."""
        start_time = datetime.now()

        # Initialize tracer
        tracer = Tracer()

        async with tracer.start_span_async("HTTP Agent", SpanKind.AGENT):
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                api_start = datetime.now()
                response = await client.post(
                    self.endpoint,
                    json={
                        "query": query,
                        "context": context,
                        "enable_tracing": True,
                    },
                    headers={
                        "Content-Type": "application/json",
                        **self.headers,
                    },
                )
                response.raise_for_status()
                data = response.json()
                api_end = datetime.now()
                api_latency = (api_end - api_start).total_seconds() * 1000

            # Record the API call as an LLM span if model info is available
            model_name = self.model_config.get("name", "unknown")
            if model_name != "unknown":
                tokens_data = data.get("tokens") or data.get("metadata", {}).get("tokens", {})
                input_tokens = 0
                output_tokens = 0
                if isinstance(tokens_data, dict):
                    input_tokens = tokens_data.get("input", tokens_data.get("input_tokens", 0))
                    output_tokens = tokens_data.get("output", tokens_data.get("output_tokens", 0))

                tracer.record_llm_call(
                    model=model_name,
                    provider="http",
                    prompt=query,
                    prompt_tokens=input_tokens,
                    completion_tokens=output_tokens,
                    cost=data.get("cost", 0.0),
                    duration_ms=api_latency,
                )

            # Record tool calls from response
            steps_data = data.get("steps") or data.get("tool_calls") or []
            for step_data in steps_data:
                tool_name = (
                    step_data.get("tool")
                    or step_data.get("tool_name")
                    or step_data.get("name")
                    or "unknown"
                )
                parameters = (
                    step_data.get("parameters")
                    or step_data.get("params")
                    or step_data.get("arguments")
                    or {}
                )
                output = step_data.get("output") or step_data.get("result")

                tracer.record_tool_call(
                    tool_name=tool_name,
                    parameters=parameters,
                    result=output,
                    error=step_data.get("error"),
                    duration_ms=step_data.get("latency", 0.0),
                )

        end_time = datetime.now()

        # Store raw response for debug mode
        self._last_raw_response = data

        # Parse response and attach trace context
        trace = self._parse_response(data, start_time, end_time)
        trace.trace_context = tracer.build_trace_context()
        return trace

    def _parse_response(
        self, data: Dict[str, Any], start_time: datetime, end_time: datetime
    ) -> ExecutionTrace:
        """
        Parse HTTP response into ExecutionTrace.

        Supports multiple response formats:
        - Flat: {"response": "...", "cost": 0.05, "tokens": 1500}
        - Nested: {"response": "...", "metadata": {"cost": 0.05, "tokens": {...}}}
        - Steps: {"output": "...", "steps": [...]}

        Override this method in subclasses for custom response formats.
        """
        session_id = data.get("session_id", f"session-{int(start_time.timestamp())}")
        # Support both "steps" and "tool_calls" field names
        steps_data = data.get("steps") or data.get("tool_calls") or []
        steps = self._parse_steps(steps_data)

        # Extract output from various common fields
        final_output = (
            data.get("response")
            or data.get("output")
            or data.get("result")
            or data.get("answer")
            or ""
        )

        # Extract metadata from various locations
        metadata = data.get("metadata", data.get("meta", {}))

        # Calculate latency
        total_latency = (end_time - start_time).total_seconds() * 1000

        # Extract cost (check multiple locations)
        total_cost = (
            data.get("cost")
            or metadata.get("cost")
            or sum(step.metrics.cost for step in steps)
            or 0.0
        )

        # Extract tokens (check multiple locations and formats)
        tokens_data = data.get("tokens") or metadata.get("tokens")
        total_tokens = None

        if tokens_data:
            if isinstance(tokens_data, dict):
                # Nested format: {"input": 100, "output": 500, "cached": 50}
                total_tokens = TokenUsage(
                    input_tokens=tokens_data.get("input", tokens_data.get("input_tokens", 0)),
                    output_tokens=tokens_data.get("output", tokens_data.get("output_tokens", 0)),
                    cached_tokens=tokens_data.get("cached", tokens_data.get("cached_tokens", 0)),
                )
            elif isinstance(tokens_data, int):
                # Simple total: {"tokens": 1500}
                total_tokens = TokenUsage(
                    input_tokens=0,
                    output_tokens=tokens_data,
                    cached_tokens=0,
                )

        # If tokens provided but no cost, calculate it
        if total_tokens and total_tokens.total_tokens > 0 and total_cost == 0.0:
            model_name = self.model_config.get("name", "gpt-4")
            total_cost = calculate_cost(
                model_name=model_name,
                input_tokens=total_tokens.input_tokens,
                output_tokens=total_tokens.output_tokens,
                cached_tokens=total_tokens.cached_tokens,
            )
            logger.info(f"💰 Calculated cost from tokens: ${total_cost:.4f} ({model_name})")

        return ExecutionTrace(
            session_id=session_id,
            start_time=start_time,
            end_time=end_time,
            steps=steps,
            final_output=final_output,
            metrics=ExecutionMetrics(
                total_cost=total_cost,
                total_latency=total_latency,
                total_tokens=total_tokens,
            ),
        )

    def _parse_steps(self, steps_data: List[Dict[str, Any]]) -> List[StepTrace]:
        """Parse steps from response data.

        Supports multiple formats:
        - Steps format: {"tool": "name", "parameters": {...}, "output": ...}
        - Tool calls format: {"name": "tool_name", "arguments": {...}, "result": ...}
        """
        steps = []
        for i, step_data in enumerate(steps_data):
            # Support both "tool"/"tool_name" and "name" for tool name
            tool_name = (
                step_data.get("tool")
                or step_data.get("tool_name")
                or step_data.get("name")
                or "unknown"
            )
            # Support both "parameters"/"params" and "arguments" for parameters
            parameters = (
                step_data.get("parameters")
                or step_data.get("params")
                or step_data.get("arguments")
                or {}
            )
            # Support both "output" and "result" for output
            output = step_data.get("output") or step_data.get("result")

            step = StepTrace(
                step_id=step_data.get("id", f"step-{i}"),
                step_name=step_data.get("step_name") or step_data.get("name") or f"Step {i + 1}",
                tool_name=tool_name,
                parameters=parameters,
                output=output,
                success=step_data.get("success", True),
                error=step_data.get("error"),
                metrics=StepMetrics(
                    latency=step_data.get("latency", 0.0),
                    cost=step_data.get("cost", 0.0),
                    tokens=step_data.get("tokens"),
                ),
            )
            steps.append(step)
        return steps

    async def health_check(self) -> bool:
        """Check if the agent endpoint is reachable."""
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(self.endpoint.replace("/api/", "/health"))
                return response.status_code == 200
        except Exception:
            return False
