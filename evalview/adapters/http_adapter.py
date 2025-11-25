"""Generic HTTP adapter for REST API agents."""

from datetime import datetime
from typing import Any, Optional, Dict, List
import httpx
import logging
from evalview.adapters.base import AgentAdapter
from evalview.core.types import (
    ExecutionTrace,
    StepTrace,
    StepMetrics,
    ExecutionMetrics,
    TokenUsage,
)
from evalview.core.pricing import calculate_cost

logger = logging.getLogger(__name__)


class HTTPAdapter(AgentAdapter):
    """Generic HTTP adapter for REST API agents."""

    def __init__(
        self,
        endpoint: str,
        headers: Optional[Dict[str, str]] = None,
        timeout: float = 30.0,
        model_config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize HTTP adapter.

        Args:
            endpoint: API endpoint URL
            headers: Optional HTTP headers
            timeout: Request timeout in seconds
            model_config: Model configuration with name and optional custom pricing
        """
        self.endpoint = endpoint
        self.headers = headers or {}
        self.timeout = timeout
        self.model_config = model_config or {}

    @property
    def name(self) -> str:
        return "http"

    async def execute(self, query: str, context: Optional[Dict[str, Any]] = None) -> ExecutionTrace:
        """Execute agent via HTTP and capture trace."""
        start_time = datetime.now()

        async with httpx.AsyncClient(timeout=self.timeout) as client:
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

        end_time = datetime.now()

        return self._parse_response(data, start_time, end_time)

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
        steps = self._parse_steps(data.get("steps", []))

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
        """Parse steps from response data."""
        steps = []
        for i, step_data in enumerate(steps_data):
            step = StepTrace(
                step_id=step_data.get("id", f"step-{i}"),
                step_name=step_data.get("name", f"Step {i + 1}"),
                tool_name=step_data.get("tool", step_data.get("tool_name", "unknown")),
                parameters=step_data.get("parameters", step_data.get("params", {})),
                output=step_data.get("output", step_data.get("result")),
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
