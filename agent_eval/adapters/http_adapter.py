"""Generic HTTP adapter for REST API agents."""

from datetime import datetime
from typing import Any, Optional, Dict, List
import httpx
from agent_eval.adapters.base import AgentAdapter
from agent_eval.core.types import (
    ExecutionTrace,
    StepTrace,
    StepMetrics,
    ExecutionMetrics,
    TokenUsage,
)
from agent_eval.core.pricing import calculate_cost


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

    async def execute(
        self, query: str, context: Optional[Dict[str, Any]] = None
    ) -> ExecutionTrace:
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

        Override this method in subclasses for custom response formats.
        """
        session_id = data.get("session_id", f"session-{int(start_time.timestamp())}")
        steps = self._parse_steps(data.get("steps", []))
        final_output = data.get("output", data.get("result", ""))

        total_latency = (end_time - start_time).total_seconds() * 1000
        total_cost = data.get("cost", sum(step.metrics.cost for step in steps))
        total_tokens = data.get("tokens", sum(step.metrics.tokens or 0 for step in steps))

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
