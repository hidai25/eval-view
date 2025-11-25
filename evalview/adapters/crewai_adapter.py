"""CrewAI-specific adapter for EvalView.

CrewAI agents typically run synchronously and output different formats.
"""

import httpx
import json
from datetime import datetime
from typing import Any, Dict, List, Optional, Set
import logging

from evalview.adapters.base import AgentAdapter
from evalview.core.types import (
    ExecutionTrace,
    StepTrace,
    StepMetrics,
    ExecutionMetrics,
)

logger = logging.getLogger(__name__)


class CrewAIAdapter(AgentAdapter):
    """Adapter for CrewAI agents.

    CrewAI response typically includes:
    - tasks: List of task executions
    - result: Final crew output
    - usage_metrics: Token usage

    Security Note:
        SSRF protection is enabled by default. URLs targeting private/internal
        networks will be rejected. Set `allow_private_urls=True` only in trusted
        development environments.
    """

    def __init__(
        self,
        endpoint: str,
        headers: Optional[Dict[str, str]] = None,
        timeout: float = 120.0,  # CrewAI can be slow
        verbose: bool = False,
        model_config: Optional[Dict[str, Any]] = None,
        allow_private_urls: bool = False,
        allowed_hosts: Optional[Set[str]] = None,
    ):
        # Set SSRF protection settings before validation
        self.allow_private_urls = allow_private_urls
        self.allowed_hosts = allowed_hosts

        # Validate endpoint URL for SSRF protection
        self.endpoint = self.validate_endpoint(endpoint)

        self.headers = headers or {"Content-Type": "application/json"}
        self.timeout = timeout
        self.verbose = verbose
        self.model_config = model_config or {}

    @property
    def name(self) -> str:
        return "crewai"

    async def execute(self, query: str, context: Optional[Dict[str, Any]] = None) -> ExecutionTrace:
        """Execute CrewAI agent and capture trace."""
        context = context or {}
        start_time = datetime.now()

        # CrewAI typically expects inputs
        payload = {"inputs": {"query": query, **context}}

        if self.verbose:
            logger.info(f"🚀 Executing CrewAI request: {query}...")
            logger.debug(f"📤 Payload: {json.dumps(payload, indent=2)}")

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(
                self.endpoint,
                json=payload,
                headers=self.headers,
            )
            response.raise_for_status()
            data = response.json()

        if self.verbose:
            logger.debug(f"📥 Response: {json.dumps(data, indent=2)[:500]}...")

        end_time = datetime.now()

        # Parse CrewAI response
        steps = self._parse_tasks(data)
        final_output = self._extract_output(data)
        metrics = self._calculate_metrics(data, steps, start_time, end_time)

        return ExecutionTrace(
            session_id=data.get("crew_id", f"crewai-{start_time.timestamp()}"),
            start_time=start_time,
            end_time=end_time,
            steps=steps,
            final_output=final_output,
            metrics=metrics,
        )

    def _parse_tasks(self, data: Dict[str, Any]) -> List[StepTrace]:
        """Parse tasks from CrewAI response."""
        steps = []

        # CrewAI includes task execution details
        if "tasks" in data:
            for i, task in enumerate(data["tasks"]):
                step = StepTrace(
                    step_id=task.get("id", f"task-{i}"),
                    step_name=task.get("description", f"Task {i + 1}"),
                    tool_name=task.get("tool"),
                    parameters=task.get("inputs", {}),
                    output=task.get("output", ""),
                    success=task.get("status") == "completed",
                    error=task.get("error"),
                    metrics=StepMetrics(
                        latency=task.get("duration", 0.0),
                        cost=0.0,  # CrewAI doesn't typically expose per-task cost
                        tokens=task.get("tokens"),
                    ),
                )
                steps.append(step)

        # Some CrewAI setups use "agent_executions"
        elif "agent_executions" in data:
            for i, execution in enumerate(data["agent_executions"]):
                step = StepTrace(
                    step_id=f"exec-{i}",
                    step_name=execution.get("agent_name", f"Agent {i + 1}"),
                    tool_name=execution.get("tool_used"),
                    parameters={},
                    output=execution.get("output", ""),
                    success=True,
                    metrics=StepMetrics(latency=0.0, cost=0.0),
                )
                steps.append(step)

        return steps

    def _extract_output(self, data: Dict[str, Any]) -> str:
        """Extract final output from CrewAI response."""
        # Try different possible locations
        if "result" in data:
            result = data["result"]
            if isinstance(result, str):
                return result
            elif isinstance(result, dict):
                return result.get("output", str(result))

        if "output" in data:
            return str(data["output"])

        if "final_output" in data:
            return str(data["final_output"])

        return ""

    def _calculate_metrics(
        self,
        data: Dict[str, Any],
        steps: List[StepTrace],
        start_time: datetime,
        end_time: datetime,
    ) -> ExecutionMetrics:
        """Calculate execution metrics."""
        total_latency = (end_time - start_time).total_seconds() * 1000

        # CrewAI sometimes includes usage_metrics
        usage = data.get("usage_metrics", {})
        total_tokens = usage.get("total_tokens")
        total_cost = usage.get("total_cost", 0.0)

        # If not provided, sum from steps
        if total_cost == 0.0:
            total_cost = sum(step.metrics.cost for step in steps)

        if not total_tokens:
            total_tokens = sum(step.metrics.tokens or 0 for step in steps)
            total_tokens = total_tokens if total_tokens > 0 else None

        return ExecutionMetrics(
            total_cost=total_cost,
            total_latency=total_latency,
            total_tokens=total_tokens,
        )

    async def health_check(self) -> bool:
        """Check if CrewAI endpoint is reachable."""
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.post(
                    self.endpoint,
                    json={"inputs": {"query": "test"}},
                    headers=self.headers,
                )
                return response.status_code in [200, 201, 422]
        except Exception:
            return False
