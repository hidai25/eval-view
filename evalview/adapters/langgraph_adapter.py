"""LangGraph-specific adapter for EvalView.

Supports both LangGraph Cloud API and self-hosted LangGraph agents.
"""

import httpx
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
)

logger = logging.getLogger(__name__)


class LangGraphAdapter(AgentAdapter):
    """Adapter for LangGraph agents.

    Supports:
    - LangGraph invoke endpoint (standard)
    - LangGraph streaming endpoint
    - LangGraph Cloud API

    Response formats:
    - {"messages": [...], "steps": [...]}
    - Streaming: data: {"type": "step", "content": "...", ...}
    """

    def __init__(
        self,
        endpoint: str,
        headers: Optional[Dict[str, str]] = None,
        timeout: float = 30.0,
        streaming: bool = False,
        verbose: bool = False,
        model_config: Optional[Dict[str, Any]] = None,
        assistant_id: Optional[str] = None,
        use_cloud_api: Optional[bool] = None,  # Auto-detect if None
    ):
        self.endpoint = endpoint
        self.headers = headers or {"Content-Type": "application/json"}
        self.timeout = timeout
        self.streaming = streaming
        self.verbose = verbose
        self.model_config = model_config or {}
        self.assistant_id = assistant_id or "agent"  # Default assistant ID

        # Auto-detect Cloud API if not specified
        if use_cloud_api is None:
            # If endpoint contains /threads or port 2024, likely Cloud API
            self.use_cloud_api = "/threads" in endpoint or ":2024" in endpoint
        else:
            self.use_cloud_api = use_cloud_api

    @property
    def name(self) -> str:
        return "langgraph"

    async def execute(
        self, query: str, context: Optional[Dict[str, Any]] = None
    ) -> ExecutionTrace:
        """Execute LangGraph agent and capture trace."""
        context = context or {}
        start_time = datetime.now()

        if self.use_cloud_api:
            return await self._execute_cloud_api(query, context, start_time)
        elif self.streaming:
            return await self._execute_streaming(query, context, start_time)
        else:
            return await self._execute_standard(query, context, start_time)

    async def _execute_cloud_api(
        self, query: str, context: Dict[str, Any], start_time: datetime
    ) -> ExecutionTrace:
        """Execute LangGraph Cloud API (threads + runs pattern)."""
        import asyncio

        assistant_id = context.get("assistant_id", self.assistant_id)

        # Extract base URL (remove /threads if present)
        base_url = self.endpoint.replace("/threads", "").rstrip("/")

        if self.verbose:
            logger.info(f"🚀 Executing LangGraph Cloud API: {query}...")
            logger.debug(f"Assistant ID: {assistant_id}")

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            # Step 1: Create thread
            thread_response = await client.post(
                f"{base_url}/threads",
                json={"metadata": {"test": "evalview"}},
                headers=self.headers,
            )
            thread_response.raise_for_status()
            thread_data = thread_response.json()
            thread_id = thread_data["thread_id"]

            if self.verbose:
                logger.debug(f"Created thread: {thread_id}")

            # Step 2: Create run with streaming
            run_payload = {
                "assistant_id": assistant_id,
                "input": {"messages": [{"role": "user", "content": query}]},
                "stream_mode": ["values", "updates"],
            }

            steps: List[StepTrace] = []
            final_output = ""
            current_event = None  # Track SSE event type
            total_tokens = 0
            input_tokens = 0
            output_tokens = 0

            async with client.stream(
                "POST",
                f"{base_url}/threads/{thread_id}/runs/stream",
                json=run_payload,
                headers=self.headers,
            ) as response:
                response.raise_for_status()

                async for line in response.aiter_lines():
                    if not line:
                        continue

                    # Track event type (SSE format: "event: <type>")
                    if line.startswith("event: "):
                        current_event = line[7:].strip()
                        if self.verbose:
                            logger.debug(f"📡 SSE Event: {current_event}")
                        continue

                    # Parse data line (SSE format: "data: <json>")
                    if line.startswith("data: "):
                        try:
                            data = json.loads(line[6:])

                            if self.verbose:
                                logger.debug(f"📦 Data ({current_event}): {json.dumps(data)[:200]}")

                            # Extract from "values" or "updates" events
                            if current_event in ("values", "updates"):
                                messages_to_process = []

                                # "values" event: data = {"messages": [...]}
                                if isinstance(data, dict) and "messages" in data:
                                    messages_to_process = data["messages"]

                                # "updates" event: data = {"agent": {"messages": [...]}}
                                elif isinstance(data, dict) and "agent" in data:
                                    agent_data = data["agent"]
                                    if isinstance(agent_data, dict) and "messages" in agent_data:
                                        messages_to_process = agent_data["messages"]

                                # Process all messages
                                if messages_to_process and isinstance(messages_to_process, list):
                                    for msg in messages_to_process:
                                        if not isinstance(msg, dict):
                                            continue

                                        # Extract final output from last AI message
                                        content = msg.get("content", "")
                                        if content and isinstance(content, str):
                                            final_output = content

                                        # Extract tool calls from message
                                        tool_calls = msg.get("tool_calls", [])
                                        if tool_calls and isinstance(tool_calls, list):
                                            for tool_call in tool_calls:
                                                if not isinstance(tool_call, dict):
                                                    continue

                                                tool_id = tool_call.get("id", f"tool-{len(steps)}")
                                                tool_name = tool_call.get("name", "unknown")

                                                # Skip if we already captured this tool call
                                                if any(s.step_id == tool_id for s in steps):
                                                    continue

                                                step = StepTrace(
                                                    step_id=tool_id,
                                                    step_name=f"Call {tool_name}",
                                                    tool_name=tool_name,
                                                    parameters=tool_call.get("args", {}),
                                                    output=None,  # Tool results come in separate messages
                                                    success=True,
                                                    metrics=StepMetrics(latency=0.0, cost=0.0),
                                                )
                                                steps.append(step)

                                                if self.verbose:
                                                    logger.debug(f"🔧 Tool call: {tool_name}({tool_call.get('args', {})})")

                                        # Extract usage metadata
                                        usage_meta = msg.get("usage_metadata")
                                        if usage_meta and isinstance(usage_meta, dict):
                                            total_tokens = usage_meta.get("total_tokens", 0)
                                            input_tokens = usage_meta.get("input_tokens", 0)
                                            output_tokens = usage_meta.get("output_tokens", 0)

                                            if self.verbose:
                                                logger.debug(f"💰 Tokens: {input_tokens} in + {output_tokens} out = {total_tokens}")

                                        # Also check response_metadata for usage
                                        response_meta = msg.get("response_metadata", {})
                                        if isinstance(response_meta, dict):
                                            token_usage = response_meta.get("token_usage", {})
                                            if isinstance(token_usage, dict) and token_usage.get("total_tokens"):
                                                total_tokens = token_usage.get("total_tokens", 0)
                                                input_tokens = token_usage.get("prompt_tokens", 0)
                                                output_tokens = token_usage.get("completion_tokens", 0)

                        except json.JSONDecodeError:
                            continue

        end_time = datetime.now()

        # Calculate cost from tokens if available
        from evalview.core.types import TokenUsage
        token_usage_obj = None
        total_cost = 0.0

        if total_tokens > 0:
            token_usage_obj = TokenUsage(
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cached_tokens=0,
            )

            # Calculate cost using model pricing
            model_name = "gpt-4o"  # Default
            input_cost_per_1m = 2.50  # GPT-4o default
            output_cost_per_1m = 10.00  # GPT-4o default

            if self.model_config:
                model_name = self.model_config.get("name", "gpt-4o")
                pricing = self.model_config.get("pricing")

                if pricing:
                    # Custom pricing provided
                    input_cost_per_1m = pricing.get("input_per_1m", 2.50)
                    output_cost_per_1m = pricing.get("output_per_1m", 10.00)

            total_cost = (
                (input_tokens / 1_000_000) * input_cost_per_1m +
                (output_tokens / 1_000_000) * output_cost_per_1m
            )

            if self.verbose:
                logger.info(f"💰 Cost: ${total_cost:.4f} ({input_tokens} in + {output_tokens} out tokens)")

        metrics = self._calculate_metrics(steps, start_time, end_time, total_cost, token_usage_obj)

        if self.verbose:
            logger.info(f"✅ Run completed: {final_output[:50]}...")

        return ExecutionTrace(
            session_id=thread_id,
            start_time=start_time,
            end_time=end_time,
            steps=steps,
            final_output=final_output,
            metrics=metrics,
        )

    async def _execute_standard(
        self, query: str, context: Dict[str, Any], start_time: datetime
    ) -> ExecutionTrace:
        """Execute LangGraph invoke endpoint."""

        # LangGraph typically expects messages format
        payload = {
            "messages": [{"role": "user", "content": query}],
            **context,
        }

        if self.verbose:
            logger.info(f"🚀 Executing LangGraph request: {query}...")
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
            logger.debug(f"📥 Response: {json.dumps(data, indent=2)}")

        end_time = datetime.now()

        # Parse LangGraph response
        steps = self._parse_steps(data)
        final_output = self._extract_output(data)
        metrics = self._calculate_metrics(steps, start_time, end_time)

        return ExecutionTrace(
            session_id=data.get("thread_id", f"langgraph-{start_time.timestamp()}"),
            start_time=start_time,
            end_time=end_time,
            steps=steps,
            final_output=final_output,
            metrics=metrics,
        )

    async def _execute_streaming(
        self, query: str, context: Dict[str, Any], start_time: datetime
    ) -> ExecutionTrace:
        """Execute LangGraph streaming endpoint."""

        payload = {
            "messages": [{"role": "user", "content": query}],
            **context,
        }

        if self.verbose:
            logger.info(f"🚀 Executing LangGraph streaming request: {query}...")

        steps: List[StepTrace] = []
        final_output = ""
        thread_id = None

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            async with client.stream(
                "POST",
                self.endpoint,
                json=payload,
                headers=self.headers,
            ) as response:
                response.raise_for_status()

                async for line in response.aiter_lines():
                    if line.startswith("data: "):
                        try:
                            event = json.loads(line[6:])
                            event_type = event.get("type")

                            if event_type == "step":
                                # Parse step event
                                step = self._parse_step_event(event)
                                if step:
                                    steps.append(step)

                            elif event_type == "message":
                                # Accumulate output
                                content = event.get("content", "")
                                final_output += content

                            elif event_type == "metadata":
                                thread_id = event.get("thread_id")

                        except json.JSONDecodeError:
                            continue

        end_time = datetime.now()
        metrics = self._calculate_metrics(steps, start_time, end_time)

        return ExecutionTrace(
            session_id=thread_id or f"langgraph-{start_time.timestamp()}",
            start_time=start_time,
            end_time=end_time,
            steps=steps,
            final_output=final_output.strip(),
            metrics=metrics,
        )

    def _parse_steps(self, data: Dict[str, Any]) -> List[StepTrace]:
        """Parse steps from LangGraph response."""
        steps = []

        # LangGraph may include intermediate_steps or actions
        if "intermediate_steps" in data:
            for i, step_data in enumerate(data["intermediate_steps"]):
                step = self._create_step_from_intermediate(step_data, i)
                if step:
                    steps.append(step)

        elif "steps" in data:
            for i, step_data in enumerate(data["steps"]):
                step = self._create_step_from_data(step_data, i)
                if step:
                    steps.append(step)

        return steps

    def _create_step_from_intermediate(
        self, step_data: Any, index: int
    ) -> Optional[StepTrace]:
        """Create StepTrace from intermediate_steps format."""
        # intermediate_steps is usually [(AgentAction, output), ...]
        if isinstance(step_data, (list, tuple)) and len(step_data) >= 2:
            action, output = step_data[0], step_data[1]

            # Extract tool info
            if hasattr(action, 'tool'):
                tool_name = action.tool
                parameters = getattr(action, 'tool_input', {})
            else:
                tool_name = str(action)
                parameters = {}

            return StepTrace(
                step_id=f"step-{index}",
                step_name=f"Step {index + 1}",
                tool_name=tool_name,
                parameters=parameters,
                output=output,
                success=True,
                metrics=StepMetrics(latency=0.0, cost=0.0),
            )
        return None

    def _create_step_from_data(
        self, step_data: Dict[str, Any], index: int
    ) -> Optional[StepTrace]:
        """Create StepTrace from steps format."""
        return StepTrace(
            step_id=step_data.get("id", f"step-{index}"),
            step_name=step_data.get("name", f"Step {index + 1}"),
            tool_name=step_data.get("tool"),
            parameters=step_data.get("parameters", {}),
            output=step_data.get("output"),
            success=step_data.get("success", True),
            metrics=StepMetrics(
                latency=step_data.get("latency", 0.0),
                cost=step_data.get("cost", 0.0),
            ),
        )

    def _parse_step_event(self, event: Dict[str, Any]) -> Optional[StepTrace]:
        """Parse streaming step event."""
        content = event.get("content", "")

        # Try to detect tool usage from content
        # This is heuristic-based since streaming format varies
        if "tool" in event or "action" in event:
            return StepTrace(
                step_id=event.get("id", f"step-{datetime.now().timestamp()}"),
                step_name=content[:50] if content else "Step",
                tool_name=event.get("tool"),
                parameters=event.get("parameters", {}),
                output=content,
                success=True,
                metrics=StepMetrics(latency=0.0, cost=0.0),
            )
        return None

    def _extract_output(self, data: Dict[str, Any]) -> str:
        """Extract final output from LangGraph response."""
        # Try different possible locations
        if "messages" in data and isinstance(data["messages"], list):
            # Get last message content
            messages = data["messages"]
            if messages:
                last_msg = messages[-1]
                if isinstance(last_msg, dict):
                    return last_msg.get("content", "")
                elif hasattr(last_msg, 'content'):
                    return last_msg.content

        if "output" in data:
            return str(data["output"])

        if "result" in data:
            return str(data["result"])

        return ""

    def _calculate_metrics(
        self,
        steps: List[StepTrace],
        start_time: datetime,
        end_time: datetime,
        total_cost: float = 0.0,
        token_usage: Optional[Any] = None,
    ) -> ExecutionMetrics:
        """Calculate execution metrics."""
        from evalview.core.types import TokenUsage

        total_latency = (end_time - start_time).total_seconds() * 1000

        # Use provided cost, or sum from steps if not provided
        if total_cost == 0.0:
            total_cost = sum(step.metrics.cost for step in steps)

        return ExecutionMetrics(
            total_cost=total_cost,
            total_latency=total_latency,
            total_tokens=token_usage,
        )

    async def health_check(self) -> bool:
        """Check if LangGraph endpoint is reachable."""
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                # Try a simple request
                response = await client.post(
                    self.endpoint,
                    json={"messages": [{"role": "user", "content": "test"}]},
                    headers=self.headers,
                )
                return response.status_code in [200, 201, 422]
        except Exception:
            return False
