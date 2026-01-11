"""HuggingFace Spaces adapter for EvalView.

Supports Gradio-based agents hosted on HuggingFace Spaces with zero configuration.
Just provide a Space URL and it works.

Examples:
    # All these formats work:
    adapter = HuggingFaceAdapter("username/my-agent")
    adapter = HuggingFaceAdapter("https://huggingface.co/spaces/username/my-agent")
    adapter = HuggingFaceAdapter("https://username-my-agent.hf.space")
"""

import os
import re
import httpx
import json
import asyncio
from datetime import datetime
from typing import Any, Dict, List, Optional, Set
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


class HuggingFaceAdapter(AgentAdapter):
    """Adapter for HuggingFace Spaces (Gradio-based agents).

    Automatically handles:
    - Space URL normalization (username/space → full URL)
    - Gradio API discovery (finds the right endpoint)
    - HF token authentication
    - Async polling for results

    Security Note:
        SSRF protection is enabled by default. Set `allow_private_urls=True`
        only in trusted development environments.

    Example:
        >>> adapter = HuggingFaceAdapter("username/my-chatbot")
        >>> trace = await adapter.execute("What is 2+2?")
    """

    # Common function names for chat/agent interfaces
    CHAT_FUNCTION_NAMES = [
        "chat", "predict", "run", "generate", "respond", "answer",
        "invoke", "call", "process", "ask", "query", "submit",
    ]

    def __init__(
        self,
        endpoint: str,
        headers: Optional[Dict[str, str]] = None,
        timeout: float = 120.0,  # Longer default for AI inference
        hf_token: Optional[str] = None,
        function_name: Optional[str] = None,  # Auto-detect if None
        verbose: bool = False,
        model_config: Optional[Dict[str, Any]] = None,
        allow_private_urls: bool = False,
        allowed_hosts: Optional[Set[str]] = None,
    ):
        """Initialize HuggingFace adapter.

        Args:
            endpoint: Space identifier - can be:
                - "username/space-name"
                - "https://huggingface.co/spaces/username/space-name"
                - "https://username-space-name.hf.space"
            headers: Optional additional HTTP headers
            timeout: Request timeout in seconds (default: 120s for AI)
            hf_token: HuggingFace token (uses HF_TOKEN env var if not set)
            function_name: Gradio function to call (auto-detected if None)
            verbose: Enable verbose logging
            model_config: Model configuration for cost calculation
            allow_private_urls: Allow private/internal URLs
            allowed_hosts: Set of allowed hostnames
        """
        # Set SSRF protection settings
        self.allow_private_urls = allow_private_urls
        self.allowed_hosts = allowed_hosts or {"huggingface.co", "hf.space"}

        # Normalize and validate the Space URL
        self.space_url = self._normalize_space_url(endpoint)
        self.endpoint = self.validate_endpoint(self.space_url)

        # Auth setup - HF_TOKEN is the standard env var
        self.hf_token = hf_token or os.getenv("HF_TOKEN")
        self.headers = headers or {}
        if self.hf_token:
            self.headers["Authorization"] = f"Bearer {self.hf_token}"

        self.timeout = timeout
        self.function_name = function_name
        self.verbose = verbose
        self.model_config = model_config or {}
        self._api_info: Optional[Dict[str, Any]] = None
        self._last_raw_response = None

    @property
    def name(self) -> str:
        return "huggingface"

    def _normalize_space_url(self, endpoint: str) -> str:
        """Convert various Space URL formats to the canonical hf.space URL.

        Handles:
        - username/space-name → https://username-space-name.hf.space
        - huggingface.co/spaces/username/space-name → https://username-space-name.hf.space
        - username-space-name.hf.space → https://username-space-name.hf.space
        """
        endpoint = endpoint.strip()

        # Already a full hf.space URL
        if ".hf.space" in endpoint:
            if not endpoint.startswith("http"):
                endpoint = f"https://{endpoint}"
            return endpoint.rstrip("/")

        # HuggingFace.co URL format
        hf_match = re.match(
            r"https?://huggingface\.co/spaces/([^/]+)/([^/]+)/?",
            endpoint
        )
        if hf_match:
            username, space = hf_match.groups()
            return f"https://{username}-{space}.hf.space"

        # Short format: username/space-name
        if "/" in endpoint and not endpoint.startswith("http"):
            parts = endpoint.split("/")
            if len(parts) == 2:
                username, space = parts
                # Replace underscores with hyphens (HF convention)
                space = space.replace("_", "-")
                return f"https://{username}-{space}.hf.space"

        # Assume it's already a valid URL
        if not endpoint.startswith("http"):
            endpoint = f"https://{endpoint}"

        return endpoint.rstrip("/")

    async def _get_api_info(self) -> Dict[str, Any]:
        """Fetch Gradio API info to discover available endpoints."""
        if self._api_info is not None:
            return self._api_info

        info_url = f"{self.endpoint}/gradio_api/info"

        if self.verbose:
            logger.info(f"🔍 Discovering API endpoints: {info_url}")

        async with httpx.AsyncClient(timeout=30.0) as client:
            try:
                response = await client.get(info_url, headers=self.headers)
                response.raise_for_status()
                self._api_info = response.json()

                if self.verbose:
                    endpoints = list(self._api_info.get("named_endpoints", {}).keys())
                    logger.info(f"📋 Found endpoints: {endpoints}")

                return self._api_info

            except httpx.HTTPStatusError as e:
                if e.response.status_code == 404:
                    raise RuntimeError(
                        f"Not a Gradio Space or API not enabled. "
                        f"Check that {self.endpoint} is a valid Gradio Space "
                        f"with API access enabled."
                    )
                raise

    async def _discover_function(self) -> str:
        """Auto-discover the best function to call for chat/agent interaction."""
        if self.function_name:
            return self.function_name

        api_info = await self._get_api_info()
        named_endpoints = api_info.get("named_endpoints", {})
        unnamed_endpoints = api_info.get("unnamed_endpoints", {})

        # First, try named endpoints with common chat function names
        for name in self.CHAT_FUNCTION_NAMES:
            if f"/{name}" in named_endpoints:
                if self.verbose:
                    logger.info(f"✅ Auto-selected function: /{name}")
                return name

        # Check if any endpoint name contains chat-related keywords
        for endpoint_name in named_endpoints.keys():
            clean_name = endpoint_name.lstrip("/").lower()
            for keyword in self.CHAT_FUNCTION_NAMES:
                if keyword in clean_name:
                    if self.verbose:
                        logger.info(f"✅ Auto-selected function: {endpoint_name}")
                    return endpoint_name.lstrip("/")

        # Fall back to first named endpoint
        if named_endpoints:
            first_endpoint = list(named_endpoints.keys())[0].lstrip("/")
            if self.verbose:
                logger.info(f"⚠️ Using first available endpoint: {first_endpoint}")
            return first_endpoint

        # Last resort: use unnamed endpoint 0
        if unnamed_endpoints:
            if self.verbose:
                logger.info("⚠️ No named endpoints, using endpoint index 0")
            return "0"

        raise RuntimeError(
            f"No API endpoints found in Space. "
            f"Ensure the Gradio app has at least one function exposed."
        )

    async def execute(self, query: str, context: Optional[Dict[str, Any]] = None) -> ExecutionTrace:
        """Execute query on HuggingFace Space and capture trace."""
        context = context or {}
        start_time = datetime.now()

        # Initialize tracer
        tracer = Tracer()

        # Discover the function to call
        fn_name = await self._discover_function()

        if self.verbose:
            logger.info(f"🚀 Calling {self.endpoint}/gradio_api/call/{fn_name}")
            logger.info(f"📝 Query: {query[:100]}...")

        # Build the request payload
        # Gradio expects {"data": [...args...]}
        # Most chat interfaces take (message, history) or just (message,)
        chat_history = context.get("history", [])
        payload = {"data": [query, chat_history]}

        async with tracer.start_span_async("HuggingFace Space", SpanKind.AGENT):
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                # Step 1: Submit the request
                submit_url = f"{self.endpoint}/gradio_api/call/{fn_name}"

                try:
                    submit_response = await client.post(
                        submit_url,
                        json=payload,
                        headers={"Content-Type": "application/json", **self.headers},
                    )
                    submit_response.raise_for_status()
                    submit_data = submit_response.json()

                except httpx.HTTPStatusError as e:
                    # Try simpler payload format (just message, no history)
                    if e.response.status_code == 422:
                        payload = {"data": [query]}
                        submit_response = await client.post(
                            submit_url,
                            json=payload,
                            headers={"Content-Type": "application/json", **self.headers},
                        )
                        submit_response.raise_for_status()
                        submit_data = submit_response.json()
                    else:
                        raise

                event_id = submit_data.get("event_id")
                if not event_id:
                    # Some Gradio versions return result directly
                    trace = self._parse_direct_response(submit_data, start_time, query, tracer)
                    return trace

                if self.verbose:
                    logger.info(f"📨 Got event_id: {event_id}")

                # Step 2: Poll for results (SSE stream)
                result_url = f"{self.endpoint}/gradio_api/call/{fn_name}/{event_id}"

                final_output = ""
                steps: List[StepTrace] = []
                raw_data = None

                async with client.stream("GET", result_url, headers=self.headers) as response:
                    response.raise_for_status()

                    async for line in response.aiter_lines():
                        line = line.strip()
                        if not line:
                            continue

                        if line.startswith("data: "):
                            try:
                                data = json.loads(line[6:])
                                raw_data = data

                                if self.verbose:
                                    logger.debug(f"📦 Received: {json.dumps(data)[:200]}")

                                # Extract output from various formats
                                if isinstance(data, list) and len(data) > 0:
                                    # Format: [response_text, updated_history]
                                    final_output = self._extract_text(data[0])
                                elif isinstance(data, dict):
                                    final_output = self._extract_text(data)

                                # Try to extract tool calls if present
                                extracted_steps = self._extract_tool_calls(data)
                                if extracted_steps:
                                    steps.extend(extracted_steps)

                            except json.JSONDecodeError:
                                # Might be partial streaming data
                                if self.verbose:
                                    logger.debug(f"⚠️ Non-JSON line: {line[:50]}")

                        elif line.startswith("event: "):
                            event_type = line[7:]
                            if self.verbose:
                                logger.debug(f"📡 Event: {event_type}")

            # Record tool spans for extracted steps
            for step in steps:
                tracer.record_tool_call(
                    tool_name=step.tool_name,
                    parameters=step.parameters,
                    result=step.output,
                    error=step.error,
                    duration_ms=step.metrics.latency if step.metrics else 0.0,
                )

        end_time = datetime.now()

        self._last_raw_response = raw_data

        return ExecutionTrace(
            session_id=event_id or f"hf-{int(start_time.timestamp())}",
            start_time=start_time,
            end_time=end_time,
            steps=steps,
            final_output=final_output,
            metrics=self._calculate_metrics(steps, start_time, end_time),
            trace_context=tracer.build_trace_context(),
        )

    def _extract_text(self, data: Any) -> str:
        """Extract text content from various response formats."""
        if isinstance(data, str):
            return data

        if isinstance(data, dict):
            # Common Gradio response fields
            for key in ["value", "text", "response", "output", "message", "content"]:
                if key in data:
                    return self._extract_text(data[key])

            # Chatbot format: {"role": "assistant", "content": "..."}
            if data.get("role") == "assistant":
                return data.get("content", "")

        if isinstance(data, list):
            # Chatbot history format: [[user, bot], [user, bot], ...]
            if data and isinstance(data[-1], (list, tuple)) and len(data[-1]) >= 2:
                return self._extract_text(data[-1][1])
            # Simple list of messages
            if data:
                return self._extract_text(data[-1])

        return str(data) if data else ""

    def _extract_tool_calls(self, data: Any) -> List[StepTrace]:
        """Extract tool calls from response if present."""
        steps = []

        if isinstance(data, dict):
            # Check for tool_calls field
            tool_calls = data.get("tool_calls") or data.get("tools") or data.get("actions")
            if tool_calls and isinstance(tool_calls, list):
                for i, tc in enumerate(tool_calls):
                    if isinstance(tc, dict):
                        steps.append(StepTrace(
                            step_id=tc.get("id", f"tool-{i}"),
                            step_name=tc.get("name", f"Tool {i+1}"),
                            tool_name=tc.get("name") or tc.get("tool") or "unknown",
                            parameters=tc.get("arguments") or tc.get("parameters") or {},
                            output=tc.get("result") or tc.get("output"),
                            success=tc.get("success", True),
                            metrics=StepMetrics(latency=0.0, cost=0.0),
                        ))

            # Check for steps field
            steps_data = data.get("steps") or data.get("intermediate_steps")
            if steps_data and isinstance(steps_data, list):
                for i, step in enumerate(steps_data):
                    if isinstance(step, dict):
                        steps.append(StepTrace(
                            step_id=step.get("id", f"step-{i}"),
                            step_name=step.get("name", f"Step {i+1}"),
                            tool_name=step.get("tool") or step.get("action") or "unknown",
                            parameters=step.get("parameters") or step.get("input") or {},
                            output=step.get("output") or step.get("result"),
                            success=step.get("success", True),
                            metrics=StepMetrics(
                                latency=step.get("latency", 0.0),
                                cost=step.get("cost", 0.0),
                            ),
                        ))

        return steps

    def _parse_direct_response(
        self, data: Dict[str, Any], start_time: datetime, query: str, tracer: Tracer
    ) -> ExecutionTrace:
        """Parse response when Gradio returns result directly (no event_id)."""
        end_time = datetime.now()

        # Extract output
        output_data = data.get("data", data)
        final_output = self._extract_text(output_data)
        steps = self._extract_tool_calls(data)

        # Record tool spans for extracted steps
        for step in steps:
            tracer.record_tool_call(
                tool_name=step.tool_name,
                parameters=step.parameters,
                result=step.output,
                error=step.error,
                duration_ms=step.metrics.latency if step.metrics else 0.0,
            )

        return ExecutionTrace(
            session_id=f"hf-{int(start_time.timestamp())}",
            start_time=start_time,
            end_time=end_time,
            steps=steps,
            final_output=final_output,
            metrics=self._calculate_metrics(steps, start_time, end_time),
            trace_context=tracer.build_trace_context(),
        )

    def _calculate_metrics(
        self,
        steps: List[StepTrace],
        start_time: datetime,
        end_time: datetime,
    ) -> ExecutionMetrics:
        """Calculate execution metrics."""
        total_latency = (end_time - start_time).total_seconds() * 1000
        total_cost = sum(step.metrics.cost for step in steps)

        return ExecutionMetrics(
            total_cost=total_cost,
            total_latency=total_latency,
            total_tokens=None,  # Gradio typically doesn't expose token counts
        )

    async def health_check(self) -> bool:
        """Check if the HuggingFace Space is reachable and awake."""
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                # Check the Gradio API info endpoint
                response = await client.get(
                    f"{self.endpoint}/gradio_api/info",
                    headers=self.headers,
                )
                return response.status_code == 200
        except Exception as e:
            if self.verbose:
                logger.warning(f"Health check failed: {e}")
            return False

    async def wake_space(self) -> bool:
        """Wake up a sleeping Space (free tier Spaces sleep after inactivity).

        Returns:
            True if Space is awake, False if failed to wake
        """
        if self.verbose:
            logger.info(f"☕ Waking up Space: {self.endpoint}")

        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                # Hit the main page to trigger wake
                response = await client.get(self.endpoint, headers=self.headers)

                if response.status_code == 200:
                    # Wait a moment for Space to initialize
                    await asyncio.sleep(2)

                    # Verify it's actually ready
                    return await self.health_check()

                return False

        except Exception as e:
            if self.verbose:
                logger.warning(f"Failed to wake Space: {e}")
            return False
