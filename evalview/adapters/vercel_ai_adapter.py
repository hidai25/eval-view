"""Vercel AI SDK adapter for EvalView.

Supports testing Vercel AI agents by parsing their streaming protocol format.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional, Set
import json
import httpx
import logging

from evalview.adapters.base import AgentAdapter
from evalview.adapters.http_adapter import AgentConnectionError
from evalview.core.types import (
    ExecutionTrace,
    StepTrace,
    StepMetrics,
    ExecutionMetrics,
    SpanKind,
)
from evalview.core.tracing import Tracer

logger = logging.getLogger(__name__)


class VercelAIAdapter(AgentAdapter):
    """Adapter for Vercel AI SDK agent endpoints.

    Connects to an HTTP endpoint running a Vercel AI agent and captures
    streamed responses in the Vercel AI protocol format.

    The adapter parses streaming responses where each line is prefixed with
    a type indicator:
    - "0:" for text chunks
    - "2:" for tool call arrays
    - Other prefixes are ignored

    Args:
        endpoint: URL of the Vercel AI agent endpoint
        headers: Optional HTTP headers (e.g., for authentication)
        timeout: Request timeout in seconds
        model_config: Model configuration with name and optional pricing
        allow_private_urls: Allow requests to private/internal networks
        allowed_hosts: Set of explicitly allowed hostnames
    """

    def __init__(
        self,
        endpoint: str,
        headers: Optional[Dict[str, str]] = None,
        timeout: float = 30.0,
        model_config: Optional[Dict[str, Any]] = None,
        allow_private_urls: bool = False,
        allowed_hosts: Optional[Set[str]] = None,
        verbose: bool = False,
    ):
        self.allow_private_urls = allow_private_urls
        self.allowed_hosts = allowed_hosts
        self.endpoint = self.validate_endpoint(endpoint)
        self.headers = headers or {}
        self.timeout = timeout
        self.model_config = model_config or {}
        self.verbose = verbose

    @property
    def name(self) -> str:
        return "vercel-ai"

    async def execute(self, query: str, context: Optional[Dict[str, Any]] = None) -> ExecutionTrace:
        """Execute agent via Vercel AI endpoint and capture trace."""
        start_time = datetime.now()
        tracer = Tracer()

        final_text = ""
        tool_calls_list: List[Dict[str, Any]] = []

        try:
            async with tracer.start_span_async("Vercel AI Agent", SpanKind.AGENT):
                async with httpx.AsyncClient(timeout=self.timeout) as client:
                    api_start = datetime.now()

                    response = await client.post(
                        self.endpoint,
                        json={
                            "query": query,
                            "context": context or {},
                        },
                        headers={
                            "Content-Type": "application/json",
                            **self.headers,
                        },
                    )

                    response.raise_for_status()

                    async for line in response.aiter_lines():
                        line = line.strip()
                        if not line:
                            continue

                        # Parse Vercel AI streaming protocol
                        if line.startswith("0:"):
                            # Text chunk
                            json_str = line[2:]
                            try:
                                text_fragment = json.loads(json_str)
                                final_text += text_fragment
                            except json.JSONDecodeError:
                                logger.warning(f"Failed to decode text chunk: {json_str}")

                        elif line.startswith("2:"):
                            # Tool calls array
                            json_str = line[2:]
                            try:
                                tool_calls_raw = json.loads(json_str)
                                if isinstance(tool_calls_raw, list):
                                    for tool_call in tool_calls_raw:
                                        # Map Vercel field names to standard names
                                        standardized = {
                                            "name": tool_call.get("toolName", "unknown"),
                                            "arguments": tool_call.get("args", {}),
                                        }
                                        tool_calls_list.append(standardized)
                            except json.JSONDecodeError:
                                logger.warning(f"Failed to decode tool calls: {json_str}")

                        # Ignore other prefixes (1:, 3:, 9:, etc.)

                    api_end = datetime.now()
                    api_latency = (api_end - api_start).total_seconds() * 1000

        except httpx.ConnectError as e:
            error_msg = str(e)
            if "Name or service not known" in error_msg or "nodename nor servname" in error_msg:
                raise AgentConnectionError(
                    "DNS resolution failed. Check that the hostname is correct.",
                    endpoint=self.endpoint,
                    error_type="dns",
                ) from None
            raise AgentConnectionError(
                "Connection refused. Is your agent running?",
                endpoint=self.endpoint,
                error_type="refused",
            ) from None

        except httpx.TimeoutException:
            raise AgentConnectionError(
                f"Request timed out after {self.timeout}s. Try increasing --timeout.",
                endpoint=self.endpoint,
                error_type="timeout",
            ) from None

        except httpx.RemoteProtocolError:
            raise AgentConnectionError(
                "Protocol error. Check that the endpoint URL is correct.",
                endpoint=self.endpoint,
                error_type="protocol",
            ) from None

        except httpx.StreamError:
            raise AgentConnectionError(
                "Connection closed unexpectedly. Check your network and try again.",
                endpoint=self.endpoint,
                error_type="stream",
            ) from None

        except httpx.HTTPStatusError as e:
            raise AgentConnectionError(
                f"HTTP {e.response.status_code}: {e.response.reason_phrase}",
                endpoint=self.endpoint,
                error_type="http",
            ) from None

        # Build steps from tool calls
        steps: List[StepTrace] = []
        for idx, tool_call in enumerate(tool_calls_list, 1):
            steps.append(
                StepTrace(
                    step_id=f"step-{idx}",
                    step_name=f"tool_call_{idx}",
                    tool_name=tool_call.get("name", "unknown"),
                    parameters=tool_call.get("arguments", {}),
                    output=None,
                    success=True,
                    metrics=StepMetrics(
                        latency=api_latency / max(len(tool_calls_list), 1),
                        cost=0.0,
                    ),
                )
            )

        end_time = datetime.now()
        total_latency = (end_time - start_time).total_seconds() * 1000

        # Record tool calls in tracer
        for tool_call in tool_calls_list:
            tracer.record_tool_call(
                tool_name=tool_call.get("name", "unknown"),
                parameters=tool_call.get("arguments", {}),
                result=None,
                duration_ms=api_latency / max(len(tool_calls_list), 1),
            )

        # Create trace
        trace = ExecutionTrace(
            session_id=f"session-vercel-{int(start_time.timestamp())}",
            start_time=start_time,
            end_time=end_time,
            steps=steps,
            final_output=final_text,
            metrics=ExecutionMetrics(
                total_cost=0.0,
                total_latency=total_latency,
                total_tokens=None,
            ),
        )

        trace.trace_context = tracer.build_trace_context()

        if self.verbose:
            logger.info(f"✓ Vercel AI execution complete: {len(steps)} tool calls, output length {len(final_text)}")

        return trace
