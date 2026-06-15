"""Vercel AI SDK adapter for EvalView.

Tests Vercel AI SDK agents by streaming their HTTP response and parsing the
two protocols the SDK ships:

* **Data Stream Protocol** (AI SDK v3/v4) — each line is ``<prefix>:<json>``::

    0:"Hello "                                              # text delta
    9:{"toolCallId":"c1","toolName":"search","args":{...}}  # tool call
    a:{"toolCallId":"c1","result":{...}}                    # tool result
    d:{"finishReason":"stop","usage":{...}}                 # finish + usage

* **SSE Protocol** (AI SDK v5) — Server-Sent Events whose ``data:`` payload is a
  JSON object carrying a ``type`` field::

    data: {"type":"text-delta","delta":"Hello "}
    data: {"type":"tool-call","toolCallId":"c1","toolName":"search","input":{...}}
    data: {"type":"tool-result","toolCallId":"c1","output":{...}}
    data: [DONE]

Both are normalized into the standard :class:`ExecutionTrace`.
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
    TokenUsage,
    SpanKind,
)
from evalview.core.tracing import Tracer

logger = logging.getLogger(__name__)


class VercelAIAdapter(AgentAdapter):
    """Adapter for Vercel AI SDK agent endpoints.

    Connects to an HTTP endpoint running a Vercel AI agent, streams the
    response, and captures text, tool calls, tool results, and token usage.
    Both the v3/v4 Data Stream Protocol and the v5 SSE protocol are supported;
    the format is detected per line, so a single adapter handles either server.

    Args:
        endpoint: URL of the Vercel AI agent endpoint (e.g. ``/api/chat``).
        headers: Optional HTTP headers (e.g. for authentication).
        timeout: Request timeout in seconds.
        model_config: Model configuration (name / pricing). Reserved for cost
            estimation; currently unused because the SDK reports usage directly.
        allow_private_urls: Allow requests to private/internal networks (SSRF).
        allowed_hosts: Set of explicitly allowed hostnames (SSRF).
        verbose: Emit per-execution debug logging.
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
        # Steps keyed by toolCallId so tool results can be matched back to the
        # call that produced them. Calls without an id fall back to a counter.
        steps: List[StepTrace] = []
        steps_by_call_id: Dict[str, StepTrace] = {}
        usage: Optional[TokenUsage] = None
        error_message: Optional[str] = None

        # Vercel AI SDK route handlers (useChat / streamText) expect a
        # `messages` array. Extra fields from `context` are merged at the top
        # level, mirroring how the SDK's `body` option is delivered server-side.
        payload: Dict[str, Any] = {"messages": [{"role": "user", "content": query}]}
        if context:
            payload.update(context)

        def add_tool_call(call_id: Optional[str], tool_name: str, args: Any) -> None:
            idx = len(steps) + 1
            step_id = call_id or f"step-{idx}"
            step = StepTrace(
                step_id=step_id,
                step_name=f"tool_call_{idx}",
                tool_name=tool_name or "unknown",
                parameters=args if isinstance(args, dict) else {"value": args},
                output=None,
                success=True,
                metrics=StepMetrics(latency=0.0, cost=0.0),
            )
            steps.append(step)
            if call_id:
                steps_by_call_id[call_id] = step

        def attach_tool_result(call_id: Optional[str], result: Any) -> None:
            step = steps_by_call_id.get(call_id) if call_id else None
            if step is None and steps:
                # No id to match on — attach to the most recent open call.
                step = next((s for s in reversed(steps) if s.output is None), None)
            if step is not None:
                step.output = result

        try:
            async with tracer.start_span_async("Vercel AI Agent", SpanKind.AGENT):
                async with httpx.AsyncClient(timeout=self.timeout) as client:
                    async with client.stream(
                        "POST",
                        self.endpoint,
                        json=payload,
                        headers={"Content-Type": "application/json", **self.headers},
                    ) as response:
                        response.raise_for_status()

                        async for raw_line in response.aiter_lines():
                            line = raw_line.strip()
                            if not line:
                                continue

                            if line.startswith("data:"):
                                # v5 SSE protocol
                                payload_str = line[len("data:") :].strip()
                                if not payload_str or payload_str == "[DONE]":
                                    continue
                                event = self._loads(payload_str)
                                if not isinstance(event, dict):
                                    continue
                                etype = event.get("type")
                                if etype in ("text-delta", "text"):
                                    delta = event.get("delta", event.get("textDelta", ""))
                                    if isinstance(delta, str):
                                        final_text += delta
                                elif etype == "tool-call":
                                    add_tool_call(
                                        event.get("toolCallId"),
                                        event.get("toolName", "unknown"),
                                        event.get("input", event.get("args", {})),
                                    )
                                elif etype == "tool-result":
                                    attach_tool_result(
                                        event.get("toolCallId"),
                                        event.get("output", event.get("result")),
                                    )
                                elif etype == "error":
                                    error_message = str(
                                        event.get("errorText", event.get("error", ""))
                                    )
                                elif etype in ("finish", "finish-step"):
                                    usage = (
                                        self._parse_usage(
                                            event.get("totalUsage", event.get("usage"))
                                        )
                                        or usage
                                    )
                                continue

                            # v3/v4 Data Stream Protocol: "<prefix>:<json>"
                            sep = line.find(":")
                            if sep == -1:
                                continue
                            prefix, body = line[:sep], line[sep + 1 :]

                            if prefix == "0":
                                text_fragment = self._loads(body)
                                if isinstance(text_fragment, str):
                                    final_text += text_fragment
                            elif prefix == "9":
                                call = self._loads(body)
                                if isinstance(call, dict):
                                    add_tool_call(
                                        call.get("toolCallId"),
                                        call.get("toolName", "unknown"),
                                        call.get("args", call.get("input", {})),
                                    )
                            elif prefix == "a":
                                tr = self._loads(body)
                                if isinstance(tr, dict):
                                    attach_tool_result(
                                        tr.get("toolCallId"),
                                        tr.get("result", tr.get("output")),
                                    )
                            elif prefix == "3":
                                err = self._loads(body)
                                error_message = err if isinstance(err, str) else json.dumps(err)
                            elif prefix in ("d", "e"):
                                finish = self._loads(body)
                                if isinstance(finish, dict):
                                    usage = self._parse_usage(finish.get("usage")) or usage
                            # All other prefixes (1:, 2:, 8:, ...) are metadata/data — ignored.

                # Mirror tool calls (with their results) into the tracer while the
                # agent span is still current, so they nest under it in the trace.
                for step in steps:
                    tracer.record_tool_call(
                        tool_name=step.tool_name,
                        parameters=step.parameters,
                        result=step.output,
                        duration_ms=0.0,
                    )

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

        end_time = datetime.now()
        total_latency = (end_time - start_time).total_seconds() * 1000

        trace = ExecutionTrace(
            session_id=f"session-vercel-{int(start_time.timestamp())}",
            start_time=start_time,
            end_time=end_time,
            steps=steps,
            final_output=final_text,
            metrics=ExecutionMetrics(
                total_cost=0.0,
                total_latency=total_latency,
                total_tokens=usage,
            ),
        )

        trace.trace_context = tracer.build_trace_context()

        if error_message:
            logger.warning(f"Vercel AI stream reported an error: {error_message}")
        if self.verbose:
            logger.info(
                f"✓ Vercel AI execution complete: {len(steps)} tool calls, "
                f"output length {len(final_text)}"
            )

        return trace

    @staticmethod
    def _loads(raw: str) -> Any:
        """Parse a JSON fragment, returning None on malformed input."""
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            logger.warning(f"Failed to decode Vercel AI stream fragment: {raw[:120]}")
            return None

    @staticmethod
    def _parse_usage(raw: Any) -> Optional[TokenUsage]:
        """Normalize a usage object from either protocol into TokenUsage."""
        if not isinstance(raw, dict):
            return None
        # v4 data-stream uses promptTokens/completionTokens; v5 SSE uses
        # inputTokens/outputTokens. Accept either spelling.
        input_tokens = raw.get("inputTokens", raw.get("promptTokens", 0)) or 0
        output_tokens = raw.get("outputTokens", raw.get("completionTokens", 0)) or 0
        try:
            return TokenUsage(input_tokens=int(input_tokens), output_tokens=int(output_tokens))
        except (TypeError, ValueError):
            return None
