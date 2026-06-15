"""Tests for the Vercel AI SDK adapter.

Covers both protocols the SDK ships: the v3/v4 Data Stream Protocol
(``0:``/``9:``/``a:``/``d:`` line prefixes) and the v5 SSE protocol
(``data: {"type": ...}`` events).
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
import httpx

from evalview.adapters.vercel_ai_adapter import VercelAIAdapter
from evalview.core.types import ExecutionTrace


# ---------------------------------------------------------------------------
# Streaming mock helpers
# ---------------------------------------------------------------------------


def _aiter(items):
    """Build an async iterator over ``items``."""

    async def _gen():
        for item in items:
            yield item

    return _gen()


class _MockResponse:
    """Stand-in for an httpx streaming response."""

    def __init__(self, lines, raise_for_status_exc=None):
        self._lines = lines
        self._exc = raise_for_status_exc

    def raise_for_status(self):
        if self._exc is not None:
            raise self._exc

    def aiter_lines(self):
        return _aiter(self._lines)


class _StreamCM:
    """Async context manager returned by ``client.stream(...)``."""

    def __init__(self, response):
        self._response = response

    async def __aenter__(self):
        return self._response

    async def __aexit__(self, *exc):
        return False


def _mock_client(stream_lines=None, stream_exc=None, raise_for_status_exc=None):
    """Build a mocked httpx.AsyncClient instance.

    ``client.stream(...)`` is a regular (non-awaited) call returning an async
    context manager, so it is mocked with a ``MagicMock``.
    """
    client = AsyncMock()
    client.__aenter__.return_value = client
    client.__aexit__.return_value = None
    if stream_exc is not None:
        client.stream = MagicMock(side_effect=stream_exc)
    else:
        response = _MockResponse(stream_lines or [], raise_for_status_exc)
        client.stream = MagicMock(return_value=_StreamCM(response))
    return client


def _patch(client):
    """Patch httpx.AsyncClient so instantiation returns ``client``."""
    patcher = patch("httpx.AsyncClient")
    mock_cls = patcher.start()
    mock_cls.return_value = client
    return patcher


class TestVercelAIAdapter:
    """Tests for VercelAIAdapter streaming and protocol parsing."""

    # --- v4 Data Stream Protocol ------------------------------------------

    @pytest.mark.asyncio
    async def test_v4_basic_text(self):
        adapter = VercelAIAdapter(endpoint="http://test.com/api")
        client = _mock_client(['0:"Hello "', '0:"world"'])
        patcher = _patch(client)
        try:
            trace = await adapter.execute("test query")
        finally:
            patcher.stop()

        assert isinstance(trace, ExecutionTrace)
        assert trace.final_output == "Hello world"
        assert len(trace.steps) == 0

    @pytest.mark.asyncio
    async def test_v4_tool_call_and_result(self):
        adapter = VercelAIAdapter(endpoint="http://test.com/api")
        client = _mock_client(
            [
                '0:"Searching "',
                '9:{"toolCallId":"c1","toolName":"search","args":{"query":"Paris"}}',
                'a:{"toolCallId":"c1","result":{"answer":"Paris"}}',
                '0:"done"',
            ]
        )
        patcher = _patch(client)
        try:
            trace = await adapter.execute("capital of France?")
        finally:
            patcher.stop()

        assert trace.final_output == "Searching done"
        assert len(trace.steps) == 1
        step = trace.steps[0]
        assert step.tool_name == "search"
        assert step.parameters == {"query": "Paris"}
        # Tool result is matched back to the call by toolCallId.
        assert step.output == {"answer": "Paris"}
        assert step.step_id == "c1"

    @pytest.mark.asyncio
    async def test_v4_multiple_tool_calls(self):
        adapter = VercelAIAdapter(endpoint="http://test.com/api")
        client = _mock_client(
            [
                '9:{"toolCallId":"c1","toolName":"search","args":{"q":"a"}}',
                '9:{"toolCallId":"c2","toolName":"format","args":{"t":"b"}}',
                'a:{"toolCallId":"c2","result":"formatted"}',
                'a:{"toolCallId":"c1","result":"results"}',
            ]
        )
        patcher = _patch(client)
        try:
            trace = await adapter.execute("test")
        finally:
            patcher.stop()

        assert [s.tool_name for s in trace.steps] == ["search", "format"]
        # Out-of-order results still land on the right call.
        assert trace.steps[0].output == "results"
        assert trace.steps[1].output == "formatted"

    @pytest.mark.asyncio
    async def test_v4_usage_captured(self):
        adapter = VercelAIAdapter(endpoint="http://test.com/api")
        client = _mock_client(
            [
                '0:"hi"',
                'd:{"finishReason":"stop","usage":{"promptTokens":12,"completionTokens":5}}',
            ]
        )
        patcher = _patch(client)
        try:
            trace = await adapter.execute("test")
        finally:
            patcher.stop()

        assert trace.metrics.total_tokens is not None
        assert trace.metrics.total_tokens.input_tokens == 12
        assert trace.metrics.total_tokens.output_tokens == 5

    # --- v5 SSE Protocol --------------------------------------------------

    @pytest.mark.asyncio
    async def test_v5_sse_text(self):
        adapter = VercelAIAdapter(endpoint="http://test.com/api")
        client = _mock_client(
            [
                'data: {"type":"text-delta","delta":"Hello "}',
                'data: {"type":"text-delta","delta":"world"}',
                "data: [DONE]",
            ]
        )
        patcher = _patch(client)
        try:
            trace = await adapter.execute("test")
        finally:
            patcher.stop()

        assert trace.final_output == "Hello world"

    @pytest.mark.asyncio
    async def test_v5_sse_tool_call_and_result(self):
        adapter = VercelAIAdapter(endpoint="http://test.com/api")
        client = _mock_client(
            [
                'data: {"type":"tool-call","toolCallId":"c1","toolName":"weather","input":{"city":"NYC"}}',
                'data: {"type":"tool-result","toolCallId":"c1","output":{"temp":72}}',
                'data: {"type":"finish","totalUsage":{"inputTokens":8,"outputTokens":3}}',
                "data: [DONE]",
            ]
        )
        patcher = _patch(client)
        try:
            trace = await adapter.execute("weather in NYC?")
        finally:
            patcher.stop()

        assert len(trace.steps) == 1
        step = trace.steps[0]
        assert step.tool_name == "weather"
        assert step.parameters == {"city": "NYC"}
        assert step.output == {"temp": 72}
        assert trace.metrics.total_tokens.input_tokens == 8
        assert trace.metrics.total_tokens.output_tokens == 3

    # --- robustness -------------------------------------------------------

    @pytest.mark.asyncio
    async def test_ignores_unknown_prefixes(self):
        adapter = VercelAIAdapter(endpoint="http://test.com/api")
        client = _mock_client(
            [
                '0:"Hello"',
                '1:{"type":"metadata"}',
                '2:[{"some":"data"}]',
                '8:{"id":"msg-1"}',
                '0:" world"',
            ]
        )
        patcher = _patch(client)
        try:
            trace = await adapter.execute("test")
        finally:
            patcher.stop()

        assert trace.final_output == "Hello world"
        assert len(trace.steps) == 0

    @pytest.mark.asyncio
    async def test_empty_lines_ignored(self):
        adapter = VercelAIAdapter(endpoint="http://test.com/api")
        client = _mock_client(['0:"Part1"', "", "   ", '0:"Part2"'])
        patcher = _patch(client)
        try:
            trace = await adapter.execute("test")
        finally:
            patcher.stop()

        assert trace.final_output == "Part1Part2"

    @pytest.mark.asyncio
    async def test_malformed_json_ignored(self):
        adapter = VercelAIAdapter(endpoint="http://test.com/api", verbose=True)
        client = _mock_client(['0:"Start"', "0:{invalid json", '0:" End"'])
        patcher = _patch(client)
        try:
            trace = await adapter.execute("test")
        finally:
            patcher.stop()

        assert trace.final_output == "Start End"

    @pytest.mark.asyncio
    async def test_non_dict_tool_args_wrapped(self):
        adapter = VercelAIAdapter(endpoint="http://test.com/api")
        client = _mock_client(['9:{"toolCallId":"c1","toolName":"echo","args":"plain"}'])
        patcher = _patch(client)
        try:
            trace = await adapter.execute("test")
        finally:
            patcher.stop()

        assert trace.steps[0].parameters == {"value": "plain"}

    # --- request shaping --------------------------------------------------

    @pytest.mark.asyncio
    async def test_request_uses_messages_body(self):
        adapter = VercelAIAdapter(endpoint="http://test.com/api")
        client = _mock_client(['0:"ok"'])
        patcher = _patch(client)
        try:
            await adapter.execute("what is 2+2?")
        finally:
            patcher.stop()

        kwargs = client.stream.call_args.kwargs
        assert kwargs["json"]["messages"] == [{"role": "user", "content": "what is 2+2?"}]

    @pytest.mark.asyncio
    async def test_context_merged_into_body(self):
        adapter = VercelAIAdapter(endpoint="http://test.com/api")
        client = _mock_client(['0:"ok"'])
        patcher = _patch(client)
        try:
            await adapter.execute("test", context={"user_id": "123"})
        finally:
            patcher.stop()

        body = client.stream.call_args.kwargs["json"]
        assert body["user_id"] == "123"
        assert body["messages"][0]["content"] == "test"

    @pytest.mark.asyncio
    async def test_custom_headers_sent(self):
        adapter = VercelAIAdapter(
            endpoint="http://test.com/api",
            headers={"Authorization": "Bearer token123"},
        )
        client = _mock_client(['0:"ok"'])
        patcher = _patch(client)
        try:
            await adapter.execute("test")
        finally:
            patcher.stop()

        headers = client.stream.call_args.kwargs["headers"]
        assert headers["Authorization"] == "Bearer token123"
        assert headers["Content-Type"] == "application/json"

    @pytest.mark.asyncio
    async def test_uses_post_streaming(self):
        adapter = VercelAIAdapter(endpoint="http://test.com/api")
        client = _mock_client(['0:"ok"'])
        patcher = _patch(client)
        try:
            await adapter.execute("test")
        finally:
            patcher.stop()

        # True streaming: client.stream("POST", ...), never a buffered post().
        assert client.stream.call_args.args[0] == "POST"
        client.post.assert_not_called()

    # --- metadata + errors ------------------------------------------------

    @pytest.mark.asyncio
    async def test_adapter_name(self):
        adapter = VercelAIAdapter(endpoint="http://test.com/api")
        assert adapter.name == "vercel-ai"

    @pytest.mark.asyncio
    async def test_trace_structure(self):
        adapter = VercelAIAdapter(endpoint="http://test.com/api")
        client = _mock_client(
            [
                '0:"Hello "',
                '9:{"toolCallId":"c1","toolName":"greet","args":{"name":"World"}}',
            ]
        )
        patcher = _patch(client)
        try:
            trace = await adapter.execute("test")
        finally:
            patcher.stop()

        assert trace.session_id
        assert trace.start_time
        assert trace.end_time
        assert trace.final_output == "Hello "
        assert trace.metrics.total_latency >= 0
        assert trace.metrics.total_cost >= 0
        assert trace.steps[0].step_id == "c1"
        assert trace.steps[0].success is True

    @pytest.mark.asyncio
    async def test_tool_spans_nest_under_agent_span(self):
        """Tool-call spans must be children of the agent span, not orphan roots."""
        adapter = VercelAIAdapter(endpoint="http://test.com/api")
        client = _mock_client(['9:{"toolCallId":"c1","toolName":"search","args":{"q":"x"}}'])
        patcher = _patch(client)
        try:
            trace = await adapter.execute("test")
        finally:
            patcher.stop()

        spans = trace.trace_context.spans
        agent = next(s for s in spans if s.kind == "agent")
        tool_spans = [s for s in spans if s.kind == "tool"]
        assert tool_spans, "expected a tool span"
        assert all(s.parent_span_id == agent.span_id for s in tool_spans)

    @pytest.mark.asyncio
    async def test_connection_error(self):
        adapter = VercelAIAdapter(endpoint="http://test.com/api")
        client = _mock_client(stream_exc=httpx.ConnectError("Connection refused"))
        patcher = _patch(client)
        try:
            with pytest.raises(Exception) as exc_info:
                await adapter.execute("test")
        finally:
            patcher.stop()
        assert "Connection" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_timeout_error(self):
        adapter = VercelAIAdapter(endpoint="http://test.com/api", timeout=5.0)
        client = _mock_client(stream_exc=httpx.TimeoutException("Timeout"))
        patcher = _patch(client)
        try:
            with pytest.raises(Exception) as exc_info:
                await adapter.execute("test")
        finally:
            patcher.stop()
        assert "timeout" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_http_status_error(self):
        adapter = VercelAIAdapter(endpoint="http://test.com/api")
        request = httpx.Request("POST", "http://test.com/api")
        response = httpx.Response(500, request=request)
        status_exc = httpx.HTTPStatusError("err", request=request, response=response)
        client = _mock_client(raise_for_status_exc=status_exc)
        patcher = _patch(client)
        try:
            with pytest.raises(Exception) as exc_info:
                await adapter.execute("test")
        finally:
            patcher.stop()
        assert "500" in str(exc_info.value)
