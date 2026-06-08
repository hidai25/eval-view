"""Tests for Vercel AI adapter."""

import pytest
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch
import httpx

from evalview.adapters.vercel_ai_adapter import VercelAIAdapter
from evalview.core.types import ExecutionTrace


def async_iter(items):
    """Helper to create async iterator function."""
    async def _iter():
        for item in items:
            yield item
    return _iter()


class MockStreamResponse:
    """Mock response with streaming support."""
    def __init__(self, stream_lines):
        self.stream_lines = stream_lines

    def aiter_lines(self):
        """Return async iterator for lines."""
        return async_iter(self.stream_lines)

    def raise_for_status(self):
        """Simulate status check."""
        pass


class TestVercelAIAdapter:
    """Tests for VercelAIAdapter streaming and tool call parsing."""

    @pytest.mark.asyncio
    async def test_execute_basic(self):
        """Test basic execution with simple text response."""
        adapter = VercelAIAdapter(endpoint="http://test.com/api")

        stream_lines = [
            '0:"Hello "',
            '0:"world"',
        ]

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_response = MockStreamResponse(stream_lines)
            mock_client.post.return_value = mock_response
            mock_client.__aenter__.return_value = mock_client
            mock_client.__aexit__.return_value = None
            mock_client_class.return_value = mock_client

            trace = await adapter.execute("test query")

            assert isinstance(trace, ExecutionTrace)
            assert trace.final_output == "Hello world"
            assert len(trace.steps) == 0
            mock_client.post.assert_called_once()

    @pytest.mark.asyncio
    async def test_execute_with_tool_calls(self):
        """Test execution with tool calls."""
        adapter = VercelAIAdapter(endpoint="http://test.com/api")

        stream_lines = [
            '0:"Searching for "',
            '2:[{"toolName":"search","args":{"query":"Paris"}}]',
            '0:"the capital"',
        ]

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_response = MockStreamResponse(stream_lines)
            mock_client.post.return_value = mock_response
            mock_client.__aenter__.return_value = mock_client
            mock_client.__aexit__.return_value = None
            mock_client_class.return_value = mock_client

            trace = await adapter.execute("What is the capital of France?")

            assert isinstance(trace, ExecutionTrace)
            assert trace.final_output == "Searching for the capital"
            assert len(trace.steps) == 1
            assert trace.steps[0].tool_name == "search"
            assert trace.steps[0].parameters == {"query": "Paris"}
            assert trace.steps[0].step_name == "tool_call_1"

    @pytest.mark.asyncio
    async def test_execute_with_multiple_tool_calls(self):
        """Test execution with multiple sequential tool calls."""
        adapter = VercelAIAdapter(endpoint="http://test.com/api")

        stream_lines = [
            '0:"Searching "',
            '2:[{"toolName":"search","args":{"query":"Paris"}}]',
            '0:" and "',
            '2:[{"toolName":"format","args":{"text":"Paris is capital"}}]',
            '0:"done"',
        ]

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_response = MockStreamResponse(stream_lines)
            mock_client.post.return_value = mock_response
            mock_client.__aenter__.return_value = mock_client
            mock_client.__aexit__.return_value = None
            mock_client_class.return_value = mock_client

            trace = await adapter.execute("What is the capital of France?")

            assert trace.final_output == "Searching  and done"
            assert len(trace.steps) == 2
            assert trace.steps[0].tool_name == "search"
            assert trace.steps[1].tool_name == "format"

    @pytest.mark.asyncio
    async def test_execute_with_array_of_tool_calls(self):
        """Test execution with array of tool calls in one message."""
        adapter = VercelAIAdapter(endpoint="http://test.com/api")

        stream_lines = [
            '0:"Processing "',
            '2:[{"toolName":"tool1","args":{"a":"1"}},{"toolName":"tool2","args":{"b":"2"}}]',
            '0:"complete"',
        ]

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_response = MockStreamResponse(stream_lines)
            mock_client.post.return_value = mock_response
            mock_client.__aenter__.return_value = mock_client
            mock_client.__aexit__.return_value = None
            mock_client_class.return_value = mock_client

            trace = await adapter.execute("test")

            assert trace.final_output == "Processing complete"
            assert len(trace.steps) == 2
            assert trace.steps[0].tool_name == "tool1"
            assert trace.steps[0].parameters == {"a": "1"}
            assert trace.steps[1].tool_name == "tool2"
            assert trace.steps[1].parameters == {"b": "2"}

    @pytest.mark.asyncio
    async def test_execute_ignores_unknown_prefixes(self):
        """Test that unknown message prefixes are ignored."""
        adapter = VercelAIAdapter(endpoint="http://test.com/api")

        stream_lines = [
            '0:"Hello"',
            '1:{"type":"metadata"}',
            '3:{"other":"data"}',
            '0:" world"',
        ]

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_response = MockStreamResponse(stream_lines)
            mock_client.post.return_value = mock_response
            mock_client.__aenter__.return_value = mock_client
            mock_client.__aexit__.return_value = None
            mock_client_class.return_value = mock_client

            trace = await adapter.execute("test")

            assert trace.final_output == "Hello world"
            assert len(trace.steps) == 0

    @pytest.mark.asyncio
    async def test_execute_empty_lines_ignored(self):
        """Test that empty lines are properly ignored."""
        adapter = VercelAIAdapter(endpoint="http://test.com/api")

        stream_lines = [
            '0:"Part1"',
            "",
            "   ",
            '0:"Part2"',
        ]

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_response = MockStreamResponse(stream_lines)
            mock_client.post.return_value = mock_response
            mock_client.__aenter__.return_value = mock_client
            mock_client.__aexit__.return_value = None
            mock_client_class.return_value = mock_client

            trace = await adapter.execute("test")

            assert trace.final_output == "Part1Part2"

    @pytest.mark.asyncio
    async def test_execute_malformed_json_ignored(self):
        """Test that malformed JSON is handled gracefully."""
        adapter = VercelAIAdapter(endpoint="http://test.com/api", verbose=True)

        stream_lines = [
            '0:"Start"',
            '0:"{invalid json"',
            '0:" End"',
        ]

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_response = MockStreamResponse(stream_lines)
            mock_client.post.return_value = mock_response
            mock_client.__aenter__.return_value = mock_client
            mock_client.__aexit__.return_value = None
            mock_client_class.return_value = mock_client

            trace = await adapter.execute("test")

            assert "Start" in trace.final_output
            assert "End" in trace.final_output

    @pytest.mark.asyncio
    async def test_adapter_name(self):
        """Test that adapter returns correct name."""
        adapter = VercelAIAdapter(endpoint="http://test.com/api")
        assert adapter.name == "vercel-ai"

    @pytest.mark.asyncio
    async def test_execute_connection_error(self):
        """Test handling of connection errors."""
        adapter = VercelAIAdapter(endpoint="http://test.com/api")

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.post.side_effect = httpx.ConnectError("Connection refused")
            mock_client.__aenter__.return_value = mock_client
            mock_client.__aexit__.return_value = None
            mock_client_class.return_value = mock_client

            with pytest.raises(Exception) as exc_info:
                await adapter.execute("test")

            assert "Connection" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_execute_timeout_error(self):
        """Test handling of timeout errors."""
        adapter = VercelAIAdapter(endpoint="http://test.com/api", timeout=5.0)

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.post.side_effect = httpx.TimeoutException("Timeout")
            mock_client.__aenter__.return_value = mock_client
            mock_client.__aexit__.return_value = None
            mock_client_class.return_value = mock_client

            with pytest.raises(Exception) as exc_info:
                await adapter.execute("test")

            assert "timeout" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_execute_with_headers(self):
        """Test that custom headers are sent."""
        headers = {"Authorization": "Bearer token123"}
        adapter = VercelAIAdapter(
            endpoint="http://test.com/api",
            headers=headers,
        )

        stream_lines = ['0:"response"']

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_response = MockStreamResponse(stream_lines)
            mock_client.post.return_value = mock_response
            mock_client.__aenter__.return_value = mock_client
            mock_client.__aexit__.return_value = None
            mock_client_class.return_value = mock_client

            trace = await adapter.execute("test")

            call_kwargs = mock_client.post.call_args.kwargs
            assert "Authorization" in call_kwargs["headers"]
            assert call_kwargs["headers"]["Authorization"] == "Bearer token123"

    @pytest.mark.asyncio
    async def test_execute_with_context(self):
        """Test that context is sent in the request."""
        adapter = VercelAIAdapter(endpoint="http://test.com/api")

        stream_lines = ['0:"response"']

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_response = MockStreamResponse(stream_lines)
            mock_client.post.return_value = mock_response
            mock_client.__aenter__.return_value = mock_client
            mock_client.__aexit__.return_value = None
            mock_client_class.return_value = mock_client

            context = {"user_id": "123", "session": "abc"}
            trace = await adapter.execute("test query", context=context)

            call_args = mock_client.post.call_args
            assert call_args.kwargs["json"]["context"] == context

    @pytest.mark.asyncio
    async def test_trace_structure(self):
        """Test that the returned trace has all required fields."""
        adapter = VercelAIAdapter(endpoint="http://test.com/api")

        stream_lines = [
            '0:"Hello "',
            '2:[{"toolName":"greet","args":{"name":"World"}}]',
        ]

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_response = MockStreamResponse(stream_lines)
            mock_client.post.return_value = mock_response
            mock_client.__aenter__.return_value = mock_client
            mock_client.__aexit__.return_value = None
            mock_client_class.return_value = mock_client

            trace = await adapter.execute("test")

            assert trace.session_id
            assert trace.start_time
            assert trace.end_time
            assert trace.final_output == "Hello "
            assert trace.metrics
            assert trace.metrics.total_latency >= 0
            assert trace.metrics.total_cost >= 0
            assert len(trace.steps) == 1
            assert trace.steps[0].step_id == "step-1"
            assert trace.steps[0].success is True

    @pytest.mark.asyncio
    async def test_tool_call_mapping(self):
        """Test that Vercel tool names are correctly mapped."""
        adapter = VercelAIAdapter(endpoint="http://test.com/api")

        stream_lines = [
            '2:[{"toolName":"my_custom_tool","args":{"param1":"value1","param2":123}}]',
        ]

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_response = MockStreamResponse(stream_lines)
            mock_client.post.return_value = mock_response
            mock_client.__aenter__.return_value = mock_client
            mock_client.__aexit__.return_value = None
            mock_client_class.return_value = mock_client

            trace = await adapter.execute("test")

            step = trace.steps[0]
            assert step.tool_name == "my_custom_tool"
            assert step.parameters["param1"] == "value1"
            assert step.parameters["param2"] == 123
