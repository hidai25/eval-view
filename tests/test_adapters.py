"""Tests for agent adapters."""

import pytest
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch
import httpx

from evalview.adapters.http_adapter import HTTPAdapter
from evalview.core.types import ExecutionTrace


class TestHTTPAdapter:
    """Tests for HTTPAdapter response parsing and execution."""

    @pytest.mark.asyncio
    async def test_execute_basic(self, http_response_flat):
        """Test basic execution with flat response."""
        adapter = HTTPAdapter(endpoint="http://test.com/api")

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_response = MagicMock()  # json() is sync, so use MagicMock
            mock_response.status_code = 200
            mock_response.json.return_value = http_response_flat
            mock_response.raise_for_status = MagicMock()
            mock_client.post.return_value = mock_response
            mock_client.__aenter__.return_value = mock_client
            mock_client.__aexit__.return_value = None
            mock_client_class.return_value = mock_client

            trace = await adapter.execute("test query")

            assert isinstance(trace, ExecutionTrace)
            assert trace.final_output == "Paris is the capital of France."
            assert trace.metrics.total_cost == 0.05
            mock_client.post.assert_called_once()

    @pytest.mark.asyncio
    async def test_parse_flat_response(self, http_response_flat):
        """Test parsing a flat response structure."""
        adapter = HTTPAdapter(endpoint="http://test.com/api")
        start_time = datetime.now()
        end_time = datetime(start_time.year, start_time.month, start_time.day, 12, 0, 3)

        trace = adapter._parse_response(http_response_flat, start_time, end_time)

        assert trace.final_output == "Paris is the capital of France."
        assert trace.metrics.total_cost == 0.05
        assert trace.metrics.total_tokens.output_tokens == 150
        assert len(trace.steps) == 0

    @pytest.mark.asyncio
    async def test_parse_nested_response(self, http_response_nested):
        """Test parsing a response with nested metadata."""
        adapter = HTTPAdapter(endpoint="http://test.com/api")
        start_time = datetime.now()
        end_time = datetime(start_time.year, start_time.month, start_time.day, 12, 0, 3)

        trace = adapter._parse_response(http_response_nested, start_time, end_time)

        assert trace.final_output == "Paris is the capital of France."
        assert trace.metrics.total_cost == 0.05
        assert trace.metrics.total_tokens.input_tokens == 50
        assert trace.metrics.total_tokens.output_tokens == 100
        assert trace.metrics.total_tokens.cached_tokens == 0

    @pytest.mark.asyncio
    async def test_parse_response_with_steps(self, http_response_with_steps):
        """Test parsing a response with detailed steps."""
        adapter = HTTPAdapter(endpoint="http://test.com/api")
        start_time = datetime.now()
        end_time = datetime(start_time.year, start_time.month, start_time.day, 12, 0, 3)

        trace = adapter._parse_response(http_response_with_steps, start_time, end_time)

        assert trace.session_id == "session-123"
        assert trace.final_output == "Paris is the capital of France."
        assert len(trace.steps) == 2

        # Check first step
        step1 = trace.steps[0]
        assert step1.step_id == "step-1"
        assert step1.step_name == "Search"
        assert step1.tool_name == "search"
        assert step1.parameters == {"query": "capital of France"}
        assert step1.success is True
        assert step1.metrics.latency == 1500
        assert step1.metrics.cost == 0.02

        # Check second step
        step2 = trace.steps[1]
        assert step2.step_id == "step-2"
        assert step2.tool_name == "summarize"

    @pytest.mark.asyncio
    async def test_parse_minimal_response(self, http_response_minimal):
        """Test parsing a minimal response with only result field."""
        adapter = HTTPAdapter(endpoint="http://test.com/api")
        start_time = datetime.now()
        end_time = datetime(start_time.year, start_time.month, start_time.day, 12, 0, 1)

        trace = adapter._parse_response(http_response_minimal, start_time, end_time)

        assert trace.final_output == "Test output"
        assert trace.metrics.total_cost == 0.0
        assert len(trace.steps) == 0

    @pytest.mark.asyncio
    async def test_parse_response_output_field_priority(self):
        """Test that output fields are tried in order: response, output, result, answer."""
        adapter = HTTPAdapter(endpoint="http://test.com/api")
        start_time = datetime.now()
        end_time = datetime(start_time.year, start_time.month, start_time.day, 12, 0, 1)

        # Test with 'answer' field
        response_with_answer = {"answer": "Answer text"}
        trace = adapter._parse_response(response_with_answer, start_time, end_time)
        assert trace.final_output == "Answer text"

        # Test with 'result' field
        response_with_result = {"result": "Result text"}
        trace = adapter._parse_response(response_with_result, start_time, end_time)
        assert trace.final_output == "Result text"

        # Test with 'output' field
        response_with_output = {"output": "Output text"}
        trace = adapter._parse_response(response_with_output, start_time, end_time)
        assert trace.final_output == "Output text"

        # Test with 'response' field (highest priority)
        response_with_all = {
            "response": "Response text",
            "output": "Output text",
            "result": "Result text",
        }
        trace = adapter._parse_response(response_with_all, start_time, end_time)
        assert trace.final_output == "Response text"

    @pytest.mark.asyncio
    async def test_parse_response_no_output(self):
        """Test parsing response with no output field."""
        adapter = HTTPAdapter(endpoint="http://test.com/api")
        start_time = datetime.now()
        end_time = datetime(start_time.year, start_time.month, start_time.day, 12, 0, 1)

        response = {"cost": 0.01}
        trace = adapter._parse_response(response, start_time, end_time)

        assert trace.final_output == ""

    @pytest.mark.asyncio
    async def test_parse_response_with_tokens_calculates_cost(self, http_response_with_tokens_only):
        """Test that cost is calculated from tokens when not provided."""
        adapter = HTTPAdapter(endpoint="http://test.com/api", model_config={"name": "gpt-4"})
        start_time = datetime.now()
        end_time = datetime(start_time.year, start_time.month, start_time.day, 12, 0, 1)

        with patch("evalview.adapters.http_adapter.calculate_cost") as mock_calculate:
            mock_calculate.return_value = 0.0035  # Mocked calculated cost

            trace = adapter._parse_response(http_response_with_tokens_only, start_time, end_time)

            mock_calculate.assert_called_once_with(
                model_name="gpt-4",
                input_tokens=100,
                output_tokens=200,
                cached_tokens=50,
            )
            assert trace.metrics.total_cost == 0.0035

    @pytest.mark.asyncio
    async def test_parse_response_cost_from_metadata(self):
        """Test extracting cost from nested metadata."""
        adapter = HTTPAdapter(endpoint="http://test.com/api")
        start_time = datetime.now()
        end_time = datetime(start_time.year, start_time.month, start_time.day, 12, 0, 1)

        response = {"output": "test", "metadata": {"cost": 0.12}}
        trace = adapter._parse_response(response, start_time, end_time)

        assert trace.metrics.total_cost == 0.12

    @pytest.mark.asyncio
    async def test_parse_response_cost_from_steps(self):
        """Test calculating total cost from step costs when not provided."""
        adapter = HTTPAdapter(endpoint="http://test.com/api")
        start_time = datetime.now()
        end_time = datetime(start_time.year, start_time.month, start_time.day, 12, 0, 1)

        response = {
            "output": "test",
            "steps": [
                {"tool": "tool1", "cost": 0.02},
                {"tool": "tool2", "cost": 0.03},
            ],
        }
        trace = adapter._parse_response(response, start_time, end_time)

        assert trace.metrics.total_cost == 0.05

    @pytest.mark.asyncio
    async def test_parse_steps_with_minimal_fields(self):
        """Test parsing steps with minimal required fields."""
        adapter = HTTPAdapter(endpoint="http://test.com/api")

        steps_data = [{"tool": "test_tool"}]
        steps = adapter._parse_steps(steps_data)

        assert len(steps) == 1
        assert steps[0].step_id == "step-0"  # Auto-generated
        assert steps[0].step_name == "Step 1"  # Auto-generated
        assert steps[0].tool_name == "test_tool"
        assert steps[0].parameters == {}
        assert steps[0].success is True
        assert steps[0].metrics.latency == 0.0
        assert steps[0].metrics.cost == 0.0

    @pytest.mark.asyncio
    async def test_parse_steps_with_tool_name_field(self):
        """Test that tool_name field is also recognized."""
        adapter = HTTPAdapter(endpoint="http://test.com/api")

        steps_data = [{"tool_name": "alternate_tool"}]
        steps = adapter._parse_steps(steps_data)

        assert steps[0].tool_name == "alternate_tool"

    @pytest.mark.asyncio
    async def test_parse_steps_parameters_aliases(self):
        """Test that both 'parameters' and 'params' are recognized."""
        adapter = HTTPAdapter(endpoint="http://test.com/api")

        steps_data = [
            {"tool": "tool1", "params": {"key": "value"}},
            {"tool": "tool2", "parameters": {"key2": "value2"}},
        ]
        steps = adapter._parse_steps(steps_data)

        assert steps[0].parameters == {"key": "value"}
        assert steps[1].parameters == {"key2": "value2"}

    @pytest.mark.asyncio
    async def test_parse_steps_output_aliases(self):
        """Test that both 'output' and 'result' are recognized."""
        adapter = HTTPAdapter(endpoint="http://test.com/api")

        steps_data = [
            {"tool": "tool1", "result": "result1"},
            {"tool": "tool2", "output": "output2"},
        ]
        steps = adapter._parse_steps(steps_data)

        assert steps[0].output == "result1"
        assert steps[1].output == "output2"

    @pytest.mark.asyncio
    async def test_parse_steps_with_errors(self):
        """Test parsing steps with error information."""
        adapter = HTTPAdapter(endpoint="http://test.com/api")

        steps_data = [
            {
                "tool": "failing_tool",
                "success": False,
                "error": "Connection timeout",
            }
        ]
        steps = adapter._parse_steps(steps_data)

        assert steps[0].success is False
        assert steps[0].error == "Connection timeout"

    @pytest.mark.asyncio
    async def test_parse_response_session_id_generation(self):
        """Test that session_id is generated if not provided."""
        adapter = HTTPAdapter(endpoint="http://test.com/api")
        start_time = datetime(2025, 1, 1, 12, 0, 0)
        end_time = datetime(2025, 1, 1, 12, 0, 1)

        response = {"response": "test"}
        trace = adapter._parse_response(response, start_time, end_time)

        assert trace.session_id.startswith("session-")
        assert "1735732800" in trace.session_id  # Unix timestamp

    @pytest.mark.asyncio
    async def test_parse_response_latency_calculation(self):
        """Test that latency is calculated from start/end times."""
        adapter = HTTPAdapter(endpoint="http://test.com/api")
        start_time = datetime(2025, 1, 1, 12, 0, 0)
        end_time = datetime(2025, 1, 1, 12, 0, 3)  # 3 seconds later

        response = {"response": "test"}
        trace = adapter._parse_response(response, start_time, end_time)

        assert trace.metrics.total_latency == 3000.0  # 3 seconds in milliseconds

    @pytest.mark.asyncio
    async def test_parse_response_tokens_integer_format(self):
        """Test parsing tokens when provided as simple integer."""
        adapter = HTTPAdapter(endpoint="http://test.com/api")
        start_time = datetime.now()
        end_time = datetime(start_time.year, start_time.month, start_time.day, 12, 0, 1)

        response = {"response": "test", "tokens": 500}
        trace = adapter._parse_response(response, start_time, end_time)

        assert trace.metrics.total_tokens.input_tokens == 0
        assert trace.metrics.total_tokens.output_tokens == 500
        assert trace.metrics.total_tokens.cached_tokens == 0

    @pytest.mark.asyncio
    async def test_parse_response_tokens_dict_format(self):
        """Test parsing tokens when provided as dictionary."""
        adapter = HTTPAdapter(endpoint="http://test.com/api")
        start_time = datetime.now()
        end_time = datetime(start_time.year, start_time.month, start_time.day, 12, 0, 1)

        response = {
            "response": "test",
            "tokens": {"input": 100, "output": 200, "cached": 50},
        }
        trace = adapter._parse_response(response, start_time, end_time)

        assert trace.metrics.total_tokens.input_tokens == 100
        assert trace.metrics.total_tokens.output_tokens == 200
        assert trace.metrics.total_tokens.cached_tokens == 50

    @pytest.mark.asyncio
    async def test_parse_response_tokens_from_metadata(self):
        """Test parsing tokens from metadata section."""
        adapter = HTTPAdapter(endpoint="http://test.com/api")
        start_time = datetime.now()
        end_time = datetime(start_time.year, start_time.month, start_time.day, 12, 0, 1)

        response = {
            "response": "test",
            "metadata": {"tokens": {"input": 75, "output": 125}},
        }
        trace = adapter._parse_response(response, start_time, end_time)

        assert trace.metrics.total_tokens.input_tokens == 75
        assert trace.metrics.total_tokens.output_tokens == 125

    @pytest.mark.asyncio
    async def test_execute_with_context(self):
        """Test execute with context parameter."""
        adapter = HTTPAdapter(endpoint="http://test.com/api")

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_response = MagicMock()  # json() is sync, so use MagicMock
            mock_response.status_code = 200
            mock_response.json.return_value = {"response": "test"}
            mock_response.raise_for_status = MagicMock()
            mock_client.post.return_value = mock_response
            mock_client.__aenter__.return_value = mock_client
            mock_client.__aexit__.return_value = None
            mock_client_class.return_value = mock_client

            await adapter.execute("test query", context={"user_id": "123"})

            # Check that context was included in request
            call_args = mock_client.post.call_args
            assert call_args[1]["json"]["context"] == {"user_id": "123"}

    @pytest.mark.asyncio
    async def test_execute_with_custom_headers(self):
        """Test execute with custom headers."""
        adapter = HTTPAdapter(
            endpoint="http://test.com/api",
            headers={"Authorization": "Bearer token123"},
        )

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_response = MagicMock()  # json() is sync, so use MagicMock
            mock_response.status_code = 200
            mock_response.json.return_value = {"response": "test"}
            mock_response.raise_for_status = MagicMock()
            mock_client.post.return_value = mock_response
            mock_client.__aenter__.return_value = mock_client
            mock_client.__aexit__.return_value = None
            mock_client_class.return_value = mock_client

            await adapter.execute("test query")

            # Check that custom headers were included
            call_args = mock_client.post.call_args
            headers = call_args[1]["headers"]
            assert "Authorization" in headers
            assert headers["Authorization"] == "Bearer token123"

    @pytest.mark.asyncio
    async def test_execute_with_timeout(self):
        """Test execute with custom timeout."""
        adapter = HTTPAdapter(endpoint="http://test.com/api", timeout=60.0)

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_response = MagicMock()  # json() is sync, so use MagicMock
            mock_response.status_code = 200
            mock_response.json.return_value = {"response": "test"}
            mock_response.raise_for_status = MagicMock()
            mock_client.post.return_value = mock_response
            mock_client.__aenter__.return_value = mock_client
            mock_client.__aexit__.return_value = None
            mock_client_class.return_value = mock_client

            await adapter.execute("test query")

            # Check that timeout was passed to client
            call_args = mock_client_class.call_args
            assert call_args[1]["timeout"] == 60.0

    @pytest.mark.asyncio
    async def test_execute_http_error(self):
        """Test execute with HTTP error response."""
        adapter = HTTPAdapter(endpoint="http://test.com/api")

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_response = MagicMock()  # json() is sync, so use MagicMock
            mock_response.status_code = 500
            mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
                "Server Error", request=MagicMock(), response=mock_response
            )
            mock_client.post.return_value = mock_response
            mock_client.__aenter__.return_value = mock_client
            mock_client.__aexit__.return_value = None
            mock_client_class.return_value = mock_client

            with pytest.raises(httpx.HTTPStatusError):
                await adapter.execute("test query")

    @pytest.mark.asyncio
    async def test_health_check_success(self):
        """Test successful health check."""
        adapter = HTTPAdapter(endpoint="http://test.com/api/")

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_response = AsyncMock()
            mock_response.status_code = 200
            mock_client.get.return_value = mock_response
            mock_client.__aenter__.return_value = mock_client
            mock_client.__aexit__.return_value = None
            mock_client_class.return_value = mock_client

            result = await adapter.health_check()

            assert result is True
            mock_client.get.assert_called_once_with("http://test.com/health")

    @pytest.mark.asyncio
    async def test_health_check_failure(self):
        """Test failed health check."""
        adapter = HTTPAdapter(endpoint="http://test.com/api/")

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_response = AsyncMock()
            mock_response.status_code = 500
            mock_client.get.return_value = mock_response
            mock_client.__aenter__.return_value = mock_client
            mock_client.__aexit__.return_value = None
            mock_client_class.return_value = mock_client

            result = await adapter.health_check()

            assert result is False

    @pytest.mark.asyncio
    async def test_health_check_exception(self):
        """Test health check with network exception."""
        adapter = HTTPAdapter(endpoint="http://test.com/api/")

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.get.side_effect = httpx.ConnectError("Connection refused")
            mock_client.__aenter__.return_value = mock_client
            mock_client.__aexit__.return_value = None
            mock_client_class.return_value = mock_client

            result = await adapter.health_check()

            assert result is False

    def test_adapter_name_property(self):
        """Test that adapter name property returns correct value."""
        adapter = HTTPAdapter(endpoint="http://test.com/api")
        assert adapter.name == "http"

    @pytest.mark.asyncio
    async def test_parse_response_empty_steps_list(self):
        """Test parsing response with empty steps list."""
        adapter = HTTPAdapter(endpoint="http://test.com/api")
        start_time = datetime.now()
        end_time = datetime(start_time.year, start_time.month, start_time.day, 12, 0, 1)

        response = {"response": "test", "steps": []}
        trace = adapter._parse_response(response, start_time, end_time)

        assert len(trace.steps) == 0

    @pytest.mark.asyncio
    async def test_execute_enables_tracing(self):
        """Test that execute sends enable_tracing flag."""
        adapter = HTTPAdapter(endpoint="http://test.com/api")

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_response = MagicMock()  # json() is sync, so use MagicMock
            mock_response.status_code = 200
            mock_response.json.return_value = {"response": "test"}
            mock_response.raise_for_status = MagicMock()
            mock_client.post.return_value = mock_response
            mock_client.__aenter__.return_value = mock_client
            mock_client.__aexit__.return_value = None
            mock_client_class.return_value = mock_client

            await adapter.execute("test query")

            # Check that enable_tracing was sent
            call_args = mock_client.post.call_args
            assert call_args[1]["json"]["enable_tracing"] is True
