"""Tests for the Pydantic AI adapter's tool-call and schema-validation extraction.

Requires ``pydantic-ai-slim`` (Python 3.10+). Skipped entirely on older
interpreters or when the dependency isn't installed, via ``importorskip``.
"""

import pytest

pydantic_ai = pytest.importorskip("pydantic_ai")

from pydantic_ai import Agent, ModelRetry  # noqa: E402
from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart  # noqa: E402
from pydantic_ai.models.function import AgentInfo, FunctionModel  # noqa: E402

from evalview.adapters.pydantic_ai_adapter import PydanticAIAdapter  # noqa: E402


def _make_agent(model_function):
    agent = Agent(FunctionModel(model_function))

    @agent.tool_plain
    def lookup_order(order_id: int) -> str:
        """Look up an order by ID."""
        return f"Order {order_id} found"

    return agent


class TestPydanticAIAdapterToolCalls:
    """Tests for the adapter's typed message-history extraction."""

    @pytest.mark.asyncio
    async def test_valid_tool_call_marks_validation_valid(self):
        """A successful tool call records valid=True and the tool's return value."""
        calls = {"n": 0}

        def model_function(messages, info: AgentInfo) -> ModelResponse:
            calls["n"] += 1
            if calls["n"] == 1:
                return ModelResponse(parts=[ToolCallPart(tool_name="lookup_order", args={"order_id": 123})])
            return ModelResponse(parts=[TextPart(content="done")])

        adapter = PydanticAIAdapter(agent=_make_agent(model_function))
        trace = await adapter.execute("look up order 123")

        assert len(trace.steps) == 1
        step = trace.steps[0]
        assert step.tool_name == "lookup_order"
        assert step.parameters == {"order_id": 123}
        assert step.success is True
        assert step.output == "Order 123 found"
        assert step.tool_argument_validation is not None
        assert step.tool_argument_validation.valid is True
        assert step.tool_argument_validation.errors == []

    @pytest.mark.asyncio
    async def test_invalid_tool_args_flagged_as_schema_invalid(self):
        """Pydantic AI rejecting arguments before the tool runs is captured as invalid."""
        calls = {"n": 0}

        def model_function(messages, info: AgentInfo) -> ModelResponse:
            calls["n"] += 1
            if calls["n"] == 1:
                # order_id should be an int; this triggers a Pydantic validation error.
                return ModelResponse(
                    parts=[ToolCallPart(tool_name="lookup_order", args={"order_id": "not-a-number"})]
                )
            if calls["n"] == 2:
                return ModelResponse(parts=[ToolCallPart(tool_name="lookup_order", args={"order_id": 123})])
            return ModelResponse(parts=[TextPart(content="done")])

        adapter = PydanticAIAdapter(agent=_make_agent(model_function))
        trace = await adapter.execute("look up order 123")

        # Both the invalid attempt and the corrected retry are preserved as
        # separate steps, correlated by their own tool_call_id.
        assert len(trace.steps) == 2

        invalid_step, valid_step = trace.steps

        assert invalid_step.tool_name == "lookup_order"
        assert invalid_step.parameters == {"order_id": "not-a-number"}
        assert invalid_step.success is False
        assert invalid_step.tool_argument_validation is not None
        assert invalid_step.tool_argument_validation.valid is False
        assert invalid_step.tool_argument_validation.source == "pydantic-ai"
        assert len(invalid_step.tool_argument_validation.errors) == 1
        error = invalid_step.tool_argument_validation.errors[0]
        assert error["type"] == "int_parsing"
        assert error["loc"] == ["order_id"]

        assert valid_step.success is True
        assert valid_step.tool_argument_validation.valid is True
        assert valid_step.output == "Order 123 found"

    @pytest.mark.asyncio
    async def test_model_retry_is_not_misclassified_as_schema_invalid(self):
        """A ModelRetry raised by the tool body must not be flagged as a schema violation."""
        calls = {"n": 0}

        def model_function(messages, info: AgentInfo) -> ModelResponse:
            calls["n"] += 1
            if calls["n"] == 1:
                return ModelResponse(parts=[ToolCallPart(tool_name="lookup_order", args={"order_id": 5})])
            return ModelResponse(parts=[TextPart(content="done")])

        agent = Agent(FunctionModel(model_function))

        @agent.tool_plain
        def lookup_order(order_id: int) -> str:
            if order_id == 5:
                raise ModelRetry("order not found, please double check the id")
            return f"Order {order_id} found"

        adapter = PydanticAIAdapter(agent=agent)
        trace = await adapter.execute("look up order 5")

        assert len(trace.steps) == 1
        step = trace.steps[0]
        assert step.success is False
        assert step.error == "order not found, please double check the id"
        # A ModelRetry means the arguments were valid and the tool executed;
        # it must not be treated as a schema violation.
        assert step.tool_argument_validation is not None
        assert step.tool_argument_validation.valid is True
        assert step.tool_argument_validation.errors == []
