"""Tests for the OpenAI adapters on the Responses API.

These cover the 2026 migration off the removed Assistants API: execution
goes through client.responses.create (with client.conversations.create
supplying session state) and traces are read from typed output items
instead of the Run Steps API.
"""

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from evalview.adapters.openai_assistants_adapter import OpenAIAssistantsAdapter
from evalview.skills.adapters.base import AgentTimeoutError
from evalview.skills.adapters.openai_assistants_adapter import (
    AssistantConfig,
    OpenAIAssistantsSkillAdapter,
    ResponsesConfig,
)
from evalview.skills.agent_types import AgentConfig, AgentType


@pytest.fixture(autouse=True)
def _clear_openai_env(monkeypatch):
    """Isolate tests from ambient OpenAI configuration."""
    for var in (
        "OPENAI_ASSISTANT_ID",
        "OPENAI_PROMPT_ID",
        "OPENAI_VECTOR_STORE_IDS",
        "OPENAI_MODEL",
    ):
        monkeypatch.delenv(var, raising=False)


def make_response(status="completed", with_tools=True):
    """Build a mock Responses API response with typed output items."""
    items = []
    if with_tools:
        function_call = MagicMock(
            type="function_call",
            id="fc_1",
            call_id="call_1",
            arguments=json.dumps({"city": "Tokyo"}),
        )
        function_call.name = "get_weather"
        items.append(function_call)

        code_call = MagicMock(type="code_interpreter_call", id="ci_1", code="print(6*7)")
        log_output = MagicMock()
        log_output.type = "logs"
        log_output.logs = "42"
        image_output = MagicMock()
        image_output.type = "image"
        image_output.url = "https://example.com/img.png"
        code_call.outputs = [log_output, image_output]
        items.append(code_call)

        file_search_call = MagicMock(type="file_search_call", id="fs_1")
        file_search_call.queries = ["quarterly report"]
        items.append(file_search_call)

        items.append(MagicMock(type="web_search_call", id="ws_1"))
        items.append(MagicMock(type="reasoning", id="rs_1"))

    items.append(MagicMock(type="message", id="msg_1", role="assistant"))

    response = MagicMock(
        id="resp_1",
        status=status,
        model="gpt-4o",
        usage=MagicMock(input_tokens=100, output_tokens=50),
        error=None,
        incomplete_details=None,
        output=items,
        output_text="The answer is 42.",
    )
    if status == "failed":
        response.error = MagicMock(code="server_error", message="boom")
    return response


def make_client(response):
    client = AsyncMock()
    client.conversations.create = AsyncMock(return_value=MagicMock(id="conv_1"))
    client.responses.create = AsyncMock(return_value=response)
    client.models.list = AsyncMock(return_value=MagicMock(data=[]))
    return client


class TestOpenAIResponsesAdapter:
    """Core adapter: evalview/adapters/openai_assistants_adapter.py."""

    async def test_malformed_function_arguments_do_not_crash_trace(self):
        response = make_response(with_tools=False)
        bad_call = MagicMock(type="function_call", id="fc_bad", call_id="call_bad", arguments="{not json")
        bad_call.name = "broken_tool"
        response.output.insert(0, bad_call)
        client = make_client(response)
        with patch("openai.AsyncOpenAI", return_value=client):
            adapter = OpenAIAssistantsAdapter()
            trace = await adapter.execute("q")

        assert trace.steps[0].tool_name == "broken_tool"
        assert trace.steps[0].parameters == {"raw": "{not json"}
        assert trace.final_output == "The answer is 42."

    async def test_execute_captures_tool_steps_and_output(self):
        client = make_client(make_response())
        with patch("openai.AsyncOpenAI", return_value=client):
            adapter = OpenAIAssistantsAdapter(model_config={"name": "gpt-4o"})
            trace = await adapter.execute("What is 6*7?")

        # Session id comes from the conversation (threads replacement)
        assert trace.session_id == "conv_1"
        assert [s.tool_name for s in trace.steps] == [
            "get_weather",
            "code_interpreter",
            "file_search",
            "web_search",
        ]
        assert trace.steps[0].parameters == {"city": "Tokyo"}
        assert trace.steps[1].output == "42"
        assert trace.steps[2].parameters == {"queries": ["quarterly report"]}
        assert trace.final_output == "The answer is 42."
        assert trace.metrics.total_tokens.input_tokens == 100
        assert trace.metrics.total_tokens.output_tokens == 50
        assert trace.metrics.total_cost > 0
        assert trace.rationale_events

        kwargs = client.responses.create.call_args.kwargs
        assert kwargs["conversation"] == "conv_1"
        assert kwargs["input"] == "What is 6*7?"
        assert kwargs["tools"] == [
            {"type": "code_interpreter", "container": {"type": "auto"}}
        ]

    async def test_failed_response_raises(self):
        client = make_client(make_response(status="failed"))
        with patch("openai.AsyncOpenAI", return_value=client):
            adapter = OpenAIAssistantsAdapter()
            with pytest.raises(RuntimeError, match="failed.*boom"):
                await adapter.execute("q")

    async def test_legacy_assistant_id_is_ignored(self, caplog):
        client = make_client(make_response(with_tools=False))
        with patch("openai.AsyncOpenAI", return_value=client):
            adapter = OpenAIAssistantsAdapter(assistant_id="asst_old")
            trace = await adapter.execute("hi")

        assert "assistant_id" in caplog.text
        assert trace.final_output == "The answer is 42."
        # No assistant params leak into the request
        assert "assistant_id" not in client.responses.create.call_args.kwargs

    async def test_prompt_id_instructions_and_tool_shorthands(self):
        client = make_client(make_response(with_tools=False))
        with patch("openai.AsyncOpenAI", return_value=client):
            adapter = OpenAIAssistantsAdapter(
                instructions="Be terse.",
                prompt_id="pmpt_1",
                tools=["web_search", "file_search"],  # no vector stores set
            )
            await adapter.execute("hi")

        kwargs = client.responses.create.call_args.kwargs
        assert kwargs["prompt"] == {"id": "pmpt_1"}
        assert kwargs["instructions"] == "Be terse."
        # file_search is skipped without OPENAI_VECTOR_STORE_IDS
        assert kwargs["tools"] == [{"type": "web_search"}]

    async def test_prompt_id_alone_sends_no_default_tools(self):
        # A dashboard Prompt object defines its own tools; injecting the
        # default code_interpreter would silently override them.
        client = make_client(make_response(with_tools=False))
        with patch("openai.AsyncOpenAI", return_value=client):
            adapter = OpenAIAssistantsAdapter(prompt_id="pmpt_1")
            await adapter.execute("hi")

        assert "tools" not in client.responses.create.call_args.kwargs

    async def test_timeout_raises_timeout_error(self):
        client = make_client(make_response())

        async def never(**kwargs):
            await asyncio.sleep(30)

        client.responses.create = never
        with patch("openai.AsyncOpenAI", return_value=client):
            adapter = OpenAIAssistantsAdapter(timeout=0.05)
            with pytest.raises(TimeoutError, match="timeout"):
                await adapter.execute("q")

    async def test_health_check_uses_models_list(self):
        client = make_client(make_response())
        with patch("openai.AsyncOpenAI", return_value=client):
            adapter = OpenAIAssistantsAdapter()
            assert await adapter.health_check() is True
        client.models.list.assert_awaited()


class TestOpenAIResponsesSkillAdapter:
    """Skills adapter: evalview/skills/adapters/openai_assistants_adapter.py."""

    @pytest.fixture
    def skill(self):
        skill = MagicMock()
        skill.metadata.name = "test-skill"
        skill.metadata.description = "A test skill"
        skill.instructions = "Do the thing."
        return skill

    @pytest.fixture
    def agent_config(self):
        return AgentConfig(
            type=AgentType.OPENAI_ASSISTANTS, env={"OPENAI_API_KEY": "test-key"}
        )

    def test_assistant_config_alias(self):
        assert AssistantConfig is ResponsesConfig

    async def test_execute_captures_trace(self, skill, agent_config):
        adapter = OpenAIAssistantsSkillAdapter(agent_config)
        client = make_client(make_response())
        adapter._client = client

        trace = await adapter.execute(skill, "run it", context={"test_name": "t1"})

        assert trace.final_output == "The answer is 42."
        assert trace.total_input_tokens == 100
        assert trace.total_output_tokens == 50
        assert "code_interpreter" in trace.tool_calls
        assert "get_weather" in trace.tool_calls
        assert any("[python]" in c for c in trace.commands_ran)
        assert "[generated_image:https://example.com/img.png]" in trace.files_created
        assert not trace.errors

        kwargs = client.responses.create.call_args.kwargs
        assert "SKILL: test-skill" in kwargs["instructions"]
        assert kwargs["max_output_tokens"] == 4096
        assert kwargs["tools"] == [
            {"type": "code_interpreter", "container": {"type": "auto"}}
        ]

    async def test_failed_response_recorded_as_error(self, skill, agent_config):
        adapter = OpenAIAssistantsSkillAdapter(agent_config)
        adapter._client = make_client(make_response(status="failed"))

        trace = await adapter.execute(skill, "run it")

        assert trace.errors
        assert "boom" in trace.errors[0]

    async def test_timeout_maps_to_agent_timeout_error(self, skill, agent_config):
        adapter = OpenAIAssistantsSkillAdapter(agent_config)
        client = make_client(make_response())

        async def never(**kwargs):
            await asyncio.sleep(30)

        client.responses.create = never
        adapter._client = client
        adapter.config.timeout = 0.05

        with pytest.raises(AgentTimeoutError):
            await adapter.execute(skill, "run it")

    async def test_prompt_id_alone_sends_no_default_tools(self, skill):
        config = AgentConfig(
            type=AgentType.OPENAI_ASSISTANTS,
            env={"OPENAI_API_KEY": "test-key", "OPENAI_PROMPT_ID": "pmpt_1"},
        )
        adapter = OpenAIAssistantsSkillAdapter(config)
        client = make_client(make_response(with_tools=False))
        adapter._client = client

        await adapter.execute(skill, "run it")

        kwargs = client.responses.create.call_args.kwargs
        assert kwargs["prompt"] == {"id": "pmpt_1"}
        assert "tools" not in kwargs

    async def test_cleanup_is_noop(self, agent_config):
        adapter = OpenAIAssistantsSkillAdapter(agent_config)
        await adapter.cleanup()
