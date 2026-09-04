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
from evalview.skills.adapters.base import AgentTimeoutError, SkillAgentAdapterError
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


def make_function_call(name="get_weather", arguments=None):
    item = MagicMock(
        type="function_call", id="fc_1", call_id="call_1",
        arguments=json.dumps({"city": "Tokyo"}) if arguments is None else arguments,
        status="completed",
    )
    item.name = name
    return item


def make_response(status="completed", with_tools=True):
    """Build a mock Responses API response with typed output items."""
    items = []
    if with_tools:
        code_call = MagicMock(type="code_interpreter_call", id="ci_1", code="print(6*7)", status="completed")
        log_output = MagicMock()
        log_output.type = "logs"
        log_output.logs = "42"
        image_output = MagicMock()
        image_output.type = "image"
        image_output.url = "https://example.com/img.png"
        code_call.outputs = [log_output, image_output]
        items.append(code_call)

        file_search_call = MagicMock(type="file_search_call", id="fs_1", status="completed")
        file_search_call.queries = ["quarterly report"]
        items.append(file_search_call)

        items.append(MagicMock(type="web_search_call", id="ws_1", status="completed"))
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

    @pytest.mark.parametrize("arguments", ["{not json", "[]", "null"])
    def test_malformed_function_arguments_do_not_crash_trace(self, arguments):
        response = make_response(with_tools=False)
        bad_call = make_function_call("broken_tool", arguments)
        response.output.insert(0, bad_call)
        steps = OpenAIAssistantsAdapter()._extract_steps(response)

        assert steps[0].tool_name == "broken_tool"
        assert steps[0].parameters == {"raw": arguments}
        assert steps[0].success is False
        assert "not executed" in steps[0].error

    async def test_custom_functions_cannot_masquerade_as_completed_execution(self):
        response = make_response(with_tools=False)
        response.output.insert(0, make_function_call())
        client = make_client(response)
        with patch("openai.AsyncOpenAI", return_value=client):
            with pytest.raises(RuntimeError, match="not executed: get_weather.*HTTP adapter"):
                await OpenAIAssistantsAdapter().execute("weather?")
        assert client.responses.create.await_count == 1

    async def test_execute_captures_tool_steps_and_output(self):
        client = make_client(make_response())
        with patch("openai.AsyncOpenAI", return_value=client):
            adapter = OpenAIAssistantsAdapter(model_config={"name": "gpt-4o"})
            trace = await adapter.execute("What is 6*7?")

        # Session id comes from the conversation (threads replacement)
        assert trace.session_id == "conv_1"
        assert [s.tool_name for s in trace.steps] == [
            "code_interpreter",
            "file_search",
            "web_search",
        ]
        assert trace.steps[0].output == "42"
        assert trace.steps[1].parameters == {"queries": ["quarterly report"]}
        assert all(step.success for step in trace.steps)
        assert trace.final_output == "The answer is 42."
        assert trace.metrics.total_tokens.input_tokens == 100
        assert trace.metrics.total_tokens.output_tokens == 50
        assert trace.metrics.total_cost > 0
        assert trace.rationale_events
        assert trace.model_id == "gpt-4o"
        assert trace.model_provider == "openai"

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
        client.__aexit__.assert_awaited_once()

    async def test_client_closes_after_successful_execution(self):
        client = make_client(make_response())
        with patch("openai.AsyncOpenAI", return_value=client):
            await OpenAIAssistantsAdapter().execute("q")
        client.__aexit__.assert_awaited_once()

    async def test_conversation_creation_is_bounded_and_closes_client(self):
        client = make_client(make_response())

        async def never_completes():
            await asyncio.Event().wait()

        client.conversations.create.side_effect = never_completes
        with patch("openai.AsyncOpenAI", return_value=client):
            with pytest.raises(TimeoutError, match="Conversation creation exceeded"):
                await OpenAIAssistantsAdapter(timeout=0.01).execute("q")
        client.responses.create.assert_not_awaited()
        client.__aexit__.assert_awaited_once()

    @pytest.mark.parametrize("source", ["config", "context", "env"])
    async def test_legacy_only_assistant_id_requires_migration(self, source, monkeypatch):
        config = {"assistant_id": "asst_old"} if source == "config" else {}
        context = {"assistant_id": "asst_old"} if source == "context" else {}
        if source == "env":
            monkeypatch.setenv("OPENAI_ASSISTANT_ID", "asst_old")
        with patch("openai.AsyncOpenAI") as client:
            with pytest.raises(ValueError, match="Migrate.*prompt_id"):
                await OpenAIAssistantsAdapter(**config).execute("hi", context)
        client.assert_not_called()

    async def test_legacy_id_with_explicit_replacement_warns(self, caplog):
        client = make_client(make_response(with_tools=False))
        with patch("openai.AsyncOpenAI", return_value=client):
            adapter = OpenAIAssistantsAdapter(assistant_id="asst_old", prompt_id="pmpt_migrated")
            trace = await adapter.execute("hi")

        assert "assistant_id" in caplog.text
        assert trace.final_output == "The answer is 42."
        # No assistant params leak into the request
        assert "assistant_id" not in client.responses.create.call_args.kwargs

    async def test_model_alone_does_not_replace_legacy_assistant_behavior(self):
        with patch("openai.AsyncOpenAI") as client:
            adapter = OpenAIAssistantsAdapter(assistant_id="asst_old", model_config={"name": "gpt-4o"})
            with pytest.raises(ValueError, match="Migrate its model, instructions"):
                await adapter.execute("hi")
        client.assert_not_called()

    async def test_prompt_id_instructions_and_tool_shorthands(self, monkeypatch):
        monkeypatch.setenv("OPENAI_VECTOR_STORE_IDS", "vs_1, vs_2")
        client = make_client(make_response(with_tools=False))
        with patch("openai.AsyncOpenAI", return_value=client):
            adapter = OpenAIAssistantsAdapter(
                instructions="Be terse.",
                prompt_id="pmpt_1",
                tools=["web_search", "file_search"],
            )
            await adapter.execute("hi")

        kwargs = client.responses.create.call_args.kwargs
        assert kwargs["prompt"] == {"id": "pmpt_1"}
        assert kwargs["instructions"] == "Be terse."
        assert kwargs["tools"] == [
            {"type": "web_search"},
            {"type": "file_search", "vector_store_ids": ["vs_1", "vs_2"]},
        ]

    @pytest.mark.parametrize("tools, message", [
        (["file_search"], "vector store ids"),
        (["typo_search"], "Unknown OpenAI tool"),
    ])
    async def test_invalid_explicit_tools_fail_before_calling_openai(self, tools, message):
        with patch("openai.AsyncOpenAI") as client:
            with pytest.raises(ValueError, match=message):
                await OpenAIAssistantsAdapter(tools=tools).execute("hi")
        client.assert_not_called()

    async def test_prompt_id_alone_sends_no_default_tools(self):
        # A dashboard Prompt object defines its own tools; injecting the
        # default code_interpreter would silently override them.
        client = make_client(make_response(with_tools=False))
        with patch("openai.AsyncOpenAI", return_value=client):
            adapter = OpenAIAssistantsAdapter(prompt_id="pmpt_1")
            await adapter.execute("hi")

        assert "tools" not in client.responses.create.call_args.kwargs
        assert "model" not in client.responses.create.call_args.kwargs

    @pytest.mark.parametrize("model_config", ["gpt-4.1", {"name": "gpt-4.1"}])
    async def test_explicit_model_overrides_prompt_model(self, model_config):
        client = make_client(make_response(with_tools=False))
        with patch("openai.AsyncOpenAI", return_value=client):
            await OpenAIAssistantsAdapter(prompt_id="pmpt_1", model_config=model_config).execute("hi")
        assert client.responses.create.call_args.kwargs["model"] == "gpt-4.1"

    async def test_empty_tools_explicitly_disable_prompt_tools(self):
        client = make_client(make_response(with_tools=False))
        with patch("openai.AsyncOpenAI", return_value=client):
            await OpenAIAssistantsAdapter(prompt_id="pmpt_1", tools=[]).execute("hi")
        assert client.responses.create.call_args.kwargs["tools"] == []

    def test_failed_hosted_tool_is_not_reported_as_success(self):
        response = make_response()
        response.output[0].status = "failed"
        step = OpenAIAssistantsAdapter()._extract_steps(response)[0]
        assert step.success is False
        assert step.error == "Tool call status: failed"

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
        assert "web_search" in trace.tool_calls
        assert any("[python]" in c for c in trace.commands_ran)
        assert "[generated_image:https://example.com/img.png]" in trace.files_created
        assert not trace.errors

        kwargs = client.responses.create.call_args.kwargs
        assert "SKILL: test-skill" in kwargs["instructions"]
        assert kwargs["max_output_tokens"] == 4096
        assert kwargs["tools"] == [
            {"type": "code_interpreter", "container": {"type": "auto"}}
        ]

    async def test_failed_response_raises_adapter_error(self, skill, agent_config):
        adapter = OpenAIAssistantsSkillAdapter(agent_config)
        adapter._client = make_client(make_response(status="failed"))

        with pytest.raises(SkillAgentAdapterError, match="boom"):
            await adapter.execute(skill, "run it")

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
        assert "model" not in kwargs

    def test_legacy_only_skill_config_requires_migration(self):
        config = AgentConfig(type=AgentType.OPENAI_ASSISTANTS, env={
            "OPENAI_API_KEY": "test-key", "OPENAI_ASSISTANT_ID": "asst_old", "OPENAI_MODEL": "gpt-4o",
        })
        with pytest.raises(SkillAgentAdapterError, match="Migrate.*OPENAI_PROMPT_ID"):
            OpenAIAssistantsSkillAdapter(config)

    async def test_skill_model_override_and_empty_tools_are_preserved(self, skill):
        config = AgentConfig(
            type=AgentType.OPENAI_ASSISTANTS, model="gpt-4.1", tools=[],
            env={"OPENAI_API_KEY": "test-key", "OPENAI_PROMPT_ID": "pmpt_1"},
        )
        adapter = OpenAIAssistantsSkillAdapter(config)
        adapter._client = make_client(make_response(with_tools=False))
        await adapter.execute(skill, "hi")
        request = adapter._client.responses.create.call_args.kwargs
        assert request["model"] == "gpt-4.1"
        assert request["tools"] == []

    @pytest.mark.parametrize("tools, message", [
        (["file_search"], "vector store ids"),
        (["typo_search"], "Unsupported OpenAI skill tools"),
    ])
    async def test_skill_invalid_tools_fail_before_api_call(self, skill, agent_config, tools, message):
        agent_config.tools = tools
        adapter = OpenAIAssistantsSkillAdapter(agent_config)
        adapter._client = make_client(make_response(with_tools=False))
        with pytest.raises(SkillAgentAdapterError, match=message):
            await adapter.execute(skill, "hi")
        adapter._client.responses.create.assert_not_awaited()

    async def test_skill_custom_functions_fail_without_claiming_execution(self, skill, agent_config):
        adapter = OpenAIAssistantsSkillAdapter(agent_config)
        response = make_response(with_tools=False)
        response.output.insert(0, make_function_call("write_file", '{"path": "report.md"}'))
        adapter._client = make_client(response)
        with pytest.raises(SkillAgentAdapterError, match="not executed: write_file"):
            await adapter.execute(skill, "write it")

    @pytest.mark.parametrize("name, arguments", [
        ("write_file", '{"path": "report.md"}'),
        ("run_command", '{"command": "echo done"}'),
    ])
    def test_skill_function_proposals_never_invent_files_or_commands(self, agent_config, name, arguments):
        adapter = OpenAIAssistantsSkillAdapter(agent_config)
        events, calls, files, commands = [], [], [], []
        adapter._process_output_item(make_function_call(name, arguments), events, calls, files, commands)
        assert calls == [name]
        assert events[0].tool_success is False
        assert "not executed" in events[0].tool_error
        assert files == []
        assert commands == []

    async def test_skill_replays_conversation_history_without_mutating_it(self, skill, agent_config):
        adapter = OpenAIAssistantsSkillAdapter(agent_config)
        adapter._client = make_client(make_response(with_tools=False))
        history = [{"role": "user", "content": "Remember 42"}, {"role": "assistant", "content": "OK"}]
        await adapter.execute(skill, "What number?", {"conversation_history": history})
        assert adapter._client.responses.create.call_args.kwargs["input"] == [
            *history, {"role": "user", "content": "What number?"},
        ]
        assert len(history) == 2

    async def test_cleanup_is_noop(self, agent_config):
        adapter = OpenAIAssistantsSkillAdapter(agent_config)
        await adapter.cleanup()


class TestOpenAIConfigurationPaths:
    """Exercise the real config loaders and command factories before mocking HTTP."""

    @pytest.fixture
    def config_data(self):
        return {
            "adapter": "openai-assistants",
            "timeout": 45.0,
            "model": {"name": "gpt-4.1"},
            "instructions": "Answer in one sentence.",
            "prompt_id": "pmpt_project",
            "tools": [{"type": "web_search"}],
        }

    @pytest.fixture
    def test_case(self):
        from evalview.core.types import TestCase

        return TestCase.model_validate({
            "name": "OpenAI config parity",
            "input": {"query": "What changed?"},
            "expected": {"tools": []},
            "thresholds": {"min_score": 0},
        })

    @pytest.mark.parametrize("factory", ["run", "snapshot_check", "programmatic"])
    @pytest.mark.parametrize("adapter_name", ["openai", "openai-assistants"])
    async def test_endpointless_project_config_reaches_every_request(
        self, factory, adapter_name, config_data, test_case, tmp_path, monkeypatch,
    ):
        import yaml
        from evalview.commands.run._adapters import build_adapter
        from evalview.commands.shared import _build_adapter_for_tc, _load_config_if_exists
        from evalview.core.adapter_factory import create_adapter_from_config

        config_data["adapter"] = adapter_name
        monkeypatch.chdir(tmp_path)
        (tmp_path / ".evalview").mkdir()
        (tmp_path / ".evalview/config.yaml").write_text(yaml.safe_dump(config_data))
        config = _load_config_if_exists()
        assert config.endpoint == ""

        if factory == "run":
            adapter = build_adapter(adapter_name, None, config_data, config.model, False, True)
        elif factory == "snapshot_check":
            adapter = _build_adapter_for_tc(test_case, config, config.timeout)
        else:
            adapter = create_adapter_from_config(config)

        assert adapter.timeout == 45.0
        client = make_client(make_response(with_tools=False))
        with patch("openai.AsyncOpenAI", return_value=client):
            await adapter.execute(test_case.input.query)
        request = client.responses.create.call_args.kwargs
        assert request["model"] == "gpt-4.1"
        assert request["instructions"] == config_data["instructions"]
        assert request["prompt"] == {"id": "pmpt_project"}
        assert request["tools"] == [{"type": "web_search"}]

    @pytest.mark.parametrize("factory", ["run", "snapshot_check"])
    @pytest.mark.parametrize("explicit_adapter", [False, True])
    async def test_per_test_overrides_preserve_unspecified_project_settings(
        self, factory, explicit_adapter, config_data, test_case,
    ):
        from evalview.commands.run._adapters import get_test_adapter
        from evalview.commands.shared import _build_adapter_for_tc
        from evalview.core.adapter_factory import create_adapter_from_config
        from evalview.core.config import EvalViewConfig

        config = EvalViewConfig.model_validate(config_data)
        test_case.adapter = "openai-assistants" if explicit_adapter else None
        test_case.adapter_config = {
            "model": "gpt-4.1-mini", "prompt_id": "pmpt_test", "tools": [],
        }
        if factory == "run":
            adapter = get_test_adapter(
                test_case, create_adapter_from_config(config), config.model,
                True, False, MagicMock(),
            )
        else:
            adapter = _build_adapter_for_tc(test_case, config, config.timeout)
        client = make_client(make_response(with_tools=False))
        with patch("openai.AsyncOpenAI", return_value=client):
            await adapter.execute(test_case.input.query)
        request = client.responses.create.call_args.kwargs
        assert request["model"] == "gpt-4.1-mini"
        assert request["instructions"] == config.instructions
        assert request["prompt"] == {"id": "pmpt_test"}
        assert request["tools"] == []

    def test_http_config_still_requires_endpoint(self):
        from pydantic import ValidationError
        from evalview.core.config import EvalViewConfig

        with pytest.raises(ValidationError, match="endpoint"):
            EvalViewConfig(adapter="http")

    async def test_multi_turn_execution_replays_prior_messages_once(self):
        from evalview.commands.shared import _execute_multi_turn_trace
        from evalview.core.types import TestCase

        test_case = TestCase.model_validate({
            "name": "OpenAI remembers previous turn",
            "turns": [{"query": "Remember 42"}, {"query": "Which number?"}],
            "expected": {"tools": []}, "thresholds": {"min_score": 0},
        })
        first = make_response(with_tools=False)
        first.output_text = "I will remember 42."
        second = make_response(with_tools=False)
        second.output_text = "42"
        client = make_client(first)
        client.responses.create.side_effect = [first, second]
        adapter = OpenAIAssistantsAdapter(tools=[])
        with patch("openai.AsyncOpenAI", return_value=client):
            trace = await _execute_multi_turn_trace(test_case, adapter)
        requests = [call.kwargs for call in client.responses.create.await_args_list]
        assert requests[0]["input"] == "Remember 42"
        assert requests[1]["input"] == [
            {"role": "user", "content": "Remember 42"},
            {"role": "assistant", "content": "I will remember 42."},
            {"role": "user", "content": "Which number?"},
        ]
        assert trace.final_output == "42"
        assert [turn.output for turn in trace.turns] == ["I will remember 42.", "42"]
