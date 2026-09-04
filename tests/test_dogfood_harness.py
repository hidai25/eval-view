"""The live dogfood harness must preserve the behavior it claims to measure."""

import subprocess
import sys
from unittest.mock import Mock

import pytest
from fastapi import HTTPException

from dogfood import agent
from evalview.core.llm_provider import LLMProvider


@pytest.fixture
def fake_chat(monkeypatch):
    class FakeChat:
        def __init__(self):
            self.prompts = []
            self.responses = []

        async def stream_response(self, prompt, *, raise_errors=False):
            assert raise_errors
            self.prompts.append(prompt)
            response = self.responses.pop(0)
            if isinstance(response, Exception):
                raise response
            yield response

    chat = FakeChat()
    monkeypatch.setattr(agent, "ChatSession", lambda **kwargs: chat)
    monkeypatch.setattr(agent, "get_provider", lambda: (LLMProvider.OPENAI, "test-model"))
    return chat


@pytest.mark.asyncio
async def test_explanation_never_executes_illustrative_commands(fake_chat, monkeypatch):
    fake_chat.responses = ["Example:\n```command\nevalview run --diff\n```"]
    execute = Mock()
    monkeypatch.setattr(agent, "execute_command", execute)
    result = await agent.execute(agent.ExecuteRequest(query="Explain baselines"))
    assert result.tool_calls == []
    execute.assert_not_called()


@pytest.mark.asyncio
async def test_final_answer_follows_actual_tool_evidence(fake_chat, monkeypatch):
    fake_chat.responses = ["```command\nevalview adapters\n```", "Observed: http, langgraph."]
    execute = Mock(return_value="ACTUAL CLI OUTPUT: http, langgraph")
    monkeypatch.setattr(agent, "execute_command", execute)
    result = await agent.execute(agent.ExecuteRequest(
        query="Run evalview adapters", context={"allowed_commands": ["evalview adapters"]}
    ))
    assert result.output == "Observed: http, langgraph."
    assert result.tool_calls[0].result == "ACTUAL CLI OUTPUT: http, langgraph"
    assert "ACTUAL CLI OUTPUT" in fake_chat.prompts[1]
    execute.assert_called_once()
    assert execute.call_args.args[1] != str(agent.os.path.dirname(agent.__file__))


@pytest.mark.asyncio
async def test_unapproved_execution_is_a_failed_request(fake_chat, monkeypatch):
    fake_chat.responses = ["```command\nevalview golden save .evalview/results/xxx.json\n```"]
    execute = Mock()
    monkeypatch.setattr(agent, "execute_command", execute)
    with pytest.raises(HTTPException) as error:
        await agent.execute(agent.ExecuteRequest(
            query="Run help", context={"allowed_commands": ["evalview snapshot --help"]}
        ))
    assert error.value.status_code == 422
    execute.assert_not_called()


@pytest.mark.asyncio
async def test_provider_failure_is_not_scored_as_an_answer(fake_chat):
    fake_chat.responses = [RuntimeError("provider unavailable")]
    with pytest.raises(HTTPException) as error:
        await agent.execute(agent.ExecuteRequest(query="Explain baselines"))
    assert error.value.status_code == 503


@pytest.mark.asyncio
async def test_failed_command_does_not_produce_successful_trace(fake_chat, monkeypatch):
    fake_chat.responses = ["```command\nevalview demo\n```"]
    monkeypatch.setattr(agent, "execute_command", Mock(side_effect=RuntimeError("exit 1")))
    with pytest.raises(HTTPException) as error:
        await agent.execute(agent.ExecuteRequest(
            query="Run demo", context={"allowed_commands": ["evalview demo"]}
        ))
    assert error.value.status_code == 502
    assert len(fake_chat.prompts) == 1


def test_command_uses_current_python_without_shell_and_isolated_cwd(monkeypatch, tmp_path):
    run = Mock(return_value=subprocess.CompletedProcess([], 0, "real output", ""))
    monkeypatch.setattr(agent.subprocess, "run", run)
    assert agent.execute_command("evalview adapters", str(tmp_path)) == "real output"
    assert run.call_args.args[0] == [sys.executable, "-m", "evalview", "adapters"]
    assert run.call_args.kwargs["cwd"] == str(tmp_path)
    assert not run.call_args.kwargs.get("shell", False)


@pytest.mark.parametrize("command", ["evalview demo; touch /tmp/injected", "evalview run", "bash"])
def test_no_shell_or_unbounded_commands(command, monkeypatch, tmp_path):
    run = Mock()
    monkeypatch.setattr(agent.subprocess, "run", run)
    with pytest.raises(ValueError):
        agent.execute_command(command, str(tmp_path))
    run.assert_not_called()


def test_cli_nonzero_is_not_presented_as_success(monkeypatch, tmp_path):
    monkeypatch.setattr(agent.subprocess, "run", Mock(
        return_value=subprocess.CompletedProcess([], 2, "", "invalid command")
    ))
    with pytest.raises(RuntimeError, match="exited with 2"):
        agent.execute_command("evalview demo", str(tmp_path))


def test_real_cli_command_works_from_isolated_directory(tmp_path):
    output = agent.execute_command("evalview adapters", str(tmp_path))
    assert "http" in output
    assert "langgraph" in output
