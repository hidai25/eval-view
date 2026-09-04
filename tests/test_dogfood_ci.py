"""A public reliability check must fail visibly and distinguish missing evidence."""

import json
import os
from pathlib import Path
import subprocess
import sys
from unittest.mock import Mock

import httpx
import openai
import pytest
import yaml

from dogfood import preflight
from dogfood.summarize import CHECKS, summarize


def workflow_step(step_id):
    workflow = Path(__file__).resolve().parents[1] / ".github/workflows/dogfood.yml"
    job = yaml.safe_load(workflow.read_text())["jobs"]["dogfood"]
    assert job["defaults"]["run"]["shell"] == "bash"
    return next(step for step in job["steps"] if step.get("id") == step_id)["run"]


def test_failed_snapshot_stops_real_workflow_before_check(tmp_path):
    stubs = """
    uv() {
      if [[ "$*" == *"evalview snapshot"* ]]; then
        echo 'snapshot failed'
        return 13
      fi
      if [[ "$*" == *"evalview check"* ]]; then
        echo 'CHECK MUST NOT RUN'
      fi
    }
    curl() { return 0; }
    cp() { return 0; }
    """
    result = subprocess.run(
        ["bash", "--noprofile", "--norc", "-eo", "pipefail", "-c", stubs + workflow_step("regression")],
        cwd=tmp_path, capture_output=True, text=True,
    )
    assert result.returncode == 13
    assert "snapshot failed" in result.stdout
    assert "CHECK MUST NOT RUN" not in result.stdout


@pytest.mark.parametrize("monitor_exit", [2, 143])
def test_monitor_workflow_preserves_unexpected_exit_even_with_noise_log(tmp_path, monitor_exit):
    (tmp_path / ".evalview").mkdir()
    (tmp_path / ".evalview/noise.jsonl").write_text('{"cycle": 1}\n')
    stubs = f"""
    uv() {{ return 0; }}
    curl() {{ return 0; }}
    timeout() {{ echo 'monitor evidence'; return {monitor_exit}; }}
    """
    result = subprocess.run(
        ["bash", "--noprofile", "--norc", "-eo", "pipefail", "-c", stubs + workflow_step("monitor")],
        cwd=tmp_path, capture_output=True, text=True,
    )
    assert result.returncode == (0 if monitor_exit == 143 else monitor_exit)


@pytest.mark.parametrize("exit_code", [0, 7])
def test_log_capture_preserves_real_exit_code_and_evidence(tmp_path, exit_code):
    script = Path(__file__).resolve().parents[1] / "dogfood/run-check.sh"
    output = tmp_path / "github-output"
    completed = subprocess.run(
        ["bash", str(script), "sample", sys.executable, "-c",
         f"print('visible evidence'); raise SystemExit({exit_code})"],
        cwd=tmp_path, env={**os.environ, "GITHUB_OUTPUT": str(output)}, capture_output=True, text=True,
    )
    assert completed.returncode == exit_code
    assert "visible evidence" in completed.stdout
    assert (tmp_path / "sample-output.txt").read_text() == "visible evidence\n"
    assert output.read_text() == f"exit_code={exit_code}\n"


def test_continue_on_error_cannot_turn_failed_outcome_green():
    steps = {name: {"outcome": "success"} for name in CHECKS}
    steps["pytest"] = {"outcome": "failure", "conclusion": "success"}
    failed, _ = summarize(steps)
    assert failed == ["pytest"]


def test_only_complete_success_is_green():
    steps = {name: {"outcome": "success"} for name in CHECKS}
    assert summarize(steps)[0] == []
    del steps["dogfood"]
    assert summarize(steps)[0] == ["dogfood"]
    assert len(summarize({})[0]) == len(CHECKS)


def test_provider_outage_blocks_live_checks_without_calling_them_passed():
    steps = {name: {"outcome": "success"} for name in CHECKS}
    steps["provider"] = {"outcome": "failure"}
    steps["pytest_llm"] = steps["dogfood"] = {"outcome": "skipped"}
    failed, summary = summarize(steps)
    assert failed == ["provider", "pytest-llm", "dogfood"]
    assert "blocked by provider preflight" in summary
    assert "unavailable, not passing" in summary


def test_credit_exhaustion_is_infrastructure_not_agent_regression():
    response = httpx.Response(429, request=httpx.Request("POST", "https://api.openai.com/v1/chat/completions"))
    error = openai.RateLimitError("no credits", response=response, body={
        "code": "credit_balance_exhausted", "type": "insufficient_quota"
    })
    assert preflight.classify_provider_error(error)["category"] == "provider_quota"


@pytest.mark.parametrize("status,category", [
    (401, "provider_auth"), (403, "provider_auth"), (404, "provider_model_access"),
    (429, "provider_rate_limit"), (500, "provider_unavailable"),
])
def test_other_provider_failure_categories(status, category):
    error = RuntimeError("failure")
    error.status_code = status
    assert preflight.classify_provider_error(error)["category"] == category


def test_missing_provider_is_failed_not_silently_skipped(monkeypatch, capsys):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    client = Mock()
    monkeypatch.setattr(preflight, "OpenAI", client)
    assert preflight.main() == 1
    assert json.loads(capsys.readouterr().out)["category"] == "provider_not_configured"
    client.assert_not_called()


def test_preflight_stops_at_first_failure_without_retries(monkeypatch, capsys):
    monkeypatch.setenv("OPENAI_API_KEY", "secret-must-not-appear")
    client = Mock()
    client.chat.completions.create.side_effect = RuntimeError("secret-must-not-appear")
    context = Mock()
    context.__enter__ = Mock(return_value=client)
    context.__exit__ = Mock(return_value=False)
    create_client = Mock(return_value=context)
    monkeypatch.setattr(preflight, "OpenAI", create_client)
    assert preflight.main() == 1
    create_client.assert_called_once_with(timeout=20.0, max_retries=0)
    client.chat.completions.create.assert_called_once()
    output = capsys.readouterr().out
    assert "secret-must-not-appear" not in output
    assert json.loads(output)["category"] == "provider_unavailable"
