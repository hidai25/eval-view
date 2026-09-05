"""Paid integration checks must be opted into, not run by ordinary CI events."""

import os
from pathlib import Path
import subprocess

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[1]
WORKFLOWS = ROOT / ".github/workflows"


def load_workflow(filename):
    # BaseLoader preserves GitHub's `on` key rather than treating it as YAML 1.1 true.
    return yaml.load((WORKFLOWS / filename).read_text(), Loader=yaml.BaseLoader)


def live_steps():
    return load_workflow("dogfood-live.yml")["jobs"]["live"]["steps"]


def test_paid_provider_secret_is_only_in_manual_live_workflow():
    for path in WORKFLOWS.glob("*.yml"):
        text = path.read_text()
        if "secrets.OPENAI_API_KEY" not in text:
            continue
        assert path.name == "dogfood-live.yml"
        assert set(load_workflow(path.name)["on"]) == {"workflow_dispatch"}
    assert "secrets.OPENAI_API_KEY" in (WORKFLOWS / "dogfood-live.yml").read_text()


def test_live_workflow_requires_explicit_consent_before_checkout_or_credentials():
    workflow = load_workflow("dogfood-live.yml")
    consent = workflow["on"]["workflow_dispatch"]["inputs"]["confirm_paid_api"]
    assert consent == {
        "description": "Run live OpenAI tests using paid API credits",
        "type": "boolean", "required": "true", "default": "false",
    }
    assert "OPENAI_API_KEY" not in workflow.get("env", {})
    job = workflow["jobs"]["live"]
    assert "OPENAI_API_KEY" not in job.get("env", {})
    steps = job["steps"]
    assert steps[0]["id"] == "consent"
    assert "env" not in steps[1]  # Checkout follows consent without provider credentials.
    paid_steps = [step for step in steps if "OPENAI_API_KEY" in step.get("env", {})]
    assert {step["id"] for step in paid_steps} == {"provider", "pytest_llm", "dogfood"}
    assert not any("always()" in step.get("if", "") for step in paid_steps)


@pytest.mark.parametrize("ref,consent,allowed", [
    ("refs/heads/main", "true", True),
    ("refs/heads/main", "false", False),
    ("refs/heads/main", "", False),
    ("refs/heads/feature", "true", False),
    ("refs/pull/1/merge", "true", False),
    ("refs/tags/main", "true", False),
])
def test_actual_consent_gate_rejects_unapproved_runs(tmp_path, ref, consent, allowed):
    gate = live_steps()[0]
    summary = tmp_path / "summary.md"
    result = subprocess.run(
        ["bash", "--noprofile", "--norc", "-eo", "pipefail", "-c", gate["run"]],
        env={**os.environ, "GITHUB_REF": ref, "CONFIRM_PAID_API": consent,
             "GITHUB_STEP_SUMMARY": str(summary)},
        cwd=tmp_path, text=True, capture_output=True,
    )
    assert (result.returncode == 0) is allowed
    if not allowed:
        assert "NOT RUN" in summary.read_text()
        assert "No provider requests were made" in result.stdout


def test_live_execution_is_bounded_and_fails_fast_without_whole_agent_retries():
    workflow = load_workflow("dogfood-live.yml")
    assert workflow["concurrency"] == {
        "group": "live-provider-checks", "cancel-in-progress": "false",
    }
    job = workflow["jobs"]["live"]
    assert int(job["timeout-minutes"]) == 25
    steps = {step["id"]: step for step in job["steps"] if "id" in step}
    assert int(steps["provider"]["timeout-minutes"]) <= 1
    assert int(steps["pytest_llm"]["timeout-minutes"]) <= 5
    assert int(steps["dogfood"]["timeout-minutes"]) <= 8
    assert "steps.provider.outcome == 'success'" in steps["pytest_llm"]["if"]
    assert "-x -m requires_api_key" in steps["pytest_llm"]["run"]
    assert "steps.pytest_llm.outcome == 'success'" in steps["dogfood"]["if"]
    assert "--sequential" in steps["dogfood"]["run"]
    assert "--max-retries 0" in steps["dogfood"]["run"]
    # Setup plus paid work leaves six minutes for reporting and runner overhead.
    before_reporting = job["steps"][1:job["steps"].index(steps["check"])]
    assert sum(int(step["timeout-minutes"]) for step in before_reporting) <= 19


def test_live_evidence_and_failure_gate_cannot_be_silently_skipped():
    steps = live_steps()
    summary = next(step for step in steps if step.get("id") == "check")
    assert summary["env"]["DOGFOOD_SCOPE"] == "live"
    assert "always()" in summary["if"]
    assert "steps.consent.outcome == 'success'" in summary["if"]
    assert steps[-1]["if"] == "always()"
    assert steps[-1]["run"] == 'test "$HAS_FAILURES" = false'
    artifacts = next(step for step in steps if step.get("uses", "").startswith("actions/upload-artifact@"))
    assert artifacts["with"]["name"].startswith("live-dogfood-evidence-")
    issue = next(step for step in steps if step.get("uses", "").startswith("actions/github-script@"))
    assert issue["env"]["DOGFOOD_SCOPE"] == "live"
    assert "steps.consent.outcome == 'success'" in issue["if"]
    assert "github.ref == 'refs/heads/main'" in issue["if"]


def test_standard_ci_excludes_live_tests_explicitly():
    workflow = load_workflow("ci.yml")
    test_command = next(
        step["run"] for step in workflow["jobs"]["test"]["steps"]
        if step.get("name") == "Run tests with pytest"
    )
    assert "-m 'not requires_api_key'" in test_command
    assert "secrets.OPENAI_API_KEY" not in (WORKFLOWS / "ci.yml").read_text()


def test_shell_steps_use_valid_bash_syntax():
    for step in live_steps():
        if "run" in step:
            result = subprocess.run(["bash", "-n"], input=step["run"], text=True, capture_output=True)
            assert result.returncode == 0, (step.get("name"), result.stderr)
