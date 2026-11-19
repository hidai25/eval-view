"""Release checks must enforce forbidden actions independently of baseline similarity."""

import json
from datetime import datetime, timezone
from unittest.mock import AsyncMock

import pytest
import yaml
from click.testing import CliRunner

from evalview.api import gate
from evalview.commands.check_cmd import check
from evalview.commands.snapshot_cmd import snapshot
from evalview.core.diff import DiffStatus
from evalview.core.golden import GoldenStore
from evalview.core.quarantine import QuarantineStore
from evalview.core.types import ExecutionTrace


def trace(tool, output="Refund processed."):
    now = datetime.now(timezone.utc)
    return ExecutionTrace.model_validate({
        "session_id": "session",
        "start_time": now,
        "end_time": now,
        "steps": [{
            "step_id": "1", "step_name": tool, "tool_name": tool,
            "parameters": {}, "output": "ok", "success": True,
            "metrics": {"cost": 0, "latency": 10},
        }],
        "final_output": output,
        "metrics": {"total_cost": 0, "total_latency": 10},
    })


@pytest.fixture
def project(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("CI", "1")
    monkeypatch.setenv("EVALVIEW_TELEMETRY_DISABLED", "1")
    (tmp_path / "tests").mkdir()
    (tmp_path / ".evalview").mkdir()
    (tmp_path / ".evalview/config.yaml").write_text("adapter: http\nendpoint: http://localhost/execute\n")
    spec = {
        "name": "refund", "input": {"query": "Refund this order"},
        "expected": {}, "thresholds": {"min_score": 0},
    }
    path = tmp_path / "tests/refund.yaml"
    path.write_text(yaml.safe_dump(spec))
    adapter = AsyncMock()
    monkeypatch.setattr("evalview.commands.shared._build_adapter_for_tc", lambda *a, **k: adapter)
    monkeypatch.setattr("evalview.commands.check_cmd._create_adapter", lambda *a, **k: adapter)
    return path, spec, adapter


def capture(adapter, tool, *options):
    adapter.execute.return_value = trace(tool)
    result = CliRunner().invoke(snapshot, ["--no-judge", "--json", *options])
    assert result.exit_code == 0, (result.exception, result.output)


@pytest.mark.parametrize("variant", [False, True])
def test_forbidden_matching_baseline_blocks_cli_sdk_and_report(project, tmp_path, variant):
    path, spec, adapter = project
    if variant:
        capture(adapter, "lookup")
        capture(adapter, "issue_refund", "--variant", "refund-route")
    else:
        capture(adapter, "issue_refund")
    control = CliRunner().invoke(check, ["--strict", "--no-judge", "--json"])
    assert control.exit_code == 0, (control.exception, control.output)

    spec["expected"]["forbidden_tools"] = ["issue_refund"]
    path.write_text(yaml.safe_dump(spec))
    QuarantineStore().add("refund", "testing quarantine precedence", owner="team")
    adapter.execute.reset_mock()

    result = CliRunner().invoke(check, [
        "--no-judge", "--json", "--heal", "--report", "report.html",
        *(["--strict"] if not variant else ["--fail-on", "OUTPUT_CHANGED", "--budget", "1"]),
    ])
    assert result.exit_code == 1, (result.exception, result.output)
    data = json.loads(result.output)
    assert data["verdict"]["verdict"] == "block_release"
    assert data["summary"]["effective_all_passed"] is False
    assert data["diffs"][0]["status"] == "regression"
    assert data["diffs"][0]["score_delta"] == 0
    assert data["diffs"][0]["forbidden_tool_violations"] == ["issue_refund"]
    assert data["diffs"][0]["root_cause"]["category"] == "forbidden_tool"
    assert data["healing"]["total_blocked"] == 1
    assert adapter.execute.await_count == 1
    assert "Forbidden tool called: issue_refund" in (tmp_path / "report.html").read_text()

    sdk = gate(test_dir="tests", quick=True, fail_on={DiffStatus.OUTPUT_CHANGED})
    assert not sdk.passed and sdk.exit_code == 1
    assert sdk.diffs[0].status == DiffStatus.REGRESSION
    assert sdk.raw_json["diffs"][0]["forbidden_tool_violations"] == ["issue_refund"]


def test_violation_on_healing_retry_preserves_original_diff_and_blocks_release(project, tmp_path):
    path, spec, adapter = project
    capture(adapter, "lookup")
    capture(adapter, "issue_refund", "--variant", "former-route")
    spec["expected"]["forbidden_tools"] = ["issue_refund"]
    path.write_text(yaml.safe_dump(spec))
    adapter.execute.side_effect = [trace("lookup", "A different answer"), trace("issue_refund")]

    result = CliRunner().invoke(check, ["--no-judge", "--heal", "--json", "--fail-on", "TOOLS_CHANGED", "--report", "retry.html"])
    assert result.exit_code == 1, (result.exception, result.output)
    data = json.loads(result.output)
    assert data["diffs"][0]["status"] == "output_changed"
    assert data["diffs"][0]["forbidden_tool_violations"] == []
    assert data["healing"]["total_blocked"] == 1
    retry = data["healing"]["results"][0]
    assert retry["attempted"] and not retry["healed"] and not retry["proposed"]
    assert retry["diagnosis"]["details"]["violations"] == ["issue_refund"]
    assert retry["final_status"] == "regression"
    assert data["verdict"]["verdict"] == "block_release"
    assert any("issue_refund" in reason for reason in data["verdict"]["reasons"])
    assert GoldenStore().count_variants("refund") == 2
    html = (tmp_path / "retry.html").read_text()
    assert "Retry called forbidden tool: issue_refund" in html
    assert "Final Outcome Failing" in html
    assert "Accept: <code>" not in html


def test_turn_specific_forbidden_tool_blocks_unchanged_workflow(project):
    path, spec, adapter = project
    adapter.execute.side_effect = lambda *args: trace("issue_refund")
    spec["turns"] = [{"query": "Preview the refund"}, {"query": "Confirm the refund"}]
    path.write_text(yaml.safe_dump(spec))
    capture(adapter, "issue_refund")
    spec["turns"][0]["expected"] = {"forbidden_tools": ["issue_refund"]}
    path.write_text(yaml.safe_dump(spec))

    result = CliRunner().invoke(check, ["--no-judge", "--json"])
    assert result.exit_code == 1, (result.exception, result.output)
    data = json.loads(result.output)
    assert data["diffs"][0]["forbidden_tool_violations"] == ["issue_refund"]
    assert data["verdict"]["verdict"] == "block_release"
