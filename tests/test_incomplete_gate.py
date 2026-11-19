"""Cross-entry-point contracts for incomplete checks and repeated runs."""
import asyncio
import json
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest
from click.testing import CliRunner

from evalview.api import DiffStatus, gate
from evalview.commands.check_cmd import check
from evalview.core.golden import GoldenStore
from evalview.core.loader import TestCaseLoader
from evalview.core.types import ExecutionMetrics, ExecutionTrace
from evalview.evaluators.evaluator import Evaluator


def _write_case(directory, name):
    path = directory / f"{name}.yaml"
    path.write_text(
        f"name: {name}\ninput:\n  query: {name}\nexpected:\n  tools: []\n"
        "thresholds:\n  min_score: 0\n",
        encoding="utf-8",
    )
    return TestCaseLoader.load_from_file(path)


@pytest.fixture
def project(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("EVALVIEW_TELEMETRY_DISABLED", "1")
    monkeypatch.setenv("CI", "1")
    for variable in ("EVALVIEW_API_TOKEN", "OPENAI_API_KEY", "ANTHROPIC_API_KEY"):
        monkeypatch.delenv(variable, raising=False)
    directory = tmp_path / "tests"
    directory.mkdir()
    now = datetime.now(timezone.utc)
    baseline = ExecutionTrace(
        session_id="test", start_time=now, end_time=now, steps=[],
        final_output="Refund processed.", metrics=ExecutionMetrics(total_latency=0, total_cost=0.01),
    )
    evaluator = Evaluator(skip_llm_judge=True)
    for name in ("refund", "lookup"):
        case = _write_case(directory, name)
        GoldenStore().save_golden(asyncio.run(evaluator.evaluate(case, baseline)))

    outcomes = {}

    async def execute(query, context):
        response = outcomes.get(query, baseline.final_output)
        if isinstance(response, Exception):
            raise response
        return baseline.model_copy(update={"final_output": response})

    adapter = SimpleNamespace(execute=execute)
    monkeypatch.setattr("evalview.commands.shared._build_adapter_for_tc", lambda *args: adapter)
    return directory, outcomes


def _check(*flags):
    result = CliRunner().invoke(check, ["tests", "--json", "--no-judge", *flags])
    assert result.exit_code in (0, 1), result.output
    return result.exit_code, json.loads(result.output)


@pytest.mark.parametrize("breakage", ["adapter", "missing-baseline"])
def test_partial_suite_fails_closed_across_entry_points(project, breakage):
    directory, outcomes = project
    if breakage == "adapter":
        outcomes["refund"] = RuntimeError("agent unavailable")
        missing = "refund"
        requested = 2
    else:
        _write_case(directory, "extra")
        missing = "extra"
        requested = 3

    sdk = gate(str(directory), quick=True, fail_on=set())
    assert not sdk.passed
    assert sdk.status is DiffStatus.INCOMPLETE
    assert sdk.exit_code == 1
    assert sdk.summary.requested_tests == requested
    assert sdk.summary.total == requested - 1
    assert sdk.summary.execution_failures == 1
    assert not sdk.raw_json["analysis"]["all_passed"]

    for flags, repeats in [([], 1), (["--statistical", "3"], 3)]:
        code, payload = _check(*flags)
        assert code == 1
        assert payload["error"] and payload["status"] == "incomplete"
        assert payload["summary"]["effective_all_passed"] is False
        assert payload["summary"]["execution_failures"] == repeats
        assert payload["summary"]["requested_tests"] == repeats * requested
        assert payload["summary"]["total_tests"] == repeats * (requested - 1)
        if repeats == 1:
            assert payload["missing_tests"] == [missing]
        else:
            assert all(run["missing_tests"] == [missing] for run in payload["runs"])


def test_statistical_repetitions_apply_the_selected_fail_policy(project):
    _, outcomes = project
    code, payload = _check("--statistical", "3", "--strict")
    assert code == 0 and payload["passed"] and payload["summary"]["unchanged"] == 6
    outcomes["refund"] = "I cannot process this request."
    for flags, expected_code in [([], 0), (["--strict"], 1), (["--fail-on", "OUTPUT_CHANGED"], 1)]:
        code, payload = _check("--statistical", "3", *flags)
        assert code == expected_code
        assert payload["passed"] == (expected_code == 0)
        assert payload["summary"]["output_changed"] == 3
        assert len(payload["runs"]) == 3
        assert "flakiness" in payload["statistics"]["refund"]


@pytest.mark.parametrize("include_malformed_file", [False, True])
def test_missing_and_invalid_suites_return_structured_errors(tmp_path, monkeypatch, include_malformed_file):
    monkeypatch.chdir(tmp_path)
    test_dir = tmp_path / "tests"
    if include_malformed_file:
        test_dir.mkdir()
        _write_case(test_dir, "valid")
        (test_dir / "broken.yaml").write_text("name: [\n", encoding="utf-8")
    sdk = gate(str(test_dir), quick=True)
    assert sdk.status is DiffStatus.INCOMPLETE and sdk.exit_code == 1
    code, payload = _check("--statistical", "3")
    assert code == 1 and payload["status"] == "incomplete"
    assert sdk.raw_json["error"] and payload["error"]
    assert not payload["summary"]["effective_all_passed"]


def test_zero_matches_and_zero_completed_runs_fail(project):
    directory, outcomes = project
    renamed = directory.parent / "renamed"
    renamed.mkdir()
    _write_case(renamed, "changed-name")
    runner = CliRunner()
    result = runner.invoke(check, [str(renamed), "--json", "--no-judge"])
    assert result.exit_code == 1
    assert json.loads(result.output)["summary"]["requested_tests"] == 1
    outcomes.update(refund=RuntimeError("down"), lookup=RuntimeError("down"))
    code, payload = _check("--statistical", "3", "--strict")
    assert code == 1 and payload["summary"]["execution_failures"] == 6
    assert payload["summary"]["total_tests"] == 0


@pytest.mark.parametrize(
    ("flags", "requested"),
    [(["--test", "refund"], 1), ([], 2), (["--statistical", "3"], 6)],
)
def test_budget_stop_counts_comparisons_and_blocks_release(project, flags, requested):
    code, payload = _check("--budget", "0.01", *flags)
    assert code == 1 and payload["status"] == "incomplete"
    assert payload["summary"]["requested_tests"] == requested
    assert payload["summary"]["total_tests"] == 1
    assert payload["summary"]["execution_failures"] == requested - 1
    assert not payload["summary"]["effective_all_passed"]
    assert "budget" in payload["error"].lower()
    if "--statistical" not in flags:
        assert payload["verdict"]["verdict"] == "block_release"


def test_invalid_statistical_run_count_returns_json(project):
    code, payload = _check("--statistical", "0")
    assert code == 1 and "at least 3" in payload["error"]


def test_invalid_configuration_reports_the_requested_count(project):
    directory, _ = project
    (directory.parent / ".evalview/config.yaml").write_text("adapter: [\n", encoding="utf-8")
    sdk = gate(str(directory), quick=True)
    assert sdk.status is DiffStatus.INCOMPLETE and sdk.summary.requested_tests == 2
    code, payload = _check("--statistical", "3")
    assert code == 1 and payload["summary"]["requested_tests"] == 6
    assert "configuration" in payload["error"]


def test_openclaw_and_mcp_propagate_incomplete_runs(tmp_path, monkeypatch):
    from evalview.mcp_server import MCPServer
    from evalview.openclaw import check_and_decide, gate_or_revert

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("evalview.openclaw.subprocess.run", lambda *a, **kw: pytest.fail("unexpected revert"))
    missing = str(tmp_path / "missing")
    assert not gate_or_revert(missing, quick=True)
    decision = check_and_decide(missing)
    assert decision.action == "review" and not decision.reverted
    mcp = json.loads(MCPServer()._run_check_direct({"test_path": missing}))
    assert mcp["status"] == "incomplete" and not mcp["summary"]["all_passed"]


def test_fresh_cached_and_auto_updated_badges_show_incomplete_coverage(project):
    from evalview.commands.badge_cmd import _badge_from_check, _badge_from_last_result

    directory, outcomes = project
    outcomes["lookup"] = RuntimeError("agent unavailable")
    fresh = _badge_from_check(quick=True, test_dir=str(directory))
    assert fresh["color"] == "red" and "1/2 compared" in fresh["message"]

    badge_path = directory.parent / ".evalview/badge.json"
    badge_path.write_text("{}", encoding="utf-8")
    code, payload = _check()
    assert code == 1
    assert json.loads(badge_path.read_text()) == fresh
    results = directory.parent / ".evalview/results"
    results.mkdir(exist_ok=True)
    (results / "check-ci.json").write_text(json.dumps(payload), encoding="utf-8")
    assert _badge_from_last_result() == fresh

    _check("--statistical", "3")
    assert "3/6 compared" in json.loads(badge_path.read_text())["message"]
