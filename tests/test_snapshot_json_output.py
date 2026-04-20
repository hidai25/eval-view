"""Tests for `evalview snapshot --json` CI output contract."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from click.testing import CliRunner

from evalview.core.types import (
    ContainsChecks,
    CostEvaluation,
    EvaluationResult,
    Evaluations,
    ExecutionMetrics,
    ExecutionTrace,
    LatencyEvaluation,
    OutputEvaluation,
    SequenceEvaluation,
    ToolEvaluation,
)


def _result(name: str, *, passed: bool, score: float = 90.0) -> EvaluationResult:
    now = datetime.now()
    return EvaluationResult(
        test_case=name,
        passed=passed,
        score=score,
        evaluations=Evaluations(
            tool_accuracy=ToolEvaluation(accuracy=1.0 if passed else 0.0),
            sequence_correctness=SequenceEvaluation(
                correct=passed, expected_sequence=[], actual_sequence=[]
            ),
            output_quality=OutputEvaluation(
                score=score,
                rationale="ok",
                contains_checks=ContainsChecks(),
                not_contains_checks=ContainsChecks(),
            ),
            cost=CostEvaluation(total_cost=0.0, threshold=1.0, passed=True),
            latency=LatencyEvaluation(total_latency=10.0, threshold=1000.0, passed=True),
        ),
        trace=ExecutionTrace(
            session_id="s1",
            start_time=now,
            end_time=now,
            steps=[],
            final_output="ok",
            metrics=ExecutionMetrics(total_cost=0.0, total_latency=10.0),
        ),
        timestamp=now,
    )


def _write_yaml(path: Path, name: str) -> None:
    path.write_text(
        f"""name: {name}
input:
  query: hi
expected:
  output:
    contains:
      - ok
thresholds:
  min_score: 0
""",
        encoding="utf-8",
    )


def _patch_pipeline(monkeypatch, results):
    """Stub out the slow moving parts so the command runs deterministically."""
    from evalview.commands import snapshot_cmd

    monkeypatch.setattr(snapshot_cmd, "_load_config_if_exists", lambda: None)
    monkeypatch.setattr(
        snapshot_cmd,
        "_execute_snapshot_tests",
        lambda test_cases, config, **kwargs: list(results),
    )
    monkeypatch.setattr(snapshot_cmd, "_cloud_push", lambda saved_names: None)
    monkeypatch.setattr(
        "evalview.core.project_state.ProjectStateStore.is_first_snapshot",
        lambda self: False,
    )
    monkeypatch.setattr(
        "evalview.core.project_state.ProjectStateStore.update_snapshot",
        lambda self, test_count=1: None,
    )


def test_snapshot_json_emits_parseable_payload(monkeypatch, tmp_path):
    """`--json` must put a single parseable JSON document on stdout with no Rich output."""
    from evalview.commands.snapshot_cmd import snapshot

    monkeypatch.chdir(tmp_path)
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    _write_yaml(tests_dir / "alpha.yaml", "alpha")
    _write_yaml(tests_dir / "beta.yaml", "beta")

    saved_paths: dict[str, Path] = {}

    def fake_save(self, result, notes=None, variant_name=None):
        path = tmp_path / ".evalview" / "golden" / f"{result.test_case}.golden.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}", encoding="utf-8")
        saved_paths[result.test_case] = path
        return path

    monkeypatch.setattr("evalview.core.golden.GoldenStore.save_golden", fake_save)
    _patch_pipeline(
        monkeypatch,
        [_result("alpha", passed=True), _result("beta", passed=False, score=10.0)],
    )

    runner = CliRunner()
    result = runner.invoke(snapshot, ["--path", "tests", "--json"])

    assert result.exit_code == 0, result.output

    # Stdout must be a single, parseable JSON document — no banner, spinner,
    # or Rich markup before/after it.
    payload = json.loads(result.output)

    assert payload["snapshot"]["total_tests"] == 2
    assert payload["snapshot"]["passing"] == 1
    assert payload["snapshot"]["saved"] == 1
    assert payload["snapshot"]["test_path"] == "tests"

    by_name = {t["name"]: t for t in payload["tests"]}
    assert by_name["alpha"]["passed"] is True
    assert by_name["alpha"]["saved"] is True
    assert by_name["alpha"]["golden_file"] == str(saved_paths["alpha"])
    assert by_name["beta"]["passed"] is False
    assert by_name["beta"]["saved"] is False
    assert by_name["beta"]["golden_file"] is None

    # No banner/spinner/Rich markers should leak into stdout.
    assert "Catch agent regressions" not in result.output
    assert "Snapshotting" not in result.output
    assert "✓" not in result.output


def test_snapshot_json_marks_failed_saves_accurately(monkeypatch, tmp_path):
    """If save_golden raises for one passing test, that test's `saved` must be False."""
    from evalview.commands.snapshot_cmd import snapshot

    monkeypatch.chdir(tmp_path)
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    _write_yaml(tests_dir / "good.yaml", "good")
    _write_yaml(tests_dir / "broken.yaml", "broken")

    def flaky_save(self, result, notes=None, variant_name=None):
        if result.test_case == "broken":
            raise RuntimeError("disk full")
        path = tmp_path / ".evalview" / "golden" / f"{result.test_case}.golden.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}", encoding="utf-8")
        return path

    monkeypatch.setattr("evalview.core.golden.GoldenStore.save_golden", flaky_save)
    _patch_pipeline(
        monkeypatch,
        [_result("good", passed=True), _result("broken", passed=True)],
    )

    runner = CliRunner()
    result = runner.invoke(snapshot, ["--path", "tests", "--json"])

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    by_name = {t["name"]: t for t in payload["tests"]}

    assert payload["snapshot"]["passing"] == 2
    assert payload["snapshot"]["saved"] == 1
    assert by_name["good"]["saved"] is True
    assert by_name["good"]["golden_file"] is not None
    assert by_name["broken"]["saved"] is False
    assert by_name["broken"]["golden_file"] is None


def test_snapshot_json_uses_variant_aware_golden_path(monkeypatch, tmp_path):
    """Per-test `golden_file` must match the path GoldenStore actually wrote to."""
    from evalview.commands.snapshot_cmd import snapshot

    monkeypatch.chdir(tmp_path)
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    _write_yaml(tests_dir / "alpha.yaml", "alpha")

    written: dict[str, Path] = {}

    def fake_save(self, result, notes=None, variant_name=None):
        suffix = f".variant_{variant_name}" if variant_name else ""
        path = tmp_path / ".evalview" / "golden" / f"{result.test_case}{suffix}.golden.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}", encoding="utf-8")
        written[result.test_case] = path
        return path

    monkeypatch.setattr("evalview.core.golden.GoldenStore.save_golden", fake_save)
    _patch_pipeline(monkeypatch, [_result("alpha", passed=True)])

    runner = CliRunner()
    result = runner.invoke(snapshot, ["--path", "tests", "--json", "--variant", "slow"])

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    alpha = payload["tests"][0]
    assert alpha["golden_file"] == str(written["alpha"])
    assert "variant_slow" in alpha["golden_file"]
    assert payload["snapshot"]["variant"] == "slow"


def test_snapshot_json_rejects_preview_combo(monkeypatch, tmp_path):
    """`--preview --json` is mutually exclusive — must exit non-zero with JSON error."""
    from evalview.commands.snapshot_cmd import snapshot

    monkeypatch.chdir(tmp_path)
    (tmp_path / "tests").mkdir()

    runner = CliRunner()
    result = runner.invoke(snapshot, ["--path", "tests", "--json", "--preview"])

    assert result.exit_code != 0
    payload = json.loads(result.output)
    assert "error" in payload
    assert "preview" in payload["error"].lower()


def test_snapshot_json_no_tests_emits_error_payload(monkeypatch, tmp_path):
    """Empty suite should yield a parseable JSON error rather than Rich output."""
    from evalview.commands.snapshot_cmd import snapshot

    monkeypatch.chdir(tmp_path)
    (tmp_path / "tests").mkdir()

    runner = CliRunner()
    result = runner.invoke(snapshot, ["--path", "tests", "--json"])

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload == {"error": "no tests found"}
