"""Tests for check command edge cases."""

from __future__ import annotations

from datetime import datetime

from click.testing import CliRunner


def test_check_dry_run_handles_golden_metadata_objects(monkeypatch, tmp_path):
    """Dry-run should count baselines by name without hashing metadata models."""
    from evalview.commands.check_cmd import check
    from evalview.core.golden import GoldenMetadata

    project = tmp_path
    monkeypatch.chdir(project)

    tests_dir = project / "tests"
    tests_dir.mkdir()
    (tests_dir / "sample.yaml").write_text(
        "name: sample\ninput:\n  query: hi\nexpected:\n  tools: []\nthresholds:\n  min_score: 0\n",
        encoding="utf-8",
    )

    evalview_dir = project / ".evalview"
    evalview_dir.mkdir()
    (evalview_dir / "config.yaml").write_text(
        "adapter: http\nendpoint: http://example.com\n",
        encoding="utf-8",
    )

    runner = CliRunner()

    monkeypatch.setattr(
        "evalview.commands.check_cmd._cloud_pull",
        lambda store: None,
    )
    monkeypatch.setattr(
        "evalview.commands.check_cmd._load_config_if_exists",
        lambda: None,
    )
    monkeypatch.setattr(
        "evalview.core.golden.GoldenStore.list_golden",
        lambda self: [
            GoldenMetadata(
                test_name="sample",
                blessed_at="2026-03-13T00:00:00Z",
                score=95.0,
            )
        ],
    )

    result = runner.invoke(check, ["tests", "--dry-run"])

    assert result.exit_code == 0
    assert "With baselines: 1" in result.output


def test_check_does_not_report_clean_when_execution_failures_occur(monkeypatch, tmp_path):
    """Execution failures should fail check even if no diffs were produced."""
    from evalview.commands.check_cmd import check
    from evalview.core.golden import GoldenMetadata
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

    project = tmp_path
    monkeypatch.chdir(project)

    tests_dir = project / "tests"
    tests_dir.mkdir()
    (tests_dir / "sample.yaml").write_text(
        "name: sample\ninput:\n  query: hi\nexpected:\n  tools: []\nthresholds:\n  min_score: 0\n",
        encoding="utf-8",
    )
    (tests_dir / "stale.yaml").write_text(
        "name: stale\nadapter: mistral\nendpoint: http://localhost:8090/execute\ninput:\n  query: hi\nexpected:\n  tools: []\nthresholds:\n  min_score: 0\n",
        encoding="utf-8",
    )

    now = datetime.now()
    sample_result = EvaluationResult(
        test_case="sample",
        passed=True,
        score=90.0,
        evaluations=Evaluations(
            tool_accuracy=ToolEvaluation(accuracy=1.0),
            sequence_correctness=SequenceEvaluation(correct=True, expected_sequence=[], actual_sequence=[]),
            output_quality=OutputEvaluation(
                score=90.0,
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

    runner = CliRunner()

    monkeypatch.setattr("evalview.commands.check_cmd._cloud_pull", lambda store: None)
    monkeypatch.setattr("evalview.commands.check_cmd._load_config_if_exists", lambda: None)
    monkeypatch.setattr(
        "evalview.core.golden.GoldenStore.list_golden",
        lambda self: [
            GoldenMetadata(test_name="sample", blessed_at="2026-03-13T00:00:00Z", score=95.0),
            GoldenMetadata(test_name="stale", blessed_at="2026-03-13T00:00:00Z", score=95.0),
        ],
    )
    monkeypatch.setattr(
        "evalview.commands.check_cmd._execute_check_tests",
        lambda test_cases, config, json_output, semantic_diff=False, timeout=30.0, skip_llm_judge=False, budget_tracker=None: ([], [sample_result], None, {}),
    )

    # Provide input for interactive judge picker + skip it via env var
    monkeypatch.setenv("EVAL_MODEL", "gpt-5.4-mini")
    result = runner.invoke(check, ["tests"])

    assert result.exit_code == 1
    assert "Everything matches the baseline" not in result.output
    assert "execution failure" in result.output


def test_check_uses_active_test_path_when_no_path_is_given(monkeypatch, tmp_path):
    """Plain `check` should follow the remembered active suite instead of raw tests/."""
    from evalview.commands.check_cmd import check
    from evalview.core.project_state import ProjectStateStore
    from evalview.core.golden import GoldenMetadata

    monkeypatch.chdir(tmp_path)
    active_dir = tmp_path / "tests" / "generated-from-init"
    active_dir.mkdir(parents=True)
    (active_dir / "sample.yaml").write_text(
        "name: sample\ninput:\n  query: hi\nexpected:\n  tools: []\nthresholds:\n  min_score: 0\n",
        encoding="utf-8",
    )
    ProjectStateStore().set_active_test_path("tests/generated-from-init")

    runner = CliRunner()
    captured = {}

    monkeypatch.setattr("evalview.commands.check_cmd._cloud_pull", lambda store: None)
    monkeypatch.setattr("evalview.commands.check_cmd._load_config_if_exists", lambda: None)
    monkeypatch.setattr(
        "evalview.core.golden.GoldenStore.list_golden",
        lambda self: [GoldenMetadata(test_name="sample", blessed_at="2026-03-13T00:00:00Z", score=95.0)],
    )

    def _fake_execute(test_cases, config, json_output, semantic_diff=False, timeout=30.0, skip_llm_judge=False, budget_tracker=None):
        captured["names"] = [tc.name for tc in test_cases]
        return [], [], None, {}

    monkeypatch.setattr("evalview.commands.check_cmd._execute_check_tests", _fake_execute)

    result = runner.invoke(check, [])

    assert result.exit_code == 1
    assert captured["names"] == ["sample"]


def test_check_auto_generates_html_report_on_failures(monkeypatch, tmp_path):
    """Local failing checks should auto-write a browser report without extra flags."""
    from evalview.commands.check_cmd import check
    from evalview.core.diff import DiffStatus
    from evalview.core.golden import GoldenMetadata
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

    project = tmp_path
    monkeypatch.chdir(project)
    monkeypatch.delenv("CI", raising=False)

    tests_dir = project / "tests"
    tests_dir.mkdir()
    (tests_dir / "sample.yaml").write_text(
        "name: sample\ninput:\n  query: hi\nexpected:\n  tools: []\nthresholds:\n  min_score: 0\n",
        encoding="utf-8",
    )

    now = datetime.now()
    sample_result = EvaluationResult(
        test_case="sample",
        passed=False,
        score=55.0,
        evaluations=Evaluations(
            tool_accuracy=ToolEvaluation(accuracy=0.0),
            sequence_correctness=SequenceEvaluation(correct=False, expected_sequence=[], actual_sequence=[]),
            output_quality=OutputEvaluation(
                score=55.0,
                rationale="changed",
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
            final_output="changed",
            metrics=ExecutionMetrics(total_cost=0.0, total_latency=10.0),
        ),
        timestamp=now,
    )

    class _Diff:
        overall_severity = DiffStatus.REGRESSION
        score_diff = -30.0
        tool_diffs = []
        output_diff = None

    called = {}
    runner = CliRunner()

    monkeypatch.setattr("evalview.commands.check_cmd._cloud_pull", lambda store: None)
    monkeypatch.setattr("evalview.commands.check_cmd._load_config_if_exists", lambda: None)
    monkeypatch.setattr(
        "evalview.core.golden.GoldenStore.list_golden",
        lambda self: [GoldenMetadata(test_name="sample", blessed_at="2026-03-13T00:00:00Z", score=95.0)],
    )
    monkeypatch.setattr(
        "evalview.commands.check_cmd._execute_check_tests",
        lambda test_cases, config, json_output, semantic_diff=False, timeout=30.0, skip_llm_judge=False, budget_tracker=None: ([("sample", _Diff())], [sample_result], None, {}),
    )
    monkeypatch.setattr(
        "evalview.commands.check_cmd._display_check_results",
        lambda *args, **kwargs: None,
    )
    def _fake_generate_visual_report(**kwargs):
        called["path"] = kwargs["output_path"]
        called["healing"] = kwargs.get("healing_summary")
        called["effective"] = kwargs.get("effective_all_passed")
        called["active_tags"] = kwargs.get("active_tags")
        called["test_metadata"] = kwargs.get("test_metadata")
        return kwargs["output_path"]

    monkeypatch.setattr(
        "evalview.visualization.generate_visual_report",
        _fake_generate_visual_report,
    )

    result = runner.invoke(check, ["tests"])

    assert result.exit_code == 1
    assert called["path"] == ".evalview/latest-check.html"
    assert called["effective"] is False
    assert called["active_tags"] == []
    assert called["test_metadata"]["sample"]["tags"] == []
    assert "Failure report:" in result.output


def test_check_passes_tag_context_to_html_report(monkeypatch, tmp_path):
    """HTML reports should receive active behavior tags and per-test tag metadata."""
    from evalview.commands.check_cmd import check
    from evalview.core.diff import DiffStatus
    from evalview.core.golden import GoldenMetadata
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

    project = tmp_path
    monkeypatch.chdir(project)
    monkeypatch.delenv("CI", raising=False)

    tests_dir = project / "tests"
    tests_dir.mkdir()
    (tests_dir / "sample.yaml").write_text(
        "name: sample\ntags:\n  - tool_use\ninput:\n  query: hi\nexpected:\n  tools: []\nthresholds:\n  min_score: 0\n",
        encoding="utf-8",
    )

    now = datetime.now()
    sample_result = EvaluationResult(
        test_case="sample",
        passed=False,
        score=55.0,
        evaluations=Evaluations(
            tool_accuracy=ToolEvaluation(accuracy=0.0),
            sequence_correctness=SequenceEvaluation(correct=False, expected_sequence=[], actual_sequence=[]),
            output_quality=OutputEvaluation(
                score=55.0,
                rationale="changed",
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
            final_output="changed",
            metrics=ExecutionMetrics(total_cost=0.0, total_latency=10.0),
        ),
        timestamp=now,
    )

    class _Diff:
        overall_severity = DiffStatus.REGRESSION
        score_diff = -30.0
        tool_diffs = []
        output_diff = None
        model_changed = False

    called = {}
    runner = CliRunner()

    monkeypatch.setattr("evalview.commands.check_cmd._cloud_pull", lambda store: None)
    monkeypatch.setattr("evalview.commands.check_cmd._load_config_if_exists", lambda: None)
    monkeypatch.setattr(
        "evalview.core.golden.GoldenStore.list_golden",
        lambda self: [GoldenMetadata(test_name="sample", blessed_at="2026-03-13T00:00:00Z", score=95.0)],
    )
    monkeypatch.setattr(
        "evalview.commands.check_cmd._execute_check_tests",
        lambda test_cases, config, json_output, semantic_diff=False, timeout=30.0, skip_llm_judge=False, budget_tracker=None: ([("sample", _Diff())], [sample_result], None, {}),
    )
    monkeypatch.setattr("evalview.commands.check_cmd._display_check_results", lambda *args, **kwargs: None)

    def _fake_generate_visual_report(**kwargs):
        called["active_tags"] = kwargs.get("active_tags")
        called["test_metadata"] = kwargs.get("test_metadata")
        return kwargs["output_path"]

    monkeypatch.setattr("evalview.visualization.generate_visual_report", _fake_generate_visual_report)

    result = runner.invoke(check, ["tests", "--tag", "tool_use"])

    assert result.exit_code == 1
    assert called["active_tags"] == ["tool_use"]
    assert called["test_metadata"]["sample"]["tags"] == ["tool_use"]


def test_check_warns_when_heal_leaves_unresolved_but_default_fail_policy_exits_zero(monkeypatch, tmp_path):
    """`check --heal` should explain zero exit under REGRESSION-only fail policy."""
    from evalview.commands.check_cmd import check
    from evalview.core.diff import DiffStatus
    from evalview.core.golden import GoldenMetadata, GoldenTrace
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

    project = tmp_path
    monkeypatch.chdir(project)
    tests_dir = project / "tests"
    tests_dir.mkdir()
    (tests_dir / "sample.yaml").write_text(
        "name: sample\nadapter: http\nendpoint: http://example.com\ninput:\n  query: hi\nexpected:\n  tools: []\nthresholds:\n  min_score: 0\n",
        encoding="utf-8",
    )

    now = datetime.now()
    sample_result = EvaluationResult(
        test_case="sample",
        passed=True,
        score=92.0,
        evaluations=Evaluations(
            tool_accuracy=ToolEvaluation(accuracy=1.0),
            sequence_correctness=SequenceEvaluation(correct=True, expected_sequence=[], actual_sequence=[]),
            output_quality=OutputEvaluation(
                score=92.0,
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
            model_id="gpt-4o-mini",
        ),
        timestamp=now,
    )

    class _Diff:
        overall_severity = DiffStatus.TOOLS_CHANGED
        score_diff = 2.0
        tool_diffs = [object()]
        output_diff = None
        model_changed = False

    runner = CliRunner()
    monkeypatch.setattr("evalview.commands.check_cmd._cloud_pull", lambda store: None)
    monkeypatch.setattr("evalview.commands.check_cmd._load_config_if_exists", lambda: None)
    monkeypatch.setattr(
        "evalview.core.golden.GoldenStore.list_golden",
        lambda self: [GoldenMetadata(test_name="sample", blessed_at="2026-03-13T00:00:00Z", score=95.0)],
    )
    monkeypatch.setattr(
        "evalview.commands.check_cmd._execute_check_tests",
        lambda test_cases, config, json_output, semantic_diff=False, timeout=30.0, skip_llm_judge=False, budget_tracker=None: ([("sample", _Diff())], [sample_result], None, {"sample": GoldenTrace(metadata=GoldenMetadata(test_name="sample", blessed_at=now, score=95.0, model_id="gpt-4o-mini"), trace=sample_result.trace, tool_sequence=[], output_hash="abc")}),
    )
    monkeypatch.setattr("evalview.commands.check_cmd._display_check_results", lambda *args, **kwargs: None)
    monkeypatch.setattr("evalview.commands.check_cmd._should_auto_generate_report", lambda **kwargs: False)
    monkeypatch.setattr("evalview.commands.badge_cmd.update_badge_after_check", lambda *args, **kwargs: None)

    result = runner.invoke(check, ["tests", "--heal"])

    assert result.exit_code == 0
    assert "Unresolved healing review items remain" in result.output
    assert "--strict or --fail-on REGRESSION,TOOLS_CHANGED,OUTPUT_CHANGED" in result.output


def test_check_shows_last_snapshot_timestamp(monkeypatch, tmp_path):
    """Human check output should show when the current baseline was last snapshotted."""
    from evalview.commands.check_cmd import check
    from evalview.core.golden import GoldenMetadata
    from evalview.core.project_state import ProjectState

    monkeypatch.chdir(tmp_path)
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    (tests_dir / "sample.yaml").write_text(
        "name: sample\ninput:\n  query: hi\nexpected:\n  tools: []\nthresholds:\n  min_score: 0\n",
        encoding="utf-8",
    )

    runner = CliRunner()

    monkeypatch.setattr("evalview.commands.check_cmd._cloud_pull", lambda store: None)
    monkeypatch.setattr("evalview.commands.check_cmd._load_config_if_exists", lambda: None)
    monkeypatch.setattr(
        "evalview.core.golden.GoldenStore.list_golden",
        lambda self: [GoldenMetadata(test_name="sample", blessed_at="2026-03-13T00:00:00Z", score=95.0)],
    )
    monkeypatch.setattr(
        "evalview.core.project_state.ProjectStateStore.load",
        lambda self: ProjectState(
            last_snapshot_at=datetime(2026, 3, 14, 9, 45),
            total_snapshots=1,
        ),
    )

    result = runner.invoke(check, ["tests", "--dry-run"])

    assert result.exit_code == 0
    assert "1 baseline" in result.output
    assert "snapshot:" in result.output
    assert "2026-03-1" in result.output  # date from blessed_at


def test_check_dry_run_filters_by_tag(monkeypatch, tmp_path):
    """Dry-run should honor behavior tag filters and report the active slice."""
    from evalview.commands.check_cmd import check
    from evalview.core.golden import GoldenMetadata

    monkeypatch.chdir(tmp_path)
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    (tests_dir / "tool.yaml").write_text(
        "name: tool-test\ntags:\n  - tool_use\ninput:\n  query: hi\nexpected:\n  tools: []\nthresholds:\n  min_score: 0\n",
        encoding="utf-8",
    )
    (tests_dir / "memory.yaml").write_text(
        "name: memory-test\ntags:\n  - memory\ninput:\n  query: hi\nexpected:\n  tools: []\nthresholds:\n  min_score: 0\n",
        encoding="utf-8",
    )

    runner = CliRunner()
    monkeypatch.setattr("evalview.commands.check_cmd._cloud_pull", lambda store: None)
    monkeypatch.setattr("evalview.commands.check_cmd._load_config_if_exists", lambda: None)
    monkeypatch.setattr(
        "evalview.core.golden.GoldenStore.list_golden",
        lambda self: [
            GoldenMetadata(test_name="tool-test", blessed_at="2026-03-13T00:00:00Z", score=95.0),
            GoldenMetadata(test_name="memory-test", blessed_at="2026-03-13T00:00:00Z", score=95.0),
        ],
    )

    result = runner.invoke(check, ["tests", "--dry-run", "--tag", "tool_use"])

    assert result.exit_code == 0
    assert "Tests:          1" in result.output
    assert "Tags:           tool_use" in result.output


def test_check_exits_when_no_tests_match_tag(monkeypatch, tmp_path):
    """Behavior-tag filters should fail fast when nothing matches."""
    from evalview.commands.check_cmd import check
    from evalview.core.golden import GoldenMetadata

    monkeypatch.chdir(tmp_path)
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    (tests_dir / "sample.yaml").write_text(
        "name: sample\ntags:\n  - retrieval\ninput:\n  query: hi\nexpected:\n  tools: []\nthresholds:\n  min_score: 0\n",
        encoding="utf-8",
    )

    runner = CliRunner()
    monkeypatch.setattr("evalview.commands.check_cmd._cloud_pull", lambda store: None)
    monkeypatch.setattr("evalview.commands.check_cmd._load_config_if_exists", lambda: None)
    monkeypatch.setattr(
        "evalview.core.golden.GoldenStore.list_golden",
        lambda self: [GoldenMetadata(test_name="sample", blessed_at="2026-03-13T00:00:00Z", score=95.0)],
    )

    result = runner.invoke(check, ["tests", "--tag", "tool_use"])

    assert result.exit_code == 1
    assert "No tests matched tags: tool_use" in result.output
