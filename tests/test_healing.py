"""Tests for the auto-heal engine (evalview check --heal)."""
from __future__ import annotations

import json
import tempfile
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock

import pytest

from evalview.core.diff import (
    DiffEngine,
    DiffStatus,
    OutputDiff,
    ToolDiff,
    TraceDiff,
)
from evalview.core.golden import GoldenMetadata, GoldenStore, GoldenTrace
from evalview.core.healing import (
    HealingAction,
    HealingDiagnosis,
    HealingEngine,
    HealingResult,
    HealingSummary,
    HealingTrigger,
    MAX_AUTO_VARIANTS,
    ModelUpdateSummary,
    save_audit_log,
)
from evalview.core.types import (
    ContainsChecks,
    CostEvaluation,
    EvaluationResult,
    Evaluations,
    ExecutionMetrics,
    ExecutionTrace,
    ForbiddenToolEvaluation,
    LatencyEvaluation,
    OutputEvaluation,
    SequenceEvaluation,
    StepMetrics,
    StepTrace,
    TestCase,
    TestInput,
    ExpectedBehavior,
    Thresholds,
    ToolEvaluation,
)
from evalview.commands.check_cmd import _all_failures_retry_healed


# --- Helpers ---

def _make_step(tool_name: str = "search", params: Optional[Dict[str, Any]] = None) -> StepTrace:
    return StepTrace(
        step_id="1",
        step_name=tool_name,
        tool_name=tool_name,
        parameters=params or {},
        output="ok",
        success=True,
        metrics=StepMetrics(cost=0.01, latency=100),
    )


def _make_trace(
    tools: Optional[List[str]] = None,
    output: str = "hello world",
    cost: float = 0.05,
    latency: float = 500.0,
    model_id: Optional[str] = None,
) -> ExecutionTrace:
    steps = [_make_step(t) for t in (tools or ["search"])]
    return ExecutionTrace(
        session_id="s1",
        start_time=datetime.now(timezone.utc),
        end_time=datetime.now(timezone.utc),
        steps=steps,
        final_output=output,
        metrics=ExecutionMetrics(total_cost=cost, total_latency=latency),
        model_id=model_id,
    )


def _make_golden(
    test_name: str = "test-1",
    tools: Optional[List[str]] = None,
    output: str = "hello world",
    score: float = 85.0,
    cost: float = 0.05,
    latency: float = 500.0,
    model_id: Optional[str] = None,
) -> GoldenTrace:
    trace = _make_trace(tools=tools, output=output, cost=cost, latency=latency, model_id=model_id)
    return GoldenTrace(
        metadata=GoldenMetadata(
            test_name=test_name,
            blessed_at=datetime.now(timezone.utc),
            score=score,
            model_id=model_id,
        ),
        trace=trace,
        tool_sequence=tools or ["search"],
        output_hash="abc123",
    )


def _make_result(
    test_name: str = "test-1",
    score: float = 85.0,
    tools: Optional[List[str]] = None,
    output: str = "hello world",
    cost: float = 0.05,
    latency: float = 500.0,
    forbidden_violations: Optional[List[str]] = None,
    model_id: Optional[str] = None,
) -> EvaluationResult:
    trace = _make_trace(tools=tools, output=output, cost=cost, latency=latency, model_id=model_id)

    ft_eval = None
    if forbidden_violations is not None:
        ft_eval = ForbiddenToolEvaluation(
            violations=forbidden_violations,
            passed=len(forbidden_violations) == 0,
        )

    return EvaluationResult(
        test_case=test_name,
        passed=score >= 70,
        score=score,
        evaluations=Evaluations(
            tool_accuracy=ToolEvaluation(accuracy=1.0),
            sequence_correctness=SequenceEvaluation(
                correct=True,
                expected_sequence=tools or ["search"],
                actual_sequence=tools or ["search"],
            ),
            output_quality=OutputEvaluation(
                score=score,
                rationale="ok",
                contains_checks=ContainsChecks(),
                not_contains_checks=ContainsChecks(),
            ),
            cost=CostEvaluation(total_cost=cost, threshold=1.0, passed=True),
            latency=LatencyEvaluation(total_latency=latency, threshold=5000, passed=True),
            forbidden_tools=ft_eval,
        ),
        trace=trace,
        timestamp=datetime.now(timezone.utc),
    )


def _make_test_case(name: str = "test-1", tools: Optional[List[str]] = None) -> TestCase:
    return TestCase(
        name=name,
        input=TestInput(query="hello"),
        expected=ExpectedBehavior(tools=tools or ["search"]),
        thresholds=Thresholds(min_score=70),
        adapter="http",
        endpoint="http://localhost:8000",
    )


def _make_diff(
    test_name: str = "test-1",
    severity: DiffStatus = DiffStatus.REGRESSION,
    tool_diffs: Optional[List[ToolDiff]] = None,
    score_diff: float = -10.0,
    model_changed: bool = False,
    golden_model_id: Optional[str] = None,
    actual_model_id: Optional[str] = None,
    output_similarity: float = 0.7,
) -> TraceDiff:
    return TraceDiff(
        test_name=test_name,
        has_differences=severity != DiffStatus.PASSED,
        tool_diffs=tool_diffs or [],
        output_diff=OutputDiff(
            similarity=output_similarity,
            golden_preview="golden",
            actual_preview="actual",
            diff_lines=[],
            severity=severity,
        ),
        score_diff=score_diff,
        latency_diff=0.0,
        overall_severity=severity,
        model_changed=model_changed,
        golden_model_id=golden_model_id,
        actual_model_id=actual_model_id,
    )


# ============================================================
# Diagnosis tests — pure, no mocks needed for adapter
# ============================================================

class TestDiagnosis:

    @pytest.fixture
    def engine(self):
        store = MagicMock(spec=GoldenStore)
        evaluator = MagicMock()
        return HealingEngine(store, evaluator)

    def test_passed_no_action(self, engine):
        diff = _make_diff(severity=DiffStatus.PASSED, score_diff=0.0)
        result = _make_result()
        tc = _make_test_case()
        golden = _make_golden()

        diag = engine.diagnose(diff, result, tc, golden)
        assert diag.action == HealingAction.NO_ACTION
        assert diag.reason == "passed"

    def test_forbidden_tool_blocked(self, engine):
        diff = _make_diff(severity=DiffStatus.REGRESSION)
        result = _make_result(forbidden_violations=["bash", "edit_file"])
        tc = _make_test_case()
        golden = _make_golden()

        diag = engine.diagnose(diff, result, tc, golden)
        assert diag.action == HealingAction.BLOCKED
        assert diag.trigger == HealingTrigger.FORBIDDEN_TOOL
        assert "bash" in diag.reason

    def test_tool_removed_flag_review(self, engine):
        diff = _make_diff(
            severity=DiffStatus.TOOLS_CHANGED,
            tool_diffs=[
                ToolDiff(
                    type="removed", position=0,
                    golden_tool="web_search", actual_tool=None,
                    severity=DiffStatus.TOOLS_CHANGED,
                    message="removed",
                )
            ],
        )
        result = _make_result()
        tc = _make_test_case()
        golden = _make_golden()

        diag = engine.diagnose(diff, result, tc, golden)
        assert diag.action == HealingAction.FLAG_REVIEW
        assert diag.trigger == HealingTrigger.STRUCTURAL_CHANGE

    def test_tool_added_flag_review(self, engine):
        diff = _make_diff(
            severity=DiffStatus.TOOLS_CHANGED,
            tool_diffs=[
                ToolDiff(
                    type="added", position=1,
                    golden_tool=None, actual_tool="new_tool",
                    severity=DiffStatus.TOOLS_CHANGED,
                    message="added",
                )
            ],
        )
        result = _make_result()
        tc = _make_test_case()
        golden = _make_golden()

        diag = engine.diagnose(diff, result, tc, golden)
        assert diag.action == HealingAction.FLAG_REVIEW
        assert diag.trigger == HealingTrigger.STRUCTURAL_CHANGE

    def test_tool_reordered_flag_review(self, engine):
        """Tool name swapped (changed type, different golden vs actual) -> FLAG_REVIEW."""
        diff = _make_diff(
            severity=DiffStatus.TOOLS_CHANGED,
            tool_diffs=[
                ToolDiff(
                    type="changed", position=0,
                    golden_tool="search", actual_tool="lookup",
                    severity=DiffStatus.TOOLS_CHANGED,
                    message="changed",
                )
            ],
        )
        result = _make_result()
        tc = _make_test_case()
        golden = _make_golden()

        diag = engine.diagnose(diff, result, tc, golden)
        assert diag.action == HealingAction.FLAG_REVIEW
        assert diag.trigger == HealingTrigger.STRUCTURAL_CHANGE

    def test_output_drift_retry(self, engine):
        """Output changed, no tool changes -> RETRY (non-determinism)."""
        diff = _make_diff(
            severity=DiffStatus.OUTPUT_CHANGED,
            tool_diffs=[],
            score_diff=-5.0,
        )
        result = _make_result(score=80.0)
        tc = _make_test_case()
        golden = _make_golden(score=85.0)

        diag = engine.diagnose(diff, result, tc, golden)
        assert diag.action == HealingAction.RETRY
        assert diag.trigger == HealingTrigger.NONDETERMINISM

    def test_score_only_retry(self, engine):
        """Score dropped, tools same -> RETRY."""
        diff = _make_diff(
            severity=DiffStatus.REGRESSION,
            tool_diffs=[],
            score_diff=-15.0,
        )
        result = _make_result(score=70.0)
        tc = _make_test_case()
        golden = _make_golden(score=85.0)

        diag = engine.diagnose(diff, result, tc, golden)
        assert diag.action == HealingAction.RETRY
        assert diag.trigger == HealingTrigger.NONDETERMINISM

    def test_score_improved_flag_review(self, engine):
        """Score went UP -> FLAG_REVIEW (not auto-accept!)."""
        diff = _make_diff(
            severity=DiffStatus.OUTPUT_CHANGED,
            tool_diffs=[],
            score_diff=5.0,
        )
        result = _make_result(score=90.0)
        tc = _make_test_case()
        golden = _make_golden(score=85.0)

        diag = engine.diagnose(diff, result, tc, golden)
        assert diag.action == HealingAction.FLAG_REVIEW
        assert diag.trigger == HealingTrigger.SCORE_IMPROVEMENT
        assert "snapshot" in diag.reason

    def test_cost_spike_flag_review(self, engine):
        """Cost >2x baseline -> FLAG_REVIEW."""
        diff = _make_diff(
            severity=DiffStatus.REGRESSION,
            tool_diffs=[],
            score_diff=-5.0,
        )
        result = _make_result(cost=0.20)
        tc = _make_test_case()
        golden = _make_golden(cost=0.05)

        diag = engine.diagnose(diff, result, tc, golden)
        assert diag.action == HealingAction.FLAG_REVIEW
        assert diag.trigger == HealingTrigger.COST_SPIKE

    def test_latency_spike_flag_review(self, engine):
        """Latency >3x baseline -> FLAG_REVIEW."""
        diff = _make_diff(
            severity=DiffStatus.REGRESSION,
            tool_diffs=[],
            score_diff=-5.0,
        )
        result = _make_result(latency=2000.0)
        tc = _make_test_case()
        golden = _make_golden(latency=500.0)

        diag = engine.diagnose(diff, result, tc, golden)
        assert diag.action == HealingAction.FLAG_REVIEW
        assert diag.trigger == HealingTrigger.LATENCY_SPIKE

    def test_param_change_same_tool_flag_review(self, engine):
        """Same tool, params changed -> FLAG_REVIEW (conservative v1)."""
        from evalview.core.diff import ParameterDiff

        diff = _make_diff(
            severity=DiffStatus.TOOLS_CHANGED,
            tool_diffs=[
                ToolDiff(
                    type="changed", position=0,
                    golden_tool="search", actual_tool="search",
                    severity=DiffStatus.TOOLS_CHANGED,
                    message="params changed",
                    parameter_diffs=[
                        ParameterDiff(
                            param_name="query",
                            golden_value="old",
                            actual_value="new",
                            diff_type="value_changed",
                        )
                    ],
                )
            ],
            score_diff=-3.0,
        )
        result = _make_result(score=82.0)
        tc = _make_test_case()
        golden = _make_golden(score=85.0)

        diag = engine.diagnose(diff, result, tc, golden)
        assert diag.action == HealingAction.FLAG_REVIEW
        assert diag.trigger == HealingTrigger.PARAM_CHANGE

    def test_no_tool_diffs_retry(self, engine):
        """Output changed, zero tool diffs -> RETRY (pure non-determinism)."""
        diff = _make_diff(
            severity=DiffStatus.OUTPUT_CHANGED,
            tool_diffs=[],
            score_diff=-2.0,
        )
        result = _make_result(score=83.0)
        tc = _make_test_case()
        golden = _make_golden(score=85.0)

        diag = engine.diagnose(diff, result, tc, golden)
        assert diag.action == HealingAction.RETRY
        assert diag.trigger == HealingTrigger.NONDETERMINISM

    def test_model_changed_retry(self, engine):
        """model_changed=True -> RETRY with trigger=MODEL_UPDATE."""
        diff = _make_diff(
            severity=DiffStatus.OUTPUT_CHANGED,
            tool_diffs=[],
            score_diff=-8.0,
            model_changed=True,
            golden_model_id="gpt-4o-2024-08-06",
            actual_model_id="gpt-4o-2024-11-20",
        )
        result = _make_result(score=77.0)
        tc = _make_test_case()
        golden = _make_golden(score=85.0, model_id="gpt-4o-2024-08-06")

        diag = engine.diagnose(diff, result, tc, golden)
        assert diag.action == HealingAction.RETRY
        assert diag.trigger == HealingTrigger.MODEL_UPDATE
        assert "gpt-4o-2024-11-20" in diag.reason

    def test_model_changed_with_tool_removal_flag_review(self, engine):
        """Model changed BUT tool removed -> FLAG_REVIEW (structural wins)."""
        diff = _make_diff(
            severity=DiffStatus.TOOLS_CHANGED,
            tool_diffs=[
                ToolDiff(
                    type="removed", position=0,
                    golden_tool="web_search", actual_tool=None,
                    severity=DiffStatus.TOOLS_CHANGED,
                    message="removed",
                )
            ],
            score_diff=-5.0,
            model_changed=True,
            golden_model_id="gpt-4o-2024-08-06",
            actual_model_id="gpt-4o-2024-11-20",
        )
        result = _make_result(score=80.0)
        tc = _make_test_case()
        golden = _make_golden(score=85.0, model_id="gpt-4o-2024-08-06")

        diag = engine.diagnose(diff, result, tc, golden)
        assert diag.action == HealingAction.FLAG_REVIEW
        assert diag.trigger == HealingTrigger.STRUCTURAL_CHANGE


# ============================================================
# Healing flow tests — mock adapter + golden store
# ============================================================

class TestHealingFlow:

    @pytest.fixture
    def tmp_dir(self):
        d = tempfile.mkdtemp()
        yield Path(d)
        shutil.rmtree(d)

    @pytest.mark.asyncio
    async def test_retry_succeeds(self, tmp_dir):
        """Retry returns PASSED diff -> healed=True, proposed=False."""
        store = GoldenStore(base_path=tmp_dir)
        evaluator = AsyncMock()

        # Retry result passes
        retry_trace = _make_trace(output="hello world")
        retry_result = _make_result(score=85.0)
        evaluator.evaluate = AsyncMock(return_value=retry_result)

        adapter = AsyncMock()
        adapter.execute = AsyncMock(return_value=retry_trace)

        diff_engine = MagicMock(spec=DiffEngine)
        passed_diff = _make_diff(severity=DiffStatus.PASSED, score_diff=0.0)
        diff_engine.compare_multi_reference_async = AsyncMock(return_value=passed_diff)

        engine = HealingEngine(store, evaluator)

        original_diff = _make_diff(severity=DiffStatus.REGRESSION, tool_diffs=[], score_diff=-10.0)
        original_result = _make_result(score=75.0)
        tc = _make_test_case()
        golden = _make_golden()

        hr = await engine.heal_test(
            original_diff, original_result, tc, [golden], adapter, diff_engine
        )

        assert hr.healed is True
        assert hr.proposed is False
        assert hr.final_status == DiffStatus.PASSED.value
        assert hr.diagnosis.action == HealingAction.RETRY
        assert hr.retry_score == 85.0

    @pytest.mark.asyncio
    async def test_retry_fails_proposes_variant(self, tmp_dir):
        """Retry still fails, score >= 70, no tool changes -> proposed=True."""
        store = GoldenStore(base_path=tmp_dir)
        # Need to create the golden dir and a baseline for count_variants
        store.golden_dir.mkdir(parents=True, exist_ok=True)

        evaluator = AsyncMock()
        retry_trace = _make_trace(output="different output")
        retry_result = _make_result(score=75.0, output="different output")
        evaluator.evaluate = AsyncMock(return_value=retry_result)

        adapter = AsyncMock()
        adapter.execute = AsyncMock(return_value=retry_trace)

        diff_engine = MagicMock(spec=DiffEngine)
        still_failed_diff = _make_diff(
            severity=DiffStatus.OUTPUT_CHANGED, tool_diffs=[], score_diff=-10.0
        )
        diff_engine.compare_multi_reference_async = AsyncMock(return_value=still_failed_diff)

        engine = HealingEngine(store, evaluator)

        original_diff = _make_diff(severity=DiffStatus.REGRESSION, tool_diffs=[], score_diff=-10.0)
        original_result = _make_result(score=75.0)
        tc = _make_test_case()
        golden = _make_golden()

        hr = await engine.heal_test(
            original_diff, original_result, tc, [golden], adapter, diff_engine
        )

        assert hr.healed is False
        assert hr.proposed is True
        assert hr.variant_saved is not None
        assert hr.variant_saved.startswith("auto_heal_")
        assert hr.variant_path is not None
        assert hr.retry_status == DiffStatus.OUTPUT_CHANGED.value
        assert hr.diagnosis.action == HealingAction.PROPOSE_VARIANT

    @pytest.mark.asyncio
    async def test_retry_fails_score_too_low(self, tmp_dir):
        """Retry fails, score < 70 -> healed=False, proposed=False, FLAG_REVIEW."""
        store = GoldenStore(base_path=tmp_dir)
        store.golden_dir.mkdir(parents=True, exist_ok=True)

        evaluator = AsyncMock()
        retry_trace = _make_trace(output="bad output")
        retry_result = _make_result(score=50.0, output="bad output")
        evaluator.evaluate = AsyncMock(return_value=retry_result)

        adapter = AsyncMock()
        adapter.execute = AsyncMock(return_value=retry_trace)

        diff_engine = MagicMock(spec=DiffEngine)
        still_failed_diff = _make_diff(
            severity=DiffStatus.REGRESSION, tool_diffs=[], score_diff=-35.0
        )
        diff_engine.compare_multi_reference_async = AsyncMock(return_value=still_failed_diff)

        engine = HealingEngine(store, evaluator)

        original_diff = _make_diff(severity=DiffStatus.REGRESSION, tool_diffs=[], score_diff=-35.0)
        original_result = _make_result(score=50.0)
        tc = _make_test_case()
        golden = _make_golden()

        hr = await engine.heal_test(
            original_diff, original_result, tc, [golden], adapter, diff_engine
        )

        assert hr.healed is False
        assert hr.proposed is False
        assert hr.diagnosis.action == HealingAction.FLAG_REVIEW
        assert hr.retry_score == 50.0
        assert hr.retry_status == DiffStatus.REGRESSION.value
        assert "variant_blocked_by" in hr.diagnosis.details

    @pytest.mark.asyncio
    async def test_variant_limit_respected(self, tmp_dir):
        """3+ auto variants exist -> falls back to FLAG_REVIEW."""
        store = GoldenStore(base_path=tmp_dir)
        store.golden_dir.mkdir(parents=True, exist_ok=True)

        # Create MAX_AUTO_VARIANTS existing variants so the store is full
        for i in range(MAX_AUTO_VARIANTS):
            dummy_result = _make_result(test_name="test-1")
            store.save_golden(dummy_result, variant_name=f"v{i}")

        evaluator = AsyncMock()
        retry_trace = _make_trace(output="different output")
        retry_result = _make_result(score=75.0, output="different output")
        evaluator.evaluate = AsyncMock(return_value=retry_result)

        adapter = AsyncMock()
        adapter.execute = AsyncMock(return_value=retry_trace)

        diff_engine = MagicMock(spec=DiffEngine)
        still_failed = _make_diff(severity=DiffStatus.OUTPUT_CHANGED, tool_diffs=[], score_diff=-10.0)
        diff_engine.compare_multi_reference_async = AsyncMock(return_value=still_failed)

        engine = HealingEngine(store, evaluator)

        original_diff = _make_diff(severity=DiffStatus.REGRESSION, tool_diffs=[], score_diff=-10.0)
        original_result = _make_result(score=75.0)
        tc = _make_test_case()
        golden = _make_golden()

        hr = await engine.heal_test(
            original_diff, original_result, tc, [golden], adapter, diff_engine
        )

        assert hr.healed is False
        assert hr.proposed is False
        assert hr.diagnosis.action == HealingAction.FLAG_REVIEW

    def test_audit_log_saved(self, tmp_dir):
        """Verify JSON file written to .evalview/healing/."""
        import os
        original_cwd = os.getcwd()
        try:
            os.chdir(tmp_dir)
            summary = HealingSummary(
                results=[
                    HealingResult(
                        test_name="test-1",
                        original_status="regression",
                        diagnosis=HealingDiagnosis(
                            action=HealingAction.RETRY,
                            trigger=HealingTrigger.NONDETERMINISM,
                            reason="retried",
                        ),
                        healed=True,
                        final_status="passed",
                    )
                ],
                total_healed=1,
                total_proposed=0,
                total_review=0,
                total_blocked=0,
            )

            path = save_audit_log(summary)
            assert Path(path).exists()
            data = json.loads(Path(path).read_text())
            assert data["total_healed"] == 1
            assert len(data["results"]) == 1
            assert data["policy_version"] == "v1"
        finally:
            os.chdir(original_cwd)

    def test_audit_log_not_saved_when_all_passed(self, tmp_dir):
        """No audit log when nothing to heal (empty results)."""
        import os
        original_cwd = os.getcwd()
        try:
            os.chdir(tmp_dir)
            # The save_audit_log function is only called when healing_results is non-empty.
            # We verify the contract: if called with empty results, file is still created
            # but the real guard is in check_cmd.py (only calls when healing_results is non-empty).
            summary = HealingSummary(
                results=[],
                total_healed=0,
                total_proposed=0,
                total_review=0,
                total_blocked=0,
            )
            path = save_audit_log(summary)
            assert Path(path).exists()
            data = json.loads(Path(path).read_text())
            assert data["total_healed"] == 0
            assert len(data["results"]) == 0
        finally:
            os.chdir(original_cwd)


# ============================================================
# Model update tests
# ============================================================

class TestModelUpdate:

    def test_model_update_summary_populated(self):
        """When model_changed on 2+ tests, ModelUpdateSummary has correct counts."""
        mu = ModelUpdateSummary(
            golden_model="gpt-4o-2024-08-06",
            actual_model="gpt-4o-2024-11-20",
            affected_count=3,
            healed_count=2,
            failed_count=1,
        )
        assert mu.affected_count == 3
        assert mu.healed_count == 2
        assert mu.failed_count == 1

    def test_model_update_all_healed_message(self):
        """All model-affected tests retry-healed -> summary reflects this."""
        mu = ModelUpdateSummary(
            golden_model="gpt-4o-2024-08-06",
            actual_model="gpt-4o-2024-11-20",
            affected_count=3,
            healed_count=3,
            failed_count=0,
        )
        assert mu.healed_count == mu.affected_count
        assert mu.failed_count == 0

    def test_model_update_partial_message(self):
        """Some healed, some not."""
        mu = ModelUpdateSummary(
            golden_model="gpt-4o-2024-08-06",
            actual_model="gpt-4o-2024-11-20",
            affected_count=3,
            healed_count=1,
            failed_count=2,
        )
        assert 0 < mu.healed_count < mu.affected_count

    def test_model_update_all_failed_message(self):
        """None healed -> broke all tests."""
        mu = ModelUpdateSummary(
            golden_model="gpt-4o-2024-08-06",
            actual_model="gpt-4o-2024-11-20",
            affected_count=3,
            healed_count=0,
            failed_count=3,
        )
        assert mu.healed_count == 0
        assert mu.failed_count == mu.affected_count


# ============================================================
# Exit code tests
# ============================================================

class TestExitCode:

    def test_all_failures_retry_healed_requires_full_coverage(self):
        """Missing healing result for one failure must keep exit nonzero."""
        diffs = [
            ("t1", _make_diff(test_name="t1", severity=DiffStatus.REGRESSION)),
            ("t2", _make_diff(test_name="t2", severity=DiffStatus.OUTPUT_CHANGED)),
        ]
        summary = HealingSummary(
            results=[
                HealingResult(
                    test_name="t1",
                    original_status="regression",
                    diagnosis=HealingDiagnosis(
                        action=HealingAction.RETRY,
                        trigger=HealingTrigger.NONDETERMINISM,
                        reason="retried",
                    ),
                    attempted=True,
                    healed=True,
                    final_status="passed",
                ),
            ],
            total_healed=1,
            total_proposed=0,
            total_review=0,
            total_blocked=0,
        )

        assert _all_failures_retry_healed(diffs, summary) is False

    def test_exit_0_when_all_healed(self):
        """All failures retry-healed -> exit 0."""
        diffs = [
            ("t1", _make_diff(test_name="t1", severity=DiffStatus.REGRESSION)),
            ("t2", _make_diff(test_name="t2", severity=DiffStatus.OUTPUT_CHANGED)),
        ]
        summary = HealingSummary(
            results=[
                HealingResult(
                    test_name="t1",
                    original_status="regression",
                    diagnosis=HealingDiagnosis(
                        action=HealingAction.RETRY,
                        trigger=HealingTrigger.NONDETERMINISM,
                        reason="retried",
                    ),
                    attempted=True,
                    healed=True,
                    final_status="passed",
                ),
                HealingResult(
                    test_name="t2",
                    original_status="output_changed",
                    diagnosis=HealingDiagnosis(
                        action=HealingAction.RETRY,
                        trigger=HealingTrigger.MODEL_UPDATE,
                        reason="retried",
                    ),
                    attempted=True,
                    healed=True,
                    final_status="passed",
                ),
            ],
            total_healed=2,
            total_proposed=0,
            total_review=0,
            total_blocked=0,
        )

        assert _all_failures_retry_healed(diffs, summary) is True

    def test_exit_1_when_proposed_remains(self):
        """One PROPOSED -> exit 1."""
        diffs = [("t1", _make_diff(test_name="t1", severity=DiffStatus.REGRESSION))]
        summary = HealingSummary(
            results=[
                HealingResult(
                    test_name="t1",
                    original_status="regression",
                    diagnosis=HealingDiagnosis(
                        action=HealingAction.PROPOSE_VARIANT,
                        trigger=HealingTrigger.NONDETERMINISM,
                        reason="saved variant",
                    ),
                    attempted=True,
                    healed=False,
                    proposed=True,
                    final_status="output_changed",
                ),
            ],
            total_healed=0,
            total_proposed=1,
            total_review=0,
            total_blocked=0,
        )

        assert _all_failures_retry_healed(diffs, summary) is False

    def test_exit_1_when_review_remains(self):
        """One FLAG_REVIEW -> exit 1."""
        diffs = [("t1", _make_diff(test_name="t1", severity=DiffStatus.TOOLS_CHANGED))]
        summary = HealingSummary(
            results=[
                HealingResult(
                    test_name="t1",
                    original_status="tools_changed",
                    diagnosis=HealingDiagnosis(
                        action=HealingAction.FLAG_REVIEW,
                        trigger=HealingTrigger.STRUCTURAL_CHANGE,
                        reason="tool removed",
                    ),
                    attempted=False,
                    healed=False,
                    final_status="tools_changed",
                ),
            ],
            total_healed=0,
            total_proposed=0,
            total_review=1,
            total_blocked=0,
        )

        assert _all_failures_retry_healed(diffs, summary) is False

    def test_exit_1_when_blocked(self):
        """BLOCKED -> exit 1."""
        diffs = [("t1", _make_diff(test_name="t1", severity=DiffStatus.REGRESSION))]
        summary = HealingSummary(
            results=[
                HealingResult(
                    test_name="t1",
                    original_status="regression",
                    diagnosis=HealingDiagnosis(
                        action=HealingAction.BLOCKED,
                        trigger=HealingTrigger.FORBIDDEN_TOOL,
                        reason="forbidden tool called: bash",
                    ),
                    attempted=False,
                    healed=False,
                    final_status="regression",
                ),
            ],
            total_healed=0,
            total_proposed=0,
            total_review=0,
            total_blocked=1,
        )

        assert _all_failures_retry_healed(diffs, summary) is False
