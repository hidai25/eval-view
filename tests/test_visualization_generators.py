from __future__ import annotations

from datetime import datetime

from evalview.core.diff import DiffStatus, OutputDiff, TraceDiff
from evalview.core.golden import GoldenMetadata, GoldenTrace
from evalview.core.healing import HealingAction, HealingDiagnosis, HealingResult, HealingSummary, HealingTrigger
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
    StepMetrics,
    StepTrace,
    TurnTrace,
    TokenUsage,
    ToolEvaluation,
)
from evalview.visualization.generators import generate_visual_report


def test_visual_report_shows_model_and_baseline_metadata(tmp_path):
    now = datetime(2026, 3, 15, 16, 50)
    trace = ExecutionTrace(
        session_id="s1",
        start_time=now,
        end_time=now,
        steps=[
            StepTrace(
                step_id="1",
                step_name="lookup_order",
                tool_name="lookup_order",
                parameters={"order_id": "4812"},
                output="ok",
                success=True,
                metrics=StepMetrics(latency=34.0, cost=0.0),
            )
        ],
        final_output="Refund issued.",
        metrics=ExecutionMetrics(
            total_cost=0.0,
            total_latency=34.0,
            total_tokens=TokenUsage(input_tokens=120, output_tokens=40, cached_tokens=0),
        ),
        model_id="gpt-4o-mini",
        model_provider="openai",
    )
    result = EvaluationResult(
        test_case="refund-flow",
        passed=True,
        score=85.0,
        evaluations=Evaluations(
            tool_accuracy=ToolEvaluation(accuracy=1.0, correct=["lookup_order"]),
            sequence_correctness=SequenceEvaluation(correct=True, expected_sequence=[], actual_sequence=[]),
            output_quality=OutputEvaluation(
                score=85.0,
                rationale="ok",
                contains_checks=ContainsChecks(),
                not_contains_checks=ContainsChecks(),
            ),
            cost=CostEvaluation(total_cost=0.0, threshold=1.0, passed=True),
            latency=LatencyEvaluation(total_latency=34.0, threshold=1000.0, passed=True),
        ),
        trace=trace,
        timestamp=now,
        input_query="I need a refund for order 4812.",
        actual_output="Refund issued.",
    )
    golden = GoldenTrace(
        metadata=GoldenMetadata(
            test_name="refund-flow",
            blessed_at=datetime(2026, 3, 14, 12, 15),
            score=85.0,
            model_id="gpt-4o-mini",
            model_provider="openai",
        ),
        trace=trace,
        tool_sequence=["lookup_order"],
        output_hash="abc123",
    )

    report_path = tmp_path / "report.html"
    generate_visual_report(
        results=[result],
        diffs=[],
        output_path=str(report_path),
        auto_open=False,
        golden_traces={"refund-flow": golden},
        judge_usage={
            "call_count": 2,
            "input_tokens": 220,
            "output_tokens": 44,
            "total_tokens": 264,
            "total_cost": 0.0012,
            "is_free": False,
        },
        title="EvalView Check Report",
    )

    html = report_path.read_text(encoding="utf-8")
    assert "openai/gpt-4o-mini" in html
    assert "Baseline Snapshot" in html
    assert "2026-03-14 12:15" in html
    assert "160 tokens" in html  # total tokens shown in KPI strip
    assert "Execution Cost per Query" in html
    assert "Trace Cost" in html
    assert "EvalView Judge" in html
    assert "$0.0012" in html
    assert "264 tokens across 2 judge calls" in html


def test_visual_report_falls_back_for_missing_step_latency_and_baseline_model(tmp_path):
    now = datetime(2026, 3, 15, 16, 50)
    trace = ExecutionTrace(
        session_id="s1",
        start_time=now,
        end_time=now,
        steps=[
            StepTrace(
                step_id="1",
                step_name="lookup_account",
                tool_name="lookup_account",
                parameters={},
                output="ok",
                success=True,
                metrics=StepMetrics(latency=0.0, cost=0.0),
            ),
            StepTrace(
                step_id="2",
                step_name="check_service_status",
                tool_name="check_service_status",
                parameters={},
                output="ok",
                success=True,
                metrics=StepMetrics(latency=0.0, cost=0.0),
            ),
        ],
        final_output="ok",
        metrics=ExecutionMetrics(
            total_cost=0.004,
            total_latency=400.0,
            total_tokens=TokenUsage(input_tokens=50, output_tokens=10, cached_tokens=0),
        ),
        model_id="mock-support-agent",
        model_provider=None,
    )
    result = EvaluationResult(
        test_case="vip-outage",
        passed=True,
        score=85.0,
        evaluations=Evaluations(
            tool_accuracy=ToolEvaluation(accuracy=1.0, correct=["lookup_account", "check_service_status"]),
            sequence_correctness=SequenceEvaluation(correct=True, expected_sequence=[], actual_sequence=[]),
            output_quality=OutputEvaluation(
                score=85.0,
                rationale="ok",
                contains_checks=ContainsChecks(),
                not_contains_checks=ContainsChecks(),
            ),
            cost=CostEvaluation(total_cost=0.004, threshold=1.0, passed=True),
            latency=LatencyEvaluation(total_latency=400.0, threshold=1000.0, passed=True),
        ),
        trace=trace,
        timestamp=now,
        input_query="dashboard down",
        actual_output="ok",
    )
    golden = GoldenTrace(
        metadata=GoldenMetadata(
            test_name="vip-outage",
            blessed_at=datetime(2026, 3, 14, 13, 14),
            score=85.0,
            model_id=None,
            model_provider=None,
        ),
        trace=trace.model_copy(update={"model_id": None, "model_provider": None}),
        tool_sequence=["lookup_account", "check_service_status"],
        output_hash="def456",
    )

    report_path = tmp_path / "report.html"
    generate_visual_report(
        results=[result],
        diffs=[],
        output_path=str(report_path),
        auto_open=False,
        golden_traces={"vip-outage": golden},
        title="EvalView Check Report",
    )

    html = report_path.read_text(encoding="utf-8")
    assert "Not recorded in snapshot" in html
    assert "⚡ 400.0ms" in html
    assert "💰 $0.004000" in html
    assert '"latency": 200.0' in html
    assert '"cost": 0.002' in html
    assert "mock-support-agent" in html
    assert "Trace cost comes from the agent execution trace only" in html
    assert "Baseline model: Not recorded in snapshot" in html


def test_visual_report_shows_all_multi_turn_turns_without_tool_steps(tmp_path):
    now = datetime(2026, 3, 16, 10, 24)
    trace = ExecutionTrace(
        session_id="s2",
        start_time=now,
        end_time=now,
        steps=[],
        final_output="I can help with billing, refunds, and outages.",
        metrics=ExecutionMetrics(
            total_cost=0.0,
            total_latency=102.0,
            total_tokens=TokenUsage(input_tokens=30, output_tokens=20, cached_tokens=0),
        ),
        model_id="mock-support-agent",
        turns=[
            TurnTrace(
                index=1,
                query="Hello, what can you help me with?",
                output="Please describe the support issue you need help with.",
                tools=[],
                latency_ms=48.0,
                cost=0.0,
            ),
            TurnTrace(
                index=2,
                query="I need help with a refund.",
                output="I can help with billing, refunds, and outages.",
                tools=[],
                latency_ms=54.0,
                cost=0.0,
            ),
        ],
    )
    result = EvaluationResult(
        test_case="hello-multi-turn",
        passed=True,
        score=85.0,
        evaluations=Evaluations(
            tool_accuracy=ToolEvaluation(accuracy=1.0, correct=[]),
            sequence_correctness=SequenceEvaluation(correct=True, expected_sequence=[], actual_sequence=[]),
            output_quality=OutputEvaluation(
                score=85.0,
                rationale="ok",
                contains_checks=ContainsChecks(),
                not_contains_checks=ContainsChecks(),
            ),
            cost=CostEvaluation(total_cost=0.0, threshold=1.0, passed=True),
            latency=LatencyEvaluation(total_latency=102.0, threshold=1000.0, passed=True),
        ),
        trace=trace,
        timestamp=now,
        input_query="Hello, what can you help me with?",
        actual_output="I can help with billing, refunds, and outages.",
    )

    report_path = tmp_path / "report.html"
    generate_visual_report(
        results=[result],
        diffs=[],
        output_path=str(report_path),
        auto_open=False,
        title="EvalView Run Report",
    )

    html = report_path.read_text(encoding="utf-8")
    assert "Conversation Turns" in html
    assert "Turn 1" in html
    assert "Turn 2" in html
    assert "I need help with a refund." in html
    assert "Please describe the support issue you need help with." in html
    assert "I can help with billing, refunds, and outages." in html


def test_visual_report_includes_healing_summary_and_audit_context(tmp_path):
    now = datetime(2026, 3, 16, 10, 24)
    trace = ExecutionTrace(
        session_id="s3",
        start_time=now,
        end_time=now,
        steps=[],
        final_output="ok",
        metrics=ExecutionMetrics(total_cost=0.0, total_latency=10.0),
        model_id="gpt-4o-mini",
    )
    result = EvaluationResult(
        test_case="refund-flow",
        passed=False,
        score=72.0,
        evaluations=Evaluations(
            tool_accuracy=ToolEvaluation(accuracy=1.0, correct=[]),
            sequence_correctness=SequenceEvaluation(correct=True, expected_sequence=[], actual_sequence=[]),
            output_quality=OutputEvaluation(
                score=72.0,
                rationale="changed",
                contains_checks=ContainsChecks(),
                not_contains_checks=ContainsChecks(),
            ),
            cost=CostEvaluation(total_cost=0.0, threshold=1.0, passed=True),
            latency=LatencyEvaluation(total_latency=10.0, threshold=1000.0, passed=True),
        ),
        trace=trace,
        timestamp=now,
        input_query="refund",
        actual_output="ok",
    )
    healing = HealingSummary(
        results=[
            HealingResult(
                test_name="refund-flow",
                original_status="output_changed",
                diagnosis=HealingDiagnosis(
                    action=HealingAction.PROPOSE_VARIANT,
                    trigger=HealingTrigger.NONDETERMINISM,
                    reason="saved candidate variant auto_heal_abcd (score 72.0)",
                ),
                attempted=True,
                healed=False,
                proposed=True,
                final_status="output_changed",
                retry_score=72.0,
                retry_status="output_changed",
                variant_saved="auto_heal_abcd",
            )
        ],
        total_healed=0,
        total_proposed=1,
        total_review=0,
        total_blocked=0,
        attempted_count=1,
        unresolved_count=1,
        failed_count=1,
        audit_path=".evalview/healing/sample.json",
        thresholds={
            "min_variant_score": 70.0,
            "max_auto_variants": 3.0,
        },
    )

    report_path = tmp_path / "report.html"
    generate_visual_report(
        results=[result],
        diffs=[],
        output_path=str(report_path),
        auto_open=False,
        title="EvalView Check Report",
        healing_summary=healing,
        effective_all_passed=False,
    )

    html = report_path.read_text(encoding="utf-8")
    assert "Healing Summary" in html
    assert "saved candidate variant auto_heal_abcd" in html
    assert ".evalview/healing/sample.json" in html
    assert "Final Outcome Failing" in html


def test_visual_report_shows_behavior_summary_tags_and_root_cause(tmp_path):
    now = datetime(2026, 3, 16, 10, 24)
    trace = ExecutionTrace(
        session_id="s4",
        start_time=now,
        end_time=now,
        steps=[],
        final_output="Changed answer",
        metrics=ExecutionMetrics(total_cost=0.0, total_latency=10.0),
        model_id="gpt-4o-mini",
    )
    result = EvaluationResult(
        test_case="refund-flow",
        passed=False,
        score=72.0,
        evaluations=Evaluations(
            tool_accuracy=ToolEvaluation(accuracy=1.0, correct=[]),
            sequence_correctness=SequenceEvaluation(correct=True, expected_sequence=[], actual_sequence=[]),
            output_quality=OutputEvaluation(
                score=72.0,
                rationale="changed",
                contains_checks=ContainsChecks(),
                not_contains_checks=ContainsChecks(),
            ),
            cost=CostEvaluation(total_cost=0.0, threshold=1.0, passed=True),
            latency=LatencyEvaluation(total_latency=10.0, threshold=1000.0, passed=True),
        ),
        trace=trace,
        timestamp=now,
        input_query="refund",
        actual_output="Changed answer",
    )
    diff = TraceDiff(
        test_name="refund-flow",
        has_differences=True,
        tool_diffs=[],
        output_diff=OutputDiff(
            similarity=0.42,
            golden_preview="Refunds take 5 days.",
            actual_preview="Refunds take 30 days.",
            diff_lines=["-Refunds take 5 days.", "+Refunds take 30 days."],
            severity=DiffStatus.OUTPUT_CHANGED,
            semantic_similarity=0.61,
        ),
        score_diff=-18.0,
        latency_diff=0.0,
        overall_severity=DiffStatus.OUTPUT_CHANGED,
    )

    report_path = tmp_path / "report.html"
    generate_visual_report(
        results=[result],
        diffs=[diff],
        output_path=str(report_path),
        auto_open=False,
        title="EvalView Check Report",
        test_metadata={"refund-flow": {"tags": ["tool_use", "clarification"]}},
        active_tags=["tool_use"],
    )

    html = report_path.read_text(encoding="utf-8")
    assert "Behavior Summary" in html
    assert "Filtered by tags" in html
    assert "tool_use" in html
    assert "clarification" in html
    assert "Why This Changed" in html
    assert "Same tools and parameters but output changed" in html
