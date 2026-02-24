"""Tests for the forbidden_tools safety contract.

Covers:
- ForbiddenToolEvaluation model
- ToolCallEvaluator.evaluate_forbidden()
- Evaluator hard-fail on forbidden tool violations
- HTML reporter span serialisation (with and without TraceContext)
"""
import pytest
from datetime import datetime
from unittest.mock import patch

from evalview.core.types import (
    TestCase as TestCaseModel,
    TestInput as TestInputModel,
    ExpectedBehavior,
    Thresholds,
    ExecutionTrace,
    StepTrace,
    StepMetrics,
    ExecutionMetrics,
    ForbiddenToolEvaluation,
    TraceContext,
    Span,
    SpanKind,
    LLMCallInfo,
    ToolCallInfo,
)
from evalview.evaluators.tool_call_evaluator import ToolCallEvaluator
from evalview.evaluators.evaluator import Evaluator


# ── Helpers ──────────────────────────────────────────────────────────────────


def _make_trace(*tool_names: str) -> ExecutionTrace:
    """Build a minimal ExecutionTrace with one step per tool name."""
    steps = [
        StepTrace(
            step_id=str(i),
            step_name=name,
            tool_name=name,
            parameters={},
            output="ok",
            success=True,
            metrics=StepMetrics(latency=100.0, cost=0.01),
        )
        for i, name in enumerate(tool_names, start=1)
    ]
    return ExecutionTrace(
        session_id="test-session",
        start_time=datetime.now(),
        end_time=datetime.now(),
        steps=steps,
        final_output="done",
        metrics=ExecutionMetrics(
            total_cost=0.01 * len(steps),
            total_latency=100.0 * len(steps),
        ),
    )


def _make_test_case(forbidden=None, tools=None) -> TestCaseModel:
    return TestCaseModel(
        name="test",
        input=TestInputModel(query="do the thing"),
        expected=ExpectedBehavior(tools=tools or [], forbidden_tools=forbidden),
        thresholds=Thresholds(min_score=50.0),
    )


# ── ForbiddenToolEvaluation model ────────────────────────────────────────────


class TestForbiddenToolEvaluationModel:
    def test_passed_when_no_violations(self):
        ev = ForbiddenToolEvaluation(violations=[], passed=True)
        assert ev.passed is True
        assert ev.violations == []

    def test_failed_when_violations_present(self):
        ev = ForbiddenToolEvaluation(violations=["edit_file"], passed=False)
        assert ev.passed is False
        assert "edit_file" in ev.violations

    def test_default_violations_empty(self):
        ev = ForbiddenToolEvaluation(passed=True)
        assert ev.violations == []


# ── ToolCallEvaluator.evaluate_forbidden ─────────────────────────────────────


class TestEvaluateForbidden:
    def test_returns_none_when_no_forbidden_configured(self):
        evaluator = ToolCallEvaluator()
        test_case = _make_test_case(forbidden=None)
        trace = _make_trace("bash", "search")
        assert evaluator.evaluate_forbidden(test_case, trace) is None

    def test_returns_none_when_empty_forbidden_list(self):
        evaluator = ToolCallEvaluator()
        test_case = _make_test_case(forbidden=[])
        trace = _make_trace("bash", "search")
        assert evaluator.evaluate_forbidden(test_case, trace) is None

    def test_passes_when_no_forbidden_tool_called(self):
        evaluator = ToolCallEvaluator()
        test_case = _make_test_case(forbidden=["edit_file", "bash"])
        trace = _make_trace("search", "summarize")
        result = evaluator.evaluate_forbidden(test_case, trace)
        assert result is not None
        assert result.passed is True
        assert result.violations == []

    def test_fails_when_single_forbidden_tool_called(self):
        evaluator = ToolCallEvaluator()
        test_case = _make_test_case(forbidden=["edit_file"])
        trace = _make_trace("search", "edit_file", "summarize")
        result = evaluator.evaluate_forbidden(test_case, trace)
        assert result is not None
        assert result.passed is False
        assert "edit_file" in result.violations

    def test_fails_when_multiple_forbidden_tools_called(self):
        evaluator = ToolCallEvaluator()
        test_case = _make_test_case(forbidden=["edit_file", "bash"])
        trace = _make_trace("bash", "edit_file", "search")
        result = evaluator.evaluate_forbidden(test_case, trace)
        assert result is not None
        assert result.passed is False
        assert set(result.violations) == {"bash", "edit_file"}

    def test_violations_are_deduplicated(self):
        """Calling a forbidden tool twice should appear only once in violations."""
        evaluator = ToolCallEvaluator()
        test_case = _make_test_case(forbidden=["bash"])
        trace = _make_trace("bash", "bash", "bash")
        result = evaluator.evaluate_forbidden(test_case, trace)
        assert result is not None
        assert result.passed is False
        assert result.violations.count("bash") == 1

    def test_case_insensitive_matching(self):
        """Forbidden tool names should match regardless of casing."""
        evaluator = ToolCallEvaluator()
        test_case = _make_test_case(forbidden=["EditFile"])
        # Agent calls "edit_file" (lowercase + underscore)
        trace = _make_trace("edit_file")
        result = evaluator.evaluate_forbidden(test_case, trace)
        assert result is not None
        assert result.passed is False

    def test_passes_when_trace_has_no_steps(self):
        evaluator = ToolCallEvaluator()
        test_case = _make_test_case(forbidden=["bash", "edit_file"])
        trace = _make_trace()  # zero steps
        result = evaluator.evaluate_forbidden(test_case, trace)
        assert result is not None
        assert result.passed is True
        assert result.violations == []

    def test_evaluations_object_contains_forbidden_field(self):
        """evaluate_forbidden result should be accessible on Evaluations."""
        from evalview.core.types import (
            Evaluations,
            ToolEvaluation,
            SequenceEvaluation,
            OutputEvaluation,
            ContainsChecks,
            CostEvaluation,
            LatencyEvaluation,
        )
        ev = Evaluations(
            tool_accuracy=ToolEvaluation(accuracy=1.0),
            sequence_correctness=SequenceEvaluation(
                correct=True, expected_sequence=[], actual_sequence=[]
            ),
            output_quality=OutputEvaluation(
                score=80.0,
                rationale="ok",
                contains_checks=ContainsChecks(),
                not_contains_checks=ContainsChecks(),
            ),
            cost=CostEvaluation(total_cost=0.0, threshold=float("inf"), passed=True),
            latency=LatencyEvaluation(total_latency=0.0, threshold=float("inf"), passed=True),
            forbidden_tools=ForbiddenToolEvaluation(violations=["bash"], passed=False),
        )
        assert ev.forbidden_tools is not None
        assert ev.forbidden_tools.passed is False

    def test_evaluations_forbidden_tools_defaults_to_none(self):
        """Evaluations.forbidden_tools should be None when not configured."""
        from evalview.core.types import (
            Evaluations,
            ToolEvaluation,
            SequenceEvaluation,
            OutputEvaluation,
            ContainsChecks,
            CostEvaluation,
            LatencyEvaluation,
        )
        ev = Evaluations(
            tool_accuracy=ToolEvaluation(accuracy=1.0),
            sequence_correctness=SequenceEvaluation(
                correct=True, expected_sequence=[], actual_sequence=[]
            ),
            output_quality=OutputEvaluation(
                score=80.0,
                rationale="ok",
                contains_checks=ContainsChecks(),
                not_contains_checks=ContainsChecks(),
            ),
            cost=CostEvaluation(total_cost=0.0, threshold=float("inf"), passed=True),
            latency=LatencyEvaluation(total_latency=0.0, threshold=float("inf"), passed=True),
        )
        assert ev.forbidden_tools is None


# ── Evaluator hard-fail integration ──────────────────────────────────────────


class TestEvaluatorForbiddenHardFail:
    """Integration tests: Evaluator must hard-fail on forbidden tool violations."""

    @pytest.mark.asyncio
    async def test_hard_fails_when_forbidden_tool_called(self):
        evaluator = Evaluator(skip_llm_judge=True)
        test_case = _make_test_case(forbidden=["bash"])
        trace = _make_trace("search", "bash")

        result = await evaluator.evaluate(test_case, trace)

        assert result.passed is False
        assert result.evaluations.forbidden_tools is not None
        assert result.evaluations.forbidden_tools.passed is False
        assert "bash" in result.evaluations.forbidden_tools.violations

    @pytest.mark.asyncio
    async def test_passes_when_no_forbidden_tool_called(self):
        evaluator = Evaluator(skip_llm_judge=True)
        test_case = _make_test_case(forbidden=["bash"], tools=["search"])
        trace = _make_trace("search")

        result = await evaluator.evaluate(test_case, trace)

        assert result.evaluations.forbidden_tools is not None
        assert result.evaluations.forbidden_tools.passed is True
        assert result.evaluations.forbidden_tools.violations == []
        # Whether the overall test passed depends on score, not tested here.

    @pytest.mark.asyncio
    async def test_forbidden_is_none_when_not_configured(self):
        evaluator = Evaluator(skip_llm_judge=True)
        test_case = _make_test_case(forbidden=None, tools=["search"])
        trace = _make_trace("search")

        result = await evaluator.evaluate(test_case, trace)

        assert result.evaluations.forbidden_tools is None

    @pytest.mark.asyncio
    async def test_forbidden_fail_overrides_high_score(self):
        """Even with a perfect score the test must fail on a forbidden tool call."""
        evaluator = Evaluator(skip_llm_judge=True)
        # No expected tools — only a forbidden constraint.
        test_case = _make_test_case(forbidden=["edit_file"], tools=None)
        # Agent calls edit_file, gets a high output score regardless.
        trace = _make_trace("search", "edit_file")
        trace.final_output = "Great answer with full detail."  # noqa: not a field, just test data

        result = await evaluator.evaluate(test_case, trace)

        assert result.passed is False, "Forbidden tool call must always cause hard-fail"
        assert result.evaluations.forbidden_tools is not None
        assert "edit_file" in result.evaluations.forbidden_tools.violations


# ── HTML reporter trace serialisation ────────────────────────────────────────


class TestHTMLReporterSpanSerialisation:
    """Unit tests for HTMLReporter._serialize_spans."""

    def _make_result_with_trace_context(self) -> "EvaluationResult":
        from evalview.core.types import (
            EvaluationResult,
            Evaluations,
            ToolEvaluation,
            SequenceEvaluation,
            OutputEvaluation,
            ContainsChecks,
            CostEvaluation,
            LatencyEvaluation,
        )

        tc = TraceContext(
            trace_id="abc123",
            root_span_id="root01",
            spans=[
                Span(
                    span_id="root01",
                    trace_id="abc123",
                    kind=SpanKind.AGENT,
                    name="Agent Execution",
                    start_time=datetime(2026, 1, 1, 0, 0, 0),
                    end_time=datetime(2026, 1, 1, 0, 0, 1),
                    duration_ms=1000.0,
                    status="ok",
                ),
                Span(
                    span_id="llm01",
                    parent_span_id="root01",
                    trace_id="abc123",
                    kind=SpanKind.LLM,
                    name="claude-sonnet-4-6",
                    start_time=datetime(2026, 1, 1, 0, 0, 0, 100000),
                    end_time=datetime(2026, 1, 1, 0, 0, 0, 800000),
                    duration_ms=700.0,
                    status="ok",
                    cost=0.0012,
                    llm=LLMCallInfo(
                        model="claude-sonnet-4-6",
                        provider="anthropic",
                        prompt="What is 2+2?",
                        completion="4",
                        prompt_tokens=10,
                        completion_tokens=1,
                        finish_reason="end_turn",
                    ),
                ),
                Span(
                    span_id="tool01",
                    parent_span_id="root01",
                    trace_id="abc123",
                    kind=SpanKind.TOOL,
                    name="calculator",
                    start_time=datetime(2026, 1, 1, 0, 0, 0, 900000),
                    end_time=datetime(2026, 1, 1, 0, 0, 1),
                    duration_ms=100.0,
                    status="ok",
                    tool=ToolCallInfo(
                        tool_name="calculator",
                        parameters={"expr": "2+2"},
                        result=4,
                    ),
                ),
            ],
            start_time=datetime(2026, 1, 1),
            end_time=datetime(2026, 1, 1, 0, 0, 1),
            total_llm_calls=1,
            total_tool_calls=1,
            total_cost=0.0012,
        )

        trace = _make_trace("calculator")
        trace.trace_context = tc

        evs = Evaluations(
            tool_accuracy=ToolEvaluation(accuracy=1.0, correct=["calculator"]),
            sequence_correctness=SequenceEvaluation(
                correct=True, expected_sequence=[], actual_sequence=[]
            ),
            output_quality=OutputEvaluation(
                score=90.0,
                rationale="Correct.",
                contains_checks=ContainsChecks(),
                not_contains_checks=ContainsChecks(),
            ),
            cost=CostEvaluation(total_cost=0.01, threshold=float("inf"), passed=True),
            latency=LatencyEvaluation(total_latency=100.0, threshold=float("inf"), passed=True),
        )

        return EvaluationResult(
            test_case="calculator-test",
            passed=True,
            score=90.0,
            evaluations=evs,
            trace=trace,
            timestamp=datetime.now(),
        )

    def test_serialises_from_trace_context(self):
        from evalview.reporters.html_reporter import HTMLReporter

        reporter = HTMLReporter()
        result = self._make_result_with_trace_context()
        spans = reporter._serialize_spans(result)

        assert len(spans) == 3  # agent, llm, tool

        kinds = [s["kind"] for s in spans]
        assert "agent" in kinds
        assert "llm" in kinds
        assert "tool" in kinds

    def test_llm_span_contains_prompt_and_completion(self):
        from evalview.reporters.html_reporter import HTMLReporter

        reporter = HTMLReporter()
        result = self._make_result_with_trace_context()
        spans = reporter._serialize_spans(result)

        llm_spans = [s for s in spans if s["kind"] == "llm"]
        assert len(llm_spans) == 1
        llm = llm_spans[0]
        assert llm["llm"]["prompt"] == "What is 2+2?"
        assert llm["llm"]["completion"] == "4"
        assert llm["llm"]["prompt_tokens"] == 10
        assert llm["llm"]["completion_tokens"] == 1
        assert llm["llm"]["finish_reason"] == "end_turn"

    def test_tool_span_contains_parameters_and_result(self):
        from evalview.reporters.html_reporter import HTMLReporter

        reporter = HTMLReporter()
        result = self._make_result_with_trace_context()
        spans = reporter._serialize_spans(result)

        tool_spans = [s for s in spans if s["kind"] == "tool"]
        assert len(tool_spans) == 1
        tool = tool_spans[0]
        assert tool["tool"]["tool_name"] == "calculator"
        assert "2+2" in tool["tool"]["parameters"]
        assert "4" in tool["tool"]["result"]

    def test_fallback_uses_step_traces_when_no_trace_context(self):
        from evalview.reporters.html_reporter import HTMLReporter
        from evalview.core.types import (
            EvaluationResult,
            Evaluations,
            ToolEvaluation,
            SequenceEvaluation,
            OutputEvaluation,
            ContainsChecks,
            CostEvaluation,
            LatencyEvaluation,
        )

        reporter = HTMLReporter()
        trace = _make_trace("search", "summarize")
        # Ensure no trace_context
        assert trace.trace_context is None

        evs = Evaluations(
            tool_accuracy=ToolEvaluation(accuracy=1.0),
            sequence_correctness=SequenceEvaluation(
                correct=True, expected_sequence=[], actual_sequence=[]
            ),
            output_quality=OutputEvaluation(
                score=80.0,
                rationale="ok",
                contains_checks=ContainsChecks(),
                not_contains_checks=ContainsChecks(),
            ),
            cost=CostEvaluation(total_cost=0.02, threshold=float("inf"), passed=True),
            latency=LatencyEvaluation(total_latency=200.0, threshold=float("inf"), passed=True),
        )
        result = EvaluationResult(
            test_case="fallback-test",
            passed=True,
            score=80.0,
            evaluations=evs,
            trace=trace,
            timestamp=datetime.now(),
        )

        spans = reporter._serialize_spans(result)

        assert len(spans) == 2
        assert all(s["kind"] == "tool" for s in spans)
        assert spans[0]["name"] == "search"
        assert spans[1]["name"] == "summarize"

    def test_long_prompt_is_truncated(self):
        from evalview.reporters.html_reporter import HTMLReporter

        reporter = HTMLReporter()
        result = self._make_result_with_trace_context()
        # Inject a very long prompt
        tc = result.trace.trace_context
        for sp in tc.spans:
            if sp.llm:
                sp.llm.prompt = "x" * 2000

        spans = reporter._serialize_spans(result)
        llm_spans = [s for s in spans if s["kind"] == "llm"]
        # Truncation limit is 600 chars + " …"
        assert len(llm_spans[0]["llm"]["prompt"]) <= 605
