"""Tests for core type definitions (Pydantic models)."""

import pytest
from datetime import datetime
from pydantic import ValidationError

from evalview.core.types import (
    TestCase as TestCaseModel,
    TestInput as TestInputModel,
    ExpectedBehavior,
    ExpectedOutput,
    Thresholds,
    MetricThreshold,
    ExecutionTrace,
    StepTrace,
    StepMetrics,
    ExecutionMetrics,
    TokenUsage,
    ToolEvaluation,
    SequenceEvaluation,
    OutputEvaluation,
    ContainsChecks,
    CostEvaluation,
    LatencyEvaluation,
    CostBreakdown,
    LatencyBreakdown,
    Evaluations,
    EvaluationResult,
)


# ============================================================================
# Test Case Types Tests
# ============================================================================


class TestTestInput:
    """Tests for TestInput model."""

    def test_valid_input(self):
        """Test creating a valid TestInput."""
        test_input = TestInputModel(query="test query", context={"key": "value"})
        assert test_input.query == "test query"
        assert test_input.context == {"key": "value"}

    def test_input_without_context(self):
        """Test TestInput with no context (optional field)."""
        test_input = TestInputModel(query="test query")
        assert test_input.query == "test query"
        assert test_input.context is None

    def test_input_missing_query(self):
        """Test that query is required."""
        with pytest.raises(ValidationError) as exc_info:
            TestInputModel()
        assert "query" in str(exc_info.value)


class TestExpectedOutput:
    """Tests for ExpectedOutput model."""

    def test_valid_expected_output(self):
        """Test creating a valid ExpectedOutput."""
        output = ExpectedOutput(
            contains=["Paris", "France"],
            not_contains=["error"],
            json_schema={"type": "object"},
        )
        assert output.contains == ["Paris", "France"]
        assert output.not_contains == ["error"]
        assert output.json_schema == {"type": "object"}

    def test_expected_output_all_optional(self):
        """Test ExpectedOutput with all optional fields None."""
        output = ExpectedOutput()
        assert output.contains is None
        assert output.not_contains is None
        assert output.json_schema is None


class TestMetricThreshold:
    """Tests for MetricThreshold model."""

    def test_valid_metric_threshold(self):
        """Test creating a valid MetricThreshold."""
        threshold = MetricThreshold(value=100.0, tolerance=10.0)
        assert threshold.value == 100.0
        assert threshold.tolerance == 10.0

    def test_metric_threshold_missing_fields(self):
        """Test that both value and tolerance are required."""
        with pytest.raises(ValidationError):
            MetricThreshold(value=100.0)
        with pytest.raises(ValidationError):
            MetricThreshold(tolerance=10.0)


class TestExpectedBehavior:
    """Tests for ExpectedBehavior model."""

    def test_valid_expected_behavior(self):
        """Test creating a valid ExpectedBehavior."""
        behavior = ExpectedBehavior(
            tools=["search", "summarize"],
            tool_sequence=["search", "summarize"],
            output=ExpectedOutput(contains=["test"]),
            metrics={"latency": MetricThreshold(value=1000.0, tolerance=100.0)},
        )
        assert behavior.tools == ["search", "summarize"]
        assert behavior.tool_sequence == ["search", "summarize"]
        assert behavior.output.contains == ["test"]
        assert "latency" in behavior.metrics

    def test_expected_behavior_all_optional(self):
        """Test ExpectedBehavior with all fields None."""
        behavior = ExpectedBehavior()
        assert behavior.tools is None
        assert behavior.tool_sequence is None
        assert behavior.output is None
        assert behavior.metrics is None


class TestThresholds:
    """Tests for Thresholds model."""

    def test_valid_thresholds(self):
        """Test creating valid Thresholds."""
        thresholds = Thresholds(min_score=70.0, max_cost=0.50, max_latency=5000.0)
        assert thresholds.min_score == 70.0
        assert thresholds.max_cost == 0.50
        assert thresholds.max_latency == 5000.0

    def test_thresholds_min_score_required(self):
        """Test that min_score is required."""
        with pytest.raises(ValidationError):
            Thresholds()

    def test_thresholds_optional_fields(self):
        """Test that max_cost and max_latency are optional."""
        thresholds = Thresholds(min_score=50.0)
        assert thresholds.min_score == 50.0
        assert thresholds.max_cost is None
        assert thresholds.max_latency is None

    def test_thresholds_min_score_constraint_min(self):
        """Test min_score constraint: must be >= 0."""
        with pytest.raises(ValidationError) as exc_info:
            Thresholds(min_score=-1.0)
        assert "greater than or equal to 0" in str(exc_info.value).lower()

    def test_thresholds_min_score_constraint_max(self):
        """Test min_score constraint: must be <= 100."""
        with pytest.raises(ValidationError) as exc_info:
            Thresholds(min_score=101.0)
        assert "less than or equal to 100" in str(exc_info.value).lower()

    def test_thresholds_boundary_values(self):
        """Test boundary values for min_score."""
        # Should accept 0.0
        thresholds_zero = Thresholds(min_score=0.0)
        assert thresholds_zero.min_score == 0.0

        # Should accept 100.0
        thresholds_hundred = Thresholds(min_score=100.0)
        assert thresholds_hundred.min_score == 100.0


class TestTestCase:
    """Tests for TestCase model."""

    def test_valid_test_case(self, sample_test_case):
        """Test creating a valid TestCase."""
        assert sample_test_case.name == "test_search"
        assert sample_test_case.description == "Test search functionality"
        assert sample_test_case.input.query == "What is the capital of France?"
        assert sample_test_case.expected.tools == ["search", "summarize"]
        assert sample_test_case.thresholds.min_score == 70.0

    def test_test_case_with_adapter_override(self):
        """Test TestCase with adapter override."""
        test_case = TestCaseModel(
            name="test",
            input=TestInputModel(query="test"),
            expected=ExpectedBehavior(),
            thresholds=Thresholds(min_score=50.0),
            adapter="langgraph",
            endpoint="http://localhost:8000",
            adapter_config={"streaming": True},
        )
        assert test_case.adapter == "langgraph"
        assert test_case.endpoint == "http://localhost:8000"
        assert test_case.adapter_config == {"streaming": True}

    def test_test_case_missing_required_fields(self):
        """Test that required fields are enforced."""
        with pytest.raises(ValidationError) as exc_info:
            TestCaseModel(name="test")
        errors_str = str(exc_info.value)
        assert "input" in errors_str or "expected" in errors_str or "thresholds" in errors_str


# ============================================================================
# Execution Trace Types Tests
# ============================================================================


class TestTokenUsage:
    """Tests for TokenUsage model."""

    def test_valid_token_usage(self):
        """Test creating valid TokenUsage."""
        tokens = TokenUsage(input_tokens=100, output_tokens=200, cached_tokens=50)
        assert tokens.input_tokens == 100
        assert tokens.output_tokens == 200
        assert tokens.cached_tokens == 50

    def test_token_usage_defaults_to_zero(self):
        """Test that token counts default to 0."""
        tokens = TokenUsage()
        assert tokens.input_tokens == 0
        assert tokens.output_tokens == 0
        assert tokens.cached_tokens == 0

    def test_total_tokens_property(self):
        """Test the total_tokens property calculation."""
        tokens = TokenUsage(input_tokens=100, output_tokens=200, cached_tokens=50)
        assert tokens.total_tokens == 350

    def test_total_tokens_with_zeros(self):
        """Test total_tokens with zero values."""
        tokens = TokenUsage()
        assert tokens.total_tokens == 0


class TestStepMetrics:
    """Tests for StepMetrics model."""

    def test_valid_step_metrics(self):
        """Test creating valid StepMetrics."""
        metrics = StepMetrics(
            latency=1500.0,
            cost=0.02,
            tokens=TokenUsage(input_tokens=50, output_tokens=100),
        )
        assert metrics.latency == 1500.0
        assert metrics.cost == 0.02
        assert metrics.tokens.input_tokens == 50

    def test_step_metrics_without_tokens(self):
        """Test StepMetrics with no tokens (optional)."""
        metrics = StepMetrics(latency=1000.0, cost=0.01)
        assert metrics.latency == 1000.0
        assert metrics.cost == 0.01
        assert metrics.tokens is None


class TestStepTrace:
    """Tests for StepTrace model."""

    def test_valid_step_trace(self):
        """Test creating a valid StepTrace."""
        step = StepTrace(
            step_id="step-1",
            step_name="Search",
            tool_name="search",
            parameters={"query": "test"},
            output={"results": ["result1"]},
            success=True,
            error=None,
            metrics=StepMetrics(latency=1500.0, cost=0.02),
        )
        assert step.step_id == "step-1"
        assert step.tool_name == "search"
        assert step.success is True

    def test_step_trace_with_error(self):
        """Test StepTrace with an error."""
        step = StepTrace(
            step_id="step-1",
            step_name="Failed Step",
            tool_name="search",
            parameters={},
            output=None,
            success=False,
            error="Connection timeout",
            metrics=StepMetrics(latency=5000.0, cost=0.0),
        )
        assert step.success is False
        assert step.error == "Connection timeout"


class TestExecutionMetrics:
    """Tests for ExecutionMetrics model."""

    def test_valid_execution_metrics(self):
        """Test creating valid ExecutionMetrics."""
        metrics = ExecutionMetrics(
            total_cost=0.05,
            total_latency=3000.0,
            total_tokens=TokenUsage(input_tokens=100, output_tokens=200),
        )
        assert metrics.total_cost == 0.05
        assert metrics.total_latency == 3000.0
        assert metrics.total_tokens.total_tokens == 300

    def test_execution_metrics_without_tokens(self):
        """Test ExecutionMetrics with no tokens."""
        metrics = ExecutionMetrics(total_cost=0.05, total_latency=3000.0)
        assert metrics.total_tokens is None


class TestExecutionTrace:
    """Tests for ExecutionTrace model."""

    def test_valid_execution_trace(self, sample_execution_trace):
        """Test creating a valid ExecutionTrace."""
        assert sample_execution_trace.session_id == "test-session-123"
        assert len(sample_execution_trace.steps) == 2
        assert sample_execution_trace.final_output == "The capital of France is Paris."
        assert sample_execution_trace.metrics.total_cost == 0.03

    def test_execution_trace_empty_steps(self, empty_trace):
        """Test ExecutionTrace with no steps."""
        assert len(empty_trace.steps) == 0
        assert empty_trace.final_output == ""
        assert empty_trace.metrics.total_cost == 0.0

    def test_execution_trace_time_ordering(self):
        """Test that start_time is before end_time."""
        start = datetime(2025, 1, 1, 12, 0, 0)
        end = datetime(2025, 1, 1, 12, 0, 5)
        trace = ExecutionTrace(
            session_id="test",
            start_time=start,
            end_time=end,
            steps=[],
            final_output="test",
            metrics=ExecutionMetrics(total_cost=0.0, total_latency=5000.0),
        )
        assert trace.start_time < trace.end_time


# ============================================================================
# Evaluation Result Types Tests
# ============================================================================


class TestToolEvaluation:
    """Tests for ToolEvaluation model."""

    def test_valid_tool_evaluation(self):
        """Test creating a valid ToolEvaluation."""
        eval_result = ToolEvaluation(
            accuracy=0.75,
            missing=["tool3"],
            unexpected=["tool4"],
            correct=["tool1", "tool2"],
        )
        assert eval_result.accuracy == 0.75
        assert eval_result.missing == ["tool3"]
        assert eval_result.unexpected == ["tool4"]
        assert eval_result.correct == ["tool1", "tool2"]

    def test_tool_evaluation_accuracy_constraint_min(self):
        """Test accuracy constraint: must be >= 0."""
        with pytest.raises(ValidationError):
            ToolEvaluation(accuracy=-0.1)

    def test_tool_evaluation_accuracy_constraint_max(self):
        """Test accuracy constraint: must be <= 1."""
        with pytest.raises(ValidationError):
            ToolEvaluation(accuracy=1.1)

    def test_tool_evaluation_default_lists(self):
        """Test that lists default to empty."""
        eval_result = ToolEvaluation(accuracy=1.0)
        assert eval_result.missing == []
        assert eval_result.unexpected == []
        assert eval_result.correct == []


class TestSequenceEvaluation:
    """Tests for SequenceEvaluation model."""

    def test_valid_sequence_evaluation(self):
        """Test creating a valid SequenceEvaluation."""
        eval_result = SequenceEvaluation(
            correct=True,
            expected_sequence=["tool1", "tool2"],
            actual_sequence=["tool1", "tool2"],
            violations=[],
        )
        assert eval_result.correct is True
        assert eval_result.expected_sequence == ["tool1", "tool2"]
        assert eval_result.violations == []

    def test_sequence_evaluation_with_violations(self):
        """Test SequenceEvaluation with violations."""
        eval_result = SequenceEvaluation(
            correct=False,
            expected_sequence=["tool1", "tool2"],
            actual_sequence=["tool2", "tool1"],
            violations=["Step 1: expected 'tool1', got 'tool2'"],
        )
        assert eval_result.correct is False
        assert len(eval_result.violations) == 1


class TestContainsChecks:
    """Tests for ContainsChecks model."""

    def test_valid_contains_checks(self):
        """Test creating valid ContainsChecks."""
        checks = ContainsChecks(passed=["Paris", "France"], failed=["London"])
        assert checks.passed == ["Paris", "France"]
        assert checks.failed == ["London"]

    def test_contains_checks_default_empty(self):
        """Test that lists default to empty."""
        checks = ContainsChecks()
        assert checks.passed == []
        assert checks.failed == []


class TestOutputEvaluation:
    """Tests for OutputEvaluation model."""

    def test_valid_output_evaluation(self):
        """Test creating a valid OutputEvaluation."""
        eval_result = OutputEvaluation(
            score=85.0,
            rationale="Good output quality",
            contains_checks=ContainsChecks(passed=["Paris"]),
            not_contains_checks=ContainsChecks(passed=["error"]),
        )
        assert eval_result.score == 85.0
        assert eval_result.rationale == "Good output quality"

    def test_output_evaluation_score_constraint_min(self):
        """Test score constraint: must be >= 0."""
        with pytest.raises(ValidationError):
            OutputEvaluation(
                score=-1.0,
                rationale="test",
                contains_checks=ContainsChecks(),
                not_contains_checks=ContainsChecks(),
            )

    def test_output_evaluation_score_constraint_max(self):
        """Test score constraint: must be <= 100."""
        with pytest.raises(ValidationError):
            OutputEvaluation(
                score=101.0,
                rationale="test",
                contains_checks=ContainsChecks(),
                not_contains_checks=ContainsChecks(),
            )


class TestCostEvaluation:
    """Tests for CostEvaluation model."""

    def test_valid_cost_evaluation(self):
        """Test creating a valid CostEvaluation."""
        eval_result = CostEvaluation(
            total_cost=0.05,
            threshold=0.10,
            passed=True,
            breakdown=[CostBreakdown(step_id="step-1", cost=0.05)],
        )
        assert eval_result.total_cost == 0.05
        assert eval_result.threshold == 0.10
        assert eval_result.passed is True

    def test_cost_evaluation_default_breakdown(self):
        """Test that breakdown defaults to empty list."""
        eval_result = CostEvaluation(total_cost=0.05, threshold=0.10, passed=True)
        assert eval_result.breakdown == []


class TestLatencyEvaluation:
    """Tests for LatencyEvaluation model."""

    def test_valid_latency_evaluation(self):
        """Test creating a valid LatencyEvaluation."""
        eval_result = LatencyEvaluation(
            total_latency=3000.0,
            threshold=5000.0,
            passed=True,
            breakdown=[LatencyBreakdown(step_id="step-1", latency=1500.0)],
        )
        assert eval_result.total_latency == 3000.0
        assert eval_result.threshold == 5000.0
        assert eval_result.passed is True

    def test_latency_evaluation_default_breakdown(self):
        """Test that breakdown defaults to empty list."""
        eval_result = LatencyEvaluation(total_latency=3000.0, threshold=5000.0, passed=True)
        assert eval_result.breakdown == []


class TestEvaluations:
    """Tests for Evaluations model."""

    def test_valid_evaluations(self):
        """Test creating a valid Evaluations object."""
        evaluations = Evaluations(
            tool_accuracy=ToolEvaluation(accuracy=1.0),
            sequence_correctness=SequenceEvaluation(
                correct=True, expected_sequence=[], actual_sequence=[]
            ),
            output_quality=OutputEvaluation(
                score=85.0,
                rationale="Good",
                contains_checks=ContainsChecks(),
                not_contains_checks=ContainsChecks(),
            ),
            cost=CostEvaluation(total_cost=0.05, threshold=0.10, passed=True),
            latency=LatencyEvaluation(total_latency=3000.0, threshold=5000.0, passed=True),
        )
        assert evaluations.tool_accuracy.accuracy == 1.0
        assert evaluations.output_quality.score == 85.0


class TestEvaluationResult:
    """Tests for EvaluationResult model."""

    def test_valid_evaluation_result(self, sample_execution_trace):
        """Test creating a valid EvaluationResult."""
        evaluations = Evaluations(
            tool_accuracy=ToolEvaluation(accuracy=1.0),
            sequence_correctness=SequenceEvaluation(
                correct=True, expected_sequence=[], actual_sequence=[]
            ),
            output_quality=OutputEvaluation(
                score=85.0,
                rationale="Good",
                contains_checks=ContainsChecks(),
                not_contains_checks=ContainsChecks(),
            ),
            cost=CostEvaluation(total_cost=0.05, threshold=0.10, passed=True),
            latency=LatencyEvaluation(total_latency=3000.0, threshold=5000.0, passed=True),
        )

        result = EvaluationResult(
            test_case="test_search",
            passed=True,
            score=85.0,
            evaluations=evaluations,
            trace=sample_execution_trace,
            timestamp=datetime(2025, 1, 1, 12, 0, 0),
            input_query="What is the capital of France?",
            actual_output="Paris is the capital of France.",
        )

        assert result.test_case == "test_search"
        assert result.passed is True
        assert result.score == 85.0
        assert result.input_query == "What is the capital of France?"

    def test_evaluation_result_score_constraint(self):
        """Test score constraint in EvaluationResult."""
        with pytest.raises(ValidationError):
            EvaluationResult(
                test_case="test",
                passed=False,
                score=150.0,  # Invalid: > 100
                evaluations=None,
                trace=None,
                timestamp=datetime.now(),
            )


# ============================================================================
# Type Coercion Validator Tests
# ============================================================================


class TestStepMetricsCoercion:
    """Tests for StepMetrics type coercion validators."""

    def test_coerce_none_latency_to_zero(self):
        """Test that None latency is coerced to 0.0."""
        metrics = StepMetrics(latency=None, cost=0.01)
        assert metrics.latency == 0.0

    def test_coerce_none_cost_to_zero(self):
        """Test that None cost is coerced to 0.0."""
        metrics = StepMetrics(latency=1000.0, cost=None)
        assert metrics.cost == 0.0

    def test_default_latency_and_cost(self):
        """Test that latency and cost default to 0.0."""
        metrics = StepMetrics()
        assert metrics.latency == 0.0
        assert metrics.cost == 0.0

    def test_coerce_int_tokens_to_token_usage(self):
        """Test that int tokens is coerced to TokenUsage."""
        metrics = StepMetrics(latency=1000.0, cost=0.01, tokens=1500)
        assert isinstance(metrics.tokens, TokenUsage)
        assert metrics.tokens.output_tokens == 1500
        assert metrics.tokens.input_tokens == 0
        assert metrics.tokens.cached_tokens == 0

    def test_coerce_dict_tokens_to_token_usage(self):
        """Test that dict tokens is coerced to TokenUsage."""
        metrics = StepMetrics(
            latency=1000.0,
            cost=0.01,
            tokens={"input_tokens": 100, "output_tokens": 200, "cached_tokens": 50},
        )
        assert isinstance(metrics.tokens, TokenUsage)
        assert metrics.tokens.input_tokens == 100
        assert metrics.tokens.output_tokens == 200
        assert metrics.tokens.cached_tokens == 50

    def test_token_usage_passes_through(self):
        """Test that TokenUsage object passes through unchanged."""
        token_usage = TokenUsage(input_tokens=100, output_tokens=200)
        metrics = StepMetrics(latency=1000.0, cost=0.01, tokens=token_usage)
        assert metrics.tokens is token_usage

    def test_invalid_tokens_type_raises_error(self):
        """Test that invalid tokens type raises ValueError."""
        with pytest.raises(ValidationError) as exc_info:
            StepMetrics(latency=1000.0, cost=0.01, tokens="invalid")
        assert "TokenUsage, dict, or int" in str(exc_info.value)

    def test_invalid_latency_type_raises_error(self):
        """Test that invalid latency type raises ValueError."""
        with pytest.raises(ValidationError) as exc_info:
            StepMetrics(latency="not_a_number", cost=0.01)
        assert "numeric value" in str(exc_info.value)


class TestExecutionMetricsCoercion:
    """Tests for ExecutionMetrics type coercion validators."""

    def test_coerce_int_total_tokens_to_token_usage(self):
        """Test that int total_tokens is coerced to TokenUsage."""
        metrics = ExecutionMetrics(total_cost=0.05, total_latency=3000.0, total_tokens=1500)
        assert isinstance(metrics.total_tokens, TokenUsage)
        assert metrics.total_tokens.output_tokens == 1500
        assert metrics.total_tokens.input_tokens == 0

    def test_coerce_dict_total_tokens_to_token_usage(self):
        """Test that dict total_tokens is coerced to TokenUsage."""
        metrics = ExecutionMetrics(
            total_cost=0.05,
            total_latency=3000.0,
            total_tokens={"input_tokens": 500, "output_tokens": 1000},
        )
        assert isinstance(metrics.total_tokens, TokenUsage)
        assert metrics.total_tokens.input_tokens == 500
        assert metrics.total_tokens.output_tokens == 1000

    def test_token_usage_passes_through(self):
        """Test that TokenUsage object passes through unchanged."""
        token_usage = TokenUsage(input_tokens=500, output_tokens=1000)
        metrics = ExecutionMetrics(
            total_cost=0.05, total_latency=3000.0, total_tokens=token_usage
        )
        assert metrics.total_tokens is token_usage

    def test_none_total_tokens_remains_none(self):
        """Test that None total_tokens remains None."""
        metrics = ExecutionMetrics(total_cost=0.05, total_latency=3000.0, total_tokens=None)
        assert metrics.total_tokens is None

    def test_invalid_total_tokens_type_raises_error(self):
        """Test that invalid total_tokens type raises ValueError."""
        with pytest.raises(ValidationError) as exc_info:
            ExecutionMetrics(total_cost=0.05, total_latency=3000.0, total_tokens="invalid")
        assert "TokenUsage, dict, or int" in str(exc_info.value)


class TestExecutionTraceCoercion:
    """Tests for ExecutionTrace datetime coercion validators."""

    def test_coerce_iso_string_to_datetime(self):
        """Test that ISO format string is coerced to datetime."""
        trace = ExecutionTrace(
            session_id="test-123",
            start_time="2025-01-15T10:30:00",
            end_time="2025-01-15T10:30:05",
            steps=[],
            final_output="test output",
            metrics=ExecutionMetrics(total_cost=0.0, total_latency=5000.0),
        )
        assert isinstance(trace.start_time, datetime)
        assert isinstance(trace.end_time, datetime)
        assert trace.start_time.year == 2025
        assert trace.start_time.month == 1
        assert trace.start_time.day == 15
        assert trace.start_time.hour == 10
        assert trace.start_time.minute == 30

    def test_coerce_iso_string_with_timezone(self):
        """Test that ISO format with timezone is coerced correctly."""
        trace = ExecutionTrace(
            session_id="test-123",
            start_time="2025-01-15T10:30:00+00:00",
            end_time="2025-01-15T10:30:05+00:00",
            steps=[],
            final_output="test output",
            metrics=ExecutionMetrics(total_cost=0.0, total_latency=5000.0),
        )
        assert isinstance(trace.start_time, datetime)
        assert trace.start_time.tzinfo is not None

    def test_coerce_iso_string_with_z_timezone(self):
        """Test that ISO format with Z timezone is coerced correctly."""
        trace = ExecutionTrace(
            session_id="test-123",
            start_time="2025-01-15T10:30:00Z",
            end_time="2025-01-15T10:30:05Z",
            steps=[],
            final_output="test output",
            metrics=ExecutionMetrics(total_cost=0.0, total_latency=5000.0),
        )
        assert isinstance(trace.start_time, datetime)

    def test_datetime_passes_through(self):
        """Test that datetime object passes through unchanged."""
        start = datetime(2025, 1, 15, 10, 30, 0)
        end = datetime(2025, 1, 15, 10, 30, 5)
        trace = ExecutionTrace(
            session_id="test-123",
            start_time=start,
            end_time=end,
            steps=[],
            final_output="test output",
            metrics=ExecutionMetrics(total_cost=0.0, total_latency=5000.0),
        )
        assert trace.start_time is start
        assert trace.end_time is end

    def test_invalid_datetime_string_raises_error(self):
        """Test that invalid datetime string raises ValueError."""
        with pytest.raises(ValidationError) as exc_info:
            ExecutionTrace(
                session_id="test-123",
                start_time="not-a-date",
                end_time="2025-01-15T10:30:05",
                steps=[],
                final_output="test output",
                metrics=ExecutionMetrics(total_cost=0.0, total_latency=5000.0),
            )
        assert "Invalid datetime format" in str(exc_info.value)
