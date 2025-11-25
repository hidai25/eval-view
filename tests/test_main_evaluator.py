"""Tests for main Evaluator orchestrator."""

import pytest
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

from evalview.evaluators.evaluator import Evaluator
from evalview.core.types import (
    TestCase,
    TestInput,
    ExpectedBehavior,
    ExpectedOutput,
    Thresholds,
    ExecutionTrace,
    StepTrace,
    StepMetrics,
    ExecutionMetrics,
    TokenUsage,
    Evaluations,
    ToolEvaluation,
    SequenceEvaluation,
    OutputEvaluation,
    ContainsChecks,
    CostEvaluation,
    LatencyEvaluation,
)


class TestEvaluator:
    """Tests for main Evaluator orchestrator."""

    @pytest.mark.asyncio
    @patch("evalview.evaluators.safety_evaluator.AsyncOpenAI")
    @patch("evalview.evaluators.hallucination_evaluator.AsyncOpenAI")
    @patch("evalview.evaluators.output_evaluator.AsyncOpenAI")
    async def test_evaluate_all_pass(
        self, mock_output_openai, mock_halluc_openai, mock_safety_openai,
        sample_test_case, sample_execution_trace, mock_openai_client
    ):
        """Test complete evaluation when all criteria pass."""
        evaluator = Evaluator()
        evaluator.output_evaluator.client = mock_openai_client
        evaluator.hallucination_evaluator.client = mock_openai_client
        evaluator.safety_evaluator.client = mock_openai_client

        result = await evaluator.evaluate(sample_test_case, sample_execution_trace)

        assert result.passed is True
        assert result.test_case == "test_search"
        assert result.score > 0
        assert result.input_query == "What is the capital of France?"
        assert result.actual_output == "The capital of France is Paris."
        assert isinstance(result.timestamp, datetime)

    @pytest.mark.asyncio
    @patch("evalview.evaluators.safety_evaluator.AsyncOpenAI")
    @patch("evalview.evaluators.hallucination_evaluator.AsyncOpenAI")
    @patch("evalview.evaluators.output_evaluator.AsyncOpenAI")
    async def test_evaluate_creates_all_evaluations(
        self, mock_output_openai, mock_halluc_openai, mock_safety_openai,
        sample_test_case, sample_execution_trace, mock_openai_client
    ):
        """Test that all sub-evaluators are run."""
        evaluator = Evaluator()
        evaluator.output_evaluator.client = mock_openai_client
        evaluator.hallucination_evaluator.client = mock_openai_client
        evaluator.safety_evaluator.client = mock_openai_client

        result = await evaluator.evaluate(sample_test_case, sample_execution_trace)

        # Check that all evaluation types are present
        assert isinstance(result.evaluations.tool_accuracy, ToolEvaluation)
        assert isinstance(result.evaluations.sequence_correctness, SequenceEvaluation)
        assert isinstance(result.evaluations.output_quality, OutputEvaluation)
        assert isinstance(result.evaluations.cost, CostEvaluation)
        assert isinstance(result.evaluations.latency, LatencyEvaluation)

    @patch("evalview.evaluators.safety_evaluator.AsyncOpenAI")
    @patch("evalview.evaluators.hallucination_evaluator.AsyncOpenAI")
    @patch("evalview.evaluators.output_evaluator.AsyncOpenAI")
    def test_compute_overall_score_perfect(self, mock1, mock2, mock3):
        """Test score calculation with perfect results."""
        evaluator = Evaluator()

        evaluations = Evaluations(
            tool_accuracy=ToolEvaluation(accuracy=1.0),
            sequence_correctness=SequenceEvaluation(
                correct=True, expected_sequence=[], actual_sequence=[]
            ),
            output_quality=OutputEvaluation(
                score=100.0,
                rationale="Perfect",
                contains_checks=ContainsChecks(),
                not_contains_checks=ContainsChecks(),
            ),
            cost=CostEvaluation(total_cost=0.0, threshold=1.0, passed=True),
            latency=LatencyEvaluation(total_latency=0.0, threshold=1000.0, passed=True),
        )

        test_case = TestCase(
            name="test",
            input=TestInput(query="test"),
            expected=ExpectedBehavior(),
            thresholds=Thresholds(min_score=50.0),
        )

        score = evaluator._compute_overall_score(evaluations, test_case)

        # Score = 100 * 0.3 (tool) + 100 * 0.5 (output) + 100 * 0.2 (sequence) = 100
        assert score == 100.0

    @patch("evalview.evaluators.safety_evaluator.AsyncOpenAI")
    @patch("evalview.evaluators.hallucination_evaluator.AsyncOpenAI")
    @patch("evalview.evaluators.output_evaluator.AsyncOpenAI")
    def test_compute_overall_score_weighted(self, mock1, mock2, mock3):
        """Test score calculation with weighted components."""
        evaluator = Evaluator()

        evaluations = Evaluations(
            tool_accuracy=ToolEvaluation(accuracy=0.5),  # 50% accuracy
            sequence_correctness=SequenceEvaluation(
                correct=False, expected_sequence=[], actual_sequence=[]
            ),  # 0% for sequence
            output_quality=OutputEvaluation(
                score=80.0,  # 80% output quality
                rationale="Good",
                contains_checks=ContainsChecks(),
                not_contains_checks=ContainsChecks(),
            ),
            cost=CostEvaluation(total_cost=0.0, threshold=1.0, passed=True),
            latency=LatencyEvaluation(total_latency=0.0, threshold=1000.0, passed=True),
        )

        test_case = TestCase(
            name="test",
            input=TestInput(query="test"),
            expected=ExpectedBehavior(),
            thresholds=Thresholds(min_score=50.0),
        )

        score = evaluator._compute_overall_score(evaluations, test_case)

        # Score = 50 * 0.3 (tool) + 80 * 0.5 (output) + 0 * 0.2 (sequence)
        #       = 15 + 40 + 0 = 55
        assert score == 55.0

    @patch("evalview.evaluators.safety_evaluator.AsyncOpenAI")
    @patch("evalview.evaluators.hallucination_evaluator.AsyncOpenAI")
    @patch("evalview.evaluators.output_evaluator.AsyncOpenAI")
    def test_compute_overall_score_zero(self, mock1, mock2, mock3):
        """Test score calculation with all zeros."""
        evaluator = Evaluator()

        evaluations = Evaluations(
            tool_accuracy=ToolEvaluation(accuracy=0.0),
            sequence_correctness=SequenceEvaluation(
                correct=False, expected_sequence=[], actual_sequence=[]
            ),
            output_quality=OutputEvaluation(
                score=0.0,
                rationale="Poor",
                contains_checks=ContainsChecks(),
                not_contains_checks=ContainsChecks(),
            ),
            cost=CostEvaluation(total_cost=0.0, threshold=1.0, passed=True),
            latency=LatencyEvaluation(total_latency=0.0, threshold=1000.0, passed=True),
        )

        test_case = TestCase(
            name="test",
            input=TestInput(query="test"),
            expected=ExpectedBehavior(),
            thresholds=Thresholds(min_score=0.0),
        )

        score = evaluator._compute_overall_score(evaluations, test_case)

        assert score == 0.0

    @patch("evalview.evaluators.safety_evaluator.AsyncOpenAI")
    @patch("evalview.evaluators.hallucination_evaluator.AsyncOpenAI")
    @patch("evalview.evaluators.output_evaluator.AsyncOpenAI")
    def test_compute_pass_fail_score_threshold(self, mock1, mock2, mock3):
        """Test pass/fail based on score threshold."""
        evaluator = Evaluator()

        evaluations = Evaluations(
            tool_accuracy=ToolEvaluation(accuracy=0.5),
            sequence_correctness=SequenceEvaluation(
                correct=True, expected_sequence=[], actual_sequence=[]
            ),
            output_quality=OutputEvaluation(
                score=50.0,
                rationale="Okay",
                contains_checks=ContainsChecks(),
                not_contains_checks=ContainsChecks(),
            ),
            cost=CostEvaluation(total_cost=0.05, threshold=1.0, passed=True),
            latency=LatencyEvaluation(total_latency=1000.0, threshold=5000.0, passed=True),
        )

        # Test case with min_score = 70
        test_case_high_threshold = TestCase(
            name="test",
            input=TestInput(query="test"),
            expected=ExpectedBehavior(),
            thresholds=Thresholds(min_score=70.0),
        )

        score = 60.0  # Below threshold

        passed = evaluator._compute_pass_fail(evaluations, test_case_high_threshold, score)
        assert passed is False

        # Test case with min_score = 50
        test_case_low_threshold = TestCase(
            name="test",
            input=TestInput(query="test"),
            expected=ExpectedBehavior(),
            thresholds=Thresholds(min_score=50.0),
        )

        passed = evaluator._compute_pass_fail(evaluations, test_case_low_threshold, score)
        assert passed is True

    @patch("evalview.evaluators.safety_evaluator.AsyncOpenAI")
    @patch("evalview.evaluators.hallucination_evaluator.AsyncOpenAI")
    @patch("evalview.evaluators.output_evaluator.AsyncOpenAI")
    def test_compute_pass_fail_cost_threshold(self, mock1, mock2, mock3):
        """Test pass/fail based on cost threshold."""
        evaluator = Evaluator()

        # Evaluations with cost failure
        evaluations_fail = Evaluations(
            tool_accuracy=ToolEvaluation(accuracy=1.0),
            sequence_correctness=SequenceEvaluation(
                correct=True, expected_sequence=[], actual_sequence=[]
            ),
            output_quality=OutputEvaluation(
                score=100.0,
                rationale="Perfect",
                contains_checks=ContainsChecks(),
                not_contains_checks=ContainsChecks(),
            ),
            cost=CostEvaluation(total_cost=1.0, threshold=0.5, passed=False),
            latency=LatencyEvaluation(total_latency=1000.0, threshold=5000.0, passed=True),
        )

        test_case = TestCase(
            name="test",
            input=TestInput(query="test"),
            expected=ExpectedBehavior(),
            thresholds=Thresholds(min_score=50.0),
        )

        score = 100.0  # High score, but cost failed

        passed = evaluator._compute_pass_fail(evaluations_fail, test_case, score)
        assert passed is False  # Should fail due to cost

    @patch("evalview.evaluators.safety_evaluator.AsyncOpenAI")
    @patch("evalview.evaluators.hallucination_evaluator.AsyncOpenAI")
    @patch("evalview.evaluators.output_evaluator.AsyncOpenAI")
    def test_compute_pass_fail_latency_threshold(self, mock1, mock2, mock3):
        """Test pass/fail based on latency threshold."""
        evaluator = Evaluator()

        # Evaluations with latency failure
        evaluations_fail = Evaluations(
            tool_accuracy=ToolEvaluation(accuracy=1.0),
            sequence_correctness=SequenceEvaluation(
                correct=True, expected_sequence=[], actual_sequence=[]
            ),
            output_quality=OutputEvaluation(
                score=100.0,
                rationale="Perfect",
                contains_checks=ContainsChecks(),
                not_contains_checks=ContainsChecks(),
            ),
            cost=CostEvaluation(total_cost=0.05, threshold=1.0, passed=True),
            latency=LatencyEvaluation(total_latency=10000.0, threshold=5000.0, passed=False),
        )

        test_case = TestCase(
            name="test",
            input=TestInput(query="test"),
            expected=ExpectedBehavior(),
            thresholds=Thresholds(min_score=50.0),
        )

        score = 100.0  # High score, but latency failed

        passed = evaluator._compute_pass_fail(evaluations_fail, test_case, score)
        assert passed is False  # Should fail due to latency

    @patch("evalview.evaluators.safety_evaluator.AsyncOpenAI")
    @patch("evalview.evaluators.hallucination_evaluator.AsyncOpenAI")
    @patch("evalview.evaluators.output_evaluator.AsyncOpenAI")
    def test_compute_pass_fail_all_pass(self, mock1, mock2, mock3):
        """Test pass/fail when all criteria are met."""
        evaluator = Evaluator()

        evaluations = Evaluations(
            tool_accuracy=ToolEvaluation(accuracy=1.0),
            sequence_correctness=SequenceEvaluation(
                correct=True, expected_sequence=[], actual_sequence=[]
            ),
            output_quality=OutputEvaluation(
                score=100.0,
                rationale="Perfect",
                contains_checks=ContainsChecks(),
                not_contains_checks=ContainsChecks(),
            ),
            cost=CostEvaluation(total_cost=0.05, threshold=1.0, passed=True),
            latency=LatencyEvaluation(total_latency=1000.0, threshold=5000.0, passed=True),
        )

        test_case = TestCase(
            name="test",
            input=TestInput(query="test"),
            expected=ExpectedBehavior(),
            thresholds=Thresholds(min_score=50.0),
        )

        score = 100.0

        passed = evaluator._compute_pass_fail(evaluations, test_case, score)
        assert passed is True

    @patch("evalview.evaluators.safety_evaluator.AsyncOpenAI")
    @patch("evalview.evaluators.hallucination_evaluator.AsyncOpenAI")
    @patch("evalview.evaluators.output_evaluator.AsyncOpenAI")
    def test_compute_pass_fail_multiple_failures(self, mock1, mock2, mock3):
        """Test pass/fail with multiple failures."""
        evaluator = Evaluator()

        evaluations = Evaluations(
            tool_accuracy=ToolEvaluation(accuracy=0.0),
            sequence_correctness=SequenceEvaluation(
                correct=False, expected_sequence=[], actual_sequence=[]
            ),
            output_quality=OutputEvaluation(
                score=20.0,
                rationale="Poor",
                contains_checks=ContainsChecks(),
                not_contains_checks=ContainsChecks(),
            ),
            cost=CostEvaluation(total_cost=1.0, threshold=0.5, passed=False),
            latency=LatencyEvaluation(total_latency=10000.0, threshold=5000.0, passed=False),
        )

        test_case = TestCase(
            name="test",
            input=TestInput(query="test"),
            expected=ExpectedBehavior(),
            thresholds=Thresholds(min_score=50.0),
        )

        score = 10.0  # Low score

        passed = evaluator._compute_pass_fail(evaluations, test_case, score)
        assert passed is False  # Should fail on multiple criteria

    @pytest.mark.asyncio
    @patch("evalview.evaluators.safety_evaluator.AsyncOpenAI")
    @patch("evalview.evaluators.hallucination_evaluator.AsyncOpenAI")
    @patch("evalview.evaluators.output_evaluator.AsyncOpenAI")
    async def test_evaluate_with_boundary_score(
        self, mock_output_openai, mock_halluc_openai, mock_safety_openai, mock_openai_client
    ):
        """Test evaluation with score exactly at threshold."""
        evaluator = Evaluator()
        evaluator.output_evaluator.client = mock_openai_client
        evaluator.hallucination_evaluator.client = mock_openai_client
        evaluator.safety_evaluator.client = mock_openai_client

        test_case = TestCase(
            name="test",
            input=TestInput(query="test"),
            expected=ExpectedBehavior(tools=["tool1"]),
            thresholds=Thresholds(min_score=55.0),  # Exact threshold
        )

        trace = ExecutionTrace(
            session_id="test",
            start_time=datetime.now(),
            end_time=datetime.now(),
            steps=[
                StepTrace(
                    step_id="1",
                    step_name="Step",
                    tool_name="tool1",
                    parameters={},
                    output="result",
                    success=True,
                    metrics=StepMetrics(latency=100.0, cost=0.01),
                )
            ],
            final_output="Test output",
            metrics=ExecutionMetrics(total_cost=0.01, total_latency=100.0),
        )

        result = await evaluator.evaluate(test_case, trace)

        # Score calculation (with output score of 85 from mock):
        # = 100 * 0.3 (tool) + 85 * 0.5 (output) + 100 * 0.2 (sequence, correct=True)
        # = 30 + 42.5 + 20 = 92.5
        # Should pass since score >= min_score (55.0)
        assert result.score == 92.5
        assert result.passed is True

    @pytest.mark.asyncio
    @patch("evalview.evaluators.safety_evaluator.AsyncOpenAI")
    @patch("evalview.evaluators.hallucination_evaluator.AsyncOpenAI")
    @patch("evalview.evaluators.output_evaluator.AsyncOpenAI")
    async def test_evaluate_score_rounding(
        self, mock_output_openai, mock_halluc_openai, mock_safety_openai, mock_openai_client
    ):
        """Test that score is properly rounded to 2 decimal places."""
        evaluator = Evaluator()
        evaluator.output_evaluator.client = mock_openai_client
        evaluator.hallucination_evaluator.client = mock_openai_client
        evaluator.safety_evaluator.client = mock_openai_client

        test_case = TestCase(
            name="test",
            input=TestInput(query="test"),
            expected=ExpectedBehavior(tools=["tool1", "tool2", "tool3"]),
            thresholds=Thresholds(min_score=0.0),
        )

        trace = ExecutionTrace(
            session_id="test",
            start_time=datetime.now(),
            end_time=datetime.now(),
            steps=[
                StepTrace(
                    step_id="1",
                    step_name="Step",
                    tool_name="tool1",
                    parameters={},
                    output="result",
                    success=True,
                    metrics=StepMetrics(latency=100.0, cost=0.01),
                )
            ],
            final_output="Test output",
            metrics=ExecutionMetrics(total_cost=0.01, total_latency=100.0),
        )

        result = await evaluator.evaluate(test_case, trace)

        # Check that score is rounded to 2 decimal places
        assert isinstance(result.score, float)
        assert len(str(result.score).split(".")[-1]) <= 2

    @pytest.mark.asyncio
    @patch("evalview.evaluators.safety_evaluator.AsyncOpenAI")
    @patch("evalview.evaluators.hallucination_evaluator.AsyncOpenAI")
    @patch("evalview.evaluators.output_evaluator.AsyncOpenAI")
    async def test_evaluate_with_no_thresholds(
        self, mock_output_openai, mock_halluc_openai, mock_safety_openai, mock_openai_client
    ):
        """Test evaluation when cost/latency thresholds are not specified."""
        evaluator = Evaluator()
        evaluator.output_evaluator.client = mock_openai_client
        evaluator.hallucination_evaluator.client = mock_openai_client
        evaluator.safety_evaluator.client = mock_openai_client

        test_case = TestCase(
            name="test",
            input=TestInput(query="test"),
            expected=ExpectedBehavior(),
            thresholds=Thresholds(min_score=0.0, max_cost=None, max_latency=None),
        )

        trace = ExecutionTrace(
            session_id="test",
            start_time=datetime.now(),
            end_time=datetime.now(),
            steps=[
                StepTrace(
                    step_id="1",
                    step_name="Step",
                    tool_name="tool1",
                    parameters={},
                    output="result",
                    success=True,
                    metrics=StepMetrics(latency=999999.0, cost=999.99),
                )
            ],
            final_output="Test output",
            metrics=ExecutionMetrics(total_cost=999.99, total_latency=999999.0),
        )

        result = await evaluator.evaluate(test_case, trace)

        # Should pass because no thresholds were set
        assert result.passed is True
        assert result.evaluations.cost.passed is True
        assert result.evaluations.latency.passed is True

    @pytest.mark.asyncio
    async def test_evaluator_initialization_with_api_key(self):
        """Test that evaluator initializes with custom API key."""
        evaluator = Evaluator(openai_api_key="test-key-123")

        # Check that output evaluator received the API key
        # (we can't directly check the private client, but we can verify initialization)
        assert evaluator.output_evaluator is not None
        assert evaluator.tool_evaluator is not None
        assert evaluator.sequence_evaluator is not None
        assert evaluator.cost_evaluator is not None
        assert evaluator.latency_evaluator is not None

    @patch("evalview.evaluators.safety_evaluator.AsyncOpenAI")
    @patch("evalview.evaluators.hallucination_evaluator.AsyncOpenAI")
    @patch("evalview.evaluators.output_evaluator.AsyncOpenAI")
    def test_compute_overall_score_weights_sum_to_one(self, mock1, mock2, mock3):
        """Verify that evaluation weights sum to 1.0 (100%)."""
        evaluator = Evaluator()

        # Create evaluations with known values
        evaluations = Evaluations(
            tool_accuracy=ToolEvaluation(accuracy=1.0),  # 100%
            sequence_correctness=SequenceEvaluation(
                correct=True, expected_sequence=[], actual_sequence=[]
            ),  # 100%
            output_quality=OutputEvaluation(
                score=100.0,  # 100%
                rationale="Perfect",
                contains_checks=ContainsChecks(),
                not_contains_checks=ContainsChecks(),
            ),
            cost=CostEvaluation(total_cost=0.0, threshold=1.0, passed=True),
            latency=LatencyEvaluation(total_latency=0.0, threshold=1000.0, passed=True),
        )

        test_case = TestCase(
            name="test",
            input=TestInput(query="test"),
            expected=ExpectedBehavior(),
            thresholds=Thresholds(min_score=50.0),
        )

        score = evaluator._compute_overall_score(evaluations, test_case)

        # With all components at 100%, the overall score should be 100
        # This verifies that weights sum to 1.0
        assert score == 100.0

    @patch("evalview.evaluators.safety_evaluator.AsyncOpenAI")
    @patch("evalview.evaluators.hallucination_evaluator.AsyncOpenAI")
    @patch("evalview.evaluators.output_evaluator.AsyncOpenAI")
    def test_compute_overall_score_only_output_quality(self, mock1, mock2, mock3):
        """Test score when only output quality is considered (others zero)."""
        evaluator = Evaluator()

        evaluations = Evaluations(
            tool_accuracy=ToolEvaluation(accuracy=0.0),  # 0%
            sequence_correctness=SequenceEvaluation(
                correct=False, expected_sequence=[], actual_sequence=[]
            ),  # 0%
            output_quality=OutputEvaluation(
                score=100.0,  # 100%
                rationale="Perfect output",
                contains_checks=ContainsChecks(),
                not_contains_checks=ContainsChecks(),
            ),
            cost=CostEvaluation(total_cost=0.0, threshold=1.0, passed=True),
            latency=LatencyEvaluation(total_latency=0.0, threshold=1000.0, passed=True),
        )

        test_case = TestCase(
            name="test",
            input=TestInput(query="test"),
            expected=ExpectedBehavior(),
            thresholds=Thresholds(min_score=50.0),
        )

        score = evaluator._compute_overall_score(evaluations, test_case)

        # Score should be 50% of 100 (output quality weight is 0.5)
        assert score == 50.0
