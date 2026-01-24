"""Tests for main Evaluator orchestrator."""

import pytest
from datetime import datetime
from unittest.mock import patch

from evalview.core.llm_provider import LLMProvider
from evalview.evaluators.evaluator import Evaluator
from evalview.core.types import (
    TestCase as TestCaseModel,
    TestInput as TestInputModel,
    ExpectedBehavior,
    Thresholds,
    ExecutionTrace,
    StepTrace,
    StepMetrics,
    ExecutionMetrics,
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
    @patch("evalview.core.llm_provider.LLMClient.chat_completion")
    @patch("evalview.core.llm_provider.select_provider")
    async def test_evaluate_all_pass(
        self,
        mock_select_provider,
        mock_chat_completion,
        sample_test_case,
        sample_execution_trace,
    ):
        """Test complete evaluation when all criteria pass."""
        mock_select_provider.return_value = (LLMProvider.OPENAI, "fake-key")
        mock_chat_completion.return_value = {"score": 85, "rationale": "Good answer"}

        evaluator = Evaluator()

        result = await evaluator.evaluate(sample_test_case, sample_execution_trace)

        assert result.passed is True
        assert result.test_case == "test_search"
        assert result.score > 0
        assert result.input_query == "What is the capital of France?"
        assert result.actual_output == "The capital of France is Paris."
        assert isinstance(result.timestamp, datetime)

    @pytest.mark.asyncio
    @patch("evalview.core.llm_provider.LLMClient.chat_completion")
    @patch("evalview.core.llm_provider.select_provider")
    async def test_evaluate_creates_all_evaluations(
        self,
        mock_select_provider,
        mock_chat_completion,
        sample_test_case,
        sample_execution_trace,
    ):
        """Test that all sub-evaluators are run."""
        mock_select_provider.return_value = (LLMProvider.OPENAI, "fake-key")
        mock_chat_completion.return_value = {"score": 85, "rationale": "Good answer"}

        evaluator = Evaluator()

        result = await evaluator.evaluate(sample_test_case, sample_execution_trace)

        # Check that all evaluation types are present
        assert isinstance(result.evaluations.tool_accuracy, ToolEvaluation)
        assert isinstance(result.evaluations.sequence_correctness, SequenceEvaluation)
        assert isinstance(result.evaluations.output_quality, OutputEvaluation)
        assert isinstance(result.evaluations.cost, CostEvaluation)
        assert isinstance(result.evaluations.latency, LatencyEvaluation)

    @patch("evalview.core.llm_provider.select_provider")
    def test_compute_overall_score_perfect(self, mock_select_provider):
        """Test score calculation with perfect results."""
        mock_select_provider.return_value = (LLMProvider.OPENAI, "fake-key")
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

        test_case = TestCaseModel(
            name="test",
            input=TestInputModel(query="test"),
            expected=ExpectedBehavior(),
            thresholds=Thresholds(min_score=50.0),
        )

        score = evaluator._compute_overall_score(evaluations, test_case)

        # Score = 100 * 0.3 (tool) + 100 * 0.5 (output) + 100 * 0.2 (sequence) = 100
        assert score == 100.0

    @patch("evalview.core.llm_provider.select_provider")
    def test_compute_overall_score_weighted(self, mock_select_provider):
        """Test score calculation with weighted components."""
        mock_select_provider.return_value = (LLMProvider.OPENAI, "fake-key")
        evaluator = Evaluator()

        evaluations = Evaluations(
            tool_accuracy=ToolEvaluation(accuracy=0.5),  # 50% accuracy
            sequence_correctness=SequenceEvaluation(
                correct=False, expected_sequence=[], actual_sequence=[], progress_score=0.0
            ),  # 0% for sequence (explicit progress_score=0.0)
            output_quality=OutputEvaluation(
                score=80.0,  # 80% output quality
                rationale="Good",
                contains_checks=ContainsChecks(),
                not_contains_checks=ContainsChecks(),
            ),
            cost=CostEvaluation(total_cost=0.0, threshold=1.0, passed=True),
            latency=LatencyEvaluation(total_latency=0.0, threshold=1000.0, passed=True),
        )

        test_case = TestCaseModel(
            name="test",
            input=TestInputModel(query="test"),
            expected=ExpectedBehavior(),
            thresholds=Thresholds(min_score=50.0),
        )

        score = evaluator._compute_overall_score(evaluations, test_case)

        # Score = 50 * 0.3 (tool) + 80 * 0.5 (output) + 0 * 0.2 (sequence)
        #       = 15 + 40 + 0 = 55
        assert score == 55.0

    @patch("evalview.core.llm_provider.select_provider")
    def test_compute_overall_score_zero(self, mock_select_provider):
        """Test score calculation with all zeros."""
        mock_select_provider.return_value = (LLMProvider.OPENAI, "fake-key")
        evaluator = Evaluator()

        evaluations = Evaluations(
            tool_accuracy=ToolEvaluation(accuracy=0.0),
            sequence_correctness=SequenceEvaluation(
                correct=False, expected_sequence=[], actual_sequence=[], progress_score=0.0
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

        test_case = TestCaseModel(
            name="test",
            input=TestInputModel(query="test"),
            expected=ExpectedBehavior(),
            thresholds=Thresholds(min_score=0.0),
        )

        score = evaluator._compute_overall_score(evaluations, test_case)

        # With progress_score=0.0 explicitly set, sequence contributes 0 points
        assert score == 0.0

    @patch("evalview.core.llm_provider.select_provider")
    def test_compute_pass_fail_score_threshold(self, mock_select_provider):
        """Test pass/fail based on score threshold."""
        mock_select_provider.return_value = (LLMProvider.OPENAI, "fake-key")
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
        test_case_high_threshold = TestCaseModel(
            name="test",
            input=TestInputModel(query="test"),
            expected=ExpectedBehavior(),
            thresholds=Thresholds(min_score=70.0),
        )

        score = 60.0  # Below threshold

        passed = evaluator._compute_pass_fail(evaluations, test_case_high_threshold, score)
        assert passed is False

        # Test case with min_score = 50
        test_case_low_threshold = TestCaseModel(
            name="test",
            input=TestInputModel(query="test"),
            expected=ExpectedBehavior(),
            thresholds=Thresholds(min_score=50.0),
        )

        passed = evaluator._compute_pass_fail(evaluations, test_case_low_threshold, score)
        assert passed is True

    @patch("evalview.core.llm_provider.select_provider")
    def test_compute_pass_fail_cost_threshold(self, mock_select_provider):
        """Test pass/fail based on cost threshold."""
        mock_select_provider.return_value = (LLMProvider.OPENAI, "fake-key")
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

        test_case = TestCaseModel(
            name="test",
            input=TestInputModel(query="test"),
            expected=ExpectedBehavior(),
            thresholds=Thresholds(min_score=50.0),
        )

        score = 100.0  # High score, but cost failed

        passed = evaluator._compute_pass_fail(evaluations_fail, test_case, score)
        assert passed is False  # Should fail due to cost

    @patch("evalview.core.llm_provider.select_provider")
    def test_compute_pass_fail_latency_threshold(self, mock_select_provider):
        """Test pass/fail based on latency threshold."""
        mock_select_provider.return_value = (LLMProvider.OPENAI, "fake-key")
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

        test_case = TestCaseModel(
            name="test",
            input=TestInputModel(query="test"),
            expected=ExpectedBehavior(),
            thresholds=Thresholds(min_score=50.0),
        )

        score = 100.0  # High score, but latency failed

        passed = evaluator._compute_pass_fail(evaluations_fail, test_case, score)
        assert passed is False  # Should fail due to latency

    @patch("evalview.core.llm_provider.select_provider")
    def test_compute_pass_fail_all_pass(self, mock_select_provider):
        """Test pass/fail when all criteria are met."""
        mock_select_provider.return_value = (LLMProvider.OPENAI, "fake-key")
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

        test_case = TestCaseModel(
            name="test",
            input=TestInputModel(query="test"),
            expected=ExpectedBehavior(),
            thresholds=Thresholds(min_score=50.0),
        )

        score = 100.0

        passed = evaluator._compute_pass_fail(evaluations, test_case, score)
        assert passed is True

    @patch("evalview.core.llm_provider.select_provider")
    def test_compute_pass_fail_multiple_failures(self, mock_select_provider):
        """Test pass/fail with multiple failures."""
        mock_select_provider.return_value = (LLMProvider.OPENAI, "fake-key")
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

        test_case = TestCaseModel(
            name="test",
            input=TestInputModel(query="test"),
            expected=ExpectedBehavior(),
            thresholds=Thresholds(min_score=50.0),
        )

        score = 10.0  # Low score

        passed = evaluator._compute_pass_fail(evaluations, test_case, score)
        assert passed is False  # Should fail on multiple criteria

    @pytest.mark.asyncio
    @patch("evalview.core.llm_provider.LLMClient.chat_completion")
    @patch("evalview.core.llm_provider.select_provider")
    async def test_evaluate_with_boundary_score(
        self, mock_select_provider, mock_chat_completion
    ):
        """Test evaluation with score exactly at threshold."""
        mock_select_provider.return_value = (LLMProvider.OPENAI, "fake-key")
        mock_chat_completion.return_value = {"score": 85, "rationale": "Good answer"}

        evaluator = Evaluator()

        test_case = TestCaseModel(
            name="test",
            input=TestInputModel(query="test"),
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
    @patch("evalview.core.llm_provider.LLMClient.chat_completion")
    @patch("evalview.core.llm_provider.select_provider")
    async def test_evaluate_score_rounding(
        self, mock_select_provider, mock_chat_completion
    ):
        """Test that score is properly rounded to 2 decimal places."""
        mock_select_provider.return_value = (LLMProvider.OPENAI, "fake-key")
        mock_chat_completion.return_value = {"score": 85, "rationale": "Good answer"}

        evaluator = Evaluator()

        test_case = TestCaseModel(
            name="test",
            input=TestInputModel(query="test"),
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
    @patch("evalview.core.llm_provider.LLMClient.chat_completion")
    @patch("evalview.core.llm_provider.select_provider")
    async def test_evaluate_with_no_thresholds(
        self, mock_select_provider, mock_chat_completion
    ):
        """Test evaluation when cost/latency thresholds are not specified."""
        mock_select_provider.return_value = (LLMProvider.OPENAI, "fake-key")
        mock_chat_completion.return_value = {"score": 85, "rationale": "Good answer"}

        evaluator = Evaluator()

        test_case = TestCaseModel(
            name="test",
            input=TestInputModel(query="test"),
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

    @patch("evalview.core.llm_provider.select_provider")
    def test_evaluator_initialization(self, mock_select_provider):
        """Test that evaluator initializes correctly."""
        mock_select_provider.return_value = (LLMProvider.OPENAI, "fake-key")
        evaluator = Evaluator()

        # Check that all evaluators are initialized
        assert evaluator.output_evaluator is not None
        assert evaluator.tool_evaluator is not None
        assert evaluator.sequence_evaluator is not None
        assert evaluator.cost_evaluator is not None
        assert evaluator.latency_evaluator is not None

    @patch("evalview.core.llm_provider.select_provider")
    def test_compute_overall_score_weights_sum_to_one(self, mock_select_provider):
        """Verify that evaluation weights sum to 1.0 (100%)."""
        mock_select_provider.return_value = (LLMProvider.OPENAI, "fake-key")
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

        test_case = TestCaseModel(
            name="test",
            input=TestInputModel(query="test"),
            expected=ExpectedBehavior(),
            thresholds=Thresholds(min_score=50.0),
        )

        score = evaluator._compute_overall_score(evaluations, test_case)

        # With all components at 100%, the overall score should be 100
        # This verifies that weights sum to 1.0
        assert score == 100.0

    @patch("evalview.core.llm_provider.select_provider")
    def test_compute_overall_score_only_output_quality(self, mock_select_provider):
        """Test score when only output quality is considered (others zero)."""
        mock_select_provider.return_value = (LLMProvider.OPENAI, "fake-key")
        evaluator = Evaluator()

        evaluations = Evaluations(
            tool_accuracy=ToolEvaluation(accuracy=0.0),  # 0%
            sequence_correctness=SequenceEvaluation(
                correct=False, expected_sequence=[], actual_sequence=[], progress_score=0.0
            ),  # 0% (explicit progress_score=0.0)
            output_quality=OutputEvaluation(
                score=100.0,  # 100%
                rationale="Perfect output",
                contains_checks=ContainsChecks(),
                not_contains_checks=ContainsChecks(),
            ),
            cost=CostEvaluation(total_cost=0.0, threshold=1.0, passed=True),
            latency=LatencyEvaluation(total_latency=0.0, threshold=1000.0, passed=True),
        )

        test_case = TestCaseModel(
            name="test",
            input=TestInputModel(query="test"),
            expected=ExpectedBehavior(),
            thresholds=Thresholds(min_score=50.0),
        )

        score = evaluator._compute_overall_score(evaluations, test_case)

        # Score should be 50% of 100 (output quality weight is 0.5)
        assert score == 50.0
