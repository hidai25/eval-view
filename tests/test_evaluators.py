"""Tests for all evaluator modules."""

import pytest
from datetime import datetime
from unittest.mock import patch

from evalview.evaluators.tool_call_evaluator import ToolCallEvaluator
from evalview.evaluators.sequence_evaluator import SequenceEvaluator
from evalview.evaluators.output_evaluator import OutputEvaluator
from evalview.evaluators.cost_evaluator import CostEvaluator
from evalview.evaluators.latency_evaluator import LatencyEvaluator
from evalview.core.types import (
    TestCase as TestCaseModel,
    TestInput as TestInputModel,
    ExpectedBehavior,
    ExpectedOutput,
    Thresholds,
    ExecutionTrace,
    StepTrace,
    StepMetrics,
    ExecutionMetrics,
)


# ============================================================================
# Tool Call Evaluator Tests
# ============================================================================


class TestToolCallEvaluator:
    """Tests for ToolCallEvaluator."""

    def test_perfect_accuracy(self, sample_test_case, sample_execution_trace):
        """Test 100% accuracy when all tools match."""
        evaluator = ToolCallEvaluator()
        result = evaluator.evaluate(sample_test_case, sample_execution_trace)

        assert result.accuracy == 1.0
        assert set(result.correct) == {"search", "summarize"}
        assert result.missing == []
        assert result.unexpected == []

    def test_missing_tools(self):
        """Test accuracy when some expected tools are missing."""
        test_case = TestCaseModel(
            name="test",
            input=TestInputModel(query="test"),
            expected=ExpectedBehavior(tools=["search", "summarize", "validate"]),
            thresholds=Thresholds(min_score=50.0),
        )

        trace = ExecutionTrace(
            session_id="test",
            start_time=datetime.now(),
            end_time=datetime.now(),
            steps=[
                StepTrace(
                    step_id="1",
                    step_name="Search",
                    tool_name="search",
                    parameters={},
                    output="result",
                    success=True,
                    metrics=StepMetrics(latency=100.0, cost=0.01),
                ),
                StepTrace(
                    step_id="2",
                    step_name="Summarize",
                    tool_name="summarize",
                    parameters={},
                    output="result",
                    success=True,
                    metrics=StepMetrics(latency=100.0, cost=0.01),
                ),
            ],
            final_output="test",
            metrics=ExecutionMetrics(total_cost=0.02, total_latency=200.0),
        )

        evaluator = ToolCallEvaluator()
        result = evaluator.evaluate(test_case, trace)

        assert result.accuracy == 2 / 3  # 2 correct out of 3 expected
        assert set(result.correct) == {"search", "summarize"}
        assert result.missing == ["validate"]
        assert result.unexpected == []

    def test_unexpected_tools(self):
        """Test that unexpected tools are detected."""
        test_case = TestCaseModel(
            name="test",
            input=TestInputModel(query="test"),
            expected=ExpectedBehavior(tools=["search"]),
            thresholds=Thresholds(min_score=50.0),
        )

        trace = ExecutionTrace(
            session_id="test",
            start_time=datetime.now(),
            end_time=datetime.now(),
            steps=[
                StepTrace(
                    step_id="1",
                    step_name="Search",
                    tool_name="search",
                    parameters={},
                    output="result",
                    success=True,
                    metrics=StepMetrics(latency=100.0, cost=0.01),
                ),
                StepTrace(
                    step_id="2",
                    step_name="Unexpected",
                    tool_name="unexpected_tool",
                    parameters={},
                    output="result",
                    success=True,
                    metrics=StepMetrics(latency=100.0, cost=0.01),
                ),
            ],
            final_output="test",
            metrics=ExecutionMetrics(total_cost=0.02, total_latency=200.0),
        )

        evaluator = ToolCallEvaluator()
        result = evaluator.evaluate(test_case, trace)

        assert result.accuracy == 1.0  # All expected tools were called
        assert result.correct == ["search"]
        assert result.missing == []
        assert result.unexpected == ["unexpected_tool"]

    def test_no_expected_tools(self):
        """Test accuracy when no tools are expected."""
        test_case = TestCaseModel(
            name="test",
            input=TestInputModel(query="test"),
            expected=ExpectedBehavior(tools=[]),
            thresholds=Thresholds(min_score=50.0),
        )

        trace = ExecutionTrace(
            session_id="test",
            start_time=datetime.now(),
            end_time=datetime.now(),
            steps=[],
            final_output="test",
            metrics=ExecutionMetrics(total_cost=0.0, total_latency=0.0),
        )

        evaluator = ToolCallEvaluator()
        result = evaluator.evaluate(test_case, trace)

        assert result.accuracy == 1.0  # Perfect score when no tools expected
        assert result.correct == []
        assert result.missing == []
        assert result.unexpected == []

    def test_no_expected_tools_but_tools_called(self):
        """Test when no tools expected but agent calls tools anyway."""
        test_case = TestCaseModel(
            name="test",
            input=TestInputModel(query="test"),
            expected=ExpectedBehavior(tools=[]),
            thresholds=Thresholds(min_score=50.0),
        )

        trace = ExecutionTrace(
            session_id="test",
            start_time=datetime.now(),
            end_time=datetime.now(),
            steps=[
                StepTrace(
                    step_id="1",
                    step_name="Unexpected",
                    tool_name="unexpected",
                    parameters={},
                    output="result",
                    success=True,
                    metrics=StepMetrics(latency=100.0, cost=0.01),
                )
            ],
            final_output="test",
            metrics=ExecutionMetrics(total_cost=0.01, total_latency=100.0),
        )

        evaluator = ToolCallEvaluator()
        result = evaluator.evaluate(test_case, trace)

        assert result.accuracy == 1.0  # All expected (zero) tools were called
        assert result.correct == []
        assert result.missing == []
        assert result.unexpected == ["unexpected"]

    def test_duplicate_tool_calls(self):
        """Test handling of duplicate tool calls."""
        test_case = TestCaseModel(
            name="test",
            input=TestInputModel(query="test"),
            expected=ExpectedBehavior(tools=["search"]),
            thresholds=Thresholds(min_score=50.0),
        )

        trace = ExecutionTrace(
            session_id="test",
            start_time=datetime.now(),
            end_time=datetime.now(),
            steps=[
                StepTrace(
                    step_id="1",
                    step_name="Search 1",
                    tool_name="search",
                    parameters={},
                    output="result",
                    success=True,
                    metrics=StepMetrics(latency=100.0, cost=0.01),
                ),
                StepTrace(
                    step_id="2",
                    step_name="Search 2",
                    tool_name="search",
                    parameters={},
                    output="result",
                    success=True,
                    metrics=StepMetrics(latency=100.0, cost=0.01),
                ),
            ],
            final_output="test",
            metrics=ExecutionMetrics(total_cost=0.02, total_latency=200.0),
        )

        evaluator = ToolCallEvaluator()
        result = evaluator.evaluate(test_case, trace)

        assert result.accuracy == 1.0  # Expected tool was called
        assert result.correct == ["search"]
        assert result.missing == []
        # Duplicates of expected tools are not considered unexpected
        # since the tool itself is in the expected set
        assert result.unexpected == []

    def test_none_expected_tools(self):
        """Test when expected.tools is None (not specified)."""
        test_case = TestCaseModel(
            name="test",
            input=TestInputModel(query="test"),
            expected=ExpectedBehavior(tools=None),
            thresholds=Thresholds(min_score=50.0),
        )

        trace = ExecutionTrace(
            session_id="test",
            start_time=datetime.now(),
            end_time=datetime.now(),
            steps=[
                StepTrace(
                    step_id="1",
                    step_name="Tool",
                    tool_name="any_tool",
                    parameters={},
                    output="result",
                    success=True,
                    metrics=StepMetrics(latency=100.0, cost=0.01),
                )
            ],
            final_output="test",
            metrics=ExecutionMetrics(total_cost=0.01, total_latency=100.0),
        )

        evaluator = ToolCallEvaluator()
        result = evaluator.evaluate(test_case, trace)

        # When tools is None, treated as empty list
        assert result.accuracy == 1.0
        assert result.correct == []
        assert result.missing == []
        assert result.unexpected == ["any_tool"]


# ============================================================================
# Sequence Evaluator Tests
# ============================================================================


class TestSequenceEvaluator:
    """Tests for SequenceEvaluator."""

    def test_correct_sequence(self, sample_test_case, sample_execution_trace):
        """Test evaluation with correct tool sequence."""
        evaluator = SequenceEvaluator()
        result = evaluator.evaluate(sample_test_case, sample_execution_trace)

        assert result.correct is True
        assert result.expected_sequence == ["search", "summarize"]
        assert result.actual_sequence == ["search", "summarize"]
        assert result.violations == []

    def test_incorrect_sequence_order(self):
        """Test evaluation with incorrect tool order."""
        test_case = TestCaseModel(
            name="test",
            input=TestInputModel(query="test"),
            expected=ExpectedBehavior(tool_sequence=["search", "summarize"]),
            thresholds=Thresholds(min_score=50.0),
        )

        trace = ExecutionTrace(
            session_id="test",
            start_time=datetime.now(),
            end_time=datetime.now(),
            steps=[
                StepTrace(
                    step_id="1",
                    step_name="Summarize",
                    tool_name="summarize",  # Wrong order
                    parameters={},
                    output="result",
                    success=True,
                    metrics=StepMetrics(latency=100.0, cost=0.01),
                ),
                StepTrace(
                    step_id="2",
                    step_name="Search",
                    tool_name="search",  # Wrong order
                    parameters={},
                    output="result",
                    success=True,
                    metrics=StepMetrics(latency=100.0, cost=0.01),
                ),
            ],
            final_output="test",
            metrics=ExecutionMetrics(total_cost=0.02, total_latency=200.0),
        )

        evaluator = SequenceEvaluator()
        result = evaluator.evaluate(test_case, trace)

        assert result.correct is False
        assert result.expected_sequence == ["search", "summarize"]
        assert result.actual_sequence == ["summarize", "search"]
        assert len(result.violations) == 2
        assert "Step 1" in result.violations[0]
        assert "Step 2" in result.violations[1]

    def test_length_mismatch(self):
        """Test evaluation when sequence lengths don't match."""
        test_case = TestCaseModel(
            name="test",
            input=TestInputModel(query="test"),
            expected=ExpectedBehavior(tool_sequence=["search", "summarize", "validate"]),
            thresholds=Thresholds(min_score=50.0),
        )

        trace = ExecutionTrace(
            session_id="test",
            start_time=datetime.now(),
            end_time=datetime.now(),
            steps=[
                StepTrace(
                    step_id="1",
                    step_name="Search",
                    tool_name="search",
                    parameters={},
                    output="result",
                    success=True,
                    metrics=StepMetrics(latency=100.0, cost=0.01),
                )
            ],
            final_output="test",
            metrics=ExecutionMetrics(total_cost=0.01, total_latency=100.0),
        )

        evaluator = SequenceEvaluator()
        result = evaluator.evaluate(test_case, trace)

        assert result.correct is False
        assert "Length mismatch" in result.violations[0]
        assert "expected 3 steps" in result.violations[0]
        assert "got 1" in result.violations[0]

    def test_no_expected_sequence(self):
        """Test when no sequence is expected (None)."""
        test_case = TestCaseModel(
            name="test",
            input=TestInputModel(query="test"),
            expected=ExpectedBehavior(tool_sequence=None),
            thresholds=Thresholds(min_score=50.0),
        )

        trace = ExecutionTrace(
            session_id="test",
            start_time=datetime.now(),
            end_time=datetime.now(),
            steps=[
                StepTrace(
                    step_id="1",
                    step_name="Any",
                    tool_name="any_tool",
                    parameters={},
                    output="result",
                    success=True,
                    metrics=StepMetrics(latency=100.0, cost=0.01),
                )
            ],
            final_output="test",
            metrics=ExecutionMetrics(total_cost=0.01, total_latency=100.0),
        )

        evaluator = SequenceEvaluator()
        result = evaluator.evaluate(test_case, trace)

        # When no sequence expected, should pass
        assert result.correct is True
        assert result.expected_sequence == []
        assert result.violations == []

    def test_empty_expected_sequence(self):
        """Test when expected sequence is empty list."""
        test_case = TestCaseModel(
            name="test",
            input=TestInputModel(query="test"),
            expected=ExpectedBehavior(tool_sequence=[]),
            thresholds=Thresholds(min_score=50.0),
        )

        trace = ExecutionTrace(
            session_id="test",
            start_time=datetime.now(),
            end_time=datetime.now(),
            steps=[],
            final_output="test",
            metrics=ExecutionMetrics(total_cost=0.0, total_latency=0.0),
        )

        evaluator = SequenceEvaluator()
        result = evaluator.evaluate(test_case, trace)

        assert result.correct is True
        assert result.expected_sequence == []
        assert result.actual_sequence == []


# ============================================================================
# Output Evaluator Tests
# ============================================================================


class TestOutputEvaluator:
    """Tests for OutputEvaluator."""

    @pytest.mark.asyncio
    @patch("evalview.evaluators.output_evaluator.AsyncOpenAI")
    async def test_contains_checks_all_pass(self, mock_openai_class, mock_openai_client):
        """Test contains checks when all strings are present."""
        test_case = TestCaseModel(
            name="test",
            input=TestInputModel(query="What is the capital of France?"),
            expected=ExpectedBehavior(
                output=ExpectedOutput(contains=["Paris", "France", "capital"])
            ),
            thresholds=Thresholds(min_score=50.0),
        )

        trace = ExecutionTrace(
            session_id="test",
            start_time=datetime.now(),
            end_time=datetime.now(),
            steps=[],
            final_output="The capital of France is Paris.",
            metrics=ExecutionMetrics(total_cost=0.0, total_latency=0.0),
        )

        evaluator = OutputEvaluator()
        evaluator.client = mock_openai_client

        result = await evaluator.evaluate(test_case, trace)

        assert len(result.contains_checks.passed) == 3
        assert result.contains_checks.failed == []
        assert result.score == 85  # From mock

    @pytest.mark.asyncio
    @patch("evalview.evaluators.output_evaluator.AsyncOpenAI")
    async def test_contains_checks_case_insensitive(self, mock_openai_class, mock_openai_client):
        """Test that contains checks are case-insensitive."""
        test_case = TestCaseModel(
            name="test",
            input=TestInputModel(query="test"),
            expected=ExpectedBehavior(output=ExpectedOutput(contains=["PARIS", "france"])),
            thresholds=Thresholds(min_score=50.0),
        )

        trace = ExecutionTrace(
            session_id="test",
            start_time=datetime.now(),
            end_time=datetime.now(),
            steps=[],
            final_output="The capital of France is Paris.",
            metrics=ExecutionMetrics(total_cost=0.0, total_latency=0.0),
        )

        evaluator = OutputEvaluator()
        evaluator.client = mock_openai_client

        result = await evaluator.evaluate(test_case, trace)

        assert "PARIS" in result.contains_checks.passed
        assert "france" in result.contains_checks.passed
        assert result.contains_checks.failed == []

    @pytest.mark.asyncio
    @patch("evalview.evaluators.output_evaluator.AsyncOpenAI")
    async def test_contains_checks_some_fail(self, mock_openai_class, mock_openai_client):
        """Test contains checks when some strings are missing."""
        test_case = TestCaseModel(
            name="test",
            input=TestInputModel(query="test"),
            expected=ExpectedBehavior(
                output=ExpectedOutput(contains=["Paris", "London", "Berlin"])
            ),
            thresholds=Thresholds(min_score=50.0),
        )

        trace = ExecutionTrace(
            session_id="test",
            start_time=datetime.now(),
            end_time=datetime.now(),
            steps=[],
            final_output="The capital of France is Paris.",
            metrics=ExecutionMetrics(total_cost=0.0, total_latency=0.0),
        )

        evaluator = OutputEvaluator()
        evaluator.client = mock_openai_client

        result = await evaluator.evaluate(test_case, trace)

        assert result.contains_checks.passed == ["Paris"]
        assert set(result.contains_checks.failed) == {"London", "Berlin"}

    @pytest.mark.asyncio
    @patch("evalview.evaluators.output_evaluator.AsyncOpenAI")
    async def test_not_contains_checks_all_pass(self, mock_openai_class, mock_openai_client):
        """Test not_contains checks when prohibited strings are absent."""
        test_case = TestCaseModel(
            name="test",
            input=TestInputModel(query="test"),
            expected=ExpectedBehavior(
                output=ExpectedOutput(not_contains=["error", "failed", "unknown"])
            ),
            thresholds=Thresholds(min_score=50.0),
        )

        trace = ExecutionTrace(
            session_id="test",
            start_time=datetime.now(),
            end_time=datetime.now(),
            steps=[],
            final_output="The capital of France is Paris.",
            metrics=ExecutionMetrics(total_cost=0.0, total_latency=0.0),
        )

        evaluator = OutputEvaluator()
        evaluator.client = mock_openai_client

        result = await evaluator.evaluate(test_case, trace)

        assert len(result.not_contains_checks.passed) == 3
        assert result.not_contains_checks.failed == []

    @pytest.mark.asyncio
    @patch("evalview.evaluators.output_evaluator.AsyncOpenAI")
    async def test_not_contains_checks_some_fail(self, mock_openai_class, mock_openai_client):
        """Test not_contains checks when some prohibited strings are present."""
        test_case = TestCaseModel(
            name="test",
            input=TestInputModel(query="test"),
            expected=ExpectedBehavior(output=ExpectedOutput(not_contains=["error", "Paris"])),
            thresholds=Thresholds(min_score=50.0),
        )

        trace = ExecutionTrace(
            session_id="test",
            start_time=datetime.now(),
            end_time=datetime.now(),
            steps=[],
            final_output="The capital of France is Paris.",
            metrics=ExecutionMetrics(total_cost=0.0, total_latency=0.0),
        )

        evaluator = OutputEvaluator()
        evaluator.client = mock_openai_client

        result = await evaluator.evaluate(test_case, trace)

        assert result.not_contains_checks.passed == ["error"]
        assert result.not_contains_checks.failed == ["Paris"]

    @pytest.mark.asyncio
    @patch("evalview.evaluators.output_evaluator.AsyncOpenAI")
    async def test_no_string_checks(self, mock_openai_class, mock_openai_client):
        """Test when no contains/not_contains are specified."""
        test_case = TestCaseModel(
            name="test",
            input=TestInputModel(query="test"),
            expected=ExpectedBehavior(output=ExpectedOutput()),
            thresholds=Thresholds(min_score=50.0),
        )

        trace = ExecutionTrace(
            session_id="test",
            start_time=datetime.now(),
            end_time=datetime.now(),
            steps=[],
            final_output="Test output",
            metrics=ExecutionMetrics(total_cost=0.0, total_latency=0.0),
        )

        evaluator = OutputEvaluator()
        evaluator.client = mock_openai_client

        result = await evaluator.evaluate(test_case, trace)

        assert result.contains_checks.passed == []
        assert result.contains_checks.failed == []
        assert result.not_contains_checks.passed == []
        assert result.not_contains_checks.failed == []

    @pytest.mark.asyncio
    @patch("evalview.evaluators.output_evaluator.AsyncOpenAI")
    async def test_llm_as_judge_integration(self, mock_openai_class, mock_openai_client):
        """Test LLM-as-judge integration."""
        test_case = TestCaseModel(
            name="test",
            input=TestInputModel(query="What is 2+2?"),
            expected=ExpectedBehavior(),
            thresholds=Thresholds(min_score=50.0),
        )

        trace = ExecutionTrace(
            session_id="test",
            start_time=datetime.now(),
            end_time=datetime.now(),
            steps=[],
            final_output="4",
            metrics=ExecutionMetrics(total_cost=0.0, total_latency=0.0),
        )

        evaluator = OutputEvaluator()
        evaluator.client = mock_openai_client

        result = await evaluator.evaluate(test_case, trace)

        assert result.score == 85
        assert result.rationale == "The output correctly answers the question."
        mock_openai_client.chat.completions.create.assert_called_once()

    @pytest.mark.asyncio
    @patch("evalview.evaluators.output_evaluator.AsyncOpenAI")
    async def test_empty_output(self, mock_openai_class, mock_openai_client):
        """Test evaluation with empty output."""
        test_case = TestCaseModel(
            name="test",
            input=TestInputModel(query="test"),
            expected=ExpectedBehavior(output=ExpectedOutput(contains=["something"])),
            thresholds=Thresholds(min_score=50.0),
        )

        trace = ExecutionTrace(
            session_id="test",
            start_time=datetime.now(),
            end_time=datetime.now(),
            steps=[],
            final_output="",
            metrics=ExecutionMetrics(total_cost=0.0, total_latency=0.0),
        )

        evaluator = OutputEvaluator()
        evaluator.client = mock_openai_client

        result = await evaluator.evaluate(test_case, trace)

        assert result.contains_checks.failed == ["something"]


# ============================================================================
# Cost Evaluator Tests
# ============================================================================


class TestCostEvaluator:
    """Tests for CostEvaluator."""

    def test_cost_within_threshold(self, sample_test_case, sample_execution_trace):
        """Test evaluation when cost is within threshold."""
        evaluator = CostEvaluator()
        result = evaluator.evaluate(sample_test_case, sample_execution_trace)

        assert result.passed is True
        assert result.total_cost == 0.03
        assert result.threshold == 0.50
        assert len(result.breakdown) == 2

    def test_cost_exceeds_threshold(self):
        """Test evaluation when cost exceeds threshold."""
        test_case = TestCaseModel(
            name="test",
            input=TestInputModel(query="test"),
            expected=ExpectedBehavior(),
            thresholds=Thresholds(min_score=50.0, max_cost=0.05),
        )

        trace = ExecutionTrace(
            session_id="test",
            start_time=datetime.now(),
            end_time=datetime.now(),
            steps=[
                StepTrace(
                    step_id="1",
                    step_name="Expensive",
                    tool_name="tool",
                    parameters={},
                    output="result",
                    success=True,
                    metrics=StepMetrics(latency=100.0, cost=0.10),
                )
            ],
            final_output="test",
            metrics=ExecutionMetrics(total_cost=0.10, total_latency=100.0),
        )

        evaluator = CostEvaluator()
        result = evaluator.evaluate(test_case, trace)

        assert result.passed is False
        assert result.total_cost == 0.10
        assert result.threshold == 0.05

    def test_no_cost_threshold(self):
        """Test when no cost threshold is specified (should always pass)."""
        test_case = TestCaseModel(
            name="test",
            input=TestInputModel(query="test"),
            expected=ExpectedBehavior(),
            thresholds=Thresholds(min_score=50.0, max_cost=None),
        )

        trace = ExecutionTrace(
            session_id="test",
            start_time=datetime.now(),
            end_time=datetime.now(),
            steps=[
                StepTrace(
                    step_id="1",
                    step_name="Step",
                    tool_name="tool",
                    parameters={},
                    output="result",
                    success=True,
                    metrics=StepMetrics(latency=100.0, cost=999.99),
                )
            ],
            final_output="test",
            metrics=ExecutionMetrics(total_cost=999.99, total_latency=100.0),
        )

        evaluator = CostEvaluator()
        result = evaluator.evaluate(test_case, trace)

        assert result.passed is True  # No threshold means always pass
        assert result.threshold == float("inf")

    def test_cost_breakdown(self, sample_execution_trace):
        """Test that cost breakdown includes all steps."""
        test_case = TestCaseModel(
            name="test",
            input=TestInputModel(query="test"),
            expected=ExpectedBehavior(),
            thresholds=Thresholds(min_score=50.0, max_cost=1.0),
        )

        evaluator = CostEvaluator()
        result = evaluator.evaluate(test_case, sample_execution_trace)

        assert len(result.breakdown) == 2
        assert result.breakdown[0].step_id == "step-1"
        assert result.breakdown[0].cost == 0.02
        assert result.breakdown[1].step_id == "step-2"
        assert result.breakdown[1].cost == 0.01

    def test_zero_cost(self):
        """Test evaluation with zero cost."""
        test_case = TestCaseModel(
            name="test",
            input=TestInputModel(query="test"),
            expected=ExpectedBehavior(),
            thresholds=Thresholds(min_score=50.0, max_cost=0.10),
        )

        trace = ExecutionTrace(
            session_id="test",
            start_time=datetime.now(),
            end_time=datetime.now(),
            steps=[],
            final_output="test",
            metrics=ExecutionMetrics(total_cost=0.0, total_latency=100.0, total_tokens=None),
        )

        evaluator = CostEvaluator()
        with patch("evalview.evaluators.cost_evaluator.logger") as mock_logger:
            result = evaluator.evaluate(test_case, trace)

            # Should log warning about zero cost
            mock_logger.warning.assert_called_once()
            assert "$0.00" in str(mock_logger.warning.call_args)

        assert result.passed is True  # Zero cost is within threshold
        assert result.total_cost == 0.0


# ============================================================================
# Latency Evaluator Tests
# ============================================================================


class TestLatencyEvaluator:
    """Tests for LatencyEvaluator."""

    def test_latency_within_threshold(self, sample_test_case, sample_execution_trace):
        """Test evaluation when latency is within threshold."""
        evaluator = LatencyEvaluator()
        result = evaluator.evaluate(sample_test_case, sample_execution_trace)

        assert result.passed is True
        assert result.total_latency == 3000.0
        assert result.threshold == 5000.0
        assert len(result.breakdown) == 2

    def test_latency_exceeds_threshold(self):
        """Test evaluation when latency exceeds threshold."""
        test_case = TestCaseModel(
            name="test",
            input=TestInputModel(query="test"),
            expected=ExpectedBehavior(),
            thresholds=Thresholds(min_score=50.0, max_latency=1000.0),
        )

        trace = ExecutionTrace(
            session_id="test",
            start_time=datetime.now(),
            end_time=datetime.now(),
            steps=[
                StepTrace(
                    step_id="1",
                    step_name="Slow",
                    tool_name="tool",
                    parameters={},
                    output="result",
                    success=True,
                    metrics=StepMetrics(latency=2000.0, cost=0.01),
                )
            ],
            final_output="test",
            metrics=ExecutionMetrics(total_cost=0.01, total_latency=2000.0),
        )

        evaluator = LatencyEvaluator()
        result = evaluator.evaluate(test_case, trace)

        assert result.passed is False
        assert result.total_latency == 2000.0
        assert result.threshold == 1000.0

    def test_no_latency_threshold(self):
        """Test when no latency threshold is specified (should always pass)."""
        test_case = TestCaseModel(
            name="test",
            input=TestInputModel(query="test"),
            expected=ExpectedBehavior(),
            thresholds=Thresholds(min_score=50.0, max_latency=None),
        )

        trace = ExecutionTrace(
            session_id="test",
            start_time=datetime.now(),
            end_time=datetime.now(),
            steps=[
                StepTrace(
                    step_id="1",
                    step_name="Step",
                    tool_name="tool",
                    parameters={},
                    output="result",
                    success=True,
                    metrics=StepMetrics(latency=999999.0, cost=0.01),
                )
            ],
            final_output="test",
            metrics=ExecutionMetrics(total_cost=0.01, total_latency=999999.0),
        )

        evaluator = LatencyEvaluator()
        result = evaluator.evaluate(test_case, trace)

        assert result.passed is True  # No threshold means always pass
        assert result.threshold == float("inf")

    def test_latency_breakdown(self, sample_execution_trace):
        """Test that latency breakdown includes all steps."""
        test_case = TestCaseModel(
            name="test",
            input=TestInputModel(query="test"),
            expected=ExpectedBehavior(),
            thresholds=Thresholds(min_score=50.0, max_latency=10000.0),
        )

        evaluator = LatencyEvaluator()
        result = evaluator.evaluate(test_case, sample_execution_trace)

        assert len(result.breakdown) == 2
        assert result.breakdown[0].step_id == "step-1"
        assert result.breakdown[0].latency == 1500.0
        assert result.breakdown[1].step_id == "step-2"
        assert result.breakdown[1].latency == 1000.0

    def test_exact_threshold_match(self):
        """Test when latency exactly matches threshold (should pass)."""
        test_case = TestCaseModel(
            name="test",
            input=TestInputModel(query="test"),
            expected=ExpectedBehavior(),
            thresholds=Thresholds(min_score=50.0, max_latency=1000.0),
        )

        trace = ExecutionTrace(
            session_id="test",
            start_time=datetime.now(),
            end_time=datetime.now(),
            steps=[],
            final_output="test",
            metrics=ExecutionMetrics(total_cost=0.0, total_latency=1000.0),
        )

        evaluator = LatencyEvaluator()
        result = evaluator.evaluate(test_case, trace)

        assert result.passed is True  # Exactly at threshold should pass
        assert result.total_latency == 1000.0
        assert result.threshold == 1000.0
