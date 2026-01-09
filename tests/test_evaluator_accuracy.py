"""Evaluator accuracy tests with ground truth.

These tests verify that EvalView's evaluators produce correct scores
for known-good and known-bad agent outputs. This is real dogfooding -
testing the core evaluation logic, not just the chat interface.
"""

import pytest
from datetime import datetime
from evalview.core.types import (
    TestCase,
    ExecutionTrace,
    ExecutionMetrics,
    StepTrace,
    StepMetrics,
    TestInput,
    ExpectedBehavior,
    ExpectedOutput,
    Thresholds,
)
from evalview.evaluators.tool_call_evaluator import ToolCallEvaluator
from evalview.evaluators.sequence_evaluator import SequenceEvaluator
from evalview.evaluators.output_evaluator import OutputEvaluator
from evalview.evaluators.hallucination_evaluator import HallucinationEvaluator


# =============================================================================
# Test Fixtures - Ground Truth Data
# =============================================================================


def make_test_case(
    query: str = "test query",
    expected_tools: list = None,
    expected_sequence: list = None,
    contains: list = None,
    not_contains: list = None,
) -> TestCase:
    """Create a test case with specified expectations."""
    return TestCase(
        name="accuracy_test",
        input=TestInput(query=query),
        expected=ExpectedBehavior(
            tools=expected_tools or [],
            tool_sequence=expected_sequence or [],
            output=ExpectedOutput(
                contains=contains or [],
                not_contains=not_contains or [],
            ),
        ),
        thresholds=Thresholds(min_score=70.0),
    )


def make_trace(
    tool_calls: list = None,
    final_output: str = "",
) -> ExecutionTrace:
    """Create an execution trace with specified tool calls and output."""
    steps = []
    for i, tool_name in enumerate(tool_calls or []):
        steps.append(
            StepTrace(
                step_id=f"step_{i}",
                step_name=f"Step {i+1}",
                tool_name=tool_name,
                parameters={},
                output="tool output",
                success=True,
                metrics=StepMetrics(latency=100.0, cost=0.001),
            )
        )
    now = datetime.now()
    return ExecutionTrace(
        session_id="test_session",
        start_time=now,
        end_time=now,
        steps=steps,
        final_output=final_output,
        metrics=ExecutionMetrics(total_latency=100.0, total_cost=0.001),
    )


# =============================================================================
# Tool Call Evaluator Accuracy Tests
# =============================================================================


class TestToolCallEvaluatorAccuracy:
    """Test that tool call evaluator correctly identifies tool usage."""

    def setup_method(self):
        self.evaluator = ToolCallEvaluator()

    def test_perfect_match_scores_100(self):
        """All expected tools called -> 100% accuracy."""
        test_case = make_test_case(expected_tools=["calculator", "search"])
        trace = make_trace(tool_calls=["calculator", "search"])

        result = self.evaluator.evaluate(test_case, trace)

        assert result.accuracy == 1.0
        assert set(result.correct) == {"calculator", "search"}
        assert result.missing == []

    def test_missing_tool_reduces_score(self):
        """Missing expected tool -> reduced accuracy."""
        test_case = make_test_case(expected_tools=["calculator", "search"])
        trace = make_trace(tool_calls=["calculator"])  # missing search

        result = self.evaluator.evaluate(test_case, trace)

        assert result.accuracy == 0.5  # 1 of 2 tools
        assert "search" in result.missing
        assert "calculator" in result.correct

    def test_no_tools_called_scores_zero(self):
        """No tools called when expected -> 0% accuracy."""
        test_case = make_test_case(expected_tools=["calculator", "search"])
        trace = make_trace(tool_calls=[])

        result = self.evaluator.evaluate(test_case, trace)

        assert result.accuracy == 0.0
        assert set(result.missing) == {"calculator", "search"}

    def test_extra_tools_dont_reduce_score(self):
        """Extra unexpected tools don't reduce accuracy."""
        test_case = make_test_case(expected_tools=["calculator"])
        trace = make_trace(tool_calls=["calculator", "search", "weather"])

        result = self.evaluator.evaluate(test_case, trace)

        assert result.accuracy == 1.0  # All expected tools called
        assert "search" in result.unexpected
        assert "weather" in result.unexpected

    def test_no_expectations_scores_100(self):
        """No expected tools -> 100% accuracy (vacuous truth)."""
        test_case = make_test_case(expected_tools=[])
        trace = make_trace(tool_calls=["anything"])

        result = self.evaluator.evaluate(test_case, trace)

        assert result.accuracy == 1.0

    def test_partial_match_calculates_correctly(self):
        """Partial match calculates accurate percentage."""
        test_case = make_test_case(expected_tools=["a", "b", "c", "d"])
        trace = make_trace(tool_calls=["a", "c"])  # 2 of 4

        result = self.evaluator.evaluate(test_case, trace)

        assert result.accuracy == 0.5
        assert set(result.correct) == {"a", "c"}
        assert set(result.missing) == {"b", "d"}


# =============================================================================
# Sequence Evaluator Accuracy Tests
# =============================================================================


class TestSequenceEvaluatorAccuracy:
    """Test that sequence evaluator correctly identifies tool ordering."""

    def setup_method(self):
        self.evaluator = SequenceEvaluator()

    def test_correct_sequence_passes(self):
        """Correct tool sequence -> correct=True."""
        test_case = make_test_case(expected_sequence=["search", "summarize"])
        trace = make_trace(tool_calls=["search", "summarize"])

        result = self.evaluator.evaluate(test_case, trace)

        assert result.correct is True
        assert result.violations == []

    def test_wrong_order_fails(self):
        """Wrong tool order -> correct=False."""
        test_case = make_test_case(expected_sequence=["search", "summarize"])
        trace = make_trace(tool_calls=["summarize", "search"])  # reversed

        result = self.evaluator.evaluate(test_case, trace)

        assert result.correct is False
        assert len(result.violations) > 0

    def test_missing_step_fails(self):
        """Missing step in sequence -> correct=False."""
        test_case = make_test_case(expected_sequence=["a", "b", "c"])
        trace = make_trace(tool_calls=["a", "c"])  # missing b

        result = self.evaluator.evaluate(test_case, trace)

        assert result.correct is False
        assert "Length mismatch" in result.violations[0]

    def test_extra_step_fails(self):
        """Extra step in sequence -> correct=False."""
        test_case = make_test_case(expected_sequence=["a", "b"])
        trace = make_trace(tool_calls=["a", "b", "c"])  # extra c

        result = self.evaluator.evaluate(test_case, trace)

        assert result.correct is False

    def test_no_expected_sequence_passes(self):
        """No expected sequence -> passes (no requirement)."""
        test_case = make_test_case(expected_sequence=[])
        trace = make_trace(tool_calls=["anything", "here"])

        result = self.evaluator.evaluate(test_case, trace)

        assert result.correct is True

    def test_identifies_specific_violation(self):
        """Identifies which step is wrong."""
        test_case = make_test_case(expected_sequence=["a", "b", "c"])
        trace = make_trace(tool_calls=["a", "X", "c"])  # b -> X

        result = self.evaluator.evaluate(test_case, trace)

        assert result.correct is False
        assert any("Step 2" in v and "'b'" in v and "'X'" in v for v in result.violations)


# =============================================================================
# Output Contains/NotContains Tests (Deterministic)
# =============================================================================


class TestOutputContainsAccuracy:
    """Test deterministic string matching in output evaluator."""

    def setup_method(self):
        # We'll test the internal methods directly to avoid LLM calls
        self.evaluator = OutputEvaluator.__new__(OutputEvaluator)

    def test_contains_all_present(self):
        """All required strings present -> all pass."""
        output = "The capital of France is Paris. The population is 2 million."

        result = self.evaluator._check_contains(output, ["Paris", "France", "capital"])

        assert set(result.passed) == {"Paris", "France", "capital"}
        assert result.failed == []

    def test_contains_some_missing(self):
        """Some required strings missing -> partial pass."""
        output = "The capital of France is Paris."

        result = self.evaluator._check_contains(output, ["Paris", "London", "Berlin"])

        assert result.passed == ["Paris"]
        assert set(result.failed) == {"London", "Berlin"}

    def test_contains_case_insensitive(self):
        """Contains check is case insensitive."""
        output = "PARIS is the capital"

        result = self.evaluator._check_contains(output, ["paris", "Paris", "PARIS"])

        assert len(result.passed) == 3
        assert result.failed == []

    def test_not_contains_all_absent(self):
        """All prohibited strings absent -> all pass."""
        output = "The weather in Tokyo is sunny."

        result = self.evaluator._check_not_contains(output, ["error", "failed", "exception"])

        assert set(result.passed) == {"error", "failed", "exception"}
        assert result.failed == []

    def test_not_contains_some_present(self):
        """Some prohibited strings present -> partial fail."""
        output = "An error occurred while processing."

        result = self.evaluator._check_not_contains(output, ["error", "success", "warning"])

        assert set(result.passed) == {"success", "warning"}
        assert result.failed == ["error"]

    def test_empty_requirements_pass(self):
        """Empty requirements -> pass."""
        output = "anything here"

        contains_result = self.evaluator._check_contains(output, [])
        not_contains_result = self.evaluator._check_not_contains(output, [])

        assert contains_result.passed == []
        assert contains_result.failed == []
        assert not_contains_result.passed == []
        assert not_contains_result.failed == []


# =============================================================================
# LLM-Based Evaluator Tests (Require API Key)
# =============================================================================


@pytest.mark.asyncio
@pytest.mark.skipif(
    not pytest.importorskip("os").environ.get("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY not set"
)
class TestOutputEvaluatorLLMAccuracy:
    """Test LLM-as-judge scoring with obvious good/bad responses.

    These tests use clear-cut cases where the correct score is unambiguous.
    """

    @pytest.fixture
    def evaluator(self):
        return OutputEvaluator()

    async def test_correct_answer_scores_high(self, evaluator):
        """Correct, complete answer should score >= 80."""
        test_case = make_test_case(
            query="What is the capital of France?",
            contains=["Paris"],
        )
        trace = make_trace(
            final_output="The capital of France is Paris. Paris has been the capital since the 10th century and is the largest city in France with a population of over 2 million in the city proper."
        )

        result = await evaluator.evaluate(test_case, trace)

        assert result.score >= 80, f"Expected >= 80, got {result.score}: {result.rationale}"

    async def test_wrong_answer_scores_low(self, evaluator):
        """Completely wrong answer should score <= 30."""
        test_case = make_test_case(
            query="What is the capital of France?",
            contains=["Paris"],
        )
        trace = make_trace(
            final_output="The capital of France is London. London is a beautiful city in France."
        )

        result = await evaluator.evaluate(test_case, trace)

        assert result.score <= 30, f"Expected <= 30, got {result.score}: {result.rationale}"

    async def test_empty_answer_scores_low(self, evaluator):
        """Empty or non-answer should score <= 20."""
        test_case = make_test_case(
            query="Explain quantum entanglement.",
            contains=["quantum"],
        )
        trace = make_trace(final_output="I don't know.")

        result = await evaluator.evaluate(test_case, trace)

        assert result.score <= 20, f"Expected <= 20, got {result.score}: {result.rationale}"

    async def test_partial_answer_scores_medium(self, evaluator):
        """Partial but correct answer should score 40-70."""
        test_case = make_test_case(
            query="List the first 5 prime numbers and explain why they are prime.",
            contains=["2", "3", "5"],
        )
        trace = make_trace(
            final_output="The first 5 prime numbers are 2, 3, 5, 7, 11."
            # Missing explanation
        )

        result = await evaluator.evaluate(test_case, trace)

        assert 40 <= result.score <= 80, f"Expected 40-80, got {result.score}: {result.rationale}"


@pytest.mark.asyncio
@pytest.mark.skipif(
    not pytest.importorskip("os").environ.get("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY not set"
)
class TestHallucinationEvaluatorAccuracy:
    """Test hallucination detection with obvious true/false cases."""

    @pytest.fixture
    def evaluator(self):
        return HallucinationEvaluator()

    async def test_factual_response_no_hallucination(self, evaluator):
        """Factually correct response based on tool output -> no hallucination."""
        test_case = make_test_case(query="What is 2 + 2?")

        # Create trace with tool that returns the answer
        now = datetime.now()
        trace = ExecutionTrace(
            session_id="test",
            start_time=now,
            end_time=now,
            steps=[
                StepTrace(
                    step_id="step_0",
                    step_name="Calculate",
                    tool_name="calculator",
                    parameters={"expression": "2 + 2"},
                    output="4",
                    success=True,
                    metrics=StepMetrics(latency=10.0, cost=0.0),
                )
            ],
            final_output="Based on the calculation, 2 + 2 equals 4.",
            metrics=ExecutionMetrics(total_latency=100.0, total_cost=0.001),
        )

        result = await evaluator.evaluate(test_case, trace)

        assert result.has_hallucination is False, f"False positive: {result.details}"

    async def test_fabricated_facts_is_hallucination(self, evaluator):
        """Response with fabricated facts not from tools -> hallucination detected."""
        test_case = make_test_case(query="What is the weather?")

        # No tool output, but response claims specific data
        now = datetime.now()
        trace = ExecutionTrace(
            session_id="test",
            start_time=now,
            end_time=now,
            steps=[],  # No tools called
            final_output="The weather in Paris is exactly 23.5°C with 45% humidity and winds from the northwest at 12 km/h. The forecast shows rain at 3:47 PM.",
            metrics=ExecutionMetrics(total_latency=100.0, total_cost=0.001),
        )

        result = await evaluator.evaluate(test_case, trace)

        # Should detect hallucination with some confidence
        assert result.has_hallucination is True or result.confidence > 0.5, \
            f"Missed hallucination: {result.details}"


# =============================================================================
# Integration: Full Evaluation Pipeline
# =============================================================================


class TestEvaluatorIntegration:
    """Test evaluators work together correctly."""

    def test_good_agent_passes_all_checks(self):
        """A good agent response passes tool and sequence checks."""
        test_case = make_test_case(
            query="Calculate 10 * 5 and format the result",
            expected_tools=["calculator", "formatter"],
            expected_sequence=["calculator", "formatter"],
            contains=["50"],
            not_contains=["error"],
        )
        trace = make_trace(
            tool_calls=["calculator", "formatter"],
            final_output="The result of 10 * 5 is 50.",
        )

        tool_eval = ToolCallEvaluator()
        seq_eval = SequenceEvaluator()
        output_eval = OutputEvaluator.__new__(OutputEvaluator)

        tool_result = tool_eval.evaluate(test_case, trace)
        seq_result = seq_eval.evaluate(test_case, trace)
        contains_result = output_eval._check_contains(trace.final_output, ["50"])
        not_contains_result = output_eval._check_not_contains(trace.final_output, ["error"])

        assert tool_result.accuracy == 1.0, "Tool accuracy should be 100%"
        assert seq_result.correct is True, "Sequence should be correct"
        assert contains_result.failed == [], "Should contain '50'"
        assert not_contains_result.failed == [], "Should not contain 'error'"

    def test_bad_agent_fails_checks(self):
        """A bad agent response fails multiple checks."""
        test_case = make_test_case(
            query="Calculate 10 * 5",
            expected_tools=["calculator"],
            expected_sequence=["calculator"],
            contains=["50"],
            not_contains=["error"],
        )
        trace = make_trace(
            tool_calls=["weather"],  # Wrong tool
            final_output="An error occurred. The result is 42.",  # Wrong answer + error
        )

        tool_eval = ToolCallEvaluator()
        seq_eval = SequenceEvaluator()
        output_eval = OutputEvaluator.__new__(OutputEvaluator)

        tool_result = tool_eval.evaluate(test_case, trace)
        seq_result = seq_eval.evaluate(test_case, trace)
        contains_result = output_eval._check_contains(trace.final_output, ["50"])
        not_contains_result = output_eval._check_not_contains(trace.final_output, ["error"])

        assert tool_result.accuracy == 0.0, "Tool accuracy should be 0%"
        assert seq_result.correct is False, "Sequence should be wrong"
        assert "50" in contains_result.failed, "Should fail '50' check"
        assert "error" in not_contains_result.failed, "Should fail 'error' check"
