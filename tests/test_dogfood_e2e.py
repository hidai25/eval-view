"""End-to-end dogfood tests for EvalView.

These tests verify that EvalView correctly evaluates agent behavior by:
1. Running a deterministic mock agent with known outputs
2. Using EvalView to evaluate the agent
3. Verifying EvalView produces correct scores

This is true dogfooding - we're testing that EvalView's evaluation logic
correctly identifies good vs bad agent behavior.
"""

import os
import signal
import subprocess
import sys
import time
from typing import Generator

import pytest

from evalview.adapters.http_adapter import HTTPAdapter
from evalview.core.types import (
    TestCase,
    TestInput,
    ExpectedBehavior,
    ExpectedOutput,
    Thresholds,
)
from evalview.evaluators.evaluator import Evaluator
from evalview.evaluators.tool_call_evaluator import ToolCallEvaluator
from evalview.evaluators.sequence_evaluator import SequenceEvaluator


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture(scope="module")
def mock_agent() -> Generator[subprocess.Popen, None, None]:
    """Start the mock agent server for testing."""
    # Start mock agent
    agent_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "dogfood", "mock_agent.py"
    )

    proc = subprocess.Popen(
        [sys.executable, agent_path],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    # Wait for agent to be ready
    import httpx

    for _ in range(30):
        try:
            response = httpx.get("http://localhost:8002/health", timeout=1.0)
            if response.status_code == 200:
                break
        except Exception:
            pass
        time.sleep(0.5)
    else:
        proc.kill()
        raise RuntimeError("Mock agent failed to start")

    yield proc

    # Cleanup
    proc.send_signal(signal.SIGTERM)
    proc.wait(timeout=5)


@pytest.fixture
def adapter() -> HTTPAdapter:
    """Create HTTP adapter for mock agent."""
    return HTTPAdapter(
        endpoint="http://localhost:8002/execute",
        allow_private_urls=True,  # Allow localhost for testing
    )


@pytest.fixture
def evaluator() -> Evaluator:
    """Create evaluator in deterministic mode (no LLM judge needed)."""
    return Evaluator(skip_llm_judge=True)


# =============================================================================
# Test Cases - Good Agent Behavior (Should Score HIGH)
# =============================================================================


class TestGoodAgentBehavior:
    """Test that EvalView correctly scores good agent behavior HIGH."""

    @pytest.mark.asyncio
    async def test_correct_calculation_scores_high(self, mock_agent, adapter, evaluator):
        """Agent uses correct tool and returns correct answer -> high score."""
        test_case = TestCase(
            name="correct_calculation",
            input=TestInput(query="Calculate 15 * 7"),
            expected=ExpectedBehavior(
                tools=["calculator"],
                output=ExpectedOutput(contains=["105"]),
            ),
            thresholds=Thresholds(min_score=70.0),
        )

        trace = await adapter.execute(test_case.input.query)
        result = await evaluator.evaluate(test_case, trace)

        # Verify EvalView correctly identifies this as good behavior
        assert result.passed, f"Good agent should pass, got score {result.score}"
        assert result.score >= 70, f"Expected >= 70, got {result.score}"
        assert result.evaluations.tool_accuracy.accuracy == 1.0, "Should have 100% tool accuracy"

    @pytest.mark.asyncio
    async def test_correct_search_scores_high(self, mock_agent, adapter, evaluator):
        """Agent uses search tool correctly -> high score."""
        test_case = TestCase(
            name="correct_search",
            input=TestInput(query="Search for Python tutorials"),
            expected=ExpectedBehavior(
                tools=["search"],
                output=ExpectedOutput(contains=["Python", "results"]),
            ),
            thresholds=Thresholds(min_score=70.0),
        )

        trace = await adapter.execute(test_case.input.query)
        result = await evaluator.evaluate(test_case, trace)

        assert result.passed, f"Good agent should pass, got score {result.score}"
        assert result.evaluations.tool_accuracy.accuracy == 1.0

    @pytest.mark.asyncio
    async def test_correct_sequence_scores_high(self, mock_agent, adapter, evaluator):
        """Agent executes tools in correct sequence -> high score."""
        test_case = TestCase(
            name="correct_sequence",
            input=TestInput(query="Search for EvalView and summarize the results"),
            expected=ExpectedBehavior(
                tools=["search", "summarize"],
                tool_sequence=["search", "summarize"],
                output=ExpectedOutput(contains=["EvalView"]),
            ),
            thresholds=Thresholds(min_score=70.0),
        )

        trace = await adapter.execute(test_case.input.query)
        result = await evaluator.evaluate(test_case, trace)

        assert result.passed, f"Good agent should pass, got score {result.score}"
        assert result.evaluations.sequence_correctness.correct is True


# =============================================================================
# Test Cases - Bad Agent Behavior (Should Score LOW or FAIL)
# =============================================================================


class TestBadAgentBehavior:
    """Test that EvalView correctly identifies bad agent behavior."""

    @pytest.mark.asyncio
    async def test_wrong_answer_detected(self, mock_agent, adapter, evaluator):
        """Agent returns wrong answer -> EvalView should detect and score low."""
        test_case = TestCase(
            name="wrong_answer",
            input=TestInput(query="Calculate 10 + 5 wrong"),  # Triggers wrong answer
            expected=ExpectedBehavior(
                tools=["calculator"],
                output=ExpectedOutput(
                    contains=["15"],  # Correct answer
                    not_contains=["1014"],  # Wrong answer agent gives
                ),
            ),
            thresholds=Thresholds(min_score=80.0),
        )

        trace = await adapter.execute(test_case.input.query)
        result = await evaluator.evaluate(test_case, trace)

        # Agent returns "1014" instead of "15", should fail contains check
        assert (
            not result.passed or result.evaluations.output_quality.contains_checks.failed
        ), "EvalView should detect wrong answer"

    @pytest.mark.asyncio
    async def test_wrong_tool_detected(self, mock_agent, adapter, evaluator):
        """Agent uses wrong tool -> EvalView should detect and reduce score."""
        test_case = TestCase(
            name="wrong_tool",
            input=TestInput(query="Calculate 8 * 4 wrong tool"),  # Triggers wrong tool
            expected=ExpectedBehavior(
                tools=["calculator"],  # We expect calculator
                output=ExpectedOutput(contains=["32"]),
            ),
            thresholds=Thresholds(min_score=80.0),
        )

        trace = await adapter.execute(test_case.input.query)
        result = await evaluator.evaluate(test_case, trace)

        # Agent uses "weather" instead of "calculator"
        assert (
            result.evaluations.tool_accuracy.accuracy == 0.0
        ), "EvalView should detect wrong tool (expected calculator, got weather)"
        assert "calculator" in result.evaluations.tool_accuracy.missing

    @pytest.mark.asyncio
    async def test_missing_tool_detected(self, mock_agent, adapter, evaluator):
        """Agent doesn't use required tool -> EvalView should detect."""
        test_case = TestCase(
            name="missing_tool",
            input=TestInput(query="Answer with no tools please"),  # Triggers no tools
            expected=ExpectedBehavior(
                tools=["calculator"],  # We require calculator
            ),
            thresholds=Thresholds(min_score=80.0),
        )

        trace = await adapter.execute(test_case.input.query)
        result = await evaluator.evaluate(test_case, trace)

        # Agent doesn't call any tools
        assert (
            result.evaluations.tool_accuracy.accuracy == 0.0
        ), "EvalView should detect missing required tool"
        assert "calculator" in result.evaluations.tool_accuracy.missing


# =============================================================================
# Regression Tests - Verify Specific Evaluation Logic
# =============================================================================


class TestEvaluationLogicRegression:
    """Regression tests to verify evaluation logic doesn't break."""

    @pytest.mark.asyncio
    async def test_tool_accuracy_calculation(self, mock_agent, adapter):
        """Verify tool accuracy is calculated correctly."""
        tool_eval = ToolCallEvaluator()

        # Test case expecting 2 tools
        test_case = TestCase(
            name="multi_tool",
            input=TestInput(query="Search for EvalView and summarize"),
            expected=ExpectedBehavior(tools=["search", "summarize"]),
            thresholds=Thresholds(min_score=70.0),
        )

        trace = await adapter.execute(test_case.input.query)
        result = tool_eval.evaluate(test_case, trace)

        # Should have both tools
        assert result.accuracy == 1.0, f"Expected 100% accuracy, got {result.accuracy}"
        assert set(result.correct) == {"search", "summarize"}

    @pytest.mark.asyncio
    async def test_sequence_validation(self, mock_agent, adapter):
        """Verify sequence validation works correctly."""
        seq_eval = SequenceEvaluator()

        test_case = TestCase(
            name="sequence_test",
            input=TestInput(query="Search for EvalView and summarize"),
            expected=ExpectedBehavior(tool_sequence=["search", "summarize"]),
            thresholds=Thresholds(min_score=70.0),
        )

        trace = await adapter.execute(test_case.input.query)
        result = seq_eval.evaluate(test_case, trace)

        assert result.correct is True, f"Sequence should be correct: {result.violations}"

    @pytest.mark.asyncio
    async def test_output_contains_check(self, mock_agent, adapter, evaluator):
        """Verify output contains check works correctly."""
        test_case = TestCase(
            name="contains_test",
            input=TestInput(query="Calculate 12 + 8"),
            expected=ExpectedBehavior(
                tools=["calculator"],
                output=ExpectedOutput(contains=["20"]),
            ),
            thresholds=Thresholds(min_score=70.0),
        )

        trace = await adapter.execute(test_case.input.query)
        result = await evaluator.evaluate(test_case, trace)

        # Output should contain "20"
        assert "20" in trace.final_output, f"Output missing '20': {trace.final_output}"
        assert (
            result.evaluations.output_quality.contains_checks.passed
        ), f"Contains check should pass: {result.evaluations.output_quality.contains_checks}"


# =============================================================================
# Meta Tests - EvalView Evaluating Itself
# =============================================================================


class TestMetaEvaluation:
    """Meta tests that verify EvalView's evaluation matches expectations."""

    @pytest.mark.asyncio
    async def test_evalview_distinguishes_good_from_bad(self, mock_agent, adapter, evaluator):
        """EvalView should score good behavior higher than bad behavior."""
        # Good case
        good_case = TestCase(
            name="good",
            input=TestInput(query="Calculate 5 * 5"),
            expected=ExpectedBehavior(
                tools=["calculator"],
                output=ExpectedOutput(contains=["25"]),
            ),
            thresholds=Thresholds(min_score=70.0),
        )

        # Bad case - wrong tool (uses weather instead of calculator)
        bad_case = TestCase(
            name="bad",
            input=TestInput(query="Calculate 5 * 5 wrong tool"),
            expected=ExpectedBehavior(
                tools=["calculator"],  # We expect calculator, agent uses weather
                output=ExpectedOutput(contains=["25"]),
            ),
            thresholds=Thresholds(min_score=80.0),  # Higher threshold so bad case fails
        )

        good_trace = await adapter.execute(good_case.input.query)
        bad_trace = await adapter.execute(bad_case.input.query)

        good_result = await evaluator.evaluate(good_case, good_trace)
        bad_result = await evaluator.evaluate(bad_case, bad_trace)

        # Good should score higher than bad
        assert (
            good_result.score > bad_result.score
        ), f"Good ({good_result.score}) should score higher than bad ({bad_result.score})"

        # Good should pass
        assert good_result.passed, "Good behavior should pass"

        # Bad should have lower tool accuracy (wrong tool used)
        assert (
            bad_result.evaluations.tool_accuracy.accuracy < 1.0
        ), f"Bad agent used wrong tool, should have reduced accuracy: {bad_result.evaluations.tool_accuracy}"

    @pytest.mark.asyncio
    async def test_evalview_catches_missing_tool(self, mock_agent, adapter, evaluator):
        """EvalView should catch missing required tool."""
        test_case = TestCase(
            name="missing_tool",
            input=TestInput(query="Answer with no tools"),
            expected=ExpectedBehavior(tools=["calculator"]),
            thresholds=Thresholds(min_score=80.0),
        )

        trace = await adapter.execute(test_case.input.query)
        result = await evaluator.evaluate(test_case, trace)

        # Missing tool should have 0% tool accuracy
        assert (
            result.evaluations.tool_accuracy.accuracy == 0.0
        ), f"Missing tool should have 0% accuracy: {result.evaluations.tool_accuracy}"
        assert "calculator" in result.evaluations.tool_accuracy.missing

    @pytest.mark.asyncio
    async def test_evalview_catches_wrong_tool(self, mock_agent, adapter, evaluator):
        """EvalView should catch wrong tool usage."""
        test_case = TestCase(
            name="wrong_tool",
            input=TestInput(query="Calculate 3 * 3 wrong tool"),
            expected=ExpectedBehavior(tools=["calculator"]),
            thresholds=Thresholds(min_score=80.0),
        )

        trace = await adapter.execute(test_case.input.query)
        result = await evaluator.evaluate(test_case, trace)

        # Wrong tool (weather instead of calculator) should have 0% tool accuracy
        assert (
            result.evaluations.tool_accuracy.accuracy == 0.0
        ), f"Wrong tool should have 0% accuracy: {result.evaluations.tool_accuracy}"
        assert "calculator" in result.evaluations.tool_accuracy.missing
        assert "weather" in result.evaluations.tool_accuracy.unexpected
