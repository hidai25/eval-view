"""Tests for per-turn multi-turn evaluation."""

from __future__ import annotations

import asyncio
import tempfile
import shutil
from datetime import datetime
from typing import List, Optional

import pytest

from evalview.core.types import (
    ConversationTurn,
    EvaluationResult,
    ExecutionMetrics,
    ExecutionTrace,
    ExpectedBehavior,
    ExpectedOutput,
    StepMetrics,
    StepTrace,
    TestCase,
    TestInput,
    Thresholds,
    TurnEvaluation,
    TurnTrace,
)
from evalview.evaluators.evaluator import Evaluator


def _make_test_case(
    turns: List[ConversationTurn],
    name: str = "multi-turn-test",
    expected: Optional[ExpectedBehavior] = None,
) -> TestCase:
    """Build a multi-turn TestCase from turn definitions."""
    return TestCase(
        name=name,
        input=TestInput(query=turns[0].query),
        turns=turns,
        expected=expected or ExpectedBehavior(),
        thresholds=Thresholds(min_score=0),
    )


def _make_trace(
    turn_steps: List[List[str]],
    turn_outputs: Optional[List[str]] = None,
) -> ExecutionTrace:
    """Build an ExecutionTrace with per-turn steps and outputs."""
    steps: List[StepTrace] = []
    turns: List[TurnTrace] = []

    for turn_idx, tools in enumerate(turn_steps, start=1):
        for i, tool in enumerate(tools):
            steps.append(StepTrace(
                step_id=f"s{turn_idx}_{i}",
                step_name=tool,
                tool_name=tool,
                parameters={},
                output="ok",
                success=True,
                metrics=StepMetrics(),
                turn_index=turn_idx,
            ))
        output = (turn_outputs[turn_idx - 1] if turn_outputs and turn_idx <= len(turn_outputs) else "")
        turns.append(TurnTrace(
            index=turn_idx,
            query=f"query {turn_idx}",
            output=output,
            tools=tools,
        ))

    return ExecutionTrace(
        session_id="test",
        start_time=datetime.now(),
        end_time=datetime.now(),
        steps=steps,
        final_output=turn_outputs[-1] if turn_outputs else "done",
        metrics=ExecutionMetrics(total_cost=0, total_latency=0),
        turns=turns,
    )


class TestPerTurnEvaluation:
    """Tests for the Evaluator._evaluate_per_turn method."""

    def _run(self, test_case: TestCase, trace: ExecutionTrace) -> EvaluationResult:
        evaluator = Evaluator(skip_llm_judge=True)
        return asyncio.run(evaluator.evaluate(test_case, trace))

    def test_tools_match_passes(self):
        """Turn with expected tools that match actual tools should pass."""
        tc = _make_test_case(turns=[
            ConversationTurn(query="q1", expected=ExpectedBehavior(tools=["search"])),
            ConversationTurn(query="q2", expected=ExpectedBehavior(tools=["book"])),
        ])
        trace = _make_trace([["search"], ["book"]], ["found it", "booked"])
        result = self._run(tc, trace)

        assert result.turn_evaluations is not None
        assert len(result.turn_evaluations) == 2
        assert all(te.passed for te in result.turn_evaluations)
        assert result.turn_evaluations[0].tool_accuracy == 1.0
        assert result.turn_evaluations[1].tool_accuracy == 1.0

    def test_tools_mismatch_fails(self):
        """Turn with expected tools that don't match should fail."""
        tc = _make_test_case(turns=[
            ConversationTurn(query="q1", expected=ExpectedBehavior(tools=["search"])),
            ConversationTurn(query="q2", expected=ExpectedBehavior(tools=["book", "confirm"])),
        ])
        trace = _make_trace([["search"], ["book"]], ["found it", "booked"])
        result = self._run(tc, trace)

        assert result.turn_evaluations is not None
        assert result.turn_evaluations[0].passed is True
        assert result.turn_evaluations[1].passed is False
        assert result.turn_evaluations[1].tool_accuracy == 0.5
        assert result.passed is False  # Hard fail propagated

    def test_forbidden_tools_violation(self):
        """Forbidden tool in a turn should hard-fail the test."""
        tc = _make_test_case(turns=[
            ConversationTurn(query="q1"),  # No expected → not evaluated
            ConversationTurn(
                query="q2",
                expected=ExpectedBehavior(forbidden_tools=["delete_booking"]),
            ),
        ])
        trace = _make_trace([["search"], ["delete_booking"]], ["found", "deleted"])
        result = self._run(tc, trace)

        assert result.turn_evaluations is not None
        # Turn 1 has no expected, so only turn 2 is evaluated
        assert len(result.turn_evaluations) == 1
        assert result.turn_evaluations[0].passed is False
        assert "delete_booking" in result.turn_evaluations[0].forbidden_violations
        assert result.passed is False

    def test_contains_check_pass(self):
        """Turn output with contains check that matches should pass."""
        tc = _make_test_case(turns=[
            ConversationTurn(query="q1"),
            ConversationTurn(
                query="q2",
                expected=ExpectedBehavior(
                    output=ExpectedOutput(contains=["confirmation", "Paris"]),
                ),
            ),
        ])
        trace = _make_trace(
            [["search"], ["book"]],
            ["results", "Your confirmation for Paris is ready"],
        )
        result = self._run(tc, trace)

        assert result.turn_evaluations is not None
        assert len(result.turn_evaluations) == 1  # Only turn 2 has expected
        te = result.turn_evaluations[0]
        assert te.passed is True
        assert "confirmation" in te.contains_passed
        assert "Paris" in te.contains_passed

    def test_contains_check_fail(self):
        """Turn output missing contains pattern should fail."""
        tc = _make_test_case(turns=[
            ConversationTurn(query="q1"),
            ConversationTurn(
                query="q2",
                expected=ExpectedBehavior(
                    output=ExpectedOutput(contains=["confirmation"]),
                ),
            ),
        ])
        trace = _make_trace(
            [["search"], ["book"]],
            ["results", "Booking failed, no availability"],
        )
        result = self._run(tc, trace)

        assert result.turn_evaluations is not None
        te = result.turn_evaluations[0]
        assert te.passed is False
        assert "confirmation" in te.contains_failed
        assert result.passed is False

    def test_not_contains_check(self):
        """Turn output with not_contains violation should fail."""
        tc = _make_test_case(turns=[
            ConversationTurn(query="q1"),
            ConversationTurn(
                query="q2",
                expected=ExpectedBehavior(
                    output=ExpectedOutput(not_contains=["error", "failed"]),
                ),
            ),
        ])
        trace = _make_trace(
            [["search"], ["book"]],
            ["results", "Booking failed due to error"],
        )
        result = self._run(tc, trace)

        assert result.turn_evaluations is not None
        te = result.turn_evaluations[0]
        assert te.passed is False
        assert "error" in te.not_contains_failed
        assert "failed" in te.not_contains_failed

    def test_mixed_some_pass_some_fail(self):
        """Mixed results: some turns pass, some fail."""
        tc = _make_test_case(turns=[
            ConversationTurn(query="q1", expected=ExpectedBehavior(tools=["search"])),
            ConversationTurn(
                query="q2",
                expected=ExpectedBehavior(
                    tools=["book"],
                    output=ExpectedOutput(contains=["confirmation"]),
                ),
            ),
            ConversationTurn(
                query="q3",
                expected=ExpectedBehavior(tools=["status"]),
            ),
        ])
        trace = _make_trace(
            [["search"], ["book"], ["cancel"]],
            ["found", "Booking done", "cancelled"],
        )
        result = self._run(tc, trace)

        assert result.turn_evaluations is not None
        assert len(result.turn_evaluations) == 3
        assert result.turn_evaluations[0].passed is True   # tools match
        assert result.turn_evaluations[1].passed is False   # missing "confirmation"
        assert result.turn_evaluations[2].passed is False   # tools mismatch
        assert result.passed is False

    def test_single_turn_unaffected(self):
        """Single-turn tests should not have turn evaluations."""
        tc = TestCase(
            name="single-turn",
            input=TestInput(query="test query"),
            expected=ExpectedBehavior(tools=["search"]),
            thresholds=Thresholds(min_score=0),
        )
        trace = _make_trace([["search"]], ["result"])
        # Remove turns to simulate single-turn
        trace.turns = None

        result = self._run(tc, trace)
        assert result.turn_evaluations is None

    def test_no_expected_on_any_turn(self):
        """Multi-turn test with no expected on any turn should return None."""
        tc = _make_test_case(turns=[
            ConversationTurn(query="q1"),
            ConversationTurn(query="q2"),
        ])
        trace = _make_trace([["search"], ["book"]], ["found", "booked"])
        result = self._run(tc, trace)
        assert result.turn_evaluations is None

    def test_evaluation_attached_to_turn_trace(self):
        """TurnEvaluation should be attached to the corresponding TurnTrace."""
        tc = _make_test_case(turns=[
            ConversationTurn(query="q1", expected=ExpectedBehavior(tools=["search"])),
            ConversationTurn(query="q2", expected=ExpectedBehavior(tools=["book"])),
        ])
        trace = _make_trace([["search"], ["book"]], ["found", "booked"])
        result = self._run(tc, trace)

        assert result.trace.turns is not None
        assert result.trace.turns[0].evaluation is not None
        assert result.trace.turns[0].evaluation.passed is True
        assert result.trace.turns[1].evaluation is not None
        assert result.trace.turns[1].evaluation.passed is True


class TestGoldenPerTurnOutputs:
    """Tests for golden baseline per_turn_outputs field."""

    def test_save_golden_includes_per_turn_outputs(self):
        """Saving golden should capture per-turn outputs."""
        from evalview.core.golden import GoldenStore

        tmp_dir = tempfile.mkdtemp()
        try:
            store = GoldenStore(base_path=None)
            store.golden_dir = __import__("pathlib").Path(tmp_dir) / "golden"

            trace = _make_trace(
                [["search"], ["book"]],
                ["Found flights to Paris", "Booking confirmed"],
            )
            result = EvaluationResult(
                test_case="golden-test",
                passed=True,
                score=90.0,
                evaluations=__import__("evalview.core.types", fromlist=["Evaluations"]).Evaluations(
                    tool_accuracy=__import__("evalview.core.types", fromlist=["ToolEvaluation"]).ToolEvaluation(accuracy=1.0),
                    sequence_correctness=__import__("evalview.core.types", fromlist=["SequenceEvaluation"]).SequenceEvaluation(
                        correct=True, expected_sequence=["search", "book"], actual_sequence=["search", "book"],
                    ),
                    output_quality=__import__("evalview.core.types", fromlist=["OutputEvaluation"]).OutputEvaluation(
                        score=90.0, rationale="good",
                        contains_checks=__import__("evalview.core.types", fromlist=["ContainsChecks"]).ContainsChecks(),
                        not_contains_checks=__import__("evalview.core.types", fromlist=["ContainsChecks"]).ContainsChecks(),
                    ),
                    cost=__import__("evalview.core.types", fromlist=["CostEvaluation"]).CostEvaluation(
                        total_cost=0, threshold=1.0, passed=True,
                    ),
                    latency=__import__("evalview.core.types", fromlist=["LatencyEvaluation"]).LatencyEvaluation(
                        total_latency=0, threshold=10000, passed=True,
                    ),
                ),
                trace=trace,
                timestamp=datetime.now(),
            )

            store.save_golden(result)
            golden = store.load_golden("golden-test")

            assert golden is not None
            assert golden.per_turn_outputs is not None
            assert golden.per_turn_outputs == ["Found flights to Paris", "Booking confirmed"]
        finally:
            shutil.rmtree(tmp_dir)

    def test_golden_backward_compatibility_no_per_turn_outputs(self):
        """Old golden files without per_turn_outputs should load fine."""
        from evalview.core.golden import GoldenTrace, GoldenMetadata

        golden = GoldenTrace(
            metadata=GoldenMetadata(
                test_name="old-test",
                blessed_at=datetime.now(),
                score=90.0,
            ),
            trace=_make_trace([["search"]], ["result"]),
            tool_sequence=["search"],
            output_hash="abc123",
        )
        # per_turn_outputs should default to None
        assert golden.per_turn_outputs is None


class TestDiffPerTurnOutputs:
    """Tests for per-turn output comparison in diff engine."""

    def test_turn_output_similarity_computed(self):
        """Diff engine should compute per-turn output similarity."""
        from evalview.core.diff import DiffEngine
        from evalview.core.golden import GoldenTrace, GoldenMetadata

        golden = GoldenTrace(
            metadata=GoldenMetadata(
                test_name="test",
                blessed_at=datetime.now(),
                score=90.0,
            ),
            trace=_make_trace([["search"], ["book"]], ["Found flights", "Confirmed"]),
            tool_sequence=["search", "book"],
            output_hash="abc",
            per_turn_tool_sequences=[["search"], ["book"]],
            per_turn_outputs=["Found flights", "Confirmed"],
        )

        actual = _make_trace(
            [["search"], ["book"]],
            ["Found flights", "Something completely different"],
        )

        engine = DiffEngine()
        diff = engine.compare(golden, actual, 90.0)

        assert diff.turn_diffs is not None
        assert len(diff.turn_diffs) == 2

        # Turn 1: same tools, same output
        assert diff.turn_diffs[0].output_similarity is not None
        assert diff.turn_diffs[0].output_similarity > 0.9

        # Turn 2: same tools, different output
        assert diff.turn_diffs[1].output_similarity is not None
        assert diff.turn_diffs[1].output_similarity < 0.5

    def test_no_per_turn_outputs_backward_compat(self):
        """Diff with old golden (no per_turn_outputs) should still work."""
        from evalview.core.diff import DiffEngine
        from evalview.core.golden import GoldenTrace, GoldenMetadata

        golden = GoldenTrace(
            metadata=GoldenMetadata(
                test_name="test",
                blessed_at=datetime.now(),
                score=90.0,
            ),
            trace=_make_trace([["search"]], ["result"]),
            tool_sequence=["search"],
            output_hash="abc",
            per_turn_tool_sequences=[["search"]],
            # No per_turn_outputs
        )

        actual = _make_trace([["search"]], ["result"])

        engine = DiffEngine()
        diff = engine.compare(golden, actual, 90.0)

        assert diff.turn_diffs is not None
        assert diff.turn_diffs[0].output_similarity is None  # Not computed


class TestCICommentTurnDetail:
    """Tests for turn-level detail in PR comments."""

    def test_turn_diffs_in_check_comment(self):
        """Check comment should include turn-level info when present."""
        from evalview.ci.comment import generate_check_pr_comment

        check_data = {
            "summary": {
                "total_tests": 2,
                "unchanged": 1,
                "regressions": 0,
                "tools_changed": 1,
                "output_changed": 0,
            },
            "diffs": [
                {
                    "test_name": "booking-flow",
                    "status": "tools_changed",
                    "score_delta": -8.2,
                    "tool_diffs": [{"type": "changed", "tool": "search"}],
                    "output_similarity": 0.90,
                    "turn_diffs": [
                        {"turn": 1, "status": "passed"},
                        {"turn": 2, "status": "tools_changed", "output_similarity": 0.65},
                        {"turn": 3, "status": "output_changed", "output_similarity": 0.70},
                    ],
                },
                {
                    "test_name": "clean-test",
                    "status": "passed",
                    "score_delta": 0,
                },
            ],
        }

        comment = generate_check_pr_comment(check_data)

        assert "Turn 2 tools changed" in comment
        assert "65% similar" in comment
        assert "Turn 3 output changed" in comment
        assert "70% similar" in comment
        # Turn 1 passed, should not appear
        assert "Turn 1" not in comment

    def test_no_turn_diffs_still_works(self):
        """Comments without turn_diffs should render normally."""
        from evalview.ci.comment import generate_check_pr_comment

        check_data = {
            "summary": {
                "total_tests": 1,
                "unchanged": 0,
                "regressions": 1,
                "tools_changed": 0,
                "output_changed": 0,
            },
            "diffs": [
                {
                    "test_name": "simple-test",
                    "status": "regression",
                    "score_delta": -10.0,
                    "tool_diffs": [],
                    "output_similarity": 0.5,
                },
            ],
        }

        comment = generate_check_pr_comment(check_data)
        assert "simple-test" in comment
        assert "Turn" not in comment
