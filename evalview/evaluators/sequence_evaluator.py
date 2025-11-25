"""Tool sequence correctness evaluator."""

from typing import List
from evalview.core.types import TestCase, ExecutionTrace, SequenceEvaluation


class SequenceEvaluator:
    """Evaluates whether tools were called in the correct order."""

    def evaluate(self, test_case: TestCase, trace: ExecutionTrace) -> SequenceEvaluation:
        """
        Evaluate tool call sequence correctness.

        Args:
            test_case: Test case with expected sequence
            trace: Execution trace with actual sequence

        Returns:
            SequenceEvaluation with correctness check
        """
        expected_sequence = test_case.expected.tool_sequence or []
        actual_sequence = [step.tool_name for step in trace.steps]

        violations: List[str] = []
        correct = True

        # If no expected sequence, pass by default
        if not expected_sequence:
            return SequenceEvaluation(
                correct=True,
                expected_sequence=expected_sequence,
                actual_sequence=actual_sequence,
                violations=[],
            )

        # Check if sequences match
        if len(expected_sequence) != len(actual_sequence):
            correct = False
            violations.append(
                f"Length mismatch: expected {len(expected_sequence)} steps, "
                f"got {len(actual_sequence)}"
            )
        else:
            for i, (expected, actual) in enumerate(zip(expected_sequence, actual_sequence)):
                if expected != actual:
                    correct = False
                    violations.append(f"Step {i + 1}: expected '{expected}', got '{actual}'")

        return SequenceEvaluation(
            correct=correct,
            expected_sequence=expected_sequence,
            actual_sequence=actual_sequence,
            violations=violations,
        )
