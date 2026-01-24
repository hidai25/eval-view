"""Tool sequence correctness evaluator."""

from typing import List, Literal
from evalview.core.types import TestCase, ExecutionTrace, SequenceEvaluation


# Sequence matching modes
SequenceMode = Literal["exact", "subsequence", "unordered"]


class SequenceEvaluator:
    """Evaluates whether tools were called in the correct order.

    Supports three matching modes:
    - exact: Tools must match exactly in order and count (legacy behavior)
    - subsequence: Expected tools must appear in order, but other tools can exist between them
    - unordered: Expected tools must be called, order doesn't matter

    The 'subsequence' mode is recommended as it avoids penalizing agents for finding
    valid alternative paths that include the expected tools.
    """

    def __init__(self, default_mode: SequenceMode = "subsequence"):
        """
        Initialize the sequence evaluator.

        Args:
            default_mode: Default matching mode ("exact", "subsequence", or "unordered")
        """
        self.default_mode = default_mode

    def evaluate(
        self,
        test_case: TestCase,
        trace: ExecutionTrace,
        mode: SequenceMode = None,
    ) -> SequenceEvaluation:
        """
        Evaluate tool call sequence correctness.

        Args:
            test_case: Test case with expected sequence
            trace: Execution trace with actual sequence
            mode: Matching mode override (defaults to instance default)

        Returns:
            SequenceEvaluation with correctness check
        """
        # Get mode from test case config, parameter, or instance default
        effective_mode = mode or self._get_mode_from_test_case(test_case) or self.default_mode

        expected_sequence = test_case.expected.tool_sequence or test_case.expected.sequence or []
        actual_sequence = [step.tool_name for step in trace.steps]

        # If no expected sequence, pass by default with perfect progress
        if not expected_sequence:
            return SequenceEvaluation(
                correct=True,
                expected_sequence=expected_sequence,
                actual_sequence=actual_sequence,
                violations=[],
                progress_score=1.0,
            )

        # Dispatch to appropriate matching strategy
        if effective_mode == "exact":
            return self._evaluate_exact(expected_sequence, actual_sequence)
        elif effective_mode == "subsequence":
            return self._evaluate_subsequence(expected_sequence, actual_sequence)
        elif effective_mode == "unordered":
            return self._evaluate_unordered(expected_sequence, actual_sequence)
        else:
            # Fallback to subsequence (safest default)
            return self._evaluate_subsequence(expected_sequence, actual_sequence)

    def _get_mode_from_test_case(self, test_case: TestCase) -> SequenceMode:
        """Extract sequence mode from test case config if specified."""
        # Check if test case has adapter_config with sequence_mode
        if test_case.adapter_config and "sequence_mode" in test_case.adapter_config:
            return test_case.adapter_config["sequence_mode"]
        return None

    def _evaluate_exact(
        self, expected: List[str], actual: List[str]
    ) -> SequenceEvaluation:
        """
        Exact matching: sequences must be identical.

        This is the legacy behavior - use with caution as it penalizes agents
        for finding valid alternative paths.

        Progress score: counts matching positions / total expected.
        """
        violations: List[str] = []
        correct = True
        matching_positions = 0

        if len(expected) != len(actual):
            correct = False
            violations.append(
                f"Length mismatch: expected {len(expected)} steps, "
                f"got {len(actual)}"
            )
            # For length mismatch, count matches up to min length
            for i in range(min(len(expected), len(actual))):
                if expected[i] == actual[i]:
                    matching_positions += 1
        else:
            for i, (exp, act) in enumerate(zip(expected, actual)):
                if exp == act:
                    matching_positions += 1
                else:
                    correct = False
                    violations.append(f"Step {i + 1}: expected '{exp}', got '{act}'")

        # Calculate progress score: proportion of expected steps matched in position
        progress_score = matching_positions / len(expected) if expected else 1.0

        return SequenceEvaluation(
            correct=correct,
            expected_sequence=expected,
            actual_sequence=actual,
            violations=violations,
            progress_score=round(progress_score, 4),
        )

    def _evaluate_subsequence(
        self, expected: List[str], actual: List[str]
    ) -> SequenceEvaluation:
        """
        Subsequence matching: expected tools must appear in order within actual.

        This is the recommended mode - it verifies the agent followed the critical
        path without penalizing additional intermediate steps.

        Example:
            expected: [search, analyze, respond]
            actual: [search, think, analyze, verify, respond]
            Result: PASS (expected tools appear in order)

        Progress score: found_in_order / total_expected
        Example: found 3/5 expected tools in order = 0.6
        """
        violations: List[str] = []

        # Track position in expected sequence
        expected_idx = 0

        # Scan actual sequence looking for expected tools in order
        for actual_idx, tool in enumerate(actual):
            if expected_idx < len(expected) and tool == expected[expected_idx]:
                expected_idx += 1

        # Check if all expected tools were found in order
        correct = expected_idx == len(expected)

        # Calculate progress score: proportion of expected sequence completed
        progress_score = expected_idx / len(expected) if expected else 1.0

        if not correct:
            missing = expected[expected_idx:]
            violations.append(
                f"Missing tools in sequence: {missing}. "
                f"Found {expected_idx}/{len(expected)} expected tools in order."
            )
            # Show what was found vs expected for debugging
            found_in_order = expected[:expected_idx]
            if found_in_order:
                violations.append(f"Found in order: {found_in_order}")

        return SequenceEvaluation(
            correct=correct,
            expected_sequence=expected,
            actual_sequence=actual,
            violations=violations,
            progress_score=round(progress_score, 4),
        )

    def _evaluate_unordered(
        self, expected: List[str], actual: List[str]
    ) -> SequenceEvaluation:
        """
        Unordered matching: expected tools must be called, order doesn't matter.

        Use this mode when you care that certain tools are used but the agent
        has flexibility in execution order.

        Example:
            expected: [search, analyze]
            actual: [analyze, think, search]
            Result: PASS (both expected tools were called)

        Progress score: satisfied_tool_requirements / total_expected_requirements
        Handles duplicates: if expected=[a, a, b] and actual=[a, b], score = 2/3
        """
        violations: List[str] = []

        # Convert to multisets to handle duplicates properly
        expected_counts = {}
        for tool in expected:
            expected_counts[tool] = expected_counts.get(tool, 0) + 1

        actual_counts = {}
        for tool in actual:
            actual_counts[tool] = actual_counts.get(tool, 0) + 1

        # Check that all expected tools were called (at least) the expected number of times
        # Also track satisfied count for progress score
        missing = []
        total_expected = len(expected)  # Total expected tool calls (with duplicates)
        satisfied_count = 0

        for tool, expected_count in expected_counts.items():
            actual_count = actual_counts.get(tool, 0)
            # Count how many of this tool's requirements were satisfied
            satisfied_for_tool = min(actual_count, expected_count)
            satisfied_count += satisfied_for_tool

            if actual_count < expected_count:
                missing.append(f"{tool} (expected {expected_count}, got {actual_count})")

        correct = len(missing) == 0

        # Calculate progress score: proportion of expected tool calls satisfied
        progress_score = satisfied_count / total_expected if total_expected > 0 else 1.0

        if not correct:
            violations.append(f"Missing or insufficient tool calls: {missing}")

        return SequenceEvaluation(
            correct=correct,
            expected_sequence=expected,
            actual_sequence=actual,
            violations=violations,
            progress_score=round(progress_score, 4),
        )
