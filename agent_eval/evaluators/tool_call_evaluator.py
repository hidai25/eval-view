"""Tool call accuracy evaluator."""

from typing import List
from agent_eval.core.types import TestCase, ExecutionTrace, ToolEvaluation


class ToolCallEvaluator:
    """Evaluates whether the agent called the expected tools."""

    def evaluate(self, test_case: TestCase, trace: ExecutionTrace) -> ToolEvaluation:
        """
        Evaluate tool call accuracy.

        Args:
            test_case: Test case with expected tools
            trace: Execution trace with actual tool calls

        Returns:
            ToolEvaluation with accuracy metrics
        """
        expected_tools = set(test_case.expected.tools or [])
        actual_tools = [step.tool_name for step in trace.steps]

        correct: List[str] = []
        missing: List[str] = []
        unexpected: List[str] = []

        # Check for expected tools
        for tool in expected_tools:
            if tool in actual_tools:
                correct.append(tool)
            else:
                missing.append(tool)

        # Check for unexpected tools
        for tool in actual_tools:
            if tool not in expected_tools:
                unexpected.append(tool)

        # Calculate accuracy
        accuracy = 1.0 if len(expected_tools) == 0 else len(correct) / len(expected_tools)

        return ToolEvaluation(
            accuracy=accuracy,
            correct=correct,
            missing=missing,
            unexpected=unexpected,
        )
