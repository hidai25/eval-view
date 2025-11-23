"""Main evaluator orchestrator."""

from datetime import datetime
from typing import Optional
from evalview.core.types import (
    TestCase,
    ExecutionTrace,
    EvaluationResult,
    Evaluations,
)
from evalview.evaluators.tool_call_evaluator import ToolCallEvaluator
from evalview.evaluators.sequence_evaluator import SequenceEvaluator
from evalview.evaluators.output_evaluator import OutputEvaluator
from evalview.evaluators.cost_evaluator import CostEvaluator
from evalview.evaluators.latency_evaluator import LatencyEvaluator


class Evaluator:
    """Main evaluator that orchestrates all evaluation components."""

    def __init__(self, openai_api_key: Optional[str] = None):
        """
        Initialize evaluator.

        Args:
            openai_api_key: OpenAI API key for LLM-as-judge
        """
        self.tool_evaluator = ToolCallEvaluator()
        self.sequence_evaluator = SequenceEvaluator()
        self.output_evaluator = OutputEvaluator(openai_api_key)
        self.cost_evaluator = CostEvaluator()
        self.latency_evaluator = LatencyEvaluator()

    async def evaluate(
        self, test_case: TestCase, trace: ExecutionTrace
    ) -> EvaluationResult:
        """
        Run complete evaluation on a test case.

        Args:
            test_case: Test case with expected behavior
            trace: Execution trace from agent

        Returns:
            Complete evaluation result
        """
        # Run all evaluations
        evaluations = Evaluations(
            tool_accuracy=self.tool_evaluator.evaluate(test_case, trace),
            sequence_correctness=self.sequence_evaluator.evaluate(test_case, trace),
            output_quality=await self.output_evaluator.evaluate(test_case, trace),
            cost=self.cost_evaluator.evaluate(test_case, trace),
            latency=self.latency_evaluator.evaluate(test_case, trace),
        )

        # Compute overall score
        score = self._compute_overall_score(evaluations, test_case)

        # Determine pass/fail
        passed = self._compute_pass_fail(evaluations, test_case, score)

        return EvaluationResult(
            test_case=test_case.name,
            passed=passed,
            score=score,
            evaluations=evaluations,
            trace=trace,
            timestamp=datetime.now(),
            input_query=test_case.input.query,
            actual_output=trace.final_output,
        )

    def _compute_overall_score(
        self, evaluations: Evaluations, test_case: TestCase
    ) -> float:
        """
        Compute weighted overall score.

        Weights:
        - Tool accuracy: 30%
        - Output quality: 50%
        - Sequence correctness: 20%
        """
        weights = {
            "tool_accuracy": 0.3,
            "output_quality": 0.5,
            "sequence_correctness": 0.2,
        }

        score = (
            evaluations.tool_accuracy.accuracy * 100 * weights["tool_accuracy"]
            + evaluations.output_quality.score * weights["output_quality"]
            + (100 if evaluations.sequence_correctness.correct else 0)
            * weights["sequence_correctness"]
        )

        return round(score, 2)

    def _compute_pass_fail(
        self, evaluations: Evaluations, test_case: TestCase, score: float
    ) -> bool:
        """Determine if test case passed all criteria."""
        # Must pass score threshold
        if score < test_case.thresholds.min_score:
            return False

        # Must pass cost threshold (if specified)
        if not evaluations.cost.passed:
            return False

        # Must pass latency threshold (if specified)
        if not evaluations.latency.passed:
            return False

        return True
