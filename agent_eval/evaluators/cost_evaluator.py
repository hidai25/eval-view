"""Cost threshold evaluator."""

from agent_eval.core.types import (
    TestCase,
    ExecutionTrace,
    CostEvaluation,
    CostBreakdown,
)


class CostEvaluator:
    """Evaluates whether execution stayed within cost thresholds."""

    def evaluate(self, test_case: TestCase, trace: ExecutionTrace) -> CostEvaluation:
        """
        Evaluate cost against threshold.

        Args:
            test_case: Test case with cost threshold
            trace: Execution trace with actual costs

        Returns:
            CostEvaluation with pass/fail status
        """
        total_cost = trace.metrics.total_cost
        threshold = test_case.thresholds.max_cost or float("inf")

        # Build breakdown by step
        breakdown = [
            CostBreakdown(step_id=step.step_id, cost=step.metrics.cost)
            for step in trace.steps
        ]

        passed = total_cost <= threshold

        return CostEvaluation(
            total_cost=total_cost,
            threshold=threshold,
            passed=passed,
            breakdown=breakdown,
        )
