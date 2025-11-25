"""Cost threshold evaluator."""

import logging
from evalview.core.types import (
    TestCase,
    ExecutionTrace,
    CostEvaluation,
    CostBreakdown,
)

logger = logging.getLogger(__name__)


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

        # Warn if cost tracking isn't working
        if total_cost == 0.0 and trace.metrics.total_tokens is None:
            logger.warning(
                "⚠️  Cost tracking shows $0.00. Your agent may not be emitting cost data.\n"
                "   For streaming agents: emit {'type': 'usage', 'data': {...}} events\n"
                "   For REST agents: include 'cost' or 'tokens' in response metadata\n"
                "   See docs/BACKEND_REQUIREMENTS.md for details"
            )

        # Build breakdown by step
        breakdown = [
            CostBreakdown(step_id=step.step_id, cost=step.metrics.cost) for step in trace.steps
        ]

        passed = total_cost <= threshold

        return CostEvaluation(
            total_cost=total_cost,
            threshold=threshold,
            passed=passed,
            breakdown=breakdown,
        )
