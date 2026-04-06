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
        max_cost = test_case.thresholds.max_cost
        threshold = max_cost if max_cost is not None else float("inf")

        # Local/free adapters (opencode with local models, goose, ollama) don't emit cost data — that's expected.
        # Only warn for cloud HTTP adapters where $0.00 signals a misconfiguration.
        _FREE_ADAPTERS = {"opencode", "goose", "ollama"}
        adapter_type = (test_case.adapter or "").lower()
        is_local = adapter_type in _FREE_ADAPTERS

        if total_cost == 0.0 and trace.metrics.total_tokens is None and not is_local:
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

        passed = total_cost >= 0 and total_cost <= threshold

        return CostEvaluation(
            total_cost=total_cost,
            threshold=threshold,
            passed=passed,
            breakdown=breakdown,
        )
