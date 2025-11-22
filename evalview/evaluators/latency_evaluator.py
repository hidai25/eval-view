"""Latency threshold evaluator."""

from evalview.core.types import (
    TestCase,
    ExecutionTrace,
    LatencyEvaluation,
    LatencyBreakdown,
)


class LatencyEvaluator:
    """Evaluates whether execution stayed within latency thresholds."""

    def evaluate(self, test_case: TestCase, trace: ExecutionTrace) -> LatencyEvaluation:
        """
        Evaluate latency against threshold.

        Args:
            test_case: Test case with latency threshold
            trace: Execution trace with actual latencies

        Returns:
            LatencyEvaluation with pass/fail status
        """
        total_latency = trace.metrics.total_latency
        threshold = test_case.thresholds.max_latency or float("inf")

        # Build breakdown by step
        breakdown = [
            LatencyBreakdown(step_id=step.step_id, latency=step.metrics.latency)
            for step in trace.steps
        ]

        passed = total_latency <= threshold

        return LatencyEvaluation(
            total_latency=total_latency,
            threshold=threshold,
            passed=passed,
            breakdown=breakdown,
        )
