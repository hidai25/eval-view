"""Statistical evaluator for variance-aware test execution.

This module provides functionality to run tests multiple times and compute
statistical metrics for pass/fail decisions, addressing the inherent
non-determinism in LLM-based agent testing.
"""

import math
import statistics
from datetime import datetime
from typing import List, Optional, Callable, Awaitable, Tuple

from evalview.core.types import (
    TestCase,
    ExecutionTrace,
    EvaluationResult,
    VarianceConfig,
    StatisticalMetrics,
    FlakinessScore,
    StatisticalEvaluationResult,
)


def compute_statistical_metrics(
    values: List[float],
    confidence_level: float = 0.95,
) -> StatisticalMetrics:
    """
    Compute comprehensive statistical metrics for a list of values.

    Args:
        values: List of numeric values to analyze
        confidence_level: Confidence level for interval calculation (default 0.95)

    Returns:
        StatisticalMetrics with all computed statistics
    """
    if not values:
        raise ValueError("Cannot compute statistics on empty list")

    n = len(values)
    sorted_values = sorted(values)

    # Basic statistics
    mean = statistics.mean(values)
    std_dev = statistics.stdev(values) if n > 1 else 0.0
    variance = std_dev ** 2

    # Percentiles
    def percentile(data: List[float], p: float) -> float:
        """Compute percentile using linear interpolation."""
        if not data:
            return 0.0
        k = (len(data) - 1) * p / 100
        f = math.floor(k)
        c = math.ceil(k)
        if f == c:
            return data[int(k)]
        return data[int(f)] * (c - k) + data[int(c)] * (k - f)

    median = percentile(sorted_values, 50)
    p25 = percentile(sorted_values, 25)
    p75 = percentile(sorted_values, 75)
    p95 = percentile(sorted_values, 95)

    # Confidence interval (using t-distribution approximation for small samples)
    # For simplicity, using normal approximation with z-scores
    z_scores = {
        0.90: 1.645,
        0.95: 1.96,
        0.99: 2.576,
    }
    z = z_scores.get(confidence_level, 1.96)

    if n > 1:
        margin = z * (std_dev / math.sqrt(n))
        ci_lower = mean - margin
        ci_upper = mean + margin
    else:
        ci_lower = mean
        ci_upper = mean

    return StatisticalMetrics(
        mean=round(mean, 4),
        std_dev=round(std_dev, 4),
        variance=round(variance, 4),
        min_value=round(min(values), 4),
        max_value=round(max(values), 4),
        median=round(median, 4),
        percentile_25=round(p25, 4),
        percentile_75=round(p75, 4),
        percentile_95=round(p95, 4),
        confidence_interval_lower=round(ci_lower, 4),
        confidence_interval_upper=round(ci_upper, 4),
        confidence_level=confidence_level,
    )


def compute_flakiness_score(
    results: List[EvaluationResult],
    score_stats: StatisticalMetrics,
) -> FlakinessScore:
    """
    Compute a flakiness score for a test based on its variance characteristics.

    The flakiness score is a composite measure that considers:
    - Pass/fail consistency
    - Score variance (coefficient of variation)
    - Output similarity across runs

    Args:
        results: List of evaluation results from multiple runs
        score_stats: Pre-computed statistical metrics for scores

    Returns:
        FlakinessScore with detailed assessment
    """
    if not results:
        raise ValueError("Cannot compute flakiness on empty results")

    total_runs = len(results)
    passed_runs = sum(1 for r in results if r.passed)
    pass_rate = passed_runs / total_runs

    # Coefficient of variation (CV) - relative measure of dispersion
    # CV = std_dev / mean (expressed as percentage)
    cv = (score_stats.std_dev / score_stats.mean * 100) if score_stats.mean > 0 else 0

    # Contributing factors analysis
    factors = []

    # Factor 1: Pass/fail inconsistency
    if 0 < pass_rate < 1:
        factors.append(f"inconsistent_pass_fail ({passed_runs}/{total_runs} passed)")

    # Factor 2: High score variance
    if cv > 20:
        factors.append(f"high_score_variance (CV={cv:.1f}%)")
    elif cv > 10:
        factors.append(f"moderate_score_variance (CV={cv:.1f}%)")

    # Factor 3: Wide score range
    score_range = score_stats.max_value - score_stats.min_value
    if score_range > 30:
        factors.append(f"wide_score_range ({score_range:.1f} points)")

    # Factor 4: Check tool call consistency
    tool_sets = []
    for r in results:
        tools = set(r.evaluations.tool_accuracy.correct +
                   r.evaluations.tool_accuracy.unexpected)
        tool_sets.append(frozenset(tools))

    unique_tool_patterns = len(set(tool_sets))
    if unique_tool_patterns > 1:
        factors.append(f"varying_tool_patterns ({unique_tool_patterns} unique patterns)")

    # Compute composite flakiness score (0 = stable, 1 = flaky)
    flakiness_components = []

    # Component 1: Pass rate deviation from 0 or 1 (max at 0.5)
    pass_rate_flakiness = 1 - abs(pass_rate - 0.5) * 2
    if pass_rate in (0, 1):
        pass_rate_flakiness = 0
    flakiness_components.append(pass_rate_flakiness * 0.4)  # 40% weight

    # Component 2: Coefficient of variation normalized (0-50% CV maps to 0-1)
    cv_flakiness = min(cv / 50, 1.0)
    flakiness_components.append(cv_flakiness * 0.3)  # 30% weight

    # Component 3: Score range normalized (0-50 points maps to 0-1)
    range_flakiness = min(score_range / 50, 1.0)
    flakiness_components.append(range_flakiness * 0.2)  # 20% weight

    # Component 4: Tool pattern consistency
    pattern_flakiness = (unique_tool_patterns - 1) / max(total_runs - 1, 1)
    flakiness_components.append(pattern_flakiness * 0.1)  # 10% weight

    flakiness_score = sum(flakiness_components)

    # Categorize flakiness
    if flakiness_score < 0.1:
        category = "stable"
    elif flakiness_score < 0.25:
        category = "low_variance"
    elif flakiness_score < 0.5:
        category = "moderate_variance"
    elif flakiness_score < 0.75:
        category = "high_variance"
    else:
        category = "flaky"

    return FlakinessScore(
        score=round(flakiness_score, 4),
        category=category,
        pass_rate=pass_rate,
        score_coefficient_of_variation=round(cv, 4),
        output_consistency=None,  # Could be enhanced with semantic similarity
        contributing_factors=factors if factors else ["none"],
    )


class StatisticalEvaluator:
    """
    Evaluator that runs tests multiple times and computes statistical metrics.

    This addresses the non-determinism inherent in LLM-based agents by:
    1. Running each test N times
    2. Computing variance metrics (mean, std dev, percentiles)
    3. Calculating a "flakiness score"
    4. Making pass/fail decisions based on statistical thresholds
    """

    def __init__(self, default_runs: int = 10, default_pass_rate: float = 0.8):
        """
        Initialize the statistical evaluator.

        Args:
            default_runs: Default number of test executions (can be overridden per-test)
            default_pass_rate: Default required pass rate (can be overridden per-test)
        """
        self.default_runs = default_runs
        self.default_pass_rate = default_pass_rate

    async def evaluate(
        self,
        test_case: TestCase,
        execute_fn: Callable[[TestCase], Awaitable[Tuple[ExecutionTrace, EvaluationResult]]],
        progress_callback: Optional[Callable[[int, int, Optional[EvaluationResult]], None]] = None,
    ) -> StatisticalEvaluationResult:
        """
        Run a test case multiple times and compute statistical evaluation.

        Args:
            test_case: The test case to evaluate
            execute_fn: Async function that executes the test and returns (trace, result)
            progress_callback: Optional callback(current_run, total_runs, last_result)

        Returns:
            StatisticalEvaluationResult with comprehensive statistics
        """
        # Get variance configuration
        variance_config = test_case.thresholds.variance or VarianceConfig()
        num_runs = variance_config.runs
        required_pass_rate = variance_config.pass_rate
        confidence_level = variance_config.confidence_level

        # Run the test multiple times
        results: List[EvaluationResult] = []

        for i in range(num_runs):
            try:
                _, result = await execute_fn(test_case)
                results.append(result)

                if progress_callback:
                    progress_callback(i + 1, num_runs, result)

            except Exception:
                # The execute_fn should handle most errors, but this is a safety net
                if progress_callback:
                    progress_callback(i + 1, num_runs, None)

        if not results:
            raise ValueError(f"All {num_runs} runs failed for test: {test_case.name}")

        # Compute statistics
        return self._compute_statistical_result(
            test_case=test_case,
            results=results,
            variance_config=variance_config,
            required_pass_rate=required_pass_rate,
            confidence_level=confidence_level,
        )

    def evaluate_from_results(
        self,
        test_case: TestCase,
        results: List[EvaluationResult],
        variance_config: Optional[VarianceConfig] = None,
    ) -> StatisticalEvaluationResult:
        """
        Compute statistical evaluation from pre-existing results.

        Useful for analyzing historical test data or cached results.

        Args:
            test_case: The test case these results belong to
            results: List of evaluation results to analyze
            variance_config: Optional config override

        Returns:
            StatisticalEvaluationResult
        """
        config = variance_config or test_case.thresholds.variance or VarianceConfig()

        return self._compute_statistical_result(
            test_case=test_case,
            results=results,
            variance_config=config,
            required_pass_rate=config.pass_rate,
            confidence_level=config.confidence_level,
        )

    def _compute_statistical_result(
        self,
        test_case: TestCase,
        results: List[EvaluationResult],
        variance_config: VarianceConfig,
        required_pass_rate: float,
        confidence_level: float,
    ) -> StatisticalEvaluationResult:
        """
        Compute the final statistical evaluation result.

        Args:
            test_case: Test case definition
            results: List of individual evaluation results
            variance_config: Variance configuration
            required_pass_rate: Required pass rate threshold
            confidence_level: Confidence level for intervals

        Returns:
            Complete StatisticalEvaluationResult
        """
        total_runs = len(results)
        successful_runs = sum(1 for r in results if r.passed)
        failed_runs = total_runs - successful_runs
        pass_rate = successful_runs / total_runs

        # Extract scores for statistics
        scores = [r.score for r in results]
        score_stats = compute_statistical_metrics(scores, confidence_level)

        # Extract costs and latencies (filter None values)
        costs = [
            r.trace.metrics.total_cost
            for r in results
            if r.trace.metrics and r.trace.metrics.total_cost is not None
        ]
        cost_stats = compute_statistical_metrics(costs, confidence_level) if costs else None

        latencies = [
            r.trace.metrics.total_latency
            for r in results
            if r.trace.metrics and r.trace.metrics.total_latency is not None
        ]
        latency_stats = compute_statistical_metrics(latencies, confidence_level) if latencies else None

        # Compute flakiness score
        flakiness = compute_flakiness_score(results, score_stats)

        # Determine pass/fail and reasons
        failure_reasons = []
        passed = True

        # Check 1: Pass rate threshold
        if pass_rate < required_pass_rate:
            passed = False
            failure_reasons.append(
                f"Pass rate {pass_rate:.1%} below required {required_pass_rate:.1%}"
            )

        # Check 2: Minimum mean score (if configured)
        if variance_config.min_mean_score is not None:
            if score_stats.mean < variance_config.min_mean_score:
                passed = False
                failure_reasons.append(
                    f"Mean score {score_stats.mean:.2f} below required {variance_config.min_mean_score}"
                )

        # Check 3: Maximum standard deviation (if configured)
        if variance_config.max_std_dev is not None:
            if score_stats.std_dev > variance_config.max_std_dev:
                passed = False
                failure_reasons.append(
                    f"Score std dev {score_stats.std_dev:.2f} exceeds max {variance_config.max_std_dev}"
                )

        # Check 4: Confidence interval must be above min_score
        min_score = test_case.thresholds.min_score
        if score_stats.confidence_interval_lower < min_score:
            # This is a warning, not necessarily a failure
            failure_reasons.append(
                f"CI lower bound {score_stats.confidence_interval_lower:.2f} below min_score {min_score}"
            )

        # Compute industry-standard reliability metrics
        # pass@k: probability of at least one success in k trials
        # Formula: 1 - (1 - p)^k where p = pass_rate, k = total_runs
        pass_at_k = 1 - ((1 - pass_rate) ** total_runs) if pass_rate < 1 else 1.0

        # pass^k: probability of all k trials succeeding
        # Formula: p^k where p = pass_rate, k = total_runs
        pass_power_k = pass_rate ** total_runs

        return StatisticalEvaluationResult(
            test_case=test_case.name,
            passed=passed,
            total_runs=total_runs,
            successful_runs=successful_runs,
            failed_runs=failed_runs,
            score_stats=score_stats,
            cost_stats=cost_stats,
            latency_stats=latency_stats,
            flakiness=flakiness,
            pass_rate=pass_rate,
            required_pass_rate=required_pass_rate,
            failure_reasons=failure_reasons if failure_reasons else [],
            pass_at_k=round(pass_at_k, 4),
            pass_power_k=round(pass_power_k, 4),
            individual_results=results,
            timestamp=datetime.now(),
            variance_config=variance_config,
        )


def is_statistical_mode(test_case: TestCase) -> bool:
    """
    Check if a test case should run in statistical mode.

    Args:
        test_case: Test case to check

    Returns:
        True if variance config is present, False for deterministic mode
    """
    return test_case.thresholds.variance is not None
