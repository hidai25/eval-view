"""Regression detection and comparison logic."""

import subprocess
from pathlib import Path
from typing import Optional, Dict, Any, List
from dataclasses import asdict, dataclass

from evalview.core.types import EvaluationResult
from evalview.core.drift_tracker import _compute_slope
from evalview.tracking.database import TrackingDatabase


# Minimum absolute slope, in score points per run, for a trend to count as
# significant. Scores are on a 0-100 scale, so 1.0 means "a point of score per
# run" — roughly 10 points of drift across a 10-run window.
DEFAULT_TREND_THRESHOLD = 1.0

# An OLS fit through fewer than three points describes noise, not a trend.
MIN_TREND_RUNS = 3


@dataclass
class RegressionReport:
    """Report of regression analysis."""

    test_name: str
    current_score: float
    baseline_score: Optional[float]
    score_delta: Optional[float]
    score_delta_percent: Optional[float]
    current_cost: Optional[float]
    baseline_cost: Optional[float]
    cost_delta: Optional[float]
    cost_delta_percent: Optional[float]
    current_latency: Optional[float]
    baseline_latency: Optional[float]
    latency_delta: Optional[float]
    latency_delta_percent: Optional[float]
    is_regression: bool
    severity: str  # "none", "minor", "moderate", "critical"
    issues: List[str]


@dataclass
class ScoreTrend:
    """Direction of a test's score across its recent runs."""

    slope: float  # score points per run; negative means degrading
    direction: str  # "improving", "worsening", "stable"
    significant: bool  # mirrors direction != "stable"
    run_count: int


def _compute_trend(
    scores_oldest_first: List[float],
    threshold: float = DEFAULT_TREND_THRESHOLD,
) -> ScoreTrend:
    """
    Classify the direction of chronologically ordered scores.

    Delegates the fit to :func:`evalview.core.drift_tracker._compute_slope` so
    there is a single OLS implementation in the codebase. That helper is used
    rather than ``statistics.linear_regression`` because the latter is Python
    3.10+ and EvalView supports 3.9.

    ``_compute_slope`` guards ``n < 2`` to keep its own division safe; the
    stricter ``MIN_TREND_RUNS`` guard here is a separate, statistical judgement
    that two points are not enough to call a direction.

    Args:
        scores_oldest_first: Scores ordered oldest run first. Callers reading
            from ``TrackingDatabase.get_test_history`` must reverse it, since
            that query returns newest first.
        threshold: Minimum absolute slope, in score points per run, that counts
            as a real trend rather than noise.

    Returns:
        Trend over the supplied scores. Fewer than ``MIN_TREND_RUNS`` points
        yields a zero-slope "stable" trend. ``significant`` is a convenience
        mirror of ``direction != "stable"``, kept so callers can gate CI on a
        single boolean.
    """
    run_count = len(scores_oldest_first)
    if run_count < MIN_TREND_RUNS:
        return ScoreTrend(slope=0.0, direction="stable", significant=False, run_count=run_count)

    # Round before classifying so the reported slope always reproduces the
    # reported direction.
    slope = round(_compute_slope(scores_oldest_first), 4)

    if slope > threshold:
        direction = "improving"
    elif slope < -threshold:
        direction = "worsening"
    else:
        direction = "stable"

    return ScoreTrend(
        slope=slope,
        direction=direction,
        significant=direction != "stable",
        run_count=run_count,
    )


class RegressionTracker:
    """Track test results and detect regressions."""

    def __init__(self, db_path: Path = Path(".evalview/tracking.db")):
        """
        Initialize regression tracker.

        Args:
            db_path: Path to tracking database
        """
        self.db = TrackingDatabase(db_path)

    def store_result(self, result: EvaluationResult) -> int:
        """
        Store evaluation result in database.

        Args:
            result: Evaluation result to store

        Returns:
            ID of stored result
        """
        # Extract git information
        git_commit, git_branch = self._get_git_info()

        # Extract metrics from evaluation
        tool_accuracy = result.evaluations.tool_accuracy.accuracy
        output_quality = result.evaluations.output_quality.score
        sequence_correct = result.evaluations.sequence_correctness.correct

        hallucination_detected = None
        if result.evaluations.hallucination:
            hallucination_detected = result.evaluations.hallucination.has_hallucination

        safety_passed = None
        if result.evaluations.safety:
            safety_passed = result.evaluations.safety.is_safe

        # Store in database
        return self.db.store_result(
            test_name=result.test_case,
            score=result.score,
            passed=result.passed,
            cost=result.trace.metrics.total_cost if result.trace.metrics else None,
            latency=result.trace.metrics.total_latency if result.trace.metrics else None,
            tool_accuracy=tool_accuracy,
            output_quality=output_quality,
            sequence_correct=sequence_correct,
            hallucination_detected=hallucination_detected,
            safety_passed=safety_passed,
            git_commit=git_commit,
            git_branch=git_branch,
        )

    def compare_to_baseline(self, result: EvaluationResult) -> RegressionReport:
        """
        Compare result to baseline and detect regressions.

        Args:
            result: Current evaluation result

        Returns:
            RegressionReport with comparison details
        """
        # Get baseline
        baseline = self.db.get_baseline(result.test_case)

        # Extract current metrics
        current_score = result.score
        current_cost = result.trace.metrics.total_cost if result.trace.metrics else None
        current_latency = result.trace.metrics.total_latency if result.trace.metrics else None

        if not baseline:
            return RegressionReport(
                test_name=result.test_case,
                current_score=current_score,
                baseline_score=None,
                score_delta=None,
                score_delta_percent=None,
                current_cost=current_cost,
                baseline_cost=None,
                cost_delta=None,
                cost_delta_percent=None,
                current_latency=current_latency,
                baseline_latency=None,
                latency_delta=None,
                latency_delta_percent=None,
                is_regression=False,
                severity="none",
                issues=["No baseline set"],
            )

        # Calculate deltas
        baseline_score = baseline["score"]
        score_delta = current_score - baseline_score
        score_delta_percent = (score_delta / baseline_score * 100) if baseline_score > 0 else 0

        baseline_cost = baseline.get("cost")
        cost_delta = None
        cost_delta_percent = None
        if current_cost is not None and baseline_cost is not None:
            cost_delta = current_cost - baseline_cost
            cost_delta_percent = (cost_delta / baseline_cost * 100) if baseline_cost > 0 else 0

        baseline_latency = baseline.get("latency")
        latency_delta = None
        latency_delta_percent = None
        if current_latency is not None and baseline_latency is not None:
            latency_delta = current_latency - baseline_latency
            latency_delta_percent = (
                (latency_delta / baseline_latency * 100) if baseline_latency > 0 else 0
            )

        # Detect regressions
        issues = []
        is_regression = False
        severity = "none"

        # Score regression (>10% drop is moderate, >20% is critical)
        if score_delta_percent < -10:
            is_regression = True
            if score_delta_percent < -20:
                severity = "critical"
                issues.append(f"Critical score regression: {score_delta_percent:.1f}% drop")
            else:
                severity = "moderate"
                issues.append(f"Moderate score regression: {score_delta_percent:.1f}% drop")
        elif score_delta_percent < -5:
            severity = "minor" if severity == "none" else severity
            issues.append(f"Minor score regression: {score_delta_percent:.1f}% drop")

        # Cost regression (>20% increase)
        if cost_delta_percent and cost_delta_percent > 20:
            is_regression = True
            if cost_delta_percent > 50:
                severity = "critical" if severity != "critical" else severity
                issues.append(f"Critical cost increase: {cost_delta_percent:.1f}%")
            else:
                severity = "moderate" if severity in ["none", "minor"] else severity
                issues.append(f"Moderate cost increase: {cost_delta_percent:.1f}%")

        # Latency regression (>30% increase)
        if latency_delta_percent and latency_delta_percent > 30:
            is_regression = True
            if latency_delta_percent > 100:
                severity = "critical" if severity != "critical" else severity
                issues.append(f"Critical latency increase: {latency_delta_percent:.1f}%")
            else:
                severity = "moderate" if severity in ["none", "minor"] else severity
                issues.append(f"Moderate latency increase: {latency_delta_percent:.1f}%")

        # Pass/fail regression (previously passed, now fails)
        baseline_passed = baseline.get("passed", 1)
        if baseline_passed and not result.passed:
            is_regression = True
            severity = "critical"
            issues.append("Test now fails (previously passed)")

        return RegressionReport(
            test_name=result.test_case,
            current_score=current_score,
            baseline_score=baseline_score,
            score_delta=score_delta,
            score_delta_percent=score_delta_percent,
            current_cost=current_cost,
            baseline_cost=baseline_cost,
            cost_delta=cost_delta,
            cost_delta_percent=cost_delta_percent,
            current_latency=current_latency,
            baseline_latency=baseline_latency,
            latency_delta=latency_delta,
            latency_delta_percent=latency_delta_percent,
            is_regression=is_regression,
            severity=severity,
            issues=issues if issues else ["No regressions detected"],
        )

    def set_baseline_from_result(self, result: EvaluationResult):
        """
        Set baseline from an evaluation result.

        Args:
            result: Evaluation result to use as baseline
        """
        git_commit, git_branch = self._get_git_info()

        tool_accuracy = result.evaluations.tool_accuracy.accuracy
        output_quality = result.evaluations.output_quality.score

        self.db.set_baseline(
            test_name=result.test_case,
            score=result.score,
            cost=result.trace.metrics.total_cost if result.trace.metrics else None,
            latency=result.trace.metrics.total_latency if result.trace.metrics else None,
            tool_accuracy=tool_accuracy,
            output_quality=output_quality,
            git_commit=git_commit,
            git_branch=git_branch,
        )

    def set_baseline_from_latest(self, test_name: str):
        """
        Set baseline from the most recent test result.

        Args:
            test_name: Name of test to set baseline for
        """
        history = self.db.get_test_history(test_name, days=7)

        if not history:
            raise ValueError(f"No recent results found for test: {test_name}")

        latest = history[0]

        self.db.set_baseline(
            test_name=test_name,
            score=latest["score"],
            cost=latest.get("cost"),
            latency=latest.get("latency"),
            tool_accuracy=latest.get("tool_accuracy"),
            output_quality=latest.get("output_quality"),
            git_commit=latest.get("git_commit"),
            git_branch=latest.get("git_branch"),
        )

    def get_statistics(self, test_name: str, days: int = 30) -> Dict[str, Any]:
        """
        Get statistics for a test over time.

        Args:
            test_name: Name of test
            days: Number of days to analyze

        Returns:
            Statistics dictionary. ``score["trend"]`` holds the OLS slope over
            the window, which reveals monotonic drift that avg/min/max hide.
        """
        history = self.db.get_test_history(test_name, days)

        if not history:
            return {
                "test_name": test_name,
                "total_runs": 0,
                "period_days": days,
            }

        scores = [h["score"] for h in history]
        costs = [h["cost"] for h in history if h.get("cost") is not None]
        latencies = [h["latency"] for h in history if h.get("latency") is not None]

        passed_count = sum(1 for h in history if h["passed"])
        failed_count = len(history) - passed_count

        # get_test_history orders newest first; the fit needs oldest first or the
        # slope comes out sign-flipped.
        trend = _compute_trend(list(reversed(scores)))

        return {
            "test_name": test_name,
            "total_runs": len(history),
            "passed_runs": passed_count,
            "failed_runs": failed_count,
            "pass_rate": (passed_count / len(history) * 100) if history else 0,
            "period_days": days,
            "score": {
                "current": scores[0] if scores else None,
                "avg": sum(scores) / len(scores) if scores else None,
                "min": min(scores) if scores else None,
                "max": max(scores) if scores else None,
                "trend": asdict(trend),
            },
            "cost": {
                "current": costs[0] if costs else None,
                "avg": sum(costs) / len(costs) if costs else None,
                "min": min(costs) if costs else None,
                "max": max(costs) if costs else None,
            },
            "latency": {
                "current": latencies[0] if latencies else None,
                "avg": sum(latencies) / len(latencies) if latencies else None,
                "min": min(latencies) if latencies else None,
                "max": max(latencies) if latencies else None,
            },
        }

    def _get_git_info(self) -> tuple[Optional[str], Optional[str]]:
        """
        Get current git commit and branch.

        Returns:
            Tuple of (commit_hash, branch_name)
        """
        try:
            # Get commit hash
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                check=True,
                timeout=5,
            )
            commit = result.stdout.strip()[:8]  # Short hash

            # Get branch name
            result = subprocess.run(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                capture_output=True,
                text=True,
                check=True,
                timeout=5,
            )
            branch = result.stdout.strip()

            return commit, branch

        except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError):
            return None, None
