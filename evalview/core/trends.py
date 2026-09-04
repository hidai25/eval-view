"""Shared helpers for score and similarity trend analysis."""

from dataclasses import dataclass
from typing import List, Literal


TrendDirection = Literal["improving", "worsening", "stable"]

# Scores use a 0-100 scale. A one-point change per run is therefore roughly
# one percent of the full scale per run.
DEFAULT_SCORE_TREND_THRESHOLD = 1.0
DEFAULT_SCORE_TREND_WINDOW = 10
MIN_TREND_RUNS = 3


@dataclass
class ScoreTrend:
    """Threshold-based direction of a score series."""

    slope: float
    direction: TrendDirection
    significant: bool
    run_count: int


def compute_slope(values: List[float]) -> float:
    """Compute the OLS regression slope for chronologically ordered values."""
    n = len(values)
    if n < 2:
        return 0.0

    x_mean = (n - 1) / 2.0
    y_mean = sum(values) / n
    numerator = sum((i - x_mean) * (value - y_mean) for i, value in enumerate(values))
    denominator = sum((i - x_mean) ** 2 for i in range(n))

    if denominator == 0.0:
        return 0.0
    return numerator / denominator


def compute_score_trend(
    scores_oldest_first: List[float],
    threshold: float = DEFAULT_SCORE_TREND_THRESHOLD,
) -> ScoreTrend:
    """Classify a chronological score series using a magnitude threshold.

    ``significant`` means the rounded absolute slope exceeds ``threshold``;
    it is not a statistical hypothesis test and does not estimate variance.
    """
    if threshold < 0:
        raise ValueError("trend threshold must be non-negative")

    run_count = len(scores_oldest_first)
    if run_count < MIN_TREND_RUNS:
        return ScoreTrend(
            slope=0.0,
            direction="stable",
            significant=False,
            run_count=run_count,
        )

    # Classify the same rounded number returned to callers so the direction is
    # reproducible from the public payload.
    slope = round(compute_slope(scores_oldest_first), 4)

    direction: TrendDirection
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


__all__ = [
    "DEFAULT_SCORE_TREND_THRESHOLD",
    "DEFAULT_SCORE_TREND_WINDOW",
    "MIN_TREND_RUNS",
    "ScoreTrend",
    "TrendDirection",
    "compute_score_trend",
    "compute_slope",
]
