"""Temporal drift tracking for gradual output degradation detection.

Stores per-check results in .evalview/history.jsonl and detects gradual
decline in output similarity that individual per-check thresholds would miss.

Usage:
    tracker = DriftTracker()
    tracker.record_check("my-test", diff)
    warning = tracker.detect_gradual_drift("my-test")
    if warning:
        console.print(f"[yellow]⚠ {warning}[/yellow]")

Why this matters:
    Individual checks use a fixed threshold (e.g., similarity < 0.95 triggers
    OUTPUT_CHANGED). A model that degrades from 0.97 → 0.95 → 0.93 over three
    weeks passes each individual check, but DriftTracker catches the declining
    trend and warns before it crosses the threshold.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from math import sqrt
from typing import Any, Dict, List, Optional, Tuple

from evalview.core.diff import TraceDiff

logger = logging.getLogger(__name__)

# Maximum total lines kept in history.jsonl across all tests.
# At ~200 bytes per line, 10 000 entries ≈ 2 MB — a reasonable ceiling.
# When this is exceeded, record_check() trims the oldest entries.
_MAX_HISTORY_ENTRIES = 10_000


def _compute_slope(values: List[float]) -> float:
    """Compute the OLS (ordinary least squares) regression slope.

    This is the mathematically correct definition of a linear regression
    slope — not the naive endpoint difference (first vs. last value), which
    is sensitive to outliers and ignores all intermediate data points.

    Args:
        values: Sequence of numeric values ordered chronologically
                (e.g., output similarities over successive checks).

    Returns:
        Slope of the best-fit line through the points (x=index, y=value).
        Negative slope = declining trend. Returns 0.0 for fewer than 2 points.

    Example:
        >>> _compute_slope([0.95, 0.93, 0.91, 0.89])
        -0.02  # declining 2% per check
        >>> _compute_slope([0.95, 0.93, 0.97, 0.91])
        -0.013  # noisy but slightly declining
    """
    n = len(values)
    if n < 2:
        return 0.0

    # x values are just the indices 0, 1, ..., n-1
    x_mean = (n - 1) / 2.0
    y_mean = sum(values) / n

    numerator = sum((i - x_mean) * (v - y_mean) for i, v in enumerate(values))
    denominator = sum((i - x_mean) ** 2 for i in range(n))

    if denominator == 0.0:
        return 0.0
    return numerator / denominator


class DriftTracker:
    """Tracks gradual drift in agent output quality over time.

    Appends each check result to .evalview/history.jsonl and provides
    trend analysis to surface slow-burning regressions that single-check
    thresholds would miss.

    The history file uses JSONL format (one JSON object per line) for easy
    streaming reads and append-only writes. Each line contains:

        {
          "ts": "2025-01-15T10:30:00",
          "test": "weather-lookup",
          "status": "passed",
          "score_diff": 0.5,
          "output_similarity": 0.97,
          "tool_changes": 0,
          "model_changed": false
        }
    """

    def __init__(self, base_path: Optional[Path] = None):
        """Initialize DriftTracker.

        Args:
            base_path: Root directory for .evalview/ data. Defaults to CWD.
        """
        self.base_path = base_path or Path(".")
        self.history_path = self.base_path / ".evalview" / "history.jsonl"

    def record_check(self, test_name: str, diff: TraceDiff) -> None:
        """Append a check result to the history log.

        Args:
            test_name: Name of the test that was checked.
            diff: TraceDiff result from this check run.
        """
        self.history_path.parent.mkdir(parents=True, exist_ok=True)

        output_similarity = diff.output_diff.similarity if diff.output_diff else 1.0

        entry: Dict[str, Any] = {
            "ts": datetime.now().isoformat(),
            "test": test_name,
            "status": diff.overall_severity.value,
            "score_diff": round(diff.score_diff, 4),
            "output_similarity": round(output_similarity, 4),
            "tool_changes": len(diff.tool_diffs),
            "model_changed": getattr(diff, "model_changed", False),
        }

        try:
            with open(self.history_path, "a") as f:
                f.write(json.dumps(entry) + "\n")
            self._prune_if_needed()
        except OSError as e:
            logger.warning(f"Failed to write drift history: {e}")

    def detect_gradual_drift(
        self,
        test_name: str,
        window: int = 10,
        slope_threshold: float = -0.02,
    ) -> Optional[str]:
        """Detect if output similarity has been gradually declining.

        Uses OLS regression slope to distinguish genuine trends from random
        noise — unlike endpoint comparison, a single outlier won't trigger
        a false alarm.

        Args:
            test_name: Test to analyze.
            window: Number of recent checks to include in trend analysis.
                    Default 10 gives a ~2-week view for daily CI runs.
            slope_threshold: Flag drift when slope is below this value.
                             Default -0.02 means 2%+ decline per check.
                             Tighten (e.g., -0.01) for sensitive tests.

        Returns:
            Human-readable warning string if drift is detected, else None.

        Example:
            warning = tracker.detect_gradual_drift("summarize-test")
            # "Output similarity declining over last 8 checks: 0.97 → 0.91
            #  (slope: -1.0%/check). May indicate gradual model drift."
        """
        recent = self._load_recent(test_name, window)
        if len(recent) < 3:
            return None  # Not enough data for a reliable trend estimate

        similarities = [r["output_similarity"] for r in recent]
        slope = _compute_slope(similarities)

        if slope < slope_threshold:
            first_val = similarities[0]
            last_val = similarities[-1]
            n = len(similarities)
            return (
                f"Output similarity declining over the last {n} checks: "
                f"{first_val:.0%} → {last_val:.0%} "
                f"(slope: {slope * 100:.1f}%/check). "
                f"This may indicate gradual model drift. "
                f"Run 'evalview check' more frequently or inspect recent changes."
            )

        return None

    def get_test_history(
        self, test_name: str, limit: int = 50
    ) -> List[Dict[str, Any]]:
        """Return recent check history for a test (newest first).

        Args:
            test_name: Test to retrieve history for.
            limit: Maximum number of entries to return.

        Returns:
            List of check records, newest first.
        """
        return list(reversed(self._load_recent(test_name, limit)))

    def compute_variance(
        self,
        test_name: str,
        window: int = 20,
    ) -> Tuple[float, float, int]:
        """Compute mean and standard deviation of output_similarity for a test.

        Args:
            test_name: Test to analyze.
            window: Number of recent entries to include.

        Returns:
            Tuple of (mean, stddev, sample_count). Returns (0.0, 0.0, 0) if
            no history exists.
        """
        recent = self._load_recent(test_name, window)
        if not recent:
            return (0.0, 0.0, 0)

        similarities = [r["output_similarity"] for r in recent]
        n = len(similarities)
        mean = sum(similarities) / n

        if n < 2:
            return (mean, 0.0, n)

        variance = sum((v - mean) ** 2 for v in similarities) / n
        return (mean, sqrt(variance), n)

    def compute_confidence(
        self,
        test_name: str,
        current_similarity: float,
        window: int = 20,
    ) -> Optional[Tuple[float, str]]:
        """Compute confidence that a change is a real signal vs. noise.

        Uses z-score: how many standard deviations the current value is from
        the historical mean. Higher z-score = more confidence it's a real change.

        Args:
            test_name: Test to analyze.
            current_similarity: The output_similarity from this check run.
            window: Historical window size.

        Returns:
            Tuple of (confidence_pct, label) where label is one of:
            "high", "medium", "low", "insufficient_history".
            Returns None if no history at all.
        """
        mean, stddev, count = self.compute_variance(test_name, window)

        if count == 0:
            return None

        if count < 3:
            return (0.0, "insufficient_history")

        if stddev == 0.0:
            # All historical values are identical
            if abs(current_similarity - mean) > 0.001:
                return (99.0, "high")
            return (5.0, "low")

        z = abs(current_similarity - mean) / stddev
        confidence_pct = min(99.0, 50.0 + z * 25.0)

        if z >= 2.0:
            return (confidence_pct, "high")
        elif z >= 1.0:
            return (confidence_pct, "medium")
        return (confidence_pct, "low")

    def get_pass_rate_trend(self, window: int = 10) -> List[float]:
        """Compute per-check-cycle pass rates over the last N check cycles.

        Groups entries by timestamp (truncated to the minute) to identify
        distinct check cycles, then computes the pass rate for each.

        Args:
            window: Number of recent cycles to return.

        Returns:
            List of pass rates (0.0-1.0) for each cycle, oldest first.
        """
        if not self.history_path.exists():
            return []

        # Load ALL entries (not filtered by test)
        entries: List[Dict[str, Any]] = []
        try:
            with open(self.history_path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        entries.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        except OSError:
            return []

        if not entries:
            return []

        # Group by timestamp truncated to the minute (dict preserves insertion order)
        cycles: Dict[str, List[str]] = {}
        for entry in entries:
            ts = entry.get("ts", "")
            # Truncate to minute: "2025-01-15T10:30:45" -> "2025-01-15T10:30"
            cycle_key = ts[:16] if len(ts) >= 16 else ts
            status = entry.get("status", "")
            if cycle_key not in cycles:
                cycles[cycle_key] = []
            cycles[cycle_key].append(status)

        # Compute pass rate per cycle
        rates: List[float] = []
        for statuses in cycles.values():
            total = len(statuses)
            passed = sum(1 for s in statuses if s == "passed")
            rates.append(passed / total if total > 0 else 0.0)

        return rates[-window:]

    def _prune_if_needed(self) -> None:
        """Trim history file to _MAX_HISTORY_ENTRIES if it has grown too large.

        Uses a fast file-size stat() check to skip the expensive full read in
        the common case where the file is well under the limit.
        At ~200 bytes per entry, the file won't need pruning until ~2 MB.
        """
        try:
            # Fast guard: skip the read unless the file is large enough to
            # plausibly exceed _MAX_HISTORY_ENTRIES. Lower bound: 150 bytes/line.
            try:
                if self.history_path.stat().st_size < _MAX_HISTORY_ENTRIES * 150:
                    return
            except OSError:
                return  # File not written yet; nothing to prune.

            with open(self.history_path) as f:
                lines = f.readlines()
            if len(lines) > _MAX_HISTORY_ENTRIES:
                with open(self.history_path, "w") as f:
                    f.writelines(lines[-_MAX_HISTORY_ENTRIES:])
                logger.debug(
                    f"Pruned drift history to {_MAX_HISTORY_ENTRIES} entries"
                )
        except OSError as e:
            logger.warning(f"Failed to prune drift history: {e}")

    def _load_recent(self, test_name: str, window: int) -> List[Dict[str, Any]]:
        """Load the most recent `window` entries for test_name (oldest first)."""
        if not self.history_path.exists():
            return []

        entries = []
        try:
            with open(self.history_path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        entry = json.loads(line)
                        if entry.get("test") == test_name:
                            entries.append(entry)
                    except json.JSONDecodeError:
                        logger.debug(
                            "Skipping malformed JSON line in drift history: %.80r",
                            line,
                        )
                        continue
        except OSError as e:
            logger.warning(f"Failed to read drift history: {e}")
            return []

        # Return the most recent `window` entries in chronological order
        return entries[-window:]
