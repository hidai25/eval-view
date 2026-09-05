"""Tests for evalview/tracking/regression.py score-trend analysis."""

import pytest

from evalview.tracking.regression import (
    DEFAULT_TREND_THRESHOLD,
    DEFAULT_TREND_WINDOW,
    MIN_TREND_RUNS,
    RegressionTracker,
    ScoreTrend,
    _compute_trend,
)


class TestComputeTrend:
    """Unit tests for the trend classifier."""

    def test_monotonic_decline_is_worsening(self):
        # 5 points per run down, well past the significance threshold.
        trend = _compute_trend([90.0, 85.0, 80.0, 75.0, 70.0])
        assert trend.slope == pytest.approx(-5.0)
        assert trend.direction == "worsening"
        assert trend.significant is True
        assert trend.run_count == 5

    def test_monotonic_improvement_is_improving(self):
        trend = _compute_trend([70.0, 75.0, 80.0, 85.0, 90.0])
        assert trend.slope == pytest.approx(5.0)
        assert trend.direction == "improving"
        assert trend.significant is True

    def test_flat_scores_are_stable(self):
        trend = _compute_trend([80.0, 80.0, 80.0, 80.0])
        assert trend.slope == pytest.approx(0.0)
        assert trend.direction == "stable"
        assert trend.significant is False

    def test_drift_below_threshold_is_stable(self):
        # 0.1 points per run: real but under DEFAULT_TREND_THRESHOLD, so it must
        # not be reported as a trend or CI gates would fire on noise.
        trend = _compute_trend([80.0, 79.9, 79.8, 79.7, 79.6])
        assert trend.slope == pytest.approx(-0.1)
        assert abs(trend.slope) < DEFAULT_TREND_THRESHOLD
        assert trend.direction == "stable"
        assert trend.significant is False

    def test_average_hides_decline_that_trend_surfaces(self):
        # The motivating case: avg/min/max look healthy while every run is worse
        # than the last.
        scores = [80.0, 77.0, 74.0, 71.0, 68.0]
        assert sum(scores) / len(scores) == pytest.approx(74.0)
        trend = _compute_trend(scores)
        assert trend.direction == "worsening"

    def test_fewer_than_minimum_runs_is_stable(self):
        for scores in ([], [50.0], [90.0, 10.0]):
            trend = _compute_trend(list(scores))
            assert trend.slope == 0.0
            assert trend.direction == "stable"
            assert trend.significant is False
            assert trend.run_count == len(scores)

    def test_exactly_minimum_runs_is_evaluated(self):
        trend = _compute_trend([90.0, 80.0, 70.0])
        assert trend.run_count == MIN_TREND_RUNS
        assert trend.direction == "worsening"

    def test_threshold_is_configurable(self):
        scores = [80.0, 79.5, 79.0, 78.5]  # -0.5 per run
        assert _compute_trend(scores).direction == "stable"
        assert _compute_trend(scores, threshold=0.1).direction == "worsening"

    def test_negative_threshold_is_rejected(self):
        with pytest.raises(ValueError, match="non-negative"):
            _compute_trend([80.0, 79.0, 78.0], threshold=-1.0)

    def test_noisy_series_uses_all_points_not_endpoints(self):
        # Endpoints are equal, but the bulk of the series declines, so an
        # endpoint-difference implementation would wrongly report "stable".
        trend = _compute_trend([80.0, 70.0, 68.0, 66.0, 80.0])
        assert trend.slope < 0
        assert trend.run_count == 5

    def test_slope_exactly_at_threshold_is_stable(self):
        # slope is exactly +1.0; classification uses strict >, so this is noise.
        trend = _compute_trend([0.0, 1.0, 2.0])
        assert trend.slope == pytest.approx(DEFAULT_TREND_THRESHOLD)
        assert trend.direction == "stable"
        assert trend.significant is False

    def test_slope_exactly_at_negative_threshold_is_stable(self):
        trend = _compute_trend([2.0, 1.0, 0.0])
        assert trend.slope == pytest.approx(-DEFAULT_TREND_THRESHOLD)
        assert trend.direction == "stable"
        assert trend.significant is False

    def test_just_past_threshold_is_a_trend(self):
        trend = _compute_trend([0.0, 1.01, 2.02])
        assert trend.direction == "improving"
        assert trend.significant is True

    def test_reported_slope_reproduces_reported_direction(self):
        # The slope is rounded for display; classification must use the same
        # rounded value, or the output cannot be reproduced from the number.
        trend = _compute_trend([0.0, 1.00002, 2.00004])
        assert trend.slope == pytest.approx(1.0)
        assert trend.direction == "stable"
        assert trend.significant is False

    def test_significant_mirrors_direction(self):
        for scores in (
            [90.0, 85.0, 80.0],
            [80.0, 85.0, 90.0],
            [80.0, 80.0, 80.0],
            [0.0, 1.0, 2.0],
        ):
            trend = _compute_trend(scores)
            assert trend.significant is (trend.direction != "stable")


class TestGetStatisticsTrend:
    """Integration tests for the trend key on get_statistics()."""

    def _tracker(self, tmp_path):
        return RegressionTracker(db_path=tmp_path / "tracking.db")

    def test_declining_test_reports_worsening(self, tmp_path):
        tracker = self._tracker(tmp_path)
        # Stored oldest first, so the newest run is the worst.
        for score in (90.0, 85.0, 80.0, 75.0, 70.0):
            tracker.db.store_result(test_name="checkout", score=score, passed=True)

        stats = tracker.get_statistics("checkout", days=30)

        # Guards against timestamp collisions silently collapsing the history.
        assert stats["total_runs"] == 5
        trend = stats["score"]["trend"]
        assert trend["direction"] == "worsening"
        assert trend["slope"] < 0
        assert trend["significant"] is True
        assert trend["run_count"] == 5

    def test_improving_test_reports_improving(self, tmp_path):
        tracker = self._tracker(tmp_path)
        for score in (70.0, 75.0, 80.0, 85.0, 90.0):
            tracker.db.store_result(test_name="checkout", score=score, passed=True)

        trend = tracker.get_statistics("checkout", days=30)["score"]["trend"]
        assert trend["direction"] == "improving"
        assert trend["slope"] > 0

    def test_history_is_read_oldest_first(self, tmp_path):
        """get_test_history returns newest first; an unreversed fit flips the sign."""
        tracker = self._tracker(tmp_path)
        for score in (90.0, 85.0, 80.0, 75.0, 70.0):
            tracker.db.store_result(test_name="checkout", score=score, passed=True)

        history = tracker.db.get_test_history("checkout", 30)
        assert [h["score"] for h in history] == [70.0, 75.0, 80.0, 85.0, 90.0]

        stats = tracker.get_statistics("checkout", days=30)
        assert stats["score"]["current"] == 70.0
        # Same sign as the chronological fit, not the raw newest-first order.
        assert stats["score"]["trend"]["slope"] < 0

    def test_trend_is_additive(self, tmp_path):
        tracker = self._tracker(tmp_path)
        for score in (80.0, 80.0, 80.0):
            tracker.db.store_result(test_name="checkout", score=score, passed=True)

        score_stats = tracker.get_statistics("checkout", days=30)["score"]
        assert set(score_stats) == {"current", "avg", "min", "max", "trend"}
        assert set(score_stats["trend"]) == {
            "slope",
            "direction",
            "significant",
            "run_count",
        }

    def test_single_run_is_stable(self, tmp_path):
        tracker = self._tracker(tmp_path)
        tracker.db.store_result(test_name="checkout", score=80.0, passed=True)

        trend = tracker.get_statistics("checkout", days=30)["score"]["trend"]
        assert trend["direction"] == "stable"
        assert trend["run_count"] == 1

    def test_trend_uses_only_the_most_recent_runs(self, tmp_path):
        tracker = self._tracker(tmp_path)
        scores = [10.0, 20.0, 30.0, 40.0, 50.0]
        scores.extend([90.0 - 2.0 * index for index in range(DEFAULT_TREND_WINDOW)])
        for score in scores:
            tracker.db.store_result(test_name="checkout", score=score, passed=True)

        trend = tracker.get_statistics("checkout", days=30)["score"]["trend"]

        assert trend["run_count"] == DEFAULT_TREND_WINDOW
        assert trend["slope"] == pytest.approx(-2.0)
        assert trend["direction"] == "worsening"

    def test_trend_window_is_configurable(self, tmp_path):
        tracker = self._tracker(tmp_path)
        for score in (90.0, 80.0, 70.0, 60.0, 50.0):
            tracker.db.store_result(test_name="checkout", score=score, passed=True)

        trend = tracker.get_statistics("checkout", trend_window=3)["score"]["trend"]

        assert trend["run_count"] == 3
        assert trend["slope"] == pytest.approx(-10.0)

    def test_non_positive_trend_window_is_rejected(self, tmp_path):
        tracker = self._tracker(tmp_path)
        with pytest.raises(ValueError, match="positive"):
            tracker.get_statistics("checkout", trend_window=0)

    def test_no_history_omits_score_block(self, tmp_path):
        tracker = self._tracker(tmp_path)
        stats = tracker.get_statistics("never-run", days=30)
        assert stats["total_runs"] == 0
        assert "score" not in stats

    def test_score_trend_is_publicly_importable(self):
        from evalview.tracking import ScoreTrend as Exported

        assert Exported is ScoreTrend
        trend = Exported(slope=-3.0, direction="worsening", significant=True, run_count=5)
        assert trend.slope == -3.0


class TestTrendsCommandDisplay:
    """The trend has to reach the terminal, not just the stats dict."""

    def _seed(self, scores):
        from evalview.tracking import RegressionTracker as Tracker

        tracker = Tracker()
        for score in scores:
            tracker.db.store_result(test_name="flow", score=score, passed=True)

    def _run(self, scores):
        from click.testing import CliRunner

        from evalview.commands.trends_cmd import trends

        runner = CliRunner()
        with runner.isolated_filesystem():
            self._seed(scores)
            return runner.invoke(trends, ["--test", "flow"])

    def test_worsening_trend_is_reported(self):
        result = self._run([90.0, 85.0, 80.0, 75.0, 70.0])
        assert result.exit_code == 0
        assert "worsening" in result.output
        assert "-5.00/run" in result.output

    def test_improving_trend_is_reported(self):
        result = self._run([70.0, 75.0, 80.0, 85.0, 90.0])
        assert result.exit_code == 0
        assert "improving" in result.output

    def test_sub_threshold_drift_is_marked_as_below_threshold(self):
        result = self._run([80.0, 79.9, 79.8, 79.7, 79.6])
        assert result.exit_code == 0
        assert "stable" in result.output
        assert "below threshold" in result.output

    def test_trend_suppressed_below_minimum_runs(self):
        result = self._run([80.0, 70.0])
        assert result.exit_code == 0
        assert "Trend:" not in result.output

    def test_trend_reported_even_when_latest_score_is_zero(self):
        result = self._run([60.0, 40.0, 20.0, 0.0])
        assert result.exit_code == 0
        assert "Score:" in result.output
        assert "Current: 0.0" in result.output
        assert "Average: 30.0" in result.output
        assert "Range: 0.0 - 60.0" in result.output
        assert "worsening" in result.output
