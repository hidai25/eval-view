"""Tests for evalview/core/drift_tracker.py."""

import json
import shutil
import tempfile
from pathlib import Path

import pytest

from evalview.core.diff import DiffStatus, TraceDiff, OutputDiff


def _make_diff(similarity: float, status: DiffStatus = DiffStatus.PASSED) -> TraceDiff:
    """Helper: create a minimal TraceDiff with a given output similarity."""
    output_diff = OutputDiff(
        similarity=similarity,
        golden_preview="golden",
        actual_preview="actual",
        diff_lines=[],
        severity=status,
    )
    return TraceDiff(
        test_name="test",
        has_differences=(status != DiffStatus.PASSED),
        tool_diffs=[],
        output_diff=output_diff,
        score_diff=0.0,
        latency_diff=0.0,
        overall_severity=status,
    )


class TestComputeSlope:
    """Unit tests for the OLS slope function."""

    def test_perfect_decline(self):
        from evalview.core.drift_tracker import _compute_slope
        vals = [0.95, 0.93, 0.91, 0.89]  # -0.02 per step
        slope = _compute_slope(vals)
        assert slope == pytest.approx(-0.02, abs=1e-9)

    def test_flat_sequence(self):
        from evalview.core.drift_tracker import _compute_slope
        vals = [0.90, 0.90, 0.90, 0.90]
        assert _compute_slope(vals) == pytest.approx(0.0)

    def test_perfect_rise(self):
        from evalview.core.drift_tracker import _compute_slope
        vals = [0.80, 0.85, 0.90, 0.95]
        slope = _compute_slope(vals)
        assert slope > 0.0

    def test_single_value_returns_zero(self):
        from evalview.core.drift_tracker import _compute_slope
        assert _compute_slope([0.9]) == 0.0

    def test_empty_returns_zero(self):
        from evalview.core.drift_tracker import _compute_slope
        assert _compute_slope([]) == 0.0

    def test_differs_from_naive_endpoint_diff(self):
        """OLS must use all data points, not just start/end."""
        from evalview.core.drift_tracker import _compute_slope
        # Noisy: goes down then recovers — endpoint diff ≈ 0
        vals = [0.95, 0.70, 0.95, 0.89]
        naive = (vals[-1] - vals[0]) / len(vals)
        ols = _compute_slope(vals)
        assert naive != pytest.approx(ols), "OLS must differ from naive endpoint diff on noisy data"

    def test_two_values(self):
        from evalview.core.drift_tracker import _compute_slope
        vals = [0.90, 0.80]
        slope = _compute_slope(vals)
        assert slope == pytest.approx(-0.10, abs=1e-9)


class TestDriftTrackerRecordAndLoad:
    """Tests for record_check() and _load_recent()."""

    @pytest.fixture
    def tmp_dir(self):
        d = tempfile.mkdtemp()
        yield Path(d)
        shutil.rmtree(d)

    def test_record_creates_history_file(self, tmp_dir):
        from evalview.core.drift_tracker import DriftTracker
        tracker = DriftTracker(base_path=tmp_dir)
        tracker.record_check("my-test", _make_diff(0.95))
        assert tracker.history_path.exists()

    def test_record_appends_entry(self, tmp_dir):
        from evalview.core.drift_tracker import DriftTracker
        tracker = DriftTracker(base_path=tmp_dir)
        tracker.record_check("my-test", _make_diff(0.95))
        tracker.record_check("my-test", _make_diff(0.93))
        history = tracker.get_test_history("my-test")
        assert len(history) == 2

    def test_record_stores_correct_fields(self, tmp_dir):
        from evalview.core.drift_tracker import DriftTracker
        tracker = DriftTracker(base_path=tmp_dir)
        tracker.record_check("my-test", _make_diff(0.91, DiffStatus.OUTPUT_CHANGED))
        history = tracker.get_test_history("my-test")
        entry = history[0]
        assert entry["test"] == "my-test"
        assert entry["status"] == "output_changed"
        assert entry["output_similarity"] == pytest.approx(0.91, abs=0.001)

    def test_filters_by_test_name(self, tmp_dir):
        from evalview.core.drift_tracker import DriftTracker
        tracker = DriftTracker(base_path=tmp_dir)
        tracker.record_check("test-a", _make_diff(0.95))
        tracker.record_check("test-b", _make_diff(0.80))
        tracker.record_check("test-a", _make_diff(0.93))
        history_a = tracker.get_test_history("test-a")
        assert len(history_a) == 2
        assert all(e["test"] == "test-a" for e in history_a)

    def test_history_returned_newest_first(self, tmp_dir):
        from evalview.core.drift_tracker import DriftTracker
        tracker = DriftTracker(base_path=tmp_dir)
        for sim in [0.95, 0.93, 0.91]:
            tracker.record_check("my-test", _make_diff(sim))
        history = tracker.get_test_history("my-test")
        # newest first — last recorded similarity is 0.91
        assert history[0]["output_similarity"] == pytest.approx(0.91, abs=0.001)

    def test_no_history_returns_empty(self, tmp_dir):
        from evalview.core.drift_tracker import DriftTracker
        tracker = DriftTracker(base_path=tmp_dir)
        assert tracker.get_test_history("nonexistent") == []

    def test_handles_missing_history_file_gracefully(self, tmp_dir):
        from evalview.core.drift_tracker import DriftTracker
        tracker = DriftTracker(base_path=tmp_dir)
        # No file written — should not raise
        result = tracker.detect_gradual_drift("any-test")
        assert result is None

    def test_skips_malformed_jsonl_lines(self, tmp_dir):
        from evalview.core.drift_tracker import DriftTracker
        tracker = DriftTracker(base_path=tmp_dir)
        tracker.history_path.parent.mkdir(parents=True, exist_ok=True)
        with open(tracker.history_path, "w") as f:
            f.write('{"test": "my-test", "output_similarity": 0.95}\n')
            f.write("this is not json\n")
            f.write('{"test": "my-test", "output_similarity": 0.93}\n')
        history = tracker.get_test_history("my-test")
        assert len(history) == 2  # malformed line skipped


class TestDriftDetection:
    """Tests for detect_gradual_drift()."""

    @pytest.fixture
    def tmp_dir(self):
        d = tempfile.mkdtemp()
        yield Path(d)
        shutil.rmtree(d)

    def test_detects_clear_decline(self, tmp_dir):
        from evalview.core.drift_tracker import DriftTracker
        tracker = DriftTracker(base_path=tmp_dir)
        # Slope ≈ -0.04/step — well below the default -0.02 threshold
        for sim in [0.95, 0.91, 0.87, 0.83, 0.79]:
            tracker.record_check("my-test", _make_diff(sim))
        warning = tracker.detect_gradual_drift("my-test")
        assert warning is not None
        assert "declining" in warning.lower()

    def test_no_warning_for_stable(self, tmp_dir):
        from evalview.core.drift_tracker import DriftTracker
        tracker = DriftTracker(base_path=tmp_dir)
        for sim in [0.95, 0.95, 0.94, 0.95, 0.95]:
            tracker.record_check("my-test", _make_diff(sim))
        assert tracker.detect_gradual_drift("my-test") is None

    def test_no_warning_for_rising(self, tmp_dir):
        from evalview.core.drift_tracker import DriftTracker
        tracker = DriftTracker(base_path=tmp_dir)
        for sim in [0.85, 0.88, 0.91, 0.94, 0.97]:
            tracker.record_check("my-test", _make_diff(sim))
        assert tracker.detect_gradual_drift("my-test") is None

    def test_requires_at_least_3_data_points(self, tmp_dir):
        from evalview.core.drift_tracker import DriftTracker
        tracker = DriftTracker(base_path=tmp_dir)
        tracker.record_check("my-test", _make_diff(0.95))
        tracker.record_check("my-test", _make_diff(0.80))
        # Only 2 points — not enough for reliable trend
        assert tracker.detect_gradual_drift("my-test") is None

    def test_custom_slope_threshold(self, tmp_dir):
        from evalview.core.drift_tracker import DriftTracker
        tracker = DriftTracker(base_path=tmp_dir)
        # Slope ≈ -0.005/step — below default (-0.02) but above -0.001
        for sim in [0.95, 0.945, 0.94, 0.935, 0.93]:
            tracker.record_check("my-test", _make_diff(sim))
        # With tight threshold: should flag
        assert tracker.detect_gradual_drift("my-test", slope_threshold=-0.001) is not None
        # With default threshold: should not flag
        assert tracker.detect_gradual_drift("my-test", slope_threshold=-0.02) is None


class TestPruning:
    """Tests for _MAX_HISTORY_ENTRIES pruning."""

    @pytest.fixture
    def tmp_dir(self):
        d = tempfile.mkdtemp()
        yield Path(d)
        shutil.rmtree(d)

    def test_prunes_when_over_limit(self, tmp_dir, monkeypatch):
        from evalview.core import drift_tracker as dt_module
        from evalview.core.drift_tracker import DriftTracker

        monkeypatch.setattr(dt_module, "_MAX_HISTORY_ENTRIES", 5)
        tracker = DriftTracker(base_path=tmp_dir)

        # Write 8 entries — should be pruned to 5
        for i in range(8):
            tracker.record_check("my-test", _make_diff(0.90))

        with open(tracker.history_path) as f:
            lines = [l for l in f.readlines() if l.strip()]
        assert len(lines) == 5

    def test_does_not_prune_under_limit(self, tmp_dir, monkeypatch):
        from evalview.core import drift_tracker as dt_module
        from evalview.core.drift_tracker import DriftTracker

        monkeypatch.setattr(dt_module, "_MAX_HISTORY_ENTRIES", 100)
        tracker = DriftTracker(base_path=tmp_dir)

        for i in range(10):
            tracker.record_check("my-test", _make_diff(0.90))

        with open(tracker.history_path) as f:
            lines = [l for l in f.readlines() if l.strip()]
        assert len(lines) == 10
