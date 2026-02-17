"""Unit tests for project state tracking."""

import json
import tempfile
import shutil
from pathlib import Path
from datetime import datetime, timedelta
import pytest

from evalview.core.project_state import ProjectState, ProjectStateStore


class TestProjectState:
    """Test ProjectState model validation and defaults."""

    def test_default_values(self):
        """Test that ProjectState has correct default values."""
        state = ProjectState()

        assert state.last_snapshot_at is None
        assert state.last_check_at is None
        assert isinstance(state.created_at, datetime)
        assert state.last_check_status is None
        assert state.current_streak == 0
        assert state.longest_streak == 0
        assert state.regression_count == 0
        assert state.total_snapshots == 0
        assert state.total_checks == 0
        assert state.milestones_hit == []
        assert state.conversion_suggestion_shown is False

    def test_milestone_tracking(self):
        """Test that milestones can be added and tracked."""
        state = ProjectState()

        state.milestones_hit.append("streak_5")
        state.milestones_hit.append("streak_10")

        assert len(state.milestones_hit) == 2
        assert "streak_5" in state.milestones_hit
        assert "streak_10" in state.milestones_hit

    def test_serialization_roundtrip(self):
        """Test that ProjectState can be serialized and deserialized."""
        original = ProjectState(
            last_snapshot_at=datetime.now(),
            last_check_at=datetime.now(),
            current_streak=5,
            longest_streak=10,
            regression_count=2,
            total_snapshots=3,
            total_checks=15,
            milestones_hit=["streak_5", "streak_10"],
            conversion_suggestion_shown=True
        )

        # Serialize
        json_str = original.model_dump_json()

        # Deserialize
        data = json.loads(json_str)
        restored = ProjectState.model_validate(data)

        assert restored.current_streak == original.current_streak
        assert restored.longest_streak == original.longest_streak
        assert restored.regression_count == original.regression_count
        assert restored.total_snapshots == original.total_snapshots
        assert restored.total_checks == original.total_checks
        assert restored.milestones_hit == original.milestones_hit
        assert restored.conversion_suggestion_shown == original.conversion_suggestion_shown


class TestProjectStateStore:
    """Test ProjectStateStore file operations."""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for testing."""
        tmpdir = tempfile.mkdtemp()
        yield Path(tmpdir)
        shutil.rmtree(tmpdir)

    def test_load_creates_new_state_if_missing(self, temp_dir):
        """Test that load() creates a new ProjectState if file doesn't exist."""
        store = ProjectStateStore(temp_dir)
        state = store.load()

        assert isinstance(state, ProjectState)
        assert state.current_streak == 0
        assert state.total_checks == 0

    def test_save_and_load(self, temp_dir):
        """Test that state can be saved and loaded."""
        store = ProjectStateStore(temp_dir)

        original = ProjectState(
            current_streak=5,
            longest_streak=10,
            total_checks=15
        )

        store.save(original)

        loaded = store.load()

        assert loaded.current_streak == 5
        assert loaded.longest_streak == 10
        assert loaded.total_checks == 15

    def test_load_handles_corrupted_file(self, temp_dir):
        """Test that load() handles corrupted JSON gracefully."""
        store = ProjectStateStore(temp_dir)

        # Create corrupted state file
        store.state_file.parent.mkdir(parents=True, exist_ok=True)
        store.state_file.write_text("not valid json {]}")

        # Should return new state instead of crashing
        state = store.load()

        assert isinstance(state, ProjectState)
        assert state.current_streak == 0

    def test_update_snapshot(self, temp_dir):
        """Test that update_snapshot() updates state correctly."""
        store = ProjectStateStore(temp_dir)

        state = store.update_snapshot(test_count=3)

        assert state.total_snapshots == 3
        assert state.last_snapshot_at is not None
        assert isinstance(state.last_snapshot_at, datetime)

        # Update again
        state = store.update_snapshot(test_count=2)
        assert state.total_snapshots == 5  # 3 + 2

    def test_update_check_clean(self, temp_dir):
        """Test that update_check() increments streak on clean checks."""
        store = ProjectStateStore(temp_dir)

        # First clean check
        state = store.update_check(has_regressions=False)

        assert state.current_streak == 1
        assert state.longest_streak == 1
        assert state.total_checks == 1
        assert state.regression_count == 0
        assert state.last_check_status == "passed"

        # Second clean check
        state = store.update_check(has_regressions=False)

        assert state.current_streak == 2
        assert state.longest_streak == 2
        assert state.total_checks == 2

    def test_update_check_regression_breaks_streak(self, temp_dir):
        """Test that regression breaks the current streak."""
        store = ProjectStateStore(temp_dir)

        # Build up a streak
        store.update_check(has_regressions=False)
        store.update_check(has_regressions=False)
        store.update_check(has_regressions=False)

        state = store.load()
        assert state.current_streak == 3

        # Regression breaks it
        state = store.update_check(has_regressions=True, status="regression")

        assert state.current_streak == 0
        assert state.longest_streak == 3  # Preserved
        assert state.regression_count == 1
        assert state.last_check_status == "regression"

    def test_milestone_detection(self, temp_dir):
        """Test that milestones are detected and tracked."""
        store = ProjectStateStore(temp_dir)

        # Get to milestone 3
        for _ in range(3):
            store.update_check(has_regressions=False)

        state = store.load()
        assert "streak_3" in state.milestones_hit

        # Get to milestone 5
        for _ in range(2):
            store.update_check(has_regressions=False)

        state = store.load()
        assert "streak_5" in state.milestones_hit
        assert "streak_3" in state.milestones_hit  # Still there

    def test_milestone_not_duplicated(self, temp_dir):
        """Test that milestones are not added twice."""
        store = ProjectStateStore(temp_dir)

        # Hit milestone 5
        for _ in range(5):
            store.update_check(has_regressions=False)

        state = store.load()
        assert state.milestones_hit.count("streak_5") == 1

        # Break streak and rebuild to 5
        store.update_check(has_regressions=True)
        for _ in range(5):
            store.update_check(has_regressions=False)

        state = store.load()
        # Should still be only one instance of streak_5
        assert state.milestones_hit.count("streak_5") == 1

    def test_longest_streak_updated(self, temp_dir):
        """Test that longest_streak is updated when exceeded."""
        store = ProjectStateStore(temp_dir)

        # Build streak to 5
        for _ in range(5):
            store.update_check(has_regressions=False)

        state = store.load()
        assert state.longest_streak == 5

        # Break and rebuild to 3 (shouldn't update longest)
        store.update_check(has_regressions=True)
        for _ in range(3):
            store.update_check(has_regressions=False)

        state = store.load()
        assert state.longest_streak == 5  # Still 5

        # Build to 10 (should update)
        for _ in range(7):
            store.update_check(has_regressions=False)

        state = store.load()
        assert state.longest_streak == 10

    def test_days_since_last_check(self, temp_dir):
        """Test calculation of days since last check."""
        store = ProjectStateStore(temp_dir)

        # No checks yet
        assert store.days_since_last_check() is None

        # Add a check
        store.update_check(has_regressions=False)

        # Should be 0 days
        assert store.days_since_last_check() == 0

        # Manually set to yesterday
        state = store.load()
        state.last_check_at = datetime.now() - timedelta(days=1)
        store.save(state)

        assert store.days_since_last_check() == 1

        # Manually set to 7 days ago
        state = store.load()
        state.last_check_at = datetime.now() - timedelta(days=7)
        store.save(state)

        assert store.days_since_last_check() == 7

    def test_is_first_snapshot(self, temp_dir):
        """Test detection of first snapshot."""
        store = ProjectStateStore(temp_dir)

        assert store.is_first_snapshot() is True

        store.update_snapshot(test_count=1)

        assert store.is_first_snapshot() is False

    def test_is_first_check(self, temp_dir):
        """Test detection of first check."""
        store = ProjectStateStore(temp_dir)

        assert store.is_first_check() is True

        store.update_check(has_regressions=False)

        assert store.is_first_check() is False

    def test_mark_conversion_shown(self, temp_dir):
        """Test marking conversion suggestion as shown."""
        store = ProjectStateStore(temp_dir)

        state = store.load()
        assert state.conversion_suggestion_shown is False

        store.mark_conversion_shown()

        state = store.load()
        assert state.conversion_suggestion_shown is True


class TestStreakEdgeCases:
    """Test edge cases in streak tracking logic."""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for testing."""
        tmpdir = tempfile.mkdtemp()
        yield Path(tmpdir)
        shutil.rmtree(tmpdir)

    def test_negative_streak_impossible(self, temp_dir):
        """Test that streak never goes negative."""
        store = ProjectStateStore(temp_dir)

        # Even with multiple regressions
        store.update_check(has_regressions=True)
        store.update_check(has_regressions=True)
        store.update_check(has_regressions=True)

        state = store.load()
        assert state.current_streak == 0
        assert state.current_streak >= 0

    def test_large_streak_no_overflow(self, temp_dir):
        """Test that large streaks don't cause overflow."""
        store = ProjectStateStore(temp_dir)

        # Manually set very large streak
        state = ProjectState(
            current_streak=999999,
            longest_streak=999999,
            total_checks=999999
        )
        store.save(state)

        # Increment
        state = store.update_check(has_regressions=False)

        assert state.current_streak == 1000000
        assert state.longest_streak == 1000000

    def test_alternating_checks(self, temp_dir):
        """Test alternating clean/regression pattern."""
        store = ProjectStateStore(temp_dir)

        for i in range(10):
            has_regression = (i % 2 == 0)
            store.update_check(has_regressions=has_regression)

        state = store.load()

        # Should end on streak of 1 (last was clean since i=9, odd)
        assert state.current_streak == 1
        # Longest streak should be 1 (always broken by next check)
        assert state.longest_streak == 1
        # 5 regressions, 5 clean
        assert state.regression_count == 5
        assert state.total_checks == 10
