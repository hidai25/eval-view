"""Project state tracking for EvalView.

Tracks project history, streaks, and milestones to enable:
- Streak tracking and celebrations
- "Since last time" recaps
- Reactivation nudges
- Health score visualization

State is persisted in .evalview/state.json
"""

from datetime import datetime
from pathlib import Path
from typing import Optional, List
from pydantic import BaseModel, Field
import json
import logging

logger = logging.getLogger(__name__)

# Milestone thresholds for streak tracking
MILESTONE_THRESHOLDS = [3, 5, 10, 25, 50, 100]


class ProjectState(BaseModel):
    """Persistent project state for habit formation and progress tracking."""

    # Timestamps
    last_snapshot_at: Optional[datetime] = None
    last_check_at: Optional[datetime] = None
    created_at: datetime = Field(default_factory=datetime.now)

    # Check status
    last_check_status: Optional[str] = None  # "passed", "regression", "tools_changed"

    # Streak tracking (for habit formation)
    current_streak: int = 0  # Consecutive clean checks
    longest_streak: int = 0  # Best streak achieved

    # Aggregate stats
    regression_count: int = 0
    total_snapshots: int = 0
    total_checks: int = 0

    # Milestones (for celebration tracking)
    milestones_hit: List[str] = Field(default_factory=list)  # ["streak_5", "streak_10", etc.]

    # Onboarding
    conversion_suggestion_shown: bool = False


class ProjectStateStore:
    """Manages persistent project state in .evalview/state.json"""

    def __init__(self, base_path: Path = Path(".")):
        """
        Initialize state store.

        Args:
            base_path: Base directory for .evalview (default: current dir)
        """
        self.state_file = base_path / ".evalview" / "state.json"
        self.state_file.parent.mkdir(parents=True, exist_ok=True)

    def load(self) -> ProjectState:
        """Load project state from disk.

        Returns:
            ProjectState instance (creates new if file doesn't exist)
        """
        if not self.state_file.exists():
            return ProjectState()

        try:
            data = json.loads(self.state_file.read_text())
            return ProjectState.model_validate(data)
        except (json.JSONDecodeError, ValueError) as e:
            # Corrupted state file, start fresh
            logger.warning(f"Could not load state file ({e}), creating new state")
            return ProjectState()

    def save(self, state: ProjectState) -> None:
        """Save project state to disk.

        Args:
            state: ProjectState to persist
        """
        try:
            with open(self.state_file, 'w') as f:
                f.write(state.model_dump_json(indent=2, exclude_none=False))
        except IOError as e:
            logger.error(f"Failed to save project state: {e}")
            raise

    def update_snapshot(self, test_count: int = 1) -> ProjectState:
        """Update state after snapshot operation.

        Args:
            test_count: Number of tests snapshotted

        Returns:
            Updated ProjectState
        """
        state = self.load()
        state.last_snapshot_at = datetime.now()
        state.total_snapshots += test_count
        self.save(state)
        return state

    def update_check(self, has_regressions: bool, status: str = "passed") -> ProjectState:
        """Update state after check operation.

        Args:
            has_regressions: Whether regressions were found
            status: Overall check status ("passed", "regression", "tools_changed", etc.)

        Returns:
            Updated ProjectState
        """
        state = self.load()
        state.last_check_at = datetime.now()
        state.total_checks += 1

        if has_regressions:
            state.regression_count += 1
            state.current_streak = 0  # Streak broken
            state.last_check_status = status
        else:
            state.current_streak += 1
            state.last_check_status = "passed"

            # Update longest streak if we beat the record
            if state.current_streak > state.longest_streak:
                state.longest_streak = state.current_streak

            # Track milestones
            milestone = f"streak_{state.current_streak}"
            if state.current_streak in MILESTONE_THRESHOLDS and milestone not in state.milestones_hit:
                state.milestones_hit.append(milestone)

        self.save(state)
        return state

    def mark_conversion_shown(self) -> None:
        """Mark that the conversion suggestion has been shown."""
        state = self.load()
        state.conversion_suggestion_shown = True
        self.save(state)

    def days_since_last_check(self) -> Optional[int]:
        """Calculate days since last check.

        Returns:
            Number of days, or None if no previous check
        """
        state = self.load()
        if state.last_check_at is None:
            return None
        delta = datetime.now() - state.last_check_at
        return delta.days

    def is_first_snapshot(self) -> bool:
        """Check if this is the first snapshot ever.

        Returns:
            True if no snapshots have been taken yet
        """
        state = self.load()
        return state.total_snapshots == 0

    def is_first_check(self) -> bool:
        """Check if this is the first check ever.

        Returns:
            True if no checks have been run yet
        """
        state = self.load()
        return state.total_checks == 0
