"""Unit tests for celebration and delight moments."""

import io
from unittest.mock import patch
import pytest

from evalview.core.celebrations import (
    Celebrations,
    STREAK_MILESTONE_START,
    STREAK_MILESTONE_SMALL,
    STREAK_MILESTONE_MEDIUM,
    STREAK_MILESTONE_LARGE,
    STREAK_MILESTONE_LEGENDARY,
    STREAK_MILESTONE_INCREDIBLE,
    STREAK_BREAK_EMPATHY_THRESHOLD,
    BANNER_WIDTH
)
from evalview.core.project_state import ProjectState


class TestConstants:
    """Test that celebration constants are defined correctly."""

    def test_milestone_constants_defined(self):
        """Test that all milestone constants exist and are positive."""
        assert STREAK_MILESTONE_START == 1
        assert STREAK_MILESTONE_SMALL == 3
        assert STREAK_MILESTONE_MEDIUM == 5
        assert STREAK_MILESTONE_LARGE == 10
        assert STREAK_MILESTONE_LEGENDARY == 25
        assert STREAK_MILESTONE_INCREDIBLE == 50

    def test_milestone_progression(self):
        """Test that milestones are in ascending order."""
        milestones = [
            STREAK_MILESTONE_START,
            STREAK_MILESTONE_SMALL,
            STREAK_MILESTONE_MEDIUM,
            STREAK_MILESTONE_LARGE,
            STREAK_MILESTONE_LEGENDARY,
            STREAK_MILESTONE_INCREDIBLE
        ]

        assert milestones == sorted(milestones)

    def test_empathy_threshold_reasonable(self):
        """Test that empathy threshold is reasonable."""
        assert STREAK_BREAK_EMPATHY_THRESHOLD >= 3
        assert STREAK_BREAK_EMPATHY_THRESHOLD <= 10

    def test_banner_width_reasonable(self):
        """Test that banner width fits terminal."""
        assert BANNER_WIDTH >= 40
        assert BANNER_WIDTH <= 100


class TestFirstSnapshot:
    """Test first snapshot celebration."""

    @patch('sys.stdout', new_callable=io.StringIO)
    def test_first_snapshot_displays_banner(self, mock_stdout):
        """Test that first snapshot shows celebration banner."""
        Celebrations.first_snapshot(test_count=3)

        output = mock_stdout.getvalue()

        assert "BASELINE CAPTURED" in output
        assert "3 test(s)" in output
        assert "ACTIVE" in output
        assert "evalview check" in output

    @patch('sys.stdout', new_callable=io.StringIO)
    def test_first_snapshot_banner_width(self, mock_stdout):
        """Test that banner uses correct width."""
        Celebrations.first_snapshot(test_count=1)

        output = mock_stdout.getvalue()

        assert "=" * BANNER_WIDTH in output

    @patch('sys.stdout', new_callable=io.StringIO)
    def test_first_snapshot_plural_singular(self, mock_stdout):
        """Test correct plural/singular for test count."""
        Celebrations.first_snapshot(test_count=1)
        output = mock_stdout.getvalue()
        assert "1 test(s)" in output

        mock_stdout.truncate(0)
        mock_stdout.seek(0)

        Celebrations.first_snapshot(test_count=5)
        output = mock_stdout.getvalue()
        assert "5 test(s)" in output


class TestCleanCheckStreak:
    """Test streak celebration logic."""

    @patch('sys.stdout', new_callable=io.StringIO)
    def test_streak_1_acknowledgment(self, mock_stdout):
        """Test that streak of 1 shows simple acknowledgment."""
        state = ProjectState(current_streak=1)

        Celebrations.clean_check_streak(state)

        output = mock_stdout.getvalue()
        assert "Streak started" in output or "1" in output

    @patch('sys.stdout', new_callable=io.StringIO)
    def test_streak_3_celebration(self, mock_stdout):
        """Test 3-check streak celebration."""
        state = ProjectState(current_streak=3)

        Celebrations.clean_check_streak(state)

        output = mock_stdout.getvalue()
        assert "3" in output
        assert "roll" in output.lower() or "clean" in output.lower()

    @patch('sys.stdout', new_callable=io.StringIO)
    def test_streak_5_celebration(self, mock_stdout):
        """Test 5-check streak celebration."""
        state = ProjectState(current_streak=5)

        Celebrations.clean_check_streak(state)

        output = mock_stdout.getvalue()
        assert "5" in output
        assert "STREAK" in output.upper()

    @patch('sys.stdout', new_callable=io.StringIO)
    def test_streak_10_celebration(self, mock_stdout):
        """Test 10-check streak celebration with ASCII art."""
        state = ProjectState(current_streak=10)

        Celebrations.clean_check_streak(state)

        output = mock_stdout.getvalue()
        assert "10" in output
        assert "⭐" in output  # ASCII art stars
        assert "Champion" in output or "STREAK" in output

    @patch('sys.stdout', new_callable=io.StringIO)
    def test_streak_25_celebration(self, mock_stdout):
        """Test 25-check streak legendary celebration."""
        state = ProjectState(current_streak=25)

        Celebrations.clean_check_streak(state)

        output = mock_stdout.getvalue()
        assert "25" in output
        assert "LEGENDARY" in output.upper() or "production" in output.lower()

    @patch('sys.stdout', new_callable=io.StringIO)
    def test_streak_50_celebration(self, mock_stdout):
        """Test 50-check streak incredible celebration."""
        state = ProjectState(current_streak=50)

        Celebrations.clean_check_streak(state)

        output = mock_stdout.getvalue()
        assert "50" in output
        assert "INCREDIBLE" in output.upper() or "rock solid" in output.lower()

    @patch('sys.stdout', new_callable=io.StringIO)
    def test_streak_above_50_multiples_of_10(self, mock_stdout):
        """Test that multiples of 10 above 50 are celebrated."""
        state = ProjectState(current_streak=60)

        Celebrations.clean_check_streak(state)

        output = mock_stdout.getvalue()
        assert "60" in output

    @patch('sys.stdout', new_callable=io.StringIO)
    def test_streak_non_milestone_quiet(self, mock_stdout):
        """Test that non-milestone streaks don't print much."""
        state = ProjectState(current_streak=7)

        Celebrations.clean_check_streak(state)

        output = mock_stdout.getvalue()
        # Should be minimal or empty for non-milestone
        assert len(output) < 100 or output.strip() == ""


class TestStreakBroken:
    """Test empathetic streak break messages."""

    @patch('sys.stdout', new_callable=io.StringIO)
    def test_small_streak_break_no_empathy(self, mock_stdout):
        """Test that breaking small streak (<5) doesn't show empathy."""
        state = ProjectState(current_streak=3)

        Celebrations.streak_broken(state, "regression")

        output = mock_stdout.getvalue()
        # Should be empty or minimal for small streaks
        assert len(output) < 50 or output.strip() == ""

    @patch('sys.stdout', new_callable=io.StringIO)
    def test_medium_streak_break_shows_empathy(self, mock_stdout):
        """Test that breaking streak >=5 shows empathy."""
        state = ProjectState(current_streak=7)

        Celebrations.streak_broken(state, "tools_changed")

        output = mock_stdout.getvalue()
        assert "Streak ended" in output or "😔" in output
        assert "7" in output
        assert "tools_changed" in output

    @patch('sys.stdout', new_callable=io.StringIO)
    def test_large_streak_break_shows_encouragement(self, mock_stdout):
        """Test that breaking large streak includes encouragement."""
        state = ProjectState(current_streak=15)

        Celebrations.streak_broken(state, "regression")

        output = mock_stdout.getvalue()
        assert "15" in output
        assert ("It happens" in output or "Fix" in output or "start" in output)

    @patch('sys.stdout', new_callable=io.StringIO)
    def test_streak_break_at_threshold_boundary(self, mock_stdout):
        """Test behavior at empathy threshold boundary."""
        state = ProjectState(current_streak=STREAK_BREAK_EMPATHY_THRESHOLD)

        Celebrations.streak_broken(state, "regression")

        output = mock_stdout.getvalue()
        assert str(STREAK_BREAK_EMPATHY_THRESHOLD) in output


class TestHealthSummary:
    """Test health summary display."""

    @patch('sys.stdout', new_callable=io.StringIO)
    def test_health_summary_displays(self, mock_stdout):
        """Test that health summary shows project statistics."""
        state = ProjectState(
            total_checks=20,
            regression_count=2,
            current_streak=5,
            longest_streak=10
        )

        Celebrations.health_summary(state)

        output = mock_stdout.getvalue()
        assert "Health" in output or "20" in output  # total checks
        assert "5" in output  # current streak

    @patch('sys.stdout', new_callable=io.StringIO)
    def test_health_summary_shows_percentage(self, mock_stdout):
        """Test that health summary includes success percentage."""
        state = ProjectState(
            total_checks=10,
            regression_count=1,  # 90% success
            current_streak=3,
            longest_streak=5
        )

        Celebrations.health_summary(state)

        output = mock_stdout.getvalue()
        # Should show something like 90%
        assert "90" in output or "9" in output


class TestShareableBadge:
    """Test shareable achievement badges."""

    @patch('sys.stdout', new_callable=io.StringIO)
    def test_shareable_badge_displays(self, mock_stdout):
        """Test that shareable badge is offered."""
        Celebrations.shareable_badge(streak=25)

        output = mock_stdout.getvalue()
        assert "25" in output
        assert ("share" in output.lower() or "badge" in output.lower() or "🏆" in output)

    @patch('sys.stdout', new_callable=io.StringIO)
    def test_shareable_badge_includes_github_link(self, mock_stdout):
        """Test that badge includes GitHub link."""
        Celebrations.shareable_badge(streak=50)

        output = mock_stdout.getvalue()
        assert "github" in output.lower() or "evalview" in output.lower()


class TestRegressionGuidance:
    """Test helpful regression guidance."""

    @patch('sys.stdout', new_callable=io.StringIO)
    def test_regression_guidance_helpful(self, mock_stdout):
        """Test that regression guidance is actionable."""
        Celebrations.regression_guidance("Score dropped")

        output = mock_stdout.getvalue()
        assert "evalview" in output.lower() or "snapshot" in output.lower() or "fix" in output.lower()

    @patch('sys.stdout', new_callable=io.StringIO)
    def test_regression_guidance_shows_details(self, mock_stdout):
        """Test that guidance references the diff summary."""
        Celebrations.regression_guidance("Tools changed: +search, -calculator")

        output = mock_stdout.getvalue()
        # Should display or reference the summary
        assert len(output) > 0


class TestFirstCheck:
    """Test first check celebration."""

    @patch('sys.stdout', new_callable=io.StringIO)
    def test_first_check_celebration(self, mock_stdout):
        """Test that first check is celebrated."""
        Celebrations.first_check()

        output = mock_stdout.getvalue()
        assert "first" in output.lower() or "1" in output


class TestEdgeCases:
    """Test edge cases and error conditions."""

    @patch('sys.stdout', new_callable=io.StringIO)
    def test_zero_streak_doesnt_crash(self, mock_stdout):
        """Test that zero streak doesn't cause errors."""
        state = ProjectState(current_streak=0)

        Celebrations.clean_check_streak(state)

        # Should not crash

    @patch('sys.stdout', new_callable=io.StringIO)
    def test_negative_streak_doesnt_crash(self, mock_stdout):
        """Test that negative streak (shouldn't happen) doesn't crash."""
        state = ProjectState(current_streak=-1)

        Celebrations.clean_check_streak(state)

        # Should not crash

    @patch('sys.stdout', new_callable=io.StringIO)
    def test_very_large_streak(self, mock_stdout):
        """Test that very large streaks are handled."""
        state = ProjectState(current_streak=1000)

        Celebrations.clean_check_streak(state)

        output = mock_stdout.getvalue()
        assert "1000" in output

    @patch('sys.stdout', new_callable=io.StringIO)
    def test_zero_tests_snapshot(self, mock_stdout):
        """Test first snapshot with zero tests."""
        Celebrations.first_snapshot(test_count=0)

        output = mock_stdout.getvalue()
        assert "0" in output

    @patch('sys.stdout', new_callable=io.StringIO)
    def test_none_diff_status_doesnt_crash(self, mock_stdout):
        """Test that None diff status doesn't crash."""
        state = ProjectState(current_streak=10)

        Celebrations.streak_broken(state, None)

        # Should not crash


class TestConsistency:
    """Test celebration consistency and tone."""

    @patch('sys.stdout', new_callable=io.StringIO)
    def test_all_milestones_have_emoji(self, mock_stdout):
        """Test that milestone celebrations include emoji."""
        milestones = [1, 3, 5, 10, 25, 50]

        for milestone in milestones:
            mock_stdout.truncate(0)
            mock_stdout.seek(0)

            state = ProjectState(current_streak=milestone)
            Celebrations.clean_check_streak(state)

            output = mock_stdout.getvalue()
            if len(output) > 10:  # Only check if there's output
                # Should have some emoji (any emoji, not specific ones)
                emojis = ["🎯", "🔥", "🌟", "💎", "🚀", "🔄"]
                assert any(char in output for char in emojis)

    @patch('sys.stdout', new_callable=io.StringIO)
    def test_tone_is_encouraging(self, mock_stdout):
        """Test that celebrations have encouraging tone."""
        state = ProjectState(current_streak=10)

        Celebrations.clean_check_streak(state)

        output = mock_stdout.getvalue()
        # Should have positive language
        positive_words = ["great", "beautiful", "nice", "champion", "stable", "good"]
        assert any(word in output.lower() for word in positive_words)

    @patch('sys.stdout', new_callable=io.StringIO)
    def test_streak_break_is_empathetic(self, mock_stdout):
        """Test that streak breaks are empathetic, not harsh."""
        state = ProjectState(current_streak=10)

        Celebrations.streak_broken(state, "regression")

        output = mock_stdout.getvalue()
        # Should be gentle
        harsh_words = ["failed", "wrong", "error", "bad"]
        empathetic_words = ["happens", "fix", "start", "new"]

        assert not any(word in output.lower() for word in harsh_words)
        # At least one empathetic word
        if len(output) > 10:
            assert any(word in output.lower() for word in empathetic_words)
