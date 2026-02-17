"""Unit tests for message generation and rotation."""

import pytest
from evalview.core.messages import (
    get_random_checking_message,
    get_random_clean_check_message,
    get_error_message,
    CHECKING_MESSAGES,
    CLEAN_CHECK_MESSAGES,
    ERROR_MESSAGES
)


class TestMessageGeneration:
    """Test message generation functions."""

    def test_random_checking_message_returns_valid(self):
        """Test that random checking message is from the list."""
        msg = get_random_checking_message()

        assert msg in CHECKING_MESSAGES
        assert isinstance(msg, str)
        assert len(msg) > 0

    def test_random_clean_check_message_returns_valid(self):
        """Test that random clean message is from the list."""
        msg = get_random_clean_check_message()

        assert msg in CLEAN_CHECK_MESSAGES
        assert isinstance(msg, str)
        assert len(msg) > 0

    def test_checking_messages_rotation(self):
        """Test that checking messages rotate (eventually get all variants)."""
        seen_messages = set()

        # Call many times to collect all variants
        for _ in range(100):
            msg = get_random_checking_message()
            seen_messages.add(msg)

        # Should have seen all messages after 100 calls (statistically)
        assert len(seen_messages) >= len(CHECKING_MESSAGES) - 1  # Allow 1 miss

    def test_clean_check_messages_rotation(self):
        """Test that clean check messages rotate."""
        seen_messages = set()

        for _ in range(100):
            msg = get_random_clean_check_message()
            seen_messages.add(msg)

        assert len(seen_messages) >= len(CLEAN_CHECK_MESSAGES) - 1

    def test_get_error_message_known_key(self):
        """Test getting an error message with a known key."""
        msg = get_error_message("no_snapshots")

        assert "No baseline found" in msg
        assert isinstance(msg, str)

    def test_get_error_message_unknown_key(self):
        """Test getting an error message with unknown key returns default."""
        msg = get_error_message("unknown_error_type")

        assert isinstance(msg, str)
        assert len(msg) > 0  # Should return default message


class TestMessageContent:
    """Test that messages have appropriate content."""

    def test_checking_messages_have_emoji(self):
        """Test that checking messages include emoji for visual appeal."""
        for msg in CHECKING_MESSAGES:
            assert any(emoji in msg for emoji in ["🔍"])

    def test_clean_check_messages_have_positive_emoji(self):
        """Test that success messages include positive emoji."""
        positive_emoji = ["✨", "🎉", "💚", "🏅", "👌"]
        for msg in CLEAN_CHECK_MESSAGES:
            assert any(emoji in msg for emoji in positive_emoji)

    def test_checking_messages_variety(self):
        """Test that checking messages have different wording."""
        # All should be unique
        assert len(CHECKING_MESSAGES) == len(set(CHECKING_MESSAGES))

        # Should have some variety in wording
        assert not all("Comparing" in msg for msg in CHECKING_MESSAGES)

    def test_clean_check_messages_variety(self):
        """Test that clean check messages have variety."""
        assert len(CLEAN_CHECK_MESSAGES) == len(set(CLEAN_CHECK_MESSAGES))

        # Different tone/style
        assert not all("All clean" in msg for msg in CLEAN_CHECK_MESSAGES)

    def test_error_messages_are_helpful(self):
        """Test that error messages provide actionable guidance."""
        for key, msg in ERROR_MESSAGES.items():
            # Should have emoji for visual distinction
            assert any(c in msg for c in ["🤔", "🤷", "😬", "⚠️", "❌"])

            # Should provide guidance (has command or action)
            # At least some should mention commands

        # Specific checks
        assert "evalview" in ERROR_MESSAGES["no_snapshots"]
        assert "evalview" in ERROR_MESSAGES["no_tests"]

    def test_message_length_reasonable(self):
        """Test that messages aren't too long for terminal display."""
        max_length = 200  # Reasonable terminal width

        for msg in CHECKING_MESSAGES:
            assert len(msg) < max_length

        for msg in CLEAN_CHECK_MESSAGES:
            assert len(msg) < max_length

        for msg in ERROR_MESSAGES.values():
            # Error messages can be slightly longer (have Rich formatting)
            assert len(msg) < max_length + 50


class TestMessageConsistency:
    """Test message consistency and patterns."""

    def test_checking_messages_consistent_prefix(self):
        """Test that checking messages follow consistent pattern."""
        for msg in CHECKING_MESSAGES:
            assert msg.startswith("🔍")

    def test_clean_check_messages_end_with_period(self):
        """Test that clean messages end with punctuation."""
        for msg in CLEAN_CHECK_MESSAGES:
            assert msg.endswith(".") or msg.endswith("!")

    def test_no_duplicate_messages_across_types(self):
        """Test that checking and clean messages don't overlap."""
        checking_set = set(CHECKING_MESSAGES)
        clean_set = set(CLEAN_CHECK_MESSAGES)

        assert len(checking_set & clean_set) == 0  # No overlap

    def test_message_lists_not_empty(self):
        """Test that message lists have content."""
        assert len(CHECKING_MESSAGES) > 0
        assert len(CLEAN_CHECK_MESSAGES) > 0
        assert len(ERROR_MESSAGES) > 0

    def test_message_lists_have_variety(self):
        """Test that lists have enough variety (at least 3 options)."""
        assert len(CHECKING_MESSAGES) >= 3
        assert len(CLEAN_CHECK_MESSAGES) >= 3


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_repeated_calls_dont_crash(self):
        """Test that repeated calls don't cause issues."""
        for _ in range(1000):
            get_random_checking_message()
            get_random_clean_check_message()

        # If we got here, no crashes occurred

    def test_error_message_with_none_key(self):
        """Test that None key doesn't crash."""
        msg = get_error_message(None)
        assert isinstance(msg, str)

    def test_error_message_with_empty_string_key(self):
        """Test that empty string key doesn't crash."""
        msg = get_error_message("")
        assert isinstance(msg, str)
