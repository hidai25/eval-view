"""Playful, personality-driven messages for EvalView CLI.

This module provides the "soul" of EvalView - making regression detection
feel less like a chore and more like a helpful colleague checking in.

Philosophy: "Serious about regressions, playful about everything else"
"""

import random
from typing import List


# Checking messages (rotate randomly during diff operations)
CHECKING_MESSAGES: List[str] = [
    "🔍 Comparing against your baseline...",
    "🔍 Checking for drift (fingers crossed)...",
    "🔍 Running regression checks...",
    "🔍 Sniffing out changes...",
    "🔍 Looking for what changed...",
]

# Clean check messages (rotate randomly when no regressions found)
CLEAN_CHECK_MESSAGES: List[str] = [
    "✨ All clean! No regressions detected.",
    "🎉 Perfect match! Your agent is stable.",
    "💚 Looking good! Everything matches the baseline.",
    "🏅 Zero regressions. Nice work!",
    "👌 All tests passed! Your agent is behaving beautifully.",
]

# Error status messages (friendly alternatives to dry system messages)
ERROR_MESSAGES = {
    "no_snapshots": "🤔 No baseline found yet. Let's create one: [cyan]evalview snapshot[/cyan]",
    "no_tests": "🤷 No test cases found. Try: [cyan]evalview init[/cyan] or [cyan]evalview demo[/cyan]",
    "snapshot_failed": "😬 Couldn't save snapshot. Check the error above.",
    "check_failed": "⚠️ Check encountered issues. See details above.",
}


def get_random_checking_message() -> str:
    """Get a random checking message.

    Returns:
        A friendly status message for diff operations
    """
    return random.choice(CHECKING_MESSAGES)


def get_random_clean_check_message() -> str:
    """Get a random clean check message.

    Returns:
        A celebratory message for clean checks
    """
    return random.choice(CLEAN_CHECK_MESSAGES)


def get_error_message(error_type: str) -> str:
    """Get a helpful error message.

    Args:
        error_type: Type of error (e.g., "no_snapshots", "no_tests")

    Returns:
        A helpful, empathetic error message
    """
    return ERROR_MESSAGES.get(error_type, "⚠️ Something went wrong.")
