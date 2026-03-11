"""Playful, personality-driven messages for EvalView CLI.

This module provides the "soul" of EvalView - making regression detection
feel less like a chore and more like a helpful colleague checking in.

Philosophy: "Serious about regressions, playful about everything else"
"""

import os
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

# Demo-specific messages — used when EVALVIEW_DEMO=1
# Phase 1 (snapshot): clean and purposeful, no randomness
DEMO_SNAPSHOT_MESSAGE = "🔍 Locking in baseline behavior..."

# Phase 2 (check): the "moment of truth" — relatable for any dev who's
# ever held their breath after a model swap
DEMO_CHECK_MESSAGES: List[str] = [
    "🔍 Moment of truth — what did the new model change?",
    "🔍 Let's see what slipped through...",
    "🔍 Running the new model through its paces...",
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
    """Get a checking message.

    When EVALVIEW_DEMO_PHASE is set, returns a demo-specific message
    instead of a random one, preserving narrative consistency.

    Returns:
        A friendly status message for diff operations
    """
    phase = os.environ.get("EVALVIEW_DEMO_PHASE")
    if phase == "snapshot":
        return DEMO_SNAPSHOT_MESSAGE
    if phase == "check":
        return random.choice(DEMO_CHECK_MESSAGES)
    return random.choice(CHECKING_MESSAGES)


def get_random_clean_check_message() -> str:
    """Get a random clean check message.

    Returns:
        A celebratory message for clean checks
    """
    return random.choice(CLEAN_CHECK_MESSAGES)


# Monitor messages (rotate for long-running sessions)
MONITOR_START_MESSAGES: List[str] = [
    "👁️  Monitor active — watching for regressions...",
    "👁️  Standing guard over your agent...",
    "👁️  Continuous regression detection started...",
]

MONITOR_CYCLE_MESSAGES: List[str] = [
    "🔄 Running check cycle...",
    "🔄 Checking for drift...",
    "🔄 Verifying agent behavior...",
    "🔄 Regression sweep...",
]

MONITOR_CLEAN_MESSAGES: List[str] = [
    "✅ All clear",
    "✅ No regressions",
    "✅ Agent stable",
    "✅ Looking good",
]


def get_random_monitor_start_message() -> str:
    """Get a random monitor startup message."""
    return random.choice(MONITOR_START_MESSAGES)


def get_random_monitor_cycle_message() -> str:
    """Get a random monitor cycle message."""
    return random.choice(MONITOR_CYCLE_MESSAGES)


def get_random_monitor_clean_message() -> str:
    """Get a random monitor clean message."""
    return random.choice(MONITOR_CLEAN_MESSAGES)


def get_error_message(error_type: str) -> str:
    """Get a helpful error message.

    Args:
        error_type: Type of error (e.g., "no_snapshots", "no_tests")

    Returns:
        A helpful, empathetic error message
    """
    return ERROR_MESSAGES.get(error_type, "⚠️ Something went wrong.")
