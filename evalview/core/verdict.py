"""Release verdict layer — the single line humans read after `evalview check`.

A Verdict is a human-actionable decision derived from the existing diff,
quarantine, drift, and cost signals. It answers one question:

    "Should I ship this change?"

The four tiers are:
    - SAFE_TO_SHIP:        all clean, no quarantine concerns
    - SHIP_WITH_QUARANTINE: clean except for known-flaky quarantined tests
    - INVESTIGATE:         soft signals (drift, cost spike, stale quarantine)
                           that warrant a look but don't block
    - BLOCK_RELEASE:       a real regression was caught — do not ship

This module is pure: no I/O, no console output. The `render_verdict_panel`
helper is a convenience for Rich-based display but can be ignored by callers
that want to format verdicts themselves (JSON output, PR comments, etc.).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, List, Optional, Tuple


class Verdict(Enum):
    """Ship-readiness decision derived from all check signals.

    Ordered from best to worst — a higher-severity verdict never demotes
    to a lower one during derivation.
    """

    SAFE_TO_SHIP = "safe_to_ship"
    SHIP_WITH_QUARANTINE = "ship_with_quarantine"
    INVESTIGATE = "investigate"
    BLOCK_RELEASE = "block_release"

    @property
    def rank(self) -> int:
        return {
            Verdict.SAFE_TO_SHIP: 0,
            Verdict.SHIP_WITH_QUARANTINE: 1,
            Verdict.INVESTIGATE: 2,
            Verdict.BLOCK_RELEASE: 3,
        }[self]


@dataclass
class VerdictSignals:
    """Inputs to `compute_verdict`. Structured so new signals can be added
    without breaking the call sites.

    Populate whatever you have; unknown signals default to benign values.
    """

    # Diff outcomes — tuple of (test_name, diff_status_value).
    # diff_status_value is the string form of DiffStatus (e.g. "regression").
    test_statuses: List[Tuple[str, str]] = field(default_factory=list)

    # Which test names are quarantined right now (stale or fresh).
    quarantined_tests: List[str] = field(default_factory=list)

    # Subset of `quarantined_tests` whose review window has expired.
    stale_quarantined_tests: List[str] = field(default_factory=list)

    # Aggregate cost delta vs baseline as a ratio (0.14 == +14%).
    # None when no baseline cost is available.
    cost_delta_ratio: Optional[float] = None

    # Highest drift confidence observed across tests: one of
    # "high", "medium", "low", "insufficient_history", or None.
    drift_confidence: Optional[str] = None

    # True if any test shows a sustained downward drift trend.
    drift_is_downward: bool = False

    # Tests that failed to execute (connection errors, timeouts).
    execution_failures: int = 0


# Tunables — kept at module scope so they're easy to surface in config later.
# Values are deliberately conservative: we prefer to be noisy and correct
# rather than silent and wrong.
_COST_SPIKE_RATIO = 0.10         # +10% triggers INVESTIGATE
_BLOCKING_STATUSES = {"regression", "contract_drift"}
_SOFT_STATUSES = {"tools_changed", "output_changed"}


def compute_verdict(signals: VerdictSignals) -> Tuple[Verdict, List[str]]:
    """Derive a ship/don't-ship verdict from the collected signals.

    Returns:
        A (verdict, reasons) tuple. `reasons` is a list of short human-readable
        strings explaining why that verdict was chosen — these are the bullet
        points shown under the headline on the CLI and in the PR comment.

    Derivation rules (in order, highest wins):
        1. Any execution failure or unquarantined regression → BLOCK_RELEASE.
        2. Any unquarantined TOOLS_CHANGED / OUTPUT_CHANGED → INVESTIGATE.
        3. High-confidence downward drift → INVESTIGATE (flips a would-be
           SAFE_TO_SHIP into a "look before you ship").
        4. Cost delta > +10% → INVESTIGATE.
        5. Stale quarantined tests → INVESTIGATE (governance signal: your
           quarantine folder is rotting).
        6. Clean but with active quarantined tests → SHIP_WITH_QUARANTINE.
        7. Otherwise → SAFE_TO_SHIP.
    """
    reasons: List[str] = []
    verdict = Verdict.SAFE_TO_SHIP

    def bump(to: Verdict, reason: str) -> None:
        nonlocal verdict
        if to.rank > verdict.rank:
            verdict = to
        reasons.append(reason)

    quarantined = set(signals.quarantined_tests)

    if signals.execution_failures > 0:
        bump(
            Verdict.BLOCK_RELEASE,
            f"{signals.execution_failures} test(s) failed to execute",
        )

    # Walk through diff statuses once, classifying each.
    blocking_tests: List[str] = []
    soft_tests: List[str] = []
    quarantined_failures: List[str] = []

    for name, status in signals.test_statuses:
        s = (status or "").lower()
        is_quarantined = name in quarantined

        if s in _BLOCKING_STATUSES:
            if is_quarantined:
                quarantined_failures.append(name)
            else:
                blocking_tests.append(name)
        elif s in _SOFT_STATUSES:
            if is_quarantined:
                quarantined_failures.append(name)
            else:
                soft_tests.append(name)

    if blocking_tests:
        preview = ", ".join(blocking_tests[:3])
        more = "" if len(blocking_tests) <= 3 else f" (+{len(blocking_tests) - 3} more)"
        bump(
            Verdict.BLOCK_RELEASE,
            f"{len(blocking_tests)} regression(s): {preview}{more}",
        )

    if soft_tests:
        preview = ", ".join(soft_tests[:3])
        more = "" if len(soft_tests) <= 3 else f" (+{len(soft_tests) - 3} more)"
        bump(
            Verdict.INVESTIGATE,
            f"{len(soft_tests)} test(s) changed behavior: {preview}{more}",
        )

    if signals.drift_is_downward and signals.drift_confidence == "high":
        bump(
            Verdict.INVESTIGATE,
            "High-confidence downward drift detected — may indicate silent model regression",
        )

    if signals.cost_delta_ratio is not None and signals.cost_delta_ratio > _COST_SPIKE_RATIO:
        pct = int(round(signals.cost_delta_ratio * 100))
        bump(Verdict.INVESTIGATE, f"Cost up {pct}% vs baseline")

    if signals.stale_quarantined_tests:
        n = len(signals.stale_quarantined_tests)
        bump(
            Verdict.INVESTIGATE,
            f"{n} quarantined test(s) stale — review overdue",
        )

    # Quarantine-only failures: the suite is effectively clean but we want
    # to say so explicitly so the user knows the quarantine is doing work.
    if (
        verdict.rank <= Verdict.SHIP_WITH_QUARANTINE.rank
        and (quarantined_failures or (quarantined and not blocking_tests and not soft_tests))
    ):
        bump(
            Verdict.SHIP_WITH_QUARANTINE,
            f"{len(quarantined)} quarantined test(s) not blocking release",
        )

    if not reasons:
        reasons.append("All tests passed, no drift, cost stable")

    return verdict, reasons


# ───────────────────────── presentation helpers ─────────────────────────

_HEADLINE = {
    Verdict.SAFE_TO_SHIP: ("✅  SAFE TO SHIP", "green"),
    Verdict.SHIP_WITH_QUARANTINE: ("⚠️   SHIP WITH QUARANTINE", "yellow"),
    Verdict.INVESTIGATE: ("🔍  INVESTIGATE BEFORE SHIPPING", "yellow"),
    Verdict.BLOCK_RELEASE: ("🛑  BLOCK RELEASE", "red"),
}


def headline_for(verdict: Verdict) -> Tuple[str, str]:
    """Return (display text, rich color) for a verdict headline."""
    return _HEADLINE[verdict]


def render_verdict_panel(
    verdict: Verdict,
    reasons: List[str],
    console: Any,
) -> None:
    """Render the verdict as a Rich panel. Caller supplies the console so
    this module doesn't need to import Rich at import time.
    """
    from rich.panel import Panel

    text, color = headline_for(verdict)
    body_lines = [f"[bold {color}]{text}[/bold {color}]", ""]
    for reason in reasons:
        body_lines.append(f"  • {reason}")
    panel = Panel(
        "\n".join(body_lines),
        border_style=color,
        padding=(1, 2),
    )
    console.print()
    console.print(panel)
    console.print()


def verdict_to_dict(verdict: Verdict, reasons: List[str]) -> dict:
    """Structured form for --json output and PR comments."""
    text, _ = headline_for(verdict)
    return {
        "verdict": verdict.value,
        "headline": text.strip(),
        "reasons": list(reasons),
    }
