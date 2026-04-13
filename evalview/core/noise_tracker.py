"""Noise tracking and confirmation gates for alerts.

This module implements three pieces of the "less noisy product" design:

1. **Confirmation gate (n>=2)** — a failure seen in a single cycle is never
   enough to alert a human. `ConfirmationGate` tracks which failures have been
   "pending" for one cycle and promotes them to "confirmed" only when they
   re-fail in the next cycle. A pending failure that self-resolves is counted
   as a suppressed false positive.

2. **Coordinated incident detection** — when multiple tests fail together in
   the same cycle and share a common root cause (model change, runtime
   fingerprint shift, or simply large batch correlation), they collapse into
   a single incident. `detect_coordinated_incident` returns either an
   `Incident` describing the shared cause, or None to fall back to per-test
   line items.

3. **Public noise metric** — the `NoiseStats` dataclass holds the counters we
   persist to `.evalview/noise.jsonl` so `evalview slack-digest` can render
   a "this week: X alerts, Y real (Z% noise)" line with verifiable math.

Every alert is a promise. If 1 in 5 is a false alarm, users start ignoring
all of them. Optimize directly for "every alert that fires is one the user
is glad fired."
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


# The default root a confirmation gate / noise log sits under. Callers can
# override for tests. We deliberately do not import from elsewhere in the
# package to keep this module dependency-free and cheap to unit-test.
_DEFAULT_BASE = Path(".") / ".evalview"
_NOISE_LOG = "noise.jsonl"


# ─────────────────────────── confirmation gate ───────────────────────────


@dataclass
class GateDecision:
    """Result of running a cycle's failures through the confirmation gate.

    Attributes:
        confirmed: Tests that failed this cycle AND the previous cycle.
                   These are the only tests the caller should alert on.
        pending: Tests that failed this cycle for the first time. No alert
                 should fire for these — they wait for the next cycle.
        self_resolved: Tests that were pending from the previous cycle but
                       passed this cycle. These are counted as suppressed
                       false positives in the noise metric.
        carried_forward: Tests that were confirmed in an earlier cycle and
                         are still failing. Not re-alerted (already alerted
                         on first confirmation).
    """

    confirmed: Set[str] = field(default_factory=set)
    pending: Set[str] = field(default_factory=set)
    self_resolved: Set[str] = field(default_factory=set)
    carried_forward: Set[str] = field(default_factory=set)

    @property
    def alerts_to_fire(self) -> Set[str]:
        """The tests whose status should actually page a human this cycle."""
        return set(self.confirmed)


@dataclass
class ConfirmationGate:
    """State machine that suppresses n=1 alerts.

    The gate owns three sets of test names between cycles:

        pending           — failed once, waiting for confirmation
        confirmed_alerted — failed twice in a row, already alerted
        (everything else) — currently passing

    A single call to `evaluate(currently_failing)` advances the machine
    by one cycle and returns a `GateDecision` describing what changed.
    The caller fires alerts only for `decision.confirmed`, and records
    `decision.self_resolved` as suppressed false positives.
    """

    pending: Set[str] = field(default_factory=set)
    confirmed_alerted: Set[str] = field(default_factory=set)

    def evaluate(self, currently_failing: Iterable[str]) -> GateDecision:
        """Advance the gate by one cycle.

        Args:
            currently_failing: Set of test names that failed this cycle.

        Returns:
            GateDecision describing which tests were promoted to confirmed
            (and therefore should alert), which are still pending, and
            which self-resolved without alerting.
        """
        currently = set(currently_failing)

        # Pending tests that passed this cycle → self-resolved (false positive).
        self_resolved = self.pending - currently

        # Previously pending AND still failing → promote to confirmed.
        # These are the ones that actually fire alerts.
        newly_confirmed = self.pending & currently

        # Previously confirmed AND still failing → carry forward silently.
        # We already alerted on the first confirmation; don't re-page.
        still_confirmed = self.confirmed_alerted & currently

        # Currently failing, but neither pending nor previously confirmed →
        # brand-new failures. These become pending for *next* cycle.
        new_pending = currently - self.pending - self.confirmed_alerted

        # Confirmed tests that recovered → drop from confirmed set.
        # They'll trigger the existing recovery-alert path in the caller.
        # (The caller owns the recovery alert; we just clean up state.)
        #
        # We intentionally do NOT track recoveries here — that's the
        # caller's job via its own previously_failing set.

        decision = GateDecision(
            confirmed=newly_confirmed,
            pending=new_pending,
            self_resolved=self_resolved,
            carried_forward=still_confirmed,
        )

        # Update the gate's state for the next cycle.
        self.pending = new_pending
        self.confirmed_alerted = newly_confirmed | still_confirmed

        return decision


# ─────────────────────────── coordinated incidents ───────────────────────────


@dataclass
class Incident:
    """A collapsed view of many correlated failures.

    When 5 tests fail in the same cycle because `gpt-5.1` rolled out,
    we want ONE Slack card that says "5 tests shifted together — likely
    provider update," not five line items that create five separate
    moments of panic.

    Attributes:
        cause: Short human label, e.g. "likely provider update".
        confidence: "low" | "medium" | "high".
        affected: Sorted list of test names involved.
        evidence: Free-form dict of the signals that led to this grouping,
                  surfaced in the Slack card's details section.
    """

    cause: str
    confidence: str
    affected: List[str]
    evidence: Dict[str, Any] = field(default_factory=dict)

    @property
    def headline(self) -> str:
        """Single-sentence summary for Slack/PR cards."""
        n = len(self.affected)
        plural = "s" if n != 1 else ""
        return (
            f"{n} test{plural} shifted together — "
            f"{self.cause} (confidence: {self.confidence})"
        )


def detect_coordinated_incident(
    diffs: List[Tuple[str, Any]],
    min_affected: int = 3,
) -> Optional[Incident]:
    """Detect whether a batch of failures shares a common root cause.

    The collapse rules are deliberately conservative — we'd rather leave
    a real incident uncollapsed than wrongly merge unrelated failures.

    Rules, in order:

      1. If ≥ `min_affected` failing tests all report `model_changed=True`,
         classify as "likely provider update" at HIGH confidence.
      2. If ≥ `min_affected` failing tests share the same runtime model
         fingerprint that differs from baseline, classify as "runtime
         fingerprint shift" at MEDIUM confidence.
      3. If ≥ `min_affected` failing tests exist in the same batch but
         have no shared signal, classify as "correlated batch failure"
         at LOW confidence — still a useful hint that something
         infrastructure-level moved, but we don't claim to know what.

    Returns None if no rule triggers. In that case the caller renders
    per-test line items as usual.
    """
    if len(diffs) < min_affected:
        return None

    # Only look at failing diffs — a coordinated incident is about the
    # failures, not the whole suite.
    from evalview.core.diff import DiffStatus

    failing: List[Tuple[str, Any]] = [
        (name, diff)
        for name, diff in diffs
        if getattr(diff, "overall_severity", None) != DiffStatus.PASSED
    ]

    if len(failing) < min_affected:
        return None

    # Rule 1: declared model change across multiple tests.
    model_changed_tests = [
        name for name, diff in failing if getattr(diff, "model_changed", False)
    ]
    if len(model_changed_tests) >= min_affected:
        return Incident(
            cause="likely provider update",
            confidence="high",
            affected=sorted(model_changed_tests),
            evidence={
                "signal": "model_changed_flag",
                "count": len(model_changed_tests),
                "total_failing": len(failing),
            },
        )

    # Rule 2: shared runtime fingerprint different from baseline.
    fingerprints: Dict[str, List[str]] = {}
    for name, diff in failing:
        fp = getattr(diff, "runtime_model_fingerprint", None) or getattr(
            diff, "actual_model_id", None
        )
        if fp:
            fingerprints.setdefault(str(fp), []).append(name)

    for fp, tests in fingerprints.items():
        if len(tests) >= min_affected:
            return Incident(
                cause="runtime fingerprint shift",
                confidence="medium",
                affected=sorted(tests),
                evidence={
                    "signal": "runtime_fingerprint",
                    "fingerprint": fp,
                    "count": len(tests),
                    "total_failing": len(failing),
                },
            )

    # Rule 3: correlated batch with no shared signal. Only fire if the
    # failing tests are the *majority* of the batch — a random one-test
    # regression in a 100-test suite should not trigger this.
    if len(failing) >= min_affected and len(failing) >= max(
        min_affected, len(diffs) // 2
    ):
        return Incident(
            cause="correlated batch failure",
            confidence="low",
            affected=sorted(name for name, _ in failing),
            evidence={
                "signal": "batch_correlation",
                "failing": len(failing),
                "total": len(diffs),
            },
        )

    return None


# ─────────────────────────── public noise metric ───────────────────────────


@dataclass
class NoiseStats:
    """Rolling counters for the public "noise" metric.

    These are aggregated from `.evalview/noise.jsonl` across a time window
    and rendered in the Slack digest as:

        "This week: 12 alerts fired · 9 real · 3 suppressed (25% noise)"

    Fields:
        alerts_fired: Number of times an alert was actually sent to a
                      notifier (not including suppressed n=1 failures).
        real_alerts: Alerts that stayed failing in a subsequent cycle —
                     "confirmed real" by the gate downstream.
        suppressed: Failures that were pending but self-resolved before
                    ever becoming an alert. The gate saved the user from
                    these.
    """

    alerts_fired: int = 0
    real_alerts: int = 0
    suppressed: int = 0

    @property
    def false_positive_rate(self) -> Optional[float]:
        """Suppressed / (suppressed + alerts_fired). None if no signal yet.

        This is the *pre-alert* false-positive rate — how often a raw
        failure would have paged someone without the confirmation gate.
        A number close to 0 means the agent is stable; a number close
        to 1 means the gate is doing heavy lifting and real alerts are
        rare.
        """
        denom = self.alerts_fired + self.suppressed
        if denom == 0:
            return None
        return self.suppressed / denom

    def format_line(self) -> str:
        """Render as a one-line digest string."""
        fpr = self.false_positive_rate
        if fpr is None:
            return "No alert activity in this window."
        pct = int(round(fpr * 100))
        return (
            f"{self.alerts_fired} alerts fired · "
            f"{self.real_alerts} real · "
            f"{self.suppressed} suppressed ({pct}% noise)"
        )


def record_cycle_noise(
    decision: GateDecision,
    base_path: Optional[Path] = None,
    timestamp: Optional[datetime] = None,
) -> None:
    """Append one cycle's noise counters to `.evalview/noise.jsonl`.

    Called by the monitor loop after every cycle. Each line records:
        - alerts_fired:  number of confirmed failures that fired alerts
        - real_alerts:   same as alerts_fired (every confirmed firing is
                         real by definition of the gate)
        - suppressed:    number of self-resolved pendings this cycle
        - pending_count: number of pendings carried into next cycle
                         (diagnostic only, not part of the public metric)

    Writes are best-effort — a failed write logs a warning and returns.
    The monitor loop must never crash because the noise log is unwritable.
    """
    base = base_path or _DEFAULT_BASE
    path = base / _NOISE_LOG
    ts = timestamp or datetime.now(timezone.utc)

    record = {
        "ts": ts.isoformat(),
        "alerts_fired": len(decision.confirmed),
        "real_alerts": len(decision.confirmed),
        "suppressed": len(decision.self_resolved),
        "pending_count": len(decision.pending),
    }

    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
    except OSError as exc:
        logger.warning("Failed to write noise log: %s", exc)


def load_noise_stats(
    base_path: Optional[Path] = None,
    since: Optional[datetime] = None,
) -> NoiseStats:
    """Read `.evalview/noise.jsonl` and sum counters over a time window.

    Args:
        base_path: Project root containing `.evalview/`. Defaults to CWD.
        since: Only include records newer than this datetime. If None,
               include all records in the file.

    Returns:
        NoiseStats with totals over the window. Zero values if the file
        doesn't exist or is empty — callers must handle that case
        gracefully (the digest shows "no activity" rather than "0% noise",
        which would be misleading).
    """
    base = base_path or _DEFAULT_BASE
    path = base / _NOISE_LOG
    stats = NoiseStats()

    if not path.exists():
        return stats

    try:
        with path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue

                if since is not None:
                    ts_raw = record.get("ts")
                    if ts_raw:
                        try:
                            ts = datetime.fromisoformat(str(ts_raw))
                            if ts.tzinfo is None:
                                ts = ts.replace(tzinfo=timezone.utc)
                            if ts <= since:
                                continue
                        except ValueError:
                            # Malformed timestamp — include by default
                            # rather than silently drop.
                            pass

                stats.alerts_fired += int(record.get("alerts_fired", 0) or 0)
                stats.real_alerts += int(record.get("real_alerts", 0) or 0)
                stats.suppressed += int(record.get("suppressed", 0) or 0)
    except OSError as exc:
        logger.warning("Failed to read noise log: %s", exc)
        return NoiseStats()

    return stats
