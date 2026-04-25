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
from dataclasses import dataclass, field
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
        confirmed: Tests that failed this cycle AND the previous cycle
                   (or failed once while marked strict). These are the
                   tests the caller should alert on.
        pending: Tests that failed this cycle for the first time. No alert
                 should fire for these — they wait for the next cycle.
        self_resolved: Tests that were pending from the previous cycle but
                       passed this cycle. These are counted as suppressed
                       false positives in the noise metric.
        carried_forward: Tests that were confirmed in an earlier cycle and
                         are still failing. Not re-alerted (already alerted
                         on first confirmation).
        strict_immediate: Tests in the strict bypass set that failed this
                          cycle. They are rolled into `confirmed` so the
                          caller alerts on them, but tracked separately so
                          the caller can log "strict: bypassing gate" and
                          so tests can verify the bypass actually fired.
    """

    confirmed: Set[str] = field(default_factory=set)
    pending: Set[str] = field(default_factory=set)
    self_resolved: Set[str] = field(default_factory=set)
    carried_forward: Set[str] = field(default_factory=set)
    strict_immediate: Set[str] = field(default_factory=set)

    @property
    def alerts_to_fire(self) -> Set[str]:
        """The tests whose status should actually page a human this cycle."""
        return set(self.confirmed) | set(self.strict_immediate)


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

    def evaluate(
        self,
        currently_failing: Iterable[str],
        strict: Optional[Iterable[str]] = None,
    ) -> GateDecision:
        """Advance the gate by one cycle.

        Args:
            currently_failing: Set of test names that failed this cycle.
            strict: Names of tests marked `gate: strict` in their YAML —
                these bypass the confirmation gate entirely. A strict
                test that fails even once alerts immediately, without
                waiting for a second cycle to confirm. Use this for
                safety-critical behaviors (auth, payments, PII, refund
                paths) where a single false positive is cheaper than
                missing a real regression for five minutes.

        Returns:
            GateDecision describing which tests were promoted to confirmed
            (and therefore should alert), which strict tests bypassed the
            gate, which are still pending, and which self-resolved
            without alerting.
        """
        currently = set(currently_failing)
        strict_set = set(strict or ())

        # Strict tests bypass the whole state machine: any strict test
        # failing this cycle alerts immediately. They are NOT recorded
        # in `pending` or `confirmed_alerted`, because those buckets are
        # what the relaxed state machine uses to avoid re-alerting — and
        # strict tests should re-alert every cycle they stay broken so
        # the user can't accidentally forget about them.
        strict_firing = currently & strict_set

        # Clean up the relaxed-mode buckets of any tests that have been
        # promoted to strict since the last cycle. Without this, a test
        # that was previously `pending` and then flipped to `gate: strict`
        # would still be in `self.pending`, and this cycle's
        # `self_resolved = self.pending - relaxed_currently` would wrongly
        # count it as suppressed — even though it's currently firing via
        # the strict bypass. The alert fires correctly either way, but
        # the noise metric has to stay honest. Relaxed-mode bookkeeping
        # only describes relaxed-mode tests.
        self.pending -= strict_set
        self.confirmed_alerted -= strict_set

        # For the relaxed-mode logic below we only consider the subset
        # of failures that are NOT strict. This keeps the buckets clean.
        relaxed_currently = currently - strict_set

        # Pending tests that passed this cycle → self-resolved (false positive).
        self_resolved = self.pending - relaxed_currently

        # Previously pending AND still failing → promote to confirmed.
        # These are the ones that actually fire alerts.
        newly_confirmed = self.pending & relaxed_currently

        # Previously confirmed AND still failing → carry forward silently.
        # We already alerted on the first confirmation; don't re-page.
        still_confirmed = self.confirmed_alerted & relaxed_currently

        # Currently failing, but neither pending nor previously confirmed →
        # brand-new failures. These become pending for *next* cycle.
        new_pending = relaxed_currently - self.pending - self.confirmed_alerted

        decision = GateDecision(
            confirmed=newly_confirmed,
            pending=new_pending,
            self_resolved=self_resolved,
            carried_forward=still_confirmed,
            strict_immediate=strict_firing,
        )

        # Update the gate's state for the next cycle. Strict tests are
        # excluded from the internal buckets so they keep alerting every
        # cycle until they pass.
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
class SuppressedEntry:
    """Per-test summary of unconfirmed failures the gate swallowed.

    The digest uses this to render a visible list of "what got
    suppressed this week" so the user can sanity-check the gate's
    decisions — never hide the signal, just the alert.

    Attributes:
        test_name: Name of the test that self-resolved.
        count: How many times it self-resolved in the window.
        last_seen: ISO-8601 timestamp of the most recent suppression.
    """

    test_name: str
    count: int = 0
    last_seen: Optional[str] = None


@dataclass
class NoiseStats:
    """Rolling counters for the public "noise" metric.

    These are aggregated from `.evalview/noise.jsonl` across a time window
    and rendered in the Slack digest as:

        "This week: 12 alerts fired · 9 real · 3 suppressed (25% noise)"

    Fields:
        alerts_fired: Number of times an alert was actually sent to a
                      notifier (not including suppressed n=1 failures).
        real_alerts: Equal to alerts_fired by definition — every alert
                     that reaches this counter already survived the
                     confirmation gate (or bypassed it via strict mode
                     because the user explicitly asked us to skip
                     confirmation). We keep this as a separate field
                     for schema symmetry with `suppressed` and to leave
                     room for a future "alerts that escalated to paged"
                     subset if we ever split the two.
        suppressed_by_test: Per-test breakdown so the digest can render
                    "Suppressed this week: flaky-search ×3, auth-retry ×1"
                    with last-seen timestamps. Keeping this as a list
                    rather than a dict preserves ordering for rendering
                    (most recent / highest count first).
    """

    alerts_fired: int = 0
    real_alerts: int = 0
    suppressed: int = 0
    suppressed_by_test: List[SuppressedEntry] = field(default_factory=list)

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
        - alerts_fired:     number of alerts that fired this cycle
                            (confirmed + strict_immediate)
        - real_alerts:      same as alerts_fired — every fired alert is
                            real by definition of the gate
        - suppressed:       number of self-resolved pendings this cycle
        - pending_count:    number of pendings carried into next cycle
                            (diagnostic only, not part of the public metric)
        - suppressed_tests: *names* of the tests that self-resolved,
                            so the digest can show the user WHICH tests
                            were suppressed, not just a count. This is
                            critical: suppressing the alert is fine,
                            suppressing the signal is not.
        - strict_tests:     names of tests that fired via strict bypass,
                            so the audit trail records the reason.

    Writes are best-effort — a failed write logs a warning and returns.
    The monitor loop must never crash because the noise log is unwritable.
    """
    base = base_path or _DEFAULT_BASE
    path = base / _NOISE_LOG
    ts = timestamp or datetime.now(timezone.utc)

    alerts_fired = len(decision.confirmed) + len(decision.strict_immediate)
    record = {
        "ts": ts.isoformat(),
        "alerts_fired": alerts_fired,
        "real_alerts": alerts_fired,
        "suppressed": len(decision.self_resolved),
        "pending_count": len(decision.pending),
        "suppressed_tests": sorted(decision.self_resolved),
        "strict_tests": sorted(decision.strict_immediate),
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

    # Per-test aggregation — used to populate suppressed_by_test.
    # {test_name: (count, last_seen_ts_iso)}
    per_test: Dict[str, Tuple[int, str]] = {}

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

                # Aggregate per-test suppression counts so we can surface
                # "WHICH tests got silently suppressed" in the digest.
                ts_str = str(record.get("ts") or "")
                for name in record.get("suppressed_tests") or []:
                    prev_count, prev_ts = per_test.get(str(name), (0, ""))
                    # Keep the lexicographically-later (≈ most recent) ts
                    new_ts = ts_str if ts_str > prev_ts else prev_ts
                    per_test[str(name)] = (prev_count + 1, new_ts)
    except OSError as exc:
        logger.warning("Failed to read noise log: %s", exc)
        return NoiseStats()

    # Sort by (count desc, most recent first) so the digest leads with
    # the tests a user would most want to look at.
    stats.suppressed_by_test = [
        SuppressedEntry(test_name=name, count=count, last_seen=last_seen)
        for name, (count, last_seen) in sorted(
            per_test.items(),
            key=lambda kv: (-kv[1][0], kv[0]),
        )
    ]

    return stats
