"""Tests for the confirmation gate, incident collapse, and noise metric.

These three pieces together deliver the "less noisy product" initiative:

    1. ConfirmationGate: no alert ever fires on n=1.
    2. detect_coordinated_incident: many correlated failures collapse
       into one incident card.
    3. NoiseStats / record_cycle_noise / load_noise_stats: publicly
       reported false-positive rate that the gate is saving users from.

The tests below pin down the exact state machine so future refactors
can't silently break the user-facing promise of "we don't page you
until we're sure."
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from evalview.core.diff import DiffStatus
from evalview.core.noise_tracker import (
    ConfirmationGate,
    GateDecision,
    Incident,
    NoiseStats,
    detect_coordinated_incident,
    load_noise_stats,
    record_cycle_noise,
)


# ─────────────────────────── helpers ───────────────────────────


def _make_diff(
    severity: DiffStatus,
    *,
    model_changed: bool = False,
    fingerprint: str | None = None,
    actual_model_id: str | None = None,
):
    """Build a minimal diff-like object that exposes the fields the
    incident detector inspects. Using a plain MagicMock keeps the test
    decoupled from the real TraceDiff class — which would otherwise
    drag in the whole evaluator pipeline just to check one boolean."""
    diff = MagicMock()
    diff.overall_severity = severity
    diff.model_changed = model_changed
    diff.runtime_model_fingerprint = fingerprint
    diff.actual_model_id = actual_model_id
    return diff


# ─────────────────────────── confirmation gate ───────────────────────────


class TestConfirmationGate:
    """The gate must never promote a failure to an alert on the first sighting."""

    def test_first_failure_becomes_pending_not_confirmed(self):
        gate = ConfirmationGate()
        decision = gate.evaluate({"billing-flow"})
        # First time we see it — should go pending, not alert.
        assert decision.confirmed == set()
        assert decision.pending == {"billing-flow"}
        assert decision.alerts_to_fire == set()

    def test_failure_confirmed_on_second_consecutive_cycle(self):
        gate = ConfirmationGate()
        gate.evaluate({"billing-flow"})  # cycle 1 → pending
        decision = gate.evaluate({"billing-flow"})  # cycle 2 → confirmed
        assert decision.confirmed == {"billing-flow"}
        assert decision.alerts_to_fire == {"billing-flow"}
        # After confirmation it leaves the pending bucket.
        assert decision.pending == set()

    def test_flake_self_resolves_without_alerting(self):
        """A failure that appears once and disappears must never alert
        and must be counted as a suppressed false positive — this is
        the whole point of the gate."""
        gate = ConfirmationGate()
        gate.evaluate({"flaky-test"})
        decision = gate.evaluate(set())  # recovers before confirmation
        assert decision.confirmed == set()
        assert decision.self_resolved == {"flaky-test"}
        assert decision.alerts_to_fire == set()

    def test_no_re_alert_on_persistent_failure(self):
        gate = ConfirmationGate()
        gate.evaluate({"billing-flow"})  # pending
        gate.evaluate({"billing-flow"})  # confirmed → alert fired
        decision = gate.evaluate({"billing-flow"})  # still failing
        # Already alerted — do not fire again.
        assert decision.confirmed == set()
        assert decision.carried_forward == {"billing-flow"}
        assert decision.alerts_to_fire == set()

    def test_independent_tests_tracked_separately(self):
        gate = ConfirmationGate()
        gate.evaluate({"a"})  # a pending
        decision = gate.evaluate({"a", "b"})  # a confirmed, b new
        assert decision.confirmed == {"a"}
        assert decision.pending == {"b"}
        assert decision.alerts_to_fire == {"a"}

    def test_mixed_cycle_confirmed_suppressed_and_new(self):
        """One cycle can simultaneously confirm one, suppress another,
        and park a third in pending — the state machine has to keep
        all three buckets straight."""
        gate = ConfirmationGate()
        gate.evaluate({"a", "b"})  # both pending
        decision = gate.evaluate({"a", "c"})
        # a was pending, failed again → confirmed
        # b was pending, passed → suppressed
        # c is new → pending
        assert decision.confirmed == {"a"}
        assert decision.self_resolved == {"b"}
        assert decision.pending == {"c"}


# ─────────────────────────── incident detection ───────────────────────────


class TestCoordinatedIncident:
    def test_no_incident_for_single_failure(self):
        diffs = [("only-one", _make_diff(DiffStatus.REGRESSION))]
        assert detect_coordinated_incident(diffs) is None

    def test_model_change_across_multiple_tests_high_confidence(self):
        """3+ failing tests all reporting model_changed should collapse
        into a HIGH-confidence 'likely provider update' incident — this
        is the exact 'gpt-5.1 rolled out, 12 alerts became 1' scenario."""
        diffs = [
            ("a", _make_diff(DiffStatus.REGRESSION, model_changed=True)),
            ("b", _make_diff(DiffStatus.REGRESSION, model_changed=True)),
            ("c", _make_diff(DiffStatus.REGRESSION, model_changed=True)),
        ]
        incident = detect_coordinated_incident(diffs)
        assert incident is not None
        assert incident.confidence == "high"
        assert "provider update" in incident.cause
        assert set(incident.affected) == {"a", "b", "c"}
        assert "3 test" in incident.headline

    def test_shared_runtime_fingerprint_medium_confidence(self):
        diffs = [
            ("a", _make_diff(DiffStatus.REGRESSION, fingerprint="gpt-5.1-2026-04")),
            ("b", _make_diff(DiffStatus.REGRESSION, fingerprint="gpt-5.1-2026-04")),
            ("c", _make_diff(DiffStatus.REGRESSION, fingerprint="gpt-5.1-2026-04")),
        ]
        incident = detect_coordinated_incident(diffs)
        assert incident is not None
        assert incident.confidence == "medium"
        assert "fingerprint" in incident.cause

    def test_correlated_batch_low_confidence_requires_majority(self):
        """A batch where the majority of tests fail but share no signal
        still collapses — but only at LOW confidence, and only when
        failing is ≥50% of the batch."""
        diffs = [
            ("a", _make_diff(DiffStatus.REGRESSION)),
            ("b", _make_diff(DiffStatus.REGRESSION)),
            ("c", _make_diff(DiffStatus.REGRESSION)),
        ]
        incident = detect_coordinated_incident(diffs)
        assert incident is not None
        assert incident.confidence == "low"

    def test_single_failure_in_large_batch_no_incident(self):
        """One test failing in a 10-test suite should NOT trigger a
        correlated-batch incident — that's just a normal regression."""
        diffs = [("bad", _make_diff(DiffStatus.REGRESSION))]
        diffs.extend(
            (f"good-{i}", _make_diff(DiffStatus.PASSED)) for i in range(9)
        )
        assert detect_coordinated_incident(diffs) is None

    def test_min_affected_threshold_respected(self):
        diffs = [
            ("a", _make_diff(DiffStatus.REGRESSION, model_changed=True)),
            ("b", _make_diff(DiffStatus.REGRESSION, model_changed=True)),
        ]
        # Only 2 failures — below default min_affected=3.
        assert detect_coordinated_incident(diffs) is None
        # But with a lower threshold it collapses.
        incident = detect_coordinated_incident(diffs, min_affected=2)
        assert incident is not None
        assert incident.confidence == "high"

    def test_incident_headline_format(self):
        incident = Incident(
            cause="likely provider update",
            confidence="high",
            affected=["a", "b", "c"],
        )
        assert incident.headline == (
            "3 tests shifted together — likely provider update (confidence: high)"
        )


# ─────────────────────────── noise stats + persistence ───────────────────────────


class TestNoiseStats:
    def test_empty_stats_has_no_rate(self):
        stats = NoiseStats()
        assert stats.false_positive_rate is None
        assert "No alert activity" in stats.format_line()

    def test_rate_computation(self):
        stats = NoiseStats(alerts_fired=8, real_alerts=8, suppressed=2)
        # 2 suppressed out of 10 total raw failures → 20% noise
        assert stats.false_positive_rate == pytest.approx(0.2)
        line = stats.format_line()
        assert "8 alerts fired" in line
        assert "8 real" in line
        assert "2 suppressed" in line
        assert "20%" in line

    def test_all_noise(self):
        stats = NoiseStats(alerts_fired=0, real_alerts=0, suppressed=5)
        # 5/5 → 100% noise — the gate saved the user every single time.
        assert stats.false_positive_rate == pytest.approx(1.0)


class TestNoisePersistence:
    def test_record_and_load_roundtrip(self, tmp_path):
        decision = GateDecision(
            confirmed={"a"},
            pending={"b"},
            self_resolved={"c", "d"},
        )
        record_cycle_noise(decision, base_path=tmp_path)
        stats = load_noise_stats(base_path=tmp_path)
        assert stats.alerts_fired == 1
        assert stats.real_alerts == 1
        assert stats.suppressed == 2

    def test_multiple_cycles_aggregate(self, tmp_path):
        for _ in range(3):
            record_cycle_noise(
                GateDecision(confirmed={"x"}, self_resolved={"y"}),
                base_path=tmp_path,
            )
        stats = load_noise_stats(base_path=tmp_path)
        assert stats.alerts_fired == 3
        assert stats.suppressed == 3
        assert stats.false_positive_rate == pytest.approx(0.5)

    def test_load_respects_since_filter(self, tmp_path):
        """Old entries outside the window must not skew the digest's
        'this week' line — we'd rather show nothing than the wrong number."""
        old = datetime.now(timezone.utc) - timedelta(days=30)
        new = datetime.now(timezone.utc)
        record_cycle_noise(
            GateDecision(confirmed={"old"}), base_path=tmp_path, timestamp=old
        )
        record_cycle_noise(
            GateDecision(confirmed={"new"}, self_resolved={"new2"}),
            base_path=tmp_path,
            timestamp=new,
        )
        cutoff = datetime.now(timezone.utc) - timedelta(days=7)
        stats = load_noise_stats(base_path=tmp_path, since=cutoff)
        # Only the fresh record should be counted.
        assert stats.alerts_fired == 1
        assert stats.suppressed == 1

    def test_missing_file_returns_empty_stats(self, tmp_path):
        stats = load_noise_stats(base_path=tmp_path)
        assert stats == NoiseStats()
        assert stats.false_positive_rate is None

    def test_malformed_line_is_skipped(self, tmp_path):
        path = tmp_path / "noise.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        # Mix one garbage line with one valid line.
        path.write_text(
            "not-json\n"
            + json.dumps(
                {
                    "ts": datetime.now(timezone.utc).isoformat(),
                    "alerts_fired": 2,
                    "real_alerts": 2,
                    "suppressed": 1,
                }
            )
            + "\n"
        )
        stats = load_noise_stats(base_path=tmp_path)
        assert stats.alerts_fired == 2
        assert stats.suppressed == 1
