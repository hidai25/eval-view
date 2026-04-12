"""Table-driven tests for the release verdict layer.

Each row represents one representative scenario a real check would produce.
The goal is to pin down the derivation rules in one place so regressions
are obvious.
"""
from __future__ import annotations

from typing import List, Tuple

import pytest

from evalview.core.verdict import (
    Verdict,
    VerdictSignals,
    compute_verdict,
    headline_for,
    verdict_to_dict,
)


def _statuses(*pairs: Tuple[str, str]) -> List[Tuple[str, str]]:
    return list(pairs)


# ────────────────────────── happy path ──────────────────────────


def test_all_clean_is_safe_to_ship() -> None:
    signals = VerdictSignals(
        test_statuses=_statuses(
            ("login", "passed"),
            ("search", "passed"),
        ),
    )
    verdict, reasons = compute_verdict(signals)
    assert verdict == Verdict.SAFE_TO_SHIP
    assert reasons  # always a one-liner, even for the happy path


def test_empty_signals_is_safe_to_ship() -> None:
    verdict, reasons = compute_verdict(VerdictSignals())
    assert verdict == Verdict.SAFE_TO_SHIP
    assert len(reasons) >= 1


# ────────────────────────── block paths ──────────────────────────


def test_unquarantined_regression_blocks() -> None:
    signals = VerdictSignals(
        test_statuses=_statuses(
            ("login", "passed"),
            ("search", "regression"),
        ),
    )
    verdict, reasons = compute_verdict(signals)
    assert verdict == Verdict.BLOCK_RELEASE
    assert any("regression" in r.lower() for r in reasons)


def test_execution_failures_block() -> None:
    signals = VerdictSignals(execution_failures=2)
    verdict, reasons = compute_verdict(signals)
    assert verdict == Verdict.BLOCK_RELEASE
    assert any("execute" in r.lower() for r in reasons)


def test_contract_drift_blocks() -> None:
    signals = VerdictSignals(
        test_statuses=_statuses(("mcp_tool", "contract_drift")),
    )
    verdict, _ = compute_verdict(signals)
    assert verdict == Verdict.BLOCK_RELEASE


def test_regression_in_quarantine_does_not_block() -> None:
    """Quarantined tests must NOT escalate the verdict to BLOCK."""
    signals = VerdictSignals(
        test_statuses=_statuses(("flaky_login", "regression")),
        quarantined_tests=["flaky_login"],
    )
    verdict, _ = compute_verdict(signals)
    assert verdict != Verdict.BLOCK_RELEASE
    assert verdict == Verdict.SHIP_WITH_QUARANTINE


# ────────────────────────── investigate paths ──────────────────────────


def test_soft_status_triggers_investigate() -> None:
    signals = VerdictSignals(
        test_statuses=_statuses(("search", "tools_changed")),
    )
    verdict, reasons = compute_verdict(signals)
    assert verdict == Verdict.INVESTIGATE
    assert any("changed behavior" in r for r in reasons)


def test_cost_spike_triggers_investigate() -> None:
    signals = VerdictSignals(
        test_statuses=_statuses(("login", "passed")),
        cost_delta_ratio=0.25,  # +25%
    )
    verdict, reasons = compute_verdict(signals)
    assert verdict == Verdict.INVESTIGATE
    assert any("cost" in r.lower() for r in reasons)


def test_small_cost_delta_does_not_escalate() -> None:
    signals = VerdictSignals(
        test_statuses=_statuses(("login", "passed")),
        cost_delta_ratio=0.05,  # +5% is within noise band
    )
    verdict, _ = compute_verdict(signals)
    assert verdict == Verdict.SAFE_TO_SHIP


def test_cost_delta_at_exact_threshold_does_not_escalate() -> None:
    """Boundary test: the rule is strictly `> 0.10`, not `>=`."""
    signals = VerdictSignals(
        test_statuses=_statuses(("login", "passed")),
        cost_delta_ratio=0.10,  # exactly at threshold
    )
    verdict, _ = compute_verdict(signals)
    assert verdict == Verdict.SAFE_TO_SHIP


def test_cost_delta_just_above_threshold_escalates() -> None:
    signals = VerdictSignals(
        test_statuses=_statuses(("login", "passed")),
        cost_delta_ratio=0.1001,
    )
    verdict, _ = compute_verdict(signals)
    assert verdict == Verdict.INVESTIGATE


def test_high_downward_drift_flips_safe_to_investigate() -> None:
    signals = VerdictSignals(
        test_statuses=_statuses(("search", "passed")),
        drift_is_downward=True,
        drift_confidence="high",
    )
    verdict, reasons = compute_verdict(signals)
    assert verdict == Verdict.INVESTIGATE
    assert any("drift" in r.lower() for r in reasons)


def test_low_confidence_drift_does_not_flip() -> None:
    signals = VerdictSignals(
        test_statuses=_statuses(("search", "passed")),
        drift_is_downward=True,
        drift_confidence="low",  # not high enough to investigate
    )
    verdict, _ = compute_verdict(signals)
    assert verdict == Verdict.SAFE_TO_SHIP


def test_stale_quarantine_triggers_investigate() -> None:
    signals = VerdictSignals(
        test_statuses=_statuses(("login", "passed")),
        quarantined_tests=["old_retry_test"],
        stale_quarantined_tests=["old_retry_test"],
    )
    verdict, reasons = compute_verdict(signals)
    assert verdict == Verdict.INVESTIGATE
    assert any("stale" in r.lower() for r in reasons)


# ────────────────────────── ship-with-quarantine path ──────────────────────────


def test_clean_with_active_quarantine_is_ship_with_quarantine() -> None:
    signals = VerdictSignals(
        test_statuses=_statuses(
            ("login", "passed"),
            ("flaky_one", "passed"),
        ),
        quarantined_tests=["flaky_one"],
    )
    verdict, reasons = compute_verdict(signals)
    assert verdict == Verdict.SHIP_WITH_QUARANTINE
    assert any("quarantine" in r.lower() for r in reasons)


# ────────────────────────── precedence ──────────────────────────


def test_block_always_wins_over_investigate() -> None:
    """BLOCK must beat INVESTIGATE even if both are triggered."""
    signals = VerdictSignals(
        test_statuses=_statuses(
            ("login", "regression"),         # block
            ("search", "tools_changed"),     # investigate
        ),
        cost_delta_ratio=0.50,               # investigate
        drift_is_downward=True,              # investigate
        drift_confidence="high",
    )
    verdict, _ = compute_verdict(signals)
    assert verdict == Verdict.BLOCK_RELEASE


def test_rank_ordering() -> None:
    assert Verdict.SAFE_TO_SHIP.rank < Verdict.SHIP_WITH_QUARANTINE.rank
    assert Verdict.SHIP_WITH_QUARANTINE.rank < Verdict.INVESTIGATE.rank
    assert Verdict.INVESTIGATE.rank < Verdict.BLOCK_RELEASE.rank


# ────────────────────────── presentation helpers ──────────────────────────


@pytest.mark.parametrize("verdict", list(Verdict))
def test_headline_for_all_verdicts(verdict: Verdict) -> None:
    text, color = headline_for(verdict)
    assert text.strip()
    assert color in {"green", "yellow", "red"}


def test_verdict_to_dict_serializable_shape() -> None:
    verdict, reasons = compute_verdict(
        VerdictSignals(test_statuses=_statuses(("login", "regression")))
    )
    payload = verdict_to_dict(verdict, reasons)
    assert payload["verdict"] == "block_release"
    assert payload["headline"]
    assert isinstance(payload["reasons"], list)
    assert payload["reasons"]
