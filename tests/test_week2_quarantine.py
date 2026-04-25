"""Tests for Week 2 quarantine governance.

Covers:
  - `add` requires owner + reason (raises otherwise)
  - Stale detection (expiry, review overdue, improving-count exemption)
  - flaky_count_history tracking + trend glyph (↗/→/↘)
  - list_stale() helper
  - Backwards-compat load of old YAML without owner/reason
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
import yaml

from evalview.core.quarantine import (
    DEFAULT_REVIEW_DAYS,
    QuarantineEntry,
    QuarantineOwnerRequired,
    QuarantineReasonRequired,
    QuarantineStore,
)


# ────────────────────────── governance enforcement ──────────────────────────


def test_add_without_owner_raises(tmp_path: Path) -> None:
    store = QuarantineStore(path=tmp_path / "q.yaml")
    with pytest.raises(QuarantineOwnerRequired):
        store.add("t1", reason="because")


def test_add_without_reason_raises(tmp_path: Path) -> None:
    store = QuarantineStore(path=tmp_path / "q.yaml")
    with pytest.raises(QuarantineReasonRequired):
        store.add("t1", reason="", owner="@hidai")


def test_add_with_whitespace_only_reason_raises(tmp_path: Path) -> None:
    store = QuarantineStore(path=tmp_path / "q.yaml")
    with pytest.raises(QuarantineReasonRequired):
        store.add("t1", reason="   ", owner="@hidai")


def test_add_with_owner_and_reason_succeeds(tmp_path: Path) -> None:
    store = QuarantineStore(path=tmp_path / "q.yaml")
    entry = store.add("t1", reason="prompt iter", owner="@hidai")
    assert entry.owner == "@hidai"
    assert entry.reason == "prompt iter"
    assert entry.review_after_days == DEFAULT_REVIEW_DAYS
    assert entry.added_at  # ISO string populated
    assert store.is_quarantined("t1")


def test_add_persists_and_reloads(tmp_path: Path) -> None:
    path = tmp_path / "q.yaml"
    store1 = QuarantineStore(path=path)
    store1.add("t1", reason="flake", owner="@hidai", review_after_days=7)
    store2 = QuarantineStore(path=path)
    entry = store2.entries["t1"]
    assert entry.owner == "@hidai"
    assert entry.reason == "flake"
    assert entry.review_after_days == 7


# ────────────────────────── stale detection ──────────────────────────


def _force_age(entry: QuarantineEntry, days: int) -> None:
    """Rewind `added_at` so the entry appears to have been added `days` days ago."""
    then = datetime.now(timezone.utc) - timedelta(days=days)
    entry.added_at = then.isoformat()


def test_fresh_entry_is_not_stale(tmp_path: Path) -> None:
    store = QuarantineStore(path=tmp_path / "q.yaml")
    entry = store.add("t1", reason="r", owner="@h", review_after_days=14)
    assert not entry.stale


def test_review_overdue_is_stale(tmp_path: Path) -> None:
    store = QuarantineStore(path=tmp_path / "q.yaml")
    entry = store.add("t1", reason="r", owner="@h", review_after_days=3)
    _force_age(entry, 10)
    assert entry.is_review_overdue
    assert entry.stale


def test_review_overdue_but_improving_is_not_stale(tmp_path: Path) -> None:
    """Improving flaky_count exempts an overdue entry from being stale.

    This is the "don't nag me if I'm already fixing it" rule.
    """
    store = QuarantineStore(path=tmp_path / "q.yaml")
    entry = store.add("t1", reason="r", owner="@h", review_after_days=3)
    _force_age(entry, 10)
    entry.flaky_count_history = [5, 4, 3, 2]
    assert entry.is_review_overdue
    assert entry.flaky_trend == "down"
    assert not entry.stale


def test_hard_expiry_past_is_stale(tmp_path: Path) -> None:
    store = QuarantineStore(path=tmp_path / "q.yaml")
    entry = store.add(
        "t1", reason="r", owner="@h",
        expiry_date=(datetime.now(timezone.utc) - timedelta(days=1)).isoformat(),
    )
    assert entry.is_expired
    assert entry.stale


def test_hard_expiry_future_is_not_stale(tmp_path: Path) -> None:
    store = QuarantineStore(path=tmp_path / "q.yaml")
    entry = store.add(
        "t1", reason="r", owner="@h",
        expiry_date=(datetime.now(timezone.utc) + timedelta(days=30)).isoformat(),
    )
    assert not entry.is_expired
    assert not entry.stale


def test_list_stale_filters_correctly(tmp_path: Path) -> None:
    store = QuarantineStore(path=tmp_path / "q.yaml")
    store.add("fresh", reason="r", owner="@h", review_after_days=30)
    overdue = store.add("overdue", reason="r", owner="@h", review_after_days=1)
    _force_age(overdue, 5)

    stale = store.list_stale()
    assert {e.test_name for e in stale} == {"overdue"}
    # Guarding against future refactors: list_all still returns both.
    assert len(store.list_all()) == 2


# ────────────────────────── flaky trend ──────────────────────────


def test_flaky_trend_flat_with_short_history() -> None:
    e = QuarantineEntry(test_name="t", flaky_count_history=[])
    assert e.flaky_trend == "flat"
    e.flaky_count_history = [3]
    assert e.flaky_trend == "flat"


def test_flaky_trend_up() -> None:
    e = QuarantineEntry(test_name="t", flaky_count_history=[1, 2, 3, 4])
    assert e.flaky_trend == "up"


def test_flaky_trend_down() -> None:
    e = QuarantineEntry(test_name="t", flaky_count_history=[5, 4, 3, 2])
    assert e.flaky_trend == "down"


def test_flaky_trend_flat_when_equal() -> None:
    e = QuarantineEntry(test_name="t", flaky_count_history=[3, 4, 5, 3])
    assert e.flaky_trend == "flat"


def test_increment_flaky_updates_history(tmp_path: Path) -> None:
    store = QuarantineStore(path=tmp_path / "q.yaml")
    store.add("t1", reason="r", owner="@h")
    store.increment_flaky("t1")
    store.increment_flaky("t1")
    store.increment_flaky("t1")
    entry = store.entries["t1"]
    assert entry.flaky_count == 3
    assert entry.flaky_count_history == [1, 2, 3]


def test_increment_flaky_caps_history_length(tmp_path: Path) -> None:
    store = QuarantineStore(path=tmp_path / "q.yaml")
    store.add("t1", reason="r", owner="@h")
    for _ in range(25):
        store.increment_flaky("t1")
    entry = store.entries["t1"]
    # Retained window is 20 — ensure we didn't blow up to 25.
    assert len(entry.flaky_count_history) == 20
    assert entry.flaky_count == 25


# ────────────────────────── backwards compat ──────────────────────────


def test_loads_legacy_yaml_without_owner_or_reason(tmp_path: Path) -> None:
    """Old YAML files (pre-Week-2) had no owner/review_after_days/history.

    New store must load them cleanly — they just show up as "unknown
    owner" and default review window. Breaking existing repos on
    upgrade would be unforgivable.
    """
    path = tmp_path / "q.yaml"
    legacy = {
        "quarantined": {
            "legacy_test": {
                "reason": "",
                "added_at": "2026-01-01T00:00:00+00:00",
                "flaky_count": 2,
            }
        }
    }
    path.write_text(yaml.dump(legacy))
    store = QuarantineStore(path=path)
    entry = store.entries["legacy_test"]
    assert entry.owner == ""  # empty, not missing
    assert entry.reason == ""
    assert entry.review_after_days == DEFAULT_REVIEW_DAYS
    assert entry.flaky_count_history == []


def test_to_dict_includes_computed_fields(tmp_path: Path) -> None:
    store = QuarantineStore(path=tmp_path / "q.yaml")
    entry = store.add("t1", reason="r", owner="@h")
    payload = entry.to_dict()
    assert payload["owner"] == "@h"
    assert payload["reason"] == "r"
    assert "age_days" in payload
    assert "stale" in payload
    assert "flaky_trend" in payload
