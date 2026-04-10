"""Unit tests for core/model_snapshots.py."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from evalview.core.model_snapshots import (
    ModelCheckPromptResult,
    ModelSnapshot,
    ModelSnapshotMetadata,
    ModelSnapshotStore,
    SnapshotSuiteMismatchError,
)


# --------------------------------------------------------------------------- #
# Fixtures / helpers
# --------------------------------------------------------------------------- #


def _make_snapshot(
    *,
    model_id: str = "claude-opus-4-5-20251101",
    provider: str = "anthropic",
    snapshot_at: datetime | None = None,
    suite_hash: str = "sha256:aaa",
    suite_version: str = "v1.public",
    temperature: float = 0.0,
    top_p: float = 1.0,
    runs_per_prompt: int = 3,
    fingerprint: str | None = "claude-opus-4-5-20251101",
    results: list[ModelCheckPromptResult] | None = None,
) -> ModelSnapshot:
    if snapshot_at is None:
        snapshot_at = datetime(2026, 4, 9, 14, 3, 11, tzinfo=timezone.utc)
    if results is None:
        results = [
            ModelCheckPromptResult(
                prompt_id="tool_choice_refund",
                category="tool_choice",
                pass_rate=1.0,
                n_runs=3,
                per_run_passed=[True, True, True],
            ),
            ModelCheckPromptResult(
                prompt_id="json_user_profile",
                category="json_schema",
                pass_rate=2 / 3,
                n_runs=3,
                per_run_passed=[True, False, True],
            ),
        ]
    return ModelSnapshot(
        metadata=ModelSnapshotMetadata(
            model_id=model_id,
            provider=provider,
            snapshot_at=snapshot_at,
            suite_name="canary",
            suite_version=suite_version,
            suite_hash=suite_hash,
            temperature=temperature,
            top_p=top_p,
            runs_per_prompt=runs_per_prompt,
            provider_fingerprint=fingerprint,
            fingerprint_confidence="weak",
            cost_total_usd=0.12,
            evalview_version="0.7.0",
        ),
        results=results,
    )


@pytest.fixture
def store(tmp_path: Path) -> ModelSnapshotStore:
    return ModelSnapshotStore(base_path=tmp_path)


# --------------------------------------------------------------------------- #
# Save / load roundtrip
# --------------------------------------------------------------------------- #


def test_save_and_load_reference_roundtrip(store: ModelSnapshotStore):
    snap = _make_snapshot()
    path = store.save_snapshot(snap)

    assert path.exists()
    assert path.parent.name == "claude-opus-4-5-20251101"
    assert path.name.endswith(".json")

    # First save auto-pins as reference.
    reference = store.load_reference(snap.metadata.model_id)
    assert reference is not None
    assert reference.metadata.model_id == snap.metadata.model_id
    assert reference.metadata.is_reference is True
    # Non-reference snapshot copy still has is_reference=False on disk.
    raw = ModelSnapshot.model_validate_json(path.read_text())
    assert raw.metadata.is_reference is False


def test_overall_pass_rate_and_counts():
    snap = _make_snapshot()
    # 1.0 and 0.666... → mean = 0.833...
    assert snap.overall_pass_rate == pytest.approx(0.8333, abs=0.001)
    assert snap.total_count == 2
    assert snap.passed_count == 1  # only the strict 1.0 prompt counts


# --------------------------------------------------------------------------- #
# Reference pinning
# --------------------------------------------------------------------------- #


def test_second_save_does_not_overwrite_reference(store: ModelSnapshotStore):
    first = _make_snapshot(snapshot_at=datetime(2026, 4, 1, tzinfo=timezone.utc))
    store.save_snapshot(first)

    second = _make_snapshot(snapshot_at=datetime(2026, 4, 8, tzinfo=timezone.utc))
    store.save_snapshot(second)

    reference = store.load_reference(first.metadata.model_id)
    assert reference is not None
    assert reference.metadata.snapshot_at == first.metadata.snapshot_at


def test_pin_reference_replaces_existing(store: ModelSnapshotStore):
    first = _make_snapshot(snapshot_at=datetime(2026, 4, 1, tzinfo=timezone.utc))
    store.save_snapshot(first)

    later = _make_snapshot(snapshot_at=datetime(2026, 4, 8, tzinfo=timezone.utc))
    store.pin_reference(first.metadata.model_id, later)

    reference = store.load_reference(first.metadata.model_id)
    assert reference is not None
    assert reference.metadata.snapshot_at == later.metadata.snapshot_at
    assert reference.metadata.is_reference is True


def test_reset_reference_removes_pin(store: ModelSnapshotStore):
    snap = _make_snapshot()
    store.save_snapshot(snap)

    assert store.reset_reference(snap.metadata.model_id) is True
    assert store.load_reference(snap.metadata.model_id) is None
    # Second reset is a no-op, not an error.
    assert store.reset_reference(snap.metadata.model_id) is False


# --------------------------------------------------------------------------- #
# latest prior / list
# --------------------------------------------------------------------------- #


def test_load_latest_returns_most_recent(store: ModelSnapshotStore):
    base = datetime(2026, 4, 1, tzinfo=timezone.utc)
    for i in range(3):
        store.save_snapshot(_make_snapshot(snapshot_at=base + timedelta(days=i)))

    latest = store.load_latest("claude-opus-4-5-20251101")
    assert latest is not None
    assert latest.metadata.snapshot_at == base + timedelta(days=2)


def test_load_latest_respects_exclude(store: ModelSnapshotStore):
    base = datetime(2026, 4, 1, tzinfo=timezone.utc)
    paths = [
        store.save_snapshot(_make_snapshot(snapshot_at=base + timedelta(days=i)))
        for i in range(3)
    ]
    # Excluding the newest path should return the day before.
    latest = store.load_latest("claude-opus-4-5-20251101", exclude=paths[-1])
    assert latest is not None
    assert latest.metadata.snapshot_at == base + timedelta(days=1)


def test_list_snapshots_sorted_and_excludes_reference(store: ModelSnapshotStore):
    base = datetime(2026, 4, 1, tzinfo=timezone.utc)
    for i in range(3):
        store.save_snapshot(_make_snapshot(snapshot_at=base + timedelta(days=i)))

    metas = store.list_snapshots("claude-opus-4-5-20251101")
    assert len(metas) == 3
    assert metas[0].snapshot_at < metas[1].snapshot_at < metas[2].snapshot_at
    # reference.json must NOT appear in the list.
    assert not any(m.is_reference for m in metas)


def test_list_models_returns_known_ids(store: ModelSnapshotStore):
    store.save_snapshot(_make_snapshot(model_id="claude-opus-4-5-20251101"))
    store.save_snapshot(_make_snapshot(model_id="gpt-5.4-20260101"))
    assert set(store.list_models()) == {
        "claude-opus-4-5-20251101",
        "gpt-5.4-20260101",
    }


# --------------------------------------------------------------------------- #
# Pruning
# --------------------------------------------------------------------------- #


def test_prune_keeps_newest_n_and_leaves_reference(store: ModelSnapshotStore):
    base = datetime(2026, 4, 1, tzinfo=timezone.utc)
    for i in range(10):
        store.save_snapshot(_make_snapshot(snapshot_at=base + timedelta(days=i)))

    deleted = store.prune("claude-opus-4-5-20251101", keep_last=3)
    assert deleted == 7

    metas = store.list_snapshots("claude-opus-4-5-20251101")
    assert len(metas) == 3
    assert metas[0].snapshot_at == base + timedelta(days=7)
    assert metas[-1].snapshot_at == base + timedelta(days=9)
    # Reference is untouched.
    assert store.load_reference("claude-opus-4-5-20251101") is not None


def test_prune_no_op_when_under_limit(store: ModelSnapshotStore):
    store.save_snapshot(_make_snapshot())
    assert store.prune("claude-opus-4-5-20251101", keep_last=5) == 0


# --------------------------------------------------------------------------- #
# Compatibility enforcement
# --------------------------------------------------------------------------- #


def test_assert_comparable_accepts_matching_metadata():
    a = _make_snapshot()
    b = _make_snapshot(snapshot_at=datetime(2026, 4, 10, tzinfo=timezone.utc))
    # Should not raise.
    ModelSnapshotStore.assert_comparable(a, b)


def test_assert_comparable_rejects_suite_hash_mismatch():
    a = _make_snapshot(suite_hash="sha256:aaa")
    b = _make_snapshot(suite_hash="sha256:bbb")
    with pytest.raises(SnapshotSuiteMismatchError, match="Suite hash differs"):
        ModelSnapshotStore.assert_comparable(a, b)


def test_assert_comparable_rejects_temperature_mismatch():
    a = _make_snapshot(temperature=0.0)
    b = _make_snapshot(temperature=0.7)
    with pytest.raises(SnapshotSuiteMismatchError, match="Sampling configuration"):
        ModelSnapshotStore.assert_comparable(a, b)


def test_assert_comparable_rejects_provider_mismatch():
    a = _make_snapshot(provider="anthropic")
    b = _make_snapshot(provider="openai")
    with pytest.raises(SnapshotSuiteMismatchError, match="Provider differs"):
        ModelSnapshotStore.assert_comparable(a, b)


# --------------------------------------------------------------------------- #
# Path safety
# --------------------------------------------------------------------------- #


def test_unsafe_model_id_characters_are_sanitized(store: ModelSnapshotStore):
    snap = _make_snapshot(model_id="weird/model:v2")
    path = store.save_snapshot(snap)
    # Slashes and colons must NOT appear in the directory name.
    dir_name = path.parent.name
    assert "/" not in dir_name
    assert ":" not in dir_name
    assert "weird" in dir_name
    # Loading round-trips the sanitized id.
    assert store.load_reference("weird/model:v2") is not None
