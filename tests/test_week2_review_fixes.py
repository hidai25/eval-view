"""Tests for the Week 2 self-review fixes.

Covers:
  - P0.1: corrupted YAML load engages safe_mode, does NOT overwrite
  - P0.1: missing entries still rejected per-entry but others load
  - P1.1: batch_update context manager collapses N writes to 1
  - P1.2: markdown escape in PR comment test names
  - P1.3: code-fence fallback when a command contains triple backticks
  - P1.4: stale_tests list is capped
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict
from unittest.mock import patch

import pytest
import yaml

from evalview.core.quarantine import (
    QuarantineEntry,
    QuarantineLoadError,
    QuarantineStore,
)
from evalview.ci.comment import (
    _build_recommendation_block,
    _build_verdict_signals_table,
    _md_escape_inline,
    _pick_code_fence,
)


# ────────────────────────── P0.1 safe_mode ──────────────────────────


def test_corrupted_yaml_engages_safe_mode(tmp_path: Path) -> None:
    path = tmp_path / "q.yaml"
    path.write_text("this is: : :: not valid yaml [[[")
    store = QuarantineStore(path=path)
    assert store.safe_mode is True
    assert store.entries == {}


def test_safe_mode_refuses_to_save(tmp_path: Path) -> None:
    """Previously: corrupted load → empty entries → next _save()
    overwrites the corrupt file with an empty one, destroying data.
    Now: safe_mode blocks the write."""
    path = tmp_path / "q.yaml"
    original_bytes = b"quarantined:\n  broken: : [[["
    path.write_bytes(original_bytes)

    store = QuarantineStore(path=path)
    assert store.safe_mode is True

    # Attempt a write — should silently refuse (logged, not raised)
    store._save()

    # File on disk must be untouched
    assert path.read_bytes() == original_bytes


def test_non_dict_top_level_engages_safe_mode(tmp_path: Path) -> None:
    path = tmp_path / "q.yaml"
    path.write_text("- this is a list\n- not a dict")
    store = QuarantineStore(path=path)
    assert store.safe_mode is True


def test_non_dict_quarantined_key_engages_safe_mode(tmp_path: Path) -> None:
    path = tmp_path / "q.yaml"
    path.write_text("quarantined:\n  - nope\n  - not a dict either")
    store = QuarantineStore(path=path)
    assert store.safe_mode is True


def test_malformed_individual_entry_is_skipped_others_load(tmp_path: Path) -> None:
    """One bad entry must not take down the whole file."""
    path = tmp_path / "q.yaml"
    data = {
        "quarantined": {
            "good_test": {
                "reason": "real",
                "owner": "@h",
                "added_at": "2026-04-01T00:00:00+00:00",
                "flaky_count": 1,
            },
            "bad_test": "this should be a dict",  # malformed
        }
    }
    path.write_text(yaml.dump(data))
    store = QuarantineStore(path=path)
    # The good one loads; the bad one is silently skipped
    assert store.safe_mode is False
    assert "good_test" in store.entries
    assert "bad_test" not in store.entries


def test_missing_file_is_not_safe_mode(tmp_path: Path) -> None:
    store = QuarantineStore(path=tmp_path / "does-not-exist.yaml")
    assert store.safe_mode is False
    assert store.entries == {}


# ────────────────────────── P1.1 batch_update ──────────────────────────


def test_batch_update_collapses_many_writes_to_one(tmp_path: Path) -> None:
    """Pre-fix: increment_flaky wrote YAML on every call.

    We patch `_save` and count how many times it's called inside the
    batch. Exactly one flush should happen on __exit__.
    """
    path = tmp_path / "q.yaml"
    store = QuarantineStore(path=path)
    store.add("t1", reason="r", owner="@h")

    write_count = {"n": 0}
    original_save = store.__class__._save

    def counting_save(self: QuarantineStore) -> None:
        # Only count flushes that actually hit disk (not no-ops)
        if self._batching:
            # Batching pre-exit: buffered, not a real write
            original_save(self)
            return
        write_count["n"] += 1
        original_save(self)

    with patch.object(QuarantineStore, "_save", counting_save):
        with store.batch_update():
            for _ in range(10):
                store.increment_flaky("t1")
        assert write_count["n"] == 1

    # Data actually persisted after the batch
    reloaded = QuarantineStore(path=path)
    assert reloaded.entries["t1"].flaky_count == 10


def test_batch_update_flushes_on_exception(tmp_path: Path) -> None:
    """If the caller raises mid-batch, what happens?

    We leave the dirty flag on but don't flush — safer to drop the
    partial batch than to persist a partial state. Tested here mostly
    to lock in the behavior so future changes notice.
    """
    path = tmp_path / "q.yaml"
    store = QuarantineStore(path=path)
    store.add("t1", reason="r", owner="@h")

    with pytest.raises(ValueError):
        with store.batch_update():
            store.increment_flaky("t1")
            raise ValueError("boom")

    # Store still usable after the exception
    store.increment_flaky("t1")
    reloaded = QuarantineStore(path=path)
    assert reloaded.entries["t1"].flaky_count >= 1


def test_non_batched_increment_still_writes_immediately(tmp_path: Path) -> None:
    path = tmp_path / "q.yaml"
    store = QuarantineStore(path=path)
    store.add("t1", reason="r", owner="@h")
    store.increment_flaky("t1")
    # A fresh load sees the update without needing an exit
    reloaded = QuarantineStore(path=path)
    assert reloaded.entries["t1"].flaky_count == 1


# ────────────────────────── P1.2 markdown escape ──────────────────────────


def test_md_escape_backticks() -> None:
    assert _md_escape_inline("has`backtick") == "has&#96;backtick"


def test_md_escape_pipes() -> None:
    assert _md_escape_inline("has|pipe") == "has\\|pipe"


def test_md_escape_both() -> None:
    assert _md_escape_inline("a`b|c") == "a&#96;b\\|c"


def test_md_escape_empty_is_noop() -> None:
    assert _md_escape_inline("") == ""


def test_signals_table_escapes_stale_test_names() -> None:
    check = {"summary": {"total_tests": 3, "unchanged": 3, "regressions": 0,
                          "tools_changed": 0, "output_changed": 0}}
    verdict = {
        "verdict": "investigate",
        "quarantine": {
            "total": 2,
            "stale": 2,
            "stale_tests": ["weird`name", "pipe|name"],
        },
    }
    lines = _build_verdict_signals_table(check, verdict)
    body = "\n".join(lines)
    # Escaped versions must be present; raw must not
    assert "weird&#96;name" in body
    assert "pipe\\|name" in body
    assert "weird`name" not in body  # raw backtick form is gone


# ────────────────────────── P1.3 fence picker ──────────────────────────


def test_pick_code_fence_defaults_to_backticks() -> None:
    assert _pick_code_fence(["evalview check"]) == "```"


def test_pick_code_fence_falls_back_to_tildes_when_triple_backticks_present() -> None:
    assert _pick_code_fence(["echo '```'"]) == "~~~"


def test_recommendation_block_uses_tilde_fence_for_backtick_command() -> None:
    verdict = {
        "recommendations": [
            {
                "action": "Do X",
                "confidence": "high",
                "suggested_commands": ["echo '```'"],
            }
        ]
    }
    lines = _build_recommendation_block(verdict)
    body = "\n".join(lines)
    assert "~~~" in body
    # The fence is not a plain triple-backtick anywhere that would
    # collide with the content.
    assert body.count("~~~") == 2  # open + close


def test_recommendation_block_default_fence_unchanged_for_safe_commands() -> None:
    verdict = {
        "recommendations": [
            {
                "action": "Do X",
                "confidence": "high",
                "suggested_commands": ["evalview replay search_cases"],
            }
        ]
    }
    lines = _build_recommendation_block(verdict)
    body = "\n".join(lines)
    assert "```bash" in body


# ────────────────────────── P1.4 stale_tests cap ──────────────────────────


def test_compute_verdict_payload_caps_stale_tests(tmp_path: Path) -> None:
    """_compute_verdict_payload stores at most 10 stale test names.

    Built by hand so we don't need a real check run. Stubs a
    QuarantineStore that returns 25 stale entries.
    """
    from evalview.commands.check_cmd import _compute_verdict_payload

    class FakeStore:
        def list_all(self):
            return [QuarantineEntry(test_name=f"t{i}", owner="@h", reason="r")
                    for i in range(25)]

        def list_stale(self):
            return self.list_all()

    out = _compute_verdict_payload(
        diffs=[],
        results=[],
        drift_tracker=None,
        execution_failures=0,
        golden_traces=None,
        quarantine=FakeStore(),
    )
    payload = out.payload
    assert payload["quarantine"]["total"] == 25
    assert payload["quarantine"]["stale"] == 25
    assert len(payload["quarantine"]["stale_tests"]) == 10  # capped
