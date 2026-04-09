"""Model snapshot storage for `evalview model-check`.

A model snapshot captures how a specific closed model behaved on a fixed
canary suite at a point in time. Later runs compare against two anchors:

- **reference** — a pinned snapshot that never auto-updates. Enables
  detection of *gradual* drift by keeping a fixed comparison point.
- **latest prior** — the most recent snapshot before the current run.
  Surfaces day-over-day change.

Storage layout::

  .evalview/model_snapshots/
    <safe-model-id>/
      2026-04-09T14-03-11Z.json        # timestamped snapshots
      2026-04-09T14-18-44Z.json
      reference.json                    # copy of the pinned reference

This file is the counterpart of ``core/mcp_contract.py`` — same idea, same
shape, different domain. Keep the two in sync when extending patterns.
"""
from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Models
# --------------------------------------------------------------------------- #


class ModelCheckPromptResult(BaseModel):
    """Per-prompt structural result, aggregated across N runs."""

    prompt_id: str
    category: str
    pass_rate: float = Field(ge=0.0, le=1.0)
    n_runs: int = Field(ge=1)
    per_run_passed: List[bool]
    latency_ms_mean: Optional[float] = None
    latency_ms_stdev: Optional[float] = None
    notes: Optional[str] = None

    @property
    def passed(self) -> bool:
        """True if every run passed (strict structural success)."""
        return self.pass_rate >= 0.999


class ModelSnapshotMetadata(BaseModel):
    """Metadata captured alongside every model snapshot.

    Versioned explicitly so future snapshot format changes can be detected
    and rejected with a clear error instead of silently misbehaving.
    """

    schema_version: int = 1
    model_id: str
    provider: str
    snapshot_at: datetime
    suite_name: str
    suite_version: str
    suite_hash: str
    temperature: float
    top_p: float
    runs_per_prompt: int
    provider_fingerprint: Optional[str] = None
    fingerprint_confidence: str = "weak"  # "strong" | "medium" | "weak"
    is_reference: bool = False
    cost_total_usd: float = 0.0
    evalview_version: Optional[str] = None
    notes: Optional[str] = None


class ModelSnapshot(BaseModel):
    """A full model snapshot: metadata plus per-prompt results."""

    metadata: ModelSnapshotMetadata
    results: List[ModelCheckPromptResult] = Field(default_factory=list)

    @property
    def overall_pass_rate(self) -> float:
        if not self.results:
            return 0.0
        return sum(r.pass_rate for r in self.results) / len(self.results)

    @property
    def passed_count(self) -> int:
        return sum(1 for r in self.results if r.passed)

    @property
    def total_count(self) -> int:
        return len(self.results)


# --------------------------------------------------------------------------- #
# Store
# --------------------------------------------------------------------------- #


class SnapshotSuiteMismatchError(Exception):
    """Raised when a drift comparison is attempted across incompatible suites.

    The canary suite is content-hashed; changing any prompt, scorer, or
    expected outcome changes the hash and invalidates comparisons against
    older snapshots. Callers must catch this and guide the user to re-pin
    the reference snapshot.
    """


# Exactly one timestamp format used everywhere. Kept strict so filenames
# round-trip through parsing without ambiguity. Includes microseconds so
# rapid back-to-back saves never collide on disk.
_TIMESTAMP_FMT = "%Y-%m-%dT%H-%M-%S.%fZ"
_SAFE_NAME_RE = re.compile(r"[^A-Za-z0-9._-]")


def _safe_name(raw: str) -> str:
    """Replace filesystem-unsafe characters while keeping names readable."""
    return _SAFE_NAME_RE.sub("_", raw)


def _format_timestamp(ts: datetime) -> str:
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    return ts.astimezone(timezone.utc).strftime(_TIMESTAMP_FMT)


class ModelSnapshotStore:
    """Manages on-disk storage of model snapshots.

    All file I/O is synchronous. The store is intentionally dumb — no
    indexing, no caching. Fifty snapshots per model at ~10 KB each is
    trivial to scan.
    """

    def __init__(self, base_path: Optional[Path] = None):
        self.base_path = base_path or Path(".")
        self.root = self.base_path / ".evalview" / "model_snapshots"

    # ------------------------------------------------------------------ #
    # Path helpers
    # ------------------------------------------------------------------ #

    def _model_dir(self, model_id: str) -> Path:
        return self.root / _safe_name(model_id)

    def _snapshot_path(self, model_id: str, ts: datetime) -> Path:
        return self._model_dir(model_id) / f"{_format_timestamp(ts)}.json"

    def _reference_path(self, model_id: str) -> Path:
        return self._model_dir(model_id) / "reference.json"

    # ------------------------------------------------------------------ #
    # Save / load
    # ------------------------------------------------------------------ #

    def save_snapshot(self, snapshot: ModelSnapshot) -> Path:
        """Persist a snapshot. Auto-pins as reference if none exists yet."""
        model_dir = self._model_dir(snapshot.metadata.model_id)
        model_dir.mkdir(parents=True, exist_ok=True)

        path = self._snapshot_path(snapshot.metadata.model_id, snapshot.metadata.snapshot_at)
        path.write_text(snapshot.model_dump_json(indent=2))
        logger.info("Saved model snapshot: %s", path)

        # First-ever snapshot becomes the reference automatically so users
        # can diff on the second run without any extra flags.
        if not self._reference_path(snapshot.metadata.model_id).exists():
            self._write_reference(snapshot)

        return path

    def _write_reference(self, snapshot: ModelSnapshot) -> None:
        ref = snapshot.model_copy(deep=True)
        ref.metadata.is_reference = True
        ref_path = self._reference_path(snapshot.metadata.model_id)
        ref_path.write_text(ref.model_dump_json(indent=2))
        logger.info("Pinned reference snapshot: %s", ref_path)

    def pin_reference(self, model_id: str, snapshot: ModelSnapshot) -> None:
        """Explicitly replace the reference for a model."""
        self._model_dir(model_id).mkdir(parents=True, exist_ok=True)
        self._write_reference(snapshot)

    def reset_reference(self, model_id: str) -> bool:
        """Delete the reference so the next snapshot becomes the new one.

        Returns True if a reference was deleted, False if there was none.
        """
        ref_path = self._reference_path(model_id)
        if ref_path.exists():
            ref_path.unlink()
            return True
        return False

    def load_reference(self, model_id: str) -> Optional[ModelSnapshot]:
        path = self._reference_path(model_id)
        if not path.exists():
            return None
        return ModelSnapshot.model_validate_json(path.read_text())

    def load_latest(self, model_id: str, *, exclude: Optional[Path] = None) -> Optional[ModelSnapshot]:
        """Load the most recent timestamped snapshot for a model.

        Args:
            model_id: model identifier
            exclude: optional path to skip (typically the snapshot just saved
                by the current run, so "latest prior" means *before* now)
        """
        entries = self._list_snapshot_files(model_id)
        if exclude is not None:
            exclude_resolved = exclude.resolve()
            entries = [p for p in entries if p.resolve() != exclude_resolved]
        if not entries:
            return None
        latest = entries[-1]  # already sorted ascending
        return ModelSnapshot.model_validate_json(latest.read_text())

    def list_snapshots(self, model_id: str) -> List[ModelSnapshotMetadata]:
        """List metadata for every timestamped snapshot, oldest first."""
        out: List[ModelSnapshotMetadata] = []
        for path in self._list_snapshot_files(model_id):
            try:
                data = json.loads(path.read_text())
                out.append(ModelSnapshotMetadata.model_validate(data["metadata"]))
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning("Failed to read snapshot %s: %s", path, exc)
        return out

    def _list_snapshot_files(self, model_id: str) -> List[Path]:
        model_dir = self._model_dir(model_id)
        if not model_dir.exists():
            return []
        # Reference file is stored in the same directory but is not a
        # timestamped snapshot; filter it out here so "latest" and "list"
        # only see real runs.
        entries = [p for p in model_dir.glob("*.json") if p.name != "reference.json"]
        entries.sort(key=lambda p: p.name)
        return entries

    # ------------------------------------------------------------------ #
    # Maintenance
    # ------------------------------------------------------------------ #

    def prune(self, model_id: str, keep_last: int = 50) -> int:
        """Delete all but the most recent `keep_last` timestamped snapshots.

        Never touches the reference file. Returns the number of files deleted.
        """
        entries = self._list_snapshot_files(model_id)
        if len(entries) <= keep_last:
            return 0
        to_delete = entries[:-keep_last]
        for path in to_delete:
            try:
                path.unlink()
            except OSError as exc:  # pragma: no cover - defensive
                logger.warning("Failed to prune %s: %s", path, exc)
        return len(to_delete)

    def list_models(self) -> List[str]:
        """List model ids (directory names) that have any snapshots."""
        if not self.root.exists():
            return []
        return sorted(p.name for p in self.root.iterdir() if p.is_dir())

    # ------------------------------------------------------------------ #
    # Compatibility checks
    # ------------------------------------------------------------------ #

    @staticmethod
    def assert_comparable(current: ModelSnapshot, other: ModelSnapshot) -> None:
        """Refuse to compare snapshots produced by different suites or configs.

        Raises SnapshotSuiteMismatchError with a message that the CLI can
        surface directly to the user. We check the signals that actually
        matter for drift detection:

        - suite_hash: if the canary changed, old results mean nothing
        - temperature / top_p: sampling configuration must match
        - provider: comparing OpenAI vs Anthropic is nonsense
        """
        cm = current.metadata
        om = other.metadata
        if cm.suite_hash != om.suite_hash:
            raise SnapshotSuiteMismatchError(
                f"Suite hash differs: current {cm.suite_hash[:12]}… vs "
                f"prior {om.suite_hash[:12]}…. The canary suite changed; old "
                f"snapshots are not comparable. Run with --reset-reference to "
                f"start a new baseline."
            )
        if cm.temperature != om.temperature or cm.top_p != om.top_p:
            raise SnapshotSuiteMismatchError(
                f"Sampling configuration differs: current temp={cm.temperature} "
                f"top_p={cm.top_p}, prior temp={om.temperature} top_p={om.top_p}. "
                f"Drift comparisons require identical sampling configuration."
            )
        if cm.provider != om.provider:
            raise SnapshotSuiteMismatchError(
                f"Provider differs: current '{cm.provider}' vs prior "
                f"'{om.provider}'. Cross-provider comparisons are not supported."
            )


__all__ = [
    "ModelCheckPromptResult",
    "ModelSnapshot",
    "ModelSnapshotMetadata",
    "ModelSnapshotStore",
    "SnapshotSuiteMismatchError",
]
