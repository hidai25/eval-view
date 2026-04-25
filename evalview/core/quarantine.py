"""
Flake Quarantine — known-flaky tests that run but don't block CI.

Quarantined tests still execute and report, but failures are excluded from
the exit code calculation. This prevents teams from disabling CI gating
because of a few noisy tests.

Week 2 adds **governance** on top of convenience:

    - `owner`: who quarantined it (required on add)
    - `reason`: why (required on add — no silent quarantining)
    - `added_at` / `review_after_days` / `expiry_date`
    - `flaky_count_history`: rolling trend of flaky_count per check
    - `stale` property: true when the review window is overdue

Without governance, quarantine becomes the "misc drawer of shame" — tests
bury themselves in YAML forever. The stale flag surfaces rot automatically
in `evalview check`, `evalview since`, and the PR comment.

Storage: `.evalview/quarantine.yaml`
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import yaml

logger = logging.getLogger(__name__)


QUARANTINE_FILE = ".evalview/quarantine.yaml"

# Default review window — if a test sits in quarantine longer than this
# without its flaky_count improving, it's flagged stale.
DEFAULT_REVIEW_DAYS = 14

# Maximum number of historical flaky_count values we retain per entry.
# Keeps the YAML file bounded without losing enough signal to see a trend.
_MAX_HISTORY_LEN = 20


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _parse_iso(raw: str) -> Optional[datetime]:
    if not raw:
        return None
    try:
        return datetime.fromisoformat(raw)
    except ValueError:
        return None


@dataclass
class QuarantineEntry:
    """A single quarantined test with governance metadata.

    `owner` and `reason` are intentionally required on creation (enforced
    by `QuarantineStore.add`) — silent quarantining is the failure mode
    we're trying to prevent. Existing entries loaded from disk that lack
    these fields are tolerated for backwards compatibility but will show
    up as "unknown owner" in the list view.
    """

    test_name: str
    reason: str = ""
    owner: str = ""
    added_at: str = ""
    expiry_date: str = ""  # ISO string; empty means "no hard expiry"
    review_after_days: int = DEFAULT_REVIEW_DAYS
    flaky_count: int = 0
    flaky_count_history: List[int] = field(default_factory=list)

    # ── governance helpers ──

    @property
    def age_days(self) -> Optional[int]:
        added = _parse_iso(self.added_at)
        if added is None:
            return None
        now = datetime.now(timezone.utc)
        if added.tzinfo is None:
            added = added.replace(tzinfo=timezone.utc)
        return max(0, (now - added).days)

    @property
    def is_expired(self) -> bool:
        """Has the hard expiry_date passed?"""
        dt = _parse_iso(self.expiry_date)
        if dt is None:
            return False
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return datetime.now(timezone.utc) >= dt

    @property
    def is_review_overdue(self) -> bool:
        """Has the soft review window elapsed?"""
        age = self.age_days
        if age is None:
            return False
        return age > max(1, self.review_after_days)

    @property
    def stale(self) -> bool:
        """Stale = past hard expiry OR review overdue without improvement.

        "Without improvement" means the flaky_count trend is flat or
        rising. A test whose flaky_count is decreasing shouldn't be
        flagged just because the calendar says so.
        """
        if self.is_expired:
            return True
        if not self.is_review_overdue:
            return False
        return self.flaky_trend != "down"

    @property
    def flaky_trend(self) -> str:
        """Return 'up', 'down', or 'flat' based on flaky_count_history.

        Uses the last vs first sample in the retained window — this is
        cheap and matches the ↗/→/↘ glyph shown in `quarantine list`.
        """
        hist = self.flaky_count_history
        if len(hist) < 2:
            return "flat"
        delta = hist[-1] - hist[0]
        if delta > 0:
            return "up"
        if delta < 0:
            return "down"
        return "flat"

    def to_dict(self) -> dict:
        return {
            "test_name": self.test_name,
            "reason": self.reason,
            "owner": self.owner,
            "added_at": self.added_at,
            "expiry_date": self.expiry_date,
            "review_after_days": self.review_after_days,
            "flaky_count": self.flaky_count,
            "flaky_count_history": list(self.flaky_count_history),
            # Computed (convenience for JSON consumers — not stored on disk)
            "age_days": self.age_days,
            "stale": self.stale,
            "flaky_trend": self.flaky_trend,
        }


class QuarantineOwnerRequired(ValueError):
    """Raised when `add()` is called without an owner.

    Silent quarantining is the anti-pattern we're preventing; a CLI that
    refuses to quarantine without attribution is the fix.
    """


class QuarantineReasonRequired(ValueError):
    """Raised when `add()` is called without a reason."""


class QuarantineLoadError(RuntimeError):
    """Raised when the quarantine file exists but can't be parsed.

    Callers that want to treat this as a warning can catch it; anything
    that writes must refuse to do so until the corruption is resolved.
    """


@dataclass
class QuarantineStore:
    """Manages the quarantine list.

    Governance is enforced at the store level (not just the CLI) so any
    programmatic caller — tests, scripts, plugins — gets the same
    guarantees.

    Corruption safety:
        `_load` distinguishes three cases:
          1. File missing           → fresh store, normal operation
          2. File present, parses   → entries populated
          3. File present, corrupt  → `safe_mode` engaged:
                                         - entries stay empty
                                         - `_save` refuses to write
                                         - a loud warning is logged

        Safe mode is the "stop digging" behavior we want — better to
        leave the corrupt file alone than overwrite it with an empty
        one, which was the old behavior and would erase every
        quarantine entry on the next `increment_flaky()` call.
    """

    path: Path = field(default_factory=lambda: Path(QUARANTINE_FILE))
    entries: Dict[str, QuarantineEntry] = field(default_factory=dict)
    safe_mode: bool = False

    # ── batching support ──
    # When a caller enters a `batch_update` block we buffer writes and
    # flush once on exit. A 50-test check with 10 flaky classifications
    # would otherwise do 10 YAML writes; batching makes it one.
    _batching: bool = field(default=False, repr=False)
    _batch_dirty: bool = field(default=False, repr=False)

    def __post_init__(self) -> None:
        try:
            self._load()
        except QuarantineLoadError as exc:
            # Surface to humans (via Rich console, if available) and to
            # logs. We intentionally do NOT re-raise — a corrupted file
            # shouldn't crash every subsequent `evalview check` — but
            # safe_mode prevents the quiet-overwrite failure mode.
            logger.warning("Quarantine file is corrupted: %s", exc)
            try:
                from evalview.commands.shared import console
                console.print(
                    f"[red]⚠  Quarantine file corrupted: {exc}[/red]\n"
                    f"[dim]   File: {self.path}[/dim]\n"
                    "[dim]   Running in safe mode — no quarantine writes "
                    "will happen until this is fixed.[/dim]"
                )
            except Exception:
                pass  # Console rendering must never escalate an error

    # ── persistence ──

    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            raw = self.path.read_text()
        except OSError as exc:
            self.safe_mode = True
            raise QuarantineLoadError(f"cannot read {self.path}: {exc}") from exc

        try:
            data = yaml.safe_load(raw) or {}
        except yaml.YAMLError as exc:
            self.safe_mode = True
            raise QuarantineLoadError(f"invalid YAML in {self.path}: {exc}") from exc

        if not isinstance(data, dict):
            self.safe_mode = True
            raise QuarantineLoadError(
                f"{self.path} did not parse to a dict (got {type(data).__name__})"
            )

        quarantined = data.get("quarantined") or {}
        if not isinstance(quarantined, dict):
            self.safe_mode = True
            raise QuarantineLoadError(
                f"`quarantined` key in {self.path} is not a dict"
            )

        for name, info in quarantined.items():
            if not isinstance(info, dict):
                # Skip malformed individual entries rather than aborting
                # the whole load — the other entries are still useful.
                logger.warning("Skipping malformed quarantine entry: %s", name)
                continue
            self.entries[name] = QuarantineEntry(
                test_name=name,
                reason=info.get("reason", ""),
                owner=info.get("owner", ""),
                added_at=info.get("added_at", ""),
                expiry_date=info.get("expiry_date", ""),
                review_after_days=info.get("review_after_days", DEFAULT_REVIEW_DAYS),
                flaky_count=info.get("flaky_count", 0),
                flaky_count_history=list(info.get("flaky_count_history", []) or []),
            )

    def _save(self) -> None:
        if self.safe_mode:
            # Refuse to overwrite a corrupted file.
            logger.warning(
                "Refusing to save quarantine — store is in safe mode due to a "
                "previous load failure. Fix or delete %s and rerun.",
                self.path,
            )
            return

        if self._batching:
            self._batch_dirty = True
            return

        self.path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "quarantined": {
                e.test_name: {
                    "reason": e.reason,
                    "owner": e.owner,
                    "added_at": e.added_at,
                    "expiry_date": e.expiry_date,
                    "review_after_days": e.review_after_days,
                    "flaky_count": e.flaky_count,
                    "flaky_count_history": list(e.flaky_count_history),
                }
                for e in self.entries.values()
            }
        }
        self.path.write_text(yaml.dump(data, default_flow_style=False, sort_keys=False))

    # ── batching ──

    def batch_update(self) -> "_BatchContext":
        """Buffer writes for the duration of a `with` block.

        Usage:
            with store.batch_update():
                for test_name in flaky_tests:
                    store.increment_flaky(test_name)
            # one YAML write happens here, on exit
        """
        return _BatchContext(self)

    # ── mutations ──

    def add(
        self,
        test_name: str,
        reason: str = "",
        *,
        owner: str = "",
        review_after_days: int = DEFAULT_REVIEW_DAYS,
        expiry_date: str = "",
    ) -> QuarantineEntry:
        """Add a test to the quarantine list.

        Raises:
            QuarantineOwnerRequired: if `owner` is blank
            QuarantineReasonRequired: if `reason` is blank

        Governance is non-optional: the product's job is to prevent the
        quarantine from becoming a dumping ground, and the simplest way
        is to refuse to accept entries that don't carry attribution.
        """
        if not owner.strip():
            raise QuarantineOwnerRequired(
                f"Cannot quarantine '{test_name}' without an owner. "
                "Use --owner @handle so future reviewers know who to ask."
            )
        if not reason.strip():
            raise QuarantineReasonRequired(
                f"Cannot quarantine '{test_name}' without a reason. "
                "Use --reason \"…\" so future reviewers know why."
            )

        entry = QuarantineEntry(
            test_name=test_name,
            reason=reason.strip(),
            owner=owner.strip(),
            added_at=_utcnow_iso(),
            expiry_date=expiry_date.strip(),
            review_after_days=max(1, review_after_days),
        )
        self.entries[test_name] = entry
        self._save()
        return entry

    def remove(self, test_name: str) -> bool:
        if test_name in self.entries:
            del self.entries[test_name]
            self._save()
            return True
        return False

    def is_quarantined(self, test_name: str) -> bool:
        return test_name in self.entries

    def increment_flaky(self, test_name: str) -> int:
        """Record that a test was classified as flaky.

        Updates `flaky_count` and appends to `flaky_count_history`
        (capped at _MAX_HISTORY_LEN). Returns the new count.

        Note: this may be called on a test that is not yet formally
        quarantined; we track the count anyway so `should_quarantine()`
        can trigger auto-quarantine once the threshold is crossed.
        In that case the entry has blank owner/reason — the CLI that
        actually quarantines the test must backfill them.
        """
        if test_name not in self.entries:
            self.entries[test_name] = QuarantineEntry(
                test_name=test_name,
                added_at=_utcnow_iso(),
            )
        entry = self.entries[test_name]
        entry.flaky_count += 1
        entry.flaky_count_history.append(entry.flaky_count)
        if len(entry.flaky_count_history) > _MAX_HISTORY_LEN:
            entry.flaky_count_history = entry.flaky_count_history[-_MAX_HISTORY_LEN:]
        self._save()
        return entry.flaky_count

    def should_quarantine(self, test_name: str, threshold: int = 3) -> bool:
        """Returns True if a test has been flaky enough times to quarantine."""
        entry = self.entries.get(test_name)
        if not entry:
            return False
        return entry.flaky_count >= threshold and not entry.reason

    # ── queries ──

    def list_all(self) -> List[QuarantineEntry]:
        return list(self.entries.values())

    def list_stale(self) -> List[QuarantineEntry]:
        """Return entries whose review window has lapsed without improvement.

        This is the "raccoon drawer" detector — the reason quarantine
        governance exists. Consumed by the verdict layer (Week 1),
        `evalview check` stale alerts (Week 2), and the PR comment
        upgrade (Week 2).
        """
        return [e for e in self.entries.values() if e.stale]


class _BatchContext:
    """Context manager that defers `_save()` calls until exit.

    Paired with `QuarantineStore.batch_update()`. Nested batches collapse
    into the outermost — only the outermost writes on exit.
    """

    def __init__(self, store: QuarantineStore) -> None:
        self._store = store
        self._was_already_batching = False

    def __enter__(self) -> QuarantineStore:
        self._was_already_batching = self._store._batching
        self._store._batching = True
        return self._store

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._was_already_batching:
            return  # Not the outermost context — leave batching on
        self._store._batching = False
        if self._store._batch_dirty:
            self._store._batch_dirty = False
            try:
                self._store._save()
            except Exception:
                logger.exception("Failed to flush batched quarantine writes")
