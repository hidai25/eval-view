"""
Flake Quarantine — known-flaky tests that run but don't block CI.

Quarantined tests still execute and report, but failures are excluded from
the exit code calculation. This prevents teams from disabling CI gating
because of a few noisy tests.

Storage: `.evalview/quarantine.yaml`
"""

import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import yaml


QUARANTINE_FILE = ".evalview/quarantine.yaml"


@dataclass
class QuarantineEntry:
    """A single quarantined test."""

    test_name: str
    reason: str = ""
    added_at: str = ""
    flaky_count: int = 0  # how many times classified as flaky

    def to_dict(self) -> dict:
        return {
            "test_name": self.test_name,
            "reason": self.reason,
            "added_at": self.added_at,
            "flaky_count": self.flaky_count,
        }


@dataclass
class QuarantineStore:
    """Manages the quarantine list."""

    path: Path = field(default_factory=lambda: Path(QUARANTINE_FILE))
    entries: Dict[str, QuarantineEntry] = field(default_factory=dict)

    def __post_init__(self):
        self._load()

    def _load(self):
        if not self.path.exists():
            return
        try:
            data = yaml.safe_load(self.path.read_text()) or {}
            for name, info in data.get("quarantined", {}).items():
                self.entries[name] = QuarantineEntry(
                    test_name=name,
                    reason=info.get("reason", ""),
                    added_at=info.get("added_at", ""),
                    flaky_count=info.get("flaky_count", 0),
                )
        except Exception:
            pass

    def _save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "quarantined": {
                e.test_name: {
                    "reason": e.reason,
                    "added_at": e.added_at,
                    "flaky_count": e.flaky_count,
                }
                for e in self.entries.values()
            }
        }
        self.path.write_text(yaml.dump(data, default_flow_style=False, sort_keys=False))

    def add(self, test_name: str, reason: str = "") -> QuarantineEntry:
        entry = QuarantineEntry(
            test_name=test_name,
            reason=reason,
            added_at=datetime.now(timezone.utc).isoformat(),
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

    def list_all(self) -> List[QuarantineEntry]:
        return list(self.entries.values())
