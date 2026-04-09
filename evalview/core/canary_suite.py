"""Loader and hasher for canary suites used by `evalview model-check`.

Canary suites are *not* regular EvalView test suites. They use a much
simpler schema — each prompt has a prompt string, a scorer name, and a
scorer-specific ``expected`` block — because the canary is run by a
different command against the raw provider, not an agent, and none of
the machinery of ``core/types.py:TestCase`` applies.

We keep the schema tight and validate aggressively. Every drift
comparison depends on the suite hash being stable, so silent acceptance
of malformed YAML would be worse than a hard failure.
"""
from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml  # type: ignore[import-untyped]

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Dataclasses
# --------------------------------------------------------------------------- #


@dataclass
class CanaryPrompt:
    """A single canary prompt with its structural scoring config."""

    id: str
    category: str
    prompt: str
    scorer: str
    expected: Dict[str, Any] = field(default_factory=dict)
    notes: Optional[str] = None


@dataclass
class CanarySuite:
    """A loaded canary suite with metadata plus a content hash.

    ``suite_hash`` is computed over the raw YAML bytes so ANY change —
    prompt text, scorer, expected block, even whitespace inside a quoted
    string — yields a new hash and invalidates prior snapshots. This is
    intentional: if the suite changes, drift comparisons are meaningless.
    """

    suite_name: str
    version: str
    description: str
    prompts: List[CanaryPrompt]
    suite_hash: str
    source_path: Optional[Path] = None


# --------------------------------------------------------------------------- #
# Loader
# --------------------------------------------------------------------------- #


_VALID_SCORERS = {"tool_choice", "json_schema", "refusal", "exact_match"}


class CanarySuiteError(ValueError):
    """Raised when a canary suite fails to load or validate."""


def hash_suite_bytes(raw: bytes) -> str:
    """Canonical SHA-256 hash of raw suite bytes, prefixed with 'sha256:'."""
    return "sha256:" + hashlib.sha256(raw).hexdigest()


def load_canary_suite(path: Path) -> CanarySuite:
    """Load and validate a canary suite YAML file.

    Raises:
        CanarySuiteError: on any structural problem. The message is safe
            to surface directly to the CLI user.
    """
    if not path.exists():
        raise CanarySuiteError(f"Canary suite not found: {path}")

    raw = path.read_bytes()
    try:
        data = yaml.safe_load(raw)
    except yaml.YAMLError as exc:
        raise CanarySuiteError(f"Invalid YAML in {path}: {exc}") from exc

    if not isinstance(data, dict):
        raise CanarySuiteError(
            f"Canary suite root must be a mapping; got {type(data).__name__}"
        )

    suite_name = data.get("suite_name")
    version = data.get("version")
    if not suite_name or not version:
        raise CanarySuiteError(
            "Canary suite must declare 'suite_name' and 'version'"
        )

    description = data.get("description") or ""
    raw_prompts = data.get("prompts")
    if not isinstance(raw_prompts, list) or not raw_prompts:
        raise CanarySuiteError("Canary suite must contain a non-empty 'prompts' list")

    prompts: List[CanaryPrompt] = []
    seen_ids: set[str] = set()
    for idx, entry in enumerate(raw_prompts):
        if not isinstance(entry, dict):
            raise CanarySuiteError(f"Prompt #{idx} must be a mapping")

        pid = entry.get("id")
        if not pid or not isinstance(pid, str):
            raise CanarySuiteError(f"Prompt #{idx} is missing a string 'id'")
        if pid in seen_ids:
            raise CanarySuiteError(f"Duplicate prompt id: {pid!r}")
        seen_ids.add(pid)

        scorer = entry.get("scorer")
        if scorer not in _VALID_SCORERS:
            raise CanarySuiteError(
                f"Prompt '{pid}': unknown scorer {scorer!r}. "
                f"Valid: {sorted(_VALID_SCORERS)}"
            )

        category = entry.get("category") or scorer
        prompt_text = entry.get("prompt")
        if not prompt_text or not isinstance(prompt_text, str):
            raise CanarySuiteError(f"Prompt '{pid}': missing or non-string 'prompt'")

        expected = entry.get("expected")
        if expected is None:
            expected = {}
        if not isinstance(expected, dict):
            raise CanarySuiteError(
                f"Prompt '{pid}': 'expected' must be a mapping if present"
            )

        prompts.append(
            CanaryPrompt(
                id=pid,
                category=str(category),
                prompt=prompt_text,
                scorer=str(scorer),
                expected=expected,
                notes=entry.get("notes"),
            )
        )

    return CanarySuite(
        suite_name=str(suite_name),
        version=str(version),
        description=str(description),
        prompts=prompts,
        suite_hash=hash_suite_bytes(raw),
        source_path=path,
    )


__all__ = [
    "CanaryPrompt",
    "CanarySuite",
    "CanarySuiteError",
    "hash_suite_bytes",
    "load_canary_suite",
]
