"""VCR-style record/replay cassettes for hermetic tool replay.

The :mod:`simulation` engine already exposes a tool-mock seam via the
``tool_executor`` adapter attribute. Cassettes turn that into a
record-once / replay-forever workflow:

* :class:`RecordingToolExecutor` wraps a real ``tool_executor`` and
  captures every ``(tool, params) -> result`` (or error) into an
  in-memory :class:`Cassette`. The orchestrator persists it to
  ``.evalview/cassettes/<test-name>.json`` after the run.
* :class:`ReplayToolExecutor` serves calls from a previously recorded
  cassette, consuming entries per-tool in declaration order. Mismatches
  raise :class:`CassetteMismatchError` under ``strict=True``; under
  the default lenient mode they fall through to the wrapped real
  executor (or return ``None`` if there is none).

The matching strategy is **per-tool sequential**: calls to a given
tool name consume that tool's recorded entries in order. This is
robust to inter-tool ordering drift (the agent may legitimately call
``lookup`` before or after ``check_policy`` on different runs) while
still pinning intra-tool state (a tool called twice with the same
params can return different values).

Scope of the v1 cassette: tool calls only. LLM-response and HTTP
cassettes follow the same shape but require adapter opt-in via
``install_mock_interceptor`` and are tracked separately under the
``response`` / ``http`` interaction kinds for future expansion. The
JSON schema reserves the fields today so cassettes recorded now
remain forward-compatible.
"""

from __future__ import annotations

import json
import logging
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Deque, Dict, List, Optional

from evalview.core.types import Cassette, Interaction

logger = logging.getLogger(__name__)


CASSETTE_FORMAT_VERSION = 1
DEFAULT_CASSETTE_DIR = Path(".evalview/cassettes")


ToolExecutor = Callable[[str, Dict[str, Any]], Any]


class CassetteError(RuntimeError):
    """Base class for cassette errors."""


class CassetteMismatchError(CassetteError):
    """Raised when a strict replay encounters an unmatched call."""


def cassette_path_for(test_name: str, root: Path = DEFAULT_CASSETTE_DIR) -> Path:
    """Default on-disk location for a test's cassette.

    Slashes in the name are replaced with ``__`` so multi-segment names
    (e.g. ``billing/refund``) stay flat under the cassette dir.
    """
    safe = test_name.replace("/", "__").replace("\\", "__")
    return root / f"{safe}.json"


def save_cassette(cassette: Cassette, path: Path) -> None:
    """Write a cassette to disk, creating parent directories as needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(cassette.model_dump_json(indent=2))


def load_cassette(path: Path) -> Cassette:
    """Load a cassette JSON, validating the schema."""
    raw = json.loads(path.read_text())
    if raw.get("version", 0) > CASSETTE_FORMAT_VERSION:
        raise CassetteError(
            f"Cassette {path} was recorded with format v{raw['version']} "
            f"but this evalview only understands up to v{CASSETTE_FORMAT_VERSION}. "
            "Upgrade evalview or re-record."
        )
    return Cassette.model_validate(raw)


@dataclass
class RecordingToolExecutor:
    """Wraps a real executor and records every call.

    On each call, the wrapped executor runs first; the result (or the
    error it raised) is then appended to ``interactions``. Errors are
    captured and re-raised so the agent sees the same behavior it would
    see live — recording is transparent.
    """

    real: ToolExecutor
    interactions: List[Interaction] = field(default_factory=list)

    def __call__(self, tool_name: str, params: Dict[str, Any]) -> Any:
        try:
            result = self.real(tool_name, params)
        except Exception as exc:
            self.interactions.append(Interaction(
                kind="tool",
                tool=tool_name,
                params=dict(params or {}),
                returns=None,
                error=f"{type(exc).__name__}: {exc}",
            ))
            raise
        self.interactions.append(Interaction(
            kind="tool",
            tool=tool_name,
            params=dict(params or {}),
            returns=result,
            error=None,
        ))
        return result


class ReplayToolExecutor:
    """Serves tool calls from a cassette, consuming per-tool in order.

    Per-tool sequential matching: each tool name has its own queue of
    recorded interactions. Successive calls to the same tool consume
    its queue in declaration order, which keeps replay deterministic
    even when the agent shuffles inter-tool ordering between runs.

    Behavior on a miss:
    - ``strict=True``: raise :class:`CassetteMismatchError`.
    - ``strict=False``: fall through to ``real`` if provided; else
      return ``None`` and log at DEBUG.
    """

    def __init__(
        self,
        cassette: Cassette,
        real: Optional[ToolExecutor] = None,
        strict: bool = False,
    ) -> None:
        self._cassette = cassette
        self._real = real
        self._strict = strict
        # Deques give O(1) popleft; lists are O(n) on pop(0), which
        # would degrade replay of long recordings.
        self._queues: Dict[str, Deque[Interaction]] = {}
        for entry in cassette.interactions:
            if entry.kind != "tool" or entry.tool is None:
                continue
            self._queues.setdefault(entry.tool, deque()).append(entry)
        self.replays: List[Interaction] = []

    def __call__(self, tool_name: str, params: Dict[str, Any]) -> Any:
        queue = self._queues.get(tool_name)
        if queue:
            entry = queue.popleft()
            self.replays.append(entry)
            if entry.error:
                raise RuntimeError(entry.error)
            return entry.returns

        if self._strict:
            raise CassetteMismatchError(
                f"Cassette has no remaining recording for tool '{tool_name}' "
                f"(strict mode). Re-record with --record or relax to lenient."
            )
        if self._real is not None:
            logger.debug(
                "Cassette miss for '%s'; falling through to real executor.",
                tool_name,
            )
            return self._real(tool_name, params)
        logger.debug("Cassette miss for '%s' with no real executor; returning None.", tool_name)
        return None

    def remaining(self) -> Dict[str, int]:
        """Per-tool counts of unused recordings — surfaces over/under-replay."""
        return {tool: len(q) for tool, q in self._queues.items() if q}


def new_cassette(test_name: str, adapter: Optional[str] = None) -> Cassette:
    """Build an empty cassette stamped with the current UTC time."""
    return Cassette(
        test_name=test_name,
        recorded_at=datetime.now(timezone.utc).isoformat(),
        adapter=adapter,
        interactions=[],
    )


__all__ = [
    "CASSETTE_FORMAT_VERSION",
    "DEFAULT_CASSETTE_DIR",
    "Cassette",
    "CassetteError",
    "CassetteMismatchError",
    "Interaction",
    "RecordingToolExecutor",
    "ReplayToolExecutor",
    "cassette_path_for",
    "load_cassette",
    "new_cassette",
    "save_cassette",
]
