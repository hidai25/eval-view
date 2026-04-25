"""Decision-rationale capture for agent execution traces.

The :class:`RationaleCollector` is the single entry point adapters use
to record why an agent picked one option over another. Rationales come
from four sources:

1. Model-reported reasoning — Anthropic ``thinking`` blocks, OpenAI
   reasoning summaries on o-series models.
2. Tool-choice dispatch — each assistant turn that picks a tool is a
   ``decision_type="tool_choice"`` event, carrying the chosen tool and
   the list of tools the agent had available.
3. Branch handoffs — multi-agent frameworks (LangGraph, CrewAI) emit
   a ``decision_type="branch"`` event at every node transition.
4. Refusals / retries — surfaced explicitly by adapters when the model
   declines or retries.

The collector enforces the caps defined in
:mod:`evalview.core.types` so a runaway agent can't blow up memory or
wire payload size: at most
:data:`~evalview.core.types.RATIONALE_MAX_EVENTS_PER_RUN` events per
run, each with ``rationale_text`` truncated to
:data:`~evalview.core.types.RATIONALE_MAX_TEXT_BYTES` bytes. Events
past the cap are silently dropped after a one-shot warning; that's the
same shape the cloud Zod validator uses so behavior stays consistent
across the wire.

This module has no network I/O, no LLM calls, and no framework
imports — it's safe to include from every adapter.
"""

from __future__ import annotations

import hashlib
import json
import logging
from typing import Any, Dict, Iterable, List, Optional, Sequence

from evalview.core.types import (
    RATIONALE_MAX_EVENTS_PER_RUN,
    RATIONALE_MAX_TEXT_BYTES,
    DecisionType,
    RationaleEvent,
)

logger = logging.getLogger(__name__)


# Canonical plain-language descriptions for each decision_type value.
# Shared by local HTML replay, CI comments, and cloud UI tooltips so every
# surface explains the enum the same way. Keep keys in sync with the
# ``DecisionType`` literal in :mod:`evalview.core.types`.
DECISION_TYPE_DESCRIPTIONS: Dict[str, str] = {
    "tool_choice": "Agent picked a specific tool from the tools available to it.",
    "branch":      "Agent chose a control-flow path — a node transition in LangGraph / CrewAI, or an if/else in a handwritten agent.",
    "refusal":     "Agent declined to act on the input. Captured explicitly so policy-trained refusals don't look like silent failures.",
    "retry":       "Agent re-attempted a step after a failure. Surfaces retry loops that could mask a systematic bug.",
}


def compute_input_hash(
    prompt: Optional[str] = None,
    tool_state: Optional[Any] = None,
    extra: Optional[Any] = None,
) -> str:
    """Stable sha256 fingerprint of the agent's input state.

    Used by cloud analytics to group decisions across runs: identical
    ``input_hash`` means "same situation" so shifts in ``chosen`` are
    meaningful signal.

    ``tool_state`` and ``extra`` are normalized via ``json.dumps(...,
    sort_keys=True, default=str)`` so Python dicts with reordered keys
    still hash identically. Missing fields contribute empty strings
    rather than raising, so adapters can pass whatever context they
    have.
    """
    parts: List[str] = []
    parts.append(prompt or "")
    for obj in (tool_state, extra):
        if obj is None:
            parts.append("")
            continue
        try:
            parts.append(json.dumps(obj, sort_keys=True, default=str, ensure_ascii=False))
        except Exception:  # pragma: no cover — defensive, json handles most shapes
            parts.append(str(obj))
    joined = "\x1e".join(parts)  # record separator is safe in JSON output
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()


def _truncate_text(text: Optional[str]) -> tuple[Optional[str], bool]:
    """Truncate to the byte cap, returning (new_text, was_truncated)."""
    if text is None:
        return None, False
    encoded = text.encode("utf-8")
    if len(encoded) <= RATIONALE_MAX_TEXT_BYTES:
        return text, False
    # Decode back safely — trim on a UTF-8 boundary.
    trimmed = encoded[:RATIONALE_MAX_TEXT_BYTES].decode("utf-8", errors="ignore")
    return trimmed, True


class RationaleCollector:
    """Per-run buffer of :class:`RationaleEvent` objects.

    Adapters construct one collector at the start of a run, call
    :meth:`capture` from each decision point, and attach
    :meth:`events` to the resulting :class:`ExecutionTrace`.

    Instances are single-threaded and not meant to be shared across
    concurrent runs. Drop events silently once the per-run cap is hit —
    a warning fires on the first drop so operators notice without
    flooding logs.
    """

    __slots__ = ("_events", "_dropped", "_warned")

    def __init__(self) -> None:
        self._events: List[RationaleEvent] = []
        self._dropped: int = 0
        self._warned: bool = False

    # ------------------------------------------------------------------
    # Capture entry points
    # ------------------------------------------------------------------

    def capture(
        self,
        *,
        step_id: str,
        decision_type: DecisionType,
        chosen: str,
        input_hash: str,
        alternatives: Optional[Sequence[str]] = None,
        rationale_text: Optional[str] = None,
        turn: Optional[int] = None,
        model_reported_confidence: Optional[float] = None,
    ) -> Optional[RationaleEvent]:
        """Record a single decision. Returns the stored event, or None if dropped.

        Keyword-only so adapter call-sites stay self-documenting and
        reordering parameters in the future is safe.
        """
        if len(self._events) >= RATIONALE_MAX_EVENTS_PER_RUN:
            self._dropped += 1
            if not self._warned:
                logger.warning(
                    "RationaleCollector hit cap of %d events; subsequent events dropped.",
                    RATIONALE_MAX_EVENTS_PER_RUN,
                )
                self._warned = True
            return None

        text, truncated = _truncate_text(rationale_text)
        event = RationaleEvent(
            step_id=step_id,
            turn=turn,
            decision_type=decision_type,
            chosen=chosen,
            alternatives=list(alternatives or []),
            rationale_text=text,
            input_hash=input_hash,
            model_reported_confidence=model_reported_confidence,
            truncated=truncated,
        )
        self._events.append(event)
        return event

    def capture_tool_choice(
        self,
        *,
        step_id: str,
        chosen_tool: str,
        available_tools: Iterable[str],
        prompt: Optional[str] = None,
        tool_state: Optional[Any] = None,
        rationale_text: Optional[str] = None,
        turn: Optional[int] = None,
        model_reported_confidence: Optional[float] = None,
    ) -> Optional[RationaleEvent]:
        """Convenience wrapper for the common tool_choice shape.

        Computes the input_hash internally from prompt + tool_state and
        filters the chosen tool out of ``alternatives``.
        """
        alternatives = [t for t in available_tools if t and t != chosen_tool]
        input_hash = compute_input_hash(prompt=prompt, tool_state=tool_state)
        return self.capture(
            step_id=step_id,
            decision_type="tool_choice",
            chosen=chosen_tool,
            alternatives=alternatives,
            rationale_text=rationale_text,
            input_hash=input_hash,
            turn=turn,
            model_reported_confidence=model_reported_confidence,
        )

    def capture_branch(
        self,
        *,
        step_id: str,
        chosen_branch: str,
        available_branches: Iterable[str],
        state_summary: Optional[Any] = None,
        rationale_text: Optional[str] = None,
        turn: Optional[int] = None,
    ) -> Optional[RationaleEvent]:
        """Convenience wrapper for multi-agent / graph handoffs."""
        alternatives = [b for b in available_branches if b and b != chosen_branch]
        input_hash = compute_input_hash(prompt=None, tool_state=state_summary)
        return self.capture(
            step_id=step_id,
            decision_type="branch",
            chosen=chosen_branch,
            alternatives=alternatives,
            rationale_text=rationale_text,
            input_hash=input_hash,
            turn=turn,
        )

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    def events(self) -> List[RationaleEvent]:
        """Return the captured events in the order they were recorded."""
        return list(self._events)

    def dropped(self) -> int:
        """Number of events dropped because the cap was hit."""
        return self._dropped

    def __len__(self) -> int:
        return len(self._events)


__all__ = ["RationaleCollector", "compute_input_hash"]
