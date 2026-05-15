"""Goal-drift detection — is the agent still working on what the user asked?

Decision-rationale logging answers *what* the agent chose at each step.
Goal-drift answers the question one layer above: *is the trajectory still
about the original ask, or did the agent quietly wander?*

The classic failure mode:

    User: "Cancel my subscription and refund the last charge."
    Agent: looks up account → checks plan → reviews terms → … → answers
           a question about pricing tiers, never cancels.

By the time the user sees the response, the trajectory is 12 steps deep
and tracing through it is painful. A drift signal at step 6 ("the agent
is no longer working on a 'cancel + refund' goal — current trajectory
looks more like a 'pricing question' goal") catches it cheaply.

This module ships a **deterministic baseline** (Jaccard token overlap
between the stated goal and a trajectory-derived intent summary) plus a
plug-in slot for an LLM judge. The deterministic baseline is intentionally
crude: it fires on the obviously-wandered cases (token overlap collapses
to <0.2) without hitting an LLM. The judge slot is where smarter
contributors can drop in something better.

Pure module — no I/O, no network, no LLM by default. The judge interface
takes a callable so callers wire whatever they want.

Contributor recipe: ``docs/agent-recipes/add-goal-drift-judge.md``.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable, Iterable, List, Optional, Sequence, Tuple


# ── Tunables ────────────────────────────────────────────────────────────────

# Below this Jaccard, the deterministic baseline calls it drift.
# 0.2 is permissive on purpose — Jaccard is noisy and we'd rather miss
# borderline drift than spam the user with false positives.
DEFAULT_DRIFT_THRESHOLD = 0.2

# Goal text and trajectory text get clipped to this many chars before
# tokenization. Long goals are usually pasted-in tickets; long
# trajectories drown the signal in repeated boilerplate.
_MAX_TEXT_CHARS = 4096


# Mirror the small stoplist in evalview.core.freshness so the two modules
# behave consistently when a future refactor unifies them.
_STOPWORDS = frozenset({
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "do", "does", "did", "doing", "have", "has", "had", "having",
    "i", "you", "he", "she", "it", "we", "they", "me", "him", "her", "us", "them",
    "my", "your", "his", "its", "our", "their", "mine", "yours", "ours",
    "this", "that", "these", "those",
    "and", "or", "but", "if", "then", "else", "of", "in", "on", "at", "to",
    "for", "with", "by", "from", "as", "about", "into", "than",
    "can", "could", "would", "should", "will", "shall", "may", "might", "must",
    "not", "no", "so", "just", "also", "very", "really", "please",
})


# ── Data shapes ─────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class GoalEvent:
    """One step in the trajectory carrying signal about current intent.

    Sourced from whatever the adapter has — a model thought trace, a
    tool name, a free-text plan node, an intermediate output. Keep the
    text short (the module truncates anyway).
    """

    step_index: int
    text: str
    kind: str = "step"
    """``goal`` | ``thought`` | ``tool_choice`` | ``tool_call`` | ``output``."""


@dataclass(frozen=True)
class GoalDriftAnalysis:
    """Result of analyzing one trajectory against a stated goal."""

    stated_goal: str
    trajectory_summary: str
    similarity: float          # [0.0, 1.0] — higher = more on-goal
    drift_delta: float         # 1.0 - similarity, what OTel attribute carries
    is_drifting: bool
    threshold: float
    judge_used: bool
    evidence: dict = field(default_factory=dict)

    @property
    def severity(self) -> str:
        """Coarse label for digest rendering."""
        if not self.is_drifting:
            return "on_goal"
        if self.similarity < self.threshold * 0.5:
            return "severe"
        return "mild"


# ── Tokenization (kept local to avoid coupling to freshness module) ─────────


def _tokens(text: str) -> frozenset[str]:
    """Lower / strip / collapse digits / drop stopwords → token set.

    Same digit normalization as the freshness module: order numbers and
    other unique IDs collapse to ``<num>`` so they don't shred otherwise-
    similar text. Stopwords are dropped to amplify content overlap.
    """
    if not text:
        return frozenset()
    truncated = text[:_MAX_TEXT_CHARS].lower()
    truncated = re.sub(r"\d+", " <num> ", truncated)
    cleaned = re.sub(r"[^a-z0-9<>\s]+", " ", truncated)
    return frozenset(
        t for t in cleaned.split()
        if t and t not in _STOPWORDS and len(t) > 1
    )


def _jaccard(a: frozenset[str], b: frozenset[str]) -> float:
    if not a or not b:
        return 0.0
    union = len(a | b)
    return (len(a & b) / union) if union else 0.0


# ── Trajectory summarization ────────────────────────────────────────────────


def summarize_trajectory(events: Sequence[GoalEvent]) -> str:
    """Produce a single text summary of where the agent ended up.

    Strategy: weight the last events most heavily — the trajectory's
    *current* intent is what we care about, not the opening few steps
    (those usually echo the goal). Concatenates the text of the last
    ``min(8, n)`` events; deliberately doesn't try to be smart about
    deduplication — the Jaccard similarity is the smart layer.
    """
    if not events:
        return ""
    tail = list(events)[-8:]
    return " | ".join(e.text for e in tail if e.text)


# ── Detector interface ──────────────────────────────────────────────────────


GoalDriftJudge = Callable[[str, str], Optional[float]]
"""Signature for a pluggable LLM judge.

The callable receives ``(stated_goal, trajectory_summary)`` and returns
a float in ``[0.0, 1.0]`` (higher = more on-goal) or ``None`` to fall
back to the deterministic baseline. Returning None lets adapters fail
soft on judge errors without breaking the analysis.
"""


def analyze_goal_drift(
    stated_goal: str,
    events: Iterable[GoalEvent],
    *,
    threshold: float = DEFAULT_DRIFT_THRESHOLD,
    judge: Optional[GoalDriftJudge] = None,
) -> GoalDriftAnalysis:
    """Return a :class:`GoalDriftAnalysis` for one trajectory.

    With no ``judge`` (default), uses Jaccard token overlap between the
    stated goal and a trajectory summary. With a ``judge``, calls it and
    falls back to Jaccard when it returns None.

    Empty trajectories are not drifting — there's no trajectory yet to
    drift from. Empty goals likewise yield similarity 0.0 but
    ``is_drifting=False`` because we can't meaningfully say *what* the
    agent should be on.
    """
    summary = summarize_trajectory(list(events))

    if not stated_goal.strip() or not summary.strip():
        return GoalDriftAnalysis(
            stated_goal=stated_goal,
            trajectory_summary=summary,
            similarity=0.0,
            drift_delta=0.0,
            is_drifting=False,
            threshold=threshold,
            judge_used=False,
            evidence={"reason": "missing_goal_or_trajectory"},
        )

    judge_used = False
    similarity: Optional[float] = None
    if judge is not None:
        try:
            similarity = judge(stated_goal, summary)
            judge_used = similarity is not None
        except Exception:
            # The judge is allowed to fail; the deterministic baseline
            # is the safety net.
            similarity = None

    if similarity is None:
        similarity = _jaccard(_tokens(stated_goal), _tokens(summary))

    similarity = max(0.0, min(1.0, similarity))
    drift_delta = round(1.0 - similarity, 4)
    is_drifting = similarity < threshold

    return GoalDriftAnalysis(
        stated_goal=stated_goal,
        trajectory_summary=summary,
        similarity=round(similarity, 4),
        drift_delta=drift_delta,
        is_drifting=is_drifting,
        threshold=threshold,
        judge_used=judge_used,
        evidence={
            "tokens_goal": sorted(_tokens(stated_goal))[:20],
            "tokens_trajectory": sorted(_tokens(summary))[:20],
        },
    )


def analyze_per_step(
    stated_goal: str,
    events: Sequence[GoalEvent],
    *,
    threshold: float = DEFAULT_DRIFT_THRESHOLD,
    judge: Optional[GoalDriftJudge] = None,
) -> List[Tuple[int, GoalDriftAnalysis]]:
    """Run the drift analysis at each step prefix.

    Useful for "when did the agent wander?" answers: returns
    ``[(step_index, analysis), ...]`` so callers can render a sparkline
    of similarity over the trajectory and locate the elbow where drift
    started.

    Cost note: with a judge plugged in, this calls the judge N times.
    The deterministic baseline is cheap enough that calling it per-step
    on long trajectories is fine.
    """
    out: List[Tuple[int, GoalDriftAnalysis]] = []
    prefix: List[GoalEvent] = []
    for e in events:
        prefix.append(e)
        out.append((
            e.step_index,
            analyze_goal_drift(
                stated_goal, prefix,
                threshold=threshold, judge=judge,
            ),
        ))
    return out
