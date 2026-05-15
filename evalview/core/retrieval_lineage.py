"""Retrieval lineage — which retrieved chunks actually influenced the output?

For RAG agents, the question observability tools mostly fail to answer is:
*"of the 8 chunks I retrieved, which ones did the agent actually use?"*

Without that, you can't tell:
- which chunks are dead weight (never influence outputs → drop from index),
- which chunks dominate (every output cites them → maybe overfit),
- when retrieval quality silently degrades (chunks change but influence
  patterns don't, or vice versa).

This module ships a **deterministic baseline** — per-chunk attribution by
token-overlap between each retrieved chunk and the final output, normalized
across the chunk set — plus the same plug-in slot pattern the goal-drift
module uses, so contributors can drop in a smarter attribution method
(LLM, embedding similarity, mechanistic interp) without changing the API.

Memory lineage uses the same primitives: read events carry the entry's
text + age, and the same attribution function ranks how influential each
read was on the output.

Pure module. No I/O, no network, no LLM by default.

Contributor recipe: ``docs/agent-recipes/add-retrieval-attribution.md``.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple


# ── Tunables ────────────────────────────────────────────────────────────────

# Stoplist mirrors the freshness/goal_drift modules — keep them in sync.
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

_MAX_TEXT_CHARS = 8192

# A chunk is "influential" when its attribution score crosses this floor.
# 0.05 is empirically permissive — meant to catch chunks that contributed
# *something* identifiable, not just chunks that dominated.
DEFAULT_INFLUENCE_THRESHOLD = 0.05

# A chunk is "dead weight" when its attribution score across N output
# samples averages below this. Use this to drive index-pruning workflows.
DEFAULT_DEAD_WEIGHT_THRESHOLD = 0.01

# Memory entries older than this are flagged as "stale" in the lineage
# report. Default is 7 days (604800 seconds); applications with faster
# decay should override.
DEFAULT_STALE_AGE_SECONDS = 7 * 24 * 3600


# ── Data shapes ─────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class RetrievedChunk:
    """One chunk returned by a retrieval call."""

    chunk_id: str
    text: str
    score: float = 0.0  # the retriever's own similarity score
    rank: int = 0       # 0-based position in the returned list


@dataclass(frozen=True)
class MemoryEntry:
    """One value read from a memory store during the trajectory."""

    key: str
    text: str
    store: str = "unspecified"
    age_seconds: int = 0


@dataclass(frozen=True)
class ChunkAttribution:
    """How much a retrieved chunk influenced the agent's output.

    ``score`` is normalized across the chunk set so the values sum to
    1.0 — that makes the per-chunk numbers comparable across runs of
    different size. ``raw`` keeps the un-normalized signal for debugging.
    """

    chunk_id: str
    score: float
    raw: float
    rank_at_retrieval: int


@dataclass(frozen=True)
class RetrievalLineage:
    """The full attribution result for one retrieval → output pair."""

    output_text: str
    chunks: Tuple[ChunkAttribution, ...]
    threshold: float
    judge_used: bool
    evidence: Dict[str, object] = field(default_factory=dict)

    @property
    def influential(self) -> Tuple[ChunkAttribution, ...]:
        """Chunks scoring above ``threshold`` — the ones that mattered."""
        return tuple(c for c in self.chunks if c.score >= self.threshold)

    @property
    def dead_weight(self) -> Tuple[ChunkAttribution, ...]:
        """Chunks scoring below ``DEFAULT_DEAD_WEIGHT_THRESHOLD``.

        Surfaced separately because "this chunk added nothing measurable"
        is the actionable signal for index-pruning workflows.
        """
        return tuple(
            c for c in self.chunks if c.score < DEFAULT_DEAD_WEIGHT_THRESHOLD
        )


@dataclass(frozen=True)
class StaleMemoryFlag:
    """A memory entry old enough to be suspect for drift."""

    key: str
    age_seconds: int
    store: str


# ── Tokenization ────────────────────────────────────────────────────────────


def _tokens(text: str) -> frozenset[str]:
    if not text:
        return frozenset()
    truncated = text[:_MAX_TEXT_CHARS].lower()
    truncated = re.sub(r"\d+", " <num> ", truncated)
    cleaned = re.sub(r"[^a-z0-9<>\s]+", " ", truncated)
    return frozenset(
        t for t in cleaned.split()
        if t and t not in _STOPWORDS and len(t) > 1
    )


def _overlap(chunk_tokens: frozenset[str], output_tokens: frozenset[str]) -> float:
    """Fraction of chunk tokens that appear in the output.

    This is *recall on the chunk*, not Jaccard. We want "did the output
    use this chunk's content?" — high values when the output borrowed
    most of a chunk, low values when it ignored it. Symmetric Jaccard
    would penalize chunks for the output containing extra material the
    chunk didn't have, which is the opposite of what we want.
    """
    if not chunk_tokens:
        return 0.0
    return len(chunk_tokens & output_tokens) / len(chunk_tokens)


# ── Attribution interface ───────────────────────────────────────────────────


AttributionJudge = Callable[
    [str, Sequence[RetrievedChunk]], Optional[Sequence[float]]
]
"""Signature for a pluggable attribution judge.

The callable receives ``(output_text, chunks)`` and returns either:

- a sequence of floats parallel to ``chunks`` (raw influence scores
  in ``[0, 1]``; the module normalizes), OR
- ``None`` to fall back to the deterministic baseline.

Returning ``None`` (or raising) lets adapters fail soft on judge errors
without breaking the lineage analysis. Same fail-soft contract as the
goal-drift judge.
"""


def _normalize(raw_scores: Sequence[float]) -> List[float]:
    """Normalize a list of raw scores so they sum to 1.0.

    All-zero input stays all-zero (no chunks influenced anything;
    nothing to redistribute mass to).
    """
    total = sum(raw_scores)
    if total <= 0:
        return [0.0 for _ in raw_scores]
    return [round(s / total, 4) for s in raw_scores]


def attribute_chunks(
    output_text: str,
    chunks: Sequence[RetrievedChunk],
    *,
    threshold: float = DEFAULT_INFLUENCE_THRESHOLD,
    judge: Optional[AttributionJudge] = None,
) -> RetrievalLineage:
    """Score each chunk's influence on the output.

    With no ``judge``, uses the deterministic baseline (chunk-token recall
    in the output). With a ``judge``, calls it; returns to the baseline
    when it returns None or raises.

    The returned scores sum to 1.0 (when any chunk has nonzero raw
    influence). This makes cross-run comparison sane: "chunk X's
    influence dropped from 0.4 to 0.1 between v1 and v2" is meaningful;
    "chunk X's raw overlap dropped from 0.8 to 0.6" depends on chunk
    length and is harder to reason about.
    """
    if not chunks:
        return RetrievalLineage(
            output_text=output_text,
            chunks=(),
            threshold=threshold,
            judge_used=False,
            evidence={"reason": "no_chunks"},
        )

    judge_used = False
    raw_scores: Optional[List[float]] = None
    if judge is not None:
        try:
            judge_result = judge(output_text, chunks)
            if judge_result is not None:
                raw_scores = [
                    max(0.0, min(1.0, float(s))) for s in judge_result
                ]
                judge_used = True
        except Exception:
            raw_scores = None

    if raw_scores is None:
        out_tokens = _tokens(output_text)
        raw_scores = [_overlap(_tokens(c.text), out_tokens) for c in chunks]

    normalized = _normalize(raw_scores)
    attributions = tuple(
        ChunkAttribution(
            chunk_id=chunk.chunk_id,
            score=normalized[i],
            raw=round(raw_scores[i], 4),
            rank_at_retrieval=chunk.rank,
        )
        for i, chunk in enumerate(chunks)
    )

    return RetrievalLineage(
        output_text=output_text,
        chunks=attributions,
        threshold=threshold,
        judge_used=judge_used,
        evidence={"chunk_count": len(chunks)},
    )


# ── Memory lineage ──────────────────────────────────────────────────────────


def attribute_memory_reads(
    output_text: str,
    reads: Sequence[MemoryEntry],
    *,
    threshold: float = DEFAULT_INFLUENCE_THRESHOLD,
    judge: Optional[AttributionJudge] = None,
) -> RetrievalLineage:
    """Reuse the chunk attribution machinery for memory reads.

    Memory entries are just chunks with a different provenance. We
    rebrand the type for the public API (``MemoryEntry`` reads more
    obviously than "passing a memory store as a chunk") but the
    attribution math is identical — same fail-soft semantics, same
    judge slot, same normalization.
    """
    chunks = tuple(
        RetrievedChunk(
            chunk_id=f"memory:{entry.store}:{entry.key}",
            text=entry.text,
            score=0.0,
            rank=i,
        )
        for i, entry in enumerate(reads)
    )
    return attribute_chunks(
        output_text, chunks, threshold=threshold, judge=judge,
    )


def detect_stale_memory(
    reads: Sequence[MemoryEntry],
    *,
    stale_age_seconds: int = DEFAULT_STALE_AGE_SECONDS,
) -> List[StaleMemoryFlag]:
    """Flag memory entries old enough to suspect for drift.

    "Old" here means wall-clock-since-write; freshness of the *content*
    relative to the world is a separate problem this module doesn't try
    to solve. The flag is a starting point for a human review or for
    feeding into a confidence weighting in the next layer.
    """
    out: List[StaleMemoryFlag] = []
    for r in reads:
        if r.age_seconds >= stale_age_seconds:
            out.append(
                StaleMemoryFlag(
                    key=r.key, age_seconds=r.age_seconds, store=r.store
                )
            )
    out.sort(key=lambda f: -f.age_seconds)
    return out
