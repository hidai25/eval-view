"""Tests for `evalview.core.retrieval_lineage`.

Covers:

1. The deterministic chunk-attribution baseline.
2. The pluggable judge interface (override + fail-soft fallback).
3. Memory lineage (delegates to chunk attribution; pin the contract).
4. Stale-memory detection.
"""
from __future__ import annotations

from typing import Optional, Sequence

from evalview.core.retrieval_lineage import (
    DEFAULT_DEAD_WEIGHT_THRESHOLD,
    DEFAULT_INFLUENCE_THRESHOLD,
    DEFAULT_STALE_AGE_SECONDS,
    AttributionJudge,
    MemoryEntry,
    RetrievedChunk,
    attribute_chunks,
    attribute_memory_reads,
    detect_stale_memory,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _chunk(chunk_id: str, text: str, *, rank: int = 0) -> RetrievedChunk:
    return RetrievedChunk(chunk_id=chunk_id, text=text, rank=rank)


# ---------------------------------------------------------------------------
# Deterministic baseline
# ---------------------------------------------------------------------------


class TestAttributeChunksBaseline:
    def test_quoted_chunk_dominates(self) -> None:
        # Output borrows nearly verbatim from chunk-A. That chunk should
        # carry most of the normalized influence; chunk-B should be near
        # zero because it shares no content.
        chunks = (
            _chunk("a", "the refund policy allows a 30 day window", rank=0),
            _chunk("b", "shipping rates depend on weight and distance", rank=1),
        )
        lineage = attribute_chunks(
            "Refund policy allows a 30 day window from purchase.",
            chunks,
        )
        scores = {c.chunk_id: c.score for c in lineage.chunks}
        assert scores["a"] > scores["b"]
        # Influential set must be non-empty when at least one chunk
        # actually contributed.
        assert lineage.influential
        assert lineage.influential[0].chunk_id == "a"

    def test_unrelated_chunks_get_normalized_to_zero(self) -> None:
        # Output has nothing in common with either chunk → all-zero raw
        # scores → normalize_scores stays all-zero (no division by zero,
        # no spurious mass redistribution).
        chunks = (
            _chunk("a", "alpha bravo charlie", rank=0),
            _chunk("b", "delta echo foxtrot", rank=1),
        )
        lineage = attribute_chunks("zulu yankee xray", chunks)
        assert all(c.score == 0.0 for c in lineage.chunks)
        # Dead-weight surface must list both — they were retrieved and
        # contributed nothing measurable.
        assert len(lineage.dead_weight) == 2

    def test_normalized_scores_sum_to_one_when_any_chunk_influential(self) -> None:
        # Cross-run comparison is the whole point of normalization.
        # Pin the invariant.
        chunks = (
            _chunk("a", "refund policy allows a 30 day window", rank=0),
            _chunk("b", "refund policy", rank=1),
        )
        lineage = attribute_chunks("refund policy 30 day window", chunks)
        total = sum(c.score for c in lineage.chunks)
        assert abs(total - 1.0) < 1e-3

    def test_empty_chunks_returns_benign_lineage(self) -> None:
        # A retrieval that returned nothing is not an error; the
        # attribution is a no-op rather than a crash.
        lineage = attribute_chunks("any output", [])
        assert lineage.chunks == ()
        assert lineage.evidence["reason"] == "no_chunks"


# ---------------------------------------------------------------------------
# Judge interface
# ---------------------------------------------------------------------------


class TestAttributionJudge:
    def test_judge_scores_override_baseline(self) -> None:
        # Inputs the baseline would call all-zero; the judge says
        # chunk-b dominates. After normalization, b should carry all
        # of the mass.
        chunks = (
            _chunk("a", "alpha", rank=0),
            _chunk("b", "bravo", rank=1),
        )

        def judge(output: str, chs: Sequence[RetrievedChunk]) -> Optional[Sequence[float]]:
            return [0.0, 0.9]

        lineage = attribute_chunks("totally unrelated", chunks, judge=judge)
        assert lineage.judge_used
        scores = {c.chunk_id: c.score for c in lineage.chunks}
        assert scores["b"] > 0.99
        assert scores["a"] == 0.0

    def test_judge_returning_none_falls_back_to_baseline(self) -> None:
        chunks = (_chunk("a", "alpha bravo", rank=0),)

        def opting_out(output: str, chs: Sequence[RetrievedChunk]) -> Optional[Sequence[float]]:
            return None

        lineage = attribute_chunks("alpha bravo", chunks, judge=opting_out)
        assert not lineage.judge_used
        # Baseline still ran — chunk influence > 0.
        assert lineage.chunks[0].score > 0

    def test_judge_raising_falls_back_silently(self) -> None:
        # A flaky judge must never block the analysis. Same fail-soft
        # contract as goal_drift.
        chunks = (_chunk("a", "alpha", rank=0),)

        def flaky(output: str, chs: Sequence[RetrievedChunk]) -> Optional[Sequence[float]]:
            raise RuntimeError("upstream timeout")

        lineage = attribute_chunks("alpha output", chunks, judge=flaky)
        assert not lineage.judge_used
        # Baseline ran successfully; lineage is well-formed.
        assert len(lineage.chunks) == 1

    def test_judge_scores_clamped_to_unit_interval(self) -> None:
        chunks = (_chunk("a", "x", rank=0), _chunk("b", "y", rank=1))

        def out_of_range(output: str, chs: Sequence[RetrievedChunk]) -> Optional[Sequence[float]]:
            return [-0.5, 7.5]

        lineage = attribute_chunks("z", chunks, judge=out_of_range)
        # Negative clamped to 0.0, 7.5 clamped to 1.0; after
        # normalization b carries all the mass.
        scores = {c.chunk_id: c.score for c in lineage.chunks}
        assert scores["a"] == 0.0
        assert scores["b"] == 1.0


# ---------------------------------------------------------------------------
# Influential / dead-weight surfaces
# ---------------------------------------------------------------------------


class TestInfluentialAndDeadWeight:
    def test_influential_threshold_pinned(self) -> None:
        # Anything above DEFAULT_INFLUENCE_THRESHOLD is "influential".
        # Pinning here so a future threshold change is a deliberate, not
        # accidental, behavior shift.
        chunks = (_chunk("dom", "alpha bravo charlie", rank=0),
                  _chunk("trace", "alpha", rank=1))
        lineage = attribute_chunks("alpha bravo charlie delta", chunks)
        # Dominant chunk should always cross the influence threshold.
        assert any(c.chunk_id == "dom" for c in lineage.influential)

    def test_dead_weight_threshold_pinned(self) -> None:
        # Below DEFAULT_DEAD_WEIGHT_THRESHOLD = candidate for index
        # pruning. Surface separately so it's actionable.
        chunks = (
            _chunk("dom", "alpha bravo", rank=0),
            _chunk("dead", "zeta yankee xray whiskey", rank=1),
        )
        lineage = attribute_chunks("alpha bravo", chunks)
        ids_dead = {c.chunk_id for c in lineage.dead_weight}
        assert "dead" in ids_dead

    def test_default_thresholds_are_distinct(self) -> None:
        # If they ever collide, "influential" and "dead weight" overlap
        # and the digest stops being readable. Pin the relationship.
        assert DEFAULT_INFLUENCE_THRESHOLD > DEFAULT_DEAD_WEIGHT_THRESHOLD


# ---------------------------------------------------------------------------
# Memory lineage
# ---------------------------------------------------------------------------


class TestMemoryAttribution:
    def test_delegates_to_chunk_attribution(self) -> None:
        reads = (
            MemoryEntry(key="user.name", text="alice", store="profile"),
            MemoryEntry(key="last_order", text="laptop dock", store="episodic"),
        )
        lineage = attribute_memory_reads("alice ordered a laptop dock", reads)
        # Attribution wrapping renames chunks for memory provenance —
        # pin the prefix so consumers can rely on it.
        assert all(c.chunk_id.startswith("memory:") for c in lineage.chunks)
        # Both reads contributed → both should appear in normalized scores.
        assert sum(c.score for c in lineage.chunks) > 0


# ---------------------------------------------------------------------------
# Stale memory detection
# ---------------------------------------------------------------------------


class TestStaleMemory:
    def test_old_entries_flagged(self) -> None:
        # A 30-day-old entry crosses the 7-day default → flagged.
        reads = [
            MemoryEntry(key="recent", text="x", age_seconds=60),
            MemoryEntry(key="ancient", text="y", age_seconds=30 * 24 * 3600),
        ]
        flags = detect_stale_memory(reads)
        assert [f.key for f in flags] == ["ancient"]

    def test_oldest_first_ordering(self) -> None:
        # Ordering matters for digest rendering — the most-stale entry
        # should always be the first thing a human sees.
        reads = [
            MemoryEntry(key="middle", text="x",
                        age_seconds=10 * 24 * 3600),
            MemoryEntry(key="oldest", text="y",
                        age_seconds=30 * 24 * 3600),
            MemoryEntry(key="just_stale", text="z",
                        age_seconds=DEFAULT_STALE_AGE_SECONDS),
        ]
        flags = detect_stale_memory(reads)
        assert [f.key for f in flags] == ["oldest", "middle", "just_stale"]

    def test_custom_threshold_respected(self) -> None:
        reads = [MemoryEntry(key="k", text="x", age_seconds=120)]
        # With a 60-second threshold the entry is stale.
        assert detect_stale_memory(reads, stale_age_seconds=60)
        # With a 600-second threshold it isn't.
        assert detect_stale_memory(reads, stale_age_seconds=600) == []


# ---------------------------------------------------------------------------
# Type contract sanity
# ---------------------------------------------------------------------------


class TestTypeContract:
    def test_attribution_judge_alias_is_callable_signature(self) -> None:
        # AttributionJudge is a typing alias — verify a function with
        # the right shape is assignable to it. Catches accidental
        # type-alias drift.
        def judge(output: str, chs: Sequence[RetrievedChunk]) -> Optional[Sequence[float]]:
            return None

        ref: AttributionJudge = judge
        assert callable(ref)
