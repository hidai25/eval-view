"""Eval-set freshness: detect production query coverage gaps in the test suite.

This module is the heart of ``evalview freshness``. It answers a question that
sits next to ``autopr`` but is fundamentally different:

- ``autopr`` is **incident-driven**: something failed in production, codify it.
- ``freshness`` is **distribution-driven**: production traffic has drifted away
  from what the suite covers — *even when nothing has failed yet*.

The function is **pure**: no I/O, no network, no LLM, no embeddings. The
similarity metric is Jaccard token overlap. That keeps it fast, deterministic,
testable in CI, and aligned with the same contract ``regression_synth`` honors.
A future revision can add an optional embedding backend behind a flag without
changing the public API.

Schema of a production query record
-----------------------------------
A "production query" is just a string. Callers extract it from whatever JSONL
they have — most commonly ``.evalview/incidents.jsonl`` (written by
``evalview monitor``), where every record carries a ``query`` field.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


# ── Tunables ────────────────────────────────────────────────────────────────

# A production query is "covered" when its max similarity to any suite query
# is at least this value. 0.35 is intentionally permissive: Jaccard on bag of
# words is noisy and we'd rather under-flag than spam the user with gaps that
# are arguably handled.
DEFAULT_COVERAGE_THRESHOLD = 0.35

# Two uncovered queries cluster together when their similarity is at least
# this value. Slightly higher than the coverage threshold so gaps stay tight.
DEFAULT_CLUSTER_THRESHOLD = 0.45

# Ignore clusters smaller than this — they're usually one-off questions, not
# a genuine traffic pattern worth a new test.
DEFAULT_MIN_CLUSTER_SIZE = 2

# Cap how many example queries we keep per cluster — enough for a human to
# eyeball the pattern, not so many that the report becomes unreadable.
_MAX_EXAMPLES_PER_CLUSTER = 5


# A short English stoplist. Keeping this tiny on purpose: Jaccard is already
# coarse and overly aggressive stopword filtering throws away signal. These
# are the words whose presence is least informative for query similarity.
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


# ── Tokenization & similarity ───────────────────────────────────────────────


def normalize_query(query: str) -> frozenset[str]:
    """Lower-case, strip punctuation, collapse numbers, drop stopwords, tokenize.

    The returned set is used directly for Jaccard similarity. We deliberately
    return a *set* (not a list) because order is irrelevant for this metric
    and dedupe-on-load is exactly what we want for repeated words.

    Number normalization: any run of digits collapses to the single token
    ``<num>``. This is the difference between "track order 4812" and
    "track order 8201" being treated as identical patterns (correct: it's
    the same intent) vs. distinct queries (wrong: order IDs are noise).
    Without this step every customer's unique ID would shred clusters.
    """
    if not query:
        return frozenset()
    lowered = query.lower()
    # Collapse digit runs before stripping punctuation so '#4812' and
    # 'order-8201' both reduce to the same '<num>' bucket.
    lowered = re.sub(r"\d+", " <num> ", lowered)
    # Allow '<' and '>' through so the placeholder survives tokenization.
    cleaned = re.sub(r"[^a-z0-9<>\s]+", " ", lowered)
    tokens = {t for t in cleaned.split() if t and t not in _STOPWORDS}
    return frozenset(tokens)


def jaccard(a: frozenset[str], b: frozenset[str]) -> float:
    """Standard Jaccard set similarity in ``[0.0, 1.0]``.

    Defined as ``|A ∩ B| / |A ∪ B|``. Returns 0.0 when both sets are empty
    rather than NaN — an empty query has no information and shouldn't be
    treated as a perfect match for another empty query.
    """
    if not a and not b:
        return 0.0
    if not a or not b:
        return 0.0
    intersection = len(a & b)
    union = len(a | b)
    return intersection / union if union else 0.0


def query_similarity(a: str, b: str) -> float:
    """Convenience wrapper: tokenize both sides and return Jaccard."""
    return jaccard(normalize_query(a), normalize_query(b))


# ── Coverage report ─────────────────────────────────────────────────────────


@dataclass(frozen=True)
class CoverageMatch:
    """One production query and its best match in the suite."""

    query: str
    nearest_suite_query: Optional[str]
    similarity: float
    covered: bool


@dataclass(frozen=True)
class CoverageReport:
    """Per-query coverage matches plus aggregate stats."""

    threshold: float
    matches: Tuple[CoverageMatch, ...]

    @property
    def total(self) -> int:
        return len(self.matches)

    @property
    def covered(self) -> int:
        return sum(1 for m in self.matches if m.covered)

    @property
    def uncovered(self) -> int:
        return self.total - self.covered

    @property
    def coverage_pct(self) -> float:
        """Coverage percentage in ``[0.0, 100.0]``. 100.0 when no queries."""
        if self.total == 0:
            return 100.0
        return round(100.0 * self.covered / self.total, 1)

    def uncovered_queries(self) -> List[str]:
        """Return the raw uncovered query strings in input order."""
        return [m.query for m in self.matches if not m.covered]


def compute_coverage(
    prod_queries: Sequence[str],
    suite_queries: Sequence[str],
    threshold: float = DEFAULT_COVERAGE_THRESHOLD,
) -> CoverageReport:
    """Build a CoverageReport for ``prod_queries`` against ``suite_queries``.

    The match for each production query is the suite query with the highest
    Jaccard similarity. If the suite is empty, every production query is
    classified as uncovered with ``similarity == 0.0``.
    """
    suite_tokens: List[Tuple[str, frozenset[str]]] = [
        (q, normalize_query(q)) for q in suite_queries if q
    ]

    matches: List[CoverageMatch] = []
    for q in prod_queries:
        if not q or not q.strip():
            continue
        q_tokens = normalize_query(q)
        best_sim = 0.0
        best_match: Optional[str] = None
        for s_query, s_tokens in suite_tokens:
            sim = jaccard(q_tokens, s_tokens)
            if sim > best_sim:
                best_sim = sim
                best_match = s_query
        matches.append(
            CoverageMatch(
                query=q,
                nearest_suite_query=best_match,
                similarity=round(best_sim, 4),
                covered=best_sim >= threshold,
            )
        )
    return CoverageReport(threshold=threshold, matches=tuple(matches))


# ── Clustering ──────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class QueryCluster:
    """A group of similar uncovered queries that together form a coverage gap."""

    representative: str
    members: Tuple[str, ...]
    avg_intra_similarity: float

    @property
    def size(self) -> int:
        return len(self.members)

    def examples(self, limit: int = _MAX_EXAMPLES_PER_CLUSTER) -> List[str]:
        """Pick a stable, diverse-ish set of example queries to show humans.

        The representative is always first. Remaining slots are filled in
        original (insertion) order — deterministic and good enough; in
        practice users want to scan a few real samples, not a curated set.
        """
        out: List[str] = [self.representative]
        for m in self.members:
            if m == self.representative:
                continue
            out.append(m)
            if len(out) >= limit:
                break
        return out


def _pick_representative(
    members: Sequence[str],
    token_cache: Dict[str, frozenset[str]],
) -> Tuple[str, float]:
    """Return ``(representative, avg_intra_similarity)`` for a cluster.

    The representative is the medoid: the member with the highest mean
    similarity to all other members. Ties broken by insertion order, which
    keeps the result stable across runs.
    """
    if len(members) == 1:
        return members[0], 1.0

    best_member = members[0]
    best_score = -1.0
    for candidate in members:
        c_tokens = token_cache[candidate]
        if not c_tokens:
            continue
        total = 0.0
        n = 0
        for other in members:
            if other is candidate:
                continue
            total += jaccard(c_tokens, token_cache[other])
            n += 1
        avg = total / n if n else 0.0
        if avg > best_score:
            best_score = avg
            best_member = candidate
    return best_member, round(max(best_score, 0.0), 4)


def cluster_queries(
    queries: Sequence[str],
    threshold: float = DEFAULT_CLUSTER_THRESHOLD,
    min_cluster_size: int = DEFAULT_MIN_CLUSTER_SIZE,
) -> List[QueryCluster]:
    """Greedy single-linkage clustering of queries by Jaccard similarity.

    Each query is placed in the first existing cluster whose seed it matches
    above ``threshold``; otherwise it seeds a new cluster. Singletons below
    ``min_cluster_size`` are dropped from the returned list — they're either
    real one-offs or noise, and either way they don't justify a new test.

    The greedy approach is intentional: it's O(n·k) where k is the number
    of clusters seen so far, requires no extra dependency, and is stable in
    the sense that the same input produces the same output. For incident
    volumes typical of an early-production agent (hundreds to low thousands),
    this is plenty fast.
    """
    token_cache: Dict[str, frozenset[str]] = {}
    # Use ``id(seed)`` would be unstable across runs; instead key by the
    # seed string itself, with a counter as tiebreaker for duplicates.
    cluster_members: List[List[str]] = []
    cluster_seeds: List[frozenset[str]] = []

    for q in queries:
        if not q:
            continue
        if q not in token_cache:
            token_cache[q] = normalize_query(q)
        q_tokens = token_cache[q]
        placed = False
        for idx, seed_tokens in enumerate(cluster_seeds):
            if jaccard(q_tokens, seed_tokens) >= threshold:
                cluster_members[idx].append(q)
                placed = True
                break
        if not placed:
            cluster_seeds.append(q_tokens)
            cluster_members.append([q])

    clusters: List[QueryCluster] = []
    for members in cluster_members:
        if len(members) < min_cluster_size:
            continue
        representative, avg_sim = _pick_representative(members, token_cache)
        clusters.append(
            QueryCluster(
                representative=representative,
                members=tuple(members),
                avg_intra_similarity=avg_sim,
            )
        )
    # Sort by size (descending) so the biggest gaps surface first.
    clusters.sort(key=lambda c: (-c.size, c.representative))
    return clusters


# ── Test-stub synthesis ─────────────────────────────────────────────────────


def coverage_slug(cluster: QueryCluster) -> str:
    """Stable filesystem-safe slug for a cluster.

    Combines a short hash of the representative query with a hash of the
    sorted member list. Same cluster on the same data => same slug, so the
    command is idempotent: re-running ``freshness --propose`` won't write
    duplicates.
    """
    rep = cluster.representative
    rep_slug = re.sub(r"[^a-zA-Z0-9_\-]+", "-", rep.lower()).strip("-")
    rep_slug = rep_slug[:40] or "gap"
    member_blob = "\n".join(sorted(cluster.members))
    member_hash = hashlib.sha1(
        member_blob.encode("utf-8"), usedforsecurity=False
    ).hexdigest()[:8]
    return f"{rep_slug}-{member_hash}"


def synthesize_coverage_test(
    cluster: QueryCluster,
    *,
    min_score: float = 70.0,
    timestamp: Optional[str] = None,
) -> Dict[str, Any]:
    """Build a coverage-gap test dict from a cluster.

    The returned dict is a valid ``TestCase`` input. It is deliberately
    minimal: only the representative query and a descriptive header, no
    ``expected`` assertions. The reviewer's workflow is:

    1. ``evalview freshness --propose`` writes these stubs.
    2. Human reviews each stub; deletes any that don't represent a real gap.
    3. ``evalview snapshot`` runs the stubs and captures current agent
       behavior as the baseline — *that* becomes the regression test going
       forward.

    Why no auto-generated ``expected`` clauses? We don't know what *correct*
    behavior is for an uncovered query. Pretending to know — and pinning
    arbitrary phrases or tools — would create false confidence. The honest
    move is to capture baseline behavior on snapshot.
    """
    if not cluster.members:
        raise ValueError("cannot synthesize a coverage test from an empty cluster")

    ts = timestamp or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    date_prefix = ts[:10]
    slug = coverage_slug(cluster)

    # TestCase.name accepts [A-Za-z0-9 _\-\.] — sanitize aggressively.
    raw_name = f"coverage_{slug}_{date_prefix}"
    safe_name = re.sub(r"[^a-zA-Z0-9 _\-\.]", "-", raw_name)

    example_lines = "\n".join(f"  - {e!r}" for e in cluster.examples())
    description = (
        f"Auto-generated coverage stub from `evalview freshness` at {ts}.\n"
        f"Cluster size: {cluster.size} production queries (avg intra-similarity "
        f"{cluster.avg_intra_similarity:.2f}).\n"
        f"Examples:\n{example_lines}\n"
        f"Review the query, decide whether this represents a real gap, then run "
        f"`evalview snapshot` to capture current behavior as a baseline."
    )

    return {
        "name": safe_name,
        "description": description,
        "input": {"query": cluster.representative},
        # Intentionally empty: see docstring. The reviewer adds assertions
        # after deciding what "correct" looks like for this query.
        "expected": {},
        "thresholds": {"min_score": float(min_score)},
        # Coverage stubs are capability tests, not regressions.
        "suite_type": "capability",
        "tags": ["coverage", "freshness"],
        "meta": {
            "coverage": {
                "slug": slug,
                "cluster_size": cluster.size,
                "avg_intra_similarity": cluster.avg_intra_similarity,
                "examples": list(cluster.examples()),
                "timestamp": ts,
            }
        },
    }


# ── Production-query extraction ─────────────────────────────────────────────


def extract_queries_from_records(
    records: Iterable[Dict[str, Any]],
    *,
    field_name: str = "query",
) -> List[str]:
    """Pull non-empty query strings out of a stream of JSONL records.

    Deliberately lenient: any record missing the field or with a non-string
    value is skipped silently. The caller is the right layer to surface
    parse errors (it knows where the records came from).
    """
    out: List[str] = []
    for r in records:
        if not isinstance(r, dict):
            continue
        q = r.get(field_name)
        if isinstance(q, str) and q.strip():
            out.append(q.strip())
    return out


# ── Aggregate report (CLI convenience) ──────────────────────────────────────


@dataclass(frozen=True)
class FreshnessReport:
    """Top-level result returned by the CLI for rendering or JSON output."""

    coverage: CoverageReport
    clusters: Tuple[QueryCluster, ...]
    suite_size: int
    prod_size: int
    cluster_threshold: float
    min_cluster_size: int

    def to_dict(self) -> Dict[str, Any]:
        """Render the report as a plain dict for ``--json``.

        Floats are pre-rounded so the JSON survives a diff in CI without
        spurious trailing-digit churn.
        """
        return {
            "suite_size": self.suite_size,
            "prod_size": self.prod_size,
            "coverage": {
                "threshold": self.coverage.threshold,
                "covered": self.coverage.covered,
                "uncovered": self.coverage.uncovered,
                "total": self.coverage.total,
                "coverage_pct": self.coverage.coverage_pct,
            },
            "clusters": [
                {
                    "slug": coverage_slug(c),
                    "representative": c.representative,
                    "size": c.size,
                    "avg_intra_similarity": c.avg_intra_similarity,
                    "examples": c.examples(),
                }
                for c in self.clusters
            ],
            "cluster_threshold": self.cluster_threshold,
            "min_cluster_size": self.min_cluster_size,
        }


def build_freshness_report(
    prod_queries: Sequence[str],
    suite_queries: Sequence[str],
    *,
    coverage_threshold: float = DEFAULT_COVERAGE_THRESHOLD,
    cluster_threshold: float = DEFAULT_CLUSTER_THRESHOLD,
    min_cluster_size: int = DEFAULT_MIN_CLUSTER_SIZE,
) -> FreshnessReport:
    """One-shot helper: coverage + clustering in a single call."""
    coverage = compute_coverage(prod_queries, suite_queries, threshold=coverage_threshold)
    clusters = cluster_queries(
        coverage.uncovered_queries(),
        threshold=cluster_threshold,
        min_cluster_size=min_cluster_size,
    )
    return FreshnessReport(
        coverage=coverage,
        clusters=tuple(clusters),
        suite_size=sum(1 for q in suite_queries if q),
        prod_size=sum(1 for q in prod_queries if q),
        cluster_threshold=cluster_threshold,
        min_cluster_size=min_cluster_size,
    )
