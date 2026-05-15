"""Tests for the eval-set freshness module + command.

Two layers:

1. ``evalview.core.freshness`` — pure functions (tokenization, Jaccard,
   coverage, clustering, stub synthesis). These have no I/O and are tested
   in isolation.
2. ``evalview.commands.freshness_cmd`` — CLI wiring. Tested through Click's
   ``CliRunner`` on a tmp filesystem, mirroring the pattern in ``test_autopr``.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import yaml
from click.testing import CliRunner

from evalview.commands.freshness_cmd import freshness
from evalview.core.freshness import (
    QueryCluster,
    build_freshness_report,
    cluster_queries,
    compute_coverage,
    coverage_slug,
    extract_queries_from_records,
    jaccard,
    normalize_query,
    query_similarity,
    synthesize_coverage_test,
)
from evalview.core.types import TestCase


# ---------------------------------------------------------------------------
# Pure module: tokenization & similarity
# ---------------------------------------------------------------------------


class TestNormalizeQuery:
    def test_lowercases_and_strips_punctuation(self) -> None:
        tokens = normalize_query("Where is My Order #4812?!")
        assert "order" in tokens
        # Digits collapse to the <num> placeholder so different IDs don't
        # shred otherwise-identical queries.
        assert "<num>" in tokens
        assert "4812" not in tokens
        assert "where" in tokens
        # Punctuation must be gone — no token still attached to it.
        assert all("#" not in t and "?" not in t for t in tokens)

    def test_numbers_normalize_for_clustering(self) -> None:
        # Two queries that differ only in the order number must produce
        # identical token sets — this is what lets clustering work on real
        # production traffic where every customer has a unique ID.
        a = normalize_query("where is order 4812")
        b = normalize_query("where is order 9999")
        assert a == b

    def test_drops_stopwords(self) -> None:
        tokens = normalize_query("I want a refund for my order")
        assert "refund" in tokens
        assert "order" in tokens
        # Stopwords gone — these are not informative for similarity.
        assert "i" not in tokens
        assert "the" not in tokens
        assert "for" not in tokens

    def test_empty_returns_empty(self) -> None:
        assert normalize_query("") == frozenset()
        # ``__bool__`` on frozenset works as expected so callers can if-check.
        assert not normalize_query("")


class TestJaccard:
    def test_identical_sets_score_one(self) -> None:
        s = frozenset({"refund", "order"})
        assert jaccard(s, s) == 1.0

    def test_disjoint_sets_score_zero(self) -> None:
        assert jaccard(frozenset({"a"}), frozenset({"b"})) == 0.0

    def test_empty_inputs_score_zero(self) -> None:
        # Empty queries should never count as a match — guards against the
        # "two empty queries are perfect twins" edge case.
        assert jaccard(frozenset(), frozenset()) == 0.0
        assert jaccard(frozenset({"a"}), frozenset()) == 0.0

    def test_partial_overlap_is_ratio(self) -> None:
        a = frozenset({"refund", "order", "policy"})
        b = frozenset({"refund", "order", "delete"})
        # |A∩B| = 2, |A∪B| = 4 → 0.5
        assert jaccard(a, b) == 0.5

    def test_query_similarity_is_symmetric(self) -> None:
        a = "where is my refund"
        b = "i want a refund please"
        assert query_similarity(a, b) == query_similarity(b, a)


# ---------------------------------------------------------------------------
# Pure module: coverage
# ---------------------------------------------------------------------------


class TestComputeCoverage:
    def test_query_with_close_match_is_covered(self) -> None:
        prod = ["I want a refund for my order"]
        suite = ["refund my order please"]
        rep = compute_coverage(prod, suite, threshold=0.3)
        assert rep.covered == 1
        assert rep.uncovered == 0
        assert rep.matches[0].nearest_suite_query == suite[0]

    def test_query_with_no_overlap_is_uncovered(self) -> None:
        prod = ["cancel my subscription tomorrow"]
        suite = ["lookup weather forecast today"]
        rep = compute_coverage(prod, suite, threshold=0.3)
        assert rep.uncovered == 1
        assert rep.matches[0].covered is False

    def test_empty_suite_means_everything_uncovered(self) -> None:
        prod = ["a", "b", "c"]
        rep = compute_coverage(prod, [], threshold=0.3)
        assert rep.covered == 0
        assert rep.uncovered == 3
        assert all(m.similarity == 0.0 for m in rep.matches)

    def test_coverage_pct_is_zero_safe(self) -> None:
        rep = compute_coverage([], ["a"], threshold=0.3)
        # No production queries → 100% by convention (nothing to fail).
        assert rep.coverage_pct == 100.0
        assert rep.total == 0

    def test_empty_query_is_skipped(self) -> None:
        rep = compute_coverage(["", "  ", "real query"], ["real query"], threshold=0.3)
        # Empty strings filtered out before scoring.
        assert rep.total == 1


# ---------------------------------------------------------------------------
# Pure module: clustering
# ---------------------------------------------------------------------------


class TestClusterQueries:
    def test_similar_queries_cluster_together(self) -> None:
        queries = [
            "where is my order 4812",
            "track order 8201",
            "where is order 9999",
            "completely unrelated billing dispute resolution",
        ]
        clusters = cluster_queries(queries, threshold=0.3, min_cluster_size=2)
        assert len(clusters) == 1
        assert clusters[0].size == 3
        # All three order-tracking queries land in the same cluster.
        assert all("order" in m.lower() for m in clusters[0].members)

    def test_singletons_dropped_below_min_size(self) -> None:
        # No two queries share enough tokens to cluster.
        queries = ["alpha bravo", "charlie delta", "echo foxtrot"]
        clusters = cluster_queries(queries, threshold=0.5, min_cluster_size=2)
        assert clusters == []

    def test_representative_is_a_member(self) -> None:
        queries = ["where is my order", "track my order please", "find my order"]
        clusters = cluster_queries(queries, threshold=0.3, min_cluster_size=2)
        assert clusters
        for c in clusters:
            assert c.representative in c.members

    def test_deterministic_across_runs(self) -> None:
        queries = [
            "where is my order 1",
            "where is my order 2",
            "where is my order 3",
            "cancel subscription now",
            "cancel my subscription please",
        ]
        a = cluster_queries(queries, threshold=0.3, min_cluster_size=2)
        b = cluster_queries(queries, threshold=0.3, min_cluster_size=2)
        # Tuples + sorted output → identical structure on every run.
        assert [(c.representative, c.members) for c in a] == [
            (c.representative, c.members) for c in b
        ]

    def test_clusters_sorted_by_size_desc(self) -> None:
        queries = (
            ["small cluster only here", "small cluster fits here"]
            + ["big group same tokens"] * 4
        )
        clusters = cluster_queries(queries, threshold=0.4, min_cluster_size=2)
        assert len(clusters) >= 2
        sizes = [c.size for c in clusters]
        assert sizes == sorted(sizes, reverse=True)


# ---------------------------------------------------------------------------
# Pure module: stub synthesis
# ---------------------------------------------------------------------------


class TestSynthesizeCoverageTest:
    def _cluster(self) -> QueryCluster:
        return QueryCluster(
            representative="where is my order #4812",
            members=("where is my order #4812", "track order 8201"),
            avg_intra_similarity=0.4,
        )

    def test_produces_valid_test_case(self) -> None:
        test = synthesize_coverage_test(self._cluster())
        # The output must round-trip through TestCase — the same contract
        # the autopr synthesizer honors. This is what makes the YAML
        # downstream-compatible.
        tc = TestCase(**test)
        assert tc.input.query == "where is my order #4812"
        assert tc.suite_type == "capability"
        assert "coverage" in tc.tags
        assert "freshness" in tc.tags

    def test_no_expected_assertions_by_default(self) -> None:
        # Deliberate: we don't know what "correct" looks like for an
        # uncovered query. Synthesizer leaves expected empty so a human
        # reviews and snapshot captures baseline behavior.
        test = synthesize_coverage_test(self._cluster())
        assert test["expected"] == {}

    def test_meta_carries_cluster_provenance(self) -> None:
        test = synthesize_coverage_test(self._cluster())
        meta = test["meta"]["coverage"]
        assert meta["cluster_size"] == 2
        assert meta["slug"] == coverage_slug(self._cluster())
        assert isinstance(meta["examples"], list)
        assert meta["examples"][0] == "where is my order #4812"

    def test_slug_is_stable(self) -> None:
        # Same cluster on the same data → same slug. Idempotency contract
        # for the --propose pathway.
        c = self._cluster()
        assert coverage_slug(c) == coverage_slug(c)

    def test_slug_differs_for_different_members(self) -> None:
        c1 = self._cluster()
        c2 = QueryCluster(
            representative=c1.representative,
            members=c1.members + ("find order 9999",),
            avg_intra_similarity=0.4,
        )
        # Same representative, different member list → different slug so
        # we don't collapse two genuine gaps into one file.
        assert coverage_slug(c1) != coverage_slug(c2)


# ---------------------------------------------------------------------------
# Pure module: top-level builder
# ---------------------------------------------------------------------------


class TestBuildFreshnessReport:
    def test_end_to_end_finds_gap(self) -> None:
        prod = [
            "where is my order #4812",
            "track order 8201",
            "where is order 9999",
            "refund my order please",  # covered by suite
        ]
        suite = ["refund my order"]
        report = build_freshness_report(
            prod, suite,
            coverage_threshold=0.3,
            cluster_threshold=0.3,
            min_cluster_size=2,
        )
        assert report.coverage.uncovered == 3
        assert len(report.clusters) == 1
        # to_dict() is stable JSON-shaped output for --json mode.
        d = report.to_dict()
        assert d["coverage"]["uncovered"] == 3
        assert len(d["clusters"]) == 1


class TestExtractQueriesFromRecords:
    def test_pulls_query_field(self) -> None:
        records = [
            {"query": "foo"},
            {"query": "bar"},
            {"query": ""},          # skipped
            {"other": "baz"},        # skipped
            {"query": 12},           # skipped (non-string)
            "not a dict",            # skipped
        ]
        assert extract_queries_from_records(records) == ["foo", "bar"]


# ---------------------------------------------------------------------------
# CLI wiring
# ---------------------------------------------------------------------------


def _write_incidents(path: Path, incidents: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for inc in incidents:
            f.write(json.dumps(inc) + "\n")


def _write_test_yaml(tests_dir: Path, name: str, query: str) -> None:
    """Write a minimal valid TestCase YAML to the suite dir."""
    tests_dir.mkdir(parents=True, exist_ok=True)
    body = {
        "name": name,
        "input": {"query": query},
        "expected": {},
        "thresholds": {"min_score": 70.0},
    }
    (tests_dir / f"{name}.yaml").write_text(yaml.safe_dump(body, sort_keys=False))


class TestFreshnessCommand:
    def test_reports_coverage_against_existing_suite(self, tmp_path: Path) -> None:
        # Suite has one test about refunds; production has three new
        # order-tracking queries → one cluster.
        tests_dir = tmp_path / "tests"
        _write_test_yaml(tests_dir, "refund_flow", "refund my order please")

        incidents = tmp_path / "incidents.jsonl"
        _write_incidents(
            incidents,
            [
                {"query": "where is my order 4812"},
                {"query": "track order 8201"},
                {"query": "where is order 9999"},
            ],
        )

        runner = CliRunner()
        result = runner.invoke(
            freshness,
            [
                "--from", str(incidents),
                "--tests-dir", str(tests_dir),
                "--threshold", "0.3",
                "--cluster-threshold", "0.3",
            ],
        )
        assert result.exit_code == 0, result.output
        assert "Coverage Gap" in result.output or "coverage gap" in result.output
        assert "Production queries" in result.output

    def test_json_output_is_machine_readable(self, tmp_path: Path) -> None:
        tests_dir = tmp_path / "tests"
        _write_test_yaml(tests_dir, "refund_flow", "refund my order please")

        incidents = tmp_path / "incidents.jsonl"
        _write_incidents(
            incidents,
            [
                {"query": "track order 4812"},
                {"query": "where is order 8201"},
            ],
        )

        runner = CliRunner()
        result = runner.invoke(
            freshness,
            [
                "--from", str(incidents),
                "--tests-dir", str(tests_dir),
                "--threshold", "0.3",
                "--cluster-threshold", "0.3",
                "--json",
            ],
        )
        assert result.exit_code == 0, result.output
        # The JSON payload may be preceded by Rich notices on first run; the
        # last fully-parseable JSON object on stdout is what matters.
        payload = json.loads(result.output[result.output.index("{") :])
        assert payload["coverage"]["total"] == 2
        assert payload["coverage"]["uncovered"] >= 1

    def test_propose_writes_stubs_and_is_idempotent(self, tmp_path: Path) -> None:
        tests_dir = tmp_path / "tests"
        _write_test_yaml(tests_dir, "refund_flow", "refund my order please")
        # Keep coverage stubs OUTSIDE the suite dir so re-running freshness
        # exercises the slug-skip path. (When stubs live inside `tests/`, the
        # loader picks them up and the gap legitimately disappears — both
        # paths are correct; this test pins the slug-skip path.)
        coverage_dir = tmp_path / "coverage"

        incidents = tmp_path / "incidents.jsonl"
        _write_incidents(
            incidents,
            [
                {"query": "where is my order 4812"},
                {"query": "track order 8201"},
                {"query": "where is order 9999"},
            ],
        )

        runner = CliRunner()
        first = runner.invoke(
            freshness,
            [
                "--from", str(incidents),
                "--tests-dir", str(tests_dir),
                "--coverage-dir", str(coverage_dir),
                "--threshold", "0.3",
                "--cluster-threshold", "0.3",
                "--propose",
            ],
        )
        assert first.exit_code == 0, first.output
        stubs = list(coverage_dir.glob("*.yaml"))
        assert stubs, "expected at least one coverage stub to be written"
        # Every stub must round-trip through TestCase.
        for path in stubs:
            data = yaml.safe_load(path.read_text())
            TestCase(**data)

        # Second run: same data → no new stubs written, exits cleanly.
        second = runner.invoke(
            freshness,
            [
                "--from", str(incidents),
                "--tests-dir", str(tests_dir),
                "--coverage-dir", str(coverage_dir),
                "--threshold", "0.3",
                "--cluster-threshold", "0.3",
                "--propose",
            ],
        )
        assert second.exit_code == 0, second.output
        assert len(list(coverage_dir.glob("*.yaml"))) == len(stubs)
        assert "Skipped" in second.output or "already exist" in second.output

    def test_dry_run_writes_nothing(self, tmp_path: Path) -> None:
        tests_dir = tmp_path / "tests"
        _write_test_yaml(tests_dir, "refund_flow", "refund my order please")
        coverage_dir = tmp_path / "tests" / "coverage"

        incidents = tmp_path / "incidents.jsonl"
        _write_incidents(
            incidents,
            [
                {"query": "where is my order 4812"},
                {"query": "track order 8201"},
            ],
        )

        runner = CliRunner()
        result = runner.invoke(
            freshness,
            [
                "--from", str(incidents),
                "--tests-dir", str(tests_dir),
                "--coverage-dir", str(coverage_dir),
                "--threshold", "0.3",
                "--cluster-threshold", "0.3",
                "--propose", "--dry-run",
            ],
        )
        assert result.exit_code == 0, result.output
        assert not coverage_dir.exists()
        assert "Would write" in result.output

    def test_require_gaps_exits_one_when_gaps_found(self, tmp_path: Path) -> None:
        tests_dir = tmp_path / "tests"
        _write_test_yaml(tests_dir, "refund_flow", "refund my order")

        incidents = tmp_path / "incidents.jsonl"
        _write_incidents(
            incidents,
            [
                {"query": "where is order 4812"},
                {"query": "track order 8201"},
            ],
        )

        runner = CliRunner()
        result = runner.invoke(
            freshness,
            [
                "--from", str(incidents),
                "--tests-dir", str(tests_dir),
                "--threshold", "0.3",
                "--cluster-threshold", "0.3",
                "--require-gaps",
            ],
        )
        assert result.exit_code == 1

    def test_missing_incidents_file_is_not_an_error(self, tmp_path: Path) -> None:
        tests_dir = tmp_path / "tests"
        _write_test_yaml(tests_dir, "refund_flow", "refund my order")

        runner = CliRunner()
        result = runner.invoke(
            freshness,
            [
                "--from", str(tmp_path / "nonexistent.jsonl"),
                "--tests-dir", str(tests_dir),
            ],
        )
        # Mirrors autopr: "no incidents this cycle" is a normal outcome.
        assert result.exit_code == 0, result.output

    def test_extra_log_is_merged(self, tmp_path: Path) -> None:
        tests_dir = tmp_path / "tests"
        _write_test_yaml(tests_dir, "refund_flow", "refund my order")

        incidents = tmp_path / "incidents.jsonl"
        _write_incidents(incidents, [{"query": "where is order 4812"}])

        extra = tmp_path / "extra.jsonl"
        _write_incidents(
            extra,
            [
                {"query": "track order 8201"},
                {"query": "where is order 9999"},
            ],
        )

        runner = CliRunner()
        result = runner.invoke(
            freshness,
            [
                "--from", str(incidents),
                "--from-log", str(extra),
                "--tests-dir", str(tests_dir),
                "--threshold", "0.3",
                "--cluster-threshold", "0.3",
                "--json",
            ],
        )
        assert result.exit_code == 0, result.output
        payload = json.loads(result.output[result.output.index("{") :])
        # 1 from incidents + 2 from extra = 3 production queries
        assert payload["coverage"]["total"] == 3
