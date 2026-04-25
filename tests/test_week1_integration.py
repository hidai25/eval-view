"""Integration tests for Week 1 foundations.

Covers seams that `test_verdict.py` intentionally doesn't touch:

  - DriftTracker fingerprinting (git-repo / non-git, cache reuse)
  - evalview log run grouping (with/without git_sha)
  - Recommendation engine populates the new severity/commands fields
  - Cost-delta aggregation in `_compute_verdict_payload`
  - Recommendation dedup
  - Test-name substitution in suggested_commands
"""
from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any, Dict
from unittest.mock import MagicMock

import pytest

from evalview.commands.check_cmd import (
    _aggregate_cost_delta_ratio,
    _dedup_recommendations,
    _substitute_test_name,
)
from evalview.commands.log_cmd import _group_runs
from evalview.core.drift_tracker import (
    DriftTracker,
    _current_git_sha,
    _current_user,
    _prompt_fingerprint,
)
from evalview.core.recommendations import Recommendation, recommend


# ────────────────────────── fingerprint helpers ──────────────────────────


def test_current_git_sha_in_a_fresh_git_repo(tmp_path: Path) -> None:
    """A real git repo should yield a non-empty short SHA."""
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(
        ["git", "-c", "user.email=t@e.com", "-c", "user.name=t",
         "commit", "-q", "--allow-empty", "-m", "init"],
        cwd=tmp_path,
        check=True,
    )
    sha = _current_git_sha(tmp_path)
    assert sha is not None
    assert len(sha) >= 7  # short SHA is 7+


def test_current_git_sha_returns_none_outside_a_repo(tmp_path: Path) -> None:
    sha = _current_git_sha(tmp_path)
    assert sha is None


def test_prompt_fingerprint_without_any_prompt_files(tmp_path: Path) -> None:
    assert _prompt_fingerprint(tmp_path) is None


def test_prompt_fingerprint_is_stable_for_the_same_content(tmp_path: Path) -> None:
    prompts = tmp_path / "prompts"
    prompts.mkdir()
    (prompts / "system.md").write_text("You are a helpful assistant.")
    first = _prompt_fingerprint(tmp_path)
    second = _prompt_fingerprint(tmp_path)
    assert first is not None
    assert first == second


def test_prompt_fingerprint_changes_when_content_changes(tmp_path: Path) -> None:
    prompts = tmp_path / "prompts"
    prompts.mkdir()
    f = prompts / "system.md"
    f.write_text("v1")
    before = _prompt_fingerprint(tmp_path)
    f.write_text("v2")
    after = _prompt_fingerprint(tmp_path)
    assert before != after


def test_current_user_returns_something() -> None:
    # Not asserting the value — just that it doesn't raise and returns a str.
    result = _current_user()
    assert result is None or isinstance(result, str)


def test_drift_tracker_caches_provenance(tmp_path: Path, monkeypatch: Any) -> None:
    """_provenance() must compute exactly once per tracker instance.

    This is the P0 perf fix: without caching a 50-test check would fire
    50 subprocess calls. We verify the cache by counting invocations of
    _current_git_sha.
    """
    from evalview.core import drift_tracker as dt_module

    calls = {"git": 0, "prompt": 0, "user": 0}

    def fake_git(_bp: Path) -> None:
        calls["git"] += 1
        return None

    def fake_prompt(_bp: Path) -> None:
        calls["prompt"] += 1
        return None

    def fake_user() -> None:
        calls["user"] += 1
        return None

    monkeypatch.setattr(dt_module, "_current_git_sha", fake_git)
    monkeypatch.setattr(dt_module, "_prompt_fingerprint", fake_prompt)
    monkeypatch.setattr(dt_module, "_current_user", fake_user)

    tracker = DriftTracker(base_path=tmp_path)
    tracker._provenance()
    tracker._provenance()
    tracker._provenance()

    assert calls == {"git": 1, "prompt": 1, "user": 1}


# ────────────────────────── log grouping ──────────────────────────


def _entry(**kw: Any) -> Dict[str, Any]:
    base = {
        "ts": "2026-04-12T14:32:01",
        "test": "t",
        "status": "passed",
        "score_diff": 0.0,
        "output_similarity": 1.0,
        "tool_changes": 0,
        "model_changed": False,
        "git_sha": None,
        "prompt_hash": None,
        "model_id": None,
        "user": None,
    }
    base.update(kw)
    return base


def test_group_runs_clusters_same_sha_same_minute() -> None:
    entries = [
        _entry(test="a", git_sha="abc1234", ts="2026-04-12T14:32:01"),
        _entry(test="b", git_sha="abc1234", ts="2026-04-12T14:32:05"),
        _entry(test="c", git_sha="abc1234", ts="2026-04-12T14:32:12"),
    ]
    runs = _group_runs(entries)
    assert len(runs) == 1
    assert runs[0]["test_count"] == 3
    assert runs[0]["git_sha"] == "abc1234"


def test_group_runs_separates_distinct_shas() -> None:
    entries = [
        _entry(test="a", git_sha="aaa0000", ts="2026-04-12T14:32:01"),
        _entry(test="b", git_sha="bbb0000", ts="2026-04-12T14:32:05"),
    ]
    runs = _group_runs(entries)
    assert len(runs) == 2


def test_group_runs_without_git_sha_uses_second_precision() -> None:
    """P1 regression: two back-to-back runs without a git_sha must NOT
    collapse into one row just because they share the same minute."""
    entries = [
        _entry(test="a", ts="2026-04-12T14:32:01"),
        _entry(test="b", ts="2026-04-12T14:32:02"),  # 1s later
    ]
    runs = _group_runs(entries)
    assert len(runs) == 2


def test_group_runs_counts_regressions() -> None:
    entries = [
        _entry(test="a", git_sha="sha1", status="passed"),
        _entry(test="b", git_sha="sha1", status="regression"),
        _entry(test="c", git_sha="sha1", status="tools_changed"),
    ]
    runs = _group_runs(entries)
    assert len(runs) == 1
    run = runs[0]
    assert run["pass"] == 1
    assert run["regression"] == 1
    assert run["tools_changed"] == 1


# ────────────────────────── Recommendation fields ──────────────────────────


def test_model_changed_rec_has_all_new_fields() -> None:
    recs = recommend(status="regression", model_changed=True)
    assert recs
    rec = next(r for r in recs if r.category == "model")
    assert rec.severity == "high"
    assert rec.likely_cause  # non-empty
    assert rec.suggested_commands  # non-empty
    assert any("statistical" in c for c in rec.suggested_commands)


def test_tool_rec_fields_populated() -> None:
    recs = recommend(
        status="tools_changed",
        tools_added=["web_search"],
        tools_removed=["local_search"],
    )
    assert recs
    rec = next(r for r in recs if r.category == "tool")
    assert rec.severity == "high"
    assert rec.suggested_commands


def test_rec_to_dict_includes_new_fields() -> None:
    rec = Recommendation(
        action="Do X",
        confidence="high",
        category="tool",
        detail="because Y",
        likely_cause="tool renamed",
        severity="high",
        suggested_commands=["evalview replay foo"],
    )
    payload = rec.to_dict()
    assert payload["likely_cause"] == "tool renamed"
    assert payload["severity"] == "high"
    assert payload["suggested_commands"] == ["evalview replay foo"]


# ────────────────────────── dedup + substitution ──────────────────────────


def test_dedup_keeps_highest_confidence_copy() -> None:
    a_high = Recommendation(
        action="Pin model", confidence="high", category="model",
        detail="", likely_cause="", severity="high",
    )
    a_low = Recommendation(
        action="Pin model", confidence="low", category="model",
        detail="", likely_cause="", severity="high",
    )
    b = Recommendation(
        action="Fix prompt", confidence="medium", category="prompt",
        detail="", likely_cause="", severity="medium",
    )
    deduped = _dedup_recommendations([a_low, b, a_high])
    # Order is not guaranteed; check by membership.
    actions = {r.action for r in deduped}
    assert actions == {"Pin model", "Fix prompt"}
    pinned = next(r for r in deduped if r.action == "Pin model")
    assert pinned.confidence == "high"  # low copy was dropped


def test_substitute_test_name_replaces_placeholders() -> None:
    out = _substitute_test_name(
        ["evalview replay <test>", "evalview golden update <test_name>", "noop"],
        "search_cases",
    )
    assert out == [
        "evalview replay search_cases",
        "evalview golden update search_cases",
        "noop",
    ]


# ────────────────────────── cost delta aggregation ──────────────────────────


def _mk_trace_diff(name: str, status: str = "passed") -> Any:
    from evalview.core.diff import DiffStatus
    diff = MagicMock()
    diff.overall_severity = DiffStatus[status.upper()]
    return (name, diff)


def _mk_result(name: str, cost: float) -> Any:
    r = MagicMock()
    r.test_case = name
    r.trace.metrics.total_cost = cost
    return r


def _mk_golden(cost: float) -> Any:
    g = MagicMock()
    g.trace.metrics.total_cost = cost
    return g


def test_cost_delta_ratio_none_when_no_results() -> None:
    ratio = _aggregate_cost_delta_ratio(
        diffs=[_mk_trace_diff("a")], results=[], golden_traces={"a": _mk_golden(1.0)},
    )
    assert ratio is None


def test_cost_delta_ratio_none_when_no_goldens() -> None:
    ratio = _aggregate_cost_delta_ratio(
        diffs=[_mk_trace_diff("a")], results=[_mk_result("a", 1.0)], golden_traces={},
    )
    assert ratio is None


def test_cost_delta_ratio_computes_aggregate_delta() -> None:
    diffs = [_mk_trace_diff("a"), _mk_trace_diff("b")]
    results = [_mk_result("a", 1.20), _mk_result("b", 0.80)]  # total 2.00
    goldens = {"a": _mk_golden(1.00), "b": _mk_golden(1.00)}   # total 2.00
    ratio = _aggregate_cost_delta_ratio(diffs, results, goldens)
    assert ratio == pytest.approx(0.0)


def test_cost_delta_ratio_positive_when_current_more_expensive() -> None:
    diffs = [_mk_trace_diff("a")]
    results = [_mk_result("a", 1.25)]
    goldens = {"a": _mk_golden(1.00)}
    ratio = _aggregate_cost_delta_ratio(diffs, results, goldens)
    assert ratio == pytest.approx(0.25)


def test_cost_delta_ratio_skips_unmatched_tests() -> None:
    """Only tests with both a result AND a golden count toward the ratio."""
    diffs = [_mk_trace_diff("a"), _mk_trace_diff("b")]
    results = [_mk_result("a", 2.0), _mk_result("b", 10.0)]  # b has no golden
    goldens = {"a": _mk_golden(1.0)}
    ratio = _aggregate_cost_delta_ratio(diffs, results, goldens)
    assert ratio == pytest.approx(1.0)  # (2.0 - 1.0) / 1.0 — b excluded
