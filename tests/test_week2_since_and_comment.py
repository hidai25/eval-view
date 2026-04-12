"""Week 2 integration: `evalview since` brief + verdict-first PR comment.

Covers:
  - _parse_since handling of "yesterday", "Nd", ISO date, git SHA, None
  - _cutoff_of_previous_run distinguishes runs via (sha, prompt, ts_bucket)
  - _entries_since filtering
  - _summarize aggregates
  - Verdict-first PR comment rendering (headline + signals + rec)
  - PR comment falls back to legacy format when no verdict payload
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List

import pytest

from evalview.commands.since_cmd import (
    _cutoff_of_previous_run,
    _entries_since,
    _parse_since,
    _sparkline,
    _summarize,
)
from evalview.ci.comment import (
    _build_recommendation_block,
    _build_verdict_header,
    _build_verdict_signals_table,
    generate_check_pr_comment,
)


# ────────────────────────── _parse_since ──────────────────────────


def test_parse_since_yesterday_returns_24h_cutoff() -> None:
    cutoff, sha, label = _parse_since("yesterday", [])
    assert sha is None
    assert cutoff is not None
    delta = datetime.now(timezone.utc) - cutoff
    assert 0.9 < delta.total_seconds() / 86400 < 1.1
    assert label == "yesterday"


def test_parse_since_days_shorthand() -> None:
    cutoff, sha, label = _parse_since("7d", [])
    assert cutoff is not None
    delta = datetime.now(timezone.utc) - cutoff
    assert 6.9 < delta.total_seconds() / 86400 < 7.1
    assert label == "7 days ago"


def test_parse_since_iso_date() -> None:
    cutoff, sha, label = _parse_since("2026-01-01", [])
    assert cutoff is not None
    assert cutoff.year == 2026 and cutoff.month == 1 and cutoff.day == 1
    assert label == "2026-01-01"


def test_parse_since_git_sha_found_in_history() -> None:
    entries = [
        {"ts": "2026-04-10T10:00:00", "git_sha": "abc1234", "test": "t1"},
        {"ts": "2026-04-11T10:00:00", "git_sha": "def5678", "test": "t2"},
    ]
    cutoff, sha, label = _parse_since("abc1234", entries)
    assert sha is not None
    assert sha.startswith("abc1234")
    assert label.startswith("commit")


def test_parse_since_unknown_falls_back_to_previous_run() -> None:
    entries = [
        {"ts": "2026-04-10T10:00:00", "git_sha": "abc1234"},
        {"ts": "2026-04-11T10:00:00", "git_sha": "def5678"},
    ]
    cutoff, sha, label = _parse_since("nonsense", entries)
    # Should not raise — falls back silently
    assert sha is None
    assert label == "nonsense"  # raw label preserved so the user sees we heard them


def test_parse_since_none_uses_previous_run_cutoff() -> None:
    entries = [
        {"ts": "2026-04-10T10:00:00", "git_sha": "a"},
        {"ts": "2026-04-11T10:00:00", "git_sha": "b"},
    ]
    cutoff, sha, label = _parse_since(None, entries)
    assert sha is None
    assert cutoff is not None
    assert label == "your last check"


def test_parse_since_empty_history_returns_ever() -> None:
    cutoff, sha, label = _parse_since(None, [])
    assert cutoff is None
    assert sha is None
    assert label == "ever"


# ────────────────────────── _cutoff_of_previous_run ──────────────────────────


def test_cutoff_of_previous_run_with_two_distinct_runs() -> None:
    entries = [
        {"ts": "2026-04-10T10:00:00", "git_sha": "sha1"},
        {"ts": "2026-04-10T10:00:05", "git_sha": "sha1"},
        {"ts": "2026-04-11T10:00:00", "git_sha": "sha2"},
        {"ts": "2026-04-11T10:00:05", "git_sha": "sha2"},
    ]
    cutoff = _cutoff_of_previous_run(entries)
    assert cutoff is not None
    # Should be the max timestamp of the earlier run (sha1 @ 10:00:05)
    assert cutoff.day == 10
    assert cutoff.hour == 10
    assert cutoff.minute == 0


def test_cutoff_of_previous_run_with_single_run_returns_none() -> None:
    entries = [
        {"ts": "2026-04-10T10:00:00", "git_sha": "only"},
        {"ts": "2026-04-10T10:00:05", "git_sha": "only"},
    ]
    cutoff = _cutoff_of_previous_run(entries)
    assert cutoff is None  # nothing to compare against


# ────────────────────────── _entries_since ──────────────────────────


def test_entries_since_filters_by_cutoff() -> None:
    entries = [
        {"ts": "2026-04-09T10:00:00"},
        {"ts": "2026-04-11T10:00:00"},
    ]
    cutoff = datetime(2026, 4, 10, tzinfo=timezone.utc)
    filtered = _entries_since(entries, cutoff, None)
    assert len(filtered) == 1
    assert filtered[0]["ts"] == "2026-04-11T10:00:00"


def test_entries_since_follows_sha_cutoff() -> None:
    entries = [
        {"ts": "t1", "git_sha": "aaa1111"},
        {"ts": "t2", "git_sha": "bbb2222"},
        {"ts": "t3", "git_sha": "ccc3333"},
    ]
    filtered = _entries_since(entries, None, "bbb2222")
    # Only entries AFTER the cutoff sha are included
    assert [e["ts"] for e in filtered] == ["t3"]


def test_entries_since_none_cutoff_returns_all() -> None:
    entries = [{"ts": "a"}, {"ts": "b"}]
    assert _entries_since(entries, None, None) == entries


# ────────────────────────── _summarize ──────────────────────────


def test_summarize_empty_history() -> None:
    out = _summarize([])
    assert out["total"] == 0
    assert out["pass_rate"] is None
    assert out["tests_improved"] == []


def test_summarize_mixed_statuses() -> None:
    entries = [
        {"test": "t1", "status": "passed", "score_diff": 0.1},
        {"test": "t1", "status": "passed", "score_diff": 0.2},
        {"test": "t2", "status": "regression", "score_diff": -3.0},
    ]
    out = _summarize(entries)
    assert out["total"] == 3
    assert out["passed"] == 2
    assert out["regression"] == 1
    assert out["pass_rate"] == pytest.approx(2 / 3)
    assert "t2" in out["tests_regressed"]


def test_summarize_detects_improved_tests() -> None:
    entries = [
        {"test": "improved", "status": "passed", "score_diff": 2.5},
        {"test": "flat", "status": "passed", "score_diff": 0.5},
    ]
    out = _summarize(entries)
    assert "improved" in out["tests_improved"]
    assert "flat" not in out["tests_improved"]


# ────────────────────────── sparkline ──────────────────────────


def test_sparkline_empty_returns_empty_string() -> None:
    assert _sparkline([]) == ""


def test_sparkline_single_span() -> None:
    """All-equal values render as a middle-height bar line."""
    out = _sparkline([0.5, 0.5, 0.5])
    assert len(out) == 3
    assert len(set(out)) == 1  # all same glyph


def test_sparkline_declining_trend_renders() -> None:
    out = _sparkline([1.0, 0.9, 0.8, 0.7])
    assert len(out) == 4
    # First glyph should be higher than last (declining trend)
    glyphs = "▁▂▃▄▅▆▇█"
    assert glyphs.index(out[0]) > glyphs.index(out[-1])


# ────────────────────────── verdict-first PR comment ──────────────────────────


def _make_check_data(
    total: int = 3,
    regressions: int = 0,
    verdict: Dict[str, Any] = None,
) -> Dict[str, Any]:
    return {
        "summary": {
            "total_tests": total,
            "unchanged": total - regressions,
            "regressions": regressions,
            "tools_changed": 0,
            "output_changed": 0,
        },
        "diffs": [],
        "verdict": verdict,
    }


def test_verdict_header_renders_safe_to_ship() -> None:
    verdict = {
        "verdict": "safe_to_ship",
        "headline": "SAFE TO SHIP",
        "reasons": ["All tests passed"],
    }
    lines = _build_verdict_header(verdict)
    assert any("Safe to ship" in line for line in lines)
    assert any("All tests passed" in line for line in lines)


def test_verdict_header_renders_block_release() -> None:
    verdict = {
        "verdict": "block_release",
        "reasons": ["1 regression: search_cases"],
    }
    lines = _build_verdict_header(verdict)
    assert any("Block release" in line for line in lines)
    assert any("regression" in line for line in lines)


def test_verdict_signals_table_shows_regressions() -> None:
    check = _make_check_data(total=10, regressions=2)
    verdict = {"verdict": "block_release", "reasons": []}
    lines = _build_verdict_signals_table(check, verdict)
    body = "\n".join(lines)
    assert "Regressions" in body
    assert "2" in body
    assert "8/10" in body


def test_verdict_signals_table_shows_cost_spike() -> None:
    check = _make_check_data()
    verdict = {"verdict": "investigate", "cost_delta_ratio": 0.25}
    lines = _build_verdict_signals_table(check, verdict)
    body = "\n".join(lines)
    assert "Cost" in body
    assert "25%" in body


def test_verdict_signals_table_shows_stale_quarantine() -> None:
    check = _make_check_data()
    verdict = {
        "verdict": "investigate",
        "quarantine": {
            "total": 3,
            "stale": 2,
            "stale_tests": ["old_retry", "old_login"],
        },
    }
    lines = _build_verdict_signals_table(check, verdict)
    body = "\n".join(lines)
    assert "stale" in body.lower()
    assert "`old_retry`" in body


def test_recommendation_block_renders_command() -> None:
    verdict = {
        "recommendations": [
            {
                "action": "Pin your model version",
                "confidence": "high",
                "category": "model",
                "detail": "The model changed",
                "likely_cause": "Provider updated",
                "severity": "high",
                "suggested_commands": [
                    "evalview replay search_cases --trace",
                ],
            }
        ],
    }
    lines = _build_recommendation_block(verdict)
    body = "\n".join(lines)
    assert "Pin your model version" in body
    assert "evalview replay search_cases --trace" in body
    assert "```bash" in body


def test_recommendation_block_empty_when_no_recs() -> None:
    assert _build_recommendation_block({}) == []


def test_generate_check_pr_comment_uses_verdict_when_present() -> None:
    verdict = {
        "verdict": "block_release",
        "reasons": ["1 regression in search_cases"],
        "recommendations": [
            {
                "action": "Do X",
                "confidence": "high",
                "suggested_commands": ["evalview replay search_cases"],
                "likely_cause": "Prompt changed",
                "severity": "high",
            }
        ],
    }
    check = _make_check_data(total=5, regressions=1, verdict=verdict)
    out = generate_check_pr_comment(check)
    assert "Block release" in out
    assert "evalview replay search_cases" in out
    assert "Prompt changed" in out


def test_generate_check_pr_comment_falls_back_to_legacy_without_verdict() -> None:
    """Old check --json output (no verdict key) must still render the
    pre-Week-2 comment so CI pipelines that haven't upgraded still work."""
    check = _make_check_data(total=5, regressions=1)
    check["verdict"] = None
    out = generate_check_pr_comment(check)
    assert "REGRESSION" in out.upper()  # legacy uses upper-case label
