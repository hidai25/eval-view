"""Tests for Week 3 commands: progress, drift, slack-digest,
graded drift confidence, replay stability check.

Each test is pure and table-driven where possible so regressions are
obvious.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import pytest

from evalview.commands.progress_cmd import (
    _compute_delta,
    _consecutive_pass_count,
    _latest_status_per_test,
)
from evalview.commands.drift_cmd import (
    _build_rows,
    _classify,
    _incident_markers,
    _parse_last,
    _per_test_series,
    _sparkline,
    _status_transitions,
)
from evalview.commands.slack_digest_cmd import _build_message, _next_action
from evalview.core.drift_tracker import DriftTracker
from evalview.core.noise_tracker import NoiseStats, SuppressedEntry


def _entry(**kw: Any) -> Dict[str, Any]:
    base: Dict[str, Any] = {
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


# ────────────────────────── progress_cmd ──────────────────────────


def test_latest_status_per_test_keeps_newest() -> None:
    entries = [
        _entry(test="a", ts="2026-04-10T10:00:00", status="regression"),
        _entry(test="a", ts="2026-04-11T10:00:00", status="passed"),
        _entry(test="b", ts="2026-04-11T10:00:00", status="tools_changed"),
    ]
    latest = _latest_status_per_test(entries)
    assert latest["a"]["status"] == "passed"  # newest wins
    assert latest["b"]["status"] == "tools_changed"


def test_consecutive_pass_count_from_end() -> None:
    entries = [
        _entry(test="a", status="passed"),
        _entry(test="a", status="regression"),
        _entry(test="a", status="passed"),
        _entry(test="a", status="passed"),
        _entry(test="a", status="passed"),
    ]
    assert _consecutive_pass_count(entries, "a") == 3


def test_consecutive_pass_count_stops_at_first_non_pass() -> None:
    entries = [
        _entry(test="a", status="passed"),
        _entry(test="a", status="passed"),
        _entry(test="a", status="regression"),  # breaks the streak (from end)
    ]
    assert _consecutive_pass_count(entries, "a") == 0


def test_compute_delta_improved_transition() -> None:
    before = [_entry(test="a", status="regression")]
    after = [_entry(test="a", status="passed", git_sha="a1b2c3d")]
    delta = _compute_delta(before, after)
    assert len(delta["improved"]) == 1
    assert delta["improved"][0][0] == "a"
    assert delta["improved"][0][1] == "a1b2c3d"
    assert len(delta["regressed"]) == 0


def test_compute_delta_regression_transition() -> None:
    before = [_entry(test="a", status="passed")]
    after = [_entry(test="a", status="regression")]
    delta = _compute_delta(before, after)
    assert len(delta["regressed"]) == 1
    assert len(delta["improved"]) == 0


def test_compute_delta_still_broken_not_counted_as_improvement() -> None:
    before = [_entry(test="a", status="regression")]
    after = [_entry(test="a", status="tools_changed")]
    delta = _compute_delta(before, after)
    assert "a" in delta["still_broken"]
    assert len(delta["improved"]) == 0


def test_compute_delta_new_test_passing_counts_as_improvement() -> None:
    before: List[Dict[str, Any]] = []
    after = [_entry(test="new", status="passed")]
    delta = _compute_delta(before, after)
    assert len(delta["improved"]) == 1
    assert delta["new_tests_count"] == 1


def test_compute_delta_new_test_failing_is_still_broken() -> None:
    before: List[Dict[str, Any]] = []
    after = [_entry(test="new", status="regression")]
    delta = _compute_delta(before, after)
    assert "new" in delta["still_broken"]
    assert len(delta["improved"]) == 0


def test_compute_delta_worth_commit_high_confidence() -> None:
    """3+ consecutive passes in the after window = high confidence."""
    before = [_entry(test="a", status="regression")]
    after = [
        _entry(test="a", status="passed", ts="2026-04-11T10:00:00"),
        _entry(test="a", status="passed", ts="2026-04-11T11:00:00"),
        _entry(test="a", status="passed", ts="2026-04-11T12:00:00"),
    ]
    delta = _compute_delta(before, after)
    worth = delta["worth_commit"]
    assert len(worth) == 1
    assert worth[0] == ("a", "high")


def test_compute_delta_worth_commit_medium_confidence() -> None:
    before = [_entry(test="a", status="regression")]
    after = [_entry(test="a", status="passed")]  # 1 sample only
    delta = _compute_delta(before, after)
    assert len(delta["worth_commit"]) == 1
    assert delta["worth_commit"][0][1] == "medium"


def test_compute_delta_aggregates_similarity_lift() -> None:
    before = [_entry(output_similarity=0.80), _entry(output_similarity=0.82)]
    after = [_entry(output_similarity=0.90), _entry(output_similarity=0.92)]
    delta = _compute_delta(before, after)
    assert delta["avg_similarity_before"] == pytest.approx(0.81)
    assert delta["avg_similarity_after"] == pytest.approx(0.91)
    assert delta["avg_similarity_delta"] == pytest.approx(0.10)


# ────────────────────────── drift_cmd ──────────────────────────


def test_sparkline_renders() -> None:
    s = _sparkline([0.5, 0.6, 0.7, 0.8])
    assert len(s) == 4


def test_sparkline_all_equal_renders_mid_height() -> None:
    s = _sparkline([0.5, 0.5, 0.5])
    assert len(set(s)) == 1  # all same glyph


def test_sparkline_empty() -> None:
    assert _sparkline([]) == ""


def test_status_transitions_detects_flip() -> None:
    series = [
        ("t1", "passed"),
        ("t2", "passed"),
        ("t3", "regression"),  # flip at idx 2
        ("t4", "regression"),
        ("t5", "passed"),       # flip at idx 4
    ]
    transitions = _status_transitions(series)
    assert transitions == [2, 4]


def test_incident_markers_aligns_with_sparkline() -> None:
    markers = _incident_markers("▁▂▃▄▅", [2])
    assert markers == "  !  "


def test_parse_last_days() -> None:
    td = _parse_last("7d")
    assert td is not None
    assert td.days == 7


def test_parse_last_hours() -> None:
    td = _parse_last("24h")
    assert td is not None
    assert td.total_seconds() == 24 * 3600


def test_parse_last_minutes() -> None:
    td = _parse_last("30m")
    assert td is not None
    assert td.total_seconds() == 30 * 60


def test_parse_last_invalid_returns_none() -> None:
    assert _parse_last("nonsense") is None
    assert _parse_last(None) is None


def test_classify_insufficient_when_few_samples() -> None:
    color, label = _classify(-0.05, 2)
    # Label matches drift_tracker.classify_drift's vocabulary so both
    # surfaces agree on what "not enough data" is called.
    assert label == "insufficient_history"


def test_classify_declining_when_steep() -> None:
    color, label = _classify(-0.05, 10)
    assert label == "declining"
    assert color == "red"


def test_classify_soft_decline() -> None:
    _, label = _classify(-0.015, 10)
    assert label == "soft decline"


def test_build_rows_returns_per_test_series() -> None:
    entries = [
        _entry(test="a", output_similarity=0.9),
        _entry(test="a", output_similarity=0.85),
        _entry(test="a", output_similarity=0.80),
        _entry(test="b", output_similarity=0.95),
        _entry(test="b", output_similarity=0.96),
        _entry(test="b", output_similarity=0.97),
    ]
    per_test = _per_test_series(entries)
    rows = _build_rows(per_test, sort_worst=True, sample_cap=10)
    # Worst-first sort: 'a' (declining) should come before 'b' (improving)
    assert rows[0]["test"] == "a"
    assert rows[1]["test"] == "b"


def test_build_rows_skips_tests_without_output_similarity() -> None:
    entries = [
        _entry(test="a", output_similarity=None),
        _entry(test="a", output_similarity=None),
    ]
    per_test = _per_test_series(entries)
    rows = _build_rows(per_test, sort_worst=False, sample_cap=10)
    assert rows == []


# ────────────────────────── drift_tracker.classify_drift ──────────────────────────


def _seed_history(tmp_path: Path, test_name: str, sims: List[float]) -> DriftTracker:
    """Write a history.jsonl file with `sims` values and return a tracker."""
    history_dir = tmp_path / ".evalview"
    history_dir.mkdir()
    with (history_dir / "history.jsonl").open("w") as f:
        for i, sim in enumerate(sims):
            f.write(json.dumps({
                "ts": f"2026-04-{10 + i:02d}T10:00:00",
                "test": test_name,
                "status": "passed",
                "score_diff": 0.0,
                "output_similarity": sim,
                "tool_changes": 0,
                "model_changed": False,
            }) + "\n")
    return DriftTracker(base_path=tmp_path)


def test_classify_drift_insufficient_history(tmp_path: Path) -> None:
    tracker = _seed_history(tmp_path, "t1", [0.95, 0.94])  # only 2 samples
    tier, slope = tracker.classify_drift("t1")
    assert tier == "insufficient_history"
    assert slope is None


def test_classify_drift_stable(tmp_path: Path) -> None:
    tracker = _seed_history(tmp_path, "t1", [0.95, 0.95, 0.95, 0.95])
    tier, _slope = tracker.classify_drift("t1")
    assert tier == "stable"


def test_classify_drift_high_confidence(tmp_path: Path) -> None:
    # Steep decline: ~3%/check
    tracker = _seed_history(tmp_path, "t1", [0.97, 0.94, 0.91, 0.88, 0.85])
    tier, slope = tracker.classify_drift("t1")
    assert tier == "high"
    assert slope is not None and slope < -0.025


def test_classify_drift_medium_confidence(tmp_path: Path) -> None:
    # Moderate decline: ~2%/check — between medium and high threshold
    tracker = _seed_history(tmp_path, "t1", [0.97, 0.95, 0.93, 0.91, 0.89])
    tier, _slope = tracker.classify_drift("t1")
    assert tier == "medium"


def test_classify_drift_low_confidence(tmp_path: Path) -> None:
    # Slight decline: ~1%/check
    tracker = _seed_history(tmp_path, "t1", [0.97, 0.96, 0.95, 0.94, 0.93])
    tier, _slope = tracker.classify_drift("t1")
    assert tier == "low"


# ────────────────────────── slack digest message builder ──────────────────────────


def test_build_message_empty_window() -> None:
    payload = _build_message(
        "yesterday",
        window={"total": 0, "pass_rate": None, "regression": 0, "tools_changed": 0, "output_changed": 0},
        drift_rows=[],
        stale_quarantine=[],
    )
    assert "No runs" in payload["text"] or "No runs" in str(payload["blocks"])


def test_build_message_with_regressions_includes_emoji() -> None:
    window = {
        "total": 10,
        "pass_rate": 0.8,
        "regression": 2,
        "tools_changed": 0,
        "output_changed": 0,
    }
    payload = _build_message("yesterday", window, [], [])
    text = json.dumps(payload)
    assert "80%" in text
    assert "regression" in text.lower()


def test_build_message_green_headline_on_full_pass() -> None:
    window = {
        "total": 20,
        "pass_rate": 1.0,
        "regression": 0,
        "tools_changed": 0,
        "output_changed": 0,
    }
    payload = _build_message("yesterday", window, [], [])
    # Inspect the raw payload — json.dumps escapes emoji to \u sequences
    # which would make the assertion lie about what's rendered in Slack.
    assert "🟢" in payload["text"]


def test_build_message_omits_noise_section_when_no_activity() -> None:
    """Empty noise stats should leave the digest tight — no
    "0% noise" line that would just add clutter on a quiet week."""
    window = {
        "total": 10,
        "pass_rate": 1.0,
        "regression": 0,
        "tools_changed": 0,
        "output_changed": 0,
    }
    payload = _build_message(
        "yesterday", window, [], [], noise_stats=NoiseStats()
    )
    text = json.dumps(payload)
    assert "Noise" not in text
    assert "suppressed" not in text


def test_build_message_renders_noise_section_when_alerts_present() -> None:
    window = {
        "total": 10,
        "pass_rate": 0.9,
        "regression": 1,
        "tools_changed": 0,
        "output_changed": 0,
    }
    noise = NoiseStats(alerts_fired=4, real_alerts=4, suppressed=1)
    payload = _build_message(
        "yesterday", window, [], [], noise_stats=noise
    )
    text = json.dumps(payload)
    assert "Noise" in text
    # Publicly reported false-positive rate: 1 / (4+1) = 20%.
    assert "20%" in text
    assert "4 fired" in text
    assert "1 suppressed" in text


def test_build_message_renders_suppressed_test_list() -> None:
    """The digest must surface the LIST of suppressed tests, not just
    a count — otherwise suppression becomes a hidden-signal failure
    mode. Users need to be able to see "flaky-search self-resolved 3
    times this week" and decide whether to investigate."""
    window = {
        "total": 10,
        "pass_rate": 0.9,
        "regression": 0,
        "tools_changed": 0,
        "output_changed": 0,
    }
    noise = NoiseStats(
        alerts_fired=2,
        real_alerts=2,
        suppressed=4,
        suppressed_by_test=[
            SuppressedEntry(test_name="flaky-search", count=3, last_seen=""),
            SuppressedEntry(test_name="auth-retry", count=1, last_seen=""),
        ],
    )
    payload = _build_message(
        "yesterday", window, [], [], noise_stats=noise
    )
    # Inspect the raw payload — json.dumps escapes unicode (× becomes
    # \u00d7) which would make the assertion lie about what's rendered.
    rendered_text = ""
    for block in payload["blocks"]:
        if block.get("type") == "section":
            rendered_text += block.get("text", {}).get("text", "") + "\n"
    assert "flaky-search" in rendered_text
    assert "auth-retry" in rendered_text
    # The count annotation appears for the tests that self-resolved
    # more than once.
    assert "× 3" in rendered_text
    assert "self-resolved" in rendered_text.lower()


def test_build_message_escapes_backticks_in_suppressed_test_names() -> None:
    """A test name containing a backtick must not break the inline
    code span that wraps it in the digest. Same markdown-safety rule
    Week 2 enforced for the PR comment."""
    window = {
        "total": 10,
        "pass_rate": 0.9,
        "regression": 0,
        "tools_changed": 0,
        "output_changed": 0,
    }
    noise = NoiseStats(
        alerts_fired=1,
        real_alerts=1,
        suppressed=1,
        suppressed_by_test=[
            SuppressedEntry(
                test_name="weird`name", count=1, last_seen=""
            ),
        ],
    )
    payload = _build_message(
        "yesterday", window, [], [], noise_stats=noise
    )
    rendered = ""
    for block in payload["blocks"]:
        if block.get("type") == "section":
            rendered += block.get("text", {}).get("text", "") + "\n"
    # Escaped form must appear, raw form must not — otherwise the
    # inline code span (wrapped in backticks) closes too early.
    assert "weird&#96;name" in rendered
    assert "`weird`name`" not in rendered  # would break the span


def test_build_message_suppressed_list_caps_at_five() -> None:
    """Bounded list so the digest doesn't become its own firehose —
    show the top 5 + a "…and N more" tail for the long tail."""
    window = {
        "total": 10,
        "pass_rate": 1.0,
        "regression": 0,
        "tools_changed": 0,
        "output_changed": 0,
    }
    entries = [
        SuppressedEntry(test_name=f"t-{i}", count=1, last_seen="")
        for i in range(8)
    ]
    noise = NoiseStats(
        alerts_fired=1, real_alerts=1, suppressed=8, suppressed_by_test=entries
    )
    payload = _build_message(
        "yesterday", window, [], [], noise_stats=noise
    )
    text = json.dumps(payload)
    assert "and 3 more" in text


def test_build_message_noise_section_green_when_clean() -> None:
    """A low-noise week should lead with the green emoji — the whole
    point of making this public is to reward being quiet."""
    window = {
        "total": 20,
        "pass_rate": 1.0,
        "regression": 0,
        "tools_changed": 0,
        "output_changed": 0,
    }
    noise = NoiseStats(alerts_fired=20, real_alerts=20, suppressed=0)
    payload = _build_message(
        "yesterday", window, [], [], noise_stats=noise
    )
    # Don't use json.dumps — it escapes emoji.
    found_noise_block = False
    for block in payload["blocks"]:
        if block.get("type") == "section":
            txt = block.get("text", {}).get("text", "")
            if "Noise" in txt:
                found_noise_block = True
                assert "🟢" in txt
                assert "0% noise" in txt
    assert found_noise_block


def test_next_action_prioritizes_regression() -> None:
    action = _next_action(
        {"regression": 1}, [("a", "▅▆▇")], [{"test_name": "q"}],
    )
    assert "fail-on REGRESSION" in action


def test_next_action_falls_through_to_drift() -> None:
    action = _next_action({"regression": 0}, [("search_cases", "▅▄▃")], [])
    assert "drift" in action
    assert "search_cases" in action


def test_next_action_clean_state() -> None:
    action = _next_action({"regression": 0}, [], [])
    assert "stable" in action.lower()


# ────────────────────────── replay stability check integration ──────────────────────────


def test_investigate_verdict_injects_stability_recommendation() -> None:
    """When the verdict is INVESTIGATE, _compute_verdict_payload must
    prepend a `--statistical 5` rerun rec at position 0 — AND it must
    survive the severity/confidence sort that runs over the existing
    recommendations (this is the P0.1 regression guard)."""
    from evalview.commands.check_cmd import _compute_verdict_payload
    from unittest.mock import MagicMock
    from evalview.core.diff import DiffStatus

    def mk_diff(name: str, status: DiffStatus, tool_changes: int = 0) -> Any:
        d = MagicMock()
        d.overall_severity = status
        # Non-empty tool_diffs so the recommendations engine produces a
        # real high-severity rec to compete with the stability rec.
        d.tool_diffs = [MagicMock(type="added", actual_tool=f"new_{name}")] * tool_changes
        d.output_diff = None
        d.score_diff = 0.0
        return (name, d)

    # Multiple soft changes → INVESTIGATE tier, AND the recommendation
    # engine will produce high-severity tool-change recs that must NOT
    # outrank the stability rec.
    diffs = [
        mk_diff("t1", DiffStatus.TOOLS_CHANGED, tool_changes=2),
        mk_diff("t2", DiffStatus.TOOLS_CHANGED, tool_changes=2),
    ]

    out = _compute_verdict_payload(
        diffs=diffs,
        results=[],
        drift_tracker=None,
        execution_failures=0,
        golden_traces=None,
        quarantine=None,
    )
    recs = out.payload.get("recommendations", [])
    assert recs, "Expected at least one recommendation on INVESTIGATE"
    top = recs[0]
    assert "statistical" in top["action"].lower(), (
        f"Stability rec should be at position 0, got: {top['action']}"
    )
    assert any("--statistical" in c for c in top.get("suggested_commands", []))


def test_block_release_does_not_inject_stability_rec() -> None:
    """Stability rec is ONLY for INVESTIGATE — BLOCK_RELEASE should
    show the real cause, not a 'rerun and see' suggestion."""
    from evalview.commands.check_cmd import _compute_verdict_payload
    from unittest.mock import MagicMock
    from evalview.core.diff import DiffStatus

    d = MagicMock()
    d.overall_severity = DiffStatus.REGRESSION
    d.tool_diffs = [MagicMock(type="added", actual_tool="new_tool")]
    d.output_diff = None
    d.score_diff = -6.0

    out = _compute_verdict_payload(
        diffs=[("t1", d)],
        results=[],
        drift_tracker=None,
        execution_failures=0,
        golden_traces=None,
        quarantine=None,
    )
    recs = out.payload.get("recommendations", [])
    if recs:
        top = recs[0]
        assert "statistical" not in top["action"].lower(), (
            "Stability rec should not fire on BLOCK_RELEASE — the verdict "
            "has enough signal to act on, we don't need the 'rerun and see' hedge"
        )
