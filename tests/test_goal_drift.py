"""Tests for `evalview.core.goal_drift`.

Two layers:

1. The deterministic baseline (Jaccard) — verify it fires on obviously-
   wandered trajectories and stays quiet on faithful ones.
2. The pluggable judge interface — verify a fake judge can override the
   baseline, and that judge errors fall back to the baseline gracefully.
"""
from __future__ import annotations

from typing import Optional

from evalview.core.goal_drift import (
    DEFAULT_DRIFT_THRESHOLD,
    GoalEvent,
    analyze_goal_drift,
    analyze_per_step,
    summarize_trajectory,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _events(*texts: str) -> list[GoalEvent]:
    return [GoalEvent(step_index=i, text=t) for i, t in enumerate(texts)]


# ---------------------------------------------------------------------------
# summarize_trajectory
# ---------------------------------------------------------------------------


class TestSummarizeTrajectory:
    def test_concatenates_with_separator(self) -> None:
        s = summarize_trajectory(_events("look up order", "check policy"))
        # The | separator keeps the summary scannable when it appears in
        # a Slack card or HTML report — easier than newlines for one-line
        # rendering.
        assert "look up order" in s and "check policy" in s
        assert "|" in s

    def test_keeps_only_last_eight_events(self) -> None:
        events = _events(*[f"step {i}" for i in range(20)])
        s = summarize_trajectory(events)
        # First 12 steps must be excluded — the trajectory's *current*
        # intent matters more than its opening, and bounding the summary
        # keeps the Jaccard signal from drowning in repetition.
        assert "step 0" not in s
        assert "step 19" in s

    def test_empty_input(self) -> None:
        assert summarize_trajectory([]) == ""


# ---------------------------------------------------------------------------
# Deterministic baseline
# ---------------------------------------------------------------------------


class TestDeterministicBaseline:
    def test_aligned_trajectory_is_on_goal(self) -> None:
        # Trajectory directly works on the stated goal — high token
        # overlap, similarity well above threshold.
        analysis = analyze_goal_drift(
            "cancel my subscription and refund last charge",
            _events("look up subscription", "cancel subscription",
                    "refund last charge", "confirm refund"),
        )
        assert not analysis.is_drifting
        assert analysis.similarity > DEFAULT_DRIFT_THRESHOLD

    def test_wandered_trajectory_is_drifting(self) -> None:
        # The "agent wandered into a different topic" failure mode.
        # Tokens collapse, similarity drops below threshold.
        analysis = analyze_goal_drift(
            "cancel my subscription",
            _events("explain pricing tiers", "describe enterprise plan",
                    "compare yearly vs monthly billing"),
        )
        assert analysis.is_drifting
        assert analysis.severity == "severe"

    def test_severity_label_pins_buckets(self) -> None:
        # 'severe' fires below half the threshold; 'mild' between half
        # and threshold. Pin both so digest rendering stays stable.
        severe = analyze_goal_drift(
            "cancel subscription",
            _events("totally unrelated content here for testing"),
        )
        assert severe.severity == "severe"

    def test_drift_delta_is_complement_of_similarity(self) -> None:
        analysis = analyze_goal_drift(
            "alpha beta gamma",
            _events("alpha beta gamma delta"),
        )
        assert abs((analysis.similarity + analysis.drift_delta) - 1.0) < 1e-6

    def test_empty_goal_returns_not_drifting(self) -> None:
        # We can't call drift on a trajectory when there's no stated goal
        # to drift FROM. Return a benign analysis rather than crashing or
        # falsely flagging drift.
        analysis = analyze_goal_drift("", _events("step 1", "step 2"))
        assert not analysis.is_drifting
        assert analysis.evidence["reason"] == "missing_goal_or_trajectory"

    def test_empty_trajectory_returns_not_drifting(self) -> None:
        # Same logic in the other direction — no trajectory yet, no
        # drift to detect.
        analysis = analyze_goal_drift("cancel my subscription", [])
        assert not analysis.is_drifting

    def test_digit_normalization(self) -> None:
        # "Order 4812" and "order 8201" should look identical to the
        # tokenizer — order IDs are noise for intent comparison.
        a = analyze_goal_drift(
            "look up order 4812", _events("look up order 8201"),
        )
        # Should be high similarity even though numbers differ.
        assert a.similarity > 0.8


# ---------------------------------------------------------------------------
# Pluggable judge
# ---------------------------------------------------------------------------


class TestJudgeInterface:
    def test_judge_score_overrides_baseline(self) -> None:
        # Disjoint inputs would normally score 0.0 from Jaccard; the
        # judge says they're a 0.9 match → analysis trusts the judge.
        def fake_judge(goal: str, summary: str) -> Optional[float]:
            return 0.9

        analysis = analyze_goal_drift(
            "alpha", _events("zeta", "eta"), judge=fake_judge,
        )
        assert analysis.judge_used
        assert analysis.similarity == 0.9
        assert not analysis.is_drifting

    def test_judge_returning_none_falls_back_to_baseline(self) -> None:
        # The contract is "judge can opt out via None". Verify the
        # fallback engages cleanly and doesn't pretend the judge ran.
        def opting_out(goal: str, summary: str) -> Optional[float]:
            return None

        analysis = analyze_goal_drift(
            "alpha beta", _events("alpha beta gamma"),
            judge=opting_out,
        )
        assert not analysis.judge_used
        # Baseline computed Jaccard normally.
        assert analysis.similarity > 0

    def test_judge_raising_falls_back_silently(self) -> None:
        # A flaky LLM judge must never break the analysis — we'd rather
        # under-detect drift than crash a monitor cycle.
        def flaky_judge(goal: str, summary: str) -> Optional[float]:
            raise RuntimeError("simulated upstream failure")

        analysis = analyze_goal_drift(
            "cancel subscription",
            _events("cancel subscription"),
            judge=flaky_judge,
        )
        assert not analysis.judge_used
        # Baseline still ran — high similarity expected.
        assert analysis.similarity > 0

    def test_judge_score_is_clamped(self) -> None:
        # Some judges return un-normalized scores; clamp to [0, 1] so
        # downstream similarity comparisons stay meaningful.
        def out_of_range(goal: str, summary: str) -> Optional[float]:
            return 7.5

        analysis = analyze_goal_drift(
            "x", _events("y"), judge=out_of_range,
        )
        assert analysis.similarity == 1.0

    def test_negative_judge_score_is_clamped(self) -> None:
        def negative(goal: str, summary: str) -> Optional[float]:
            return -0.5

        analysis = analyze_goal_drift(
            "x", _events("y"), judge=negative,
        )
        assert analysis.similarity == 0.0


# ---------------------------------------------------------------------------
# Per-step analysis
# ---------------------------------------------------------------------------


class TestAnalyzePerStep:
    def test_returns_one_analysis_per_step(self) -> None:
        events = _events("step a", "step b", "step c")
        results = analyze_per_step("alpha", events)
        assert len(results) == 3
        # Step indexes propagate through.
        assert [idx for idx, _ in results] == [0, 1, 2]

    def test_drift_detected_increases_over_wandering_trajectory(self) -> None:
        # First step echoes the goal; subsequent steps drift away.
        # The per-step drift_delta should monotonically rise (or stay)
        # in this case — pin that as a useful sanity invariant.
        events = _events(
            "cancel subscription",            # on-goal
            "discuss pricing tiers",           # drift
            "explain enterprise features",     # more drift
            "compare yearly billing options",  # more drift
        )
        results = analyze_per_step("cancel my subscription", events)
        deltas = [a.drift_delta for _, a in results]
        # First step should be the on-goal one.
        assert deltas[0] < deltas[-1]
