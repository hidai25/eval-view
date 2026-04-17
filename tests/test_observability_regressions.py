"""Regression tests for the observability stack hardening.

Each test here guards a specific bug that has been fixed once and should
not return quietly.  They assert concrete values — not "doesn't crash" or
"severity != CRITICAL" — so a no-op implementation cannot pass.
"""
from __future__ import annotations

from datetime import datetime
from typing import List, Optional

from evalview.core.benchmark_hardening import (
    FlagSeverity,
    GamingCheck,
    _check_abnormal_file_access,
    _check_score_without_work,
    _check_too_perfect,
)
from evalview.core.observability import (
    LOW_TRUST_THRESHOLD,
    OBSERVABILITY_SCHEMA_VERSION,
    ObservabilitySummary,
    extract_observability_summary,
)
from evalview.core.turn_coherence import (
    CoherenceCategory,
    _detect_tool_regression,
    analyze_coherence,
)
from evalview.core.types import (
    ExecutionMetrics,
    ExecutionTrace,
    StepMetrics,
    StepTrace,
    TurnTrace,
)


# ── Fixture helpers ─────────────────────────────────────────────────────────


def _step(name: str, turn_index: int = 1, params: Optional[dict] = None) -> StepTrace:
    return StepTrace(
        step_id=f"s-{name}-{turn_index}",
        step_name=name,
        tool_name=name,
        parameters=params or {},
        output="ok",
        success=True,
        metrics=StepMetrics(latency=100, cost=0.01),
        turn_index=turn_index,
    )


def _turn(idx: int, query: str = "q", output: str = "o") -> TurnTrace:
    return TurnTrace(index=idx, query=query, output=output, tools=[])


def _trace(
    *,
    steps: Optional[List[StepTrace]] = None,
    turns: Optional[List[TurnTrace]] = None,
    latency: float = 1000.0,
) -> ExecutionTrace:
    return ExecutionTrace(
        session_id="test",
        start_time=datetime(2025, 1, 1),
        end_time=datetime(2025, 1, 1, 0, 5),
        steps=steps or [],
        final_output="done",
        metrics=ExecutionMetrics(total_cost=0.1, total_latency=latency),
        turns=turns or [],
    )


# ── 1. Tool regression dedup (turn_coherence.py:299) ────────────────────────


class TestToolRegressionDedup:
    """A single coherent pattern across many later turns must emit at most
    one issue per later turn — not one per (i, j) pair.

    Before the dedup fix, 10 later turns dropping the same tool relative to
    turn 1 produced ~45 near-identical issues (O(n²)).  After the fix, it
    should emit exactly `(n - 1)` issues — one per later turn, referencing
    the earliest turn with the largest drop.
    """

    def test_ten_turns_dropping_tool_emits_one_issue_per_later_turn(self) -> None:
        turns = [_turn(i, f"q{i}") for i in range(1, 11)]  # 10 turns
        steps: List[StepTrace] = []
        # Turn 1 uses both search and fetch; turns 2..10 only use fetch.
        steps.append(_step("search", turn_index=1))
        for i in range(1, 11):
            steps.append(_step("fetch", turn_index=i))

        issues = _detect_tool_regression(turns, steps)

        # Exactly 9 issues (one per later turn), not 9*10/2 = 45.
        assert len(issues) == 9, f"expected 9 dedup'd issues, got {len(issues)}"
        for issue in issues:
            # All reference turn 1 — the earliest with the drop.
            assert issue.reference_turn == 1
            assert issue.category == CoherenceCategory.TOOL_REGRESSION
            assert "search" in issue.description

    def test_no_pattern_no_issues(self) -> None:
        """Sanity: turns that all use the same tool set should emit nothing."""
        turns = [_turn(i, f"q{i}") for i in range(1, 6)]
        steps = [_step("tool_a", turn_index=i) for i in range(1, 6)]
        issues = _detect_tool_regression(turns, steps)
        assert issues == []


# ── 2. Latency = 0 must be treated as unmeasured (benchmark_hardening) ──────


class TestLatencyZeroUnmeasured:
    """``_check_too_perfect`` with latency=0 must NOT mark the trace as fast.

    Before the fix, ``latency < MIN_EXPECTED_LATENCY_MS * 2`` fired for any
    zero or negative latency — producing a CRITICAL false positive on
    unmeasured runs.  The fix requires ``latency > 0`` for ``is_fast``.
    """

    def test_score_100_latency_0_one_step_is_not_critical(self) -> None:
        trace = _trace(steps=[_step("x")], latency=0)
        flags = _check_too_perfect(trace, score=100)

        # Exactly one flag, and it must be SUSPICIOUS (is_light=True,
        # is_fast=False because latency=0 is unmeasured).  CRITICAL requires
        # both is_fast AND is_light — which is the exact regression.
        assert len(flags) == 1
        assert flags[0].severity == FlagSeverity.SUSPICIOUS, (
            "latency=0 must be treated as unmeasured, not as fast"
        )

    def test_score_100_negative_latency_also_unmeasured(self) -> None:
        trace = _trace(steps=[_step("x")], latency=-1)
        flags = _check_too_perfect(trace, score=100)
        assert len(flags) == 1
        assert flags[0].severity != FlagSeverity.CRITICAL


# ── 3. Suspicious-extension boundary (benchmark_hardening) ──────────────────


class TestSuspiciousExtensionBoundary:
    """``.evalview/`` path must NOT match the ``.eval`` extension pattern.

    The regex was tightened to require a non-alphanumeric (or end) after
    the extension.  A substring fix would re-flag every ``.evalview``
    config path a CI job touches.
    """

    def test_evalview_directory_not_flagged(self) -> None:
        steps = [_step("read", params={"path": "/app/.evalview/config.yaml"})]
        flags = _check_abnormal_file_access(_trace(steps=steps))
        assert flags == []

    def test_eval_file_extension_still_flagged(self) -> None:
        """Sanity: the fix must not have over-rotated — .eval as a real
        extension must still trip the check."""
        steps = [_step("read", params={"path": "/tmp/answers.eval"})]
        flags = _check_abnormal_file_access(_trace(steps=steps))
        assert len(flags) == 1


# ── 4. MIN_EXPECTED_TOOL_CALLS now catches single-tool shortcuts ────────────


class TestMinExpectedToolCallsRaised:
    """``MIN_EXPECTED_TOOL_CALLS`` was raised from 1 to 2 so the check fires
    on "1 tool call + 95 score" — previously it only caught literally zero
    tool calls, which is trivially easy for a gaming agent to bypass."""

    def test_single_tool_high_score_flagged(self) -> None:
        trace = _trace(steps=[_step("x")])
        flags = _check_score_without_work(trace, score=95)
        assert len(flags) == 1
        # 1 tool + score >= 90 → SUSPICIOUS (not CRITICAL, since steps > 0)
        assert flags[0].severity == FlagSeverity.SUSPICIOUS

    def test_two_tools_high_score_ok(self) -> None:
        """Threshold is ``len(steps) < min_tools`` so 2 steps is fine."""
        trace = _trace(steps=[_step("a"), _step("b")])
        flags = _check_score_without_work(trace, score=95)
        assert flags == []


# ── 5. Reference_turn in coherence issues is a real earlier turn ────────────


class TestCoherenceReferenceTurn:
    """Issues in a coherence report must reference a real earlier turn
    index — not a hardcoded 0 or 1.  Before the hardening commit, a bug
    left reference_turn hardcoded."""

    def test_output_contradiction_reference_turn_is_earlier(self) -> None:
        turns = [
            _turn(1, "q1", output="The item is available now."),
            _turn(2, "q2", output="The item is not available at the moment."),
            _turn(3, "q3", output="The item is not available at the moment."),
        ]
        report = analyze_coherence(_trace(turns=turns))

        contradictions = [
            i for i in report.issues
            if i.category == CoherenceCategory.OUTPUT_CONTRADICTION
        ]
        assert contradictions, "expected at least one contradiction"
        for issue in contradictions:
            # reference_turn MUST be strictly earlier than the issue's turn.
            assert issue.reference_turn is not None
            assert issue.reference_turn < issue.turn_index, (
                f"reference_turn {issue.reference_turn} is not earlier than "
                f"turn {issue.turn_index} — probably hardcoded"
            )


# ── 6. ObservabilitySummary.to_payload() schema_version ─────────────────────


class TestObservabilityPayloadSchema:
    def test_schema_version_present_when_signals(self) -> None:
        summary = ObservabilitySummary(anomaly_count=1, anomaly_tests=["t1"])
        payload = summary.to_payload()
        assert payload["schema_version"] == OBSERVABILITY_SCHEMA_VERSION
        assert payload["anomalies"] == {"count": 1, "tests": ["t1"]}

    def test_empty_summary_returns_empty_dict(self) -> None:
        """No signals → empty dict, not {schema_version: ...} alone — callers
        rely on `if payload: ...` to skip serialization."""
        assert ObservabilitySummary().to_payload() == {}

    def test_verdict_payload_nests_under_observability(self) -> None:
        """Fixes the ambiguous-key bug where verdict-level and check-JSON both
        used ``behavioral_anomalies`` with different shapes."""
        summary = ObservabilitySummary(anomaly_count=1, anomaly_tests=["t1"])
        wrapped = summary.to_verdict_payload()
        assert set(wrapped.keys()) == {"observability"}
        assert wrapped["observability"]["anomalies"] == {"count": 1, "tests": ["t1"]}
        # And the check-JSON "behavioral_anomalies" (per-test list, emitted
        # separately by check_display) no longer collides with this key name.
        assert "behavioral_anomalies" not in wrapped
        assert "behavioral_anomalies" not in wrapped["observability"]


# ── 7. check_gaming_batch is wired into extract_observability_summary ───────


class _FakeLatency:
    def __init__(self, seconds: float) -> None:
        self.total_latency = seconds


class _FakeEvaluations:
    def __init__(self, latency_s: float) -> None:
        self.latency = _FakeLatency(latency_s)


class _FakeResult:
    """Duck-typed result object sufficient for extract_observability_summary."""

    def __init__(
        self,
        test_name: str,
        score: float,
        latency_s: float = 0.0,
        trust_score: Optional[float] = None,
    ) -> None:
        self.test_case = test_name
        self.score = score
        self.anomaly_report = None
        self.coherence_report = None
        self.evaluations = _FakeEvaluations(latency_s)
        self.trust_report = (
            {"flags": [], "trust_score": trust_score, "summary": ""}
            if trust_score is not None
            else None
        )


class TestBatchHardeningWired:
    """``check_gaming_batch`` used to be a dead public API — no production
    caller.  After the wire-up, it must run inside
    ``extract_observability_summary`` and surface batch-level flags."""

    def test_all_perfect_scores_trigger_batch_flag(self) -> None:
        results = [_FakeResult(f"t{i}", 100.0) for i in range(5)]
        summary = extract_observability_summary(results)
        assert summary.batch_hardening_flags, (
            "5 perfect-score results should trigger the batch TOO_PERFECT check"
        )
        assert any(
            f["check"] == GamingCheck.TOO_PERFECT.value
            for f in summary.batch_hardening_flags
        )
        payload = summary.to_payload()
        assert "batch_hardening" in payload

    def test_mixed_scores_no_batch_flag(self) -> None:
        results = [_FakeResult(f"t{i}", 50.0 + i * 10) for i in range(5)]
        summary = extract_observability_summary(results)
        assert summary.batch_hardening_flags == []


# ── 8. LOW_TRUST_THRESHOLD is the single source of truth ────────────────────


class TestLowTrustThresholdSingleSource:
    """Before the hardening commit, ``0.8`` was repeated in multiple files.
    Every consumer should import from ``observability.LOW_TRUST_THRESHOLD``.
    """

    def test_threshold_value(self) -> None:
        assert LOW_TRUST_THRESHOLD == 0.8

    def test_summary_uses_strict_less_than_threshold(self) -> None:
        """Trust score just under threshold triggers low-trust; exactly at
        threshold does not (strict `<` comparison)."""
        below = extract_observability_summary(
            [_FakeResult("t", 50.0, trust_score=LOW_TRUST_THRESHOLD - 0.01)]
        )
        at = extract_observability_summary(
            [_FakeResult("t", 50.0, trust_score=LOW_TRUST_THRESHOLD)]
        )

        assert below.low_trust_count == 1
        assert at.low_trust_count == 0
