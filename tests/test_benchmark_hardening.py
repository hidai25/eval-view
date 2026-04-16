"""Tests for benchmark hardening / anti-gaming checks."""

from datetime import datetime

import pytest

from evalview.core.benchmark_hardening import (
    GamingCheck,
    GamingFlag,
    FlagSeverity,
    HardeningReport,
    check_gaming,
    check_gaming_batch,
    _check_suspiciously_fast,
    _check_config_leakage,
    _check_score_without_work,
    _check_too_perfect,
    _check_abnormal_file_access,
)
from evalview.core.types import (
    ExecutionTrace,
    ExecutionMetrics,
    StepTrace,
    StepMetrics,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _step(
    tool: str,
    params: dict = None,
    output: str = "ok",
) -> StepTrace:
    return StepTrace(
        step_id=f"s-{tool}",
        step_name=tool,
        tool_name=tool,
        parameters=params or {},
        output=output,
        success=True,
        metrics=StepMetrics(latency=100, cost=0.01),
    )


_SENTINEL = object()


def _trace(
    steps=_SENTINEL,
    latency: float = 5000.0,
    cost: float = 0.1,
) -> ExecutionTrace:
    if steps is _SENTINEL:
        steps = [_step("search"), _step("analyze"), _step("respond")]
    return ExecutionTrace(
        session_id="test",
        start_time=datetime(2025, 1, 1),
        end_time=datetime(2025, 1, 1, 0, 1),
        steps=steps,
        final_output="done",
        metrics=ExecutionMetrics(total_cost=cost, total_latency=latency),
    )


# ---------------------------------------------------------------------------
# Suspiciously fast
# ---------------------------------------------------------------------------


class TestSuspiciouslyFast:
    def test_normal_latency_no_flag(self):
        trace = _trace(latency=5000)
        assert _check_suspiciously_fast(trace) == []

    def test_fast_latency_flagged(self):
        trace = _trace(latency=100)
        flags = _check_suspiciously_fast(trace)
        assert len(flags) == 1
        assert flags[0].check == GamingCheck.SUSPICIOUSLY_FAST

    def test_very_fast_is_suspicious(self):
        trace = _trace(latency=50)  # Under half the threshold
        flags = _check_suspiciously_fast(trace)
        assert flags[0].severity == FlagSeverity.SUSPICIOUS

    def test_zero_latency_skipped(self):
        """Latency=0 means not measured, should not flag."""
        trace = _trace(latency=0)
        assert _check_suspiciously_fast(trace) == []


# ---------------------------------------------------------------------------
# Config leakage
# ---------------------------------------------------------------------------


class TestConfigLeakage:
    def test_no_leakage_normal_trace(self):
        trace = _trace()
        assert _check_config_leakage(trace) == []

    def test_leakage_in_params(self):
        steps = [
            _step("read_file", {"path": "/app/.evalview/golden/test.json"}),
        ]
        trace = _trace(steps=steps)
        flags = _check_config_leakage(trace)
        assert len(flags) == 1
        assert flags[0].check == GamingCheck.CONFIG_LEAKAGE

    def test_leakage_ground_truth(self):
        steps = [
            _step("read_file", {"path": "/data/ground_truth.json"}),
        ]
        trace = _trace(steps=steps)
        flags = _check_config_leakage(trace)
        assert len(flags) == 1

    def test_leakage_in_output(self):
        steps = [
            _step("bash", {"cmd": "cat file.txt"}, output="evalview.yaml config found"),
        ]
        trace = _trace(steps=steps)
        flags = _check_config_leakage(trace)
        assert len(flags) == 1

    def test_multiple_leakage_is_critical(self):
        steps = [
            _step("read", {"path": ".evalview/golden/test.json"}),
            _step("read", {"path": "/data/answer_key.txt"}),
        ]
        trace = _trace(steps=steps)
        flags = _check_config_leakage(trace)
        assert len(flags) == 1
        assert flags[0].severity == FlagSeverity.CRITICAL


# ---------------------------------------------------------------------------
# Score without work
# ---------------------------------------------------------------------------


class TestScoreWithoutWork:
    def test_high_score_with_tools_ok(self):
        trace = _trace()  # 3 steps
        assert _check_score_without_work(trace, score=95) == []

    def test_high_score_no_tools_flagged(self):
        trace = _trace(steps=[])
        flags = _check_score_without_work(trace, score=95)
        assert len(flags) == 1
        assert flags[0].severity == FlagSeverity.CRITICAL

    def test_moderate_score_no_tools_less_severe(self):
        trace = _trace(steps=[])
        flags = _check_score_without_work(trace, score=82)
        assert len(flags) == 1
        assert flags[0].severity == FlagSeverity.INFO

    def test_low_score_no_tools_no_flag(self):
        trace = _trace(steps=[])
        assert _check_score_without_work(trace, score=50) == []


# ---------------------------------------------------------------------------
# Too perfect
# ---------------------------------------------------------------------------


class TestTooPerfect:
    def test_normal_score_no_flag(self):
        trace = _trace()
        assert _check_too_perfect(trace, score=85) == []

    def test_perfect_score_flagged(self):
        trace = _trace()
        flags = _check_too_perfect(trace, score=100)
        assert len(flags) == 1
        assert flags[0].check == GamingCheck.TOO_PERFECT

    def test_perfect_fast_light_is_critical(self):
        trace = _trace(steps=[_step("x")], latency=100)
        flags = _check_too_perfect(trace, score=100)
        assert len(flags) == 1
        assert flags[0].severity == FlagSeverity.CRITICAL


# ---------------------------------------------------------------------------
# Abnormal file access
# ---------------------------------------------------------------------------


class TestAbnormalFileAccess:
    def test_normal_params_no_flag(self):
        trace = _trace()
        assert _check_abnormal_file_access(trace) == []

    def test_suspicious_extension(self):
        steps = [_step("read", {"path": "answers.golden"})]
        trace = _trace(steps=steps)
        flags = _check_abnormal_file_access(trace)
        assert len(flags) == 1


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------


class TestCheckGaming:
    def test_clean_result(self):
        trace = _trace()
        report = check_gaming(trace, score=80)
        assert not report.has_flags
        assert report.trust_score == 1.0

    def test_multiple_flags_reduce_trust(self):
        trace = _trace(steps=[], latency=50)
        report = check_gaming(trace, score=100)
        assert report.has_flags
        assert report.trust_score < 1.0

    def test_report_summary(self):
        trace = _trace()
        report = check_gaming(trace, score=80)
        assert "No gaming signals" in report.summary()

    def test_report_to_dict(self):
        trace = _trace()
        report = check_gaming(trace, score=80)
        d = report.to_dict()
        assert "trust_score" in d
        assert "flags" in d


# ---------------------------------------------------------------------------
# Batch checks
# ---------------------------------------------------------------------------


class TestBatchChecks:
    def test_empty_batch(self):
        report = check_gaming_batch([])
        assert not report.has_flags

    def test_all_perfect_flagged(self):
        results = [{"score": 100, "trace": _trace()} for _ in range(5)]
        report = check_gaming_batch(results)
        assert report.has_flags
        assert any(f.severity == FlagSeverity.CRITICAL for f in report.flags)

    def test_mixed_scores_ok(self):
        results = [
            {"score": 100},
            {"score": 85},
            {"score": 72},
        ]
        report = check_gaming_batch(results)
        assert not report.has_flags
