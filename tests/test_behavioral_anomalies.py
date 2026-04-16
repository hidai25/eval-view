"""Tests for behavioral anomaly detection."""

from datetime import datetime

import pytest

from evalview.core.behavioral_anomalies import (
    Anomaly,
    AnomalyPattern,
    AnomalyReport,
    AnomalySeverity,
    detect_anomalies,
    _detect_tool_loops,
    _detect_progress_stalls,
    _detect_brittle_recovery,
    _detect_excessive_retries,
    _detect_skipped_steps,
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
    success: bool = True,
    error: str = None,
    step_id: str = None,
) -> StepTrace:
    """Convenience builder for test steps."""
    return StepTrace(
        step_id=step_id or f"s-{id(tool)}",
        step_name=tool,
        tool_name=tool,
        parameters=params or {},
        output="ok",
        success=success,
        error=error,
        metrics=StepMetrics(latency=100, cost=0.01),
    )


def _trace(steps: list[StepTrace]) -> ExecutionTrace:
    """Convenience builder for traces."""
    return ExecutionTrace(
        session_id="test",
        start_time=datetime(2025, 1, 1),
        end_time=datetime(2025, 1, 1, 0, 1),
        steps=steps,
        final_output="done",
        metrics=ExecutionMetrics(total_cost=0.1, total_latency=5000),
    )


# ---------------------------------------------------------------------------
# Tool loop detection
# ---------------------------------------------------------------------------


class TestToolLoops:
    def test_no_loop_in_varied_steps(self):
        steps = [_step("a"), _step("b"), _step("c")]
        assert _detect_tool_loops(steps) == []

    def test_loop_detected(self):
        steps = [
            _step("search", {"q": "x"}),
            _step("search", {"q": "x"}),
            _step("search", {"q": "x"}),
        ]
        anomalies = _detect_tool_loops(steps)
        assert len(anomalies) == 1
        assert anomalies[0].pattern == AnomalyPattern.TOOL_LOOP
        assert anomalies[0].severity == AnomalySeverity.ERROR
        assert "3 times" in anomalies[0].description

    def test_no_loop_different_params(self):
        steps = [
            _step("search", {"q": "a"}),
            _step("search", {"q": "b"}),
            _step("search", {"q": "c"}),
        ]
        assert _detect_tool_loops(steps) == []

    def test_loop_at_end_of_trace(self):
        steps = [
            _step("init"),
            _step("search", {"q": "x"}),
            _step("search", {"q": "x"}),
            _step("search", {"q": "x"}),
            _step("search", {"q": "x"}),
        ]
        anomalies = _detect_tool_loops(steps)
        assert len(anomalies) == 1
        assert anomalies[0].evidence["consecutive_count"] == 4

    def test_two_separate_loops(self):
        steps = [
            _step("a", {"x": 1}),
            _step("a", {"x": 1}),
            _step("a", {"x": 1}),
            _step("b"),
            _step("c", {"y": 2}),
            _step("c", {"y": 2}),
            _step("c", {"y": 2}),
        ]
        anomalies = _detect_tool_loops(steps)
        assert len(anomalies) == 2


# ---------------------------------------------------------------------------
# Progress stall detection
# ---------------------------------------------------------------------------


class TestProgressStalls:
    def test_no_stall_with_new_tools(self):
        steps = [_step(f"tool_{i}") for i in range(10)]
        assert _detect_progress_stalls(steps) == []

    def test_stall_detected(self):
        steps = [
            _step("a"),
            _step("b"),  # Last new tool
            _step("a"),
            _step("b"),
            _step("a"),
            _step("b"),
            _step("a"),
        ]
        anomalies = _detect_progress_stalls(steps)
        assert len(anomalies) == 1
        assert anomalies[0].pattern == AnomalyPattern.PROGRESS_STALL

    def test_no_stall_short_trace(self):
        steps = [_step("a"), _step("a"), _step("a")]
        assert _detect_progress_stalls(steps) == []


# ---------------------------------------------------------------------------
# Brittle recovery detection
# ---------------------------------------------------------------------------


class TestBrittleRecovery:
    def test_no_brittle_recovery_on_success(self):
        steps = [_step("a"), _step("b")]
        assert _detect_brittle_recovery(steps) == []

    def test_brittle_recovery_detected(self):
        steps = [
            _step("search", {"q": "x"}, success=False, error="timeout"),
            _step("search", {"q": "x"}),
            _step("search", {"q": "x"}),
        ]
        anomalies = _detect_brittle_recovery(steps)
        assert len(anomalies) == 1
        assert anomalies[0].pattern == AnomalyPattern.BRITTLE_RECOVERY
        assert "timeout" in anomalies[0].description

    def test_no_brittle_if_params_change(self):
        steps = [
            _step("search", {"q": "x"}, success=False, error="not found"),
            _step("search", {"q": "y"}),  # Different params = adapted
            _step("search", {"q": "z"}),
        ]
        assert _detect_brittle_recovery(steps) == []

    def test_single_retry_not_flagged(self):
        """One retry after failure is normal, not brittle."""
        steps = [
            _step("search", {"q": "x"}, success=False, error="timeout"),
            _step("search", {"q": "x"}),
        ]
        assert _detect_brittle_recovery(steps) == []


# ---------------------------------------------------------------------------
# Excessive retry detection
# ---------------------------------------------------------------------------


class TestExcessiveRetries:
    def test_no_excess_few_calls(self):
        steps = [_step("a"), _step("b"), _step("a")]
        assert _detect_excessive_retries(steps) == []

    def test_excessive_retries_detected(self):
        steps = [_step("search") for _ in range(6)]
        anomalies = _detect_excessive_retries(steps)
        assert len(anomalies) == 1
        assert anomalies[0].pattern == AnomalyPattern.EXCESSIVE_RETRIES
        assert anomalies[0].evidence["total_calls"] == 6

    def test_severity_escalates_with_failures(self):
        steps = [
            _step("api_call", success=False, error="500") for _ in range(5)
        ]
        anomalies = _detect_excessive_retries(steps)
        assert len(anomalies) == 1
        assert anomalies[0].severity == AnomalySeverity.ERROR


# ---------------------------------------------------------------------------
# Skipped steps detection
# ---------------------------------------------------------------------------


class TestSkippedSteps:
    def test_no_skip_when_all_present(self):
        steps = [_step("search"), _step("book")]
        assert _detect_skipped_steps(steps, ["search", "book"]) == []

    def test_skip_detected(self):
        steps = [_step("search")]
        anomalies = _detect_skipped_steps(steps, ["search", "book", "confirm"])
        assert len(anomalies) == 1
        assert anomalies[0].pattern == AnomalyPattern.SKIPPED_STEPS
        assert "book" in anomalies[0].description
        assert "confirm" in anomalies[0].description

    def test_no_skip_without_required_tools(self):
        steps = [_step("search")]
        assert _detect_skipped_steps(steps, None) == []
        assert _detect_skipped_steps(steps, []) == []


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------


class TestDetectAnomalies:
    def test_clean_trace(self):
        trace = _trace([_step("a"), _step("b"), _step("c")])
        report = detect_anomalies(trace)
        assert not report.has_anomalies
        assert report.total_steps == 3
        assert report.unique_tools == 3

    def test_multiple_anomalies(self):
        steps = [
            _step("search", {"q": "x"}, success=False, error="timeout"),
            _step("search", {"q": "x"}),
            _step("search", {"q": "x"}),
            _step("search", {"q": "x"}),
            _step("search", {"q": "x"}),
            _step("search", {"q": "x"}),
        ]
        trace = _trace(steps)
        report = detect_anomalies(trace)
        assert report.has_anomalies
        patterns = {a.pattern for a in report.anomalies}
        assert AnomalyPattern.EXCESSIVE_RETRIES in patterns

    def test_report_summary(self):
        trace = _trace([_step("a")])
        report = detect_anomalies(trace)
        assert "No behavioral anomalies" in report.summary()

    def test_report_to_dict(self):
        trace = _trace([_step("a")])
        report = detect_anomalies(trace)
        d = report.to_dict()
        assert "anomalies" in d
        assert "summary" in d
        assert d["total_steps"] == 1
