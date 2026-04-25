"""Tests for behavioral anomaly detection."""

from datetime import datetime


from evalview.core.behavioral_anomalies import (
    AnomalyPattern,
    AnomalySeverity,
    detect_anomalies,
    _detect_tool_loops,
    _detect_progress_stalls,
    _detect_brittle_recovery,
    _detect_excessive_retries,
    _detect_skipped_steps,
    _step_fingerprint,
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


# ---------------------------------------------------------------------------
# Step fingerprinting
# ---------------------------------------------------------------------------


class TestStepFingerprint:
    def test_same_tool_same_params_same_fingerprint(self):
        s1 = _step("search", {"q": "hello"})
        s2 = _step("search", {"q": "hello"})
        assert _step_fingerprint(s1) == _step_fingerprint(s2)

    def test_different_params_different_fingerprint(self):
        s1 = _step("search", {"q": "hello"})
        s2 = _step("search", {"q": "world"})
        assert _step_fingerprint(s1) != _step_fingerprint(s2)

    def test_nested_dict_params_stable(self):
        s1 = _step("api", {"filters": {"a": 1, "b": 2}})
        s2 = _step("api", {"filters": {"b": 2, "a": 1}})
        assert _step_fingerprint(s1) == _step_fingerprint(s2)

    def test_large_params_truncated(self):
        long_val = "x" * 500
        s = _step("tool", {"data": long_val})
        fp = _step_fingerprint(s)
        # The fingerprint should contain truncation marker
        assert "..." in fp


# ---------------------------------------------------------------------------
# Tool loop boundary
# ---------------------------------------------------------------------------


class TestToolLoopBoundary:
    def test_two_identical_not_flagged(self):
        """Exactly 2 identical calls is below LOOP_MIN_CONSECUTIVE (3)."""
        steps = [
            _step("search", {"q": "x"}),
            _step("search", {"q": "x"}),
        ]
        assert _detect_tool_loops(steps) == []


# ---------------------------------------------------------------------------
# Progress stall boundary
# ---------------------------------------------------------------------------


class TestProgressStallBoundary:
    def test_exactly_at_threshold(self):
        """Exactly 5 non-new-tool calls (STALL_WINDOW) should flag."""
        # Introduce 2 tools, then repeat only those for 5 more calls
        steps = [
            _step("a"),
            _step("b"),
            _step("a"),  # stall starts here
            _step("b"),
            _step("a"),
            _step("b"),
            _step("a"),
        ]
        anomalies = _detect_progress_stalls(steps)
        assert len(anomalies) == 1
        assert anomalies[0].pattern == AnomalyPattern.PROGRESS_STALL

    def test_below_threshold(self):
        """4 non-new-tool calls should NOT flag."""
        steps = [
            _step("a"),
            _step("b"),
            _step("a"),
            _step("b"),
            _step("a"),
            _step("b"),
        ]
        anomalies = _detect_progress_stalls(steps)
        assert anomalies == []


# ---------------------------------------------------------------------------
# Brittle recovery — success vs failure retries
# ---------------------------------------------------------------------------


class TestBrittleRecoverySuccessful:
    def test_successful_retries_are_warning(self):
        """Failed step + 2 identical retries that SUCCEED should be WARNING."""
        steps = [
            _step("search", {"q": "x"}, success=False, error="timeout"),
            _step("search", {"q": "x"}, success=True),
            _step("search", {"q": "x"}, success=True),
        ]
        anomalies = _detect_brittle_recovery(steps)
        assert len(anomalies) == 1
        assert anomalies[0].severity == AnomalySeverity.WARNING

    def test_all_failed_retries_are_error(self):
        """Failed step + 2 identical failed retries should be ERROR."""
        steps = [
            _step("search", {"q": "x"}, success=False, error="timeout"),
            _step("search", {"q": "x"}, success=False, error="timeout"),
            _step("search", {"q": "x"}, success=False, error="timeout"),
        ]
        anomalies = _detect_brittle_recovery(steps)
        assert len(anomalies) == 1
        assert anomalies[0].severity == AnomalySeverity.ERROR


# ---------------------------------------------------------------------------
# Empty / minimal inputs
# ---------------------------------------------------------------------------


class TestEmptyInputs:
    def test_empty_steps_all_detectors(self):
        """Empty list should return no anomalies for all detectors."""
        assert _detect_tool_loops([]) == []
        assert _detect_progress_stalls([]) == []
        assert _detect_brittle_recovery([]) == []
        assert _detect_excessive_retries([]) == []
        assert _detect_skipped_steps([], None) == []

    def test_single_step(self):
        """Single step should return no anomalies."""
        steps = [_step("a")]
        assert _detect_tool_loops(steps) == []
        assert _detect_progress_stalls(steps) == []
        assert _detect_brittle_recovery(steps) == []
        assert _detect_excessive_retries(steps) == []
        assert _detect_skipped_steps(steps, None) == []


# ---------------------------------------------------------------------------
# Report with anomalies
# ---------------------------------------------------------------------------


class TestReportWithAnomalies:
    def test_summary_with_anomalies(self):
        """summary() should show error/warning counts when anomalies present."""
        steps = [
            _step("search", {"q": "x"}),
            _step("search", {"q": "x"}),
            _step("search", {"q": "x"}),
        ]
        trace = _trace(steps)
        report = detect_anomalies(trace)
        assert report.has_anomalies
        summary = report.summary()
        assert "error" in summary.lower()
        assert "tool_loop" in summary

    def test_error_and_warning_properties(self):
        """error_anomalies and warning_anomalies should filter correctly."""
        steps = [
            _step("a"),
            _step("b"),
            _step("a"),
            _step("b"),
            _step("a"),
            _step("b"),
            _step("a"),
        ]
        trace = _trace(steps)
        report = detect_anomalies(trace)
        # Progress stalls produce WARNING severity
        assert len(report.warning_anomalies) >= 1
        assert all(
            a.severity == AnomalySeverity.WARNING
            for a in report.warning_anomalies
        )
        assert all(
            a.severity == AnomalySeverity.ERROR
            for a in report.error_anomalies
        )
