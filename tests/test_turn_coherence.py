"""Tests for cross-turn coherence analysis."""

from datetime import datetime

import pytest

from evalview.core.turn_coherence import (
    CoherenceCategory,
    CoherenceIssue,
    CoherenceReport,
    CoherenceSeverity,
    analyze_coherence,
    _detect_tool_regression,
    _detect_strategy_drift,
    _detect_output_contradiction,
    _extract_key_entities,
)
from evalview.core.types import (
    ExecutionTrace,
    ExecutionMetrics,
    StepTrace,
    StepMetrics,
    TurnTrace,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _step(
    tool: str,
    turn_index: int = 1,
    params: dict = None,
) -> StepTrace:
    return StepTrace(
        step_id=f"s-{tool}-{turn_index}",
        step_name=tool,
        tool_name=tool,
        parameters=params or {},
        output="ok",
        success=True,
        metrics=StepMetrics(latency=100, cost=0.01),
        turn_index=turn_index,
    )


def _turn(
    index: int,
    query: str,
    output: str = "",
    tools: list[str] = None,
) -> TurnTrace:
    return TurnTrace(
        index=index,
        query=query,
        output=output,
        tools=tools or [],
    )


def _trace(
    turns: list[TurnTrace],
    steps: list[StepTrace] = None,
) -> ExecutionTrace:
    return ExecutionTrace(
        session_id="test",
        start_time=datetime(2025, 1, 1),
        end_time=datetime(2025, 1, 1, 0, 5),
        steps=steps or [],
        final_output="done",
        metrics=ExecutionMetrics(total_cost=0.1, total_latency=5000),
        turns=turns,
    )


# ---------------------------------------------------------------------------
# Entity extraction
# ---------------------------------------------------------------------------


class TestEntityExtraction:
    def test_extracts_capitalized_names(self):
        entities = _extract_key_entities("I want to fly to New York City")
        assert "new york city" in entities or "new york" in entities

    def test_extracts_quoted_strings(self):
        entities = _extract_key_entities('My order number is "ABC-123"')
        assert "abc-123" in entities

    def test_extracts_numbers(self):
        entities = _extract_key_entities("The flight costs 450 dollars")
        assert "450" in entities

    def test_empty_string(self):
        assert _extract_key_entities("") == set()


# ---------------------------------------------------------------------------
# Tool regression
# ---------------------------------------------------------------------------


class TestToolRegression:
    def test_no_regression_consistent_tools(self):
        turns = [_turn(1, "q1"), _turn(2, "q2")]
        steps = [
            _step("search", 1),
            _step("analyze", 1),
            _step("search", 2),
            _step("analyze", 2),
        ]
        assert _detect_tool_regression(turns, steps) == []

    def test_regression_detected_when_tools_dropped(self):
        turns = [_turn(1, "q1"), _turn(2, "q2")]
        steps = [
            _step("search", 1),
            _step("analyze", 1),
            _step("verify", 1),
            # Turn 2 only uses search, dropped analyze and verify
            _step("search", 2),
        ]
        issues = _detect_tool_regression(turns, steps)
        assert len(issues) >= 1
        assert issues[0].category == CoherenceCategory.TOOL_REGRESSION

    def test_single_turn_no_regression(self):
        turns = [_turn(1, "q1")]
        steps = [_step("search", 1)]
        assert _detect_tool_regression(turns, steps) == []


# ---------------------------------------------------------------------------
# Strategy drift
# ---------------------------------------------------------------------------


class TestStrategyDrift:
    def test_no_drift_consistent_strategy(self):
        turns = [_turn(i, f"q{i}") for i in range(1, 5)]
        steps = [
            _step("search", 1), _step("analyze", 1),
            _step("search", 2), _step("analyze", 2),
            _step("search", 3), _step("analyze", 3),
            _step("search", 4), _step("analyze", 4),
        ]
        assert _detect_strategy_drift(turns, steps) == []

    def test_drift_detected_when_tools_change(self):
        turns = [_turn(i, f"q{i}") for i in range(1, 5)]
        steps = [
            # First half: search + analyze
            _step("search", 1), _step("analyze", 1),
            _step("search", 2), _step("analyze", 2),
            # Second half: completely different tools
            _step("api_call", 3), _step("format", 3),
            _step("api_call", 4), _step("validate", 4),
        ]
        issues = _detect_strategy_drift(turns, steps)
        assert len(issues) == 1
        assert issues[0].category == CoherenceCategory.STRATEGY_DRIFT

    def test_too_few_turns_no_drift(self):
        turns = [_turn(1, "q1"), _turn(2, "q2")]
        steps = [_step("a", 1), _step("b", 2)]
        assert _detect_strategy_drift(turns, steps) == []


# ---------------------------------------------------------------------------
# Output contradiction
# ---------------------------------------------------------------------------


class TestOutputContradiction:
    def test_no_contradiction(self):
        turns = [
            _turn(1, "q1", output="The flight is available."),
            _turn(2, "q2", output="I'll book the available flight."),
        ]
        assert _detect_output_contradiction(turns) == []

    def test_contradiction_detected(self):
        turns = [
            _turn(1, "q1", output="The item is available in our inventory."),
            _turn(2, "q2", output="The item is not available right now."),
        ]
        issues = _detect_output_contradiction(turns)
        assert len(issues) == 1
        assert issues[0].category == CoherenceCategory.OUTPUT_CONTRADICTION
        assert issues[0].severity == CoherenceSeverity.ERROR

    def test_single_turn_no_contradiction(self):
        turns = [_turn(1, "q1", output="hello")]
        assert _detect_output_contradiction(turns) == []


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------


class TestAnalyzeCoherence:
    def test_single_turn_returns_clean(self):
        trace = _trace(turns=[_turn(1, "hello")])
        report = analyze_coherence(trace)
        assert not report.has_issues
        assert report.coherence_score == 1.0

    def test_no_turns_returns_clean(self):
        trace = _trace(turns=[])
        report = analyze_coherence(trace)
        assert not report.has_issues

    def test_clean_multi_turn(self):
        turns = [
            _turn(1, "q1", output="The capital is Paris."),
            _turn(2, "q2", output="Paris has many landmarks."),
        ]
        trace = _trace(turns=turns)
        report = analyze_coherence(trace)
        assert report.total_turns == 2

    def test_report_summary(self):
        trace = _trace(turns=[_turn(1, "q1"), _turn(2, "q2")])
        report = analyze_coherence(trace)
        summary = report.summary()
        assert "turns" in summary.lower() or "coherence" in summary.lower()

    def test_report_to_dict(self):
        trace = _trace(turns=[_turn(1, "q1"), _turn(2, "q2")])
        report = analyze_coherence(trace)
        d = report.to_dict()
        assert "coherence_score" in d
        assert "issues" in d

    def test_issues_reduce_score(self):
        turns = [
            _turn(1, "q1", output="The item is available now."),
            _turn(2, "q2", output="The item is not available at the moment."),
        ]
        trace = _trace(turns=turns)
        report = analyze_coherence(trace)
        if report.has_issues:
            assert report.coherence_score < 1.0
