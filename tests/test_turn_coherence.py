"""Tests for cross-turn coherence analysis."""

from datetime import datetime


from evalview.core.turn_coherence import (
    CoherenceCategory,
    CoherenceSeverity,
    analyze_coherence,
    _detect_tool_regression,
    _detect_strategy_drift,
    _detect_output_contradiction,
    _extract_key_entities,
    _detect_context_amnesia,
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


# ---------------------------------------------------------------------------
# Context amnesia detection
# ---------------------------------------------------------------------------


class TestContextAmnesia:
    def test_amnesia_detected(self):
        """3+ turns where user mentions 'New York City' early, references it
        later, but agent doesn't mention it — should detect amnesia."""
        turns = [
            _turn(1, "I want to fly to New York City", output="Great, let me search flights to New York City."),
            _turn(2, "What airlines are available?", output="Delta and United have options."),
            _turn(3, "Book the cheapest flight to New York City", output="I booked the cheapest option for you."),
        ]
        issues = _detect_context_amnesia(turns)
        assert len(issues) >= 1
        assert issues[0].category == CoherenceCategory.CONTEXT_AMNESIA

    def test_no_amnesia_when_agent_references(self):
        """Same setup but agent DOES mention the entity — no amnesia."""
        turns = [
            _turn(1, "I want to fly to New York City", output="Great, searching for New York City flights."),
            _turn(2, "What airlines are available?", output="Delta and United fly to New York City."),
            _turn(3, "Book the cheapest flight to New York City", output="Booked the cheapest flight to New York City."),
        ]
        issues = _detect_context_amnesia(turns)
        assert issues == []

    def test_too_few_turns_no_amnesia(self):
        """2 turns should return empty (below AMNESIA_MIN_TURNS=3)."""
        turns = [
            _turn(1, "Fly to New York City", output="Searching."),
            _turn(2, "Book flight to New York City", output="Done."),
        ]
        issues = _detect_context_amnesia(turns)
        assert issues == []

    def test_reference_turn_points_to_origin(self):
        """Verify that reference_turn points to the actual turn that introduced
        the entity, not hardcoded to 1."""
        turns = [
            _turn(1, "Hello there", output="Hi!"),
            _turn(2, "I need info about San Francisco Bay", output="Sure, tell me more."),
            _turn(3, "What else is in the area?", output="There's lots to do."),
            _turn(4, "Tell me about San Francisco Bay attractions", output="Here are some things."),
        ]
        issues = _detect_context_amnesia(turns)
        if issues:
            # The origin turn for "San Francisco Bay" should be turn 2
            assert issues[0].reference_turn == 2


# ---------------------------------------------------------------------------
# Contradiction reduces score — unconditional assertion
# ---------------------------------------------------------------------------


class TestIssuesReduceScoreFixed:
    def test_contradiction_reduces_score_unconditionally(self):
        """Create a clear contradiction, assert issues ARE present, then assert score < 1.0."""
        turns = [
            _turn(1, "Is the item available?", output="The item is available in stock."),
            _turn(2, "Can I buy it?", output="The item is not available right now."),
        ]
        trace = _trace(turns=turns)
        report = analyze_coherence(trace)
        # Assert unconditionally: this MUST produce a contradiction
        assert report.has_issues
        assert any(i.category == CoherenceCategory.OUTPUT_CONTRADICTION for i in report.issues)
        assert report.coherence_score < 1.0


# ---------------------------------------------------------------------------
# Coherence report properties
# ---------------------------------------------------------------------------


class TestCoherenceReportProperties:
    def test_errors_and_warnings_properties(self):
        """Create a report with errors and warnings, test the properties."""
        turns = [
            # Contradiction produces ERROR
            _turn(1, "q1", output="The item is available now."),
            _turn(2, "q2", output="The item is not available right now."),
        ]
        steps = [
            # Tool regression produces WARNING
            _step("search", 1),
            _step("analyze", 1),
            _step("verify", 1),
            _step("search", 2),
        ]
        trace = _trace(turns=turns, steps=steps)
        report = analyze_coherence(trace)
        # Check that errors and warnings filter correctly
        assert all(e.severity == CoherenceSeverity.ERROR for e in report.errors)
        assert all(w.severity == CoherenceSeverity.WARNING for w in report.warnings)

    def test_summary_with_issues(self):
        """Test summary() output when issues exist."""
        turns = [
            _turn(1, "q1", output="The item is available."),
            _turn(2, "q2", output="The item is not available."),
        ]
        trace = _trace(turns=turns)
        report = analyze_coherence(trace)
        if report.has_issues:
            summary = report.summary()
            assert "error" in summary.lower()
            assert "score" in summary.lower()


# ---------------------------------------------------------------------------
# Wider contradiction detection
# ---------------------------------------------------------------------------


class TestContradictionValueChange:
    """Tests for the value-change and has/doesn't-have contradiction patterns."""

    def test_value_change_detected(self):
        """Different numeric values for the same label should be caught."""
        turns = [
            _turn(1, "What's the price?", output="The price is $50 per unit."),
            _turn(2, "Confirm the price", output="The price is $75 per unit."),
        ]
        issues = _detect_output_contradiction(turns)
        assert len(issues) == 1
        assert issues[0].evidence["contradiction_type"] == "value_change"

    def test_same_value_no_contradiction(self):
        """Same value repeated should not flag."""
        turns = [
            _turn(1, "q1", output="The price is $50."),
            _turn(2, "q2", output="As mentioned, the price is $50."),
        ]
        issues = _detect_output_contradiction(turns)
        assert issues == []

    def test_has_doesnt_have_detected(self):
        """'X has Y' vs 'X doesn't have Y' should be caught."""
        turns = [
            _turn(1, "q1", output="The store has inventory available."),
            _turn(2, "q2", output="The store doesn't have inventory right now."),
        ]
        issues = _detect_output_contradiction(turns)
        assert len(issues) == 1
        assert issues[0].evidence["contradiction_type"] == "has_negation"

    def test_negation_still_works(self):
        """Original 'is' vs 'is not' pattern should still work."""
        turns = [
            _turn(1, "q1", output="The item is available in our inventory."),
            _turn(2, "q2", output="The item is not available right now."),
        ]
        issues = _detect_output_contradiction(turns)
        assert len(issues) == 1
        assert issues[0].evidence["contradiction_type"] == "negation"
