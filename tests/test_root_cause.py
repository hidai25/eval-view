"""Unit tests for root cause attribution."""

from typing import List, Optional
from unittest.mock import AsyncMock, patch

import pytest

from evalview.core.diff import (
    DiffStatus,
    OutputDiff,
    ParameterDiff,
    ToolDiff,
    TraceDiff,
)
from evalview.core.root_cause import (
    Confidence,
    RootCauseCategory,
    analyze_root_cause,
    enrich_with_ai,
    enrich_diffs_with_ai,
    enrich_with_narrative,
    enrich_diffs_with_narrative,
    _build_ai_user_prompt,
    _build_narrative_prompt,
    _format_params_brief,
    _strip_markdown_fences,
)


def _make_trace_diff(
    tool_diffs: Optional[List[ToolDiff]] = None,
    output_similarity: float = 1.0,
    score_diff: float = -10.0,
    overall_severity: DiffStatus = DiffStatus.REGRESSION,
    model_changed: bool = False,
    golden_model_id: Optional[str] = None,
    actual_model_id: Optional[str] = None,
) -> TraceDiff:
    """Helper to build a TraceDiff for testing."""
    return TraceDiff(
        test_name="test-case",
        has_differences=True,
        tool_diffs=tool_diffs or [],
        output_diff=OutputDiff(
            similarity=output_similarity,
            golden_preview="golden output",
            actual_preview="actual output",
            diff_lines=[],
            severity=DiffStatus.PASSED if output_similarity >= 0.95 else DiffStatus.OUTPUT_CHANGED,
        ),
        score_diff=score_diff,
        latency_diff=0.0,
        overall_severity=overall_severity,
        model_changed=model_changed,
        golden_model_id=golden_model_id,
        actual_model_id=actual_model_id,
    )


class TestRootCauseCategories:
    """Test each root cause category is correctly identified."""

    def test_passed_returns_none(self):
        """PASSED diffs have no root cause to analyze."""
        diff = _make_trace_diff(overall_severity=DiffStatus.PASSED, score_diff=0.0)
        assert analyze_root_cause(diff) is None

    def test_tool_removed(self):
        """Detect when a baseline tool was skipped."""
        diff = _make_trace_diff(
            tool_diffs=[
                ToolDiff(
                    type="removed",
                    position=1,
                    golden_tool="search_api",
                    actual_tool=None,
                    severity=DiffStatus.TOOLS_CHANGED,
                    message="Tool removed: 'search_api' was at step 2",
                ),
            ],
        )
        result = analyze_root_cause(diff)
        assert result is not None
        assert result.category == RootCauseCategory.TOOL_REMOVED
        assert result.root_tool == "search_api"
        assert result.confidence == Confidence.HIGH
        assert "search_api" in result.summary
        assert "search_api" in result.suggested_fix

    def test_tool_added(self):
        """Detect when a new unexpected tool was called."""
        diff = _make_trace_diff(
            tool_diffs=[
                ToolDiff(
                    type="added",
                    position=2,
                    golden_tool=None,
                    actual_tool="debug_logger",
                    severity=DiffStatus.TOOLS_CHANGED,
                    message="Tool added: 'debug_logger' at step 3",
                ),
            ],
        )
        result = analyze_root_cause(diff)
        assert result is not None
        assert result.category == RootCauseCategory.TOOL_ADDED
        assert result.root_tool == "debug_logger"
        assert result.confidence == Confidence.HIGH

    def test_tool_reordered(self):
        """Detect when tools are reordered (changed with different names)."""
        diff = _make_trace_diff(
            tool_diffs=[
                ToolDiff(
                    type="changed",
                    position=0,
                    golden_tool="fetch_data",
                    actual_tool="validate_input",
                    severity=DiffStatus.TOOLS_CHANGED,
                    message="Tool changed: 'fetch_data' -> 'validate_input' at step 1",
                ),
            ],
        )
        result = analyze_root_cause(diff)
        assert result is not None
        assert result.category == RootCauseCategory.TOOL_REORDERED
        assert result.confidence == Confidence.MEDIUM

    def test_parameter_changed(self):
        """Detect when a tool's parameters changed."""
        diff = _make_trace_diff(
            tool_diffs=[
                ToolDiff(
                    type="changed",
                    position=0,
                    golden_tool="search_api",
                    actual_tool="search_api",
                    severity=DiffStatus.TOOLS_CHANGED,
                    message="Tool 'search_api' parameters changed at step 1",
                    parameter_diffs=[
                        ParameterDiff(
                            param_name="limit",
                            golden_value=50,
                            actual_value=5,
                            diff_type="value_changed",
                        ),
                    ],
                ),
            ],
        )
        result = analyze_root_cause(diff)
        assert result is not None
        assert result.category == RootCauseCategory.PARAMETER_CHANGED
        assert result.root_tool == "search_api"
        assert result.confidence == Confidence.HIGH
        assert len(result.parameter_diffs) == 1
        assert result.parameter_diffs[0].param_name == "limit"
        assert "limit" in result.summary
        assert "50" in result.summary or "5" in result.summary

    def test_parameter_missing(self):
        """Detect when a parameter was dropped."""
        diff = _make_trace_diff(
            tool_diffs=[
                ToolDiff(
                    type="changed",
                    position=0,
                    golden_tool="search_api",
                    actual_tool="search_api",
                    severity=DiffStatus.TOOLS_CHANGED,
                    message="Tool 'search_api' parameters changed at step 1",
                    parameter_diffs=[
                        ParameterDiff(
                            param_name="max_results",
                            golden_value=10,
                            actual_value=None,
                            diff_type="missing",
                        ),
                    ],
                ),
            ],
        )
        result = analyze_root_cause(diff)
        assert result is not None
        assert result.category == RootCauseCategory.PARAMETER_CHANGED
        assert "max_results" in result.summary
        assert "missing" in result.summary.lower() or "missing" in result.suggested_fix.lower()

    def test_output_drifted(self):
        """Detect output drift when tools and params match but output changed."""
        diff = _make_trace_diff(
            tool_diffs=[],
            output_similarity=0.65,
        )
        result = analyze_root_cause(diff)
        assert result is not None
        assert result.category == RootCauseCategory.OUTPUT_DRIFTED
        assert result.confidence == Confidence.LOW
        assert "65%" in result.summary

    def test_output_drifted_with_model_change(self):
        """Output drift with model change gets medium confidence."""
        diff = _make_trace_diff(
            tool_diffs=[],
            output_similarity=0.70,
            model_changed=True,
            golden_model_id="gpt-4o-2024-01",
            actual_model_id="gpt-4o-2024-03",
        )
        result = analyze_root_cause(diff)
        assert result is not None
        assert result.category == RootCauseCategory.OUTPUT_DRIFTED
        assert result.confidence == Confidence.MEDIUM
        assert "gpt-4o-2024-01" in result.summary
        assert "gpt-4o-2024-03" in result.summary

    def test_score_only(self):
        """Detect score-only regression when output is similar."""
        diff = _make_trace_diff(
            tool_diffs=[],
            output_similarity=0.98,
            score_diff=-15.0,
            overall_severity=DiffStatus.REGRESSION,
        )
        result = analyze_root_cause(diff)
        assert result is not None
        assert result.category == RootCauseCategory.SCORE_ONLY
        assert result.confidence == Confidence.LOW
        assert "15.0" in result.summary


class TestConfidenceLevels:
    """Verify confidence is assigned correctly."""

    def test_tool_changes_are_high_confidence(self):
        diff = _make_trace_diff(
            tool_diffs=[
                ToolDiff(type="removed", position=0, golden_tool="x", actual_tool=None,
                         severity=DiffStatus.TOOLS_CHANGED, message="removed"),
            ],
        )
        assert analyze_root_cause(diff).confidence == Confidence.HIGH

    def test_parameter_changes_are_high_confidence(self):
        diff = _make_trace_diff(
            tool_diffs=[
                ToolDiff(
                    type="changed", position=0, golden_tool="x", actual_tool="x",
                    severity=DiffStatus.TOOLS_CHANGED, message="changed",
                    parameter_diffs=[
                        ParameterDiff(param_name="a", golden_value=1, actual_value=2, diff_type="value_changed"),
                    ],
                ),
            ],
        )
        assert analyze_root_cause(diff).confidence == Confidence.HIGH

    def test_reorder_is_medium_confidence(self):
        diff = _make_trace_diff(
            tool_diffs=[
                ToolDiff(type="changed", position=0, golden_tool="a", actual_tool="b",
                         severity=DiffStatus.TOOLS_CHANGED, message="changed"),
            ],
        )
        assert analyze_root_cause(diff).confidence == Confidence.MEDIUM

    def test_score_only_is_low_confidence(self):
        diff = _make_trace_diff(tool_diffs=[], output_similarity=0.99, score_diff=-10.0)
        assert analyze_root_cause(diff).confidence == Confidence.LOW


class TestEdgeCases:
    """Edge cases and complex scenarios."""

    def test_multiple_tool_changes(self):
        """When multiple tools are removed, report the first one."""
        diff = _make_trace_diff(
            tool_diffs=[
                ToolDiff(type="removed", position=0, golden_tool="tool_a", actual_tool=None,
                         severity=DiffStatus.TOOLS_CHANGED, message="removed"),
                ToolDiff(type="removed", position=1, golden_tool="tool_b", actual_tool=None,
                         severity=DiffStatus.TOOLS_CHANGED, message="removed"),
            ],
        )
        result = analyze_root_cause(diff)
        assert result is not None
        assert result.category == RootCauseCategory.TOOL_REMOVED
        assert result.root_tool == "tool_a"
        # Both tools should be mentioned in summary
        assert "tool_a" in result.summary
        assert "tool_b" in result.summary

    def test_mixed_added_and_removed_prefers_removed(self):
        """Removed tools take priority over added tools."""
        diff = _make_trace_diff(
            tool_diffs=[
                ToolDiff(type="removed", position=0, golden_tool="important_tool", actual_tool=None,
                         severity=DiffStatus.TOOLS_CHANGED, message="removed"),
                ToolDiff(type="added", position=1, golden_tool=None, actual_tool="new_tool",
                         severity=DiffStatus.TOOLS_CHANGED, message="added"),
            ],
        )
        result = analyze_root_cause(diff)
        assert result is not None
        assert result.category == RootCauseCategory.TOOL_REMOVED

    def test_no_parameter_data_falls_through(self):
        """Changed tools without parameter diffs fall to reorder category."""
        diff = _make_trace_diff(
            tool_diffs=[
                ToolDiff(
                    type="changed", position=0, golden_tool="tool_a", actual_tool="tool_b",
                    severity=DiffStatus.TOOLS_CHANGED, message="changed",
                    parameter_diffs=[],  # No parameter data
                ),
            ],
        )
        result = analyze_root_cause(diff)
        assert result is not None
        assert result.category == RootCauseCategory.TOOL_REORDERED

    def test_tools_changed_severity(self):
        """Non-regression severity still gets root cause analysis."""
        diff = _make_trace_diff(
            overall_severity=DiffStatus.TOOLS_CHANGED,
            tool_diffs=[
                ToolDiff(type="added", position=0, golden_tool=None, actual_tool="extra_tool",
                         severity=DiffStatus.TOOLS_CHANGED, message="added"),
            ],
        )
        result = analyze_root_cause(diff)
        assert result is not None
        assert result.category == RootCauseCategory.TOOL_ADDED

    def test_output_changed_severity(self):
        """OUTPUT_CHANGED severity gets root cause analysis."""
        diff = _make_trace_diff(
            overall_severity=DiffStatus.OUTPUT_CHANGED,
            tool_diffs=[],
            output_similarity=0.80,
            score_diff=0.0,
        )
        result = analyze_root_cause(diff)
        assert result is not None
        assert result.category == RootCauseCategory.OUTPUT_DRIFTED

    def test_multiple_param_diffs_collected(self):
        """All parameter diffs across multiple tools are collected."""
        diff = _make_trace_diff(
            tool_diffs=[
                ToolDiff(
                    type="changed", position=0, golden_tool="tool_a", actual_tool="tool_a",
                    severity=DiffStatus.TOOLS_CHANGED, message="changed",
                    parameter_diffs=[
                        ParameterDiff(param_name="x", golden_value=1, actual_value=2, diff_type="value_changed"),
                    ],
                ),
                ToolDiff(
                    type="changed", position=1, golden_tool="tool_b", actual_tool="tool_b",
                    severity=DiffStatus.TOOLS_CHANGED, message="changed",
                    parameter_diffs=[
                        ParameterDiff(param_name="y", golden_value="a", actual_value="b", diff_type="value_changed"),
                    ],
                ),
            ],
        )
        result = analyze_root_cause(diff)
        assert result is not None
        assert result.category == RootCauseCategory.PARAMETER_CHANGED
        assert len(result.parameter_diffs) == 2
        param_names = {pd.param_name for pd in result.parameter_diffs}
        assert param_names == {"x", "y"}


class TestSerialization:
    """Test to_dict serialization for JSON output."""

    def test_to_dict(self):
        diff = _make_trace_diff(
            tool_diffs=[
                ToolDiff(type="removed", position=0, golden_tool="my_tool", actual_tool=None,
                         severity=DiffStatus.TOOLS_CHANGED, message="removed"),
            ],
        )
        result = analyze_root_cause(diff)
        d = result.to_dict()
        assert d["category"] == "tool_removed"
        assert d["root_tool"] == "my_tool"
        assert d["confidence"] == "high"
        assert isinstance(d["summary"], str)
        assert isinstance(d["suggested_fix"], str)
        assert isinstance(d["parameter_diffs"], list)

    def test_to_dict_with_parameter_diffs(self):
        diff = _make_trace_diff(
            tool_diffs=[
                ToolDiff(
                    type="changed", position=0, golden_tool="api", actual_tool="api",
                    severity=DiffStatus.TOOLS_CHANGED, message="changed",
                    parameter_diffs=[
                        ParameterDiff(param_name="limit", golden_value=50, actual_value=5,
                                      diff_type="value_changed", similarity=None),
                    ],
                ),
            ],
        )
        result = analyze_root_cause(diff)
        d = result.to_dict()
        assert len(d["parameter_diffs"]) == 1
        assert d["parameter_diffs"][0]["param"] == "limit"
        assert d["parameter_diffs"][0]["golden"] == 50
        assert d["parameter_diffs"][0]["actual"] == 5


class TestDisplayIntegration:
    """Test that root cause integrates with _display_check_results."""

    def test_json_output_includes_root_cause(self):
        """Verify root_cause field appears in JSON output structure."""
        diff = _make_trace_diff(
            tool_diffs=[
                ToolDiff(type="removed", position=0, golden_tool="my_tool", actual_tool=None,
                         severity=DiffStatus.TOOLS_CHANGED, message="removed"),
            ],
        )
        root_cause = analyze_root_cause(diff)
        assert root_cause is not None
        d = root_cause.to_dict()
        # Verify the dict is JSON-serializable
        import json
        serialized = json.dumps(d)
        parsed = json.loads(serialized)
        assert parsed["category"] == "tool_removed"


class TestAIEnrichment:
    """Test AI-powered root cause enrichment."""

    @pytest.mark.asyncio
    async def test_high_confidence_skips_ai(self):
        """High-confidence attributions should not call the LLM."""
        diff = _make_trace_diff(
            tool_diffs=[
                ToolDiff(type="removed", position=0, golden_tool="tool_a", actual_tool=None,
                         severity=DiffStatus.TOOLS_CHANGED, message="removed"),
            ],
        )
        analysis = analyze_root_cause(diff)
        assert analysis.confidence == Confidence.HIGH

        with patch("evalview.core.llm_provider.LLMClient") as mock_cls:
            result = await enrich_with_ai(analysis, diff)
            mock_cls.assert_not_called()
            assert result.ai_explanation is None

    @pytest.mark.asyncio
    async def test_low_confidence_calls_ai(self):
        """Low-confidence attributions should call the LLM."""
        diff = _make_trace_diff(
            tool_diffs=[],
            output_similarity=0.60,
        )
        analysis = analyze_root_cause(diff)
        assert analysis.confidence == Confidence.LOW

        mock_client = AsyncMock()
        mock_client.chat_completion.return_value = {
            "explanation": "The model was updated from v1 to v2, causing different phrasing."
        }

        with patch("evalview.core.llm_provider.LLMClient", return_value=mock_client):
            result = await enrich_with_ai(analysis, diff)
            mock_client.chat_completion.assert_called_once()
            assert result.ai_explanation == "The model was updated from v1 to v2, causing different phrasing."

    @pytest.mark.asyncio
    async def test_medium_confidence_calls_ai(self):
        """Medium-confidence attributions should also call the LLM."""
        diff = _make_trace_diff(
            tool_diffs=[
                ToolDiff(type="changed", position=0, golden_tool="a", actual_tool="b",
                         severity=DiffStatus.TOOLS_CHANGED, message="changed"),
            ],
        )
        analysis = analyze_root_cause(diff)
        assert analysis.confidence == Confidence.MEDIUM

        mock_client = AsyncMock()
        mock_client.chat_completion.return_value = {
            "explanation": "Tool reordering is likely caused by an ambiguous prompt."
        }

        with patch("evalview.core.llm_provider.LLMClient", return_value=mock_client):
            result = await enrich_with_ai(analysis, diff)
            assert result.ai_explanation is not None

    @pytest.mark.asyncio
    async def test_ai_failure_degrades_gracefully(self):
        """If the LLM call fails, the original analysis is returned unchanged."""
        diff = _make_trace_diff(
            tool_diffs=[],
            output_similarity=0.60,
        )
        analysis = analyze_root_cause(diff)

        mock_client = AsyncMock()
        mock_client.chat_completion.side_effect = RuntimeError("API down")

        with patch("evalview.core.llm_provider.LLMClient", return_value=mock_client):
            result = await enrich_with_ai(analysis, diff)
            assert result.ai_explanation is None
            assert result.category == RootCauseCategory.OUTPUT_DRIFTED

    @pytest.mark.asyncio
    async def test_ai_no_provider_degrades_gracefully(self):
        """If no LLM provider is available, degrade gracefully."""
        diff = _make_trace_diff(
            tool_diffs=[],
            output_similarity=0.60,
        )
        analysis = analyze_root_cause(diff)

        with patch("evalview.core.llm_provider.LLMClient", side_effect=ValueError("No API key")):
            result = await enrich_with_ai(analysis, diff)
            assert result.ai_explanation is None

    @pytest.mark.asyncio
    async def test_ai_empty_explanation_ignored(self):
        """Empty AI explanation should not be set."""
        diff = _make_trace_diff(
            tool_diffs=[],
            output_similarity=0.60,
        )
        analysis = analyze_root_cause(diff)

        mock_client = AsyncMock()
        mock_client.chat_completion.return_value = {"explanation": ""}

        with patch("evalview.core.llm_provider.LLMClient", return_value=mock_client):
            result = await enrich_with_ai(analysis, diff)
            assert result.ai_explanation is None

    @pytest.mark.asyncio
    async def test_enrich_diffs_batch(self):
        """enrich_diffs_with_ai processes multiple diffs and skips PASSED tests."""
        failing_diff = TraceDiff(
            test_name="test-1",
            has_differences=True,
            tool_diffs=[],
            output_diff=OutputDiff(
                similarity=0.60,
                golden_preview="golden output",
                actual_preview="actual output",
                diff_lines=[],
                severity=DiffStatus.OUTPUT_CHANGED,
            ),
            score_diff=-10.0,
            latency_diff=0.0,
            overall_severity=DiffStatus.REGRESSION,
        )
        passing_diff = TraceDiff(
            test_name="test-2",
            has_differences=False,
            tool_diffs=[],
            output_diff=OutputDiff(
                similarity=1.0,
                golden_preview="",
                actual_preview="",
                diff_lines=[],
                severity=DiffStatus.PASSED,
            ),
            score_diff=0.0,
            latency_diff=0.0,
            overall_severity=DiffStatus.PASSED,
        )

        mock_client = AsyncMock()
        mock_client.chat_completion.return_value = {"explanation": "Drift detected."}

        with patch("evalview.core.llm_provider.LLMClient", return_value=mock_client):
            results = await enrich_diffs_with_ai(
                [("test-1", failing_diff), ("test-2", passing_diff)]
            )
            assert "test-1" in results
            assert results["test-1"].ai_explanation == "Drift detected."
            # PASSED test must not appear in results
            assert "test-2" not in results

    @pytest.mark.asyncio
    async def test_enrich_diffs_batch_high_confidence_not_enriched(self):
        """HIGH-confidence diffs in the batch should not call the LLM."""
        high_conf_diff = TraceDiff(
            test_name="test-hc",
            has_differences=True,
            tool_diffs=[
                ToolDiff(type="removed", position=0, golden_tool="tool_x", actual_tool=None,
                         severity=DiffStatus.TOOLS_CHANGED, message="removed"),
            ],
            output_diff=OutputDiff(
                similarity=1.0, golden_preview="", actual_preview="",
                diff_lines=[], severity=DiffStatus.PASSED,
            ),
            score_diff=-5.0,
            latency_diff=0.0,
            overall_severity=DiffStatus.REGRESSION,
        )
        assert analyze_root_cause(high_conf_diff).confidence == Confidence.HIGH

        mock_client = AsyncMock()
        mock_client.chat_completion.return_value = {"explanation": "Should not be called."}

        with patch("evalview.core.llm_provider.LLMClient", return_value=mock_client):
            results = await enrich_diffs_with_ai([("test-hc", high_conf_diff)])
            assert "test-hc" in results
            mock_client.chat_completion.assert_not_called()
            assert results["test-hc"].ai_explanation is None

    @pytest.mark.asyncio
    async def test_enrich_diffs_batch_empty_input(self):
        """Empty diffs list returns an empty dict without error."""
        results = await enrich_diffs_with_ai([])
        assert results == {}

    @pytest.mark.asyncio
    async def test_ai_chat_completion_returns_none(self):
        """None response from chat_completion is handled without raising."""
        diff = _make_trace_diff(tool_diffs=[], output_similarity=0.60)
        analysis = analyze_root_cause(diff)

        mock_client = AsyncMock()
        mock_client.chat_completion.return_value = None  # type: ignore[assignment]

        with patch("evalview.core.llm_provider.LLMClient", return_value=mock_client):
            result = await enrich_with_ai(analysis, diff)
            # Should degrade gracefully — no ai_explanation set, no exception raised
            assert result.ai_explanation is None

    def test_to_dict_includes_ai_explanation(self):
        """to_dict should include ai_explanation when set."""
        diff = _make_trace_diff(
            tool_diffs=[],
            output_similarity=0.60,
        )
        analysis = analyze_root_cause(diff)
        analysis.ai_explanation = "This is caused by model drift."
        d = analysis.to_dict()
        assert d["ai_explanation"] == "This is caused by model drift."

    def test_to_dict_omits_ai_explanation_when_none(self):
        """to_dict should not include ai_explanation when None."""
        diff = _make_trace_diff(
            tool_diffs=[
                ToolDiff(type="removed", position=0, golden_tool="x", actual_tool=None,
                         severity=DiffStatus.TOOLS_CHANGED, message="removed"),
            ],
        )
        analysis = analyze_root_cause(diff)
        assert analysis.ai_explanation is None
        d = analysis.to_dict()
        assert "ai_explanation" not in d


class TestAIPromptBuilding:
    """Test the prompt construction for AI enrichment."""

    def test_prompt_includes_test_name(self):
        diff = _make_trace_diff(tool_diffs=[], output_similarity=0.70)
        analysis = analyze_root_cause(diff)
        prompt = _build_ai_user_prompt(analysis, diff)
        assert "test-case" in prompt

    def test_prompt_includes_model_change(self):
        diff = _make_trace_diff(
            tool_diffs=[],
            output_similarity=0.70,
            model_changed=True,
            golden_model_id="gpt-4o-v1",
            actual_model_id="gpt-4o-v2",
        )
        analysis = analyze_root_cause(diff)
        prompt = _build_ai_user_prompt(analysis, diff)
        assert "gpt-4o-v1" in prompt
        assert "gpt-4o-v2" in prompt

    def test_prompt_includes_parameter_diffs(self):
        diff = _make_trace_diff(
            tool_diffs=[
                ToolDiff(
                    type="changed", position=0, golden_tool="api", actual_tool="api",
                    severity=DiffStatus.TOOLS_CHANGED, message="changed",
                    parameter_diffs=[
                        ParameterDiff(param_name="limit", golden_value=50, actual_value=5,
                                      diff_type="value_changed"),
                    ],
                ),
            ],
        )
        analysis = analyze_root_cause(diff)
        prompt = _build_ai_user_prompt(analysis, diff)
        assert "limit" in prompt
        assert "50" in prompt

    def test_prompt_includes_output_previews(self):
        diff = _make_trace_diff(tool_diffs=[], output_similarity=0.70)
        analysis = analyze_root_cause(diff)
        prompt = _build_ai_user_prompt(analysis, diff)
        assert "Baseline output preview" in prompt
        assert "Current output preview" in prompt


class TestParamFormatEdgeCases:
    """Test parameter formatting helpers for all diff types."""

    def test_type_changed_summary(self):
        """type_changed produces a meaningful summary."""
        diff = _make_trace_diff(
            tool_diffs=[
                ToolDiff(
                    type="changed", position=0, golden_tool="api", actual_tool="api",
                    severity=DiffStatus.TOOLS_CHANGED, message="changed",
                    parameter_diffs=[
                        ParameterDiff(param_name="count", golden_value=10, actual_value="10",
                                      diff_type="type_changed"),
                    ],
                ),
            ],
        )
        result = analyze_root_cause(diff)
        assert result is not None
        assert result.category == RootCauseCategory.PARAMETER_CHANGED
        assert "count" in result.summary
        assert "type" in result.summary.lower()
        # suggested_fix should mention type change
        assert "type" in result.suggested_fix.lower() or "schema" in result.suggested_fix.lower()

    def test_added_param_summary(self):
        """added parameter produces correct summary."""
        diff = _make_trace_diff(
            tool_diffs=[
                ToolDiff(
                    type="changed", position=0, golden_tool="api", actual_tool="api",
                    severity=DiffStatus.TOOLS_CHANGED, message="changed",
                    parameter_diffs=[
                        ParameterDiff(param_name="verbose", golden_value=None,
                                      actual_value=True, diff_type="added"),
                    ],
                ),
            ],
        )
        result = analyze_root_cause(diff)
        assert result is not None
        assert "verbose" in result.summary
        assert "new" in result.summary.lower()

    def test_unknown_diff_type_fallback(self):
        """Unknown diff types get a generic summary."""
        diff = _make_trace_diff(
            tool_diffs=[
                ToolDiff(
                    type="changed", position=0, golden_tool="api", actual_tool="api",
                    severity=DiffStatus.TOOLS_CHANGED, message="changed",
                    parameter_diffs=[
                        ParameterDiff(param_name="x", golden_value=1,
                                      actual_value=2, diff_type="unknown_type"),
                    ],
                ),
            ],
        )
        result = analyze_root_cause(diff)
        assert result is not None
        assert "x" in result.summary


class TestContractDriftEdgeCase:
    """Test handling of CONTRACT_DRIFT severity."""

    def test_contract_drift_gets_analysis(self):
        """CONTRACT_DRIFT is not PASSED, so it should get a root cause."""
        diff = _make_trace_diff(
            overall_severity=DiffStatus.CONTRACT_DRIFT,
            tool_diffs=[
                ToolDiff(type="removed", position=0, golden_tool="old_api", actual_tool=None,
                         severity=DiffStatus.CONTRACT_DRIFT, message="removed"),
            ],
        )
        result = analyze_root_cause(diff)
        assert result is not None
        assert result.category == RootCauseCategory.TOOL_REMOVED

    def test_contract_drift_no_tool_diffs_falls_through(self):
        """CONTRACT_DRIFT with no tool diffs and similar output gets score_only."""
        diff = _make_trace_diff(
            overall_severity=DiffStatus.CONTRACT_DRIFT,
            tool_diffs=[],
            output_similarity=0.99,
            score_diff=-8.0,
        )
        result = analyze_root_cause(diff)
        assert result is not None
        assert result.category == RootCauseCategory.SCORE_ONLY


class TestSlackNotifierRootCause:
    """Test that Slack alerts include root cause summaries."""

    @pytest.mark.asyncio
    async def test_regression_alert_includes_root_cause(self):
        """Regression alerts should include the root cause summary."""
        from unittest.mock import AsyncMock, patch
        from evalview.core.slack_notifier import SlackNotifier

        diff = _make_trace_diff(
            tool_diffs=[
                ToolDiff(type="removed", position=0, golden_tool="search_api", actual_tool=None,
                         severity=DiffStatus.TOOLS_CHANGED, message="removed"),
            ],
            overall_severity=DiffStatus.REGRESSION,
        )
        analysis = {"has_regressions": True, "has_tools_changed": False,
                     "has_output_changed": False, "all_passed": False}

        notifier = SlackNotifier("https://hooks.slack.com/test")

        with patch("httpx.AsyncClient") as mock_httpx:
            mock_client = AsyncMock()
            mock_response = AsyncMock()
            mock_response.raise_for_status = lambda: None
            mock_client.post.return_value = mock_response
            mock_httpx.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            mock_httpx.return_value.__aexit__ = AsyncMock(return_value=False)

            await notifier.send_regression_alert(
                [("billing-flow", diff)], analysis
            )

            # Verify the posted payload includes root cause
            call_args = mock_client.post.call_args
            payload = call_args.kwargs.get("json") or call_args[1].get("json")
            text = payload["text"]
            assert "search_api" in text
            assert "was expected but not called" in text

    @pytest.mark.asyncio
    async def test_tools_changed_alert_includes_root_cause(self):
        """TOOLS_CHANGED alerts should also include root cause."""
        from evalview.core.root_cause import analyze_root_cause

        diff = _make_trace_diff(
            tool_diffs=[
                ToolDiff(type="added", position=0, golden_tool=None, actual_tool="extra_tool",
                         severity=DiffStatus.TOOLS_CHANGED, message="added"),
            ],
            overall_severity=DiffStatus.TOOLS_CHANGED,
        )

        # Verify root cause is generated for this diff
        rc = analyze_root_cause(diff)
        assert rc is not None
        assert "extra_tool" in rc.summary


# ---------------------------------------------------------------------------
# Helpers shared by narrative tests
# ---------------------------------------------------------------------------

class _FakeMetrics:
    latency = 42.0


class _FakeStep:
    """Minimal stand-in for StepTrace."""
    def __init__(self, tool_name: str, parameters=None, output=None):
        self.tool_name = tool_name
        self.step_name = tool_name
        self.parameters = parameters or {}
        self.output = output
        self.metrics = _FakeMetrics()


class _FakeResult:
    """Minimal stand-in for EvaluationResult."""
    class _Trace:
        def __init__(self, steps):
            self.steps = steps
    def __init__(self, test_case: str, steps):
        self.test_case = test_case
        self.trace = self._Trace(steps)


class _FakeGolden:
    """Minimal stand-in for GoldenTrace."""
    class _Trace:
        def __init__(self, steps):
            self.steps = steps
    def __init__(self, steps):
        self.trace = self._Trace(steps)


class TestNarrativeEnrichment:
    """14 tests for the --explain narrative enrichment feature."""

    # 1 ── always calls LLM even for HIGH-confidence diffs
    @pytest.mark.asyncio
    async def test_narrative_always_calls_ai_for_high_confidence(self):
        """enrich_with_narrative calls LLM even for HIGH-confidence attributions."""
        diff = _make_trace_diff(
            tool_diffs=[
                ToolDiff(type="removed", position=0, golden_tool="critical_tool",
                         actual_tool=None, severity=DiffStatus.TOOLS_CHANGED, message="removed"),
            ],
        )
        analysis = analyze_root_cause(diff)
        assert analysis.confidence == Confidence.HIGH

        mock_client = AsyncMock()
        mock_client.chat_completion.return_value = {
            "narrative": "The critical_tool was removed because the prompt changed."
        }

        with patch("evalview.core.llm_provider.LLMClient", return_value=mock_client):
            result = await enrich_with_narrative(analysis, diff)
            mock_client.chat_completion.assert_called_once()
            assert result.narrative_root_cause == "The critical_tool was removed because the prompt changed."

    # 2 ── LOW confidence
    @pytest.mark.asyncio
    async def test_narrative_calls_ai_for_low_confidence(self):
        """LOW-confidence output drift gets a narrative."""
        diff = _make_trace_diff(tool_diffs=[], output_similarity=0.60)
        analysis = analyze_root_cause(diff)
        assert analysis.confidence == Confidence.LOW

        mock_client = AsyncMock()
        mock_client.chat_completion.return_value = {"narrative": "Output drifted due to model update."}

        with patch("evalview.core.llm_provider.LLMClient", return_value=mock_client):
            result = await enrich_with_narrative(analysis, diff)
            assert result.narrative_root_cause == "Output drifted due to model update."

    # 3 ── MEDIUM confidence
    @pytest.mark.asyncio
    async def test_narrative_calls_ai_for_medium_confidence(self):
        """MEDIUM-confidence tool reordering gets a narrative."""
        diff = _make_trace_diff(
            tool_diffs=[
                ToolDiff(type="changed", position=0, golden_tool="a", actual_tool="b",
                         severity=DiffStatus.TOOLS_CHANGED, message="changed"),
            ],
        )
        analysis = analyze_root_cause(diff)
        assert analysis.confidence == Confidence.MEDIUM

        mock_client = AsyncMock()
        mock_client.chat_completion.return_value = {"narrative": "Tools reordered by model."}

        with patch("evalview.core.llm_provider.LLMClient", return_value=mock_client):
            result = await enrich_with_narrative(analysis, diff)
            assert result.narrative_root_cause == "Tools reordered by model."

    # 4 ── LLM failure degrades gracefully
    @pytest.mark.asyncio
    async def test_narrative_failure_degrades_gracefully(self):
        """If the LLM raises, narrative_root_cause is not set and no exception propagates."""
        diff = _make_trace_diff(tool_diffs=[], output_similarity=0.60)
        analysis = analyze_root_cause(diff)

        mock_client = AsyncMock()
        mock_client.chat_completion.side_effect = RuntimeError("Network timeout")

        with patch("evalview.core.llm_provider.LLMClient", return_value=mock_client):
            result = await enrich_with_narrative(analysis, diff)
            assert result.narrative_root_cause is None
            assert result.category == RootCauseCategory.OUTPUT_DRIFTED

    # 5 ── no LLM provider
    @pytest.mark.asyncio
    async def test_narrative_no_provider_degrades_gracefully(self):
        """No API key → narrative_root_cause stays None."""
        diff = _make_trace_diff(tool_diffs=[], output_similarity=0.60)
        analysis = analyze_root_cause(diff)

        with patch("evalview.core.llm_provider.LLMClient", side_effect=ValueError("No API key")):
            result = await enrich_with_narrative(analysis, diff)
            assert result.narrative_root_cause is None

    # 6 ── empty narrative ignored
    @pytest.mark.asyncio
    async def test_narrative_empty_response_not_set(self):
        """An empty narrative string from the LLM is not stored."""
        diff = _make_trace_diff(tool_diffs=[], output_similarity=0.60)
        analysis = analyze_root_cause(diff)

        mock_client = AsyncMock()
        mock_client.chat_completion.return_value = {"narrative": ""}

        with patch("evalview.core.llm_provider.LLMClient", return_value=mock_client):
            result = await enrich_with_narrative(analysis, diff)
            assert result.narrative_root_cause is None

    # 7 ── None response from chat_completion handled
    @pytest.mark.asyncio
    async def test_narrative_none_response_handled(self):
        """None return from chat_completion degrades gracefully."""
        diff = _make_trace_diff(tool_diffs=[], output_similarity=0.60)
        analysis = analyze_root_cause(diff)

        mock_client = AsyncMock()
        mock_client.chat_completion.return_value = None  # type: ignore[assignment]

        with patch("evalview.core.llm_provider.LLMClient", return_value=mock_client):
            result = await enrich_with_narrative(analysis, diff)
            assert result.narrative_root_cause is None

    # 8 ── prompt includes golden trace tool names
    def test_narrative_prompt_includes_golden_steps(self):
        """_build_narrative_prompt includes baseline step tool names."""
        diff = _make_trace_diff(tool_diffs=[], output_similarity=0.70)
        analysis = analyze_root_cause(diff)
        golden = [_FakeStep("fetch_data"), _FakeStep("validate_schema")]
        prompt = _build_narrative_prompt(analysis, diff, golden_steps=golden)
        assert "fetch_data" in prompt
        assert "validate_schema" in prompt
        assert "BASELINE" in prompt

    # 9 ── prompt includes actual trace tool names
    def test_narrative_prompt_includes_actual_steps(self):
        """_build_narrative_prompt includes current-run step tool names."""
        diff = _make_trace_diff(tool_diffs=[], output_similarity=0.70)
        analysis = analyze_root_cause(diff)
        actual = [_FakeStep("fetch_data"), _FakeStep("write_output")]
        prompt = _build_narrative_prompt(analysis, diff, actual_steps=actual)
        assert "write_output" in prompt
        assert "CURRENT" in prompt

    # 10 ── batch skips PASSED tests
    @pytest.mark.asyncio
    async def test_narrative_batch_skips_passed_tests(self):
        """enrich_diffs_with_narrative excludes PASSED diffs from the result dict."""
        failing = TraceDiff(
            test_name="failing",
            has_differences=True,
            tool_diffs=[],
            output_diff=OutputDiff(
                similarity=0.55, golden_preview="g", actual_preview="a",
                diff_lines=[], severity=DiffStatus.OUTPUT_CHANGED,
            ),
            score_diff=-8.0, latency_diff=0.0,
            overall_severity=DiffStatus.REGRESSION,
        )
        passing = TraceDiff(
            test_name="passing",
            has_differences=False,
            tool_diffs=[],
            output_diff=OutputDiff(
                similarity=1.0, golden_preview="", actual_preview="",
                diff_lines=[], severity=DiffStatus.PASSED,
            ),
            score_diff=0.0, latency_diff=0.0,
            overall_severity=DiffStatus.PASSED,
        )

        mock_client = AsyncMock()
        mock_client.chat_completion.return_value = {"narrative": "Drift."}

        with patch("evalview.core.llm_provider.LLMClient", return_value=mock_client):
            results = await enrich_diffs_with_narrative(
                [("failing", failing), ("passing", passing)]
            )
            assert "failing" in results
            assert results["failing"].narrative_root_cause == "Drift."
            assert "passing" not in results

    # 11 ── batch processes multiple failing diffs
    @pytest.mark.asyncio
    async def test_narrative_batch_processes_multiple_failures(self):
        """enrich_diffs_with_narrative enriches all failing tests concurrently."""
        def _failing(name: str) -> TraceDiff:
            return TraceDiff(
                test_name=name,
                has_differences=True,
                tool_diffs=[],
                output_diff=OutputDiff(
                    similarity=0.60, golden_preview="g", actual_preview="a",
                    diff_lines=[], severity=DiffStatus.OUTPUT_CHANGED,
                ),
                score_diff=-5.0, latency_diff=0.0,
                overall_severity=DiffStatus.REGRESSION,
            )

        mock_client = AsyncMock()
        mock_client.chat_completion.return_value = {"narrative": "Narrative text."}

        diffs = [("t1", _failing("t1")), ("t2", _failing("t2")), ("t3", _failing("t3"))]
        with patch("evalview.core.llm_provider.LLMClient", return_value=mock_client):
            results = await enrich_diffs_with_narrative(diffs)
            assert set(results) == {"t1", "t2", "t3"}
            assert mock_client.chat_completion.call_count == 3

    # 12 ── empty input
    @pytest.mark.asyncio
    async def test_narrative_batch_empty_input(self):
        """Empty diffs list returns an empty dict without error."""
        results = await enrich_diffs_with_narrative([])
        assert results == {}

    # 13 ── narrative_root_cause in to_dict
    def test_narrative_in_to_dict(self):
        """to_dict includes narrative_root_cause when set."""
        diff = _make_trace_diff(tool_diffs=[], output_similarity=0.60)
        analysis = analyze_root_cause(diff)
        analysis.narrative_root_cause = "Model drift caused this regression."
        d = analysis.to_dict()
        assert d["narrative_root_cause"] == "Model drift caused this regression."

    # 14 ── narrative_root_cause absent from to_dict when None
    def test_narrative_omitted_from_to_dict_when_none(self):
        """to_dict does not include narrative_root_cause when it's None."""
        diff = _make_trace_diff(
            tool_diffs=[
                ToolDiff(type="removed", position=0, golden_tool="x", actual_tool=None,
                         severity=DiffStatus.TOOLS_CHANGED, message="removed"),
            ],
        )
        analysis = analyze_root_cause(diff)
        assert analysis.narrative_root_cause is None
        d = analysis.to_dict()
        assert "narrative_root_cause" not in d


class TestHelperUtilities:
    """Tests for shared utility helpers."""

    def test_strip_markdown_fences_json(self):
        """Strip ```json ... ``` wrappers."""
        raw = '```json\n{"explanation": "test"}\n```'
        assert _strip_markdown_fences(raw) == '{"explanation": "test"}'

    def test_strip_markdown_fences_plain(self):
        """Strip plain ``` ... ``` wrappers."""
        raw = '```\n{"narrative": "text"}\n```'
        assert _strip_markdown_fences(raw) == '{"narrative": "text"}'

    def test_strip_markdown_fences_clean(self):
        """Leave text without fences unchanged."""
        raw = '{"explanation": "no fences here"}'
        assert _strip_markdown_fences(raw) == raw

    def test_format_params_brief_dict(self):
        """Format a simple dict as a compact one-liner."""
        result = _format_params_brief({"limit": 10, "query": "hello"})
        assert "limit" in result
        assert "hello" in result

    def test_format_params_brief_empty(self):
        """Empty params returns empty string."""
        assert _format_params_brief({}) == ""
        assert _format_params_brief(None) == ""

    def test_format_params_brief_truncates_long_dict(self):
        """Dicts with more than 4 keys show a '+N more' suffix."""
        params = {f"key{i}": i for i in range(6)}
        result = _format_params_brief(params)
        assert "+2 more" in result
