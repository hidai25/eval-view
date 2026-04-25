"""Integration tests for the replay pipeline.

Tests the full round-trip: build a trace, replay it through a mock adapter,
verify the diff is produced correctly.
"""
from __future__ import annotations

import asyncio
from datetime import datetime
from typing import Any, Dict, Optional
from unittest.mock import AsyncMock, patch


from evalview.core.replay_pipeline import (
    ReplayPipeline,
    ReplayResult,
    ReplayBatchResult,
    trace_to_golden,
    _extract_query,
    _get_tool_sequence,
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
    params: Optional[Dict[str, Any]] = None,
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


def _trace(
    steps: Optional[list] = None,
    output: str = "done",
    turns: Optional[list] = None,
) -> ExecutionTrace:
    if steps is None:
        steps = [_step("search", {"query": "hello"}), _step("format")]
    return ExecutionTrace(
        session_id="test",
        start_time=datetime(2025, 1, 1),
        end_time=datetime(2025, 1, 1, 0, 1),
        steps=steps,
        final_output=output,
        metrics=ExecutionMetrics(total_cost=0.1, total_latency=5000),
        turns=turns,
    )


# ---------------------------------------------------------------------------
# Unit tests for helpers
# ---------------------------------------------------------------------------


class TestGetToolSequence:
    def test_extracts_tool_names(self):
        trace = _trace(steps=[_step("a"), _step("b"), _step("c")])
        assert _get_tool_sequence(trace) == ["a", "b", "c"]

    def test_empty_steps(self):
        trace = _trace(steps=[])
        assert _get_tool_sequence(trace) == []


class TestExtractQuery:
    def test_from_multi_turn(self):
        turns = [TurnTrace(index=1, query="What is the weather?", output="Sunny")]
        trace = _trace(turns=turns)
        assert _extract_query(trace) == "What is the weather?"

    def test_from_step_params(self):
        steps = [_step("search", {"query": "hello world"})]
        trace = _trace(steps=steps, turns=None)
        assert _extract_query(trace) == "hello world"

    def test_from_step_params_message_key(self):
        steps = [_step("chat", {"message": "hi there"})]
        trace = _trace(steps=steps, turns=None)
        assert _extract_query(trace) == "hi there"

    def test_fallback_to_tool_description(self):
        steps = [_step("custom_tool", {"x": 42})]
        trace = _trace(steps=steps, turns=None)
        result = _extract_query(trace)
        assert "[replay]" in result
        assert "custom_tool" in result

    def test_empty_trace_fallback(self):
        trace = _trace(steps=[], turns=None)
        result = _extract_query(trace)
        assert "[replay]" in result


class TestTraceToGolden:
    def test_creates_golden_with_correct_metadata(self):
        trace = _trace()
        golden = trace_to_golden(trace, test_name="my-test", score=95.0)
        assert golden.metadata.test_name == "my-test"
        assert golden.metadata.score == 95.0
        assert golden.metadata.blessed_by == "production-replay"
        assert golden.tool_sequence == ["search", "format"]
        assert golden.trace is trace


# ---------------------------------------------------------------------------
# ReplayResult and ReplayBatchResult
# ---------------------------------------------------------------------------


class TestReplayResult:
    def test_succeeded_when_replay_present(self):
        r = ReplayResult(original_trace=_trace(), replay_trace=_trace())
        assert r.succeeded is True

    def test_not_succeeded_on_error(self):
        r = ReplayResult(original_trace=_trace(), error="boom")
        assert r.succeeded is False

    def test_status_error(self):
        r = ReplayResult(original_trace=_trace(), error="timeout")
        assert r.status == "error"

    def test_status_unknown_no_diff(self):
        r = ReplayResult(original_trace=_trace(), replay_trace=_trace())
        assert r.status == "unknown"

    def test_summary_on_error(self):
        r = ReplayResult(original_trace=_trace(), error="timeout")
        assert "timeout" in r.summary()

    def test_summary_no_diff(self):
        r = ReplayResult(original_trace=_trace(), replay_trace=_trace())
        assert "no diff" in r.summary().lower()


class TestReplayBatchResult:
    def test_empty_batch(self):
        b = ReplayBatchResult()
        assert b.total == 0
        assert b.succeeded == 0
        assert b.failed == 0
        assert b.changed == 0
        assert b.stable == 0
        assert "No traces" in b.summary()

    def test_counts_with_mixed_results(self):
        b = ReplayBatchResult(results=[
            ReplayResult(original_trace=_trace(), replay_trace=_trace()),  # succeeded, no diff
            ReplayResult(original_trace=_trace(), error="boom"),  # failed
        ])
        assert b.total == 2
        assert b.succeeded == 1
        assert b.failed == 1
        assert b.stable == 1  # succeeded + no differences


# ---------------------------------------------------------------------------
# Full pipeline integration test (mocked adapter)
# ---------------------------------------------------------------------------


class TestReplayPipelineIntegration:
    """End-to-end test: original trace → replay through mock adapter → diff."""

    def test_replay_identical_trace_is_stable(self):
        """When replay produces the same trace, result should be stable."""
        original = _trace(
            steps=[_step("search", {"q": "weather"}), _step("format")],
            output="It's sunny",
        )
        # The mock adapter returns an identical trace
        replay_trace = _trace(
            steps=[_step("search", {"q": "weather"}), _step("format")],
            output="It's sunny",
        )

        mock_adapter = AsyncMock()
        mock_adapter.execute.return_value = replay_trace

        pipeline = ReplayPipeline(adapter_name="http", endpoint="http://test")

        with patch.object(pipeline, "_get_adapter", return_value=mock_adapter):
            result = asyncio.run(pipeline.replay_trace(
                original_trace=original,
                query="What's the weather?",
            ))

        assert result.succeeded
        assert result.error is None
        assert result.replay_trace is not None
        assert result.diff is not None
        # Identical trace = stable
        assert result.status == "passed"
        assert "stable" in result.summary().lower() or "passed" in result.status

    def test_replay_different_tools_shows_change(self):
        """When replay uses different tools, diff should show changes."""
        original = _trace(
            steps=[_step("search"), _step("analyze"), _step("format")],
            output="result A",
        )
        # Replay uses different tools
        replay_trace = _trace(
            steps=[_step("api_call"), _step("format")],
            output="result B",
        )

        mock_adapter = AsyncMock()
        mock_adapter.execute.return_value = replay_trace

        pipeline = ReplayPipeline(adapter_name="http", endpoint="http://test")

        with patch.object(pipeline, "_get_adapter", return_value=mock_adapter):
            result = asyncio.run(pipeline.replay_trace(
                original_trace=original,
                query="test query",
            ))

        assert result.succeeded
        assert result.has_differences
        assert result.status != "passed"

    def test_replay_adapter_failure_captured(self):
        """When the adapter raises, error is captured in ReplayResult."""
        original = _trace()

        mock_adapter = AsyncMock()
        mock_adapter.execute.side_effect = ConnectionError("refused")

        pipeline = ReplayPipeline(adapter_name="http", endpoint="http://test")

        with patch.object(pipeline, "_get_adapter", return_value=mock_adapter):
            result = asyncio.run(pipeline.replay_trace(
                original_trace=original,
                query="test",
            ))

        assert not result.succeeded
        assert result.error is not None
        assert "refused" in result.error
        assert result.status == "error"

    def test_replay_extracts_query_from_trace(self):
        """When no query is provided, it should be extracted from the trace."""
        original = _trace(
            steps=[_step("search", {"query": "extracted question"})],
        )

        mock_adapter = AsyncMock()
        mock_adapter.execute.return_value = _trace()

        pipeline = ReplayPipeline(adapter_name="http", endpoint="http://test")

        with patch.object(pipeline, "_get_adapter", return_value=mock_adapter):
            result = asyncio.run(pipeline.replay_trace(original_trace=original))

        # Verify the adapter was called with the extracted query
        mock_adapter.execute.assert_called_once_with("extracted question")
        assert result.succeeded

    def test_replay_batch_runs_all_traces(self):
        """Batch replay should process all traces and return aggregate results."""
        traces = [_trace() for _ in range(3)]
        queries = ["q1", "q2", "q3"]

        mock_adapter = AsyncMock()
        mock_adapter.execute.return_value = _trace()

        pipeline = ReplayPipeline(adapter_name="http", endpoint="http://test")

        with patch.object(pipeline, "_get_adapter", return_value=mock_adapter):
            batch = asyncio.run(pipeline.replay_batch(traces, queries=queries))

        assert batch.total == 3
        assert batch.succeeded == 3
        assert batch.failed == 0
        assert batch.started_at is not None
        assert batch.completed_at is not None
        assert mock_adapter.execute.call_count == 3

    def test_replay_batch_handles_mixed_results(self):
        """Batch should handle a mix of successes and failures."""
        traces = [_trace(), _trace()]
        queries = ["good", "bad"]

        call_count = 0

        async def mock_execute(query: str, context=None):
            nonlocal call_count
            call_count += 1
            if query == "bad":
                raise TimeoutError("timed out")
            return _trace()

        mock_adapter = AsyncMock()
        mock_adapter.execute = mock_execute

        pipeline = ReplayPipeline(adapter_name="http", endpoint="http://test")

        with patch.object(pipeline, "_get_adapter", return_value=mock_adapter):
            batch = asyncio.run(pipeline.replay_batch(traces, queries=queries))

        assert batch.total == 2
        assert batch.succeeded == 1
        assert batch.failed == 1
        assert "1 stable" in batch.summary() or "1 failed" in batch.summary()
