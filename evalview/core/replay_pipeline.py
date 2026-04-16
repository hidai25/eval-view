"""Production replay pipeline — take a failed trace, re-execute, and diff.

Connects the existing import, run, and diff engines into a single pipeline:

    production trace → parse → re-execute against live agent → diff vs original

This directly addresses the "demo-to-prod gap" complaint: teams can take a
real production failure, replay it against the current agent, and get a
structured diff showing what changed.

Usage:
    from evalview.core.replay_pipeline import ReplayPipeline

    pipeline = ReplayPipeline(adapter_name="http", endpoint="http://localhost:8080")
    report = await pipeline.replay_trace(original_trace)
    print(report.summary())

The pipeline is intentionally simple — it's a coordinator, not a new engine.
It reuses DiffEngine, the adapter system, and the evaluator.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

from evalview.core.diff import DiffEngine, DiffStatus, TraceDiff
from evalview.core.golden import GoldenMetadata, GoldenTrace
from evalview.core.types import (
    ExecutionTrace,
    TestCase,
    TestInput,
    ExpectedBehavior,
    Thresholds,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass
class ReplayResult:
    """Result of replaying a single production trace."""

    original_trace: ExecutionTrace
    replay_trace: Optional[ExecutionTrace] = None
    diff: Optional[TraceDiff] = None
    error: Optional[str] = None
    replay_score: float = 0.0

    @property
    def succeeded(self) -> bool:
        return self.replay_trace is not None and self.error is None

    @property
    def has_differences(self) -> bool:
        return self.diff is not None and self.diff.has_differences

    @property
    def status(self) -> str:
        if self.error:
            return "error"
        if self.diff is None:
            return "unknown"
        return self.diff.overall_severity.value

    def summary(self) -> str:
        if self.error:
            return f"Replay failed: {self.error}"
        if self.diff is None:
            return "Replay completed but no diff available"
        if not self.diff.has_differences:
            return "Replay matches original — behavior is stable"

        parts = [f"Status: {self.diff.overall_severity.value}"]
        if self.diff.tool_diffs:
            parts.append(f"{len(self.diff.tool_diffs)} tool change(s)")
        if self.diff.output_diff:
            parts.append(f"output similarity: {self.diff.output_diff.similarity:.0%}")
        if self.diff.score_diff != 0:
            parts.append(f"score delta: {self.diff.score_diff:+.1f}")
        return " | ".join(parts)


@dataclass
class ReplayBatchResult:
    """Result of replaying multiple production traces."""

    results: List[ReplayResult] = field(default_factory=list)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    @property
    def total(self) -> int:
        return len(self.results)

    @property
    def succeeded(self) -> int:
        return sum(1 for r in self.results if r.succeeded)

    @property
    def failed(self) -> int:
        return sum(1 for r in self.results if r.error)

    @property
    def changed(self) -> int:
        return sum(1 for r in self.results if r.has_differences)

    @property
    def stable(self) -> int:
        return sum(1 for r in self.results if r.succeeded and not r.has_differences)

    def summary(self) -> str:
        if not self.results:
            return "No traces replayed"
        return (
            f"Replayed {self.total} trace(s): "
            f"{self.stable} stable, {self.changed} changed, "
            f"{self.failed} failed"
        )


# ---------------------------------------------------------------------------
# Trace → TestCase conversion
# ---------------------------------------------------------------------------


def trace_to_test_case(
    trace: ExecutionTrace,
    name: str = "replay",
) -> TestCase:
    """Convert a production trace into a TestCase for re-execution.

    The test case uses the trace's original query as input and the
    tool sequence as expected behavior, so the replay can be diffed
    against the original.
    """
    tool_names = [step.tool_name for step in trace.steps]

    return TestCase(
        name=name,
        input=TestInput(query=trace.final_output),  # Use final output as fallback
        expected=ExpectedBehavior(
            tools=tool_names if tool_names else None,
        ),
        thresholds=Thresholds(min_score=0),  # Don't fail on score — we want the diff
    )


def trace_to_golden(
    trace: ExecutionTrace,
    test_name: str = "replay",
    score: float = 100.0,
) -> GoldenTrace:
    """Convert a production trace into a GoldenTrace for diffing.

    The original production trace becomes the "blessed" baseline that
    the replay is compared against.
    """
    tool_sequence = [step.tool_name for step in trace.steps]

    metadata = GoldenMetadata(
        test_name=test_name,
        blessed_at=trace.start_time,
        blessed_by="production-replay",
        score=score,
        model_id=trace.model_id,
        model_provider=trace.model_provider,
    )

    return GoldenTrace(
        metadata=metadata,
        trace=trace,
        tool_sequence=tool_sequence,
    )


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


class ReplayPipeline:
    """Orchestrates production trace replay and diffing.

    Takes a production execution trace, re-runs the same query against
    the current agent, and produces a structured diff showing what changed.
    """

    def __init__(
        self,
        adapter_name: str = "http",
        endpoint: Optional[str] = None,
        adapter_config: Optional[Dict[str, Any]] = None,
        diff_config: Optional[Any] = None,
    ):
        """Initialize the replay pipeline.

        Args:
            adapter_name: Which adapter to use for replay (e.g., "http", "anthropic").
            endpoint: Agent endpoint URL.
            adapter_config: Additional adapter configuration.
            diff_config: Optional DiffConfig for the diff engine.
        """
        self.adapter_name = adapter_name
        self.endpoint = endpoint
        self.adapter_config = adapter_config or {}
        self.diff_engine = DiffEngine(config=diff_config)

    async def replay_trace(
        self,
        original_trace: ExecutionTrace,
        query: Optional[str] = None,
        test_name: str = "replay",
        original_score: float = 100.0,
    ) -> ReplayResult:
        """Replay a single production trace and diff against original.

        Args:
            original_trace: The original production trace to replay.
            query: The original user query. If not provided, uses the
                  first step's parameters or the final output.
            test_name: Name for the test case.
            original_score: Score to assign to the original (baseline).

        Returns:
            ReplayResult with the original, replay, and diff.
        """
        # Convert original to golden baseline
        golden = trace_to_golden(original_trace, test_name, original_score)

        # Build test case from the original
        if query is None:
            # Try to extract query from trace
            query = _extract_query(original_trace)

        test_case = TestCase(
            name=test_name,
            input=TestInput(query=query),
            expected=ExpectedBehavior(
                tools=[step.tool_name for step in original_trace.steps] or None,
            ),
            thresholds=Thresholds(min_score=0),
            adapter=self.adapter_name if self.adapter_name != "http" else None,
            endpoint=self.endpoint,
            adapter_config=self.adapter_config or None,
        )

        try:
            # Get adapter and execute
            from evalview.core.adapter_factory import create_adapter

            adapter = create_adapter(
                adapter_name=self.adapter_name,
                endpoint=self.endpoint,
                **(self.adapter_config or {}),
            )

            replay_trace = await adapter.execute(test_case)

            # Diff the replay against the original
            diff = self.diff_engine.compare(golden, replay_trace, actual_score=0.0)

            return ReplayResult(
                original_trace=original_trace,
                replay_trace=replay_trace,
                diff=diff,
            )

        except Exception as e:
            logger.error("Replay failed: %s", e)
            return ReplayResult(
                original_trace=original_trace,
                error=str(e),
            )

    async def replay_batch(
        self,
        traces: List[ExecutionTrace],
        queries: Optional[List[Optional[str]]] = None,
    ) -> ReplayBatchResult:
        """Replay multiple production traces.

        Args:
            traces: List of production traces to replay.
            queries: Optional parallel list of original queries.

        Returns:
            ReplayBatchResult with per-trace results.
        """
        import asyncio

        batch = ReplayBatchResult(started_at=datetime.now())
        queries = queries or [None] * len(traces)

        results = []
        for i, (trace, query) in enumerate(zip(traces, queries)):
            result = await self.replay_trace(
                original_trace=trace,
                query=query,
                test_name=f"replay-{i + 1}",
            )
            results.append(result)

        batch.results = results
        batch.completed_at = datetime.now()
        return batch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _extract_query(trace: ExecutionTrace) -> str:
    """Best-effort extraction of the original query from a trace.

    Tries several heuristics:
    1. First step's parameters for common query fields
    2. First turn's query (multi-turn traces)
    3. Falls back to describing the tool sequence
    """
    # Try multi-turn traces first
    if trace.turns and trace.turns[0].query:
        return trace.turns[0].query

    # Try first step's parameters
    if trace.steps:
        first_params = trace.steps[0].parameters
        for key in ("query", "prompt", "message", "input", "question", "user_message"):
            if key in first_params and isinstance(first_params[key], str):
                return first_params[key]

    # Fall back to describing what the trace did
    if trace.steps:
        tools = [s.tool_name for s in trace.steps[:3]]
        return f"[replay] Original trace used: {', '.join(tools)}"

    return "[replay] No query could be extracted from trace"
