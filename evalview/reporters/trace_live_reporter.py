"""Live trace reporter for real-time execution visibility.

Provides beautiful terminal output showing spans as they complete,
with color-coding based on latency and cost thresholds.

Example usage:
    reporter = create_trace_reporter(trace_out_path="trace.jsonl")
    try:
        reporter.report_from_execution_trace(trace, "test_name")
    finally:
        reporter.close()

Or with context manager:
    with create_trace_reporter(trace_out_path="trace.jsonl") as reporter:
        reporter.report_from_execution_trace(trace, "test_name")
"""

from __future__ import annotations

import json
import logging
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Tuple, Dict, Any, TextIO, Iterator

from rich.console import Console
from rich.text import Text

from evalview.core.types import (
    Span,
    SpanKind,
    TraceContext,
    ExecutionTrace,
)

__all__ = [
    "LiveTraceReporter",
    "create_trace_reporter",
    "LATENCY_THRESHOLDS",
    "COST_THRESHOLDS",
]

logger = logging.getLogger(__name__)

# Thresholds for color coding (in ms and USD)
LATENCY_THRESHOLDS = {
    "fast": 1000,     # < 1s = green
    "moderate": 3000,  # 1-3s = yellow
    # > 3s = red
}

COST_THRESHOLDS = {
    "cheap": 0.01,     # < $0.01 = green
    "moderate": 0.05,  # $0.01-$0.05 = yellow
    # > $0.05 = red
}


class LiveTraceReporter:
    """Reports trace spans in real-time with color-coded output.

    Provides live feedback during test execution showing:
    - Each span as it completes (LLM calls, tool executions)
    - Color-coded latency and cost indicators
    - Running totals and final summary

    Note: This class is NOT thread-safe. For parallel test execution,
    create a separate reporter instance per test or synchronize access.
    """

    def __init__(
        self,
        console: Optional[Console] = None,
        jsonl_output: Optional[TextIO] = None,
    ):
        """Initialize the live trace reporter.

        Args:
            console: Rich console for output. Creates new one if not provided.
            jsonl_output: Optional file handle for JSONL export. Caller owns
                         the lifecycle unless using create_trace_reporter().
        """
        self.console = console or Console()
        self._jsonl_output = jsonl_output
        self._owns_file = False  # Set by create_trace_reporter

        # Running totals (reset per trace)
        self._total_cost: float = 0.0
        self._total_tokens: int = 0
        self._llm_calls: int = 0
        self._tool_calls: int = 0
        self._start_time: Optional[datetime] = None
        self._current_trace_id: Optional[str] = None
        self._spans_by_cost: List[Tuple[str, float]] = []
        self._spans_by_latency: List[Tuple[str, float]] = []

    def close(self) -> None:
        """Close the JSONL output file if we own it.

        Safe to call multiple times.
        """
        if self._owns_file and self._jsonl_output:
            try:
                self._jsonl_output.close()
            except Exception as e:
                logger.warning(f"Error closing trace output file: {e}")
            finally:
                self._jsonl_output = None
                self._owns_file = False

    def __enter__(self) -> "LiveTraceReporter":
        """Context manager entry."""
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Context manager exit - ensures file is closed."""
        self.close()

    def _reset_state(self) -> None:
        """Reset running totals for a new trace."""
        self._total_cost = 0.0
        self._total_tokens = 0
        self._llm_calls = 0
        self._tool_calls = 0
        self._start_time = None
        self._current_trace_id = None
        self._spans_by_cost = []
        self._spans_by_latency = []

    def _get_latency_color(self, latency_ms: float) -> str:
        """Get color based on latency threshold."""
        if latency_ms < LATENCY_THRESHOLDS["fast"]:
            return "green"
        elif latency_ms < LATENCY_THRESHOLDS["moderate"]:
            return "yellow"
        return "red"

    def _get_cost_color(self, cost: float) -> str:
        """Get color based on cost threshold."""
        if cost < COST_THRESHOLDS["cheap"]:
            return "green"
        elif cost < COST_THRESHOLDS["moderate"]:
            return "yellow"
        return "red"

    def _format_cost(self, cost: float) -> str:
        """Format cost for human-readable display."""
        if cost == 0:
            return "$0.00"
        elif cost < 0.01:
            return f"${cost:.4f}"
        return f"${cost:.2f}"

    def _format_latency(self, latency_ms: float) -> str:
        """Format latency for human-readable display."""
        if latency_ms < 1000:
            return f"{latency_ms:.0f}ms"
        return f"{latency_ms / 1000:.1f}s"

    def _write_jsonl(self, record: Dict[str, Any]) -> None:
        """Write a record to JSONL output if configured.

        Errors are logged but not raised to avoid disrupting test execution.
        """
        if not self._jsonl_output:
            return

        try:
            def serialize(obj: Any) -> str:
                if isinstance(obj, datetime):
                    return obj.isoformat()
                raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

            self._jsonl_output.write(json.dumps(record, default=serialize) + "\n")
            self._jsonl_output.flush()
        except Exception as e:
            logger.warning(f"Failed to write trace record: {e}")

    def trace_started(self, trace_id: str, test_name: str) -> None:
        """Called when a trace begins. Resets state and prints header."""
        self._reset_state()
        self._start_time = datetime.now()
        self._current_trace_id = trace_id

        self.console.print()
        self.console.print(f"[bold cyan]━━━ Trace: {test_name} ━━━[/bold cyan]")

        self._write_jsonl({
            "type": "trace_start",
            "trace_id": trace_id,
            "trace_spec_version": "1.0",
            "test_name": test_name,
            "source": "eval",
            "started_at": self._start_time,
        })

    def span_completed(self, span: Span) -> None:
        """Called when a span completes. Displays it in real-time."""
        latency_ms = span.duration_ms or 0.0
        cost = span.cost or 0.0

        # Update running totals
        self._total_cost += cost
        if span.kind == SpanKind.LLM:
            self._llm_calls += 1
            if span.llm:
                prompt_tokens = span.llm.prompt_tokens or 0
                completion_tokens = span.llm.completion_tokens or 0
                self._total_tokens += prompt_tokens + completion_tokens
        elif span.kind == SpanKind.TOOL:
            self._tool_calls += 1

        # Track for summary
        self._spans_by_cost.append((span.name, cost))
        self._spans_by_latency.append((span.name, latency_ms))

        # Build display line
        line = Text()

        # Indent child spans
        indent = "  " if span.parent_span_id else ""
        line.append(indent)

        # Span type indicator with status-based color
        span_type = span.kind.value.lower() if span.kind else "span"
        type_style = "bold red" if span.status == "error" else "bold blue"
        line.append(f"[{span_type}] ", style=type_style)

        # Span name
        line.append(span.name, style="bold")

        # LLM-specific: token counts
        if span.kind == SpanKind.LLM and span.llm:
            input_tokens = span.llm.prompt_tokens or 0
            output_tokens = span.llm.completion_tokens or 0
            line.append(f" → {input_tokens:,} in / {output_tokens:,} out", style="dim")

        # Tool-specific: success/error status
        if span.kind == SpanKind.TOOL:
            status_text = "success" if span.status == "ok" else "error"
            status_color = "green" if span.status == "ok" else "red"
            line.append(" → ", style="dim")
            line.append(status_text, style=status_color)

        # Cost (only if non-zero)
        if cost > 0:
            cost_color = self._get_cost_color(cost)
            line.append(" → ", style="dim")
            line.append(self._format_cost(cost), style=cost_color)

        # Latency (always shown)
        latency_color = self._get_latency_color(latency_ms)
        line.append(" (", style="dim")
        line.append(self._format_latency(latency_ms), style=latency_color)
        line.append(")", style="dim")

        # Error message on new line if present
        if span.status == "error" and span.error_message:
            truncated_error = span.error_message[:100]
            line.append(f"\n{indent}  └─ ", style="dim")
            line.append(truncated_error, style="red")

        self.console.print(line)

        # Write to JSONL
        self._write_jsonl({
            "type": "span",
            "span_id": span.span_id,
            "parent_span_id": span.parent_span_id,
            "trace_id": span.trace_id,
            "span_type": span.kind.value if span.kind else "unknown",
            "name": span.name,
            "start_time": span.start_time,
            "end_time": span.end_time,
            "latency_ms": latency_ms,
            "status": span.status,
            "error_message": span.error_message,
            "cost_usd": cost,
            "llm": {
                "provider": span.llm.provider,
                "model": span.llm.model,
                "input_tokens": span.llm.prompt_tokens,
                "output_tokens": span.llm.completion_tokens,
            } if span.kind == SpanKind.LLM and span.llm else None,
            "tool": {
                "tool_name": span.tool.tool_name,
                "tool_success": span.status == "ok",
            } if span.kind == SpanKind.TOOL and span.tool else None,
        })

    def trace_completed(self, trace_context: Optional[TraceContext] = None) -> None:
        """Called when a trace completes. Displays summary."""
        end_time = datetime.now()
        total_time_ms = (
            (end_time - self._start_time).total_seconds() * 1000
            if self._start_time else 0.0
        )

        # Use stored trace_id, fall back to context, then "unknown"
        trace_id = (
            self._current_trace_id
            or (trace_context.trace_id if trace_context else None)
            or "unknown"
        )

        self.console.print()
        self.console.print("[bold cyan]━━━ Trace Summary ━━━[/bold cyan]")

        # Build the 3 headline numbers
        summary = Text()

        # Cost
        cost_color = self._get_cost_color(self._total_cost)
        summary.append("💰 Total cost:    ", style="bold")
        summary.append(self._format_cost(self._total_cost), style=f"bold {cost_color}")
        summary.append("\n")

        # Time
        latency_color = self._get_latency_color(total_time_ms)
        summary.append("⏱️  Total time:    ", style="bold")
        summary.append(self._format_latency(total_time_ms), style=f"bold {latency_color}")
        summary.append("\n")

        # LLM calls
        summary.append("🔄 LLM calls:     ", style="bold")
        summary.append(str(self._llm_calls), style="bold")
        if self._total_tokens > 0:
            summary.append(f" ({self._total_tokens:,} tokens)", style="dim")
        summary.append("\n")

        # Tool calls (only if any)
        if self._tool_calls > 0:
            summary.append("🔧 Tool calls:    ", style="bold")
            summary.append(str(self._tool_calls), style="bold")
            summary.append("\n")

        self.console.print(summary)

        # Slowest span
        if self._spans_by_latency:
            slowest = max(self._spans_by_latency, key=lambda x: x[1])
            if slowest[1] > 0:
                self.console.print(
                    f"[dim]Slowest:[/dim] {slowest[0]} ({self._format_latency(slowest[1])})"
                )

        # Most expensive span
        if self._spans_by_cost:
            expensive = max(self._spans_by_cost, key=lambda x: x[1])
            if expensive[1] > 0:
                self.console.print(
                    f"[dim]Most expensive:[/dim] {expensive[0]} ({self._format_cost(expensive[1])})"
                )

        self.console.print()

        # Write closing record to JSONL
        self._write_jsonl({
            "type": "trace_end",
            "trace_id": trace_id,
            "ended_at": end_time,
            "total_cost_usd": self._total_cost,
            "total_tokens": self._total_tokens,
            "total_llm_calls": self._llm_calls,
            "total_tool_calls": self._tool_calls,
            "total_latency_ms": total_time_ms,
        })

    def report_from_execution_trace(
        self,
        execution_trace: ExecutionTrace,
        test_name: str,
    ) -> None:
        """Generate trace report from a completed ExecutionTrace.

        This is the main entry point for post-execution reporting.

        Args:
            execution_trace: The completed execution trace
            test_name: Name of the test for display
        """
        trace_id = execution_trace.session_id
        self.trace_started(trace_id, test_name)

        # Prefer detailed trace context if available
        if execution_trace.trace_context:
            for span in execution_trace.trace_context.spans:
                # Skip root agent span (already shown as header)
                if span.kind == SpanKind.AGENT and span.parent_span_id is None:
                    continue
                self.span_completed(span)
            self.trace_completed(execution_trace.trace_context)
        else:
            # Fall back to legacy step traces
            for step in execution_trace.steps:
                span = Span(
                    span_id=step.step_id[:8] if len(step.step_id) >= 8 else step.step_id,
                    parent_span_id="root",
                    trace_id=trace_id,
                    kind=SpanKind.TOOL,
                    name=step.tool_name,
                    start_time=execution_trace.start_time,
                    end_time=execution_trace.end_time,
                    duration_ms=step.metrics.latency if step.metrics else 0.0,
                    status="ok" if step.success else "error",
                    error_message=step.error,
                    cost=step.metrics.cost if step.metrics else 0.0,
                )
                self.span_completed(span)

            # Override totals from execution metrics for accuracy
            self._total_cost = execution_trace.metrics.total_cost
            self.trace_completed(None)


def create_trace_reporter(
    console: Optional[Console] = None,
    trace_out_path: Optional[str] = None,
) -> LiveTraceReporter:
    """Factory function to create a configured trace reporter.

    The returned reporter owns the file handle and should be closed
    when done, either by calling close() or using as a context manager.

    Args:
        console: Rich console for output (uses default if None)
        trace_out_path: Path for JSONL export (None = no export)

    Returns:
        Configured LiveTraceReporter instance

    Example:
        with create_trace_reporter(trace_out_path="out.jsonl") as reporter:
            reporter.report_from_execution_trace(trace, "test_name")
    """
    jsonl_file: Optional[TextIO] = None

    if trace_out_path:
        path = Path(trace_out_path)
        # Ensure parent directory exists
        path.parent.mkdir(parents=True, exist_ok=True)
        jsonl_file = open(path, "a", encoding="utf-8")

    reporter = LiveTraceReporter(
        console=console,
        jsonl_output=jsonl_file,
    )
    reporter._owns_file = jsonl_file is not None

    return reporter


@contextmanager
def trace_reporter_context(
    console: Optional[Console] = None,
    trace_out_path: Optional[str] = None,
) -> Iterator[LiveTraceReporter]:
    """Context manager for trace reporter with automatic cleanup.

    Args:
        console: Rich console for output
        trace_out_path: Path for JSONL export

    Yields:
        Configured LiveTraceReporter instance

    Example:
        with trace_reporter_context(trace_out_path="out.jsonl") as reporter:
            reporter.report_from_execution_trace(trace, "test_name")
    """
    reporter = create_trace_reporter(console=console, trace_out_path=trace_out_path)
    try:
        yield reporter
    finally:
        reporter.close()
