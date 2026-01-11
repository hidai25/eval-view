"""Tracing instrumentation for EvalView.

Provides a Tracer class and context managers for capturing LLM calls,
tool executions, and building hierarchical trace contexts.

Example usage:
    tracer = Tracer()
    with tracer.start_span("Agent", SpanKind.AGENT) as agent_span:
        with tracer.start_span("Claude", SpanKind.LLM) as llm_span:
            response = await client.messages.create(...)
            llm_span.llm = LLMCallInfo(
                model="claude-sonnet-4-5-20250929",
                provider="anthropic",
                prompt_tokens=response.usage.input_tokens,
                completion_tokens=response.usage.output_tokens,
            )
        with tracer.start_span("get_weather", SpanKind.TOOL) as tool_span:
            result = execute_tool("get_weather", {"city": "NYC"})
            tool_span.tool = ToolCallInfo(
                tool_name="get_weather",
                parameters={"city": "NYC"},
                result=result,
            )
    trace_context = tracer.build_trace_context()
"""

import uuid
import contextvars
from datetime import datetime
from typing import Optional, List, Any, Dict

__all__ = [
    "Tracer",
    "SpanContext",
    "AsyncSpanContext",
    "steps_to_trace_context",
    "generate_trace_id",
    "generate_span_id",
]

from evalview.core.types import (
    Span,
    SpanKind,
    TraceContext,
    LLMCallInfo,
    ToolCallInfo,
    StepTrace,
)


# Context variable for tracking current span in async contexts
_current_span: contextvars.ContextVar[Optional[Span]] = contextvars.ContextVar(
    "_current_span", default=None
)


def generate_trace_id() -> str:
    """Generate a unique trace ID."""
    return uuid.uuid4().hex[:16]


def generate_span_id() -> str:
    """Generate a unique span ID."""
    return uuid.uuid4().hex[:8]


class SpanContext:
    """Context manager for span lifecycle management.

    Automatically tracks start/end times and handles parent-child
    relationships in the span hierarchy.
    """

    def __init__(self, tracer: "Tracer", span: Span):
        self.tracer = tracer
        self.span = span
        self._token: Optional[contextvars.Token] = None

    def __enter__(self) -> Span:
        """Enter the span context and set as current span."""
        self._token = _current_span.set(self.span)
        return self.span

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit the span context, recording end time and status."""
        self.span.end_time = datetime.now()
        self.span.duration_ms = (
            self.span.end_time - self.span.start_time
        ).total_seconds() * 1000

        if exc_type is not None:
            self.span.status = "error"
            self.span.error_message = str(exc_val) if exc_val else str(exc_type)
        else:
            self.span.status = "ok"

        # Restore previous span context
        if self._token is not None:
            _current_span.reset(self._token)

        return False  # Don't suppress exceptions


class AsyncSpanContext:
    """Async context manager for span lifecycle management."""

    def __init__(self, tracer: "Tracer", span: Span):
        self.tracer = tracer
        self.span = span
        self._token: Optional[contextvars.Token] = None

    async def __aenter__(self) -> Span:
        """Enter the span context and set as current span."""
        self._token = _current_span.set(self.span)
        return self.span

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Exit the span context, recording end time and status."""
        self.span.end_time = datetime.now()
        self.span.duration_ms = (
            self.span.end_time - self.span.start_time
        ).total_seconds() * 1000

        if exc_type is not None:
            self.span.status = "error"
            self.span.error_message = str(exc_val) if exc_val else str(exc_type)
        else:
            self.span.status = "ok"

        # Restore previous span context
        if self._token is not None:
            _current_span.reset(self._token)

        return False  # Don't suppress exceptions


class Tracer:
    """Tracer for capturing execution spans.

    Creates and manages spans for LLM calls, tool executions, and
    agent-level operations. Builds a hierarchical trace context
    for visualization and debugging.
    """

    def __init__(self, trace_id: Optional[str] = None):
        """Initialize a new tracer.

        Args:
            trace_id: Optional trace ID. If not provided, generates a new one.
        """
        self.trace_id = trace_id or generate_trace_id()
        self.spans: List[Span] = []
        self.root_span_id: Optional[str] = None
        self.start_time: Optional[datetime] = None

    def start_span(
        self,
        name: str,
        kind: SpanKind,
        parent_span_id: Optional[str] = None,
    ) -> SpanContext:
        """Start a new span.

        Args:
            name: Name of the span (e.g., model name, tool name)
            kind: Type of span (AGENT, LLM, TOOL)
            parent_span_id: Optional explicit parent span ID.
                           If not provided, uses current span from context.

        Returns:
            SpanContext that can be used as a context manager.
        """
        # Get parent from context if not explicitly provided
        if parent_span_id is None:
            current = _current_span.get()
            if current is not None:
                parent_span_id = current.span_id

        span_id = generate_span_id()
        now = datetime.now()

        # Track root span and start time
        if self.root_span_id is None:
            self.root_span_id = span_id
            self.start_time = now

        span = Span(
            span_id=span_id,
            parent_span_id=parent_span_id,
            trace_id=self.trace_id,
            kind=kind,
            name=name,
            start_time=now,
        )

        self.spans.append(span)
        return SpanContext(self, span)

    def start_span_async(
        self,
        name: str,
        kind: SpanKind,
        parent_span_id: Optional[str] = None,
    ) -> AsyncSpanContext:
        """Start a new span for async context.

        Args:
            name: Name of the span (e.g., model name, tool name)
            kind: Type of span (AGENT, LLM, TOOL)
            parent_span_id: Optional explicit parent span ID.

        Returns:
            AsyncSpanContext that can be used as an async context manager.
        """
        # Get parent from context if not explicitly provided
        if parent_span_id is None:
            current = _current_span.get()
            if current is not None:
                parent_span_id = current.span_id

        span_id = generate_span_id()
        now = datetime.now()

        # Track root span and start time
        if self.root_span_id is None:
            self.root_span_id = span_id
            self.start_time = now

        span = Span(
            span_id=span_id,
            parent_span_id=parent_span_id,
            trace_id=self.trace_id,
            kind=kind,
            name=name,
            start_time=now,
        )

        self.spans.append(span)
        return AsyncSpanContext(self, span)

    def record_llm_call(
        self,
        model: str,
        provider: str,
        prompt: Optional[str] = None,
        completion: Optional[str] = None,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        finish_reason: Optional[str] = None,
        cost: float = 0.0,
        duration_ms: Optional[float] = None,
    ) -> Span:
        """Record an LLM call as a span.

        Convenience method for recording LLM calls without using context manager.

        Args:
            model: Model identifier (e.g., "claude-sonnet-4-5-20250929")
            provider: Provider name (e.g., "anthropic", "openai")
            prompt: Optional prompt text
            completion: Optional completion text
            prompt_tokens: Number of input tokens
            completion_tokens: Number of output tokens
            finish_reason: Reason for completion (e.g., "end_turn", "tool_use")
            cost: Cost of this LLM call
            duration_ms: Duration in milliseconds

        Returns:
            The created Span.
        """
        current = _current_span.get()
        parent_id = current.span_id if current else None

        span_id = generate_span_id()
        now = datetime.now()

        if self.root_span_id is None:
            self.root_span_id = span_id
            self.start_time = now

        span = Span(
            span_id=span_id,
            parent_span_id=parent_id,
            trace_id=self.trace_id,
            kind=SpanKind.LLM,
            name=model,
            start_time=now,
            end_time=now,
            duration_ms=duration_ms or 0.0,
            status="ok",
            cost=cost,
            llm=LLMCallInfo(
                model=model,
                provider=provider,
                prompt=prompt,
                completion=completion,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                finish_reason=finish_reason,
            ),
        )

        self.spans.append(span)
        return span

    def record_tool_call(
        self,
        tool_name: str,
        parameters: Dict[str, Any],
        result: Optional[Any] = None,
        error: Optional[str] = None,
        duration_ms: Optional[float] = None,
    ) -> Span:
        """Record a tool call as a span.

        Convenience method for recording tool calls without using context manager.

        Args:
            tool_name: Name of the tool
            parameters: Tool input parameters
            result: Tool execution result
            error: Error message if tool failed
            duration_ms: Duration in milliseconds

        Returns:
            The created Span.
        """
        current = _current_span.get()
        parent_id = current.span_id if current else None

        span_id = generate_span_id()
        now = datetime.now()

        if self.root_span_id is None:
            self.root_span_id = span_id
            self.start_time = now

        span = Span(
            span_id=span_id,
            parent_span_id=parent_id,
            trace_id=self.trace_id,
            kind=SpanKind.TOOL,
            name=tool_name,
            start_time=now,
            end_time=now,
            duration_ms=duration_ms or 0.0,
            status="error" if error else "ok",
            error_message=error,
            tool=ToolCallInfo(
                tool_name=tool_name,
                parameters=parameters,
                result=result,
            ),
        )

        self.spans.append(span)
        return span

    def build_trace_context(self) -> TraceContext:
        """Build the complete trace context from recorded spans.

        Returns:
            TraceContext with all spans, statistics, and metadata.
        """
        now = datetime.now()

        # Calculate statistics
        total_llm_calls = sum(1 for s in self.spans if s.kind == SpanKind.LLM)
        total_tool_calls = sum(1 for s in self.spans if s.kind == SpanKind.TOOL)
        total_cost = sum(s.cost for s in self.spans)

        return TraceContext(
            trace_id=self.trace_id,
            root_span_id=self.root_span_id or generate_span_id(),
            spans=self.spans,
            start_time=self.start_time or now,
            end_time=now,
            total_llm_calls=total_llm_calls,
            total_tool_calls=total_tool_calls,
            total_cost=total_cost,
        )


def steps_to_trace_context(
    steps: List[StepTrace],
    session_id: str,
    start_time: datetime,
    end_time: datetime,
) -> TraceContext:
    """Convert legacy StepTrace list to TraceContext.

    This helper enables backward compatibility by converting existing
    step traces (tool calls only) into the new TraceContext format.
    LLM call information is not available in legacy format.

    Args:
        steps: List of StepTrace from ExecutionTrace
        session_id: Session ID to use as trace ID
        start_time: Execution start time
        end_time: Execution end time

    Returns:
        TraceContext populated from the step traces.
    """
    trace_id = session_id.replace("-", "")[:16] or generate_trace_id()

    # Create root agent span
    root_span_id = generate_span_id()
    root_span = Span(
        span_id=root_span_id,
        parent_span_id=None,
        trace_id=trace_id,
        kind=SpanKind.AGENT,
        name="Agent Execution",
        start_time=start_time,
        end_time=end_time,
        duration_ms=(end_time - start_time).total_seconds() * 1000,
        status="ok",
    )

    spans = [root_span]
    total_cost = 0.0

    # Convert each step to a tool span
    for step in steps:
        step_cost = step.metrics.cost if step.metrics else 0.0
        total_cost += step_cost

        # Calculate approximate start/end times based on latency
        step_duration_ms = step.metrics.latency if step.metrics else 0.0

        span = Span(
            span_id=step.step_id[:8] if len(step.step_id) >= 8 else generate_span_id(),
            parent_span_id=root_span_id,
            trace_id=trace_id,
            kind=SpanKind.TOOL,
            name=step.tool_name,
            start_time=start_time,  # Approximate - we don't have exact timestamps
            end_time=start_time,
            duration_ms=step_duration_ms,
            status="ok" if step.success else "error",
            error_message=step.error,
            cost=step_cost,
            tool=ToolCallInfo(
                tool_name=step.tool_name,
                parameters=step.parameters,
                result=step.output,
            ),
        )
        spans.append(span)

    return TraceContext(
        trace_id=trace_id,
        root_span_id=root_span_id,
        spans=spans,
        start_time=start_time,
        end_time=end_time,
        total_llm_calls=0,  # Not available in legacy format
        total_tool_calls=len(steps),
        total_cost=total_cost,
    )
