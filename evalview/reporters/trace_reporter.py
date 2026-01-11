"""Trace reporter for visualizing execution traces.

Provides terminal-based timeline visualization of agent execution
traces, including LLM calls, tool executions, and cost breakdowns.
"""

import json
from typing import List, Optional, Dict, Any
from rich.console import Console
from rich.panel import Panel
from rich.tree import Tree
from rich.text import Text
from rich.table import Table

__all__ = ["TraceReporter"]

from evalview.core.types import (
    TraceContext,
    Span,
    SpanKind,
    EvaluationResult,
)
from evalview.core.tracing import steps_to_trace_context


class TraceReporter:
    """Generates formatted terminal output for execution traces.

    Visualizes trace contexts with hierarchical span trees,
    LLM call details, tool executions, and performance metrics.
    """

    def __init__(self):
        self.console = Console()

    def print_trace(
        self,
        trace_context: TraceContext,
        show_prompts: bool = False,
        show_completions: bool = False,
        llm_only: bool = False,
        tools_only: bool = False,
    ) -> None:
        """Print a formatted trace visualization.

        Args:
            trace_context: The trace context to visualize
            show_prompts: Whether to show LLM prompts
            show_completions: Whether to show LLM completions
            llm_only: Only show LLM spans
            tools_only: Only show tool spans
        """
        # Header with summary
        self._print_header(trace_context)

        # Build and print the span tree
        self._print_span_tree(
            trace_context,
            show_prompts=show_prompts,
            show_completions=show_completions,
            llm_only=llm_only,
            tools_only=tools_only,
        )

    def print_trace_from_result(
        self,
        result: EvaluationResult,
        show_prompts: bool = False,
        show_completions: bool = False,
        llm_only: bool = False,
        tools_only: bool = False,
    ) -> None:
        """Print trace from an evaluation result.

        Handles both new-style trace_context and legacy step traces
        via auto-conversion.

        Args:
            result: The evaluation result containing the trace
            show_prompts: Whether to show LLM prompts
            show_completions: Whether to show LLM completions
            llm_only: Only show LLM spans
            tools_only: Only show tool spans
        """
        trace = result.trace

        # Use trace_context if available, otherwise convert from steps
        if trace.trace_context is not None:
            trace_context = trace.trace_context
        else:
            # Auto-convert legacy steps to trace context
            trace_context = steps_to_trace_context(
                steps=trace.steps,
                session_id=trace.session_id,
                start_time=trace.start_time,
                end_time=trace.end_time,
            )

        self.print_trace(
            trace_context,
            show_prompts=show_prompts,
            show_completions=show_completions,
            llm_only=llm_only,
            tools_only=tools_only,
        )

    def _print_header(self, trace_context: TraceContext) -> None:
        """Print the trace header with summary statistics."""
        duration_ms = 0.0
        if trace_context.end_time and trace_context.start_time:
            duration_ms = (
                trace_context.end_time - trace_context.start_time
            ).total_seconds() * 1000

        header_text = Text()
        header_text.append("Trace ID: ", style="bold")
        header_text.append(trace_context.trace_id, style="cyan")
        header_text.append("\n")

        header_text.append("Duration: ", style="bold")
        header_text.append(f"{duration_ms:.0f}ms", style="yellow")
        header_text.append(" | ", style="dim")

        header_text.append("LLM Calls: ", style="bold")
        header_text.append(str(trace_context.total_llm_calls), style="magenta")
        header_text.append(" | ", style="dim")

        header_text.append("Tool Calls: ", style="bold")
        header_text.append(str(trace_context.total_tool_calls), style="blue")
        header_text.append(" | ", style="dim")

        header_text.append("Cost: ", style="bold")
        header_text.append(f"${trace_context.total_cost:.4f}", style="green")

        self.console.print(Panel(header_text, border_style="cyan"))
        self.console.print()

    def _print_span_tree(
        self,
        trace_context: TraceContext,
        show_prompts: bool = False,
        show_completions: bool = False,
        llm_only: bool = False,
        tools_only: bool = False,
    ) -> None:
        """Print the hierarchical span tree."""
        # Build parent-child relationships
        children: Dict[str, List[Span]] = {}

        for span in trace_context.spans:
            parent_id = span.parent_span_id or "root"
            if parent_id not in children:
                children[parent_id] = []
            children[parent_id].append(span)

        # Find root spans (no parent or parent is "root")
        root_spans = children.get("root", [])
        if not root_spans:
            # If no explicit root, use spans without parents
            root_spans = [s for s in trace_context.spans if s.parent_span_id is None]

        if not root_spans:
            self.console.print("[dim]No spans to display[/dim]")
            return

        # Create tree for each root span
        for root_span in root_spans:
            tree = self._build_span_node(
                root_span,
                children,
                show_prompts=show_prompts,
                show_completions=show_completions,
                llm_only=llm_only,
                tools_only=tools_only,
            )
            if tree:
                self.console.print(tree)
                self.console.print()

    def _build_span_node(
        self,
        span: Span,
        children: Dict[str, List[Span]],
        show_prompts: bool = False,
        show_completions: bool = False,
        llm_only: bool = False,
        tools_only: bool = False,
    ) -> Optional[Tree]:
        """Build a tree node for a span and its children."""
        # Apply filters
        if llm_only and span.kind != SpanKind.LLM and span.kind != SpanKind.AGENT:
            return None
        if tools_only and span.kind != SpanKind.TOOL and span.kind != SpanKind.AGENT:
            return None

        # Build the span label
        label = self._format_span_label(span)
        tree = Tree(label)

        # Add span details based on type
        if span.kind == SpanKind.LLM and span.llm:
            self._add_llm_details(tree, span, show_prompts, show_completions)
        elif span.kind == SpanKind.TOOL and span.tool:
            self._add_tool_details(tree, span)

        # Add error if present
        if span.error_message:
            tree.add(f"[red]! error: {span.error_message}[/red]")

        # Recursively add children
        child_spans = children.get(span.span_id, [])
        for child in child_spans:
            child_node = self._build_span_node(
                child,
                children,
                show_prompts=show_prompts,
                show_completions=show_completions,
                llm_only=llm_only,
                tools_only=tools_only,
            )
            if child_node:
                # Add child tree content to parent
                tree.add(child_node)

        return tree

    def _format_span_label(self, span: Span) -> Text:
        """Format the label for a span node."""
        label = Text()

        # Determine style based on status
        is_error = span.status == "error"

        # Build label with icon and name based on span kind
        if span.kind == SpanKind.AGENT:
            icon_style = "red" if is_error else "cyan"
            label.append("[robot] ", style=icon_style)
            label.append(span.name, style=f"bold {icon_style}")
        elif span.kind == SpanKind.LLM:
            icon_style = "red" if is_error else "magenta"
            label.append("[brain] ", style=icon_style)
            label.append(span.name, style=f"bold {icon_style}")
        elif span.kind == SpanKind.TOOL:
            icon_style = "red" if is_error else "blue"
            label.append("[wrench] ", style=icon_style)
            label.append(span.name, style=f"bold {icon_style}")

        # Add metrics
        label.append(" [", style="dim")
        if span.duration_ms is not None:
            label.append(f"{span.duration_ms:.0f}ms", style="yellow")
        if span.cost > 0:
            label.append(f" | ${span.cost:.4f}", style="green")
        label.append("]", style="dim")

        return label

    def _add_llm_details(
        self,
        tree: Tree,
        span: Span,
        show_prompts: bool = False,
        show_completions: bool = False,
    ) -> None:
        """Add LLM-specific details to the tree node."""
        if not span.llm:
            return

        llm = span.llm

        # Token usage
        token_text = Text()
        token_text.append("Tokens: ", style="dim")
        token_text.append(f"{llm.prompt_tokens:,}", style="cyan")
        token_text.append(" in / ", style="dim")
        token_text.append(f"{llm.completion_tokens:,}", style="cyan")
        token_text.append(" out", style="dim")
        tree.add(token_text)

        # Provider and model
        if llm.provider:
            tree.add(f"[dim]Provider: {llm.provider}[/dim]")

        # Finish reason
        if llm.finish_reason:
            tree.add(f"[dim]Finish: {llm.finish_reason}[/dim]")

        # Prompt (truncated)
        if show_prompts and llm.prompt:
            prompt_preview = llm.prompt[:200] + "..." if len(llm.prompt) > 200 else llm.prompt
            tree.add(f"[dim]Prompt: {prompt_preview}[/dim]")

        # Completion (truncated)
        if show_completions and llm.completion:
            completion_preview = llm.completion[:200] + "..." if len(llm.completion) > 200 else llm.completion
            tree.add(f"[dim]Completion: {completion_preview}[/dim]")

    def _add_tool_details(self, tree: Tree, span: Span) -> None:
        """Add tool-specific details to the tree node."""
        if not span.tool:
            return

        tool = span.tool

        # Parameters
        if tool.parameters:
            params_str = self._format_value(tool.parameters, max_length=80)
            tree.add(f"[dim]-> params: {params_str}[/dim]")

        # Result
        if tool.result is not None:
            result_str = self._format_value(tool.result, max_length=80)
            tree.add(f"[dim]<- result: {result_str}[/dim]")

    def _format_value(self, value: Any, max_length: int = 60) -> str:
        """Format a value for display, truncating if needed."""
        if value is None:
            return "null"
        if isinstance(value, (dict, list)):
            text = json.dumps(value, default=str)
        else:
            text = str(value)

        if len(text) > max_length:
            return text[: max_length - 3] + "..."
        return text

    def print_trace_table(self, trace_context: TraceContext) -> None:
        """Print a table view of all spans.

        Args:
            trace_context: The trace context to display
        """
        if not trace_context.spans:
            self.console.print("[dim]No spans to display[/dim]")
            return

        table = Table(
            title="Span Details",
            show_header=True,
            header_style="bold",
        )

        table.add_column("#", style="dim", width=3)
        table.add_column("Type", width=6)
        table.add_column("Name", style="cyan")
        table.add_column("Duration", justify="right")
        table.add_column("Cost", justify="right")
        table.add_column("Tokens", justify="right")
        table.add_column("Status", justify="center")

        for i, span in enumerate(trace_context.spans, 1):
            # Type icon
            type_icons = {
                SpanKind.AGENT: "[cyan]AGENT[/cyan]",
                SpanKind.LLM: "[magenta]LLM[/magenta]",
                SpanKind.TOOL: "[blue]TOOL[/blue]",
            }
            type_str = type_icons.get(span.kind, str(span.kind.value))

            # Duration
            duration_str = f"{span.duration_ms:.0f}ms" if span.duration_ms else "-"

            # Cost
            cost_str = f"${span.cost:.4f}" if span.cost > 0 else "-"

            # Tokens
            tokens_str = "-"
            if span.llm:
                total = span.llm.prompt_tokens + span.llm.completion_tokens
                tokens_str = f"{total:,}"

            # Status
            if span.status == "error":
                status_str = "[red]x[/red]"
            elif span.status == "ok":
                status_str = "[green]v[/green]"
            else:
                status_str = "[dim]-[/dim]"

            table.add_row(
                str(i),
                type_str,
                span.name[:30] + "..." if len(span.name) > 30 else span.name,
                duration_str,
                cost_str,
                tokens_str,
                status_str,
            )

        self.console.print(table)

    def print_llm_summary(self, trace_context: TraceContext) -> None:
        """Print a summary of all LLM calls.

        Args:
            trace_context: The trace context to analyze
        """
        llm_spans = [s for s in trace_context.spans if s.kind == SpanKind.LLM]

        if not llm_spans:
            self.console.print("[dim]No LLM calls in trace[/dim]")
            return

        total_prompt_tokens = sum(s.llm.prompt_tokens for s in llm_spans if s.llm)
        total_completion_tokens = sum(s.llm.completion_tokens for s in llm_spans if s.llm)
        total_cost = sum(s.cost for s in llm_spans)
        total_duration = sum(s.duration_ms or 0 for s in llm_spans)

        # Model breakdown
        models: Dict[str, Dict[str, Any]] = {}
        for span in llm_spans:
            if span.llm:
                model = span.llm.model
                if model not in models:
                    models[model] = {
                        "calls": 0,
                        "prompt_tokens": 0,
                        "completion_tokens": 0,
                        "cost": 0.0,
                    }
                models[model]["calls"] += 1
                models[model]["prompt_tokens"] += span.llm.prompt_tokens
                models[model]["completion_tokens"] += span.llm.completion_tokens
                models[model]["cost"] += span.cost

        # Print summary
        self.console.print("[bold]LLM Call Summary[/bold]")
        self.console.print()

        summary_text = Text()
        summary_text.append("Total Calls: ", style="bold")
        summary_text.append(f"{len(llm_spans)}\n")
        summary_text.append("Total Tokens: ", style="bold")
        summary_text.append(f"{total_prompt_tokens + total_completion_tokens:,} ")
        summary_text.append(f"({total_prompt_tokens:,} in / {total_completion_tokens:,} out)\n", style="dim")
        summary_text.append("Total Cost: ", style="bold")
        summary_text.append(f"${total_cost:.4f}\n", style="green")
        summary_text.append("Total Duration: ", style="bold")
        summary_text.append(f"{total_duration:.0f}ms", style="yellow")

        self.console.print(Panel(summary_text, border_style="magenta"))

        # Model breakdown table
        if len(models) > 1:
            self.console.print()
            table = Table(title="By Model", show_header=True)
            table.add_column("Model")
            table.add_column("Calls", justify="right")
            table.add_column("Tokens", justify="right")
            table.add_column("Cost", justify="right")

            for model, stats in models.items():
                total_tokens = stats["prompt_tokens"] + stats["completion_tokens"]
                table.add_row(
                    model,
                    str(stats["calls"]),
                    f"{total_tokens:,}",
                    f"${stats['cost']:.4f}",
                )

            self.console.print(table)

    def export_json(self, trace_context: TraceContext) -> str:
        """Export trace context as JSON.

        Args:
            trace_context: The trace context to export

        Returns:
            JSON string representation of the trace
        """
        return trace_context.model_dump_json(indent=2)
