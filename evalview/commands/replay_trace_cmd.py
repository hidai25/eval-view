"""Replay-trace command — replay production traces against the current agent.

Takes a JSONL file of production traces (or the output of `evalview import`),
re-executes each query against the live agent, and diffs the result against
the original production behavior.

This bridges the demo-to-prod gap: when something fails in production, you
can replay it locally and see exactly what changed.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import click

from evalview.commands.shared import console
from evalview.telemetry.decorators import track_command


@click.command("replay-trace")
@click.argument("trace_file", type=click.Path(exists=True))
@click.option(
    "--adapter", "-a",
    default=None,
    help="Adapter to use for replay (e.g. http, anthropic, openai). Uses config if omitted.",
)
@click.option(
    "--endpoint", "-e",
    default=None,
    help="Agent endpoint URL for replay.",
)
@click.option(
    "--max", "max_traces",
    default=20,
    show_default=True,
    help="Maximum number of traces to replay.",
)
@click.option(
    "--json", "json_output",
    is_flag=True,
    help="Output JSON results.",
)
@click.option(
    "--timeout",
    type=float,
    default=120.0,
    show_default=True,
    help="Timeout per replay in seconds.",
)
@track_command("replay-trace")
def replay_trace(
    trace_file: str,
    adapter: Optional[str],
    endpoint: Optional[str],
    max_traces: int,
    json_output: bool,
    timeout: float,
) -> None:
    """Replay production traces against the current agent and diff results.

    Takes a JSONL file where each line is a production trace (with at least
    a query/input field and optionally tool_calls and output). Re-executes
    each query against the live agent and shows what changed.

    \b
    Examples:
        evalview replay-trace prod-failures.jsonl
        evalview replay-trace traces.jsonl --adapter http --endpoint http://localhost:8080
        evalview replay-trace traces.jsonl --json
        evalview replay-trace traces.jsonl --max 5
    """
    import asyncio

    path = Path(trace_file)

    # Resolve adapter/endpoint from config if not provided
    from evalview.commands.shared import _load_config_if_exists
    config = _load_config_if_exists()

    effective_adapter = adapter or (config.adapter if config else None) or "http"
    effective_endpoint = endpoint or (config.endpoint if config else None)

    if not effective_endpoint and effective_adapter == "http":
        console.print(
            "[red]No endpoint configured.[/red] "
            "Pass --endpoint or set it in .evalview/config.yaml"
        )
        sys.exit(1)

    # Load traces from JSONL
    traces_data = _load_trace_file(path, max_traces)
    if not traces_data:
        console.print("[yellow]No traces found in file.[/yellow]")
        sys.exit(1)

    if not json_output:
        console.print(
            f"\n[cyan]◈ Replaying {len(traces_data)} trace(s)[/cyan] "
            f"against [bold]{effective_adapter}[/bold]"
            f"{f' @ {effective_endpoint}' if effective_endpoint else ''}\n"
        )

    # Execute replays
    from evalview.core.replay_pipeline import ReplayPipeline
    pipeline = ReplayPipeline(
        adapter_name=effective_adapter,
        endpoint=effective_endpoint,
    )

    # Build ExecutionTrace objects from the JSONL data
    from evalview.core.types import (
        ExecutionTrace, ExecutionMetrics, StepTrace, StepMetrics,
    )
    from datetime import datetime

    original_traces = []
    queries = []
    for td in traces_data:
        query = (
            td.get("query")
            or td.get("input")
            or td.get("prompt")
            or td.get("user_message")
            or td.get("question")
            or ""
        )
        queries.append(query)

        # Build a minimal ExecutionTrace from the production data
        tool_calls = td.get("tool_calls") or td.get("tools") or []
        output = td.get("output") or td.get("response") or td.get("result") or ""

        steps = []
        for i, tool in enumerate(tool_calls):
            tool_name = tool if isinstance(tool, str) else tool.get("name", f"tool_{i}")
            params = tool.get("parameters", {}) if isinstance(tool, dict) else {}
            steps.append(StepTrace(
                step_id=f"prod-{i}",
                step_name=tool_name,
                tool_name=tool_name,
                parameters=params,
                output="",
                success=True,
                metrics=StepMetrics(latency=0, cost=0),
            ))

        trace = ExecutionTrace(
            session_id=td.get("session_id", f"prod-{len(original_traces)}"),
            start_time=datetime.now(),
            end_time=datetime.now(),
            steps=steps,
            final_output=output if isinstance(output, str) else str(output),
            metrics=ExecutionMetrics(total_cost=0, total_latency=0),
        )
        original_traces.append(trace)

    # Run the replay pipeline
    try:
        batch_result = asyncio.run(
            pipeline.replay_batch(original_traces, queries=queries)
        )
    except Exception as e:
        console.print(f"[red]Replay failed: {e}[/red]")
        sys.exit(1)

    # Display results
    if json_output:
        output_data = {
            "summary": batch_result.summary(),
            "total": batch_result.total,
            "succeeded": batch_result.succeeded,
            "failed": batch_result.failed,
            "changed": batch_result.changed,
            "stable": batch_result.stable,
            "results": [
                {
                    "query": queries[i][:200] if i < len(queries) else "",
                    "status": r.status,
                    "summary": r.summary(),
                    "has_differences": r.has_differences,
                    "error": r.error,
                }
                for i, r in enumerate(batch_result.results)
            ],
        }
        print(json.dumps(output_data, indent=2))
    else:
        from evalview.core.diff import DiffStatus

        for i, r in enumerate(batch_result.results):
            query_preview = queries[i][:60] if i < len(queries) else "?"
            if r.error:
                console.print(f"  [red]✗ ERROR[/red]: {query_preview}")
                console.print(f"    [dim]{r.error}[/dim]")
            elif not r.has_differences:
                console.print(f"  [green]✓ STABLE[/green]: {query_preview}")
            else:
                status = r.status
                icon = {
                    "regression": "[red]✗ REGRESSION[/red]",
                    "tools_changed": "[yellow]⚠ TOOLS_CHANGED[/yellow]",
                    "output_changed": "[dim]~ OUTPUT_CHANGED[/dim]",
                }.get(status, f"[yellow]⚠ {status.upper()}[/yellow]")
                console.print(f"  {icon}: {query_preview}")
                console.print(f"    [dim]{r.summary()}[/dim]")
            console.print()

        # Summary
        console.print(f"[bold]{batch_result.summary()}[/bold]\n")

    sys.exit(1 if batch_result.changed > 0 else 0)


def _load_trace_file(path: Path, max_entries: int) -> List[Dict[str, Any]]:
    """Load production traces from a JSONL file."""
    entries: List[Dict[str, Any]] = []
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                    if isinstance(entry, dict):
                        entries.append(entry)
                except json.JSONDecodeError:
                    continue
                if len(entries) >= max_entries:
                    break
    except OSError as e:
        return []
    return entries
