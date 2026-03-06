"""Traces commands — query and manage local trace storage."""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import click

from evalview.commands.shared import console
from evalview.telemetry.decorators import track_command


@click.group()
def traces():
    """Query and manage local trace storage.

    \b
    Examples:
        evalview traces list              # List recent traces
        evalview traces list --last-24h   # Last 24 hours
        evalview traces show abc123       # Show specific trace
        evalview traces export abc123     # Export trace to HTML
        evalview traces cost-report       # Cost report for last 7 days
    """
    pass


@traces.command("list")
@click.option("--last-24h", "last_24h", is_flag=True, help="Show traces from last 24 hours")
@click.option("--last-7d", "last_7d", is_flag=True, help="Show traces from last 7 days")
@click.option("--source", type=click.Choice(["eval", "trace_cmd"]), help="Filter by source")
@click.option("--limit", "-n", default=20, help="Max traces to show (default: 20)")
@track_command("traces_list")
def traces_list(last_24h: bool, last_7d: bool, source: Optional[str], limit: int):
    """List recent traces."""
    from evalview.storage import TraceDB

    try:
        with TraceDB() as db:
            last_hours = 24 if last_24h else None
            last_days = 7 if last_7d else None

            traces_data = db.list_traces(
                last_hours=last_hours,
                last_days=last_days,
                source=source,
                limit=limit,
            )

            if not traces_data:
                console.print("[dim]No traces found.[/dim]")
                console.print("[dim]Run 'evalview trace <script.py>' to capture traces.[/dim]")
                return

            console.print("[bold cyan]━━━ Recent Traces ━━━[/bold cyan]")
            console.print()

            for trace in traces_data:
                # Parse timestamp
                created = trace["created_at"][:16].replace("T", " ")

                # Format cost
                cost = trace.get("total_cost", 0)
                if cost == 0:
                    cost_str = "$0.00"
                elif cost < 0.01:
                    cost_str = f"${cost:.4f}"
                else:
                    cost_str = f"${cost:.2f}"

                # Format source
                src = trace.get("source", "unknown")
                src_icon = "📊" if src == "eval" else "🔍"

                # Script name
                script = trace.get("script_name") or "-"

                console.print(
                    f"[bold]{trace['run_id']}[/bold]  {src_icon} {created}  "
                    f"{trace.get('total_calls', 0)} calls  {cost_str}  [dim]{script}[/dim]"
                )

            console.print()
            console.print(f"[dim]Showing {len(traces_data)} traces. Use --limit to see more.[/dim]")

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        sys.exit(1)


@traces.command("show")
@click.argument("trace_id")
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
@track_command("traces_show")
def traces_show(trace_id: str, as_json: bool):
    """Show details of a specific trace."""
    import json as json_module
    from evalview.storage import TraceDB

    try:
        with TraceDB() as db:
            trace = db.get_trace(trace_id)

            if not trace:
                console.print(f"[red]Trace not found: {trace_id}[/red]")
                sys.exit(1)

            spans = db.get_trace_spans(trace_id)

            if as_json:
                output = {
                    "trace": trace,
                    "spans": spans,
                }
                console.print(json_module.dumps(output, indent=2, default=str))
                return

            # Pretty print
            console.print("[bold cyan]━━━ Trace Details ━━━[/bold cyan]")
            console.print()

            console.print(f"[bold]Trace ID:[/bold]     {trace['run_id']}")
            console.print(f"[bold]Created:[/bold]      {trace['created_at'][:19].replace('T', ' ')}")
            console.print(f"[bold]Source:[/bold]       {trace.get('source', 'unknown')}")
            if trace.get("script_name"):
                console.print(f"[bold]Script:[/bold]       {trace['script_name']}")
            console.print()

            # Stats
            console.print("[bold]Summary:[/bold]")
            console.print(f"  Total calls:    {trace.get('total_calls', 0)}")
            tokens = trace.get("total_tokens", 0)
            in_tokens = trace.get("total_input_tokens", 0)
            out_tokens = trace.get("total_output_tokens", 0)
            console.print(f"  Total tokens:   {tokens:,} (in: {in_tokens:,} / out: {out_tokens:,})")

            cost = trace.get("total_cost", 0)
            cost_str = f"${cost:.4f}" if cost < 0.01 and cost > 0 else f"${cost:.2f}"
            console.print(f"  Total cost:     {cost_str}")

            latency = trace.get("total_latency_ms", 0)
            if latency < 1000:
                latency_str = f"{latency:.0f}ms"
            else:
                latency_str = f"{latency/1000:.1f}s"
            console.print(f"  Total time:     {latency_str}")
            console.print()

            # Spans
            if spans:
                console.print("[bold]LLM Calls:[/bold]")
                for i, span in enumerate(spans, 1):
                    if span.get("span_type") == "llm":
                        model = span.get("model", "unknown")
                        duration = span.get("duration_ms", 0)
                        dur_str = f"{duration:.0f}ms" if duration < 1000 else f"{duration/1000:.1f}s"
                        span_cost = span.get("cost_usd", 0)
                        span_cost_str = f"${span_cost:.4f}" if span_cost < 0.01 and span_cost > 0 else f"${span_cost:.2f}"
                        status = span.get("status", "success")
                        status_icon = "✓" if status == "success" else "✗"

                        console.print(
                            f"  {i}. {status_icon} {model:<25} {dur_str:>8}  {span_cost_str}"
                        )

            console.print()

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        sys.exit(1)


@traces.command("cost-report")
@click.option("--last-7d", "last_7d", is_flag=True, default=True, help="Report for last 7 days (default)")
@click.option("--last-30d", "last_30d", is_flag=True, help="Report for last 30 days")
@click.option("--by-model", "by_model", is_flag=True, help="Show breakdown by model")
@track_command("traces_cost_report")
def traces_cost_report(last_7d: bool, last_30d: bool, by_model: bool):
    """Show cost report for recent traces."""
    from evalview.storage import TraceDB

    try:
        with TraceDB() as db:
            days = 30 if last_30d else 7
            report = db.get_cost_report(last_days=days)

            totals = report["totals"]
            total_cost = totals.get("total_cost") or 0
            total_calls = totals.get("total_calls") or 0

            console.print(f"[bold cyan]━━━ Cost Report (Last {days} Days) ━━━[/bold cyan]")
            console.print()

            # Format total cost
            if total_cost == 0:
                cost_str = "$0.00"
            elif total_cost < 0.01:
                cost_str = f"${total_cost:.4f}"
            else:
                cost_str = f"${total_cost:.2f}"

            console.print(f"[bold]Total:[/bold]     {cost_str} across {total_calls:,} LLM calls")
            console.print()

            # By model breakdown
            models = report.get("by_model", [])
            if models:
                console.print("[bold]By Model:[/bold]")
                max_cost = max((m.get("total_cost") or 0) for m in models) if models else 1

                for m in models[:10]:
                    model_name = m.get("model") or "unknown"
                    model_cost = m.get("total_cost") or 0

                    # Calculate percentage
                    pct = (model_cost / total_cost * 100) if total_cost > 0 else 0

                    # Format cost
                    if model_cost == 0:
                        mc_str = "$0.00"
                    elif model_cost < 0.01:
                        mc_str = f"${model_cost:.4f}"
                    else:
                        mc_str = f"${model_cost:.2f}"

                    # Progress bar
                    bar_width = 16
                    filled = int((model_cost / max_cost) * bar_width) if max_cost > 0 else 0
                    bar = "█" * filled + "░" * (bar_width - filled)

                    console.print(f"  {model_name:<22} {mc_str:>8}  ({pct:>4.0f}%)  {bar}")

                console.print()

            # By day breakdown
            days_data = report.get("by_day", [])
            if days_data:
                console.print("[bold]By Day:[/bold]")
                max_day_cost = max((d.get("total_cost") or 0) for d in days_data) if days_data else 1

                for d in days_data[-7:]:  # Show last 7 days max
                    day = d.get("day", "")
                    day_cost = d.get("total_cost") or 0

                    # Format cost
                    if day_cost == 0:
                        dc_str = "$0.00"
                    elif day_cost < 0.01:
                        dc_str = f"${day_cost:.4f}"
                    else:
                        dc_str = f"${day_cost:.2f}"

                    # Progress bar
                    bar_width = 8
                    filled = int((day_cost / max_day_cost) * bar_width) if max_day_cost > 0 else 0
                    bar = "█" * filled + "░" * (bar_width - filled)

                    console.print(f"  {day}  {dc_str:>8}  {bar}")

                console.print()

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        sys.exit(1)


@traces.command("export")
@click.argument("trace_id")
@click.option("--json", "as_json", is_flag=True, help="Export as JSON instead of HTML")
@click.option("-o", "--output", "output_path", help="Output file path")
@track_command("traces_export")
def traces_export(trace_id: str, as_json: bool, output_path: Optional[str]):
    """Export a trace to HTML or JSON.

    \b
    Examples:
        evalview traces export abc123            # Export to HTML
        evalview traces export abc123 --json    # Export to JSON
        evalview traces export abc123 -o report.html
    """
    import json as json_module
    from evalview.storage import TraceDB

    try:
        with TraceDB() as db:
            trace = db.get_trace(trace_id)

            if not trace:
                console.print(f"[red]Trace not found: {trace_id}[/red]")
                sys.exit(1)

            spans = db.get_trace_spans(trace_id)

            if as_json:
                output = output_path or f"trace_{trace_id}.json"
                data = {"trace": trace, "spans": spans}
                Path(output).write_text(
                    json_module.dumps(data, indent=2, default=str),
                    encoding="utf-8",
                )
                console.print(f"[green]Exported to: {output}[/green]")
            else:
                # HTML export (default)
                try:
                    from evalview.exporters import TraceHTMLExporter
                except ImportError:
                    console.print("[red]HTML export requires jinja2. Install with:[/red]")
                    console.print("  pip install evalview[reports]")
                    sys.exit(1)

                output = output_path or f"trace_{trace_id}.html"
                exporter = TraceHTMLExporter()
                exporter.export(trace, spans, output)
                console.print(f"[green]Exported to: {output}[/green]")

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        sys.exit(1)
