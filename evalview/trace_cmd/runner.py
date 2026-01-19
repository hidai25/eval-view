"""Subprocess runner for traced execution.

Launches a Python script with automatic SDK instrumentation by injecting
a bootstrap module via PYTHONPATH.
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from rich.console import Console
from rich.text import Text

__all__ = ["run_traced_command"]

# Bootstrap code injected into the subprocess
BOOTSTRAP_CODE = '''
"""EvalView trace bootstrap - patches SDKs before user code runs."""
import os
import sys
import atexit

def _evalview_init():
    # Only run if trace output is configured
    if not os.environ.get("EVALVIEW_TRACE_OUTPUT"):
        return

    # Add evalview to path if needed
    evalview_path = os.environ.get("EVALVIEW_PACKAGE_PATH")
    if evalview_path and evalview_path not in sys.path:
        sys.path.insert(0, evalview_path)

    try:
        from evalview.trace_cmd.patcher import patch_sdks
        from evalview.trace_cmd.collector import close_collector

        # Patch SDKs
        patched = patch_sdks()
        if patched:
            print(f"[evalview] Instrumented: {', '.join(patched)}", file=sys.stderr)

        # Register cleanup
        atexit.register(close_collector)

    except Exception as e:
        print(f"[evalview] Warning: Instrumentation failed: {e}", file=sys.stderr)

_evalview_init()
'''


def _format_tokens(tokens: int) -> str:
    """Format token count with commas."""
    return f"{tokens:,}"


def _format_cost(cost: float) -> str:
    """Format cost for display."""
    if cost == 0:
        return "$0.00"
    elif cost < 0.01:
        return f"${cost:.4f}"
    return f"${cost:.2f}"


def _format_duration(ms: float) -> str:
    """Format duration for display."""
    if ms < 1000:
        return f"{ms:.0f}ms"
    return f"{ms / 1000:.1f}s"


def _parse_trace_file(trace_file: Path) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Parse a trace file and return spans and summary.

    Args:
        trace_file: Path to the JSONL trace file

    Returns:
        Tuple of (spans list, summary dict)
    """
    spans: List[Dict[str, Any]] = []
    summary: Dict[str, Any] = {}

    if not trace_file.exists():
        return spans, summary

    with open(trace_file, encoding="utf-8") as f:
        for line in f:
            try:
                record = json.loads(line)
                if record.get("type") == "span":
                    spans.append(record)
                elif record.get("type") == "trace_end":
                    summary = record
            except json.JSONDecodeError:
                continue

    return spans, summary


def _print_summary(
    console: Console,
    spans: List[Dict[str, Any]],
    summary: Dict[str, Any],
) -> None:
    """Print trace summary from parsed data."""
    # Calculate stats from spans
    total_calls = 0
    total_input_tokens = 0
    total_output_tokens = 0
    total_cost = 0.0
    total_time_ms = 0.0
    by_model: Dict[str, Dict[str, Any]] = {}
    all_calls: List[Dict[str, Any]] = []
    errors = 0

    for span in spans:
        if span.get("span_type") == "llm":
            total_calls += 1
            input_tokens = span.get("input_tokens", 0)
            output_tokens = span.get("output_tokens", 0)
            cost = span.get("cost_usd", 0.0)
            duration = span.get("duration_ms", 0.0)

            total_input_tokens += input_tokens
            total_output_tokens += output_tokens
            total_cost += cost
            total_time_ms += duration

            model = span.get("model", "unknown")
            if model not in by_model:
                by_model[model] = {"calls": 0, "cost": 0.0, "tokens": 0}
            by_model[model]["calls"] += 1
            by_model[model]["cost"] += cost
            by_model[model]["tokens"] += input_tokens + output_tokens

            all_calls.append({
                "model": model,
                "duration_ms": duration,
                "cost": cost,
            })

            if span.get("status") == "error":
                errors += 1

    # Use trace_end total time if available (more accurate)
    if summary.get("total_time_ms"):
        total_time_ms = summary["total_time_ms"]

    if total_calls == 0:
        console.print("[dim]No LLM calls captured.[/dim]")
        return

    total_tokens = total_input_tokens + total_output_tokens

    # Print summary header
    console.print()
    console.print("[bold cyan]━━━ Trace Summary ━━━[/bold cyan]")

    summary_text = Text()
    summary_text.append("Total LLM calls:  ", style="bold")
    summary_text.append(str(total_calls), style="bold")
    if errors > 0:
        summary_text.append(f" ({errors} errors)", style="red")
    summary_text.append("\n")

    summary_text.append("Total tokens:     ", style="bold")
    summary_text.append(_format_tokens(total_tokens), style="bold")
    summary_text.append(f" (in: {_format_tokens(total_input_tokens)} / out: {_format_tokens(total_output_tokens)})", style="dim")
    summary_text.append("\n")

    summary_text.append("Total cost:       ", style="bold")
    cost_color = "green" if total_cost < 0.10 else "yellow" if total_cost < 1.0 else "red"
    summary_text.append(_format_cost(total_cost), style=f"bold {cost_color}")
    summary_text.append("\n")

    summary_text.append("Total time:       ", style="bold")
    summary_text.append(_format_duration(total_time_ms), style="bold")

    console.print(summary_text)

    # Slowest calls section
    if len(all_calls) > 1:
        sorted_by_duration = sorted(all_calls, key=lambda x: x["duration_ms"], reverse=True)
        top_slowest = sorted_by_duration[:3]

        console.print()
        console.print("[dim]Slowest calls:[/dim]")
        for i, call in enumerate(top_slowest, 1):
            console.print(f"  {i}. {call['model']:<20} {_format_duration(call['duration_ms'])}")

    # Most expensive by model
    if by_model:
        console.print()
        console.print("[dim]Most expensive:[/dim]")
        sorted_models = sorted(by_model.items(), key=lambda x: x[1]["cost"], reverse=True)
        for model, stats in sorted_models[:5]:
            cost_str = _format_cost(stats["cost"])
            calls_str = f"({stats['calls']} call{'s' if stats['calls'] != 1 else ''})"
            console.print(f"  {model:<20} {cost_str} {calls_str}")

    console.print()


def _save_to_sqlite(
    spans: List[Dict[str, Any]],
    summary: Dict[str, Any],
    script_name: Optional[str] = None,
) -> Optional[str]:
    """Save trace data to SQLite database.

    Args:
        spans: List of span records
        summary: Summary record from trace_end
        script_name: Name of the script being traced

    Returns:
        The run_id if saved, None on error
    """
    try:
        from evalview.storage import TraceDB

        with TraceDB() as db:
            run_id = db.save_trace(
                source="trace_cmd",
                script_name=script_name,
                spans=spans,
                summary=summary,
            )
            return run_id
    except Exception as e:
        # Log but don't break - trace output is more important than persistence
        import sys
        print(f"[evalview] Warning: Failed to save trace to database: {e}", file=sys.stderr)
        return None


def run_traced_command(
    command: List[str],
    output_path: Optional[str] = None,
    console: Optional[Console] = None,
    save_to_db: bool = True,
) -> Tuple[int, Optional[Path]]:
    """Run a command with automatic SDK instrumentation.

    Args:
        command: Command and arguments to run (e.g., ["python", "script.py"])
        output_path: Optional path for trace output. Auto-generates if None.
        console: Rich console for output
        save_to_db: Whether to save trace to SQLite database

    Returns:
        Tuple of (exit_code, trace_file_path)
    """
    console = console or Console()

    # Extract script name from command
    script_name = None
    for arg in command:
        if arg.endswith(".py"):
            script_name = Path(arg).name
            break

    # Create temp file for trace output
    if output_path:
        trace_file = Path(output_path)
        trace_file.parent.mkdir(parents=True, exist_ok=True)
    else:
        fd, temp_path = tempfile.mkstemp(suffix=".jsonl", prefix="evalview_trace_")
        os.close(fd)
        trace_file = Path(temp_path)

    # Create bootstrap file
    fd, bootstrap_path = tempfile.mkstemp(suffix=".py", prefix="evalview_bootstrap_")
    os.close(fd)
    with open(bootstrap_path, "w", encoding="utf-8") as f:
        f.write(BOOTSTRAP_CODE)

    # Get the evalview package path
    import evalview
    evalview_path = str(Path(evalview.__file__).parent.parent)

    # Set up environment
    env = os.environ.copy()
    env["EVALVIEW_TRACE_OUTPUT"] = str(trace_file)
    env["EVALVIEW_PACKAGE_PATH"] = evalview_path

    # Prepend bootstrap directory to PYTHONPATH
    bootstrap_dir = str(Path(bootstrap_path).parent)
    existing_pythonpath = env.get("PYTHONPATH", "")
    if existing_pythonpath:
        env["PYTHONPATH"] = os.pathsep.join([bootstrap_dir, evalview_path, existing_pythonpath])
    else:
        env["PYTHONPATH"] = os.pathsep.join([bootstrap_dir, evalview_path])

    # Create sitecustomize.py in the bootstrap directory to auto-run
    sitecustomize_path = Path(bootstrap_dir) / "sitecustomize.py"
    sitecustomize_existed = sitecustomize_path.exists()
    if not sitecustomize_existed:
        with open(sitecustomize_path, "w", encoding="utf-8") as f:
            f.write(BOOTSTRAP_CODE)

    run_id: Optional[str] = None

    try:
        # Print header
        console.print("[bold cyan]━━━ EvalView Trace ━━━[/bold cyan]")
        console.print(f"[dim]Running: {' '.join(command)}[/dim]")
        console.print()

        # Run the command
        result = subprocess.run(command, env=env)

        # Parse trace file
        spans, summary = _parse_trace_file(trace_file)

        # Print summary
        _print_summary(console, spans, summary)

        # Save to SQLite
        if save_to_db and spans:
            run_id = _save_to_sqlite(spans, summary, script_name)
            if run_id:
                console.print(f"[dim]Trace ID: {run_id}[/dim]")

        if output_path:
            console.print(f"[dim]Trace saved to: {trace_file}[/dim]")

        return result.returncode, trace_file

    finally:
        # Cleanup bootstrap files
        try:
            os.unlink(bootstrap_path)
            if not sitecustomize_existed and sitecustomize_path.exists():
                os.unlink(sitecustomize_path)
        except OSError:
            pass

        # Cleanup temp trace file if no output path specified
        if not output_path and trace_file.exists():
            try:
                os.unlink(trace_file)
            except OSError:
                pass
