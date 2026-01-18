"""Subprocess runner for traced execution.

Launches a Python script with automatic SDK instrumentation by injecting
a bootstrap module via PYTHONPATH.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import List, Optional, Tuple

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


def _print_summary(console: Console, trace_file: Path) -> None:
    """Print trace summary from the trace file."""
    if not trace_file.exists():
        return

    # Parse trace file
    total_calls = 0
    total_tokens = 0
    total_cost = 0.0
    total_time_ms = 0.0
    by_model: dict = {}
    errors = 0

    with open(trace_file) as f:
        for line in f:
            try:
                record = json.loads(line)
                if record.get("type") == "span" and record.get("span_type") == "llm":
                    total_calls += 1
                    input_tokens = record.get("input_tokens", 0)
                    output_tokens = record.get("output_tokens", 0)
                    cost = record.get("cost_usd", 0.0)
                    duration = record.get("duration_ms", 0.0)

                    total_tokens += input_tokens + output_tokens
                    total_cost += cost
                    total_time_ms += duration

                    model = record.get("model", "unknown")
                    if model not in by_model:
                        by_model[model] = {"calls": 0, "cost": 0.0, "tokens": 0}
                    by_model[model]["calls"] += 1
                    by_model[model]["cost"] += cost
                    by_model[model]["tokens"] += input_tokens + output_tokens

                    if record.get("status") == "error":
                        errors += 1

                elif record.get("type") == "trace_end":
                    # Use the trace_end total time if available (more accurate)
                    total_time_ms = record.get("total_time_ms", total_time_ms)

            except json.JSONDecodeError:
                continue

    if total_calls == 0:
        console.print("[dim]No LLM calls captured.[/dim]")
        return

    # Print summary
    console.print()
    console.print("[bold cyan]━━━ Trace Summary ━━━[/bold cyan]")

    summary = Text()
    summary.append("Total LLM calls:  ", style="bold")
    summary.append(str(total_calls), style="bold")
    if errors > 0:
        summary.append(f" ({errors} errors)", style="red")
    summary.append("\n")

    summary.append("Total tokens:     ", style="bold")
    summary.append(_format_tokens(total_tokens), style="bold")
    summary.append("\n")

    summary.append("Total cost:       ", style="bold")
    cost_color = "green" if total_cost < 0.10 else "yellow" if total_cost < 1.0 else "red"
    summary.append(_format_cost(total_cost), style=f"bold {cost_color}")
    summary.append("\n")

    summary.append("Total time:       ", style="bold")
    summary.append(_format_duration(total_time_ms), style="bold")
    summary.append("\n")

    console.print(summary)

    # By model breakdown
    if len(by_model) > 1 or (len(by_model) == 1 and list(by_model.keys())[0] != "unknown"):
        console.print()
        console.print("[dim]By model:[/dim]")
        sorted_models = sorted(by_model.items(), key=lambda x: x[1]["cost"], reverse=True)
        for model, stats in sorted_models:
            cost_str = _format_cost(stats["cost"])
            console.print(f"  {model}: {stats['calls']} calls, {cost_str}")

    console.print()


def run_traced_command(
    command: List[str],
    output_path: Optional[str] = None,
    console: Optional[Console] = None,
) -> Tuple[int, Optional[Path]]:
    """Run a command with automatic SDK instrumentation.

    Args:
        command: Command and arguments to run (e.g., ["python", "script.py"])
        output_path: Optional path for trace output. Auto-generates if None.
        console: Rich console for output

    Returns:
        Tuple of (exit_code, trace_file_path)
    """
    console = console or Console()

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
    with open(bootstrap_path, "w") as f:
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
        env["PYTHONPATH"] = f"{bootstrap_dir}:{evalview_path}:{existing_pythonpath}"
    else:
        env["PYTHONPATH"] = f"{bootstrap_dir}:{evalview_path}"

    # Create sitecustomize.py in the bootstrap directory to auto-run
    sitecustomize_path = Path(bootstrap_dir) / "sitecustomize.py"
    sitecustomize_existed = sitecustomize_path.exists()
    if not sitecustomize_existed:
        with open(sitecustomize_path, "w") as f:
            f.write(BOOTSTRAP_CODE)

    try:
        # Print header
        console.print(f"[bold cyan]━━━ EvalView Trace ━━━[/bold cyan]")
        console.print(f"[dim]Running: {' '.join(command)}[/dim]")
        console.print()

        # Run the command
        result = subprocess.run(command, env=env)

        # Print summary
        _print_summary(console, trace_file)

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
