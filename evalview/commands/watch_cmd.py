"""Watch command — re-run regression checks on file changes."""
from __future__ import annotations

import asyncio
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, List, Optional

import click

from evalview.commands.shared import console
from evalview.telemetry.decorators import track_command

# Directories to exclude from watching by default
_DEFAULT_EXCLUDES = {
    ".evalview", ".git", "venv", ".venv", "env", ".env",
    "node_modules", "__pycache__", ".mypy_cache", ".pytest_cache",
    "dist", "build", ".tox", ".eggs", "*.egg-info",
}


@click.command("watch")
@click.option(
    "--path", "-p",
    multiple=True,
    help="Directories to watch (default: current directory).",
)
@click.option(
    "--test", "-t",
    "test_name",
    default=None,
    help="Only check this test (by name).",
)
@click.option(
    "--test-dir",
    default="tests",
    show_default=True,
    help="Path to test cases directory.",
)
@click.option(
    "--interval",
    default=2.0,
    show_default=True,
    type=float,
    help="Debounce interval in seconds.",
)
@click.option(
    "--quick",
    is_flag=True,
    help="Skip LLM judge — deterministic checks only ($0, sub-second).",
)
@click.option(
    "--fail-on",
    default="REGRESSION",
    show_default=True,
    help="Comma-separated statuses that count as failure.",
)
@click.option(
    "--sound",
    is_flag=True,
    help="Terminal bell on regression.",
)
@track_command("watch", lambda **kw: {"quick": kw.get("quick", False)})
def watch(
    path: tuple,
    test_name: Optional[str],
    test_dir: str,
    interval: float,
    quick: bool,
    fail_on: str,
    sound: bool,
) -> None:
    """Watch for file changes and re-run regression checks.

    Runs evalview check automatically when files change. Uses the
    same gate() API as the Python library — no subprocess overhead.

    \b
    Examples:
        evalview watch                      # Watch current dir, check on change
        evalview watch --quick              # Fast mode — no LLM judge, $0
        evalview watch --path src/ --path tests/
        evalview watch --test "my-test"     # Only check one test
        evalview watch --interval 1         # 1-second debounce
    """
    run_watch(
        watch_paths=list(path) if path else ["."],
        test_dir=test_dir,
        test_name=test_name,
        quick=quick,
        fail_on=fail_on,
        sound=sound,
        interval=interval,
    )


def run_watch(
    *,
    watch_paths: List[str],
    test_dir: str,
    test_name: Optional[str],
    quick: bool,
    fail_on: str,
    sound: bool,
    interval: float,
) -> None:
    """Run the watch loop until interrupted.

    Shared by `evalview watch` and `evalview check --watch` so the two
    entry points can't drift apart. Never returns normally — it either
    loops until KeyboardInterrupt or exits the process on a setup error.
    """
    try:
        from evalview.core.watcher import WATCHDOG_AVAILABLE
    except ImportError:
        WATCHDOG_AVAILABLE = False

    if not WATCHDOG_AVAILABLE:
        console.print(
            "[red]Watch mode requires watchdog.[/red]\n"
            "Install with: [bold]pip install evalview[watch][/bold]"
        )
        sys.exit(1)

    # Validate test directory exists
    if not Path(test_dir).exists():
        console.print(f"[red]Test directory not found:[/red] {test_dir}")
        sys.exit(1)

    # Parse fail_on statuses
    from evalview.commands.shared import _parse_fail_statuses
    fail_statuses = _parse_fail_statuses(fail_on)

    # Show startup banner
    _print_banner(watch_paths, test_dir, test_name, quick, interval)

    # Run initial check
    _run_check(
        test_dir=test_dir,
        test_name=test_name,
        quick=quick,
        fail_statuses=fail_statuses,
        sound=sound,
        trigger_path=None,
    )

    # Start watching
    try:
        asyncio.run(_watch_loop(
            watch_paths=watch_paths,
            test_dir=test_dir,
            test_name=test_name,
            quick=quick,
            fail_statuses=fail_statuses,
            sound=sound,
            debounce_seconds=interval,
        ))
    except KeyboardInterrupt:
        console.print("\n[dim]Watch stopped.[/dim]")


def _print_banner(
    watch_paths: list,
    test_dir: str,
    test_name: Optional[str],
    quick: bool,
    interval: float,
) -> None:
    """Print the watch mode startup banner."""
    from rich.panel import Panel

    paths_str = ", ".join(watch_paths)
    mode_str = "[cyan]quick[/cyan] (no judge, $0)" if quick else "[cyan]full[/cyan] (with judge)"
    test_str = f"[cyan]{test_name}[/cyan]" if test_name else f"all in [cyan]{test_dir}/[/cyan]"

    content = (
        f"  Watching   {paths_str}\n"
        f"  Tests      {test_str}\n"
        f"  Mode       {mode_str}\n"
        f"  Debounce   {interval}s\n"
        f"\n"
        f"  [dim]Press Ctrl+C to stop[/dim]"
    )
    console.print(Panel(content, title="EvalView Watch", border_style="blue"))
    console.print()


def _run_check(
    test_dir: str,
    test_name: Optional[str],
    quick: bool,
    fail_statuses: set,
    sound: bool,
    trigger_path: Optional[str],
) -> None:
    """Run a single check cycle synchronously. Used for the initial run only.

    Inside the watch loop use :func:`_run_check_async` — the synchronous
    ``gate()`` calls ``asyncio.run()``, which raises once a loop is running.
    """
    from evalview.api import gate

    _announce_check(trigger_path)

    try:
        result = gate(
            test_dir=test_dir,
            test_name=test_name,
            quick=quick,
            fail_on=fail_statuses,
        )
    except Exception as e:
        console.print(f"[red]Check failed:[/red] {e}\n")
        return

    _render_check_result(result, sound)


async def _run_check_async(
    test_dir: str,
    test_name: Optional[str],
    quick: bool,
    fail_statuses: set,
    sound: bool,
    trigger_path: Optional[str],
) -> None:
    """Run a single check cycle from inside the watch loop's event loop.

    Must use ``gate_async``: the loop is already running, so the synchronous
    ``gate()`` would fail with "asyncio.run() cannot be called from a running
    event loop" and every file-change trigger would be swallowed as a failed
    check.
    """
    from evalview.api import gate_async

    _announce_check(trigger_path)

    try:
        result = await gate_async(
            test_dir=test_dir,
            test_name=test_name,
            quick=quick,
            fail_on=fail_statuses,
        )
    except Exception as e:
        console.print(f"[red]Check failed:[/red] {e}\n")
        return

    _render_check_result(result, sound)


def _announce_check(trigger_path: Optional[str]) -> None:
    """Print the header line for a check cycle."""
    timestamp = datetime.now().strftime("%H:%M:%S")

    if trigger_path:
        console.print(f"[dim]{timestamp}[/dim]  Change detected: [cyan]{trigger_path}[/cyan]")
    else:
        console.print(f"[dim]{timestamp}[/dim]  Running initial check...")

    console.print()


def _render_check_result(result: Any, sound: bool) -> None:
    """Display the scorecard and per-test details for one check cycle."""
    from evalview.api import DiffStatus
    from evalview.core.dashboard import render_scorecard
    from evalview.core.project_state import ProjectStateStore

    if not result.diffs:
        console.print("[yellow]No tests matched any baselines.[/yellow]")
        console.print("[dim]Run evalview snapshot to capture baselines first.[/dim]\n")
        return

    # Load state for streak info
    try:
        state_store = ProjectStateStore()
        state = state_store.load()
        current_streak = int(state.current_streak or 0)
        longest_streak = int(state.longest_streak or 0)
    except Exception:
        current_streak = 0
        longest_streak = 0

    s = result.summary
    health_pct = (s.unchanged / s.total * 100) if s.total > 0 else 0.0

    scorecard = render_scorecard(
        passed=s.unchanged,
        tools_changed=s.tools_changed,
        output_changed=s.output_changed,
        regressions=s.regressions,
        execution_failures=s.execution_failures,
        current_streak=current_streak,
        longest_streak=longest_streak,
        health_pct=health_pct,
    )
    console.print(scorecard)

    # Show per-test details for non-passing tests
    if not result.passed:
        console.print()
        for diff in result.diffs:
            if not diff.passed:
                status_icon = {
                    DiffStatus.REGRESSION: "[red]  REGRESSION[/red]",
                    DiffStatus.TOOLS_CHANGED: "[yellow]  TOOLS_CHANGED[/yellow]",
                    DiffStatus.OUTPUT_CHANGED: "[dim]  OUTPUT_CHANGED[/dim]",
                }.get(diff.status, "  ?")

                parts = [status_icon, f"  [bold]{diff.test_name}[/bold]"]
                if abs(diff.score_delta) > 1:
                    sign = "+" if diff.score_delta > 0 else ""
                    parts.append(f"  score {sign}{diff.score_delta:.1f}")
                if diff.tool_changes > 0:
                    parts.append(f"  {diff.tool_changes} tool change(s)")
                if diff.output_similarity is not None and diff.output_similarity < 0.95:
                    parts.append(f"  output {diff.output_similarity:.0%} similar")

                console.print("".join(parts))

        if sound:
            # Terminal bell
            sys.stdout.write("\a")
            sys.stdout.flush()
    else:
        console.print(f"[green]  All {s.total} tests clean.[/green]")

    console.print()
    console.print("[dim]Watching for changes...[/dim]\n")


async def _watch_loop(
    watch_paths: list,
    test_dir: str,
    test_name: Optional[str],
    quick: bool,
    fail_statuses: set,
    sound: bool,
    debounce_seconds: float,
) -> None:
    """Main async watch loop with simple debounce.

    Uses watchdog's Observer directly with a lightweight event handler.
    Debounce is handled here (not in the async DebouncedTestHandler) to
    avoid nested event loop issues and keep the logic straightforward.
    """
    import threading
    from watchdog.observers import Observer
    from watchdog.events import FileSystemEventHandler as _BaseHandler

    # Shared state between watchdog thread and our async loop
    _lock = threading.Lock()
    _pending_path: Optional[str] = None
    _last_event_time: float = 0
    _running_check = False

    class _Handler(_BaseHandler):
        def _handle(self, event) -> None:  # type: ignore[override]
            nonlocal _pending_path, _last_event_time
            if event.is_directory:
                return
            path = event.src_path
            # Check file extension
            if not any(path.endswith(ext) for ext in (
                ".py", ".yaml", ".yml", ".json", ".md",
                ".txt", ".toml", ".cfg", ".ini",
            )):
                return
            # Check exclusions
            for exclude in _DEFAULT_EXCLUDES:
                if f"/{exclude}/" in path or f"\\{exclude}\\" in path:
                    return
            with _lock:
                _pending_path = path
                _last_event_time = time.time()

        def on_modified(self, event) -> None:  # type: ignore[override]
            self._handle(event)

        def on_created(self, event) -> None:  # type: ignore[override]
            self._handle(event)

    handler = _Handler()
    observer = Observer()
    for wp in watch_paths:
        p = Path(wp).resolve()
        if p.exists():
            observer.schedule(handler, str(p), recursive=True)

    observer.start()

    try:
        while True:
            await asyncio.sleep(0.3)

            with _lock:
                if _pending_path is None:
                    continue
                # Wait for debounce period after last event
                if time.time() - _last_event_time < debounce_seconds:
                    continue
                if _running_check:
                    continue
                trigger = _pending_path
                _pending_path = None
                _running_check = True

            try:
                await _run_check_async(
                    test_dir=test_dir,
                    test_name=test_name,
                    quick=quick,
                    fail_statuses=fail_statuses,
                    sound=sound,
                    trigger_path=trigger,
                )
            finally:
                with _lock:
                    _running_check = False
    finally:
        observer.stop()
        observer.join(timeout=2)
