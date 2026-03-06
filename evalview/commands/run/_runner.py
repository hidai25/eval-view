"""Test execution runners for the run command.

Two execution strategies:
- `run_sequential`: simple Progress-bar driven loop, one test at a time.
- `run_parallel`: concurrent execution via `execute_tests_parallel` with a
  Rich Live panel that shows elapsed time, running tests, and pass/fail counts.

Both return ``(results, passed, failed, execution_errors)``.
"""
from __future__ import annotations

import asyncio
import sys
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from rich.live import Live
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn


async def run_sequential(
    test_cases: List[Any],
    execute_fn: Callable,
    console: Any,
    config: Dict[str, Any],
) -> Tuple[List[Any], int, int, int]:
    """Execute tests one at a time, showing a spinner for each.

    Returns:
        (results, passed, failed, execution_errors)
    """
    results: List[Any] = []
    passed = failed = execution_errors = 0

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        for test_case in test_cases:
            task = progress.add_task(f"Running {test_case.name}...", total=None)
            try:
                test_passed, result = await execute_fn(test_case)
                results.append(result)
                if test_passed:
                    passed += 1
                    progress.update(
                        task,
                        description=f"[green]✅ {test_case.name} - PASSED (score: {result.score})[/green]",
                    )
                else:
                    failed += 1
                    progress.update(
                        task,
                        description=f"[red]❌ {test_case.name} - FAILED (score: {result.score})[/red]",
                    )
            except Exception as exc:
                execution_errors += 1
                error_msg = _format_error(exc, config)
                progress.update(
                    task,
                    description=f"[red]⚠ {test_case.name} - EXECUTION ERROR: {error_msg}[/red]",
                )
            finally:
                progress.remove_task(task)

    return results, passed, failed, execution_errors


async def run_parallel(
    test_cases: List[Any],
    execute_fn: Callable,
    max_workers: int,
    verbose: bool,
    console: Any,
    config: Dict[str, Any],
) -> Tuple[List[Any], int, int, int]:
    """Execute tests concurrently with a Live status panel.

    In non-interactive (CI) environments, falls back to simple output without
    the Live panel.

    Returns:
        (results, passed, failed, execution_errors)
    """
    import time as time_module
    import httpx as _httpx
    from evalview.core.parallel import execute_tests_parallel
    from evalview.core.llm_provider import judge_cost_tracker

    results: List[Any] = []

    # Mutable state shared across callbacks
    passed = 0
    failed = 0
    execution_errors = 0
    tests_running: set = set()
    tests_completed = 0
    start_time = time_module.time()

    judge_cost_tracker.reset()

    # ── Callback helpers ──────────────────────────────────────────────────────

    def _on_start(test_name: str) -> None:
        nonlocal tests_running
        tests_running.add(test_name[:30])
        if verbose:
            console.print(f"[dim]  ▶ Starting: {test_name}[/dim]")

    def _on_complete(test_name: str, test_passed: bool, result: Any) -> None:
        nonlocal passed, failed, tests_running, tests_completed
        tests_running.discard(test_name[:30])
        tests_completed += 1
        if test_passed:
            passed += 1
            console.print(f"[green]✅ {test_name} - PASSED (score: {result.score})[/green]")
        else:
            failed += 1
            console.print(f"[red]❌ {test_name} - FAILED (score: {result.score})[/red]")

    def _on_error(test_name: str, exc: Exception) -> None:
        nonlocal execution_errors, tests_running, tests_completed
        tests_running.discard(test_name[:30])
        tests_completed += 1
        execution_errors += 1
        error_msg = _format_error(exc, config)
        console.print(f"[red]⚠ {test_name} - EXECUTION ERROR: {error_msg}[/red]")

    # ── Live display helpers ──────────────────────────────────────────────────

    def _format_elapsed() -> str:
        elapsed = time_module.time() - start_time
        mins, secs = divmod(elapsed, 60)
        secs_int = int(secs)
        ms = int((secs - secs_int) * 1000)
        return f"{int(mins):02d}:{secs_int:02d}.{ms:03d}"

    _spinner_frames = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
    _spinner_idx = 0

    def _get_status_panel() -> Panel:
        nonlocal _spinner_idx
        spinner = _spinner_frames[_spinner_idx % len(_spinner_frames)]
        _spinner_idx += 1

        running_lines = "\n".join(
            [f"  [yellow]{spinner}[/yellow] [dim]{t}...[/dim]" for t in list(tests_running)[:3]]
        ) or f"  [yellow]{spinner}[/yellow] [dim]Starting tests...[/dim]"

        status = "[bold red]● Running[/bold red]" if failed > 0 else "[green]● Running[/green]"
        judge_cost = judge_cost_tracker.get_summary()

        content = (
            f"  {status}\n\n"
            f"  [bold]⏱️  Elapsed:[/bold]    [yellow]{_format_elapsed()}[/yellow]\n"
            f"  [bold]📋 Progress:[/bold]   {tests_completed}/{len(test_cases)} tests\n"
            f"  [bold]💰 Judge:[/bold]      [dim]{judge_cost}[/dim]\n\n"
            f"{running_lines}\n\n"
            f"  [green]✓ Passed:[/green] {passed}    [red]✗ Failed:[/red] {failed}"
        )

        return Panel(
            content,
            title="[bold]Test Execution[/bold]",
            border_style="red" if failed > 0 else "cyan",
            padding=(0, 1),
        )

    # ── Execution ─────────────────────────────────────────────────────────────

    console.print(
        f"[dim]Executing {len(test_cases)} tests with up to {max_workers} parallel workers...[/dim]\n"
    )

    if sys.stdin.isatty():
        async def _update_display(live: Live) -> None:
            while tests_completed < len(test_cases):
                live.update(_get_status_panel())
                await asyncio.sleep(0.1)
            live.update(_get_status_panel())

        with Live(_get_status_panel(), console=console, refresh_per_second=10) as live:
            parallel_task = execute_tests_parallel(
                test_cases,
                execute_fn,
                max_workers=max_workers,
                on_start=_on_start,
                on_complete=_on_complete,
                on_error=_on_error,
            )
            parallel_results, _ = await asyncio.gather(
                parallel_task, _update_display(live), return_exceptions=True
            )

        _print_completion_box(passed, failed, execution_errors, _format_elapsed(), judge_cost_tracker, console)

    else:
        parallel_results = await execute_tests_parallel(
            test_cases,
            execute_fn,
            max_workers=max_workers,
            on_start=_on_start,
            on_complete=_on_complete,
            on_error=_on_error,
        )

    # Collect successful results
    if isinstance(parallel_results, BaseException):
        import logging as _logging
        _logging.getLogger(__name__).error(f"parallel_results is an exception: {parallel_results}")
        console.print(f"[red]Error in parallel execution: {parallel_results}[/red]")
    elif parallel_results:
        for pr in list(parallel_results):  # type: ignore[arg-type]
            if pr.success and pr.result:
                results.append(pr.result)

    return results, passed, failed, execution_errors


# ── Private helpers ───────────────────────────────────────────────────────────


def _format_error(exc: Exception, config: Dict[str, Any]) -> str:
    """Return a human-readable error message from an execution exception."""
    try:
        import httpx as _httpx
        if isinstance(exc, _httpx.ConnectError):
            endpoint = config.get("endpoint", "unknown endpoint")
            return f"Cannot connect to {endpoint}"
        if isinstance(exc, _httpx.TimeoutException):
            return "Request timeout"
    except ImportError:
        pass
    return str(exc)


def _print_completion_box(
    passed: int,
    failed: int,
    execution_errors: int,
    elapsed: str,
    judge_cost_tracker: Any,
    console: Any,
) -> None:
    """Print the ASCII completion box shown after parallel execution."""
    final_judge_cost = judge_cost_tracker.get_summary()
    console.print()
    console.print("[bold cyan]╔══════════════════════════════════════════════════════════════════╗[/bold cyan]")
    console.print("[bold cyan]║[/bold cyan]                                                                  [bold cyan]║[/bold cyan]")
    if execution_errors > 0:
        console.print("[bold cyan]║[/bold cyan]  [bold red]⚠ EXECUTION ERRORS OCCURRED[/bold red]                                  [bold cyan]║[/bold cyan]")
    elif failed == 0:
        console.print("[bold cyan]║[/bold cyan]  [bold green]✓ AGENT HEALTHY[/bold green]                                               [bold cyan]║[/bold cyan]")
    else:
        console.print("[bold cyan]║[/bold cyan]  [bold red]✗ REGRESSION DETECTED[/bold red]                                        [bold cyan]║[/bold cyan]")
    console.print("[bold cyan]║[/bold cyan]                                                                  [bold cyan]║[/bold cyan]")
    if execution_errors > 0:
        console.print(f"[bold cyan]║[/bold cyan]  [green]✓ Passed:[/green] {passed:<4}  [red]✗ Failed:[/red] {failed:<4}  [red]⚠ Errors:[/red] {execution_errors:<4}         [bold cyan]║[/bold cyan]")
    else:
        console.print(f"[bold cyan]║[/bold cyan]  [green]✓ Passed:[/green] {passed:<4}  [red]✗ Failed:[/red] {failed:<4}  [dim]Time:[/dim] {elapsed}               [bold cyan]║[/bold cyan]")
    console.print(f"[bold cyan]║[/bold cyan]  [dim]💰 Judge cost:[/dim] {final_judge_cost:<45}[bold cyan]║[/bold cyan]")
    console.print("[bold cyan]║[/bold cyan]                                                                  [bold cyan]║[/bold cyan]")
    console.print("[bold cyan]╚══════════════════════════════════════════════════════════════════╝[/bold cyan]")
    console.print()
