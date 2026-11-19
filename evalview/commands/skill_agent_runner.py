"""Shared runner for agent-based skill tests."""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any, Optional

from evalview.commands.shared import console
from evalview.skills.test_helpers import (
    build_results_table,
    build_summary_panel,
    format_results_as_json,
    handle_test_completion,
    load_test_suite,
    print_detailed_test_results,
    print_suite_info,
    validate_and_parse_agent_type,
)
from evalview.skills.ui_utils import print_evalview_banner


def run_agent_skill_test(
    test_file: str,
    agent: Optional[str],
    trace_dir: Optional[str],
    no_rubric: bool,
    cwd: Optional[str],
    max_turns: Optional[int],
    verbose: bool,
    output_json: bool,
    model: Optional[str],
) -> None:
    """Run agent-based skill tests and render results."""
    agent_type_enum = validate_and_parse_agent_type(agent, console)
    suite, runner = load_test_suite(
        test_file,
        agent_type_enum,
        trace_dir,
        no_rubric,
        cwd,
        max_turns,
        verbose,
        model or "",
        console,
    )

    print_evalview_banner(console, subtitle="[dim]Agent-Based Skill Testing[/dim]")
    print_suite_info(suite, trace_dir, console)

    start_time = time.time()
    total_tests = len(suite.tests)
    completed_count = [0]

    console.print(f"[cyan]Running {total_tests} tests in parallel...[/cyan]\n")

    def on_test_complete(test_result: Any) -> None:
        completed_count[0] += 1
        icon = "[green]✓[/green]" if test_result.passed else "[red]✗[/red]"
        score_str = f"[dim]{test_result.score:.0f}%[/dim]"
        latency_str = f"[dim]{test_result.latency_ms / 1000:.1f}s[/dim]"
        console.print(
            f"  {icon} [{completed_count[0]}/{total_tests}] "
            f"[bold]{test_result.test_name}[/bold]  {score_str}  {latency_str}"
        )

    run_error = None
    result = None
    try:
        loop = asyncio.new_event_loop()
        try:
            result = loop.run_until_complete(
                runner.run_suite(suite, on_test_complete=on_test_complete)
            )
        finally:
            loop.close()
    except Exception as exc:
        run_error = exc

    console.print()

    if run_error:
        console.print(f"[red]Error running tests: {run_error}[/red]")
        raise SystemExit(1)

    assert result is not None
    elapsed_ms = (time.time() - start_time) * 1000

    if output_json:
        console.print(json.dumps(format_results_as_json(result), indent=2))
        if not result.passed:
            raise SystemExit(1)
        return

    console.print(build_results_table(result))
    console.print()
    print_detailed_test_results(result, verbose, console)
    console.print(build_summary_panel(result, elapsed_ms))
    handle_test_completion(result, test_file, suite, console)
