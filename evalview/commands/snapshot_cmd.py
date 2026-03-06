"""Snapshot command — run tests and save passing results as baseline."""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional, TYPE_CHECKING

import click

from evalview.commands.shared import (
    console,
    _execute_snapshot_tests,
    _load_config_if_exists,
    _cloud_push,
)
from evalview.telemetry.decorators import track_command

if TYPE_CHECKING:
    from evalview.core.types import EvaluationResult


def _save_snapshot_results(
    results: List["EvaluationResult"],
    notes: Optional[str],
    variant: Optional[str] = None,
) -> int:
    """Save passing test results as golden baselines.

    Returns:
        Number of tests successfully saved
    """
    from evalview.core.golden import GoldenStore

    store = GoldenStore()

    # Filter to passing results
    passing = [r for r in results if r.passed]

    if not passing:
        console.print("\n[yellow]No passing tests to snapshot.[/yellow]")
        console.print("[dim]Fix failing tests first, then run evalview snapshot again.[/dim]\n")
        return 0

    # Save passing results as golden
    console.print()
    saved_count = 0
    saved_names = []
    for result in passing:
        try:
            store.save_golden(result, notes=notes, variant_name=variant)
            variant_label = f" (variant: {variant})" if variant else ""
            console.print(f"[green]✓ Snapshotted:[/green] {result.test_case}{variant_label}")
            saved_count += 1
            saved_names.append(result.test_case)
        except Exception as e:
            console.print(f"[red]❌ Failed to save {result.test_case}: {e}[/red]")

    # Silent cloud push — never blocks or fails the snapshot
    if saved_names:
        _cloud_push(saved_names)

    return saved_count


@click.command("snapshot")
@click.argument("test_path", default="tests", type=click.Path(exists=True))
@click.option("--notes", "-n", help="Notes about this snapshot")
@click.option("--test", "-t", help="Snapshot only this specific test (by name)")
@click.option("--variant", help="Save as a named variant for non-deterministic agents (max 5 per test)")
@track_command("snapshot")
def snapshot(test_path: str, notes: str, test: str, variant: str):
    """Run tests and snapshot passing results as baseline.

    This is the simple workflow: snapshot → check → fix → snapshot.

    TEST_PATH is the directory containing test cases (default: tests/).

    Examples:
        evalview snapshot                         # Snapshot all passing tests
        evalview snapshot --test "my-test"        # Snapshot one test only
        evalview snapshot --notes "v2.0"          # Add notes to snapshot
        evalview snapshot --variant variant1      # Save as alternate acceptable behavior
    """
    from evalview.core.loader import TestCaseLoader
    from evalview.core.project_state import ProjectStateStore
    from evalview.core.celebrations import Celebrations
    from evalview.core.messages import get_random_checking_message
    from evalview.skills.ui_utils import print_evalview_banner

    print_evalview_banner(console, subtitle="[dim]Catch agent regressions before you ship[/dim]")

    # Initialize stores
    state_store = ProjectStateStore()

    # Check if this is the first snapshot ever
    is_first = state_store.is_first_snapshot()

    console.print(f"\n[cyan]▶ {get_random_checking_message()}[/cyan]\n")

    # Load test cases
    loader = TestCaseLoader()
    try:
        test_cases = loader.load_from_directory(Path(test_path))
    except Exception as e:
        console.print(f"[red]❌ Failed to load test cases: {e}[/red]\n")
        Celebrations.no_tests_found()
        return

    if not test_cases:
        Celebrations.no_tests_found()
        return

    # Filter to specific test if requested
    if test:
        test_cases = [tc for tc in test_cases if tc.name == test]
        if not test_cases:
            console.print(f"[red]❌ No test found with name: {test}[/red]\n")
            return

    # Run tests
    console.print(f"[cyan]Running {len(test_cases)} test(s)...[/cyan]\n")

    # Load config
    config = _load_config_if_exists()

    # Execute tests
    results = _execute_snapshot_tests(test_cases, config)

    # Save passing results as golden
    saved_count = _save_snapshot_results(results, notes, variant=variant)

    if saved_count == 0:
        return

    # Update project state
    state_store.update_snapshot(test_count=saved_count)

    # Celebrate!
    if is_first:
        Celebrations.first_snapshot(saved_count)
    else:
        console.print(f"\n[green]Baseline updated: {saved_count} test(s)[/green]")
        console.print("[dim]Run: evalview check[/dim]\n")
