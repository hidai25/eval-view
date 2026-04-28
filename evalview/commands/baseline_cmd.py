"""Baseline management commands."""
from __future__ import annotations


import click

from evalview.commands.shared import console
from evalview.telemetry.decorators import track_command


@click.group()
def baseline():
    """Manage test baselines for regression detection."""
    pass


@baseline.command("set")
@click.option(
    "--test",
    help="Specific test name to set baseline for (default: all recent tests)",
)
@click.option(
    "--from-latest",
    is_flag=True,
    help="Set baseline from most recent test run",
)
@track_command("baseline_set")
def baseline_set(test: str, from_latest: bool):
    """Set baseline from recent test results."""
    from evalview.tracking import RegressionTracker

    tracker = RegressionTracker()

    if test:
        # Set baseline for specific test
        if from_latest:
            try:
                tracker.set_baseline_from_latest(test)
                console.print(f"[green]✅ Baseline set for test: {test}[/green]")
            except ValueError as e:
                console.print(f"[red]❌ Error: {e}[/red]")
        else:
            console.print("[yellow]⚠️  Must specify --from-latest or run tests first[/yellow]")
    else:
        # Set baselines for all recent tests
        results = tracker.db.get_recent_results(days=1)
        unique_tests = set(r["test_name"] for r in results)

        if not unique_tests:
            console.print("[yellow]⚠️  No recent test results found. Run tests first.[/yellow]")
            return

        for test_name in unique_tests:
            tracker.set_baseline_from_latest(test_name)

        console.print(f"[green]✅ Baselines set for {len(unique_tests)} test(s)[/green]")


@baseline.command("show")
@click.option(
    "--test",
    help="Specific test name to show baseline for",
)
@track_command("baseline_show")
def baseline_show(test: str):
    """Show current baselines."""
    from evalview.tracking import RegressionTracker
    from rich.table import Table

    tracker = RegressionTracker()

    if test:
        # Show specific baseline
        baseline_data = tracker.db.get_baseline(test)
        if not baseline_data:
            console.print(f"[yellow]⚠️  No baseline set for test: {test}[/yellow]")
            return

        console.print(f"\n[bold]Baseline for: {test}[/bold]\n")
        console.print(f"  Score: {baseline_data['score']:.2f}")
        if baseline_data.get("cost"):
            console.print(f"  Cost: ${baseline_data['cost']:.4f}")
        if baseline_data.get("latency"):
            console.print(f"  Latency: {baseline_data['latency']:.0f}ms")
        console.print(f"  Created: {baseline_data['created_at']}")
        if baseline_data.get("git_commit"):
            console.print(
                f"  Git: {baseline_data['git_commit']} ({baseline_data.get('git_branch', 'unknown')})"
            )
        console.print()
    else:
        # Show all baselines
        results = tracker.db.get_recent_results(days=30)
        unique_tests = set(r["test_name"] for r in results)

        table = Table(title="Test Baselines", show_header=True, header_style="bold cyan")
        table.add_column("Test Name", style="white")
        table.add_column("Score", justify="right", style="green")
        table.add_column("Cost", justify="right", style="yellow")
        table.add_column("Latency", justify="right", style="blue")
        table.add_column("Created", style="dim")

        has_baselines = False
        for test_name in sorted(unique_tests):
            baseline_data = tracker.db.get_baseline(test_name)
            if baseline_data:
                has_baselines = True
                table.add_row(
                    test_name,
                    f"{baseline_data['score']:.1f}",
                    f"${baseline_data.get('cost', 0):.4f}" if baseline_data.get("cost") else "N/A",
                    f"{baseline_data.get('latency', 0):.0f}ms" if baseline_data.get("latency") else "N/A",
                    baseline_data["created_at"][:10],
                )

        if not has_baselines:
            console.print(
                "[yellow]⚠️  No baselines set. Run 'evalview baseline set' first.[/yellow]"
            )
        else:
            console.print()
            console.print(table)
            console.print()


@baseline.command("clear")
@click.option(
    "--test",
    help="Specific test name to clear baseline for",
)
@click.confirmation_option(prompt="Are you sure you want to clear baselines?")
@track_command("baseline_clear")
def baseline_clear(test: str):
    """Clear baselines."""
    from evalview.tracking import RegressionTracker

    tracker = RegressionTracker()

    if test:
        # Clear specific baseline (would need to add this to DB class)
        console.print("[yellow]⚠️  Clear specific baseline not yet implemented[/yellow]")
    else:
        tracker.db.clear_baselines()
        console.print("[green]✅ All baselines cleared[/green]")
