"""Trends command — show performance trends over time."""
from __future__ import annotations

import click

from evalview.commands.shared import console
from evalview.telemetry.decorators import track_command


@click.command("trends")
@click.option(
    "--days",
    default=30,
    help="Number of days to analyze (default: 30)",
)
@click.option(
    "--test",
    help="Specific test name to show trends for",
)
@track_command("trends")
def trends(days: int, test: str):
    """Show performance trends over time."""
    from evalview.tracking import RegressionTracker
    from rich.table import Table

    tracker = RegressionTracker()

    if test:
        # Show trends for specific test
        stats = tracker.get_statistics(test, days)

        if stats["total_runs"] == 0:
            console.print(f"[yellow]⚠️  No data found for test: {test}[/yellow]")
            return

        console.print(f"\n[bold]Performance Trends: {test}[/bold]")
        console.print(f"Period: Last {days} days\n")

        console.print("[cyan]Test Runs:[/cyan]")
        console.print(f"  Total: {stats['total_runs']}")
        console.print(f"  Passed: {stats['passed_runs']} ({stats['pass_rate']:.1f}%)")
        console.print(f"  Failed: {stats['failed_runs']}")

        if stats["score"]["current"]:
            console.print("\n[cyan]Score:[/cyan]")
            console.print(f"  Current: {stats['score']['current']:.1f}")
            console.print(f"  Average: {stats['score']['avg']:.1f}")
            console.print(f"  Range: {stats['score']['min']:.1f} - {stats['score']['max']:.1f}")

        if stats["cost"]["current"]:
            console.print("\n[cyan]Cost:[/cyan]")
            console.print(f"  Current: ${stats['cost']['current']:.4f}")
            console.print(f"  Average: ${stats['cost']['avg']:.4f}")
            console.print(f"  Range: ${stats['cost']['min']:.4f} - ${stats['cost']['max']:.4f}")

        if stats["latency"]["current"]:
            console.print("\n[cyan]Latency:[/cyan]")
            console.print(f"  Current: {stats['latency']['current']:.0f}ms")
            console.print(f"  Average: {stats['latency']['avg']:.0f}ms")
            console.print(
                f"  Range: {stats['latency']['min']:.0f}ms - {stats['latency']['max']:.0f}ms"
            )

        console.print()

    else:
        # Show overall trends
        daily_trends = tracker.db.get_daily_trends(days)

        if not daily_trends:
            console.print(f"[yellow]⚠️  No trend data available for last {days} days[/yellow]")
            return

        console.print("\n[bold]Overall Performance Trends[/bold]")
        console.print(f"Period: Last {days} days\n")

        table = Table(show_header=True, header_style="bold cyan")
        table.add_column("Date", style="white")
        table.add_column("Avg Score", justify="right", style="green")
        table.add_column("Avg Cost", justify="right", style="yellow")
        table.add_column("Avg Latency", justify="right", style="blue")
        table.add_column("Tests", justify="center", style="dim")
        table.add_column("Pass Rate", justify="right", style="green")

        for trend in daily_trends[-14:]:  # Show last 14 days
            pass_rate = (
                trend["passed_tests"] / trend["total_tests"] * 100
                if trend["total_tests"] > 0
                else 0
            )

            table.add_row(
                trend["date"],
                f"{trend['avg_score']:.1f}" if trend["avg_score"] else "N/A",
                f"${trend['avg_cost']:.4f}" if trend.get("avg_cost") else "N/A",
                f"{trend['avg_latency']:.0f}ms" if trend.get("avg_latency") else "N/A",
                str(trend["total_tests"]),
                f"{pass_rate:.0f}%",
            )

        console.print(table)
        console.print()
