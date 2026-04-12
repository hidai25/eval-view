"""CLI commands for managing the flake quarantine (with governance).

Week 2 upgrade: `add` now requires `--owner` and `--reason`. `list` shows
owner, age, stale badge, and a flaky trend glyph (↗/→/↘) so teams can see
rot before the quarantine folder becomes a dumping ground.
"""

from __future__ import annotations

from typing import Optional

import click

from evalview.commands.shared import console
from evalview.core.quarantine import (
    DEFAULT_REVIEW_DAYS,
    QuarantineOwnerRequired,
    QuarantineReasonRequired,
    QuarantineStore,
)
from evalview.telemetry.decorators import track_command


_TREND_GLYPH = {
    "up": "↗",
    "flat": "→",
    "down": "↘",
}


@click.group()
def quarantine() -> None:
    """Manage quarantined (known-flaky) tests.

    Quarantined tests still run and report, but failures don't block CI.
    Governance rules:
        - `add` requires --owner and --reason (no silent quarantining)
        - `list` surfaces stale entries whose review window has lapsed
    """


@quarantine.command("add")
@click.argument("test_name")
@click.option("--owner", "-o", required=True,
              help="Who owns this quarantine entry (e.g. @hidai).")
@click.option("--reason", "-r", required=True,
              help="Why this test is being quarantined.")
@click.option("--review-after-days", type=int, default=DEFAULT_REVIEW_DAYS,
              show_default=True,
              help="Flag as stale after this many days without improvement.")
@click.option("--expires", "expiry_date", default="",
              help="ISO date after which this quarantine is stale (e.g. 2026-05-01).")
@track_command("quarantine_add")
def quarantine_add(
    test_name: str,
    owner: str,
    reason: str,
    review_after_days: int,
    expiry_date: str,
) -> None:
    """Add a test to the quarantine list."""
    store = QuarantineStore()
    if store.is_quarantined(test_name):
        existing = store.entries[test_name]
        console.print(f"[yellow]Already quarantined:[/yellow] {test_name}")
        if existing.owner:
            console.print(f"  [dim]Owner:  {existing.owner}[/dim]")
        if existing.reason:
            console.print(f"  [dim]Reason: {existing.reason}[/dim]")
        return

    try:
        entry = store.add(
            test_name,
            reason=reason,
            owner=owner,
            review_after_days=review_after_days,
            expiry_date=expiry_date,
        )
    except (QuarantineOwnerRequired, QuarantineReasonRequired) as exc:
        console.print(f"[red]✗ {exc}[/red]")
        raise click.Abort() from exc

    console.print(f"[green]Quarantined:[/green] {test_name}")
    console.print(f"  [dim]Owner:  {entry.owner}[/dim]")
    console.print(f"  [dim]Reason: {entry.reason}[/dim]")
    if entry.expiry_date:
        console.print(f"  [dim]Expires: {entry.expiry_date}[/dim]")
    console.print(
        f"  [dim]Review window: {entry.review_after_days} days "
        "(flagged stale after that without improvement)[/dim]"
    )


@quarantine.command("remove")
@click.argument("test_name")
@track_command("quarantine_remove")
def quarantine_remove(test_name: str) -> None:
    """Remove a test from the quarantine list."""
    store = QuarantineStore()
    if store.remove(test_name):
        console.print(f"[green]Removed from quarantine:[/green] {test_name}")
        console.print(
            "  [dim]This test will now block deployment if it fails.[/dim]"
        )
    else:
        console.print(f"[yellow]Not quarantined:[/yellow] {test_name}")


@quarantine.command("list")
@click.option("--stale-only", is_flag=True,
              help="Show only stale entries (review overdue).")
@click.option("--json", "json_output", is_flag=True,
              help="Emit machine-readable JSON.")
@track_command("quarantine_list")
def quarantine_list(stale_only: bool, json_output: bool) -> None:
    """Show quarantined tests with owner, age, and stale status."""
    store = QuarantineStore()
    entries = store.list_stale() if stale_only else store.list_all()

    if json_output:
        import json
        click.echo(json.dumps({"quarantined": [e.to_dict() for e in entries]}, indent=2))
        return

    if not entries:
        if stale_only:
            console.print("[green]No stale quarantine entries.[/green]")
        else:
            console.print("[dim]No quarantined tests.[/dim]")
        return

    from rich.table import Table

    stale_count = sum(1 for e in entries if e.stale)
    title = (
        f"{len(entries)} quarantined test(s)"
        + (f" — [yellow]{stale_count} stale[/yellow]" if stale_count else "")
    )

    table = Table(
        title=title,
        show_header=True,
        header_style="bold",
        padding=(0, 1),
    )
    table.add_column("Test", no_wrap=True)
    table.add_column("Owner", style="cyan", no_wrap=True)
    table.add_column("Age", justify="right", no_wrap=True)
    table.add_column("Flaky", justify="right", no_wrap=True)
    table.add_column("Trend", justify="center", no_wrap=True)
    table.add_column("Status", no_wrap=True)
    table.add_column("Reason", overflow="fold")

    for e in sorted(entries, key=lambda x: (not x.stale, x.test_name)):
        age = e.age_days
        age_str = f"{age}d" if age is not None else "—"
        trend_glyph = _TREND_GLYPH.get(e.flaky_trend, "→")
        trend_color = {
            "up": "red",
            "flat": "dim",
            "down": "green",
        }.get(e.flaky_trend, "dim")

        if e.stale:
            status = "[red]⏰ STALE[/red]"
        else:
            status = "[green]⏸ active[/green]"

        table.add_row(
            e.test_name,
            e.owner or "[red]<unknown>[/red]",
            age_str,
            str(e.flaky_count),
            f"[{trend_color}]{trend_glyph}[/{trend_color}]",
            status,
            e.reason or "[red]<no reason>[/red]",
        )

    console.print(table)

    if stale_count:
        console.print()
        console.print(
            f"[yellow]⏰ {stale_count} entr{'y' if stale_count == 1 else 'ies'} stale "
            "— review overdue.[/yellow]"
        )
        console.print(
            "[dim]   Either fix the underlying flake or remove from quarantine:[/dim]"
        )
        console.print("[dim]   evalview quarantine remove <test>[/dim]")
