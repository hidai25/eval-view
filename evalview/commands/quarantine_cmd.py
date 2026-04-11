"""CLI commands for managing the flake quarantine."""

import click
from evalview.commands.shared import console
from evalview.core.quarantine import QuarantineStore


@click.group()
def quarantine():
    """Manage quarantined (known-flaky) tests.

    Quarantined tests still run and report, but failures don't block CI.
    """
    pass


@quarantine.command("add")
@click.argument("test_name")
@click.option("--reason", "-r", default="", help="Why this test is being quarantined")
def quarantine_add(test_name: str, reason: str):
    """Add a test to the quarantine list."""
    store = QuarantineStore()
    if store.is_quarantined(test_name):
        console.print(f"[yellow]Already quarantined:[/yellow] {test_name}")
        return
    entry = store.add(test_name, reason)
    console.print(f"[green]Quarantined:[/green] {test_name}")
    if reason:
        console.print(f"  [dim]Reason: {reason}[/dim]")
    console.print(f"  [dim]This test will still run but won't block CI.[/dim]")


@quarantine.command("remove")
@click.argument("test_name")
def quarantine_remove(test_name: str):
    """Remove a test from the quarantine list."""
    store = QuarantineStore()
    if store.remove(test_name):
        console.print(f"[green]Removed from quarantine:[/green] {test_name}")
        console.print(f"  [dim]Failures will now block CI again.[/dim]")
    else:
        console.print(f"[yellow]Not quarantined:[/yellow] {test_name}")


@quarantine.command("list")
def quarantine_list():
    """Show all quarantined tests."""
    store = QuarantineStore()
    entries = store.list_all()
    if not entries:
        console.print("[dim]No quarantined tests.[/dim]")
        return
    console.print(f"[bold]{len(entries)} quarantined test(s):[/bold]\n")
    for e in entries:
        console.print(f"  [yellow]⏸[/yellow] {e.test_name}")
        if e.reason:
            console.print(f"    [dim]Reason: {e.reason}[/dim]")
        if e.added_at:
            console.print(f"    [dim]Added: {e.added_at}[/dim]")
