"""Telemetry management commands."""
from __future__ import annotations

import click

from evalview.commands.shared import console
from evalview.telemetry.config import (
    TELEMETRY_DISABLED_ENV,
    load_config as load_telemetry_config,
    set_telemetry_enabled,
)


@click.group()
def telemetry():
    """Manage anonymous usage telemetry.

    EvalView collects anonymous usage data to improve the tool.
    No personal info, API keys, or test content is collected.

    \b
    Examples:
        evalview telemetry status   # Check current status
        evalview telemetry off      # Disable telemetry
        evalview telemetry on       # Enable telemetry
    """
    pass


@telemetry.command("status")
def telemetry_status():
    """Show current telemetry status."""
    import os

    env_disabled = os.environ.get(TELEMETRY_DISABLED_ENV, "").lower() in ("1", "true", "yes")
    config = load_telemetry_config()

    console.print("\n[cyan]━━━ Telemetry Status ━━━[/cyan]\n")

    if env_disabled:
        console.print("[yellow]Status:[/yellow] [red]Disabled[/red] (via environment variable)")
        console.print(f"[dim]${TELEMETRY_DISABLED_ENV} is set[/dim]")
    elif config.enabled:
        console.print("[yellow]Status:[/yellow] [green]Enabled[/green]")
    else:
        console.print("[yellow]Status:[/yellow] [red]Disabled[/red]")

    console.print(f"[yellow]Install ID:[/yellow] [dim]{config.install_id}[/dim]")
    console.print()
    console.print("[dim]What we collect:[/dim]")
    console.print("  • Command name (run, init, chat, skill_test, etc.)")
    console.print("  • Adapter type (langgraph, crewai, etc.)")
    console.print("  • Test count, pass/fail count")
    console.print("  • OS + Python version")
    console.print("  • CI environment (github_actions, gitlab_ci, or local)")
    console.print("  • Chat session: provider, model name, message count, slash commands used")
    console.print("  • Skill commands: agent type, validation mode")
    console.print("  • Golden trace operations (save, list, delete)")
    console.print()
    console.print("[dim]What we DON'T collect:[/dim]")
    console.print("  • API keys or credentials")
    console.print("  • Test content, queries, or outputs")
    console.print("  • File paths or IP addresses")
    console.print("  • Error messages (only error class name)")
    console.print("  • Chat conversation content")
    console.print()


@telemetry.command("on")
def telemetry_on():
    """Enable anonymous telemetry."""
    import os

    env_disabled = os.environ.get(TELEMETRY_DISABLED_ENV, "").lower() in ("1", "true", "yes")

    if env_disabled:
        console.print(
            f"[yellow]Warning:[/yellow] ${TELEMETRY_DISABLED_ENV} is set. "
            "Unset it to enable telemetry."
        )
        console.print()
        return

    set_telemetry_enabled(True)
    console.print("[green]✓ Telemetry enabled[/green]")
    console.print("[dim]Thank you for helping improve EvalView![/dim]")
    console.print()


@telemetry.command("off")
def telemetry_off():
    """Disable anonymous telemetry."""
    set_telemetry_enabled(False)
    console.print("[green]✓ Telemetry disabled[/green]")
    console.print("[dim]You can re-enable anytime with: evalview telemetry on[/dim]")
    console.print()
