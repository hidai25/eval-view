"""OpenClaw integration commands."""
from __future__ import annotations


import click

from evalview.commands.shared import console


@click.group()
def openclaw():
    """OpenClaw integration — install skills and manage the regression gate."""
    pass


@openclaw.command("install")
@click.option("--target", "-t", default=".", help="Directory to install the skill into (default: current dir).")
def install(target: str):
    """Install the evalview-gate skill for OpenClaw claws.

    Copies the skill to <target>/skills/evalview-gate.md so any claw
    working in that directory can use EvalView as a regression gate.

    \b
    Examples:
        evalview openclaw install                  # Install in current directory
        evalview openclaw install --target ~/my-claw  # Install in specific claw workspace
    """
    from evalview.openclaw import install_skill

    path = install_skill(target)
    console.print(f"[green]Installed evalview-gate skill:[/green] {path}")
    console.print()
    console.print("[bold]Next steps:[/bold]")
    console.print("  1. Ensure your claw has [cyan]bash[/cyan] tool access")
    console.print("  2. The claw will automatically use EvalView after code changes")
    console.print()
    console.print("[dim]Or use the Python API directly in your autonomous loop:[/dim]")
    console.print()
    console.print("  [cyan]from evalview.openclaw import gate_or_revert[/cyan]")
    console.print("  [cyan]ok = gate_or_revert('tests/')[/cyan]")


@openclaw.command("check")
@click.option("--path", "test_dir", default="tests", help="Test directory (default: tests/).")
@click.option("--strict", is_flag=True, help="Fail on any change, not just regressions.")
@click.option("--auto-revert/--no-revert", default=True, help="Auto-revert on regression (default: yes).")
@click.option("--timeout", default=30.0, type=float, help="Per-test timeout in seconds.")
def openclaw_check(test_dir: str, strict: bool, auto_revert: bool, timeout: float):
    """Run the gate check and show the decision.

    This is what an autonomous claw does internally — runs tests,
    decides whether to continue/revert/accept.

    \b
    Examples:
        evalview openclaw check                    # Standard gate check
        evalview openclaw check --strict           # Fail on any change
        evalview openclaw check --no-revert        # Don't auto-revert, just report
    """
    from evalview.openclaw import check_and_decide

    decision = check_and_decide(
        test_dir=test_dir,
        strict=strict,
        auto_revert=auto_revert,
        timeout=timeout,
    )

    action_styles = {
        "continue": ("[green]CONTINUE[/green]", "green"),
        "revert": ("[red]REVERT[/red]", "red"),
        "accept": ("[yellow]ACCEPT[/yellow]", "yellow"),
        "review": ("[yellow]REVIEW[/yellow]", "yellow"),
    }
    label, color = action_styles.get(decision.action, ("?", "white"))

    console.print(f"\n  Decision: {label}")
    console.print(f"  Reason:   {decision.reason}")

    if decision.reverted:
        console.print("  [dim]Changes were automatically reverted.[/dim]")

    if decision.action == "accept" and decision.changed_tests:
        for name in decision.changed_tests:
            quoted = f'"{name}"' if " " in name else name
            console.print(f"  [dim]Accept:[/dim] evalview snapshot --test {quoted}")

    summary = decision.gate_result.summary
    console.print(
        f"\n  [dim]{summary.total} compared · "
        f"{summary.unchanged} unchanged · "
        f"{summary.regressions} regressions · "
        f"{summary.tools_changed} tool changes · "
        f"{summary.output_changed} output changes[/dim]\n"
    )

    raise SystemExit(decision.gate_result.exit_code)
