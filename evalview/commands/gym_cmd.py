"""Gym command — practice agent eval patterns."""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import click
import httpx
import yaml

from evalview.commands.shared import console
from evalview.telemetry.decorators import track_command


@click.command("gym")
@click.option(
    "--suite",
    type=click.Choice(["all", "failure-modes", "security"]),
    default="all",
    help="Which test suite to run (default: all)",
)
@click.option(
    "--endpoint",
    default="http://localhost:2024",
    help="Agent endpoint URL (default: http://localhost:2024)",
)
@click.option(
    "--list-only",
    is_flag=True,
    help="List scenarios without running them",
)
@track_command("gym")
def gym(suite: str, endpoint: str, list_only: bool):
    """Run the EvalView Gym - practice agent eval patterns.

    The Gym provides curated test scenarios for learning how to write
    production-grade agent evals. It includes:

    \b
    • failure-modes: 10 scenarios testing resilience (timeouts, errors, loops)
    • security: 5 scenarios testing injection/jailbreak resistance

    \b
    Quick start:
        1. Start the gym agent:
           cd gym/agents/support-bot && langgraph dev

        2. Run all scenarios:
           evalview gym

    \b
    Examples:
        evalview gym                        # Run all scenarios
        evalview gym --suite failure-modes  # Resilience tests only
        evalview gym --suite security       # Security tests only
        evalview gym --list-only            # List without running
    """
    console.print("[blue]━━━ EvalView Gym ━━━[/blue]\n")
    console.print("[dim]Practice environment for learning agent eval patterns[/dim]\n")

    # Find gym directory
    gym_paths = [
        Path("gym"),  # From repo root
        Path(__file__).parent.parent.parent / "gym",  # Relative to evalview package
    ]

    gym_dir = None
    for path in gym_paths:
        if path.exists():
            gym_dir = path
            break

    if not gym_dir:
        console.print("[red]Error: gym/ directory not found.[/red]")
        console.print("[dim]Make sure you're in the EvalView repo root or gym is installed.[/dim]")
        sys.exit(1)

    # Collect scenarios based on suite
    scenarios = []

    if suite in ("all", "failure-modes"):
        fm_dir = gym_dir / "failure-modes"
        if fm_dir.exists():
            scenarios.extend(sorted(fm_dir.glob("*.yaml")))

    if suite in ("all", "security"):
        sec_dir = gym_dir / "security"
        if sec_dir.exists():
            scenarios.extend(sorted(sec_dir.glob("*.yaml")))

    if not scenarios:
        console.print(f"[yellow]No scenarios found for suite: {suite}[/yellow]")
        sys.exit(1)

    # List only mode
    if list_only:
        console.print(f"[cyan]Scenarios in suite '{suite}':[/cyan]\n")

        for scenario_path in scenarios:
            try:
                with open(scenario_path) as f:
                    data = yaml.safe_load(f)
                name = data.get("name", scenario_path.stem)
                desc = data.get("description", "").split("\n")[0][:60]
                suite_name = scenario_path.parent.name
                console.print(f"  [{suite_name}] [bold]{name}[/bold]")
                if desc:
                    console.print(f"           [dim]{desc}[/dim]")
            except Exception:
                console.print(f"  [red]{scenario_path.name}[/red] (failed to parse)")

        console.print(f"\n[dim]Total: {len(scenarios)} scenarios[/dim]")
        return

    # Run scenarios
    console.print(f"Running {len(scenarios)} scenarios against {endpoint}\n")

    # Check if endpoint is reachable
    try:
        httpx.get(f"{endpoint.rstrip('/')}/health", timeout=5.0)
        console.print("[green]✓ Agent endpoint reachable[/green]\n")
    except Exception:
        console.print("[yellow]⚠ Could not reach agent endpoint[/yellow]")
        console.print(f"[dim]  Make sure your agent is running at {endpoint}[/dim]")
        console.print("[dim]  Start with: cd gym/agents/support-bot && langgraph dev[/dim]\n")

        if not click.confirm("Continue anyway?", default=False):
            sys.exit(1)

    # Run each scenario
    passed = 0
    failed = 0
    errors = 0

    for scenario_path in scenarios:
        try:
            with open(scenario_path) as f:
                data = yaml.safe_load(f)

            name = data.get("name", scenario_path.stem)
            suite_name = scenario_path.parent.name

            # Override endpoint
            data["endpoint"] = endpoint

            console.print(f"[dim][{suite_name}][/dim] {name}... ", end="")

            # Run the test using existing infrastructure
            from evalview.adapters.http_adapter import HTTPAdapter
            from evalview.core.loader import TestCaseLoader
            from evalview.evaluators.evaluator import Evaluator

            async def _run_gym_scenario() -> bool:
                tc = TestCaseLoader.load_from_file(scenario_path)
                tc.endpoint = endpoint
                adapter = HTTPAdapter(endpoint=endpoint)
                trace = await adapter.execute(tc.input.query, tc.input.context)
                eval_result = await Evaluator().evaluate(tc, trace)
                return eval_result.passed

            if asyncio.run(_run_gym_scenario()):
                console.print("[green]PASS[/green]")
                passed += 1
            else:
                console.print("[red]FAIL[/red]")
                failed += 1

        except Exception as e:
            console.print("[red]ERROR[/red]")
            console.print(f"    [dim]{str(e)[:80]}[/dim]")
            errors += 1

    # Summary
    console.print()
    console.print("[cyan]━━━ Summary ━━━[/cyan]")
    total = passed + failed + errors
    console.print(f"  Passed:  [green]{passed}[/green]/{total}")
    console.print(f"  Failed:  [red]{failed}[/red]/{total}")
    if errors:
        console.print(f"  Errors:  [yellow]{errors}[/yellow]/{total}")

    if failed > 0 or errors > 0:
        sys.exit(1)
