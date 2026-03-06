"""Expand command — generate test variations using LLM."""
from __future__ import annotations

import asyncio
import re
from pathlib import Path
from typing import Optional

import click

from evalview.commands.shared import console
from evalview.telemetry.decorators import track_command


@click.command("expand")
@click.argument("test_file", type=click.Path(exists=True))
@click.option(
    "--count",
    "-n",
    default=10,
    type=int,
    help="Number of variations to generate (default: 10)",
)
@click.option(
    "--output-dir",
    "-o",
    type=click.Path(),
    help="Output directory for generated tests (default: same as input)",
)
@click.option(
    "--edge-cases/--no-edge-cases",
    default=True,
    help="Include edge case variations (default: True)",
)
@click.option(
    "--focus",
    "-f",
    help="Focus variations on specific aspect (e.g., 'different stock tickers')",
)
@click.option(
    "--dry-run",
    is_flag=True,
    help="Preview generated tests without saving",
)
@track_command("expand", lambda **kw: {"count": kw.get("count"), "edge_cases": kw.get("edge_cases")})
def expand(test_file: str, count: int, output_dir: str, edge_cases: bool, focus: str, dry_run: bool):
    """Expand a test case into variations using LLM.

    Takes a base test case and generates variations with different inputs,
    edge cases, and scenarios. Great for building comprehensive test suites
    from a few seed tests.

    Example:
        evalview expand tests/test-cases/stock-basic.yaml --count 20
    """
    asyncio.run(_expand_async(test_file, count, output_dir, edge_cases, focus, dry_run))


async def _expand_async(
    test_file: str,
    count: int,
    output_dir: str,
    edge_cases: bool,
    focus: str,
    dry_run: bool,
):
    """Async implementation of expand command."""
    from evalview.expander import TestExpander
    from evalview.core.loader import TestCaseLoader
    from rich.table import Table

    console.print("[blue]🔄 Expanding test case...[/blue]\n")

    # Load base test
    test_path = Path(test_file)
    console.print(f"[dim]Loading: {test_path}[/dim]")

    try:
        base_test = TestCaseLoader.load_from_file(test_path)
        if not base_test:
            console.print(f"[red]❌ No test cases found in {test_file}[/red]")
            return
    except Exception as e:
        console.print(f"[red]❌ Failed to load test: {e}[/red]")
        return

    console.print(f"[green]✓[/green] Base test: [bold]{base_test.name}[/bold]")
    console.print(f"  Query: \"{base_test.input.query}\"")
    console.print()

    # Initialize expander
    try:
        expander = TestExpander()
    except ValueError as e:
        console.print(f"[red]❌ {e}[/red]")
        return

    # Show provider info
    if expander.message:
        console.print(f"[yellow]ℹ️  {expander.message}[/yellow]")
    console.print(f"[dim]Using {expander.provider.capitalize()} for test generation[/dim]")
    console.print()

    # Generate variations
    console.print(f"[cyan]🤖 Generating {count} variations...[/cyan]")
    if focus:
        console.print(f"[dim]   Focus: {focus}[/dim]")
    if edge_cases:
        console.print("[dim]   Including edge cases[/dim]")
    console.print()

    try:
        variations = await expander.expand(
            base_test,
            count=count,
            include_edge_cases=edge_cases,
            variation_focus=focus,
        )
    except Exception as e:
        console.print(f"[red]❌ Failed to generate variations: {e}[/red]")
        console.print("[dim]Make sure OPENAI_API_KEY or ANTHROPIC_API_KEY is set[/dim]")
        return

    if not variations:
        console.print("[yellow]⚠️  No variations generated[/yellow]")
        return

    console.print(f"[green]✓[/green] Generated {len(variations)} variations\n")

    # Convert to TestCase objects
    test_cases = [
        expander.convert_to_test_case(v, base_test, i)
        for i, v in enumerate(variations, 1)
    ]

    # Show preview table
    table = Table(title="Generated Test Variations", show_header=True, header_style="bold cyan")
    table.add_column("#", style="dim", width=3)
    table.add_column("Name", style="white", no_wrap=False)
    table.add_column("Query", style="dim", no_wrap=False)
    table.add_column("Edge?", style="yellow", justify="center", width=5)

    for i, (variation, tc) in enumerate(zip(variations, test_cases), 1):
        is_edge = "⚠️" if variation.get("is_edge_case") else ""
        query_preview = tc.input.query[:50] + "..." if len(tc.input.query) > 50 else tc.input.query
        table.add_row(str(i), tc.name, query_preview, is_edge)

    console.print(table)
    console.print()

    if dry_run:
        console.print("[yellow]Dry run - no files saved[/yellow]")
        return

    # Ask for confirmation
    if not click.confirm("Save these test variations?", default=True):
        console.print("[yellow]Cancelled[/yellow]")
        return

    # Determine output directory
    if output_dir:
        out_path = Path(output_dir)
    else:
        out_path = test_path.parent

    # Generate prefix from base test name
    prefix = re.sub(r'[^a-z0-9]+', '-', base_test.name.lower()).strip('-')[:20]
    prefix = f"{prefix}-var"

    # Save variations
    console.print(f"\n[cyan]💾 Saving to {out_path}/...[/cyan]")
    saved_paths = expander.save_variations(test_cases, out_path, prefix=prefix)

    console.print(f"\n[green]✅ Saved {len(saved_paths)} test variations:[/green]")
    for path in saved_paths[:5]:  # Show first 5
        console.print(f"   • {path.name}")
    if len(saved_paths) > 5:
        console.print(f"   • ... and {len(saved_paths) - 5} more")

    # Suggest run command with correct path (use --pattern for file matching)
    console.print(f"\n[blue]Run with:[/blue] evalview run {out_path} --pattern '{prefix}*.yaml'")
