"""Golden trace management commands."""
from __future__ import annotations

import click

from evalview.commands.shared import console
from evalview.telemetry.decorators import track_command


@click.group(hidden=True)
def golden():
    """Manage golden traces (blessed baselines for regression detection).

    Golden traces are "blessed" test results that represent expected behavior.
    Use them with `evalview run --diff` to detect regressions.

    Examples:
        evalview golden save .evalview/results/2024-01-15T10:30:00.json
        evalview golden list
        evalview golden delete "My Test Case"
    """
    pass


@golden.command("save")
@click.argument("result_file", type=click.Path(exists=True))
@click.option("--notes", "-n", help="Notes about why this is the golden baseline")
@click.option("--test", "-t", help="Save only specific test (by name)")
@click.option("--variant", "-v", help="Save as named variant (for multi-reference goldens)")
@track_command("golden_save")
def golden_save(result_file: str, notes: str, test: str, variant: str):
    """Save a test result as the golden baseline.

    RESULT_FILE is a JSON file from `evalview run` (e.g., .evalview/results/xxx.json)

    Examples:
        evalview golden save .evalview/results/latest.json
        evalview golden save results.json --notes "v1.0 release baseline"
        evalview golden save results.json --test "List Directory Contents"
        evalview golden save results.json --variant "fast-path" --notes "Optimized path"
    """
    import json
    from evalview.core.golden import GoldenStore
    from evalview.core.types import EvaluationResult

    console.print("\n[cyan]━━━ Saving Golden Trace ━━━[/cyan]\n")

    # Load result file
    with open(result_file) as f:
        data = json.load(f)

    # Handle both single result and batch results
    results = []
    if type(data).__name__ == "list":
        results = data
    elif isinstance(data, dict) and "results" in data:
        results = data["results"]
    else:
        results = [data]

    # Filter by test name if specified
    if test:
        results = [r for r in results if r.get("test_case") == test]
        if not results:
            console.print(f"[red]❌ No test found with name: {test}[/red]")
            return

    store = GoldenStore()

    for result_data in results:
        try:
            result = EvaluationResult.model_validate(result_data)

            # Check if golden already exists (for default or this specific variant)
            variant_exists = store._get_golden_path(result.test_case, variant).exists()
            if variant_exists:
                variant_label = f"variant '{variant}'" if variant else "default golden"
                if not click.confirm(
                    f"{variant_label.capitalize()} already exists for '{result.test_case}'. Overwrite?",
                    default=False,
                ):
                    console.print(f"[yellow]Skipped: {result.test_case}[/yellow]")
                    continue

            # Save golden (may raise ValueError if too many variants)
            try:
                path = store.save_golden(result, notes=notes, source_file=result_file, variant_name=variant)
            except ValueError as e:
                console.print(f"[red]❌ {e}[/red]")
                continue

            variant_label = f" (variant: {variant})" if variant else ""
            console.print(f"[green]✓ Saved golden:[/green] {result.test_case}{variant_label}")
            console.print(f"  [dim]Score: {result.score:.1f}[/dim]")
            console.print(f"  [dim]Tools: {len(result.trace.steps)} steps[/dim]")
            console.print(f"  [dim]File: {path}[/dim]")
            console.print()

        except Exception as e:
            console.print(f"[red]❌ Failed to save: {e}[/red]")

    console.print("[green]Done![/green]")
    console.print()
    console.print("[dim]⭐ EvalView saved your baseline! Star if it helped → github.com/hidai25/eval-view[/dim]\n")


@golden.command("list")
@track_command("golden_list")
def golden_list():
    """List all golden traces.

    Shows all saved golden baselines with metadata and variant counts.
    """
    from evalview.core.golden import GoldenStore

    store = GoldenStore()
    goldens_with_variants = store.list_golden_with_variants()

    if not goldens_with_variants:
        console.print("\n[yellow]No golden traces found.[/yellow]")
        console.print("[dim]Save one with: evalview golden save <result.json>[/dim]\n")
        return

    console.print("\n[cyan]━━━ Golden Traces ━━━[/cyan]\n")

    for item in sorted(goldens_with_variants, key=lambda x: x["metadata"].test_name):
        g = item["metadata"]
        variant_count = item["variant_count"]

        variant_label = f" ({variant_count} variants)" if variant_count > 1 else ""
        console.print(f"  [bold]{g.test_name}[/bold]{variant_label}")
        console.print(f"    [dim]Score: {g.score:.1f}[/dim]")
        console.print(f"    [dim]Blessed: {g.blessed_at.strftime('%Y-%m-%d %H:%M')}[/dim]")
        if g.notes:
            console.print(f"    [dim]Notes: {g.notes}[/dim]")
        console.print()

    console.print(f"[dim]Total: {len(goldens_with_variants)} test(s) with golden trace(s)[/dim]\n")


@golden.command("delete")
@click.argument("test_name")
@click.option("--force", "-f", is_flag=True, help="Skip confirmation")
@track_command("golden_delete")
def golden_delete(test_name: str, force: bool):
    """Delete a golden trace.

    TEST_NAME is the name of the test case to delete.
    """
    from evalview.core.golden import GoldenStore

    store = GoldenStore()

    if not store.has_golden(test_name):
        console.print(f"\n[yellow]No golden trace found for: {test_name}[/yellow]\n")
        return

    if not force:
        if not click.confirm(f"Delete golden trace for '{test_name}'?", default=False):
            console.print("[dim]Cancelled[/dim]")
            return

    store.delete_golden(test_name)
    console.print(f"\n[green]✓ Deleted golden trace: {test_name}[/green]\n")


@golden.command("show")
@click.argument("test_name")
@track_command("golden_show")
def golden_show(test_name: str):
    """Show details of a golden trace.

    TEST_NAME is the name of the test case.
    """
    from evalview.core.golden import GoldenStore
    from rich.panel import Panel

    store = GoldenStore()
    golden = store.load_golden(test_name)

    if not golden:
        console.print(f"\n[yellow]No golden trace found for: {test_name}[/yellow]\n")
        return

    console.print(f"\n[cyan]━━━ Golden Trace: {test_name} ━━━[/cyan]\n")

    # Metadata
    console.print("[bold]Metadata:[/bold]")
    console.print(f"  Score: {golden.metadata.score:.1f}")
    console.print(f"  Blessed: {golden.metadata.blessed_at.strftime('%Y-%m-%d %H:%M')}")
    console.print(f"  Source: {golden.metadata.source_result_file or 'N/A'}")
    if golden.metadata.notes:
        console.print(f"  Notes: {golden.metadata.notes}")
    console.print()

    # Tool sequence
    console.print("[bold]Tool Sequence:[/bold]")
    for i, tool in enumerate(golden.tool_sequence, 1):
        console.print(f"  {i}. {tool}")
    console.print()

    # Output preview
    console.print("[bold]Output Preview:[/bold]")
    preview = golden.trace.final_output[:500]
    if len(golden.trace.final_output) > 500:
        preview += "..."
    console.print(Panel(preview, border_style="dim"))
    console.print()
