"""Visual commands — inspect, visualize, compare."""
from __future__ import annotations

import asyncio
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import click
import yaml
from rich.panel import Panel
from rich.table import Table

from evalview.commands.shared import (
    console,
    _create_adapter,
)
from evalview.telemetry.decorators import track_command


@click.command("inspect")
@click.argument("target", default="latest", required=False)
@click.option("--title", default="EvalView Report", help="Report title")
@click.option("--notes", default="", help="Optional note shown in the report header")
@click.option("--no-open", is_flag=True, help="Do not auto-open in browser")
@click.option("--output", "-o", default=None, help="Output HTML path")
def inspect_cmd(target: str, title: str, notes: str, no_open: bool, output: Optional[str]) -> None:
    """Generate a beautiful visual HTML report and open it in the browser.

    TARGET can be 'latest' (default), a path to a results JSON file, or a
    timestamp string matching a file in .evalview/results/.

    \b
    Examples:
        evalview inspect
        evalview inspect latest
        evalview inspect .evalview/results/20260221_103000.json
        evalview inspect latest --notes "after refactor PR #42"
    """
    import glob as _glob

    from evalview.reporters.json_reporter import JSONReporter
    from evalview.visualization import generate_visual_report

    # Resolve target to a results file
    results_file: Optional[str] = None
    if target == "latest" or not target:
        files = sorted(_glob.glob(".evalview/results/*.json"))
        if not files:
            console.print("\n[red]No results found in .evalview/results/[/red]")
            console.print("[dim]Run [bold]evalview run[/bold] or [bold]evalview snapshot[/bold] first.[/dim]\n")
            raise SystemExit(1)
        results_file = files[-1]
    elif os.path.exists(target):
        results_file = target
    else:
        # Try as partial timestamp match
        matches = sorted(_glob.glob(f".evalview/results/*{target}*.json"))
        if not matches:
            console.print(f"\n[red]No results file found matching: {target}[/red]\n")
            raise SystemExit(1)
        results_file = matches[-1]

    console.print(f"\n[cyan]◈ Generating visual report from {results_file}...[/cyan]")

    try:
        results = JSONReporter.load_as_results(results_file)
    except Exception as exc:
        console.print(f"[red]Failed to load results: {exc}[/red]\n")
        raise SystemExit(1)

    path = generate_visual_report(
        results=results,
        title=title,
        notes=notes,
        output_path=output,
        auto_open=not no_open,
    )

    total = len(results)
    passed = sum(1 for r in results if r.passed)
    rate = round(passed / total * 100) if total else 0

    console.print(f"[green]✓ Report generated:[/green] {path}")
    console.print(f"  {passed}/{total} tests passing ({rate}%)\n")


@click.command("visualize")
@click.argument("target", default="latest", required=False)
@click.option("--compare", "-c", multiple=True, help="Additional result files to compare (use multiple times)")
@click.option("--title", default=None, help="Report title (auto-generated if omitted)")
@click.option("--notes", default="", help="Optional note shown in the report header")
@click.option("--no-open", is_flag=True, help="Do not auto-open in browser (useful in CI)")
@click.option("--output", "-o", default=None, help="Output HTML path")
def visualize_cmd(target: str, compare: tuple, title: Optional[str], notes: str, no_open: bool, output: Optional[str]) -> None:
    """Generate a visual HTML report, optionally comparing multiple runs.

    TARGET can be 'latest' (default), a path to a results JSON file, or a
    partial timestamp string matching a file in .evalview/results/.

    Use --compare to add more runs for side-by-side comparison.

    \b
    Examples:
        evalview visualize
        evalview visualize latest
        evalview visualize latest --compare .evalview/results/20260220_110044.json
        evalview visualize --compare run1.json --compare run2.json --compare run3.json
        evalview visualize latest --notes "after PR #42" --no-open
    """
    import glob as _glob

    from evalview.reporters.json_reporter import JSONReporter
    from evalview.visualization import generate_visual_report

    def _resolve(t: str) -> Optional[str]:
        if t == "latest":
            files = sorted(_glob.glob(".evalview/results/*.json"))
            return files[-1] if files else None
        if os.path.exists(t):
            return t
        matches = sorted(_glob.glob(f".evalview/results/*{t}*.json"))
        return matches[-1] if matches else None

    # Resolve primary target
    primary = _resolve(target)
    if not primary:
        console.print("\n[red]No results found.[/red] Run [bold]evalview run[/bold] first.\n")
        raise SystemExit(1)

    # Resolve comparison targets
    compare_files = []
    for c in compare:
        r = _resolve(c)
        if r:
            compare_files.append(r)
        else:
            console.print(f"[yellow]⚠ Could not find comparison file: {c}[/yellow]")

    all_files = [primary] + compare_files
    is_multi = len(all_files) > 1

    console.print(f"\n[cyan]◈ Generating visual report{'s' if is_multi else ''} from {len(all_files)} run{'s' if is_multi else ''}...[/cyan]")

    try:
        all_results = [JSONReporter.load_as_results(f) for f in all_files]
    except Exception as exc:
        console.print(f"[red]Failed to load results: {exc}[/red]\n")
        raise SystemExit(1)

    # Primary results are first; flatten for single-run view
    results = all_results[0]

    auto_title = title or (
        f"Comparison: {len(all_files)} runs" if is_multi else "EvalView Report"
    )
    auto_notes = notes or (
        " · ".join(os.path.basename(f).replace(".json", "") for f in all_files)
        if is_multi else notes
    )

    path = generate_visual_report(
        results=results,
        compare_results=all_results[1:] if is_multi else None,
        compare_labels=[os.path.basename(f).replace(".json", "") for f in all_files],
        title=auto_title,
        notes=auto_notes,
        output_path=output,
        auto_open=not no_open,
    )

    total = len(results)
    passed = sum(1 for r in results if r.passed)
    rate = round(passed / total * 100) if total else 0

    console.print(f"[green]✓ Report generated:[/green] {path}")
    if is_multi:
        console.print(f"  Comparing {len(all_files)} runs — primary: {passed}/{total} passing ({rate}%)\n")
    else:
        console.print(f"  {passed}/{total} tests passing ({rate}%)\n")


def _sanitize_label(label: str) -> str:
    """Make a label safe for use in filenames."""
    return re.sub(r"[^\w\-]", "_", label)


async def _compare_async(
    v1: str,
    v2: str,
    tests_path: str,
    adapter_type: Optional[str],
    label_v1: str,
    label_v2: str,
    no_open: bool,
    no_judge: bool,
) -> None:
    """Run same tests against two endpoints and generate a side-by-side comparison."""
    from evalview.core.parallel import execute_tests_parallel
    from evalview.core.loader import TestCaseLoader
    from evalview.evaluators.evaluator import Evaluator
    from evalview.core.llm_provider import get_or_select_provider
    from evalview.reporters.json_reporter import JSONReporter
    from evalview.visualization import generate_visual_report
    from evalview.skills.ui_utils import print_evalview_banner

    print_evalview_banner(console, subtitle="[dim]A/B Endpoint Comparison[/dim]")

    # Load config for adapter type / timeout fallback
    config_path = Path(".evalview/config.yaml")
    config: Dict[str, Any] = {}
    if config_path.exists():
        with open(config_path) as f:
            config = yaml.safe_load(f) or {}

    _adapter_type = adapter_type or config.get("adapter", "http")
    timeout = float(config.get("timeout", 30.0))

    # Provider selection (once, shared across both runs)
    if no_judge:
        console.print("[yellow]⚠  --no-judge: skipping LLM-as-judge. Deterministic scoring only.[/yellow]\n")
    else:
        if get_or_select_provider(console) is None:
            return

    evaluator = Evaluator(skip_llm_judge=no_judge)

    # Load test cases
    tests_dir = Path(tests_path)
    if not tests_dir.exists():
        console.print(f"[red]❌ Tests directory not found: {tests_path}[/red]")
        console.print("[dim]Tip: use --tests to specify a different directory[/dim]")
        return

    test_cases = TestCaseLoader.load_from_directory(tests_dir, "*.yaml")
    if not test_cases:
        console.print(f"[yellow]⚠  No test cases found in {tests_path}[/yellow]")
        return

    console.print(f"[blue]Loaded {len(test_cases)} test case(s)[/blue]\n")

    # Run all tests against a single endpoint; returns (results, error_count)
    async def _run_endpoint(endpoint: str, label: str) -> Tuple[List[Any], int]:
        console.print(f"[cyan]◈ Running against {label}:[/cyan] [dim]{endpoint}[/dim]")
        adapter = _create_adapter(_adapter_type, endpoint, timeout=timeout, allow_private_urls=True)

        async def _execute(test_case: Any) -> Any:
            context = dict(test_case.input.context) if test_case.input.context else {}
            trace = await adapter.execute(test_case.input.query, context)
            return await evaluator.evaluate(test_case, trace, adapter_name=getattr(adapter, "name", None))

        def _on_complete(name: str, ok: bool, res: Any) -> None:
            if not ok or res is None:
                console.print(f"  [yellow]⚠[/yellow] {name} [dim](execution error)[/dim]")
            elif res.passed:
                console.print(f"  [green]✓[/green] {name}")
            else:
                console.print(f"  [red]✗[/red] {name}")

        parallel_results = await execute_tests_parallel(
            test_cases,
            _execute,
            max_workers=8,
            on_complete=_on_complete,
        )

        results = [pr.result for pr in parallel_results if pr.result is not None]
        error_count = sum(1 for pr in parallel_results if pr.result is None)
        passed_count = sum(1 for r in results if r.passed)

        summary = f"  [dim]{passed_count}/{len(results)} passing"
        if error_count:
            summary += f", {error_count} error(s)"
        console.print(summary + "[/dim]\n")

        return results, error_count

    results_v1, errors_v1 = await _run_endpoint(v1, label_v1)
    results_v2, errors_v2 = await _run_endpoint(v2, label_v2)

    if not results_v1 or not results_v2:
        console.print("[red]❌ No results to compare.[/red]")
        return

    # Save both result sets
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    slug_v1 = _sanitize_label(label_v1)
    slug_v2 = _sanitize_label(label_v2)
    output_dir = Path(".evalview/results")
    output_dir.mkdir(parents=True, exist_ok=True)
    file_v1 = output_dir / f"compare_{ts}_{slug_v1}.json"
    file_v2 = output_dir / f"compare_{ts}_{slug_v2}.json"
    JSONReporter.save(results_v1, file_v1)
    JSONReporter.save(results_v2, file_v2)

    # Generate comparison visual report
    report_path = generate_visual_report(
        results=results_v1,
        compare_results=[results_v2],
        compare_labels=[label_v1, label_v2],
        title=f"Comparison: {label_v1} vs {label_v2}",
        notes=f"{v1}  →  {v2}",
        auto_open=not no_open and not os.environ.get("CI"),
    )
    console.print(f"[green]✓ Comparison report:[/green] {report_path}\n")

    # Per-test summary table
    v1_by_name = {r.test_case: r for r in results_v1}
    v2_by_name = {r.test_case: r for r in results_v2}

    only_in_v2 = set(v2_by_name) - set(v1_by_name)
    if only_in_v2:
        console.print(f"[yellow]⚠  Tests only in {label_v2} (not compared): {', '.join(sorted(only_in_v2))}[/yellow]\n")

    table = Table(show_header=True, header_style="bold cyan")
    table.add_column("Test", min_width=20)
    table.add_column(label_v1, justify="center")
    table.add_column(label_v2, justify="center")
    table.add_column("Δ Score", justify="right")
    table.add_column("Verdict", justify="center")

    improved = degraded = unchanged = 0
    for name in sorted(v1_by_name):
        r1 = v1_by_name[name]
        r2 = v2_by_name.get(name)
        if r2 is None:
            table.add_row(name, f"{r1.score:.1f}", "[dim]—[/dim]", "[dim]—[/dim]", "[dim]missing[/dim]")
            continue

        delta = r2.score - r1.score
        if delta > 1:
            delta_str = f"[green]+{delta:.1f}[/green]"
            verdict = "[green]improved[/green]"
            improved += 1
        elif delta < -1:
            delta_str = f"[red]{delta:.1f}[/red]"
            verdict = "[red]degraded[/red]"
            degraded += 1
        else:
            delta_str = "[dim]≈0[/dim]"
            verdict = "[dim]same[/dim]"
            unchanged += 1

        s1 = f"[green]{r1.score:.1f}[/green]" if r1.passed else f"[red]{r1.score:.1f}[/red]"
        s2 = f"[green]{r2.score:.1f}[/green]" if r2.passed else f"[red]{r2.score:.1f}[/red]"
        table.add_row(name, s1, s2, delta_str, verdict)

    console.print(table)
    console.print()

    # Overall verdict
    total = improved + degraded + unchanged
    if degraded == 0 and improved > 0:
        console.print(Panel(
            f"[green]{label_v2} is better[/green] — improved {improved}/{total} test(s), no regressions.\n"
            f"[dim]Safe to promote {label_v2} to production.[/dim]",
            border_style="green",
        ))
    elif degraded > 0 and improved == 0:
        console.print(Panel(
            f"[red]{label_v2} is worse[/red] — degraded {degraded}/{total} test(s), no improvements.\n"
            f"[dim]Do not promote {label_v2} to production.[/dim]",
            border_style="red",
        ))
    elif degraded > 0:
        console.print(Panel(
            f"[yellow]Mixed results[/yellow] — {improved} improved, {degraded} degraded, {unchanged} unchanged.\n"
            f"[dim]Review degraded tests before promoting {label_v2}.[/dim]",
            border_style="yellow",
        ))
    else:
        console.print(f"[dim]No meaningful difference between {label_v1} and {label_v2}.[/dim]")
    console.print()


@click.command("compare")
@click.option("--v1", required=True, help="Baseline agent endpoint URL")
@click.option("--v2", required=True, help="Candidate agent endpoint URL")
@click.option("--tests", "tests_path", default="tests", show_default=True, help="Test directory")
@click.option("--adapter", "adapter_type", default=None, help="Adapter type (default: from config or http)")
@click.option("--label-v1", default="baseline", show_default=True, help="Label for v1 in the report")
@click.option("--label-v2", default="candidate", show_default=True, help="Label for v2 in the report")
@click.option("--no-open", is_flag=True, help="Don't auto-open report in browser")
@click.option("--no-judge", is_flag=True, help="Skip LLM-as-judge, use deterministic scoring only")
@track_command("compare")
def compare_cmd(
    v1: str,
    v2: str,
    tests_path: str,
    adapter_type: Optional[str],
    label_v1: str,
    label_v2: str,
    no_open: bool,
    no_judge: bool,
) -> None:
    """Run the same tests against two agent endpoints and compare results.

    \b
    Examples:
        evalview compare --v1 http://localhost:8000 --v2 http://localhost:8001
        evalview compare --v1 http://staging-v1/agent --v2 http://staging-v2/agent --label-v1 gpt4o --label-v2 o3
        evalview compare --v1 http://old --v2 http://new --no-judge --no-open
    """
    asyncio.run(_compare_async(v1, v2, tests_path, adapter_type, label_v1, label_v2, no_open, no_judge))
