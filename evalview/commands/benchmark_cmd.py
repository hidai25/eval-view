"""Benchmark command — run curated benchmark packs against configured agent."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import click

from evalview.commands.shared import (
    console,
    _execute_snapshot_tests,
    _load_config_if_exists,
)
from evalview.telemetry.decorators import track_command


@click.command("benchmark")
@click.argument(
    "domain",
    type=click.Choice(["rag", "coding", "customer-support", "research", "all"]),
    required=False,
    default=None,
)
@click.option(
    "--list", "list_only",
    is_flag=True,
    help="List available benchmark domains without running",
)
@click.option(
    "--export-only",
    is_flag=True,
    help="Export test YAMLs to tests/benchmarks/<domain>/ without running",
)
@click.option(
    "--output-dir", "output_dir",
    default=None,
    help="Override output directory for --export-only (default: tests/benchmarks/<domain>)",
)
@click.option("--no-browser", is_flag=True, help="Don't auto-open the HTML report")
@track_command("benchmark")
def benchmark_cmd(
    domain: Optional[str],
    list_only: bool,
    export_only: bool,
    output_dir: Optional[str],
    no_browser: bool,
) -> None:
    """Run a curated benchmark against your configured agent.

    Benchmarks are portable test packs that measure agent quality on
    common tasks (RAG, coding, support) and produce a comparable score.

    \b
    Examples:
        evalview benchmark rag               # Run RAG benchmark
        evalview benchmark coding            # Run coding benchmark
        evalview benchmark all               # Run all domains
        evalview benchmark rag --export-only # Export YAMLs only
        evalview benchmark --list            # Show available domains
    """
    import tempfile
    from evalview.benchmarks import DOMAINS, get_pack, write_pack_yaml
    from evalview.core.loader import TestCaseLoader

    if list_only or domain is None:
        console.print("\n[bold]Available benchmark domains:[/bold]\n")
        for d, desc in DOMAINS.items():
            console.print(f"  [cyan]{d:<20}[/cyan] {desc}")
        console.print(
            "\n[dim]Run: evalview benchmark <domain>[/dim]\n"
        )
        return

    domains_to_run = list(DOMAINS.keys()) if domain == "all" else [domain]

    for d in domains_to_run:
        cases = get_pack(d)
        console.print(f"\n[cyan]◈ Benchmark: {d}[/cyan]  ({len(cases)} tests)\n")

        if export_only:
            dest = Path(output_dir) if output_dir else Path("tests") / "benchmarks" / d
            written = write_pack_yaml(d, dest)
            console.print(f"  [green]✓ {len(written)} files exported to {dest}/[/green]")
            console.print(f"  [dim]Run: evalview snapshot --test-path {dest}[/dim]\n")
            continue

        # Check agent is configured
        config = _load_config_if_exists()
        if not config or not getattr(config, "endpoint", None):
            console.print(
                "  [yellow]⚠ No agent configured.[/yellow] "
                "Run [bold]evalview init[/bold] first, or use "
                "[bold]--export-only[/bold] to just export the test files.\n"
            )
            continue

        # Write to a temp dir so benchmark YAMLs don't pollute the user's tests/
        with tempfile.TemporaryDirectory() as _tmpdir:
            tmp_dest = Path(_tmpdir)
            write_pack_yaml(d, tmp_dest)
            loader = TestCaseLoader()
            try:
                test_cases = loader.load_from_directory(tmp_dest)
            except Exception as e:
                console.print(f"  [red]❌ Failed to load benchmark tests: {e}[/red]\n")
                continue

        if not test_cases:
            console.print("  [yellow]No test cases loaded.[/yellow]\n")
            continue

        console.print(f"  Running {len(test_cases)} tests against {config.endpoint}...\n")
        results = _execute_snapshot_tests(test_cases, config)

        if not results:
            console.print("  [red]No results — check your agent is running.[/red]\n")
            continue

        # Score report
        passed = sum(1 for r in results if r.passed)
        scores = [r.score for r in results]
        avg_score = sum(scores) / len(scores) if scores else 0
        pass_rate = passed / len(results) * 100 if results else 0

        score_color = "green" if avg_score >= 70 else "yellow" if avg_score >= 50 else "red"
        console.print(
            f"\n  [bold]Benchmark: {d}[/bold]  "
            f"[{score_color}]{avg_score:.1f}/100[/{score_color}]  "
            f"({passed}/{len(results)} passed, {pass_rate:.0f}% pass rate)\n"
        )

        # Per-difficulty breakdown
        by_diff: Dict[str, List[float]] = {}
        for tc, r in zip(test_cases, results):
            diff = getattr(tc, "difficulty", None) or "medium"
            by_diff.setdefault(diff, []).append(r.score)

        diff_order = ["trivial", "easy", "medium", "hard", "expert"]
        for diff in diff_order:
            if diff not in by_diff:
                continue
            diff_scores = by_diff[diff]
            diff_avg = sum(diff_scores) / len(diff_scores)
            bar_len = int(diff_avg / 5)
            bar = "█" * bar_len + "░" * (20 - bar_len)
            c = "green" if diff_avg >= 70 else "yellow" if diff_avg >= 50 else "red"
            console.print(f"  {diff:<8} [{c}]{bar}[/{c}] {diff_avg:.0f}/100")

        console.print()

        # Generate HTML report
        from evalview.visualization import generate_visual_report
        report_path = generate_visual_report(
            results=results,
            auto_open=not no_browser,
            title=f"Benchmark: {d}",
        )
        console.print(f"  [green]◈ Report:[/green] {report_path}\n")
