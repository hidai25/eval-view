"""Reporting functions for the run command.

Handles:
- Golden diff display (tool diffs, parameter diffs, score/output deltas)
- Regression analysis (compare-baseline mode)
- Summary and coverage reports
- Saving results to disk (JSON + golden baseline)
- HTML report generation
- CI exit-code computation
- Trust-framing summary
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple


def display_diff_results(
    diffs_found: List[Tuple[str, Any]],
    results: List[Any],
    console: Any,
) -> None:
    """Print the golden diff report and any CI urgency messages."""
    from evalview.core.diff import DiffStatus

    if not diffs_found:
        _print_no_diffs(results, console)
        return

    console.print("\n[bold cyan]━━━ Golden Diff Report ━━━[/bold cyan]\n")

    for test_name, trace_diff in diffs_found:
        status = trace_diff.overall_severity
        if status == DiffStatus.REGRESSION:
            icon = "[red]✗ REGRESSION[/red]"
        elif status == DiffStatus.TOOLS_CHANGED:
            icon = "[yellow]⚠ TOOLS_CHANGED[/yellow]"
        elif status == DiffStatus.OUTPUT_CHANGED:
            icon = "[dim]~ OUTPUT_CHANGED[/dim]"
        else:
            icon = "[green]✓ PASSED[/green]"

        console.print(f"{icon} [bold]{test_name}[/bold]")
        console.print(f"    Summary: {trace_diff.summary()}")

        if trace_diff.tool_diffs:
            console.print("    [bold]Tool Changes:[/bold]")
            for td in trace_diff.tool_diffs[:5]:
                if td.type == "added":
                    console.print(f"      [green]+ {td.actual_tool}[/green] (new step)")
                elif td.type == "removed":
                    console.print(f"      [red]- {td.golden_tool}[/red] (missing)")
                elif td.type == "changed":
                    if td.golden_tool == td.actual_tool and td.parameter_diffs:
                        console.print(f"      [yellow]~ {td.golden_tool}[/yellow] (parameters changed)")
                    else:
                        console.print(f"      [yellow]~ {td.golden_tool} -> {td.actual_tool}[/yellow]")

                if td.parameter_diffs:
                    console.print("        [dim]Parameter differences:[/dim]")
                    for pd in td.parameter_diffs[:10]:
                        if pd.diff_type == "missing":
                            console.print(f"          [red]- {pd.param_name}[/red]: {pd.golden_value}")
                        elif pd.diff_type == "added":
                            console.print(f"          [green]+ {pd.param_name}[/green]: {pd.actual_value}")
                        elif pd.diff_type == "type_changed":
                            console.print(f"          [yellow]~ {pd.param_name}[/yellow]: type changed")
                            console.print(f"            golden: {type(pd.golden_value).__name__} = {pd.golden_value}")
                            console.print(f"            actual: {type(pd.actual_value).__name__} = {pd.actual_value}")
                        elif pd.diff_type == "value_changed":
                            sim_str = ""
                            if pd.similarity is not None:
                                sim_str = f" (similarity: {int(pd.similarity * 100)}%)"
                            console.print(f"          [yellow]~ {pd.param_name}[/yellow]:{sim_str}")
                            console.print(f"            [dim]golden:[/dim] {pd.golden_value}")
                            console.print(f"            [dim]actual:[/dim] {pd.actual_value}")

        # Per-turn evaluation results
        matching_result = next((r for r in results if r.test_case == test_name), None)
        if matching_result and getattr(matching_result, "turn_evaluations", None):
            for te in matching_result.turn_evaluations:
                parts = []
                if te.tool_accuracy is not None:
                    tool_names = ""
                    if te.tool_accuracy >= 1.0:
                        tool_names = " ✓"
                    parts.append(f"tools{tool_names}")
                if te.contains_passed:
                    parts.append(f'contains "{te.contains_passed[0]}" ✓')
                if te.contains_failed:
                    parts.append(f'missing "{te.contains_failed[0]}"')
                if te.forbidden_violations:
                    parts.append(f'forbidden "{te.forbidden_violations[0]}" used')
                if te.passed:
                    console.print(f"      [green]Turn {te.turn_index}: ✅[/green] {', '.join(parts)}")
                else:
                    console.print(f"      [red]Turn {te.turn_index}: ❌[/red] {', '.join(parts)}")

        if abs(trace_diff.score_diff) > 1:
            direction = "[green]↑[/green]" if trace_diff.score_diff > 0 else "[red]↓[/red]"
            console.print(f"    Score: {direction} {trace_diff.score_diff:+.1f}")

        console.print()

    # Urgency summary
    regressions = sum(1 for _, d in diffs_found if d.overall_severity == DiffStatus.REGRESSION)
    tools_changed = sum(1 for _, d in diffs_found if d.overall_severity == DiffStatus.TOOLS_CHANGED)
    output_changed = sum(1 for _, d in diffs_found if d.overall_severity == DiffStatus.OUTPUT_CHANGED)

    if regressions > 0:
        console.print(f"[red]✗ {regressions} REGRESSION(s) - score dropped, fix before deploy[/red]")
        console.print()
        console.print("[dim]⭐ EvalView caught this before prod! Star → github.com/hidai25/eval-view[/dim]\n")
    elif tools_changed > 0:
        console.print(f"[yellow]⚠ {tools_changed} TOOLS_CHANGED - agent behavior shifted, review before deploy[/yellow]")
        console.print()
        console.print("[dim]⭐ EvalView caught this! Star → github.com/hidai25/eval-view[/dim]\n")
    elif output_changed > 0:
        console.print(f"[dim]~ {output_changed} OUTPUT_CHANGED - response changed, review before deploy[/dim]\n")


def _print_no_diffs(results: List[Any], console: Any) -> None:
    from evalview.core.golden import GoldenStore

    store = GoldenStore()
    goldens = store.list_golden()
    matched = sum(1 for g in goldens if any(r.test_case == g.test_name for r in results))
    if matched > 0:
        console.print(
            f"[green]✓ PASSED - No differences from golden baseline ({matched} tests compared)[/green]\n"
        )
    elif goldens:
        console.print("[yellow]No golden traces match these tests[/yellow]")
    else:
        console.print("[yellow]No golden traces found[/yellow]")
        console.print("[dim]Create baseline: evalview golden save <results-file>[/dim]\n")


def display_regression_analysis(
    regression_reports: Dict[str, Any],
    console: Any,
) -> None:
    """Print the compare-baseline regression analysis section."""
    if not regression_reports:
        return

    console.print()
    console.print("[bold cyan]📊 Regression Analysis[/bold cyan]")
    console.print("━" * 60)
    console.print()

    any_regressions = False
    for test_name, report in regression_reports.items():
        if report.baseline_score is None:
            continue

        if report.is_regression:
            any_regressions = True
            if report.severity == "critical":
                status = "[red]🔴 CRITICAL REGRESSION[/red]"
            elif report.severity == "moderate":
                status = "[yellow]🟡 MODERATE REGRESSION[/yellow]"
            else:
                status = "[yellow]🟠 MINOR REGRESSION[/yellow]"
        else:
            status = "[green]✅ No regression[/green]"

        console.print(f"[bold]{test_name}[/bold]: {status}")

        if report.score_delta is not None:
            delta_str = f"{report.score_delta:+.1f}"
            pct_str = f"({report.score_delta_percent:+.1f}%)"
            direction = "[red]↓[/red]" if report.score_delta < 0 else "[green]↑[/green]"
            console.print(
                f"  Score: {report.current_score:.1f} {direction} {delta_str} {pct_str} "
                f"vs baseline {report.baseline_score:.1f}"
            )

        if report.cost_delta is not None and report.cost_delta_percent is not None:
            delta_str = f"${report.cost_delta:+.4f}"
            pct_str = f"({report.cost_delta_percent:+.1f}%)"
            if report.cost_delta_percent > 20:
                console.print(f"  Cost: ${report.current_cost:.4f} [red]↑ {delta_str}[/red] {pct_str}")
            else:
                console.print(f"  Cost: ${report.current_cost:.4f} {delta_str} {pct_str}")

        if report.latency_delta is not None and report.latency_delta_percent is not None:
            delta_str = f"{report.latency_delta:+.0f}ms"
            pct_str = f"({report.latency_delta_percent:+.1f}%)"
            if report.latency_delta_percent > 30:
                console.print(f"  Latency: {report.current_latency:.0f}ms [red]↑ {delta_str}[/red] {pct_str}")
            else:
                console.print(f"  Latency: {report.current_latency:.0f}ms {delta_str} {pct_str}")

        if report.is_regression and report.issues:
            console.print(f"  Issues: {', '.join(report.issues)}")

        console.print()

    if any_regressions:
        console.print("[red]⚠️  Regressions detected! Review changes before deploying.[/red]\n")


def save_results(
    results: List[Any],
    output: str,
    console: Any,
) -> Path:
    """Persist results to a JSON file and return the path."""
    from evalview.reporters.json_reporter import JSONReporter

    output_dir = Path(output)
    output_dir.mkdir(parents=True, exist_ok=True)
    results_file = output_dir / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    JSONReporter.save(results, results_file)
    console.print(f"\n[dim]Results saved to: {results_file}[/dim]\n")
    return results_file


def save_golden_if_requested(
    save_golden: bool,
    failed: int,
    execution_errors: int,
    results: List[Any],
    results_file: Path,
    console: Any,
) -> None:
    """Auto-save golden baselines when --save-golden is set and all tests passed."""
    if not (save_golden and failed == 0 and execution_errors == 0 and results):
        return

    try:
        from evalview.core.golden import GoldenStore

        store = GoldenStore()
        saved_count = 0
        for result in results:
            if result and result.score > 0:
                store.save_golden(result, notes="Auto-saved via --save-golden", source_file=str(results_file))
                saved_count += 1

        if saved_count > 0:
            console.print(
                f"[green]Golden baseline saved for {saved_count} test{'s' if saved_count != 1 else ''}.[/green]"
            )
            console.print("[dim]Future runs with --diff will compare against this baseline.[/dim]\n")
    except Exception as exc:
        console.print(f"[yellow]Could not save golden baseline: {exc}[/yellow]\n")


def collect_diffs(results: List[Any]) -> List[Tuple[str, Any]]:
    """Compare results against golden baselines and return (name, diff) pairs."""
    from evalview.core.golden import GoldenStore
    from evalview.core.diff import compare_to_golden

    store = GoldenStore()
    diffs: List[Tuple[str, Any]] = []
    for result in results:
        golden = store.load_golden(result.test_case)
        if golden:
            trace_diff = compare_to_golden(golden, result.trace, result.score)
            if trace_diff.has_differences:
                diffs.append((result.test_case, trace_diff))
    return diffs


def display_html_reports(
    html_report: Optional[str],
    diff_report: Optional[str],
    diff_enabled: bool,
    diffs_found: List[Tuple[str, Any]],
    results: List[Any],
    no_open: bool,
    watch: bool,
    console: Any,
) -> None:
    """Generate optional HTML results report and/or HTML diff report."""
    if html_report and results:
        try:
            from evalview.visualization import generate_visual_report

            html_path = generate_visual_report(results, output_path=html_report, auto_open=False)
            console.print("\n[bold green]📊 HTML Report Generated![/bold green]")
            console.print(f"   [link=file://{Path(html_path).absolute()}]{html_path}[/link]")
            console.print(f"   [dim]Open in browser: open {html_path}[/dim]\n")
        except Exception as exc:
            console.print(f"[yellow]⚠️  Could not generate HTML report: {exc}[/yellow]\n")

    if diff_report and results:
        if not diff_enabled:
            console.print("[yellow]⚠️  --diff-report requires --diff flag[/yellow]")
            console.print("[dim]Usage: evalview run --diff --diff-report diff.html[/dim]\n")
        elif diffs_found:
            try:
                from evalview.reporters.html_reporter import DiffReporter

                diff_path = DiffReporter().generate(
                    diffs=[d for _, d in diffs_found],
                    results=results,
                    output_path=diff_report,
                )
                console.print("\n[bold cyan]📊 Diff Report Generated![/bold cyan]")
                console.print(f"   [link=file://{Path(diff_path).absolute()}]{diff_path}[/link]")
                console.print(f"   [dim]Open in browser: open {diff_path}[/dim]\n")
            except ImportError as exc:
                console.print(f"[yellow]⚠️  Could not generate diff report: {exc}[/yellow]")
                console.print("[dim]Install with: pip install jinja2[/dim]\n")
        else:
            console.print("[dim]No differences to report - all tests match golden baseline[/dim]\n")

    if not watch and results:
        import os

        in_ci = bool(os.environ.get("CI"))
        should_open = not no_open and not in_ci
        try:
            from evalview.visualization import generate_visual_report

            report_path = generate_visual_report(
                results,
                diffs=[d for _, d in diffs_found] if diffs_found else None,
                auto_open=should_open,
                title="EvalView Run Report",
            )
            if should_open:
                console.print(f"\n[bold]📊 Report opened in browser[/bold] [dim]({report_path})[/dim]\n")
            else:
                console.print(f"\n[dim]📊 Report saved: {report_path}[/dim]\n")
        except Exception as err:
            console.print(f"[dim]⚠ Could not generate HTML report: {err}[/dim]")


def compute_exit_code(
    failed: int,
    execution_errors: int,
    diff_enabled: bool,
    diffs_found: List[Tuple[str, Any]],
    fail_on: Optional[str],
    warn_on: Optional[str],
    console: Any,
) -> int:
    """Determine the process exit code for CI use.

    Returns 0 on success, 1 on test failures or diff regressions, 2 on
    execution errors.
    """
    if execution_errors > 0:
        exit_code = 2
    elif failed > 0:
        exit_code = 1
    else:
        exit_code = 0

    if not (diff_enabled and diffs_found):
        return exit_code

    from evalview.core.diff import DiffStatus

    valid_statuses = {"REGRESSION", "TOOLS_CHANGED", "OUTPUT_CHANGED", "PASSED", "CONTRACT_DRIFT"}
    fail_statuses: Set[Any] = set()
    warn_statuses: Set[Any] = set()

    for s in (fail_on or "").upper().split(","):
        s = s.strip()
        if not s:
            continue
        if s in valid_statuses:
            fail_statuses.add(DiffStatus[s])
        else:
            console.print(
                f"[yellow]Warning: Unknown status '{s}' in --fail-on "
                f"(valid: {', '.join(valid_statuses)})[/yellow]"
            )

    for s in (warn_on or "").upper().split(","):
        s = s.strip()
        if not s:
            continue
        if s in valid_statuses:
            warn_statuses.add(DiffStatus[s])
        else:
            console.print(
                f"[yellow]Warning: Unknown status '{s}' in --warn-on "
                f"(valid: {', '.join(valid_statuses)})[/yellow]"
            )

    fail_count = 0
    warn_count = 0
    status_counts: Dict[Any, int] = {}

    for _, trace_diff in diffs_found:
        diff_status = trace_diff.overall_severity
        status_counts[diff_status] = status_counts.get(diff_status, 0) + 1
        if diff_status in fail_statuses:
            fail_count += 1
        elif diff_status in warn_statuses:
            warn_count += 1

    if fail_count > 0 or warn_count > 0:
        console.print("[bold]━━━ CI Summary ━━━[/bold]")
        for diff_status, count in sorted(status_counts.items(), key=lambda x: x[0].value):
            if diff_status in fail_statuses:
                console.print(f"  [red]✗ {count} {diff_status.value.upper()}[/red] [dim][FAIL][/dim]")
            elif diff_status in warn_statuses:
                console.print(f"  [yellow]⚠ {count} {diff_status.value.upper()}[/yellow] [dim][WARN][/dim]")
            else:
                console.print(f"  [green]✓ {count} {diff_status.value.upper()}[/green]")

        if fail_count > 0:
            exit_code = max(exit_code, 1)
            console.print(f"\n[bold red]Exit: {exit_code}[/bold red] ({fail_count} failure(s) in fail_on set)\n")
        else:
            console.print(f"\n[bold green]Exit: {exit_code}[/bold green] ({warn_count} warning(s) only)\n")

    return exit_code


def display_trust_frame(
    passed: int,
    failed: int,
    execution_errors: int,
    results: List[Any],
    results_file: Optional[Path],
    console: Any,
) -> None:
    """Print the final health summary and optional 'save golden' tip."""
    console.print("[dim]━" * 50 + "[/dim]")
    if execution_errors > 0:
        n = execution_errors
        console.print(
            f"[bold yellow]{n} test{'s' if n != 1 else ''} could not run.[/bold yellow] "
            "Check network, timeouts, or agent availability.\n"
        )
    elif failed == 0 and passed > 0:
        console.print(
            f"[bold green]Agent healthy.[/bold green] {passed}/{passed} tests passed.\n"
        )
        if results_file:
            try:
                from evalview.core.golden import GoldenStore

                store = GoldenStore()
                if not any(store.has_golden(r.test_case) for r in results if r):
                    console.print("[dim]Tip: Save this as your baseline so future runs detect regressions:[/dim]")
                    console.print("[dim]   evalview snapshot[/dim]\n")
            except Exception:
                pass
    elif failed > 0:
        console.print(
            f"[bold red]{failed} test{'s' if failed != 1 else ''} failed.[/bold red] "
            "Review scores and agent behavior above.\n"
        )
