"""Check and replay commands — regression detection against golden baselines."""
from __future__ import annotations

import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

import click

from evalview.commands.shared import (
    console,
    _load_config_if_exists,
    _cloud_pull,
    _execute_check_tests,
    _analyze_check_diffs,
    _parse_fail_statuses,
)
from evalview.commands.check_display import (
    _display_check_results,
    _print_trajectory_diff,
)
from evalview.telemetry.decorators import track_command

if TYPE_CHECKING:
    from evalview.core.types import EvaluationResult
    from evalview.core.diff import TraceDiff


def _compute_check_exit_code(
    diffs: List[Tuple[str, "TraceDiff"]],
    fail_on: Optional[str],
    strict: bool,
    execution_failures: int = 0,
) -> int:
    """Compute exit code based on diff results and fail conditions.

    Returns:
        0 if no failures match fail conditions, 1 otherwise
    """
    if strict:
        fail_on = "REGRESSION,TOOLS_CHANGED,OUTPUT_CHANGED"

    if not fail_on:
        fail_on = "REGRESSION"  # Default

    fail_statuses = _parse_fail_statuses(fail_on)

    if execution_failures > 0:
        return 1

    for _, diff in diffs:
        if diff.overall_severity in fail_statuses:
            return 1

    return 0


def _summarize_check_targets(test_cases: List[Any], config: Any) -> tuple[list[str], list[str]]:
    config_endpoint = getattr(config, "endpoint", None) if config else None
    config_adapter = getattr(config, "adapter", None) if config else None
    endpoints = sorted(
        {
            str(endpoint)
            for endpoint in ((tc.endpoint or config_endpoint) for tc in test_cases)
            if endpoint is not None
        }
    )
    adapters = sorted(
        {
            str(adapter)
            for adapter in ((tc.adapter or config_adapter) for tc in test_cases)
            if adapter is not None
        }
    )
    return endpoints, adapters


def _print_check_failure_guidance(test_cases: List[Any], config: Any) -> None:
    endpoints, adapters = _summarize_check_targets(test_cases, config)
    if len(endpoints) > 1 or len(adapters) > 1:
        console.print("[yellow]This check run mixes multiple endpoints or adapters.[/yellow]")
        if endpoints:
            console.print(f"[dim]Endpoints: {', '.join(endpoints)}[/dim]")
        if adapters:
            console.print(f"[dim]Adapters: {', '.join(adapters)}[/dim]")
        console.print("[dim]Use a narrower folder such as tests/generated-from-init or rerun evalview init to refresh config.[/dim]\n")
    else:
        console.print("[dim]Fix the failing test connections or narrow the test path, then rerun evalview check.[/dim]\n")


def _should_auto_generate_report(
    *,
    report_path: Optional[str],
    json_output: bool,
    analysis: Dict[str, Any],
    results: List[Any],
) -> bool:
    if report_path or json_output or not results:
        return False
    if bool(__import__("os").environ.get("CI")):
        return False
    # Always generate — open report for both clean checks and failures
    return True


def _judge_usage_summary() -> Dict[str, Any]:
    """Return structured judge usage for report rendering."""
    from evalview.core.llm_provider import judge_cost_tracker

    total_tokens = judge_cost_tracker.total_input_tokens + judge_cost_tracker.total_output_tokens
    model_display = ""
    pricing_display = ""
    if judge_cost_tracker.model:
        if judge_cost_tracker.provider:
            model_display = f"{judge_cost_tracker.provider}/{judge_cost_tracker.model}"
        else:
            model_display = judge_cost_tracker.model
        from evalview.core.pricing import format_pricing_line
        pricing_display = format_pricing_line(judge_cost_tracker.model) or ""
    return {
        "call_count": judge_cost_tracker.call_count,
        "input_tokens": judge_cost_tracker.total_input_tokens,
        "output_tokens": judge_cost_tracker.total_output_tokens,
        "total_tokens": total_tokens,
        "total_cost": round(judge_cost_tracker.total_cost, 6),
        "is_free": judge_cost_tracker.call_count > 0 and judge_cost_tracker.total_cost == 0,
        "model": model_display,
        "pricing": pricing_display,
    }


def _resolve_default_test_path(test_path: str) -> str:
    """Use the active onboarding/generation folder when the user omitted a path."""
    if test_path != "tests":
        return test_path
    from evalview.core.project_state import ProjectStateStore

    active = ProjectStateStore().get_active_test_path()
    if active and Path(active).exists():
        return active
    return test_path


def _format_snapshot_timestamp(snapshot_at: datetime) -> str:
    """Format the last snapshot timestamp for human-facing check output."""
    if snapshot_at.tzinfo is not None:
        snapshot_at = snapshot_at.astimezone().replace(tzinfo=None)
    return snapshot_at.strftime("%Y-%m-%d %H:%M")


def _format_baseline_timestamp(dt: datetime) -> str:
    """Format a baseline timestamp as an exact date/time string."""
    if dt.tzinfo is not None:
        dt = dt.astimezone().replace(tzinfo=None)
    return dt.strftime("%Y-%m-%d %H:%M")


def _print_baseline_context(goldens: List[Any], state: Any) -> None:
    """Print baseline context: count, date range, and model info."""
    if not goldens:
        return

    n = len(goldens)
    dates = [g.blessed_at for g in goldens if g.blessed_at]

    # Model info — collect unique model IDs
    models = {g.model_id for g in goldens if g.model_id}

    parts = [f"[dim]{n} baseline{'s' if n != 1 else ''}[/dim]"]

    if dates:
        oldest = min(dates)
        newest = max(dates)
        if oldest == newest:
            parts.append(f"[dim]snapshot: {_format_baseline_timestamp(newest)}[/dim]")
        else:
            parts.append(
                f"[dim]snapshots: {_format_baseline_timestamp(oldest)} – "
                f"{_format_baseline_timestamp(newest)}[/dim]"
            )

    if models:
        model_str = ", ".join(sorted(models))
        parts.append(f"[dim]model: {model_str}[/dim]")

    console.print("  ".join(parts))
    console.print()


@click.command("check")
@click.argument("test_path", default="tests", type=click.Path(exists=True))
@click.option("--test", "-t", help="Check only this specific test")
@click.option("--json", "json_output", is_flag=True, help="Output JSON for CI")
@click.option("--fail-on", help="Comma-separated statuses to fail on (default: REGRESSION)")
@click.option("--strict", is_flag=True, help="Fail on any change (REGRESSION, TOOLS_CHANGED, OUTPUT_CHANGED)")
@click.option("--report", "report_path", default=None, type=click.Path(), help="Generate HTML report at this path (auto-opens in browser)")
@click.option("--csv", "csv_path", default=None, type=click.Path(), help="Export results to a CSV file")
@click.option(
    "--semantic-diff/--no-semantic-diff",
    "semantic_diff",
    default=None,
    help=(
        "Enable/disable embedding-based semantic similarity. "
        "Auto-enabled when OPENAI_API_KEY is set (adds ~$0.00004/test). "
        "Use --no-semantic-diff to opt out."
    ),
)
@click.option("--budget", type=float, default=None, help="Maximum total budget in dollars.")
@click.option("--timeout", type=float, default=120.0, help="Timeout per test in seconds (default: 120.0).")
@click.option("--dry-run", "dry_run", is_flag=True, default=False, help="Preview test plan without executing.")
@click.option("--ai-root-cause", "ai_root_cause", is_flag=True, default=False, help="Use AI to explain low-confidence regressions (requires LLM provider).")
@click.option("--statistical", "statistical_runs", type=int, default=None, help="Run each test N times for variance analysis (e.g. --statistical 10).")
@click.option("--auto-variant", "auto_variant", is_flag=True, default=False, help="Auto-discover and save distinct execution paths as golden variants (use with --statistical).")
@click.option("--judge", "judge_model", default=None, help="Judge model for scoring (e.g. gpt-5.4-mini, sonnet, deepseek-chat).")
@track_command("check")
def check(test_path: str, test: str, json_output: bool, fail_on: str, strict: bool, report_path: Optional[str], csv_path: Optional[str], semantic_diff: Optional[bool], budget: Optional[float], timeout: float, dry_run: bool, ai_root_cause: bool, statistical_runs: Optional[int], auto_variant: bool, judge_model: Optional[str]):
    """Check current behavior against snapshot baseline.

    This command runs tests and compares them against your saved baselines,
    showing only what changed. Perfect for CI/CD and daily development.

    TEST_PATH is the directory containing test cases (default: tests/).

    Examples:
        evalview check                                   # Check all tests
        evalview check --test "my-test"                  # Check one test
        evalview check --json                            # JSON output for CI
        evalview check --csv results.csv                 # Export results to CSV
        evalview check --report report.html              # Generate HTML report
        evalview check --fail-on REGRESSION,TOOLS_CHANGED
        evalview check --strict                          # Fail on any change
        evalview check --no-semantic-diff                # Opt out of semantic diff
        evalview check --dry-run                         # Preview plan, no API calls
        evalview check --budget 0.50                     # Cap spend at $0.50
        evalview check --timeout 60                      # 60 second timeout per test
        evalview check --ai-root-cause                   # AI-powered regression explanation
        evalview check --statistical 10                  # Run each test 10 times, show variance
        evalview check --statistical 10 --auto-variant   # Auto-save distinct paths as variants
    """
    if budget is not None and budget <= 0:
        click.echo("Error: --budget must be a positive number.", err=True)
        sys.exit(1)

    if timeout <= 0:
        click.echo("Error: --timeout must be a positive number.", err=True)
        sys.exit(1)

    from evalview.core.loader import TestCaseLoader
    from evalview.core.golden import GoldenStore
    from evalview.core.project_state import ProjectStateStore
    from evalview.core.celebrations import Celebrations
    from evalview.core.messages import get_random_checking_message

    # Initialize stores
    store = GoldenStore()
    state_store = ProjectStateStore()
    test_path = _resolve_default_test_path(test_path)

    state = state_store.load()

    # Check if this is the first check
    is_first_check = state_store.is_first_check()

    # Show recap
    if not is_first_check and not json_output:
        days_since = state_store.days_since_last_check()
        if days_since and days_since >= 7:
            Celebrations.welcome_back(days_since)

    # Pull any missing goldens from cloud before checking locally
    _cloud_pull(store)

    # Verify snapshots exist
    goldens = store.list_golden()
    if not goldens:
        if not json_output:
            Celebrations.no_snapshot_found()
        sys.exit(1)

    if not json_output:
        _print_baseline_context(goldens, state)
        console.print(f"[cyan]▶ {get_random_checking_message()}[/cyan]\n")

    # Load test cases
    loader = TestCaseLoader()
    try:
        test_cases = loader.load_from_directory(Path(test_path))
    except Exception as e:
        console.print(f"[red]❌ Failed to load test cases: {e}[/red]\n")
        sys.exit(1)

    # Filter to specific test if requested
    if test:
        test_cases = [tc for tc in test_cases if tc.name == test]
        if not test_cases:
            console.print(f"[red]❌ No test found with name: {test}[/red]\n")
            sys.exit(1)

    test_metadata = {
        tc.name: {
            "is_multi_turn": bool(getattr(tc, "is_multi_turn", False)),
            "behavior_class": (tc.meta or {}).get("behavior_class"),
        }
        for tc in test_cases
    }

    # Load config
    config = _load_config_if_exists()

    # Apply judge config: --judge flag > interactive picker > config.yaml
    from evalview.commands.shared import apply_judge_model
    apply_judge_model(judge_model, interactive=not json_output)
    from evalview.core.config import apply_judge_config
    apply_judge_config(config)
    from evalview.core.llm_provider import judge_cost_tracker
    judge_cost_tracker.reset()

    # Resolve semantic diff: explicit flag > config file > auto-enable.
    from evalview.core.semantic_diff import SemanticDiff
    key_available = SemanticDiff.is_available()

    if semantic_diff is None:
        config_setting = config.get_diff_config().semantic_diff_enabled if config else None
        if config_setting is False:
            semantic_diff = False
        else:
            semantic_diff = key_available
        if semantic_diff and not json_output:
            state_for_notice = state_store.load()
            if not state_for_notice.semantic_auto_noticed:
                console.print(
                    "[dim]ℹ  Semantic diff auto-enabled (OPENAI_API_KEY detected). "
                    f"{SemanticDiff.cost_notice()}. "
                    "Use --no-semantic-diff to opt out.[/dim]\n"
                )
                state_for_notice.semantic_auto_noticed = True
                state_store.save(state_for_notice)
    elif semantic_diff and not key_available:
        if not json_output:
            console.print(
                "[yellow]⚠  --semantic-diff requested but OPENAI_API_KEY is not set. "
                "Falling back to lexical comparison.[/yellow]\n"
            )
        semantic_diff = False

    # Dry-run mode — show plan and exit
    if dry_run:
        golden_names = {golden.test_name for golden in goldens}
        tests_with_baselines = sum(1 for tc in test_cases if tc.name in golden_names)
        if not json_output:
            console.print(f"  Tests:          {len(test_cases)}")
            console.print(f"  With baselines: {tests_with_baselines}")
            console.print(f"  API calls:      ~{len(test_cases)} (agent) + ~{len(test_cases)} (judge)")
            if budget is not None:
                console.print(f"  Budget:         ${budget:.2f}")
            console.print()
            console.print("[dim]No API calls were made. Remove --dry-run to execute.[/dim]\n")
        else:
            print(json.dumps({"dry_run": True, "tests": len(test_cases), "with_baselines": tests_with_baselines}))
        sys.exit(0)

    # Pre-flight: skip execution if no tests have matching baselines
    golden_names = {golden.test_name for golden in goldens}
    matched_tests = [tc for tc in test_cases if tc.name in golden_names]
    if not matched_tests:
        if not json_output:
            from rich.panel import Panel as _PF
            console.print(
                _PF(
                    "[yellow]0 tests compared.[/yellow] "
                    "Your test names don't match any golden baselines.\n\n"
                    "This usually means tests were regenerated or renamed since the last snapshot.\n\n"
                    "[bold]To fix:[/bold]\n"
                    "  [bold]evalview snapshot[/bold]         capture new baselines for current tests\n"
                    "  [bold]evalview snapshot --reset[/bold]  clear old baselines first, then capture fresh",
                    border_style="yellow",
                    title="No matching baselines",
                    padding=(1, 2),
                )
            )
        sys.exit(0)

    # Budget tracking with circuit breaker
    budget_tracker = None
    if budget is not None:
        from evalview.core.budget import BudgetTracker
        budget_tracker = BudgetTracker(limit=budget)

    # Statistical mode — run tests N times and cluster results
    if statistical_runs:
        if statistical_runs < 3:
            console.print("[red]Error: --statistical requires at least 3 runs.[/red]")
            sys.exit(1)

        if not json_output:
            console.print(f"[cyan]▶ Statistical mode: running each test {statistical_runs} times...[/cyan]\n")

        from evalview.core.variant_clusterer import cluster_results, suggest_variants, format_cluster_summary
        from evalview.evaluators.statistical_evaluator import compute_statistical_metrics, compute_flakiness_score

        all_stat_results: Dict[str, List] = {}

        for run_idx in range(statistical_runs):
            if not json_output:
                console.print(f"  [dim]Run {run_idx + 1}/{statistical_runs}...[/dim]")

            run_diffs, run_results, _, _ = _execute_check_tests(
                test_cases, config, json_output=True, semantic_diff=semantic_diff, timeout=timeout,
                budget_tracker=budget_tracker,
            )

            for result in run_results:
                test_name = result.test_case
                if test_name not in all_stat_results:
                    all_stat_results[test_name] = []
                all_stat_results[test_name].append(result)

        # Cluster and display results per test
        if not json_output:
            console.print()
            for test_name, test_results in all_stat_results.items():
                clusters = cluster_results(test_results)
                scores = [r.score for r in test_results]
                stats = compute_statistical_metrics(scores)
                flakiness = compute_flakiness_score(test_results, stats)

                console.print(
                    f"[bold]{test_name}[/bold]  "
                    f"[dim]mean: {stats.mean:.1f}, std: {stats.std_dev:.1f}, "
                    f"flakiness: {flakiness.category}[/dim]"
                )
                console.print(format_cluster_summary(clusters, statistical_runs))
                console.print()

                # Auto-variant: save distinct paths
                if auto_variant:
                    suggested = suggest_variants(clusters)
                    if len(suggested) > 1:
                        existing = store.load_all_golden_variants(test_name)
                        existing_count = len(existing) if existing else 0
                        slots_left = 5 - existing_count

                        if slots_left <= 0:
                            console.print(f"  [yellow]⚠ {test_name}: already has 5 variants (max)[/yellow]")
                            continue

                        # Skip the most common cluster (already the default baseline)
                        new_variants = suggested[1:slots_left + 1]

                        if new_variants:
                            console.print(f"  [cyan]Found {len(new_variants)} distinct path(s) to save as variants:[/cyan]")
                            for v in new_variants:
                                console.print(f"    • {v.sequence_key} ({v.frequency} occurrences)")

                            import click as _click
                            if _click.confirm("    Save these as golden variants?", default=True):
                                for idx, variant_cluster in enumerate(new_variants):
                                    variant_name = f"auto-v{existing_count + idx + 1}"
                                    rep = variant_cluster.representative
                                    store.save_golden(
                                        result=rep,
                                        notes=f"Auto-variant from statistical run ({variant_cluster.frequency}/{statistical_runs} occurrences)",
                                        variant_name=variant_name,
                                    )
                                    console.print(f"    [green]✓ Saved variant '{variant_name}': {variant_cluster.sequence_key}[/green]")
                                console.print()

            console.print()

        sys.exit(0)

    # Execute tests and compare against golden — show spinner while waiting
    if not json_output:
        from evalview.commands.shared import run_with_spinner
        diffs, results, drift_tracker, golden_traces = run_with_spinner(
            lambda: _execute_check_tests(test_cases, config, json_output, semantic_diff, timeout, budget_tracker=budget_tracker),
            "Checking",
            len(test_cases),
        )
    else:
        diffs, results, drift_tracker, golden_traces = _execute_check_tests(
            test_cases, config, json_output, semantic_diff, timeout, budget_tracker=budget_tracker
        )

    golden_names = {golden.test_name for golden in goldens}
    baseline_test_cases = [tc for tc in test_cases if tc.name in golden_names]
    execution_failures = max(0, len(baseline_test_cases) - len(results))

    # Analyze diffs
    analysis = _analyze_check_diffs(diffs)
    analysis["execution_failures"] = execution_failures
    if execution_failures > 0:
        analysis["all_passed"] = False
        analysis["has_execution_failures"] = True

    # Don't treat zero-test runs as a real pass — no tests were compared
    # But execution failures still count as failures even with 0 diffs.
    actually_compared = len(diffs)
    if actually_compared == 0 and execution_failures == 0:
        analysis["all_passed"] = True  # Not a failure, but not a real check
        analysis["nothing_compared"] = True

    # Update project state (only count real checks toward streaks)
    if actually_compared > 0:
        state = state_store.update_check(
            has_regressions=(not analysis["all_passed"]),
            status="passed" if analysis["all_passed"] else "regression"
        )
    else:
        state = state_store.load()

    # Cost summary with per-test breakdown
    if results and not json_output:
        total_cost = sum(r.trace.metrics.total_cost for r in results)
        total_api_calls = sum(len(r.trace.steps) for r in results)

        console.print(
            f"[dim]💰 {len(results)} tests, {total_api_calls} API calls, "
            f"${total_cost:.4f} total[/dim]"
        )

        # Per-test cost breakdown (show top 5 most expensive)
        if len(results) > 1:
            sorted_by_cost = sorted(results, key=lambda r: r.trace.metrics.total_cost, reverse=True)
            console.print("[dim]   Top costs:[/dim]")
            for r in sorted_by_cost[:5]:
                cost = r.trace.metrics.total_cost
                if cost > 0:
                    pct = cost / total_cost * 100 if total_cost > 0 else 0
                    console.print(f"[dim]     ${cost:.4f} ({pct:.0f}%) — {r.test_case}[/dim]")

        console.print()

        if budget_tracker and budget_tracker.halted:
            console.print(
                f"[red]⚠  Budget circuit breaker tripped: "
                f"${budget_tracker.spent:.4f} spent of ${budget:.2f} limit[/red]"
            )
            skipped = len(test_cases) - len(results)
            if skipped > 0:
                console.print(f"[red]   {skipped} test(s) skipped to stay within budget[/red]")
            console.print()
            sys.exit(1)
        elif budget is not None and total_cost > budget:
            console.print(
                f"[red]⚠  Budget exceeded: ${total_cost:.4f} > ${budget:.2f} limit[/red]\n"
            )
            sys.exit(1)

    # AI root cause enrichment (opt-in)
    ai_root_causes = None
    if ai_root_cause and not analysis["all_passed"]:
        import asyncio
        from evalview.core.root_cause import enrich_diffs_with_ai
        if not json_output:
            console.print("[dim]🤖 Running AI root cause analysis...[/dim]\n")
        ai_root_causes = asyncio.run(enrich_diffs_with_ai(diffs))

    # Display results
    _display_check_results(
        diffs, analysis, state, is_first_check, json_output,
        drift_tracker=drift_tracker,
        golden_traces=golden_traces,
        results=results,
        ai_root_causes=ai_root_causes,
        test_metadata=test_metadata,
    )

    if execution_failures > 0 and not json_output:
        _print_check_failure_guidance(baseline_test_cases, config)

    auto_report = _should_auto_generate_report(
        report_path=report_path,
        json_output=json_output,
        analysis=analysis,
        results=results,
    )
    effective_report_path = report_path
    if auto_report:
        effective_report_path = str(Path(".evalview") / "latest-check.html")

    # Generate HTML report if requested or auto-enabled
    if effective_report_path and results:
        from evalview.visualization import generate_visual_report
        diff_list = [d for _, d in diffs]
        # Open to Diffs tab when there are changes, Overview when clean
        tab = "diffs" if diff_list else "overview"
        path = generate_visual_report(
            results=results,
            diffs=diff_list,
            golden_traces=golden_traces,
            judge_usage=_judge_usage_summary(),
            output_path=effective_report_path,
            auto_open=not json_output,
            title="EvalView Check Report",
            default_tab=tab,
        )
        if not json_output:
            if auto_report:
                console.print(f"[green]◈ Failure report:[/green] {path}")
                console.print("[dim]Opened automatically because this check found changes or execution failures.[/dim]\n")
            else:
                console.print(f"[green]◈ Report:[/green] {path}\n")

    # Export results to CSV if requested
    if csv_path and diffs:
        result_lookup: Dict[str, "EvaluationResult"] = {}
        if results:
            for r in results:
                result_lookup[r.test_case] = r

        timestamp = datetime.now(timezone.utc).isoformat()
        csv_file_path = Path(csv_path)
        with open(csv_file_path, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["test_name", "status", "score", "baseline_score", "diff", "timestamp"])
            for name, diff in diffs:
                current_result = result_lookup.get(name)
                current_score: Any = current_result.score if current_result else ""
                baseline_score: Any = (current_result.score - diff.score_diff) if current_result and diff.score_diff is not None else ""
                score_diff = diff.score_diff if diff.score_diff is not None else ""
                writer.writerow([
                    name,
                    diff.overall_severity.value,
                    current_score,
                    baseline_score,
                    score_diff,
                    timestamp,
                ])
        if not json_output:
            console.print(f"[green]◈ CSV exported:[/green] {csv_file_path}\n")

    # Auto-update badge if it exists
    from evalview.commands.badge_cmd import update_badge_after_check
    update_badge_after_check(diffs, len(diffs))

    # Compute and exit with code
    exit_code = _compute_check_exit_code(diffs, fail_on, strict, execution_failures=execution_failures)
    sys.exit(exit_code)


@click.command("replay")
@click.argument("test_name", required=False, default=None)
@click.option("--test-path", "test_path", default="tests", type=click.Path(exists=True), help="Directory containing tests")
@click.option("--no-browser", is_flag=True, help="Don't auto-open the HTML report")
@track_command("replay")
def replay(test_name: Optional[str], test_path: str, no_browser: bool) -> None:
    """Replay a test and show full trajectory diff vs baseline.

    Shows step-by-step what your agent did vs. the saved baseline.
    Opens an HTML report with side-by-side sequence diagrams.

    \b
    Examples:
        evalview replay my-test
        evalview replay my-test --no-browser
        evalview replay my-test --test-path ./my-tests
    """
    from evalview.core.loader import TestCaseLoader
    from evalview.core.golden import GoldenStore
    from evalview.visualization import generate_visual_report

    store = GoldenStore()
    _cloud_pull(store)

    # No test name given — list available tests with baselines
    if not test_name:
        goldens = store.list_golden()
        if not goldens:
            console.print("\n[yellow]No baselines found.[/yellow] Run [bold]evalview snapshot[/bold] first.\n")
            sys.exit(1)
        console.print("\n[bold]Available tests with baselines:[/bold]\n")
        for g in sorted(goldens, key=lambda g: g.test_name):
            console.print(f"  [cyan]{g.test_name}[/cyan]  [dim]score: {g.score:.0f}[/dim]")
        console.print(f"\n[dim]Usage: evalview replay <test_name>[/dim]\n")
        sys.exit(0)

    golden_variants = store.load_all_golden_variants(test_name)
    if not golden_variants:
        console.print(f"\n[red]❌ No baseline found for '{test_name}'[/red]")
        quoted = f'"{test_name}"' if " " in test_name else test_name
        console.print(f"[dim]Run: evalview snapshot --test {quoted}[/dim]\n")
        sys.exit(1)

    loader = TestCaseLoader()
    try:
        test_cases = loader.load_from_directory(Path(test_path))
    except Exception as e:
        console.print(f"\n[red]❌ Failed to load test cases: {e}[/red]\n")
        sys.exit(1)

    matching = [tc for tc in test_cases if tc.name == test_name]
    if not matching:
        console.print(f"\n[red]❌ No test case found with name: {test_name}[/red]")
        console.print(f"[dim]Available: {', '.join(tc.name for tc in test_cases) or 'none'}[/dim]\n")
        sys.exit(1)

    config = _load_config_if_exists()

    console.print(f"\n[cyan]◈ Replaying '{test_name}'...[/cyan]\n")

    diffs, results, _, _ = _execute_check_tests([matching[0]], config, json_output=False)

    if not results:
        console.print("[red]❌ Test execution failed — check your agent is running[/red]\n")
        sys.exit(1)

    result = results[0]
    golden = golden_variants[0]  # Primary baseline

    # Terminal: side-by-side step comparison
    _print_trajectory_diff(golden, result)

    # Diff summary
    if diffs:
        _, diff = diffs[0]
        from evalview.core.diff import DiffStatus
        status_display = {
            DiffStatus.PASSED: "[green]PASSED[/green]",
            DiffStatus.TOOLS_CHANGED: "[yellow]TOOLS_CHANGED[/yellow]",
            DiffStatus.OUTPUT_CHANGED: "[dim]OUTPUT_CHANGED[/dim]",
            DiffStatus.REGRESSION: "[red]REGRESSION[/red]",
        }.get(diff.overall_severity, str(diff.overall_severity))
        console.print(f"Status: {status_display}  |  {diff.summary()}\n")

    # Generate HTML report with side-by-side Mermaid trajectories
    golden_traces_dict = {test_name: golden}
    diff_list = [d for _, d in diffs]

    path = generate_visual_report(
        results=results,
        diffs=diff_list,
        golden_traces=golden_traces_dict,
        auto_open=not no_browser,
        title=f"Replay: {test_name}",
    )

    console.print(f"[green]◈ Report:[/green] {path}\n")
