"""Check and replay commands — regression detection against golden baselines."""
from __future__ import annotations

import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, TYPE_CHECKING

import click

from evalview.commands.shared import (
    console,
    _load_config_if_exists,
    _cloud_pull,
    _create_adapter,
    _execute_check_tests,
    _analyze_check_diffs,
)
from evalview.commands.check_display import (
    _display_check_results,
    _print_trajectory_diff,
)
from evalview.telemetry.decorators import track_command

from evalview.core.diff import DiffStatus
from evalview.commands._check_verdict import (
    _VerdictOutput,  # noqa: F401  (re-exported for backward compat)
    _aggregate_cost_delta_ratio,  # noqa: F401  (re-exported for backward compat)
    _compute_check_exit_code,
    _compute_verdict_payload,
    _dedup_recommendations,  # noqa: F401  (re-exported for backward compat)
    _render_verdict_panel,
    _substitute_test_name,  # noqa: F401  (re-exported for backward compat)
)
from evalview.commands._check_helpers import (
    _all_failures_retry_healed,
    _filter_test_cases_by_tags,
    _format_baseline_timestamp,  # noqa: F401  (re-exported for backward compat)
    _format_snapshot_timestamp,  # noqa: F401  (re-exported for backward compat)
    _judge_usage_summary,
    _normalize_requested_tags,  # noqa: F401  (re-exported for backward compat)
    _print_baseline_context,
    _print_check_failure_guidance,
    _resolve_default_test_path,
    _should_auto_generate_report,
    _summarize_check_targets,  # noqa: F401  (re-exported for backward compat)
)

if TYPE_CHECKING:
    from evalview.core.types import EvaluationResult



@click.command("check")
@click.argument("test_path", default="tests", type=click.Path(exists=True))
@click.option("--test", "-t", help="Check only this specific test")
@click.option("--tag", "tags", multiple=True, help="Check only tests tagged with this behavior (repeatable).")
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
@click.option("--ai-root-cause", "ai_root_cause", is_flag=True, default=False, help="Use AI to explain low-confidence regressions (requires LLM provider, makes LLM API calls).")
@click.option("--explain", "explain", is_flag=True, default=False, help="Deep semantic root-cause analysis: feeds full baseline+current traces to LLM for a narrative explanation (requires LLM provider, makes LLM API calls).")
@click.option("--statistical", "statistical_runs", type=int, default=None, help="Run each test N times for variance analysis (e.g. --statistical 10).")
@click.option("--auto-variant", "auto_variant", is_flag=True, default=False, help="Auto-discover and save distinct execution paths as golden variants (use with --statistical).")
@click.option("--judge", "judge_model", default=None, help="Judge model for scoring (e.g. gpt-5.4-mini, sonnet, deepseek-chat).")
@click.option("--no-judge", "no_judge", is_flag=True, default=False, help="Skip LLM-as-judge evaluation. Uses deterministic scoring only (scores capped at 75). No API key required.")
@click.option("--heal", "heal_mode", is_flag=True, default=False, help="Auto-retry flaky failures, propose candidate variants. Never touches forbidden tools.")
@track_command("check")
def check(test_path: str, test: str, tags: tuple[str, ...], json_output: bool, fail_on: str, strict: bool, report_path: Optional[str], csv_path: Optional[str], semantic_diff: Optional[bool], budget: Optional[float], timeout: float, dry_run: bool, ai_root_cause: bool, explain: bool, statistical_runs: Optional[int], auto_variant: bool, judge_model: Optional[str], no_judge: bool, heal_mode: bool):
    """Decide whether it's safe to ship this agent change.

    Replays your test suite against the saved golden baselines and emits
    a single ship/don't-ship verdict — SAFE_TO_SHIP, SHIP_WITH_QUARANTINE,
    INVESTIGATE, or BLOCK_RELEASE — derived from the diff, quarantine
    state, cost delta, and drift confidence. The verdict is the one line
    you (and your PR reviewer, and your team's Slack channel) actually
    read; the diff details are there when you want to dig in.

    Perfect for CI/CD (exits 1 on regression by default) and daily
    development (auto-opens the HTML report when something changed).

    TEST_PATH is the directory containing test cases (default: tests/).

    Examples:
        evalview check                                   # Check all tests
        evalview check --test "my-test"                  # Check one test
        evalview check --tag tool_use --tag retrieval    # Check one behavior slice
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
        evalview check --explain                         # Deep trace narrative (feeds full traces to LLM)
        evalview check --statistical 10                  # Run each test 10 times, show variance
        evalview check --statistical 10 --auto-variant   # Auto-save distinct paths as variants
        evalview check --heal                            # Auto-retry flaky failures, propose variants
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

    test_cases, active_tags = _filter_test_cases_by_tags(test_cases, tags)
    if active_tags and not test_cases:
        console.print(f"[red]❌ No tests matched tags: {', '.join(active_tags)}[/red]\n")
        sys.exit(1)

    test_metadata = {
        tc.name: {
            "is_multi_turn": bool(getattr(tc, "is_multi_turn", False)),
            "behavior_class": (tc.meta or {}).get("behavior_class"),
            "tags": list(getattr(tc, "tags", []) or []),
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
            if active_tags:
                console.print(f"  Tags:           {', '.join(active_tags)}")
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

        if (ai_root_cause or explain) and not json_output:
            console.print(
                "[yellow]⚠  --ai-root-cause / --explain are not applied in --statistical mode "
                "(variance analysis exits before enrichment).[/yellow]\n"
            )

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
                skip_llm_judge=no_judge, budget_tracker=budget_tracker,
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
            lambda: _execute_check_tests(test_cases, config, json_output, semantic_diff, timeout, skip_llm_judge=no_judge, budget_tracker=budget_tracker),
            "Checking",
            len(test_cases),
        )
    else:
        diffs, results, drift_tracker, golden_traces = _execute_check_tests(
            test_cases, config, json_output, semantic_diff, timeout, skip_llm_judge=no_judge, budget_tracker=budget_tracker
        )

    golden_names = {golden.test_name for golden in goldens}
    baseline_test_cases = [tc for tc in test_cases if tc.name in golden_names]
    execution_failures = max(0, len(baseline_test_cases) - len(results))

    # --- Healing pass (never mutates original diffs) ---
    healing_summary = None
    all_failures_retry_healed = False
    if heal_mode and diffs:
        import asyncio as _asyncio
        from evalview.core.healing import (
            HealingEngine, HealingSummary, HealingAction, HealingTrigger,
            HealingDiagnosis, HealingResult,
            ModelUpdateSummary, save_audit_log,
            MIN_VARIANT_SCORE, MAX_COST_MULTIPLIER, MAX_LATENCY_MULTIPLIER,
            MAX_AUTO_VARIANTS,
        )
        from evalview.core.diff import DiffEngine as _HealDiffEngine, DiffConfig as _HealDiffConfig
        from evalview.evaluators.evaluator import Evaluator as _HealEvaluator

        _heal_diff_config = config.get_diff_config() if config else _HealDiffConfig()
        heal_diff_engine = _HealDiffEngine(config=_heal_diff_config)
        heal_evaluator = _HealEvaluator()
        engine = HealingEngine(store, heal_evaluator)

        healing_results: List[Any] = []

        async def _heal_all() -> None:
            for (name, diff_item), result_item in zip(diffs, results):
                if diff_item.overall_severity == DiffStatus.PASSED:
                    continue
                tc = next((t for t in test_cases if t.name == name), None)
                if tc is None:
                    healing_results.append(HealingResult(
                        test_name=name,
                        original_status=diff_item.overall_severity.value,
                        diagnosis=HealingDiagnosis(
                            action=HealingAction.FLAG_REVIEW,
                            trigger=HealingTrigger.OTHER,
                            reason="heal skipped: no matching test case loaded",
                            details={"skip_reason": "missing_test_case"},
                        ),
                        attempted=False,
                        healed=False,
                        final_status=diff_item.overall_severity.value,
                        original_score=result_item.score,
                        actual_model=getattr(result_item.trace, "model_id", None),
                    ))
                    continue
                gv = store.load_all_golden_variants(name)
                if not gv:
                    healing_results.append(HealingResult(
                        test_name=name,
                        original_status=diff_item.overall_severity.value,
                        diagnosis=HealingDiagnosis(
                            action=HealingAction.FLAG_REVIEW,
                            trigger=HealingTrigger.OTHER,
                            reason="heal skipped: no baseline variants available",
                            details={"skip_reason": "missing_golden_variants"},
                        ),
                        attempted=False,
                        healed=False,
                        final_status=diff_item.overall_severity.value,
                        original_score=result_item.score,
                        actual_model=getattr(result_item.trace, "model_id", None),
                    ))
                    continue

                adapter_type = tc.adapter or (config.adapter if config else None)
                endpoint = tc.endpoint or (config.endpoint if config else None)
                if not adapter_type or not endpoint:
                    healing_results.append(HealingResult(
                        test_name=name,
                        original_status=diff_item.overall_severity.value,
                        diagnosis=HealingDiagnosis(
                            action=HealingAction.FLAG_REVIEW,
                            trigger=HealingTrigger.OTHER,
                            reason="heal skipped: adapter or endpoint missing",
                            details={"skip_reason": "missing_adapter_or_endpoint"},
                        ),
                        attempted=False,
                        healed=False,
                        final_status=diff_item.overall_severity.value,
                        original_score=result_item.score,
                        baseline_score=gv[0].metadata.score if gv else None,
                        baseline_model=gv[0].metadata.model_id if gv else None,
                        actual_model=getattr(result_item.trace, "model_id", None),
                    ))
                    continue

                try:
                    adapter = _create_adapter(adapter_type, endpoint, timeout=timeout)
                    hr = await engine.heal_test(
                        diff_item, result_item, tc, gv, adapter, heal_diff_engine
                    )
                    healing_results.append(hr)
                except Exception as exc:
                    if not json_output:
                        console.print(f"[yellow]  Heal failed for {name}: {exc}[/yellow]")
                    healing_results.append(HealingResult(
                        test_name=name,
                        original_status=diff_item.overall_severity.value,
                        diagnosis=HealingDiagnosis(
                            action=HealingAction.FLAG_REVIEW,
                            trigger=HealingTrigger.OTHER,
                            reason=f"heal error: {exc}",
                            details={"error_type": type(exc).__name__},
                        ),
                        attempted=True,
                        healed=False,
                        final_status=diff_item.overall_severity.value,
                        original_score=result_item.score,
                        baseline_score=gv[0].metadata.score if gv else None,
                        baseline_model=gv[0].metadata.model_id if gv else None,
                        actual_model=getattr(result_item.trace, "model_id", None),
                    ))

        _asyncio.run(_heal_all())

        # Build model update summary if any tests had a model/runtime change signal
        model_update = None
        model_affected = [
            (n, d) for n, d in diffs
            if d.model_changed or getattr(d, "runtime_fingerprint_changed", False)
        ]
        if model_affected:
            _, first_model_diff = model_affected[0]
            model_healed = sum(
                1 for r in healing_results
                if r.healed and r.diagnosis.trigger == HealingTrigger.MODEL_UPDATE
            )
            model_update = ModelUpdateSummary(
                golden_model=(
                    first_model_diff.golden_model_id
                    or getattr(first_model_diff, "golden_runtime_fingerprint", None)
                    or "unknown"
                ),
                actual_model=(
                    first_model_diff.actual_model_id
                    or getattr(first_model_diff, "actual_runtime_fingerprint", None)
                    or "unknown"
                ),
                affected_count=len(model_affected),
                healed_count=model_healed,
                failed_count=len(model_affected) - model_healed,
            )

        healing_summary = HealingSummary(
            results=healing_results,
            total_healed=sum(1 for r in healing_results if r.healed),
            total_proposed=sum(1 for r in healing_results if r.proposed),
            total_review=sum(
                1 for r in healing_results
                if r.diagnosis.action == HealingAction.FLAG_REVIEW
            ),
            total_blocked=sum(
                1 for r in healing_results
                if r.diagnosis.action == HealingAction.BLOCKED
            ),
            attempted_count=sum(1 for r in healing_results if r.attempted),
            unresolved_count=sum(1 for r in healing_results if not r.healed),
            failed_count=len(healing_results),
            thresholds={
                "min_variant_score": MIN_VARIANT_SCORE,
                "max_cost_multiplier": MAX_COST_MULTIPLIER,
                "max_latency_multiplier": MAX_LATENCY_MULTIPLIER,
                "max_auto_variants": float(MAX_AUTO_VARIANTS),
            },
            model_update=model_update,
        )
        if healing_results:
            healing_summary.audit_path = save_audit_log(healing_summary)
        all_failures_retry_healed = _all_failures_retry_healed(
            diffs, healing_summary, execution_failures=execution_failures
        )

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

    analysis["healing_enabled"] = bool(heal_mode)
    analysis["healing_all_resolved"] = all_failures_retry_healed
    analysis["effective_all_passed"] = (
        analysis["all_passed"] or all_failures_retry_healed
    )
    analysis["has_unresolved_failures"] = not analysis["effective_all_passed"]

    # Update project state (only count real checks toward streaks)
    if actually_compared > 0:
        state = state_store.update_check(
            has_regressions=analysis["has_unresolved_failures"],
            status="passed" if analysis["effective_all_passed"] else "regression"
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

    # AI root cause enrichment (opt-in via --ai-root-cause)
    import asyncio as _asyncio_rc
    ai_root_causes = None
    if ai_root_cause and analysis["has_unresolved_failures"]:
        from evalview.core.root_cause import enrich_diffs_with_ai
        if not json_output:
            console.print("[dim]🤖 Running AI root cause analysis...[/dim]\n")
        ai_root_causes = _asyncio_rc.run(enrich_diffs_with_ai(diffs))

    # Narrative enrichment (opt-in via --explain) — feeds full traces to LLM
    narrative_root_causes = None
    if explain and analysis["has_unresolved_failures"]:
        from evalview.core.root_cause import enrich_diffs_with_narrative
        if not json_output:
            console.print("[dim]🔍 Running deep trace analysis (--explain)...[/dim]\n")
        narrative_root_causes = _asyncio_rc.run(
            enrich_diffs_with_narrative(diffs, golden_traces=golden_traces, results=results)
        )

    # Merge AI and narrative enrichments: both fields live on the same RootCauseAnalysis
    # object so the display layer and HTML report have everything in one place.
    combined_root_causes: Optional[Dict[str, Any]] = None
    if ai_root_causes or narrative_root_causes:
        combined_root_causes = {}
        for name, _diff in diffs:
            ai_rc = (ai_root_causes or {}).get(name)
            nar_rc = (narrative_root_causes or {}).get(name)
            if nar_rc is not None and ai_rc is not None:
                # Carry ai_explanation onto the narrative-enriched object
                nar_rc.ai_explanation = ai_rc.ai_explanation
                combined_root_causes[name] = nar_rc
            elif nar_rc is not None:
                combined_root_causes[name] = nar_rc
            elif ai_rc is not None:
                combined_root_causes[name] = ai_rc

    from evalview.core.model_runtime_detector import analyze_model_runtime_change

    model_runtime_summary = analyze_model_runtime_change(
        diffs,
        healing_summary=healing_summary,
    )

    # ── Verdict layer (computed before display so --json can include it) ──
    # One QuarantineStore for the whole check run — shared between the
    # verdict computation and the exit-code computation to avoid a
    # double YAML read and a race window where the two could disagree.
    from evalview.core.quarantine import QuarantineStore as _QuarantineStore
    shared_quarantine = _QuarantineStore()

    verdict_output = _compute_verdict_payload(
        diffs=diffs,
        results=results,
        drift_tracker=drift_tracker,
        execution_failures=execution_failures,
        golden_traces=golden_traces,
        quarantine=shared_quarantine,
    )

    # --- Enrich verdict payload with observability signals ---
    from evalview.core.observability import extract_observability_summary
    _obs = extract_observability_summary(results)
    verdict_output.payload.update(_obs.to_verdict_payload())

    # Display results
    _display_check_results(
        diffs, analysis, state, is_first_check, json_output,
        drift_tracker=drift_tracker,
        golden_traces=golden_traces,
        results=results,
        ai_root_causes=combined_root_causes,
        test_metadata=test_metadata,
        healing_summary=healing_summary,
        model_runtime_summary=model_runtime_summary,
        verdict_payload=verdict_output.payload,
    )

    # Render the verdict panel as the last thing the user sees (screenshotable).
    if not json_output:
        _render_verdict_panel(verdict_output)

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
            auto_open=not json_output and not bool(__import__("os").environ.get("CI")),
            title="EvalView Check Report",
            default_tab=tab,
            healing_summary=healing_summary,
            model_runtime_summary=model_runtime_summary,
            effective_all_passed=analysis["effective_all_passed"],
            test_metadata=test_metadata,
            active_tags=active_tags,
            root_causes=combined_root_causes,
        )
        if not json_output:
            if auto_report:
                label = "Check report" if analysis["effective_all_passed"] else "Failure report"
                console.print(f"[green]◈ {label}:[/green] {path}")
                if analysis["effective_all_passed"]:
                    console.print("[dim]Opened automatically with healing details and audit context.[/dim]\n")
                else:
                    console.print("[dim]Opened automatically because this check found unresolved changes or execution failures.[/dim]\n")
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

    # ── Push results to EvalView Cloud (best-effort) ──
    # Runs after all display/report work so it never blocks the user
    # experience. Silently skips when EVALVIEW_API_TOKEN is unset.
    try:
        from evalview.cloud.push import _get_api_token, _get_git_context, _push_async
        import asyncio as _cloud_asyncio
        import hashlib as _cloud_hash

        _cloud_token = _get_api_token()
        if _cloud_token:
            _total_cost = sum(
                r.trace.metrics.total_cost for r in results
                if hasattr(r, "trace") and hasattr(r.trace, "metrics")
            ) if results else 0
            _total_latency = sum(
                r.trace.metrics.total_latency for r in results
                if hasattr(r, "trace") and hasattr(r.trace, "metrics")
            ) if results else 0

            _n_regression = sum(1 for _, d in diffs if d.overall_severity == DiffStatus.REGRESSION)
            _n_tools = sum(1 for _, d in diffs if d.overall_severity == DiffStatus.TOOLS_CHANGED)
            _n_output = sum(1 for _, d in diffs if d.overall_severity == DiffStatus.OUTPUT_CHANGED)
            _n_passed = sum(1 for _, d in diffs if d.overall_severity == DiffStatus.PASSED)

            if _n_regression > 0:
                _cloud_status = "regression"
            elif _n_tools > 0:
                _cloud_status = "tools_changed"
            elif _n_output > 0:
                _cloud_status = "output_changed"
            else:
                _cloud_status = "passed"

            _cloud_payload = {
                "run_id": _cloud_hash.md5(str(datetime.now()).encode()).hexdigest()[:8],
                "status": _cloud_status,
                "source": "ci" if __import__("os").environ.get("CI") else "cli",
                **_get_git_context(),
                "summary": {
                    "total": len(diffs),
                    "unchanged": _n_passed,
                    "regressions": _n_regression,
                    "tools_changed": _n_tools,
                    "output_changed": _n_output,
                },
                "total_cost": _total_cost,
                "total_latency_ms": _total_latency,
                "diffs": [
                    {
                        "test_name": d.test_name,
                        "status": d.overall_severity.value,
                        "score_delta": d.score_diff or 0,
                        "output_similarity": d.output_diff.similarity if d.output_diff else None,
                        "tool_changes": len(d.tool_diffs) if d.tool_diffs else 0,
                        "model_changed": d.model_changed,
                    }
                    for _, d in diffs
                ],
                "result_json": verdict_output.payload or {},
            }
            _dashboard_url = _cloud_asyncio.run(_push_async(_cloud_payload, _cloud_token))
            if _dashboard_url and not json_output:
                console.print(f"[green]☁  Pushed to cloud:[/green] {_dashboard_url}\n")
    except Exception:
        pass

    # Compute and exit with code
    exit_code = _compute_check_exit_code(
        diffs, fail_on, strict,
        execution_failures=execution_failures,
        quarantine=shared_quarantine,
    )

    # --heal exit code: 0 only if every failure was retry-healed
    if all_failures_retry_healed:
        exit_code = 0

    if (
        heal_mode
        and healing_summary
        and healing_summary.unresolved_count > 0
        and exit_code == 0
        and not json_output
    ):
        console.print(
            "[yellow]⚠ Unresolved healing review items remain, but this run exits 0 under the current "
            "--fail-on policy.[/yellow]"
        )
        console.print(
            "[dim]Use --strict or --fail-on REGRESSION,TOOLS_CHANGED,OUTPUT_CHANGED if review cases should fail CI.[/dim]\n"
        )

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
        console.print("\n[dim]Usage: evalview replay <test_name>[/dim]\n")
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

    # Observability signals from the evaluation
    ar = result.anomaly_report
    if ar is not None:
        for anom in ar.get("anomalies", [])[:5]:
            sev = anom.get("severity", "warning")
            icon = "[red]\u26a0[/red]" if sev == "error" else "[yellow]\u26a0[/yellow]"
            console.print(
                f"  {icon} [bold]{anom.get('pattern', '')}[/bold]: "
                f"{anom.get('description', '')}"
            )
        console.print()

    tr = result.trust_report
    if tr is not None:
        trust_val = tr.get("trust_score", 1.0)
        if trust_val < 1.0:
            console.print(f"  Trust: {trust_val:.0%} — {tr.get('summary', '')}")
            console.print()

    cr = result.coherence_report
    if cr is not None:
        for issue in cr.get("issues", [])[:3]:
            sev = issue.get("severity", "warning")
            icon = "[red]\u26a0[/red]" if sev == "error" else "[yellow]\u26a0[/yellow]"
            console.print(
                f"  {icon} [bold]{issue.get('category', '')}[/bold]: "
                f"{issue.get('description', '')}"
            )
        if cr.get("issues"):
            console.print()

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
