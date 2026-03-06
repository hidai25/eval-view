"""Check and replay commands — regression detection against golden baselines."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

import click

from evalview.commands.shared import (
    console,
    _create_adapter,
    _load_config_if_exists,
    _cloud_pull,
)
from evalview.telemetry.decorators import track_command

if TYPE_CHECKING:
    from evalview.core.types import EvaluationResult, TestCase
    from evalview.core.config import EvalViewConfig
    from evalview.core.diff import TraceDiff
    from evalview.core.project_state import ProjectState
    from evalview.core.drift_tracker import DriftTracker


def _execute_check_tests(
    test_cases: List["TestCase"],
    config: Optional["EvalViewConfig"],
    json_output: bool,
    semantic_diff: bool = False,
) -> Tuple[List[Tuple[str, "TraceDiff"]], List["EvaluationResult"], "DriftTracker"]:
    """Execute tests and compare against golden variants.

    Args:
        test_cases: Test cases to run.
        config: EvalView config (adapter, endpoint, thresholds).
        json_output: Suppress non-JSON console output when True.
        semantic_diff: Enable embedding-based semantic similarity (opt-in).

    Returns:
        Tuple of (diffs, results, drift_tracker) where diffs is [(test_name, TraceDiff)].
        The drift_tracker is returned so callers can reuse it for detection without
        creating a second instance that would re-read the history file.
    """
    import asyncio
    from evalview.core.golden import GoldenStore
    from evalview.core.diff import DiffEngine
    from evalview.core.config import DiffConfig
    from evalview.core.drift_tracker import DriftTracker
    from evalview.evaluators.evaluator import Evaluator

    diff_config = config.get_diff_config() if config else DiffConfig()
    # --semantic-diff flag overrides config file setting
    if semantic_diff:
        diff_config = DiffConfig(
            **{**diff_config.model_dump(), "semantic_diff_enabled": True}
        )

    store = GoldenStore()
    diff_engine = DiffEngine(config=diff_config)
    drift_tracker = DriftTracker()
    evaluator = Evaluator()

    results = []
    diffs = []

    async def _run_one(tc) -> Optional[Tuple["EvaluationResult", "TraceDiff"]]:
        """Run a single test: execute → evaluate → diff (async pipeline)."""
        adapter_type = tc.adapter or (config.adapter if config else None)
        endpoint = tc.endpoint or (config.endpoint if config else None)
        if not adapter_type or not endpoint:
            return None

        allow_private = getattr(config, "allow_private_urls", True) if config else True
        try:
            adapter = _create_adapter(adapter_type, endpoint, allow_private_urls=allow_private)
        except ValueError as e:
            if not json_output:
                console.print(f"[yellow]⚠ Skipping {tc.name}: {e}[/yellow]")
            return None

        trace = await adapter.execute(tc.input.query, tc.input.context)
        result = await evaluator.evaluate(tc, trace)

        golden_variants = store.load_all_golden_variants(tc.name)
        if not golden_variants:
            return None

        # Use async comparison to include semantic diff when enabled
        diff = await diff_engine.compare_multi_reference_async(
            golden_variants, trace, result.score
        )
        return result, diff

    # Run all tests concurrently in a single event loop.
    # return_exceptions=True means exceptions are returned as values (not raised),
    # so one failing test does not cancel the others.
    async def _run_all() -> List:
        return await asyncio.gather(*[_run_one(tc) for tc in test_cases], return_exceptions=True)

    outcomes = asyncio.run(_run_all())

    for tc, outcome in zip(test_cases, outcomes):
        if isinstance(outcome, BaseException):
            if not json_output:
                if isinstance(outcome, (asyncio.TimeoutError, asyncio.CancelledError)):
                    console.print(f"[red]✗ {tc.name}: Async execution timed out — {outcome}[/red]")
                else:
                    console.print(f"[red]✗ {tc.name}: Failed — {outcome}[/red]")
            continue
        if outcome is None:
            continue
        result, diff = outcome
        results.append(result)
        diffs.append((tc.name, diff))
        drift_tracker.record_check(tc.name, diff)

    return diffs, results, drift_tracker


def _analyze_check_diffs(diffs: List[Tuple[str, "TraceDiff"]]) -> Dict[str, Any]:
    """Analyze diffs and return summary statistics.

    Returns:
        Dict with keys: has_regressions, has_tools_changed, has_output_changed, all_passed
    """
    from evalview.core.diff import DiffStatus

    has_regressions = any(d.overall_severity == DiffStatus.REGRESSION for _, d in diffs)
    has_tools_changed = any(d.overall_severity == DiffStatus.TOOLS_CHANGED for _, d in diffs)
    has_output_changed = any(d.overall_severity == DiffStatus.OUTPUT_CHANGED for _, d in diffs)
    all_passed = not has_regressions and not has_tools_changed and not has_output_changed

    return {
        "has_regressions": has_regressions,
        "has_tools_changed": has_tools_changed,
        "has_output_changed": has_output_changed,
        "all_passed": all_passed,
    }


def _display_check_results(
    diffs: List[Tuple[str, "TraceDiff"]],
    analysis: Dict[str, Any],
    state: "ProjectState",
    is_first_check: bool,
    json_output: bool,
    drift_tracker: Optional["DriftTracker"] = None,
) -> None:
    """Display check results in JSON or console format.

    Args:
        drift_tracker: DriftTracker instance from _execute_check_tests. If None,
                       a new instance is created (legacy behaviour). Pass the
                       instance from _execute_check_tests to avoid reading the
                       history file twice.
    """
    from evalview.core.diff import DiffStatus
    from evalview.core.celebrations import Celebrations
    from evalview.core.drift_tracker import DriftTracker
    from evalview.core.messages import get_random_clean_check_message
    from rich.panel import Panel

    if json_output:
        # JSON output for CI — include model change and semantic similarity info
        output = {
            "summary": {
                "total_tests": len(diffs),
                "unchanged": sum(1 for _, d in diffs if d.overall_severity == DiffStatus.PASSED),
                "regressions": sum(1 for _, d in diffs if d.overall_severity == DiffStatus.REGRESSION),
                "tools_changed": sum(1 for _, d in diffs if d.overall_severity == DiffStatus.TOOLS_CHANGED),
                "output_changed": sum(1 for _, d in diffs if d.overall_severity == DiffStatus.OUTPUT_CHANGED),
                "model_changed": any(getattr(d, "model_changed", False) for _, d in diffs),
            },
            "diffs": [
                {
                    "test_name": name,
                    "status": diff.overall_severity.value,
                    "score_delta": diff.score_diff,
                    "has_tool_diffs": len(diff.tool_diffs) > 0,
                    "output_similarity": diff.output_diff.similarity if diff.output_diff else 1.0,
                    "semantic_similarity": (
                        diff.output_diff.semantic_similarity if diff.output_diff else None
                    ),
                    "model_changed": getattr(diff, "model_changed", False),
                    "golden_model_id": getattr(diff, "golden_model_id", None),
                    "actual_model_id": getattr(diff, "actual_model_id", None),
                }
                for name, diff in diffs
            ],
        }
        print(json.dumps(output, indent=2))
    else:
        # Console output with personality
        if is_first_check:
            Celebrations.first_check()

        # Model version change warning — shown before pass/fail output so it
        # frames the context for any regressions the user is about to see.
        model_changed_diffs = [
            (name, d) for name, d in diffs if getattr(d, "model_changed", False)
        ]
        if model_changed_diffs:
            name, d = model_changed_diffs[0]
            golden_m = getattr(d, "golden_model_id", "unknown")
            actual_m = getattr(d, "actual_model_id", "unknown")
            console.print(
                Panel(
                    f"[yellow]Model changed:[/yellow] "
                    f"[dim]{golden_m}[/dim] → [bold]{actual_m}[/bold]\n\n"
                    "Baselines were captured with a different model version. "
                    "Output changes below may be caused by the model update rather "
                    "than your code. If the new behavior looks correct, run "
                    "[bold]evalview snapshot[/bold] to update the baseline.",
                    title="⚠  Model Version Change Detected",
                    border_style="yellow",
                )
            )
            console.print()

        # Gradual drift warnings — check each test for slow-burning decline.
        # Use the passed tracker (same instance used for recording) to avoid
        # a redundant instantiation and second read of the history file.
        _drift = drift_tracker if drift_tracker is not None else DriftTracker()
        for name, _ in diffs:
            warning = _drift.detect_gradual_drift(name)
            if warning:
                console.print(f"[yellow]📉 {name}:[/yellow] {warning}\n")

        if analysis["all_passed"]:
            # Clean check!
            console.print(f"[green]{get_random_clean_check_message()}[/green]\n")

            # Show streak celebration
            if state.current_streak >= 3:
                Celebrations.clean_check_streak(state)

            # Show health summary periodically
            if state.total_checks >= 5 and state.total_checks % 5 == 0:
                Celebrations.health_summary(state)
        else:
            # Show diffs
            console.print("\n[bold]Diff Summary[/bold]")
            unchanged = sum(1 for _, d in diffs if d.overall_severity == DiffStatus.PASSED)
            console.print(f"  {unchanged}/{len(diffs)} unchanged")
            if analysis["has_regressions"]:
                count = sum(1 for _, d in diffs if d.overall_severity == DiffStatus.REGRESSION)
                console.print(f"  {count} {'regression' if count == 1 else 'regressions'}")
            if analysis["has_tools_changed"]:
                count = sum(1 for _, d in diffs if d.overall_severity == DiffStatus.TOOLS_CHANGED)
                console.print(f"  {count} tool {'change' if count == 1 else 'changes'}")
            if analysis["has_output_changed"]:
                count = sum(1 for _, d in diffs if d.overall_severity == DiffStatus.OUTPUT_CHANGED)
                console.print(f"  {count} output {'change' if count == 1 else 'changes'}")

            console.print()

            # Show details of changed tests
            for name, diff in diffs:
                if diff.overall_severity != DiffStatus.PASSED:
                    severity_icon = {
                        DiffStatus.REGRESSION: "[red]✗ REGRESSION[/red]",
                        DiffStatus.TOOLS_CHANGED: "[yellow]⚠ TOOLS_CHANGED[/yellow]",
                        DiffStatus.OUTPUT_CHANGED: "[dim]~ OUTPUT_CHANGED[/dim]",
                    }.get(diff.overall_severity, "?")

                    console.print(f"{severity_icon}: {name}")
                    console.print(f"    {diff.summary()}")
                    quoted = f'"{name}"' if " " in name else name
                    console.print(f"    [dim]→ evalview replay {quoted}[/dim]")
                    console.print()

            # Show guidance
            if analysis["has_regressions"]:
                Celebrations.regression_guidance("See details above")


def _print_trajectory_diff(golden: Any, result: Any) -> None:
    """Print a side-by-side terminal trajectory comparison (golden vs actual)."""
    from rich.table import Table
    from rich.text import Text

    golden_steps: List[Any] = []
    actual_steps: List[Any] = []
    try:
        golden_steps = golden.trace.steps or []
    except AttributeError:
        pass
    try:
        actual_steps = result.trace.steps or []
    except AttributeError:
        pass

    if not golden_steps and not actual_steps:
        console.print("[dim]No tool steps in either trace — both are direct responses.[/dim]\n")
        return

    max_steps = max(len(golden_steps), len(actual_steps))

    table = Table(
        title="Trajectory Diff",
        show_header=True,
        header_style="bold",
        show_lines=True,
    )
    table.add_column("#", style="dim", width=3)
    table.add_column("Baseline", min_width=30)
    table.add_column("Current", min_width=30)
    table.add_column("", justify="center", width=3)

    for i in range(max_steps):
        g = golden_steps[i] if i < len(golden_steps) else None
        a = actual_steps[i] if i < len(actual_steps) else None

        g_name: str = str((getattr(g, "tool_name", None) or getattr(g, "step_name", "?")) if g else "—")
        a_name: str = str((getattr(a, "tool_name", None) or getattr(a, "step_name", "?")) if a else "—")

        match = g_name == a_name
        match_str = "[green]✓[/green]" if match else "[red]✗[/red]"

        if match:
            g_style, a_style = "cyan", "cyan"
        elif a_name == "—":
            g_style, a_style = "cyan", "red"   # step was dropped
        elif g_name == "—":
            g_style, a_style = "dim", "yellow"  # new step added
        else:
            g_style, a_style = "cyan", "yellow"  # step changed

        table.add_row(str(i + 1), Text(g_name, style=g_style), Text(a_name, style=a_style), match_str)

    console.print(table)
    console.print()

    golden_seq = [str(getattr(s, "tool_name", None) or getattr(s, "step_name", "?")) for s in golden_steps]
    actual_seq = [str(getattr(s, "tool_name", None) or getattr(s, "step_name", "?")) for s in actual_steps]

    if golden_seq == actual_seq:
        console.print("[green]Tool sequence: identical[/green]\n")
    else:
        console.print("[yellow]Tool sequence changed:[/yellow]")
        console.print(f"  Baseline: {' → '.join(golden_seq) or '(none)'}")
        console.print(f"  Current:  {' → '.join(actual_seq) or '(none)'}\n")


def _compute_check_exit_code(
    diffs: List[Tuple[str, "TraceDiff"]],
    fail_on: Optional[str],
    strict: bool
) -> int:
    """Compute exit code based on diff results and fail conditions.

    Returns:
        0 if no failures match fail conditions, 1 otherwise
    """
    if strict:
        fail_on = "REGRESSION,TOOLS_CHANGED,OUTPUT_CHANGED"

    if not fail_on:
        fail_on = "REGRESSION"  # Default

    fail_statuses = set(s.strip().upper() for s in fail_on.split(","))

    for _, diff in diffs:
        if diff.overall_severity.value.upper() in fail_statuses:
            return 1

    return 0


@click.command("check")
@click.argument("test_path", default="tests", type=click.Path(exists=True))
@click.option("--test", "-t", help="Check only this specific test")
@click.option("--json", "json_output", is_flag=True, help="Output JSON for CI")
@click.option("--fail-on", help="Comma-separated statuses to fail on (default: REGRESSION)")
@click.option("--strict", is_flag=True, help="Fail on any change (REGRESSION, TOOLS_CHANGED, OUTPUT_CHANGED)")
@click.option(
    "--semantic-diff",
    "semantic_diff",
    is_flag=True,
    default=False,
    help="Enable embedding-based semantic similarity (requires OPENAI_API_KEY, adds ~$0.00004/test)",
)
@track_command("check")
def check(test_path: str, test: str, json_output: bool, fail_on: str, strict: bool, semantic_diff: bool):
    """Check current behavior against snapshot baseline.

    This command runs tests and compares them against your saved baselines,
    showing only what changed. Perfect for CI/CD and daily development.

    TEST_PATH is the directory containing test cases (default: tests/).

    Examples:
        evalview check                                   # Check all tests
        evalview check --test "my-test"                  # Check one test
        evalview check --json                            # JSON output for CI
        evalview check --fail-on REGRESSION,TOOLS_CHANGED
        evalview check --strict                          # Fail on any change
    """
    from evalview.core.loader import TestCaseLoader
    from evalview.core.golden import GoldenStore
    from evalview.core.project_state import ProjectStateStore
    from evalview.core.celebrations import Celebrations
    from evalview.core.messages import get_random_checking_message

    # Initialize stores
    store = GoldenStore()
    state_store = ProjectStateStore()

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

    # Show status message
    if not json_output:
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

    # Load config
    config = _load_config_if_exists()

    # Semantic diff notice — shown once before execution so users know outputs
    # are sent to the OpenAI embedding API and can abort if undesired.
    if semantic_diff and not json_output:
        from evalview.core.semantic_diff import SemanticDiff
        if SemanticDiff.is_available():
            console.print(
                f"[dim]ℹ  Semantic diff enabled. {SemanticDiff.cost_notice()} "
                "Agent outputs are sent to OpenAI for embedding comparison.[/dim]\n"
            )
        else:
            console.print(
                "[yellow]⚠  --semantic-diff requested but OPENAI_API_KEY is not set. "
                "Falling back to lexical comparison.[/yellow]\n"
            )

    # Execute tests and compare against golden
    diffs, results, drift_tracker = _execute_check_tests(test_cases, config, json_output, semantic_diff)

    # Analyze diffs
    analysis = _analyze_check_diffs(diffs)

    # Update project state
    state = state_store.update_check(
        has_regressions=(not analysis["all_passed"]),
        status="passed" if analysis["all_passed"] else "regression"
    )

    # Display results (reuse drift_tracker instance to avoid re-reading history file)
    _display_check_results(diffs, analysis, state, is_first_check, json_output, drift_tracker=drift_tracker)

    # Compute and exit with code
    exit_code = _compute_check_exit_code(diffs, fail_on, strict)
    sys.exit(exit_code)


@click.command("replay")
@click.argument("test_name")
@click.option("--test-path", "test_path", default="tests", type=click.Path(exists=True), help="Directory containing tests")
@click.option("--no-browser", is_flag=True, help="Don't auto-open the HTML report")
@track_command("replay")
def replay(test_name: str, test_path: str, no_browser: bool) -> None:
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

    diffs, results, _ = _execute_check_tests([matching[0]], config, json_output=False)

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
