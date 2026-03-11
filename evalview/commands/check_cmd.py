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
    from evalview.core.diff import TraceDiff, ToolDiff
    from evalview.core.project_state import ProjectState
    from evalview.core.drift_tracker import DriftTracker
    from evalview.core.golden import GoldenTrace


def _execute_check_tests(
    test_cases: List["TestCase"],
    config: Optional["EvalViewConfig"],
    json_output: bool,
    semantic_diff: bool = False,
    timeout: float = 30.0,
) -> Tuple[List[Tuple[str, "TraceDiff"]], List["EvaluationResult"], "DriftTracker", Dict[str, "GoldenTrace"]]:
    """Execute tests and compare against golden variants.

    Args:
        test_cases: Test cases to run.
        config: EvalView config (adapter, endpoint, thresholds).
        json_output: Suppress non-JSON console output when True.
        semantic_diff: Enable embedding-based semantic similarity (opt-in).

    Returns:
        Tuple of (diffs, results, drift_tracker, golden_traces) where
        diffs is [(test_name, TraceDiff)] and golden_traces maps test name
        to the primary GoldenTrace used for comparison.
    """
    import asyncio
    from evalview.core.golden import GoldenStore, GoldenTrace
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

    results: List["EvaluationResult"] = []
    diffs: List[Tuple[str, "TraceDiff"]] = []
    golden_traces: Dict[str, GoldenTrace] = {}

    async def _run_one(tc) -> Optional[Tuple["EvaluationResult", "TraceDiff", GoldenTrace]]:
        """Run a single test: execute → evaluate → diff (async pipeline)."""
        adapter_type = tc.adapter or (config.adapter if config else None)
        endpoint = tc.endpoint or (config.endpoint if config else None)
        if not adapter_type or not endpoint:
            return None

        allow_private = getattr(config, "allow_private_urls", True) if config else True
        try:
            adapter = _create_adapter(adapter_type, endpoint, timeout=timeout, allow_private_urls=allow_private)
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
        return result, diff, golden_variants[0]

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
        result, diff, golden = outcome
        results.append(result)
        diffs.append((tc.name, diff))
        golden_traces[tc.name] = golden
        drift_tracker.record_check(tc.name, diff)

    return diffs, results, drift_tracker, golden_traces


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


def _print_parameter_diffs(tool_diffs: List["ToolDiff"]) -> None:
    """Print parameter-level differences for tool calls."""
    from rich.table import Table

    has_param_diffs = any(td.parameter_diffs for td in tool_diffs)
    if not has_param_diffs:
        return

    table = Table(
        title="Parameter Changes",
        show_header=True,
        header_style="bold",
        show_lines=False,
        padding=(0, 1),
    )
    table.add_column("Step", style="dim", width=5)
    table.add_column("Tool", style="cyan", min_width=12)
    table.add_column("Param", style="bold", min_width=10)
    table.add_column("Baseline", min_width=15)
    table.add_column("Current", min_width=15)
    table.add_column("", width=8)

    for td in tool_diffs:
        if not td.parameter_diffs:
            continue
        tool_name = td.golden_tool or td.actual_tool or "?"
        for pd in td.parameter_diffs:
            # Format the change indicator
            if pd.diff_type == "missing":
                indicator = "[red]-removed[/red]"
                golden_val = str(pd.golden_value)[:40]
                actual_val = "[dim]—[/dim]"
            elif pd.diff_type == "added":
                indicator = "[green]+added[/green]"
                golden_val = "[dim]—[/dim]"
                actual_val = str(pd.actual_value)[:40]
            elif pd.similarity is not None:
                pct = int(pd.similarity * 100)
                color = "green" if pct >= 80 else "yellow" if pct >= 50 else "red"
                indicator = f"[{color}]{pct}%[/{color}]"
                golden_val = str(pd.golden_value)[:40]
                actual_val = str(pd.actual_value)[:40]
            else:
                indicator = "[yellow]~[/yellow]"
                golden_val = str(pd.golden_value)[:40]
                actual_val = str(pd.actual_value)[:40]

            table.add_row(
                str(td.position + 1),
                tool_name,
                pd.param_name,
                golden_val,
                actual_val,
                indicator,
            )

    console.print(table)
    console.print()


def _print_output_diff(diff: "TraceDiff") -> None:
    """Print output similarity and unified diff excerpt."""
    if not diff.output_diff:
        return

    od = diff.output_diff
    if od.similarity >= 0.95:
        return  # Close enough, don't show

    # Similarity line
    sim_pct = int(od.similarity * 100)
    sim_color = "green" if sim_pct >= 80 else "yellow" if sim_pct >= 50 else "red"
    parts = [f"[{sim_color}]{sim_pct}% lexical[/{sim_color}]"]
    if od.semantic_similarity is not None:
        sem_pct = int(od.semantic_similarity * 100)
        sem_color = "green" if sem_pct >= 80 else "yellow" if sem_pct >= 50 else "red"
        parts.append(f"[{sem_color}]{sem_pct}% semantic[/{sem_color}]")
    console.print(f"    Output similarity: {' / '.join(parts)}")

    # Show a few diff lines (max 8) for context
    meaningful_lines = [
        line for line in od.diff_lines
        if line.startswith("+") or line.startswith("-")
        if not line.startswith("+++") and not line.startswith("---")
    ]
    if meaningful_lines:
        for line in meaningful_lines[:8]:
            if line.startswith("+"):
                console.print(f"      [green]{line}[/green]")
            else:
                console.print(f"      [red]{line}[/red]")
        if len(meaningful_lines) > 8:
            console.print(f"      [dim]... {len(meaningful_lines) - 8} more lines[/dim]")
    console.print()


def _print_inline_trajectory(diff: "TraceDiff", golden: Optional["GoldenTrace"], result: Optional["EvaluationResult"]) -> None:
    """Print a compact inline trajectory comparison for check output."""
    golden_seq: List[str] = []
    actual_seq: List[str] = []

    if golden:
        golden_seq = golden.tool_sequence or []
    if result:
        try:
            actual_seq = [
                str(getattr(s, "tool_name", None) or getattr(s, "step_name", "?"))
                for s in (result.trace.steps or [])
            ]
        except AttributeError:
            pass

    if not golden_seq and not actual_seq:
        return

    if golden_seq != actual_seq:
        console.print(f"    [dim]Baseline:[/dim] {' → '.join(golden_seq) or '(none)'}")
        console.print(f"    [dim]Current:[/dim]  {' → '.join(actual_seq) or '(none)'}")


def _display_check_results(
    diffs: List[Tuple[str, "TraceDiff"]],
    analysis: Dict[str, Any],
    state: "ProjectState",
    is_first_check: bool,
    json_output: bool,
    drift_tracker: Optional["DriftTracker"] = None,
    golden_traces: Optional[Dict[str, "GoldenTrace"]] = None,
    results: Optional[List["EvaluationResult"]] = None,
) -> None:
    """Display check results in JSON or console format.

    Args:
        drift_tracker: DriftTracker instance from _execute_check_tests. If None,
                       a new instance is created (legacy behaviour). Pass the
                       instance from _execute_check_tests to avoid reading the
                       history file twice.
        golden_traces: Dict mapping test name to primary GoldenTrace (for inline trajectory).
        results: List of EvaluationResult objects (for inline trajectory).
    """
    from evalview.core.diff import DiffStatus
    from evalview.core.celebrations import Celebrations
    from evalview.core.drift_tracker import DriftTracker
    from evalview.core.messages import get_random_clean_check_message
    from rich.panel import Panel

    # Build result lookup by test name
    result_by_name: Dict[str, Any] = {}
    if results:
        for r in results:
            result_by_name[r.test_case] = r

    if json_output:
        # JSON output for CI — include model change, semantic similarity, and
        # per-test tool diffs and parameter changes for machine consumption.
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
                    "tool_diffs": [
                        {
                            "type": td.type,
                            "position": td.position,
                            "golden_tool": td.golden_tool,
                            "actual_tool": td.actual_tool,
                            "message": td.message,
                            "parameter_diffs": [
                                {
                                    "param": pd.param_name,
                                    "golden": pd.golden_value,
                                    "actual": pd.actual_value,
                                    "type": pd.diff_type,
                                    "similarity": pd.similarity,
                                }
                                for pd in td.parameter_diffs
                            ],
                        }
                        for td in diff.tool_diffs
                    ],
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

            # Show details of changed tests with inline trajectory + parameter diffs
            _goldens = golden_traces or {}
            for name, diff in diffs:
                if diff.overall_severity != DiffStatus.PASSED:
                    severity_icon = {
                        DiffStatus.REGRESSION: "[red]✗ REGRESSION[/red]",
                        DiffStatus.TOOLS_CHANGED: "[yellow]⚠ TOOLS_CHANGED[/yellow]",
                        DiffStatus.OUTPUT_CHANGED: "[dim]~ OUTPUT_CHANGED[/dim]",
                    }.get(diff.overall_severity, "?")

                    # Score delta
                    score_part = ""
                    if abs(diff.score_diff) > 1:
                        sign = "+" if diff.score_diff > 0 else ""
                        score_color = "green" if diff.score_diff > 0 else "red"
                        score_part = f"  [{score_color}]{sign}{diff.score_diff:.1f} pts[/{score_color}]"

                    console.print(f"{severity_icon}: {name}{score_part}")

                    # Inline trajectory comparison (tool sequence)
                    golden_for_test = _goldens.get(name)
                    result_for_test = result_by_name.get(name)
                    _print_inline_trajectory(diff, golden_for_test, result_for_test)

                    # Parameter-level diffs (compact)
                    if diff.tool_diffs:
                        _print_parameter_diffs(diff.tool_diffs)

                    # Output similarity + diff excerpt
                    _print_output_diff(diff)

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
@click.option("--report", "report_path", default=None, type=click.Path(), help="Generate HTML report at this path (auto-opens in browser)")
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
@click.option("--timeout", type=float, default=30.0, help="Timeout per test in seconds (default: 30.0).")
@click.option("--dry-run", "dry_run", is_flag=True, default=False, help="Preview test plan without executing.")
@track_command("check")
def check(test_path: str, test: str, json_output: bool, fail_on: str, strict: bool, report_path: Optional[str], semantic_diff: Optional[bool], budget: Optional[float], timeout: float, dry_run: bool):
    """Check current behavior against snapshot baseline.

    This command runs tests and compares them against your saved baselines,
    showing only what changed. Perfect for CI/CD and daily development.

    TEST_PATH is the directory containing test cases (default: tests/).

    Examples:
        evalview check                                   # Check all tests
        evalview check --test "my-test"                  # Check one test
        evalview check --json                            # JSON output for CI
        evalview check --report report.html              # Generate HTML report
        evalview check --fail-on REGRESSION,TOOLS_CHANGED
        evalview check --strict                          # Fail on any change
        evalview check --no-semantic-diff                # Opt out of semantic diff
        evalview check --dry-run                         # Preview plan, no API calls
        evalview check --budget 0.50                     # Cap spend at $0.50
        evalview check --timeout 60                      # 60 second timeout per test
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

    # Apply judge config from config.yaml (env vars / CLI flags take priority)
    from evalview.core.config import apply_judge_config
    apply_judge_config(config)

    # Resolve semantic diff: explicit flag > config file > auto-enable.
    # Priority (highest to lowest):
    #   1. --no-semantic-diff flag  → always off
    #   2. --semantic-diff flag     → on if key available, warn otherwise
    #   3. config semantic_diff_enabled: false → always off
    #   4. auto-enable when OPENAI_API_KEY is set
    from evalview.core.semantic_diff import SemanticDiff
    key_available = SemanticDiff.is_available()

    if semantic_diff is None:
        # No explicit flag — check config, then auto-enable if key is present.
        config_setting = config.get_diff_config().semantic_diff_enabled if config else None
        if config_setting is False:
            # User explicitly disabled it in config — respect that.
            semantic_diff = False
        else:
            semantic_diff = key_available
        if semantic_diff and not json_output:
            # Show one-time notice so users know this is happening
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
        # User explicitly requested it but key is missing
        if not json_output:
            console.print(
                "[yellow]⚠  --semantic-diff requested but OPENAI_API_KEY is not set. "
                "Falling back to lexical comparison.[/yellow]\n"
            )
        semantic_diff = False

    # Dry-run mode — show plan and exit
    if dry_run:
        # Use the already-loaded golden list instead of per-test file reads
        golden_names = set(goldens)
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
            import json
            print(json.dumps({"dry_run": True, "tests": len(test_cases), "with_baselines": tests_with_baselines}))
        sys.exit(0)

    # Execute tests and compare against golden
    diffs, results, drift_tracker, golden_traces = _execute_check_tests(test_cases, config, json_output, semantic_diff, timeout)

    # Analyze diffs
    analysis = _analyze_check_diffs(diffs)

    # Update project state
    state = state_store.update_check(
        has_regressions=(not analysis["all_passed"]),
        status="passed" if analysis["all_passed"] else "regression"
    )

    # Cost summary
    if results and not json_output:
        total_cost = sum(r.trace.metrics.total_cost for r in results)
        total_api_calls = sum(len(r.trace.steps) for r in results)
        console.print(
            f"[dim]💰 {len(results)} tests, {total_api_calls} API calls, "
            f"${total_cost:.4f} total[/dim]\n"
        )
        if budget is not None and total_cost > budget:
            console.print(
                f"[red]⚠  Budget exceeded: ${total_cost:.4f} > ${budget:.2f} limit[/red]\n"
            )
            sys.exit(1)

    # Display results (reuse drift_tracker instance to avoid re-reading history file)
    _display_check_results(
        diffs, analysis, state, is_first_check, json_output,
        drift_tracker=drift_tracker,
        golden_traces=golden_traces,
        results=results,
    )

    # Generate HTML report if requested
    if report_path and results:
        from evalview.visualization import generate_visual_report
        diff_list = [d for _, d in diffs]
        path = generate_visual_report(
            results=results,
            diffs=diff_list,
            golden_traces=golden_traces,
            output_path=report_path,
            auto_open=not json_output,
            title="EvalView Check Report",
        )
        if not json_output:
            console.print(f"[green]◈ Report:[/green] {path}\n")

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
