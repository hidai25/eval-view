"""Display and printing functions for check command output."""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

from evalview.commands.shared import console

if TYPE_CHECKING:
    from evalview.core.types import EvaluationResult
    from evalview.core.diff import TraceDiff, ToolDiff
    from evalview.core.project_state import ProjectState
    from evalview.core.drift_tracker import DriftTracker
    from evalview.core.golden import GoldenTrace
    from evalview.core.root_cause import RootCauseAnalysis
    from evalview.core.healing import HealingSummary


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


def _print_root_cause(root_cause: "RootCauseAnalysis") -> None:
    """Print root cause attribution for a regression."""
    from rich.panel import Panel

    confidence_color = {
        "high": "green",
        "medium": "yellow",
        "low": "dim",
    }.get(root_cause.confidence.value, "dim")

    lines = [
        f"[bold]Root cause:[/bold] {root_cause.category.value} ([{confidence_color}]{root_cause.confidence.value} confidence[/{confidence_color}])",
        f"  [dim]→[/dim] {root_cause.summary}",
    ]
    if root_cause.suggested_fix:
        lines.append(f"  [dim]Fix:[/dim] {root_cause.suggested_fix}")

    if getattr(root_cause, "ai_explanation", None):
        lines.append(f"  [cyan]🤖 AI:[/cyan] {root_cause.ai_explanation}")

    console.print(Panel(
        "\n".join(lines),
        border_style="red" if root_cause.confidence.value == "high" else "yellow",
        padding=(0, 1),
    ))


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


def _print_passed_summary(
    diffs: List[Tuple[str, "TraceDiff"]],
    result_by_name: Dict[str, Any],
    golden_traces: Optional[Dict[str, "GoldenTrace"]] = None,
) -> None:
    """Print a compact summary of what was verified when all tests pass.

    Shows each test with its score, tool match status, and output similarity
    so users have evidence that real work was done.
    """
    _goldens = golden_traces or {}
    for name, diff in diffs:
        result = result_by_name.get(name)
        score_str = f"{result.score:.1f}" if result else "?"

        parts = [f"[green]✓[/green] {name:<30s} [bold]{score_str}[/bold]"]

        # Tool match status
        if not diff.tool_diffs:
            parts.append("[green]tools ✓[/green]")
        else:
            parts.append(f"[yellow]tools ~{len(diff.tool_diffs)} changed[/yellow]")

        # Output similarity
        if diff.output_diff:
            sim_pct = int(diff.output_diff.similarity * 100)
            if sim_pct >= 95:
                parts.append(f"[green]output {sim_pct}%[/green]")
            else:
                color = "yellow" if sim_pct >= 80 else "red"
                parts.append(f"[{color}]output {sim_pct}%[/{color}]")

        # Multi-turn indicator
        if diff.turn_diffs:
            n_turns = len(diff.turn_diffs)
            parts.append(f"[dim]({n_turns} turns)[/dim]")

        console.print("  " + "  ".join(parts))

    # Summary line with test type breakdown
    n_multi = sum(1 for _, d in diffs if d.turn_diffs)
    n_single = len(diffs) - n_multi
    type_parts = []
    if n_single:
        type_parts.append(f"{n_single} single-turn")
    if n_multi:
        type_parts.append(f"{n_multi} multi-turn")
    if type_parts:
        console.print(f"  [dim]{len(diffs)} tests checked: {', '.join(type_parts)}[/dim]")
    console.print()


def _display_check_results(
    diffs: List[Tuple[str, "TraceDiff"]],
    analysis: Dict[str, Any],
    state: "ProjectState",
    is_first_check: bool,
    json_output: bool,
    drift_tracker: Optional["DriftTracker"] = None,
    golden_traces: Optional[Dict[str, "GoldenTrace"]] = None,
    results: Optional[List["EvaluationResult"]] = None,
    ai_root_causes: Optional[Dict[str, Any]] = None,
    test_metadata: Optional[Dict[str, Dict[str, Any]]] = None,
    healing_summary: Optional["HealingSummary"] = None,
) -> None:
    """Display check results in JSON or console format."""
    import json

    from evalview.core.diff import DiffStatus
    from evalview.core.celebrations import Celebrations
    from evalview.core.drift_tracker import DriftTracker
    from evalview.core.messages import get_random_clean_check_message
    from evalview.core.root_cause import analyze_root_cause
    from rich.panel import Panel

    # Build result lookup by test name
    result_by_name: Dict[str, Any] = {}
    if results:
        for r in results:
            result_by_name[r.test_case] = r

    # Pre-compute root causes (use AI-enriched versions when available)
    _ai_rc = ai_root_causes or {}
    root_cause_by_name: Dict[str, Any] = {}
    for name, diff in diffs:
        rc = _ai_rc.get(name) or analyze_root_cause(diff)
        root_cause_by_name[name] = rc

    if json_output:
        output = {
            "summary": {
                "total_tests": len(diffs),
                "unchanged": sum(1 for _, d in diffs if d.overall_severity == DiffStatus.PASSED),
                "regressions": sum(1 for _, d in diffs if d.overall_severity == DiffStatus.REGRESSION),
                "tools_changed": sum(1 for _, d in diffs if d.overall_severity == DiffStatus.TOOLS_CHANGED),
                "output_changed": sum(1 for _, d in diffs if d.overall_severity == DiffStatus.OUTPUT_CHANGED),
                "model_changed": any(getattr(d, "model_changed", False) for _, d in diffs),
                "effective_all_passed": bool(analysis.get("effective_all_passed", analysis["all_passed"])),
                "healing_all_resolved": bool(analysis.get("healing_all_resolved", False)),
                "has_unresolved_failures": bool(analysis.get("has_unresolved_failures", not analysis["all_passed"])),
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
                    "turn_diffs": [
                        {
                            "turn": td.turn_index,
                            "baseline_tools": td.baseline_tools,
                            "current_tools": td.current_tools,
                            "status": td.status.value,
                        }
                        for td in (diff.turn_diffs or [])
                    ] or None,
                    "root_cause": (
                        root_cause_by_name[name].to_dict()
                        if root_cause_by_name.get(name) is not None
                        else None
                    ),
                }
                for name, diff in diffs
            ],
        }
        if healing_summary:
            output["healing"] = {
                "total_healed": healing_summary.total_healed,
                "total_proposed": healing_summary.total_proposed,
                "total_review": healing_summary.total_review,
                "total_blocked": healing_summary.total_blocked,
                "attempted_count": healing_summary.attempted_count,
                "unresolved_count": healing_summary.unresolved_count,
                "failed_count": healing_summary.failed_count,
                "policy_version": healing_summary.policy_version,
                "thresholds": healing_summary.thresholds,
                "audit_path": healing_summary.audit_path,
                "results": [r.model_dump() for r in healing_summary.results],
            }
        print(json.dumps(output, indent=2))
    else:
        # Console output with personality
        from evalview.core.dashboard import (
            render_scorecard,
            render_sparklines,
            render_confidence_label,
            render_smart_accept_suggestion,
        )

        # --- Dashboard Scorecard ---
        passed_count = sum(1 for _, d in diffs if d.overall_severity == DiffStatus.PASSED)
        tools_changed_count = sum(1 for _, d in diffs if d.overall_severity == DiffStatus.TOOLS_CHANGED)
        output_changed_count = sum(1 for _, d in diffs if d.overall_severity == DiffStatus.OUTPUT_CHANGED)
        regression_count = sum(1 for _, d in diffs if d.overall_severity == DiffStatus.REGRESSION)
        exec_failures = int(analysis.get("execution_failures", 0) or 0)
        total_compared = len(diffs)
        health_pct = (passed_count / total_compared * 100) if total_compared > 0 else 0.0

        if diffs:
            scorecard = render_scorecard(
                passed=passed_count,
                tools_changed=tools_changed_count,
                output_changed=output_changed_count,
                regressions=regression_count,
                execution_failures=exec_failures,
                current_streak=int(state.current_streak or 0),
                longest_streak=int(state.longest_streak or 0),
                health_pct=health_pct,
            )
            console.print(scorecard)
            console.print()

        # --- Sparkline Trends ---
        if diffs and drift_tracker is not None:
            test_trends: Dict[str, List[float]] = {}
            for _name, _diff in diffs:
                history = drift_tracker.get_test_history(_name, limit=10)
                # get_test_history returns newest-first; reverse for sparklines
                similarities = [h["output_similarity"] for h in reversed(history)]
                if similarities:
                    test_trends[_name] = similarities

            pass_trend = drift_tracker.get_pass_rate_trend(window=10)
            sparkline_output = render_sparklines(test_trends, pass_trend)
            if sparkline_output:
                console.print(sparkline_output)
                console.print()

        if is_first_check:
            Celebrations.first_check()

        # Model version change warning
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

        # Gradual drift warnings
        _drift = drift_tracker if drift_tracker is not None else DriftTracker()
        for name, _ in diffs:
            warning = _drift.detect_gradual_drift(name)
            if warning:
                console.print(f"[yellow]📉 {name}:[/yellow] {warning}\n")

        effective_all_passed = bool(analysis.get("effective_all_passed", analysis["all_passed"]))

        if analysis["all_passed"]:
            # Show what was verified so users can trust the result
            if diffs:
                _print_passed_summary(diffs, result_by_name, golden_traces)
                console.print(f"[green]{get_random_clean_check_message()}[/green]\n")

                if state.current_streak >= 3:
                    Celebrations.clean_check_streak(state)

                if state.total_checks >= 5 and state.total_checks % 5 == 0:
                    Celebrations.health_summary(state)
            else:
                from rich.panel import Panel as _Panel

                console.print(
                    _Panel(
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
        else:
            # Execution failures (tests that didn't produce diffs)
            if exec_failures > 0:
                console.print(
                    f"  {exec_failures} execution "
                    f"{'failure' if exec_failures == 1 else 'failures'}"
                )
                console.print()

            # Build healing lookup
            heal_by_name: Dict[str, Any] = {}
            if healing_summary:
                from evalview.core.healing import HealingAction
                heal_by_name = {r.test_name: r for r in healing_summary.results}

            # Model update banner (before per-test display)
            if healing_summary and healing_summary.model_update:
                mu = healing_summary.model_update
                console.print(
                    f"  [yellow]\u26a0 Model update detected:[/yellow] "
                    f"[dim]{mu.golden_model}[/dim] \u2192 [bold]{mu.actual_model}[/bold] "
                    f"({mu.affected_count} test{'s' if mu.affected_count != 1 else ''} affected)\n"
                )

            _goldens = golden_traces or {}
            for name, diff in diffs:
                if diff.overall_severity == DiffStatus.PASSED:
                    continue

                # Check if healing resolved this test
                heal_result = heal_by_name.get(name)
                if heal_result and heal_result.healed:
                    console.print(
                        f"  [green]\u26a1 HEALED[/green]: {name}  "
                        f"[dim]{heal_result.diagnosis.reason}[/dim]"
                    )
                    continue
                elif heal_result and heal_result.proposed:
                    console.print(
                        f"  [cyan]\u25c8 PROPOSED[/cyan]: {name}  "
                        f"[dim]{heal_result.diagnosis.reason}[/dim]"
                    )
                    continue
                elif heal_result and heal_result.diagnosis.action == HealingAction.BLOCKED:
                    console.print(
                        f"  [red]\u2717 BLOCKED[/red]: {name}  "
                        f"[dim]{heal_result.diagnosis.reason}[/dim]"
                    )
                    continue
                elif heal_result:
                    # FLAG_REVIEW — show with review icon then fall through to details
                    console.print(
                        f"  [yellow]\u26a0 REVIEW[/yellow]: {name}  "
                        f"[dim]{heal_result.diagnosis.reason}[/dim]"
                    )
                    # Fall through to show details below

                if not heal_result:
                    # Normal (non-healing) display
                    severity_icon = {
                        DiffStatus.REGRESSION: "[red]\u2717 REGRESSION[/red]",
                        DiffStatus.TOOLS_CHANGED: "[yellow]\u26a0 TOOLS_CHANGED[/yellow]",
                        DiffStatus.OUTPUT_CHANGED: "[dim]~ OUTPUT_CHANGED[/dim]",
                    }.get(diff.overall_severity, "?")

                    score_part = ""
                    if abs(diff.score_diff) > 1:
                        sign = "+" if diff.score_diff > 0 else ""
                        score_color = "green" if diff.score_diff > 0 else "red"
                        score_part = f"  [{score_color}]{sign}{diff.score_diff:.1f} pts[/{score_color}]"

                    # Confidence label from historical variance
                    confidence_str = ""
                    if drift_tracker is not None:
                        output_sim = diff.output_diff.similarity if diff.output_diff else 1.0
                        conf_result = drift_tracker.compute_confidence(name, output_sim)
                        if conf_result is not None:
                            conf_pct, conf_label = conf_result
                            confidence_str = "  " + render_confidence_label(conf_pct, conf_label)

                    console.print(f"{severity_icon}: {name}{score_part}{confidence_str}")

                meta = (test_metadata or {}).get(name, {})
                if meta.get("is_multi_turn"):
                    behavior_class = str(meta.get("behavior_class") or "multi_turn").replace("_", " ")
                    console.print(f"    [dim]Multi-turn path:[/dim] {behavior_class}")

                golden_for_test = _goldens.get(name)
                result_for_test = result_by_name.get(name)
                _print_inline_trajectory(diff, golden_for_test, result_for_test)

                if diff.tool_diffs:
                    _print_parameter_diffs(diff.tool_diffs)

                # Per-turn breakdown for multi-turn tests
                if diff.turn_diffs:
                    changed_turns = [td for td in diff.turn_diffs if td.status != DiffStatus.PASSED]
                    if changed_turns:
                        console.print("    [dim]Per-turn breakdown:[/dim]")
                        for td in changed_turns:
                            baseline_str = ", ".join(td.baseline_tools) or "(none)"
                            current_str = ", ".join(td.current_tools) or "(none)"
                            parts = []
                            if td.baseline_tools != td.current_tools:
                                parts.append(f"[red]{baseline_str}[/red] \u2192 [green]{current_str}[/green]")
                            else:
                                parts.append(f"tools OK")
                            if td.output_similarity is not None:
                                sim_pct = int(td.output_similarity * 100)
                                sim_color = "green" if sim_pct >= 80 else "yellow" if sim_pct >= 50 else "red"
                                parts.append(f"output [{sim_color}]{sim_pct}% similar[/{sim_color}]")
                            console.print(
                                f"      [yellow]Turn {td.turn_index}:[/yellow] "
                                + ", ".join(parts)
                            )

                # Per-turn evaluation failures
                if result_for_test and getattr(result_for_test, "turn_evaluations", None):
                    failed_evals = [te for te in result_for_test.turn_evaluations if not te.passed]
                    if failed_evals:
                        console.print("    [dim]Per-turn evaluation:[/dim]")
                        for te in failed_evals:
                            console.print(
                                f"      [red]Turn {te.turn_index}: FAIL[/red] \u2014 {te.details}"
                            )

                _print_output_diff(diff)

                root_cause = root_cause_by_name.get(name)
                if root_cause is not None:
                    _print_root_cause(root_cause)

                quoted = f'"{name}"' if " " in name else name
                console.print(f"    [dim]\u2192 evalview replay {quoted}[/dim]")

                # Smart accept suggestion
                if result_for_test is not None:
                    golden_for_accept = _goldens.get(name)
                    baseline_score = (
                        result_for_test.score - diff.score_diff
                        if diff.score_diff is not None
                        else 0.0
                    )
                    baseline_tools = (
                        golden_for_accept.tool_sequence if golden_for_accept else []
                    )
                    current_tools = [
                        str(getattr(s, "tool_name", None) or getattr(s, "step_name", "?"))
                        for s in (result_for_test.trace.steps or [])
                    ]
                    accept_panel = render_smart_accept_suggestion(
                        test_name=name,
                        score_improved=(diff.score_diff > 0) if diff.score_diff is not None else False,
                        tools_changed=bool(diff.tool_diffs),
                        baseline_tools=baseline_tools,
                        current_tools=current_tools,
                        baseline_score=baseline_score,
                        current_score=result_for_test.score,
                    )
                    if accept_panel:
                        console.print(accept_panel)

                console.print()

            # Healing summary footer
            if healing_summary and healing_summary.model_update:
                mu = healing_summary.model_update
                if mu.healed_count == mu.affected_count:
                    console.print(
                        f"  [green]Model update:[/green] all {mu.affected_count} affected tests healed via retry. "
                        f"Run [bold]evalview snapshot[/bold] to rebase."
                    )
                elif mu.healed_count > 0:
                    console.print(
                        f"  [yellow]Model update:[/yellow] {mu.healed_count} of {mu.affected_count} affected tests healed. "
                        f"{mu.failed_count} still failing \u2014 review before rebasing."
                    )
                else:
                    console.print(
                        f"  [red]Model update:[/red] broke {mu.failed_count} tests. Review before rebasing."
                    )

            if healing_summary and healing_summary.results:
                parts = []
                if healing_summary.total_healed:
                    parts.append(f"{healing_summary.total_healed} resolved")
                if healing_summary.total_proposed:
                    parts.append(
                        f"{healing_summary.total_proposed} candidate "
                        f"variant{'s' if healing_summary.total_proposed != 1 else ''} saved"
                    )
                if healing_summary.total_review:
                    parts.append(
                        f"{healing_summary.total_review} "
                        f"need{'s' if healing_summary.total_review == 1 else ''} review"
                    )
                if healing_summary.total_blocked:
                    parts.append(f"{healing_summary.total_blocked} blocked")
                console.print(f"\n  {', '.join(parts)}.")
                if effective_all_passed:
                    console.print("  [green]All detected failures were resolved by bounded retry healing.[/green]")
                if healing_summary.audit_path:
                    console.print(f"  [dim]Audit log: {healing_summary.audit_path}[/dim]")
                console.print()

            if analysis.get("has_unresolved_failures", analysis["has_regressions"]):
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
