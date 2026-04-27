"""Small support helpers for `evalview check`.

Contains:
- _all_failures_retry_healed: did the heal pass clear all failures?
- target/tag summarizers: filter, dedupe, and describe the test slice
- judge usage summary: structured cost data for reports
- baseline context formatting: date ranges + model IDs

Extracted from check_cmd.py so the command body stays focused on flow.
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

from evalview.commands.shared import console
from evalview.core.diff import DiffStatus

if TYPE_CHECKING:
    from evalview.core.diff import TraceDiff


def _all_failures_retry_healed(
    diffs: List[Tuple[str, "TraceDiff"]],
    healing_summary: Optional[Any],
    execution_failures: int = 0,
) -> bool:
    """Return True only when every failing diff was resolved by a retry heal."""
    if execution_failures > 0 or not healing_summary:
        return False

    failed_names = {
        name for name, diff in diffs if diff.overall_severity != DiffStatus.PASSED
    }
    if not failed_names:
        return False

    result_by_name = {result.test_name: result for result in healing_summary.results}
    if set(result_by_name) != failed_names:
        return False

    return all(
        result.healed and result.final_status == DiffStatus.PASSED.value
        for result in result_by_name.values()
    )


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


def _normalize_requested_tags(tags: tuple[str, ...]) -> list[str]:
    normalized: list[str] = []
    for tag in tags:
        value = str(tag).strip().lower()
        if value and value not in normalized:
            normalized.append(value)
    return normalized


def _filter_test_cases_by_tags(test_cases: List[Any], requested_tags: tuple[str, ...]) -> tuple[List[Any], list[str]]:
    active_tags = _normalize_requested_tags(requested_tags)
    if not active_tags:
        return test_cases, []
    filtered = [
        tc for tc in test_cases
        if set(getattr(tc, "tags", []) or []).intersection(active_tags)
    ]
    return filtered, active_tags


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
    import os
    if report_path or json_output or not results:
        return False
    if bool(os.environ.get("CI")):
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
