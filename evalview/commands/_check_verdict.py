"""Verdict-payload computation and console rendering for `evalview check`.

These helpers turn raw diff/result/drift data into the ship/don't-ship
decision (the `_VerdictOutput`) consumed by --json output, PR comments,
and the console panel. Pure logic except _render_verdict_panel which
writes to the shared Rich console.

Extracted from check_cmd.py so the command body stays focused on
orchestration. Tests import _compute_verdict_payload from check_cmd
(re-exported there for backward compat).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

from evalview.commands.shared import console, _parse_fail_statuses

if TYPE_CHECKING:
    from evalview.core.diff import TraceDiff


def _compute_check_exit_code(
    diffs: List[Tuple[str, "TraceDiff"]],
    fail_on: Optional[str],
    strict: bool,
    execution_failures: int = 0,
    quarantine: Optional[Any] = None,
) -> int:
    """Compute exit code based on diff results and fail conditions.

    Quarantined tests are excluded from the exit code unless --strict is set.

    Args:
        quarantine: Optional pre-loaded `QuarantineStore`. Callers that also
            compute the verdict should share a single instance to avoid a
            double YAML read and the race window that comes with it.

    Returns:
        0 if no failures match fail conditions, 1 otherwise
    """
    from evalview.core.quarantine import QuarantineStore

    if strict:
        fail_on = "REGRESSION,TOOLS_CHANGED,OUTPUT_CHANGED"

    if not fail_on:
        fail_on = "REGRESSION"  # Default

    fail_statuses = _parse_fail_statuses(fail_on)

    if execution_failures > 0:
        return 1

    if quarantine is None:
        quarantine = QuarantineStore()

    for name, diff in diffs:
        if diff.overall_severity in fail_statuses:
            # Quarantined tests don't block CI (unless --strict)
            if not strict and quarantine.is_quarantined(name):
                continue
            return 1

    return 0


@dataclass
class _VerdictOutput:
    """Internal container returned by _compute_verdict_payload.

    The `.payload` dict is what goes into --json output and PR comments.
    The `.verdict` / `.reasons` / `.top_recs` fields are kept separate so
    the console renderer doesn't have to dig through the serialized dict
    (which would otherwise force an ugly `_raw_*` key convention).
    """
    payload: Dict[str, Any] = field(default_factory=dict)
    verdict: Any = None
    reasons: List[str] = field(default_factory=list)
    top_recs: List[Any] = field(default_factory=list)


def _substitute_test_name(commands: List[str], test_name: str) -> List[str]:
    """Replace `<test>` placeholders with the real test name.

    Users shouldn't have to hunt for the test name — if we know it at
    rec-time, paste it in. Supports both `<test>` and `<test_name>`.
    """
    out: List[str] = []
    for cmd in commands:
        s = cmd.replace("<test_name>", test_name).replace("<test>", test_name)
        out.append(s)
    return out


def _aggregate_cost_delta_ratio(
    diffs: List[Tuple[str, "TraceDiff"]],
    results: List[Any],
    golden_traces: Optional[Dict[str, Any]],
) -> Optional[float]:
    """Compute (current_total_cost - golden_total_cost) / golden_total_cost.

    Returns None when either side is zero/missing so a missing baseline
    never trips the cost-spike verdict rule.

    Aggregation is across only the tests that were actually compared —
    tests without a matching golden are excluded from both sides so the
    ratio is apples-to-apples.
    """
    if not results or not golden_traces or not diffs:
        return None

    compared_names = {name for name, _ in diffs}
    current_by_name: Dict[str, float] = {}
    for r in results:
        name = getattr(r, "test_case", None)
        if name is None:
            continue
        if name not in compared_names:
            continue
        cost = 0.0
        try:
            cost = float(getattr(r.trace.metrics, "total_cost", 0.0) or 0.0)
        except Exception:
            cost = 0.0
        current_by_name[name] = cost

    baseline_total = 0.0
    current_total = 0.0
    for name in compared_names:
        if name not in golden_traces or name not in current_by_name:
            continue
        try:
            g_cost = float(
                getattr(golden_traces[name].trace.metrics, "total_cost", 0.0) or 0.0
            )
        except Exception:
            g_cost = 0.0
        baseline_total += g_cost
        current_total += current_by_name[name]

    if baseline_total <= 0 or current_total <= 0:
        return None
    return (current_total - baseline_total) / baseline_total


def _dedup_recommendations(recs: List[Any]) -> List[Any]:
    """Drop duplicate recommendations produced across multiple failing tests.

    When 10 tests all fail with the same root cause (model change, tool
    rename, etc.) the engine generates the same rec for each diff. The
    user only needs to see it once — keep the highest-confidence copy.
    """
    seen: Dict[Tuple[str, str], Any] = {}
    conf_rank = {"high": 0, "medium": 1, "low": 2}
    for rec in recs:
        key = (getattr(rec, "action", ""), getattr(rec, "category", ""))
        prev = seen.get(key)
        if prev is None:
            seen[key] = rec
            continue
        if conf_rank.get(getattr(rec, "confidence", "medium"), 1) < conf_rank.get(
            getattr(prev, "confidence", "medium"), 1
        ):
            seen[key] = rec
    return list(seen.values())


def _compute_verdict_payload(
    *,
    diffs: List[Tuple[str, "TraceDiff"]],
    results: List[Any],
    drift_tracker: Any,
    execution_failures: int,
    golden_traces: Optional[Dict[str, Any]] = None,
    quarantine: Optional[Any] = None,
) -> _VerdictOutput:
    """Pure: derive the release verdict + top recs from check outputs.

    Never raises. Returns a `_VerdictOutput` so the renderer can access
    the raw Verdict/reasons/recs without parsing the serialized payload.

    Args:
        quarantine: Optional pre-loaded `QuarantineStore`. Share one
            instance with `_compute_check_exit_code` per check so the
            two stay in sync even if the YAML file changes mid-run.
    """
    from evalview.core.verdict import (
        VerdictSignals,
        compute_verdict,
        verdict_to_dict,
    )
    from evalview.core.quarantine import QuarantineStore
    from evalview.core.recommendations import recommend_from_trace_diff

    try:
        if quarantine is None:
            quarantine = QuarantineStore()
        quarantined = [q.test_name for q in quarantine.list_all()]
        stale_quarantined = [q.test_name for q in quarantine.list_stale()]
    except Exception:
        quarantined = []
        stale_quarantined = []

    # Drift signal: use the graded classify_drift() tier per test.
    # Count how many tests land in each tier so the PR comment / verdict
    # reason can say "3 of 12 tests drifting, 1 with high confidence"
    # instead of the earlier boolean "something is drifting, maybe".
    drift_counts: Dict[str, int] = {
        "high": 0, "medium": 0, "low": 0, "stable": 0, "insufficient_history": 0,
    }
    drift_warnings = 0  # tests in low/medium/high
    drift_confidence: Optional[str] = None
    drift_is_downward = False
    if drift_tracker is not None:
        for name, _ in diffs:
            try:
                tier, _slope = drift_tracker.classify_drift(name)
            except Exception:
                tier = "insufficient_history"
            drift_counts[tier] = drift_counts.get(tier, 0) + 1
            if tier in ("low", "medium", "high"):
                drift_warnings += 1

        # Aggregate confidence = highest tier observed across tests.
        # A single high-confidence drifter is enough to escalate.
        if drift_counts["high"] > 0:
            drift_confidence = "high"
            drift_is_downward = True
        elif drift_counts["medium"] > 0:
            drift_confidence = "medium"
            drift_is_downward = True
        elif drift_counts["low"] > 0:
            drift_confidence = "low"
            drift_is_downward = True

    cost_delta_ratio = _aggregate_cost_delta_ratio(diffs, results, golden_traces)

    signals = VerdictSignals(
        test_statuses=[(name, d.overall_severity.value) for name, d in diffs],
        quarantined_tests=quarantined,
        stale_quarantined_tests=stale_quarantined,
        cost_delta_ratio=cost_delta_ratio,
        drift_confidence=drift_confidence,
        drift_is_downward=drift_is_downward,
        execution_failures=execution_failures,
    )

    verdict, reasons = compute_verdict(signals)

    # Augment the drift reason with a count if more than one test is affected.
    if drift_warnings > 1:
        reasons = [
            (
                f"{drift_warnings} tests showing downward drift "
                "(OLS slope exceeds threshold)"
            )
            if r.startswith("High-confidence downward drift")
            else r
            for r in reasons
        ]

    payload = verdict_to_dict(verdict, reasons)

    # Collect recommendations across failing diffs, substituting test names
    # into placeholder commands so the user can copy-paste with zero edits.
    all_recs: List[Any] = []
    for name, d in diffs:
        if d.overall_severity.value == "passed":
            continue
        try:
            recs = recommend_from_trace_diff(d)
        except Exception:
            recs = []
        for rec in recs:
            rec.suggested_commands = _substitute_test_name(
                getattr(rec, "suggested_commands", None) or [], name
            )
        all_recs.extend(recs)

    all_recs = _dedup_recommendations(all_recs)

    severity_rank = {"high": 0, "medium": 1, "low": 2}
    all_recs.sort(
        key=lambda r: (
            severity_rank.get(getattr(r, "severity", "medium"), 1),
            severity_rank.get(getattr(r, "confidence", "medium"), 1),
        )
    )

    # W3.4 — replay stability check.
    # When the verdict is INVESTIGATE, prepend a "rerun statistically"
    # recommendation so users have a cheap one-command way to decide
    # "flake or real?" before they escalate or ignore. This is the
    # single most operational output of the verdict layer.
    #
    # IMPORTANT: insert AFTER the sort, not before. Inserting before
    # the sort lets existing high-severity recs demote this one off
    # the top-3 — which defeats the whole point of the stability
    # check (it's the *first* thing the user should do).
    from evalview.core.recommendations import Recommendation as _Rec
    from evalview.core.verdict import Verdict as _Verdict
    if verdict == _Verdict.INVESTIGATE:
        stability_rec = _Rec(
            action="Rerun statistically to distinguish flake from real drift",
            confidence="high",
            category="config",
            detail=(
                "The verdict is INVESTIGATE — some signals moved but it's "
                "not clear whether the change is a genuine regression or LLM "
                "variance. Rerunning 5x with --statistical gives a confidence "
                "interval that settles the question in under a minute."
            ),
            likely_cause="Uncertain — stability check separates flake from drift.",
            severity="high",  # tier-one concern when INVESTIGATE fires
            suggested_commands=["evalview check --statistical 5"],
        )
        all_recs.insert(0, stability_rec)

    top_recs = all_recs[:3]
    payload["recommendations"] = [r.to_dict() for r in top_recs]
    if cost_delta_ratio is not None:
        payload["cost_delta_ratio"] = round(cost_delta_ratio, 4)
    if drift_warnings > 0:
        payload["drift_affected_tests"] = drift_warnings
        payload["drift_breakdown"] = {
            k: v for k, v in drift_counts.items() if v > 0 and k != "stable"
        }
    # Cap the stale-tests list to keep the payload bounded — consumers
    # (PR comments, Slack, cloud) only ever show the first few names
    # anyway. `stale` still carries the true count.
    _STALE_PREVIEW_CAP = 10
    payload["quarantine"] = {
        "total": len(quarantined),
        "stale": len(stale_quarantined),
        "stale_tests": list(stale_quarantined[:_STALE_PREVIEW_CAP]),
    }

    return _VerdictOutput(
        payload=payload,
        verdict=verdict,
        reasons=reasons,
        top_recs=top_recs,
    )


def _render_verdict_panel(output: "_VerdictOutput") -> None:
    """Console-only rendering of the verdict output."""
    from evalview.core.verdict import render_verdict_panel

    if output.verdict is None:
        return

    render_verdict_panel(output.verdict, output.reasons, console)

    if not output.top_recs:
        return

    console.print("[bold]Likely cause & next actions:[/bold]\n")
    for i, rec in enumerate(output.top_recs, 1):
        conf = getattr(rec, "confidence", "medium")
        sev = getattr(rec, "severity", "medium")
        cause = getattr(rec, "likely_cause", "") or rec.detail
        console.print(
            f"  [bold]{i}.[/bold] {rec.action} "
            f"[dim]({sev} severity, {conf} confidence)[/dim]"
        )
        if cause:
            console.print(f"     [dim]{cause}[/dim]")
        commands = getattr(rec, "suggested_commands", None) or []
        for cmd in commands:
            console.print(f"     [cyan]→ {cmd}[/cyan]")
        console.print()
