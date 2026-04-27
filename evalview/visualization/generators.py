"""EvalView visual report generator.

Produces a single self-contained HTML file from EvaluationResult objects and
TraceDiff data.  No external files — Mermaid.js and Chart.js are loaded from
CDN.  The generated file is suitable for:
    • Auto-open in browser after ``evalview check``
    • Attaching to Slack / PRs
    • Returning as a path from the MCP ``generate_visual_report`` tool
    • Sharing with ``--share`` (future)

Usage::
    from evalview.visualization import generate_visual_report
    path = generate_visual_report(results, diffs, output_path="report.html")
"""
from __future__ import annotations

import json
import os
import webbrowser
from collections import Counter
from datetime import datetime
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from evalview.core.types import EvaluationResult
    from evalview.core.diff import TraceDiff
    from evalview.core.observability import (
        AnomalyReportDict,
        CoherenceReportDict,
        TrustReportDict,
    )


# Submodule extractions: Mermaid helpers and the HTML/Jinja2 template.
from evalview.visualization._mermaid import (
    _mermaid_from_steps,
    _mermaid_trace,
    _strip_markdown,
)
from evalview.visualization._template import _TEMPLATE

# ── KPI helpers ────────────────────────────────────────────────────────────────

def _kpis(results: List["EvaluationResult"]) -> Dict[str, Any]:
    if not results:
        return {}
    total = len(results)
    passed = sum(1 for r in results if r.passed)
    scores = [r.score for r in results]
    costs = []
    latencies = []
    total_input_tokens = 0
    total_output_tokens = 0
    for r in results:
        try:
            costs.append(r.trace.metrics.total_cost or 0)
            latencies.append(r.trace.metrics.total_latency or 0)
            if r.trace.metrics.total_tokens:
                total_input_tokens += r.trace.metrics.total_tokens.input_tokens
                total_output_tokens += r.trace.metrics.total_tokens.output_tokens
        except AttributeError:
            pass
    models = _collect_models(results)
    total_tokens = total_input_tokens + total_output_tokens
    return {
        "total": total,
        "passed": passed,
        "failed": total - passed,
        "pass_rate": round(passed / total * 100, 1),
        "avg_score": round(sum(scores) / len(scores), 1),
        "total_cost": round(sum(costs), 6),
        "avg_latency_ms": round(sum(latencies) / len(latencies), 0) if latencies else 0,
        "scores": scores,
        "test_names": [r.test_case for r in results],
        "models": models,
        "models_display": ", ".join(models) if models else "Unknown",
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
        "total_tokens": total_tokens,
    }


def _clean_model_name(model_id: str, provider: Optional[str] = None) -> str:
    """Format a model name for display — human-readable, no internal prefixes."""
    # Skip transport-layer "providers" that aren't real LLM providers
    non_providers = {"http", "mcp", "unknown", "none", ""}
    if provider and provider.lower() not in non_providers:
        return f"{provider}/{model_id}"
    return model_id


def _extract_models(result: "EvaluationResult") -> List[str]:
    """Extract best-effort model labels from a result (deduplicated by model ID)."""
    seen_ids: set[str] = set()
    labels: list[str] = []
    trace = result.trace
    model_id = getattr(trace, "model_id", None)
    model_provider = getattr(trace, "model_provider", None)
    if model_id:
        seen_ids.add(model_id)
        labels.append(_clean_model_name(model_id, model_provider))

    # Only add span models if the trace didn't already report a model_id.
    # When model_id is set (from the agent response), span models are
    # typically just the config echo from the HTTP adapter — showing both
    # creates confusing duplicates like "anthropic/claude-sonnet-4-5, claude-sonnet-4-6".
    trace_context = getattr(trace, "trace_context", None)
    if trace_context and not model_id:
        for span in trace_context.spans:
            if span.llm and span.llm.model and span.llm.model not in seen_ids:
                seen_ids.add(span.llm.model)
                provider = span.llm.provider or model_provider
                labels.append(_clean_model_name(span.llm.model, provider))

    return labels


def _extract_check_result(result: "EvaluationResult", check_name: str) -> Optional[Dict[str, Any]]:
    """Extract a check result (hallucination, safety, pii, forbidden_tools) for the template."""
    evals = getattr(result, "evaluations", None)
    if not evals:
        return None
    check = getattr(evals, check_name, None)
    if check is None:
        return None
    data: Dict[str, Any] = {"passed": getattr(check, "passed", True)}
    if check_name == "hallucination":
        data["has_hallucination"] = getattr(check, "has_hallucination", False)
        data["confidence"] = getattr(check, "confidence", 0)
        data["details"] = getattr(check, "details", "")
    elif check_name == "safety":
        data["is_safe"] = getattr(check, "is_safe", True)
        data["categories"] = getattr(check, "categories_flagged", [])
        data["severity"] = getattr(check, "severity", "safe")
        data["details"] = getattr(check, "details", "")
    elif check_name == "pii":
        data["has_pii"] = getattr(check, "has_pii", False)
        data["types"] = getattr(check, "types_detected", [])
        data["details"] = getattr(check, "details", "")
    elif check_name == "forbidden_tools":
        data["violations"] = getattr(check, "violations", [])
    return data


def _normalize_anomaly_report(
    report: Optional["AnomalyReportDict"],
) -> Optional[Dict[str, Any]]:
    """Project an anomaly report down to the fields the Jinja template reads.

    EvaluationResult.anomaly_report is validated by Pydantic against
    AnomalyReportDict, so the shape is trusted. This adapter exists solely
    to decouple the template from schema additions — new fields on the
    report do not require template edits.
    """
    if report is None:
        return None
    return {
        "anomalies": [
            {
                "pattern": a["pattern"],
                "severity": a["severity"],
                "description": a["description"],
            }
            for a in report["anomalies"]
        ],
        "summary": report["summary"],
    }


def _normalize_trust_report(
    report: Optional["TrustReportDict"],
) -> Optional[Dict[str, Any]]:
    """Project a trust report down to what the Jinja template reads, and
    embed LOW_TRUST_THRESHOLD so the template doesn't hardcode it."""
    if report is None:
        return None
    from evalview.core.observability import LOW_TRUST_THRESHOLD
    return {
        "trust_score": report["trust_score"],
        "low_trust_threshold": LOW_TRUST_THRESHOLD,
        "flags": [
            {
                "check": f["check"],
                "severity": f["severity"],
                "description": f["description"],
            }
            for f in report["flags"]
        ],
        "summary": report["summary"],
    }


def _normalize_coherence_report(
    report: Optional["CoherenceReportDict"],
) -> Optional[Dict[str, Any]]:
    """Project a coherence report down to what the Jinja template reads."""
    if report is None:
        return None
    return {
        "issues": [
            {
                "category": i["category"],
                "severity": i["severity"],
                "turn_index": i["turn_index"],
                "description": i["description"],
            }
            for i in report["issues"]
        ],
        "coherence_score": report["coherence_score"],
        "summary": report["summary"],
    }


def _collect_models(results: List["EvaluationResult"]) -> List[str]:
    """Collect model labels across a run, ordered by frequency."""
    counts: Counter[str] = Counter()
    for result in results:
        for label in _extract_models(result):
            counts[label] += 1
    return [label for label, _ in counts.most_common()]


def _baseline_meta(golden_traces: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Summarize baseline creation metadata."""
    if not golden_traces:
        return {
            "latest_created_display": "Unknown",
            "models_display": "Unknown",
        }

    blessed_times: list[datetime] = []
    model_counts: Counter[str] = Counter()
    for golden in golden_traces.values():
        metadata = getattr(golden, "metadata", None)
        if not metadata:
            continue
        blessed_at = getattr(metadata, "blessed_at", None)
        if isinstance(blessed_at, datetime):
            blessed_times.append(blessed_at)
        model_id = getattr(metadata, "model_id", None)
        model_provider = getattr(metadata, "model_provider", None)
        if model_id:
            model_counts[f"{model_provider}/{model_id}" if model_provider else str(model_id)] += 1

    latest_created = max(blessed_times).strftime("%Y-%m-%d %H:%M") if blessed_times else "Unknown"
    models = [label for label, _ in model_counts.most_common()]
    return {
        "latest_created_display": latest_created,
        "models_display": ", ".join(models) if models else "Not recorded in snapshot",
    }


# ── Diff helpers ───────────────────────────────────────────────────────────────

def _diff_rows(
    diffs: List["TraceDiff"],
    golden_traces: Optional[Dict[str, Any]] = None,
    actual_results: Optional[Dict[str, Any]] = None,
    test_metadata: Optional[Dict[str, Dict[str, Any]]] = None,
    root_causes: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    rows = []
    from evalview.core.root_cause import analyze_root_cause

    for d in diffs:
        status = str(getattr(d, "overall_severity", "passed")).lower().replace("diffstatus.", "")
        output_diff = getattr(d, "output_diff", None)
        similarity = round(getattr(output_diff, "similarity", 1.0) * 100, 1) if output_diff else 100.0
        semantic_similarity = None
        if output_diff and getattr(output_diff, "semantic_similarity", None) is not None:
            semantic_similarity = round(output_diff.semantic_similarity * 100, 1)
        golden_out = getattr(output_diff, "golden_preview", "") if output_diff else ""
        actual_out = getattr(output_diff, "actual_preview", "") if output_diff else ""
        diff_lines = getattr(output_diff, "diff_lines", []) if output_diff else []
        score_delta = getattr(d, "score_diff", 0.0) or 0.0
        test_name = getattr(d, "test_name", "")

        # Extract tool sequences from golden trace and tool_diffs
        golden_tools: List[str] = []
        actual_tools: List[str] = []
        if golden_traces and test_name in golden_traces:
            gt = golden_traces[test_name]
            golden_tools = getattr(gt, "tool_sequence", []) or []
        # Reconstruct actual tools from golden + diffs
        tool_diffs = getattr(d, "tool_diffs", []) or []
        if actual_results and test_name in actual_results:
            try:
                result = actual_results[test_name]
                actual_tools = [
                    str(getattr(s, "tool_name", None) or getattr(s, "step_name", "?"))
                    for s in (result.trace.steps or [])
                ]
            except AttributeError:
                pass

        # Extract parameter diffs for the HTML template
        param_diffs = []
        for td in tool_diffs:
            for pd in getattr(td, "parameter_diffs", []):
                sim = None
                if pd.similarity is not None:
                    sim = round(pd.similarity * 100, 1)
                param_diffs.append({
                    "step": td.position + 1,
                    "tool": td.golden_tool or td.actual_tool or "?",
                    "param": pd.param_name,
                    "golden": str(pd.golden_value)[:60] if pd.golden_value is not None else "",
                    "actual": str(pd.actual_value)[:60] if pd.actual_value is not None else "",
                    "type": pd.diff_type,
                    "similarity": sim,
                })

        # Generate side-by-side trajectory diagrams when trace data is available
        golden_diagram = ""
        actual_diagram = ""
        if golden_traces and test_name in golden_traces:
            gt = golden_traces[test_name]
            try:
                gt_steps = gt.trace.steps or []
            except AttributeError:
                gt_steps = []
            golden_diagram = _mermaid_from_steps(gt_steps)
        if actual_results and test_name in actual_results:
            actual_diagram = _mermaid_trace(actual_results[test_name])

        actual_score = None
        if actual_results and test_name in actual_results:
            actual_score = round(getattr(actual_results[test_name], "score", 0), 1)
        baseline_score = round(actual_score - score_delta, 1) if actual_score is not None else None
        # Use pre-computed (AI/narrative enriched) root cause when available;
        # fall back to deterministic analysis so the report always has context.
        if root_causes and test_name in root_causes:
            root_cause = root_causes[test_name]
        else:
            root_cause = analyze_root_cause(d)
        meta = (test_metadata or {}).get(test_name, {})
        tags = list(meta.get("tags") or [])

        # Confidence scoring from drift history
        confidence_pct = None
        confidence_label = None
        try:
            from evalview.core.drift_tracker import DriftTracker
            _dt = DriftTracker()
            output_sim = getattr(output_diff, "similarity", 1.0) if output_diff else 1.0
            conf = _dt.compute_confidence(test_name, output_sim)
            if conf is not None:
                confidence_pct = round(conf[0], 0)
                confidence_label = conf[1]
        except Exception:
            pass

        # Smart accept: suggest accepting if score improved or stayed stable
        accept_suggestion = None
        if status != "passed" and actual_score is not None and baseline_score is not None:
            s_diff = actual_score - baseline_score
            if s_diff >= -2.0:  # Score didn't drop significantly
                quoted = f'"{test_name}"' if " " in test_name else test_name
                accept_suggestion = {
                    "score_improved": s_diff > 0,
                    "command": f"evalview snapshot --test {quoted}",
                    "preview_command": f"evalview snapshot --test {quoted} --preview",
                }

        rows.append({
            "name": test_name,
            "status": status,
            "score_delta": round(score_delta, 1),
            "actual_score": actual_score,
            "baseline_score": baseline_score,
            "similarity": similarity,
            "semantic_similarity": semantic_similarity,
            "golden_tools": golden_tools,
            "actual_tools": actual_tools,
            "golden_out": golden_out[:600],
            "actual_out": actual_out[:600],
            "diff_lines": diff_lines[:50],
            "param_diffs": param_diffs,
            "golden_diagram": golden_diagram,
            "actual_diagram": actual_diagram,
            "confidence_pct": confidence_pct,
            "confidence_label": confidence_label,
            "accept_suggestion": accept_suggestion,
            "model_changed": bool(getattr(d, "model_changed", False)),
            "runtime_fingerprint_changed": bool(getattr(d, "runtime_fingerprint_changed", False)),
            "golden_runtime_fingerprint": getattr(d, "golden_runtime_fingerprint", None),
            "actual_runtime_fingerprint": getattr(d, "actual_runtime_fingerprint", None),
            "tags": tags,
            "root_cause_summary": getattr(root_cause, "summary", ""),
            "root_cause_category": getattr(getattr(root_cause, "category", None), "value", ""),
            "root_cause_fix": getattr(root_cause, "suggested_fix", None),
            "root_cause_ai": getattr(root_cause, "ai_explanation", None),
            "root_cause_narrative": getattr(root_cause, "narrative_root_cause", None),
            "recommendations": [r.to_dict() for r in _get_recommendations(d)],
        })
    return rows


def _get_recommendations(diff):
    """Generate recommendations for a TraceDiff (lazy import)."""
    try:
        from evalview.core.recommendations import recommend_from_trace_diff
        return recommend_from_trace_diff(diff)
    except Exception:
        return []


def _behavior_summary(
    results: List["EvaluationResult"],
    diff_rows: List[Dict[str, Any]],
    test_metadata: Optional[Dict[str, Dict[str, Any]]] = None,
    healing_summary: Optional[Any] = None,
) -> List[Dict[str, Any]]:
    rows_by_name = {row["name"]: row for row in diff_rows}
    heal_by_name: Dict[str, Any] = {}
    if healing_summary is not None:
        heal_by_name = {r.test_name: r for r in healing_summary.results}

    summary: Dict[str, Dict[str, int]] = {}

    def _bucket(tag: str) -> Dict[str, int]:
        if tag not in summary:
            summary[tag] = {
                "total": 0,
                "passed": 0,
                "changed": 0,
                "regressions": 0,
                "healed": 0,
                "review": 0,
                "blocked": 0,
            }
        return summary[tag]

    for result in results:
        name = result.test_case
        meta = (test_metadata or {}).get(name, {})
        tags = list(meta.get("tags") or []) or ["untagged"]
        diff_row = rows_by_name.get(name)
        heal_result = heal_by_name.get(name)
        status = diff_row["status"] if diff_row else ("passed" if result.passed else "regression")

        for tag in tags:
            row = _bucket(tag)
            row["total"] += 1
            if status == "passed":
                row["passed"] += 1
            elif status == "regression":
                row["regressions"] += 1
            else:
                row["changed"] += 1

            if heal_result:
                action = getattr(getattr(heal_result, "diagnosis", None), "action", None)
                action_value = getattr(action, "value", action)
                if heal_result.healed:
                    row["healed"] += 1
                elif action_value == "flag_review":
                    row["review"] += 1
                elif action_value == "blocked":
                    row["blocked"] += 1

    return [{"tag": tag, **counts} for tag, counts in sorted(summary.items(), key=lambda item: item[0])]


# ── Timeline helpers ───────────────────────────────────────────────────────────

def _compute_adapter_compare(results: List["EvaluationResult"]) -> Dict[str, Any]:
    """Auto-detect multi-adapter runs and build cross-model comparison data.

    Returns a dict with ``enabled=True`` only when ≥2 distinct adapter names
    share at least one task name, so the Compare tab only appears when useful.
    """
    from collections import defaultdict

    by_task: Dict[str, Dict[str, Dict]] = defaultdict(dict)
    for r in results:
        adapter = r.adapter_name or "unknown"
        task = r.test_case
        cost = 0.0
        latency = 0.0
        try:
            cost = r.trace.metrics.total_cost or 0.0
            latency = r.trace.metrics.total_latency or 0.0
        except AttributeError:
            pass
        by_task[task][adapter] = {
            "score": round(r.score, 1),
            "passed": r.passed,
            "latency_ms": latency,
            "tool_accuracy": round(r.evaluations.tool_accuracy.accuracy * 100, 1),
            "cost": cost,
        }

    all_adapters: List[str] = list(dict.fromkeys(r.adapter_name or "unknown" for r in results))
    all_tasks: List[str] = list(dict.fromkeys(r.test_case for r in results))
    tasks_shared = [t for t in all_tasks if len(by_task[t]) >= 2]
    enabled = len(all_adapters) >= 2 and len(tasks_shared) >= 1

    # Pre-compute per-adapter totals
    totals: Dict[str, Dict[str, Any]] = {}
    for adapter in all_adapters:
        cells = [by_task[t][adapter] for t in all_tasks if adapter in by_task[t]]
        if cells:
            avg_score = round(sum(c["score"] for c in cells) / len(cells), 1)
            avg_lat_s = round(sum(c["latency_ms"] for c in cells) / len(cells) / 1000, 1)
            total_cost = sum(c["cost"] for c in cells)
            passes = sum(1 for c in cells if c["passed"])
            cost_display = "free" if total_cost == 0.0 else f"${total_cost:.3f}"
        else:
            avg_score, avg_lat_s, cost_display, passes = "—", "—", "—", 0
        totals[adapter] = {
            "avg_score": avg_score,
            "avg_lat_s": avg_lat_s,
            "cost_display": cost_display,
            "pass_rate": f"{passes}/{len(cells)}" if cells else "—",
        }

    return {
        "enabled": enabled,
        "adapters": all_adapters,
        "tasks": all_tasks,
        "rows": {task: by_task[task] for task in all_tasks},
        "totals": totals,
    }


def _timeline_data(results: List["EvaluationResult"]) -> List[Dict[str, Any]]:
    rows = []
    for r in results:
        try:
            steps = r.trace.steps or []
            fallback_latency = 0.0
            fallback_cost = 0.0
            if steps:
                total_latency = float(getattr(r.trace.metrics, "total_latency", 0) or 0)
                total_cost = float(getattr(r.trace.metrics, "total_cost", 0) or 0)
                if not any((getattr(getattr(step, "metrics", None), "latency", 0) or 0) > 0 for step in steps):
                    fallback_latency = total_latency / len(steps) if total_latency > 0 else 0.0
                if not any((getattr(getattr(step, "metrics", None), "cost", 0) or 0) > 0 for step in steps):
                    fallback_cost = total_cost / len(steps) if total_cost > 0 else 0.0
            for step in steps:
                lat = getattr(step.metrics, "latency", 0) if hasattr(step, "metrics") else 0
                cost = getattr(step.metrics, "cost", 0) if hasattr(step, "metrics") else 0
                if (not lat or lat <= 0) and fallback_latency:
                    lat = fallback_latency
                if (not cost or cost <= 0) and fallback_cost:
                    cost = fallback_cost
                tool = getattr(step, "tool_name", "unknown")[:20]
                test = r.test_case[:15]
                rows.append({
                    "test": test,
                    "tool": tool,
                    "label": f"{test} \u203a {tool}",
                    "latency": round(lat, 1),
                    "cost": round(cost, 6),
                    "success": getattr(step, "success", True),
                })
        except AttributeError:
            pass
    return rows


# ── Main entry point ───────────────────────────────────────────────────────────

def generate_visual_report(
    results: List["EvaluationResult"],
    diffs: Optional[List["TraceDiff"]] = None,
    output_path: Optional[str] = None,
    auto_open: bool = True,
    title: str = "EvalView Report",
    notes: Optional[str] = None,
    compare_results: Optional[List[List["EvaluationResult"]]] = None,
    compare_labels: Optional[List[str]] = None,
    golden_traces: Optional[Dict[str, Any]] = None,
    judge_usage: Optional[Dict[str, Any]] = None,
    default_tab: Optional[str] = None,
    healing_summary: Optional[Any] = None,
    model_runtime_summary: Optional[Any] = None,
    effective_all_passed: Optional[bool] = None,
    test_metadata: Optional[Dict[str, Dict[str, Any]]] = None,
    active_tags: Optional[List[str]] = None,
    root_causes: Optional[Dict[str, Any]] = None,
) -> str:
    """Generate a self-contained visual HTML report.

    Args:
        results: List of EvaluationResult objects.
        diffs: Optional list of TraceDiff objects for diff tab.
        output_path: Where to write the HTML (default: .evalview/reports/<timestamp>.html).
        auto_open: If True, open the report in the default browser.
        title: Report title shown in the header.
        notes: Optional free-text note shown in the header.
        golden_traces: Optional dict mapping test name to GoldenTrace. When provided,
            the Diffs tab renders side-by-side baseline vs. current Mermaid diagrams.

    Returns:
        Absolute path to the generated HTML file.
    """
    if output_path is None:
        os.makedirs(".evalview/reports", exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f".evalview/reports/{ts}.html"

    kpis = _kpis(results)
    baseline = _baseline_meta(golden_traces)
    traces = []
    for r in results:
        try:
            cost = r.trace.metrics.total_cost or 0.0
            latency = r.trace.metrics.total_latency or 0.0
            tokens = None
            input_tokens = 0
            output_tokens = 0
            if r.trace.metrics.total_tokens:
                input_tokens = r.trace.metrics.total_tokens.input_tokens
                output_tokens = r.trace.metrics.total_tokens.output_tokens
                tokens = input_tokens + output_tokens
        except AttributeError:
            cost, latency, tokens = 0.0, 0.0, None
            input_tokens, output_tokens = 0, 0
        has_steps = bool(getattr(r.trace, "steps", None))
        models = _extract_models(r)
        baseline_created = ""
        baseline_model = "Unknown"
        if golden_traces and r.test_case in golden_traces:
            metadata = getattr(golden_traces[r.test_case], "metadata", None)
            if metadata:
                blessed_at = getattr(metadata, "blessed_at", None)
                if isinstance(blessed_at, datetime):
                    baseline_created = blessed_at.strftime("%Y-%m-%d %H:%M")
                model_id = getattr(metadata, "model_id", None)
                model_provider = getattr(metadata, "model_provider", None)
                if model_id:
                    baseline_model = f"{model_provider}/{model_id}" if model_provider else str(model_id)
                else:
                    trace_model_id = getattr(getattr(golden_traces[r.test_case], "trace", None), "model_id", None)
                    trace_model_provider = getattr(getattr(golden_traces[r.test_case], "trace", None), "model_provider", None)
                    if trace_model_id:
                        baseline_model = f"{trace_model_provider}/{trace_model_id}" if trace_model_provider else str(trace_model_id)
                    else:
                        baseline_model = "Not recorded in snapshot"

        # Extract turn and tool info for the trace list view
        turn_list = []
        if getattr(r.trace, "turns", None):
            for turn in getattr(r.trace, "turns", []) or []:
                turn_entry = {
                    "index": int(getattr(turn, "index", 0) or 0),
                    "query": str(getattr(turn, "query", "") or ""),
                    "output": _strip_markdown(str(getattr(turn, "output", "") or "")),
                    "tools": [str(tool) for tool in (getattr(turn, "tools", None) or [])],
                    "latency_ms": float(getattr(turn, "latency_ms", 0) or 0),
                    "cost": float(getattr(turn, "cost", 0) or 0),
                }
                # Attach per-turn evaluation if present
                eval_obj = getattr(turn, "evaluation", None)
                if eval_obj is not None:
                    turn_entry["evaluation"] = {
                        "passed": eval_obj.passed,
                        "tool_accuracy": eval_obj.tool_accuracy,
                        "forbidden_violations": eval_obj.forbidden_violations,
                        "contains_passed": eval_obj.contains_passed,
                        "contains_failed": eval_obj.contains_failed,
                        "not_contains_passed": eval_obj.not_contains_passed,
                        "not_contains_failed": eval_obj.not_contains_failed,
                    }
                turn_list.append(turn_entry)
        elif has_steps:
            current_t_idx = None
            current_turn_data = None
            turn_fallback_latency = 0.0
            turn_fallback_cost = 0.0
            if not any(getattr(step, "turn_index", None) is not None for step in r.trace.steps):
                turn_fallback_latency = float(getattr(r.trace.metrics, "total_latency", 0) or 0)
                turn_fallback_cost = float(getattr(r.trace.metrics, "total_cost", 0) or 0)
            for step in r.trace.steps:
                t_idx = getattr(step, "turn_index", None)
                if t_idx is not None:
                    if t_idx != current_t_idx:
                        current_t_idx = t_idx
                        current_turn_data = {
                            "index": t_idx,
                            "query": getattr(step, "turn_query", ""),
                            "output": "",
                            "tools": [],
                            "latency_ms": 0.0,
                            "cost": 0.0,
                        }
                        turn_list.append(current_turn_data)

                    if current_turn_data is not None:
                        tool_name = str(getattr(step, "tool_name", None) or getattr(step, "step_name", None) or "unknown")
                        current_turn_data["tools"].append(tool_name)
                        step_latency = float(getattr(getattr(step, "metrics", None), "latency", 0) or 0)
                        step_cost = float(getattr(getattr(step, "metrics", None), "cost", 0) or 0)
                        current_turn_data["latency_ms"] += step_latency
                        current_turn_data["cost"] += step_cost

            if not turn_list and has_steps:
                turn_list.append({
                    "index": 1,
                    "query": getattr(r, "input_query", "") or "",
                    "output": _strip_markdown(getattr(r, "actual_output", "") or ""),
                    "tools": [
                        str(getattr(step, "tool_name", None) or getattr(step, "step_name", None) or "unknown")
                        for step in r.trace.steps
                    ],
                    "latency_ms": turn_fallback_latency,
                    "cost": turn_fallback_cost,
                })

        # Build failure reasons list for failed tests
        failure_reasons = []
        if not r.passed:
            if r.min_score and r.score < r.min_score:
                failure_reasons.append(f"Score {round(r.score, 1)} below minimum {round(r.min_score, 1)}")
            evals = r.evaluations
            if evals.output_quality.score < 50:
                failure_reasons.append(f"Output quality: {round(evals.output_quality.score, 1)}/100")
            if evals.hallucination and getattr(evals.hallucination, "has_hallucination", False):
                conf = getattr(evals.hallucination, "confidence", None)
                conf_str = f" ({round(conf * 100)}% confidence)" if conf else ""
                failure_reasons.append(f"Hallucination detected{conf_str}")
            if evals.safety and not getattr(evals.safety, "is_safe", True):
                failure_reasons.append("Safety violation")
            if evals.forbidden_tools and getattr(evals.forbidden_tools, "violations", []):
                failure_reasons.append(f"Forbidden tools used: {', '.join(evals.forbidden_tools.violations)}")
            if evals.tool_accuracy.accuracy < 0.5:
                failure_reasons.append(f"Tool accuracy: {round(evals.tool_accuracy.accuracy * 100, 1)}%")

        output_rationale = getattr(r.evaluations.output_quality, "rationale", "") or ""

        # Score breakdown: show how the final score was calculated
        evals = r.evaluations
        tool_acc = round(evals.tool_accuracy.accuracy * 100, 1) if evals.tool_accuracy else None
        output_qual = round(evals.output_quality.score, 1) if evals.output_quality else None
        seq_obj = getattr(evals, "sequence_correctness", None)
        seq_correct = getattr(seq_obj, "correct", None) if seq_obj else None
        weights = getattr(r, "weights", None) or {}
        w_tool = weights.get("tool_accuracy", 0.3)
        w_output = weights.get("output_quality", 0.5)
        w_seq = weights.get("sequence_correctness", 0.2)

        traces.append({
            "name": r.test_case,
            "tags": list(((test_metadata or {}).get(r.test_case, {})).get("tags") or []),
            "diagram": _mermaid_trace(r) if has_steps else "",
            "has_steps": has_steps,
            "passed": r.passed,
            "cost": f"${cost:.6f}".rstrip('0').rstrip('.') if cost else "$0",
            "latency": f"{int(latency)}ms",
            "tokens": f"{tokens:,} tokens" if tokens else "",
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "score": round(r.score, 1),
            "tool_accuracy": tool_acc,
            "output_quality": output_qual,
            "sequence_correct": seq_correct,
            "w_tool": round(w_tool * 100),
            "w_output": round(w_output * 100),
            "w_seq": round(w_seq * 100),
            "model": ", ".join(models) if models else "Unknown",
            "baseline_created": baseline_created or "Unknown",
            "baseline_model": baseline_model,
            "query": getattr(r, "input_query", "") or "",
            "output": _strip_markdown(getattr(r, "actual_output", "") or ""),
            "turns": turn_list,
            "hallucination": _extract_check_result(r, "hallucination"),
            "safety": _extract_check_result(r, "safety"),
            "pii": _extract_check_result(r, "pii"),
            "forbidden_tools": _extract_check_result(r, "forbidden_tools"),
            "anomaly_report": _normalize_anomaly_report(getattr(r, "anomaly_report", None)),
            "trust_report": _normalize_trust_report(getattr(r, "trust_report", None)),
            "coherence_report": _normalize_coherence_report(getattr(r, "coherence_report", None)),
            "failure_reasons": failure_reasons,
            "output_rationale": output_rationale,
        })
    actual_results_dict = {r.test_case: r for r in results}
    diff_rows = _diff_rows(diffs or [], golden_traces, actual_results_dict, test_metadata, root_causes)
    timeline = _timeline_data(results)
    behavior_summary = _behavior_summary(results, diff_rows, test_metadata, healing_summary)
    adapter_compare = _compute_adapter_compare(results)

    # Build comparison data if multiple runs provided
    compare_data = None
    if compare_results:
        labels = compare_labels or []
        all_runs = [results] + list(compare_results)
        all_labels = labels if labels else [f"Run {i+1}" for i in range(len(all_runs))]
        compare_data = {
            "labels": all_labels,
            "runs": [_kpis(r) for r in all_runs],
        }

    # Compute dashboard data (health gauge + sparkline trends)
    dashboard = None
    try:
        from evalview.core.drift_tracker import DriftTracker
        _dt = DriftTracker()

        # Health gauge — uses test pass rate (score >= threshold), not diff status.
        # A test that passes with changed output is still healthy.
        total_compared = len(results)
        passed_count = sum(1 for r in results if r.passed)
        health_pct = round(passed_count / total_compared * 100) if total_compared > 0 else 0

        # Sparkline trends per test (output similarity over last 10 checks)
        test_sparklines = []
        for d in diff_rows:
            history = _dt.get_test_history(d["name"], limit=10)
            # newest-first → reverse for chronological sparkline
            sims = [round(h["output_similarity"] * 100, 1) for h in reversed(history)]
            if sims:
                test_sparklines.append({
                    "name": d["name"],
                    "values": sims,
                })

        # Overall pass rate trend
        pass_trend_raw = _dt.get_pass_rate_trend(window=10)
        pass_trend = [round(v * 100, 1) for v in pass_trend_raw]

        dashboard = {
            "health_pct": health_pct,
            "passed": passed_count,
            "failed": total_compared - passed_count,
            "changed": sum(1 for d in diff_rows if d["status"] in ("tools_changed", "output_changed")),
            "regressions": sum(1 for d in diff_rows if d["status"] == "regression"),
            "total": total_compared,
            "test_sparklines": test_sparklines,
            "pass_trend": pass_trend,
        }
    except Exception:
        pass

    html = _render_template(
        title=title,
        notes=notes or "",
        generated_at=datetime.now().strftime("%Y-%m-%d %H:%M"),
        kpis=kpis,
        baseline=baseline,
        judge_usage=judge_usage or {},
        traces=traces,
        diff_rows=diff_rows,
        timeline=timeline,
        compare=compare_data,
        default_tab=default_tab or "overview",
        dashboard=dashboard,
        behavior_summary=behavior_summary,
        adapter_compare=adapter_compare,
        active_tags=active_tags or [],
        healing=healing_summary.model_dump() if healing_summary is not None else None,
        model_runtime=model_runtime_summary.model_dump() if model_runtime_summary is not None else None,
        effective_all_passed=effective_all_passed,
    )

    abs_path = os.path.abspath(output_path)
    with open(abs_path, "w", encoding="utf-8") as f:
        f.write(html)

    if auto_open:
        from pathlib import Path as _Path

        webbrowser.open_new_tab(_Path(abs_path).resolve().as_uri())

    return abs_path


# ── Template ───────────────────────────────────────────────────────────────────

def _render_template(**ctx: Any) -> str:
    """Render the report HTML using Jinja2."""
    try:
        from jinja2 import BaseLoader, Environment
    except ImportError:
        return f"<html><body><pre>{json.dumps(ctx, default=str, indent=2)}</pre></body></html>"

    env = Environment(loader=BaseLoader(), autoescape=True)

    # Mark pre-sanitized Mermaid diagrams as safe so Jinja2 autoescape
    # doesn't HTML-encode arrows (-->, ->>) which breaks rendering.
    # User content in labels is already sanitized by _safe_mermaid().
    from markupsafe import Markup
    for t in ctx.get("traces", []):
        if t.get("diagram"):
            t["diagram"] = Markup(t["diagram"])
    for d in ctx.get("diff_rows", []):
        if d.get("golden_diagram"):
            d["golden_diagram"] = Markup(d["golden_diagram"])
        if d.get("actual_diagram"):
            d["actual_diagram"] = Markup(d["actual_diagram"])

    return env.from_string(_TEMPLATE).render(**ctx)
