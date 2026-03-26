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
from pathlib import Path
import webbrowser
from collections import Counter
from datetime import datetime
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from evalview.core.types import EvaluationResult
    from evalview.core.diff import TraceDiff


# ── Mermaid helpers ────────────────────────────────────────────────────────────

def _mermaid_from_steps(steps: List[Any], query: str = "", output: str = "") -> str:
    """Core Mermaid sequence diagram builder from a steps list."""
    if not steps:
        return "sequenceDiagram\n    Note over Agent: Direct response — no tools used"

    lines = ["sequenceDiagram"]
    lines.append("    participant User")
    lines.append("    participant Agent")

    seen_tools: Dict[str, str] = {}
    for step in steps:
        tool: str = str(getattr(step, "tool_name", None) or getattr(step, "step_name", None) or "unknown")
        if tool not in seen_tools:
            alias = f"T{len(seen_tools)}"
            seen_tools[tool] = alias
            short = (tool[:31] + "…") if len(tool) > 32 else tool
            lines.append(f"    participant {alias} as {short}")

    short_query = _safe_mermaid((query[:40] + "…") if len(query) > 40 else query) if query else "..."
    lines.append(f"    User->>Agent: {short_query}")

    current_turn = None

    for step in steps:
        step_turn = getattr(step, "turn_index", None)

        # Add a turn separator when the turn index changes
        if step_turn is not None and step_turn != current_turn:
            step_query = getattr(step, "turn_query", "") or ""
            safe_query = _safe_mermaid((step_query[:57] + "...") if len(step_query) > 60 else step_query)
            if safe_query:
                lines.append(f"    Note over User,Agent: Turn {step_turn} - {safe_query}")
            else:
                lines.append(f"    Note over User,Agent: Turn {step_turn}")
            current_turn = step_turn

        tool = str(getattr(step, "tool_name", None) or getattr(step, "step_name", None) or "unknown")
        alias = seen_tools.get(tool, tool)
        params = getattr(step, "parameters", {}) or {}
        param_str = ", ".join(f"{k}={str(v)[:20]}" for k, v in list(params.items())[:2])
        if len(params) > 2:
            param_str += "…"
        success = getattr(step, "success", True)
        arrow = "->>" if success else "-x"
        lines.append(f"    Agent{arrow}{alias}: {_safe_mermaid(param_str or tool)}")
        out = getattr(step, "output", None)
        out_str = str(out)[:30] if out is not None else "ok"
        lines.append(f"    {alias}-->Agent: {_safe_mermaid(out_str)}")

    short_out = _safe_mermaid((output[:40] + "…") if len(output) > 40 else output) if output else "..."
    lines.append(f"    Agent-->>User: {short_out}")

    return "\n".join(lines)


def _mermaid_trace(result: "EvaluationResult") -> str:
    """Convert an EvaluationResult into a Mermaid sequence diagram."""
    steps = []
    try:
        steps = result.trace.steps or []
    except AttributeError:
        pass
    query: str = str(getattr(result, "input_query", "") or "")
    output: str = str(getattr(result, "actual_output", "") or "")
    return _mermaid_from_steps(steps, query, output)


def _strip_markdown(text: str) -> str:
    """Remove common markdown symbols for clean display in HTML."""
    import re
    text = re.sub(r'\*{1,3}(.+?)\*{1,3}', r'\1', text, flags=re.DOTALL)  # bold/italic
    text = re.sub(r'`(.+?)`', r'\1', text, flags=re.DOTALL)               # inline code
    text = re.sub(r'^#{1,6}\s+', '', text, flags=re.MULTILINE)            # headings
    return text


def _safe_mermaid(s: str) -> str:
    """Strip everything except safe alphanumeric + basic punctuation for Mermaid labels."""
    import re
    s = s.replace("\n", " ").replace("\r", "")
    s = re.sub(r'[^\w\s\.\-_/=:,]', '', s)
    s = s[:28].strip()
    return (s + '...') if len(s) == 28 else s or '...'


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
) -> List[Dict[str, Any]]:
    rows = []
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
        })
    return rows


# ── Timeline helpers ───────────────────────────────────────────────────────────

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
            "failure_reasons": failure_reasons,
            "output_rationale": output_rationale,
        })
    actual_results_dict = {r.test_case: r for r in results}
    diff_rows = _diff_rows(diffs or [], golden_traces, actual_results_dict)
    timeline = _timeline_data(results)

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


_TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{{ title }}</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=JetBrains+Mono:wght@400;500;600&display=swap" rel="stylesheet">
<script src="https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4/dist/chart.umd.min.js"></script>
<style>
*{box-sizing:border-box;margin:0;padding:0}
:root{
  --green:#10b981;--green-bright:#34d399;
  --red:#ef4444;--red-bright:#f87171;
  --yellow:#f59e0b;--yellow-bright:#fbbf24;
  --blue:#2563eb;--blue-bright:#3b82f6;
  --teal:#0d9488;--teal-bright:#14b8a6;
  --cyan:#06b6d4;
  --bg:#060b18;--bg-card:rgba(12,20,36,.75);
  --border:rgba(51,65,85,.45);--border-light:rgba(71,85,105,.5);
  --text:#f1f5f9;--text-2:#94a3b8;--text-3:#64748b;--text-4:#475569;
  --r:16px;--r-sm:12px;--r-xs:8px;
  --font:'Inter',-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;
  --mono:'JetBrains Mono','Fira Code','SF Mono',monospace;
}
html{scroll-behavior:smooth;-webkit-font-smoothing:antialiased;-moz-osx-font-smoothing:grayscale}
body{font-family:var(--font);font-size:14px;line-height:1.6;color:var(--text);min-height:100vh;overflow-x:hidden;background:var(--bg)}

/* ── Header ── */
.header{
  position:sticky;top:0;z-index:200;
  background:rgba(6,11,24,.85);border-bottom:1px solid var(--border);
  backdrop-filter:blur(20px);-webkit-backdrop-filter:blur(20px);
  padding:0 40px;height:52px;display:flex;align-items:center;justify-content:space-between;
}
.logo{display:flex;align-items:center;gap:10px}
.logo-icon{width:28px;height:28px;border-radius:7px;flex-shrink:0;background:linear-gradient(135deg,var(--blue-bright),var(--teal));display:flex;align-items:center;justify-content:center;font-size:13px;box-shadow:0 2px 10px rgba(37,99,235,.2)}
.logo-text{font-size:14px;font-weight:700;letter-spacing:-.02em;color:var(--text)}
.logo-sub{font-size:10px;color:var(--text-4);font-weight:400}
.header-right{display:flex;align-items:center;gap:6px}

/* ── Badges ── */
.badge{display:inline-flex;align-items:center;gap:4px;padding:3px 10px;border-radius:20px;font-size:11px;font-weight:600;white-space:nowrap}
.b-green{background:rgba(16,185,129,.12);color:var(--green-bright);border:1px solid rgba(16,185,129,.25)}
.b-red{background:rgba(239,68,68,.12);color:var(--red-bright);border:1px solid rgba(239,68,68,.25)}
.b-yellow{background:rgba(245,158,11,.12);color:var(--yellow-bright);border:1px solid rgba(245,158,11,.25)}
.b-cyan{background:rgba(6,182,212,.12);color:#67e8f9;border:1px solid rgba(6,182,212,.25)}

/* ── Dashboard Gauge ── */
.health-gauge{display:flex;align-items:center;gap:16px;padding:16px 20px}
.gauge-ring{position:relative;width:80px;height:80px;flex-shrink:0}
.gauge-ring svg{transform:rotate(-90deg)}
.gauge-ring .gauge-text{position:absolute;inset:0;display:flex;align-items:center;justify-content:center;font-size:18px;font-weight:800;letter-spacing:-.02em}
.gauge-stats{display:flex;flex-direction:column;gap:4px}
.gauge-stat{font-size:12px;display:flex;align-items:center;gap:6px}
.gauge-dot{width:8px;height:8px;border-radius:50%;flex-shrink:0}
.confidence-badge{display:inline-flex;align-items:center;gap:4px;padding:2px 8px;border-radius:12px;font-size:10px;font-weight:600;margin-left:8px}
.conf-high{background:rgba(239,68,68,.15);color:var(--red-bright);border:1px solid rgba(239,68,68,.2)}
.conf-medium{background:rgba(245,158,11,.15);color:var(--yellow-bright);border:1px solid rgba(245,158,11,.2)}
.conf-low{background:rgba(100,116,139,.15);color:var(--text-3);border:1px solid rgba(100,116,139,.2)}
.conf-insufficient{background:rgba(100,116,139,.08);color:var(--text-4);border:1px solid rgba(100,116,139,.15)}
.accept-box{margin:8px 18px 12px;padding:12px 16px;border-radius:var(--r-xs);border:1px solid rgba(16,185,129,.25);background:rgba(16,185,129,.06)}
.accept-box.neutral{border-color:rgba(245,158,11,.25);background:rgba(245,158,11,.06)}
.accept-box code{background:rgba(255,255,255,.06);padding:3px 8px;border-radius:4px;font-family:var(--mono);font-size:11px;border:1px solid var(--border);user-select:all}
.b-blue{background:rgba(37,99,235,.12);color:var(--blue-bright);border:1px solid rgba(37,99,235,.25)}
.b-purple{background:rgba(13,148,136,.12);color:var(--teal-bright);border:1px solid rgba(13,148,136,.25)}

/* ── Layout ── */
.main{max-width:1160px;margin:0 auto;padding:28px 36px 80px;position:relative;z-index:1}

/* ── Tabs ── */
.tabbar{display:flex;gap:0;background:var(--bg-card);border:1px solid var(--border);border-radius:var(--r-sm);padding:3px;margin-bottom:28px}
.tab{flex:1;text-align:center;background:none;border:none;color:var(--text-4);cursor:pointer;font:600 12px/1 var(--font);padding:10px 12px;border-radius:9px;transition:all .15s}
.tab:hover{color:var(--text-2);background:rgba(255,255,255,.03)}
.tab.on{color:#fff;background:rgba(37,99,235,.18);border:1px solid rgba(37,99,235,.3)}
.panel{display:none}.panel.on{display:block}

/* ══════════════════════════════════════════════
   KPI STRIP — compact horizontal bar
   ══════════════════════════════════════════════ */
.kpi-strip{
  display:flex;align-items:center;gap:0;
  background:var(--bg-card);border:1px solid var(--border);border-radius:var(--r);
  overflow:hidden;margin-bottom:14px;padding:10px 0;flex-wrap:wrap;
}
.kpi-item{
  display:flex;align-items:center;gap:8px;padding:4px 20px;
  border-right:1px solid var(--border);white-space:nowrap;
}
.kpi-item:last-child{border-right:none}
.kpi-dot{width:8px;height:8px;border-radius:50%;flex-shrink:0}
.kpi-dot.green{background:var(--green);box-shadow:0 0 6px rgba(16,185,129,.4)}
.kpi-dot.yellow{background:var(--yellow);box-shadow:0 0 6px rgba(245,158,11,.4)}
.kpi-dot.red{background:var(--red);box-shadow:0 0 6px rgba(239,68,68,.4)}
.kpi-val{font-size:13px;font-weight:700;color:var(--text)}
.kpi-label{font-size:11px;color:var(--text-3);font-weight:500}

/* ── Card ── */
.card{background:var(--bg-card);border:1px solid var(--border);border-radius:var(--r);padding:20px 22px;margin-bottom:14px;position:relative;overflow:hidden}
.card-title{font-size:11px;font-weight:700;color:var(--text-3);text-transform:uppercase;letter-spacing:.06em;margin-bottom:14px;display:flex;align-items:center;gap:7px}
.card-title::before{content:'';width:3px;height:11px;border-radius:2px;background:var(--blue-bright)}
.chart-wrap{position:relative}

/* ── Meta row ── */
.meta-row{display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-bottom:14px}
@media(max-width:800px){.meta-row{grid-template-columns:1fr}}
.meta-card{background:var(--bg-card);border:1px solid var(--border);border-radius:var(--r-sm);padding:14px 18px}
.meta-label{font-size:10px;font-weight:700;color:var(--text-4);text-transform:uppercase;letter-spacing:.08em;margin-bottom:5px}
.meta-value{font-size:14px;font-weight:700;color:var(--text)}
.meta-sub{font-size:11px;color:var(--text-4);margin-top:3px}
.healing-grid{display:grid;grid-template-columns:1.2fr .8fr;gap:12px;margin-bottom:14px}
@media(max-width:800px){.healing-grid{grid-template-columns:1fr}}
.healing-list{display:flex;flex-direction:column;gap:8px}
.healing-row{display:flex;align-items:flex-start;justify-content:space-between;gap:10px;padding:10px 12px;border:1px solid var(--border);border-radius:var(--r-xs);background:rgba(255,255,255,.02)}
.healing-name{font-weight:700;font-size:13px;color:var(--text)}
.healing-reason{font-size:11px;color:var(--text-3);margin-top:3px}

/* ── Chart row ── */

/* ── Trace items ── */
.item{background:var(--bg-card);border:1px solid var(--border);border-radius:var(--r);margin-bottom:8px;overflow:hidden;transition:border-color .15s}
.item:hover{border-color:var(--border-light)}
.item-head{padding:12px 18px;display:flex;align-items:center;gap:10px;cursor:pointer;transition:background .1s}
.item-head:hover{background:rgba(255,255,255,.015)}
.item-name{font-weight:700;font-size:14px;flex:1;letter-spacing:-.02em}
.item-meta{display:flex;align-items:center;gap:5px;flex-shrink:0;flex-wrap:wrap}
.mc{display:inline-flex;align-items:center;gap:3px;padding:2px 7px;border-radius:4px;background:rgba(255,255,255,.035);font-size:10px;font-weight:500;color:var(--text-3);white-space:nowrap}
.chevron{color:var(--text);font-size:18px;transition:transform .2s;flex-shrink:0;width:24px;height:24px;display:inline-flex;align-items:center;justify-content:center;border-radius:6px;background:rgba(255,255,255,.05);border:1px solid var(--border)}
.item-head:hover .chevron{background:rgba(255,255,255,.08);border-color:var(--border-light)}
details[open] .turn-chevron{transform:rotate(90deg)}
.item-body{padding:18px;border-top:1px solid var(--border);background:rgba(0,0,0,.12)}
.mermaid-box{background:rgba(0,0,0,.18);border:1px solid rgba(51,65,85,.35);border-radius:var(--r-sm);padding:14px 14px;overflow-x:auto;min-height:120px}
.mermaid-box svg{min-width:400px;max-width:100%;height:auto;display:block;margin:0 auto}
.mermaid-box .mermaid{min-width:500px}
.mermaid-box line.actor-line{stroke-dasharray:4 4;stroke:rgba(100,116,139,.15) !important}

/* ── Chat turns ── */
.chat-container{margin-top:16px;padding:14px;background:rgba(0,0,0,.1);border:1px solid rgba(51,65,85,.25);border-radius:var(--r-sm)}
.chat-header{font-size:11px;font-weight:700;color:var(--text-3);text-transform:uppercase;letter-spacing:.06em;margin-bottom:12px;padding-bottom:8px;border-bottom:1px solid rgba(51,65,85,.25)}
.chat-messages{display:flex;flex-direction:column;gap:3px}
.chat-bubble{max-width:78%;padding:9px 13px;font-size:13px;line-height:1.5;border-radius:12px}
.chat-bubble.user{align-self:flex-end;background:rgba(37,99,235,.1);border:1px solid rgba(37,99,235,.18);color:var(--text);border-bottom-right-radius:3px}
.chat-bubble.agent{align-self:flex-start;background:rgba(255,255,255,.025);border:1px solid rgba(51,65,85,.35);color:var(--text-2);border-bottom-left-radius:3px}
.chat-meta{display:flex;align-items:center;gap:6px;padding:4px 2px;font-size:10px;color:var(--text-4);font-weight:500}
.chat-meta.right{justify-content:flex-end}
.chat-tool-tag{display:inline-flex;padding:1px 6px;border-radius:3px;background:rgba(37,99,235,.07);border:1px solid rgba(37,99,235,.12);font-size:10px;font-weight:600;color:var(--blue-bright);font-family:var(--mono)}
.chat-eval{padding:5px 9px;border-radius:6px;font-size:11px;font-weight:600;max-width:78%}
.chat-eval.pass{align-self:flex-start;background:rgba(16,185,129,.07);border:1px solid rgba(16,185,129,.18);color:var(--green-bright)}
.chat-eval.fail{align-self:flex-start;background:rgba(239,68,68,.07);border:1px solid rgba(239,68,68,.18);color:var(--red-bright)}

/* ── Diffs ── */
.diff-item{background:var(--bg-card);border:1px solid var(--border);border-radius:var(--r);margin-bottom:8px;overflow:hidden}
.diff-head{padding:12px 18px;display:flex;align-items:center;gap:8px;flex-wrap:wrap;border-bottom:1px solid var(--border)}
.diff-name{font-weight:700;font-size:14px;flex:1;letter-spacing:-.02em}
.diff-cols{display:grid;grid-template-columns:1fr 1fr}
.diff-col{padding:14px 18px}
.diff-col+.diff-col{border-left:1px solid var(--border)}
.col-title{font-size:10px;font-weight:700;color:var(--text-4);text-transform:uppercase;letter-spacing:.06em;margin-bottom:7px}
.tags{display:flex;flex-wrap:wrap;gap:3px;margin-bottom:7px}
.tag{background:rgba(255,255,255,.035);border:1px solid rgba(51,65,85,.45);border-radius:4px;padding:2px 7px;font-size:11px;font-family:var(--mono);font-weight:500}
.tag.add{border-color:rgba(16,185,129,.25);color:var(--green-bright);background:rgba(16,185,129,.05)}
.tag.rem{border-color:rgba(239,68,68,.25);color:var(--red-bright);background:rgba(239,68,68,.05);text-decoration:line-through}
.outbox{background:rgba(0,0,0,.18);border:1px solid rgba(51,65,85,.35);border-radius:var(--r-xs);padding:10px;font:12px/1.6 var(--mono);color:var(--text-3);white-space:pre-wrap;word-break:break-all;max-height:180px;overflow-y:auto}
.difflines{background:rgba(0,0,0,.18);border:1px solid rgba(51,65,85,.35);border-radius:var(--r-xs);padding:8px;font:11px/1.6 var(--mono);max-height:180px;overflow-y:auto;margin-top:7px}
.difflines .a{color:var(--green-bright);background:rgba(16,185,129,.04);display:block;padding:1px 4px;margin:0 -4px;border-radius:2px}
.difflines .r{color:var(--red-bright);background:rgba(239,68,68,.04);display:block;padding:1px 4px;margin:0 -4px;border-radius:2px}
.sim{display:inline-flex;align-items:center;gap:4px;font-size:11px;color:var(--text-3)}
.sim-track{width:40px;height:3px;background:rgba(255,255,255,.06);border-radius:2px;overflow:hidden;display:inline-block;vertical-align:middle}
.sim-fill{height:100%;border-radius:2px}
.sim-fill.hi{background:var(--green)}.sim-fill.mid{background:var(--yellow)}.sim-fill.lo{background:var(--red)}
.pipeline{display:flex;flex-direction:column;gap:5px;padding:12px 18px;border-top:1px solid var(--border)}
.pipeline-row{display:flex;align-items:center;gap:3px;flex-wrap:wrap}
.pipeline-label{font-size:10px;font-weight:700;color:var(--text-4);text-transform:uppercase;letter-spacing:.06em;width:60px;flex-shrink:0}
.pipe-step{display:inline-flex;padding:3px 9px;border-radius:4px;font-size:11px;font-family:var(--mono);font-weight:600;background:rgba(255,255,255,.035);border:1px solid rgba(51,65,85,.45);color:var(--text-2);position:relative}
.pipe-step+.pipe-step{margin-left:5px}
.pipe-step+.pipe-step::before{content:'→';position:absolute;left:-12px;color:var(--text-4);font-size:9px;font-family:var(--font)}
.pipe-step.match{border-color:rgba(37,99,235,.2);background:rgba(37,99,235,.04)}
.pipe-step.added{border-color:rgba(16,185,129,.25);color:var(--green-bright);background:rgba(16,185,129,.05)}
.pipe-step.removed{border-color:rgba(239,68,68,.25);color:var(--red-bright);background:rgba(239,68,68,.05);text-decoration:line-through}
.traj-grid{display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-top:12px;padding-top:12px;border-top:1px solid var(--border)}
.traj-col .col-title{padding-bottom:6px}

/* ── Tables ── */
.ev-table{width:100%;border-collapse:collapse;font-size:13px}
.ev-table th{text-align:left;padding:7px 10px;color:var(--text-4);font-size:10px;font-weight:700;text-transform:uppercase;letter-spacing:.06em;border-bottom:1px solid var(--border)}
.ev-table td{padding:9px 10px;border-bottom:1px solid rgba(51,65,85,.25)}
.ev-table tr:hover td{background:rgba(255,255,255,.012)}
.ev-table .mono{font-family:var(--mono);font-size:12px}
.ev-table .num{font-weight:700;font-variant-numeric:tabular-nums}
.param-table{width:100%;border-collapse:collapse;font-size:12px}
.param-table th{text-align:left;padding:5px 9px;color:var(--text-4);font-size:10px;font-weight:700;text-transform:uppercase;letter-spacing:.06em;border-bottom:1px solid var(--border)}
.param-table td{padding:5px 9px;border-bottom:1px solid rgba(51,65,85,.25)}
table td,table th{transition:background .1s}
.empty{text-align:center;padding:64px 40px;color:var(--text-4)}
.empty-icon{font-size:32px;margin-bottom:10px;display:block;opacity:.25}
.empty code{background:rgba(255,255,255,.05);padding:2px 7px;border-radius:4px;font-family:var(--mono);font-size:12px;border:1px solid var(--border)}
::-webkit-scrollbar{width:4px;height:4px}::-webkit-scrollbar-track{background:transparent}::-webkit-scrollbar-thumb{background:rgba(255,255,255,.07);border-radius:3px}
</style>
</head>
<body>

<header class="header">
  <div class="logo">
    <div class="logo-icon">◈</div>
    <div><div class="logo-text">{{ title }}</div><div class="logo-sub">{{ generated_at }}{% if notes %} · {{ notes }}{% endif %}</div></div>
  </div>
  <div class="header-right">
    {% if effective_all_passed is not none %}{% if effective_all_passed %}<span class="badge b-green">✓ Final Outcome Passing</span>{% else %}<span class="badge b-red">✗ Final Outcome Failing</span>{% endif %}{% endif %}
    {% if kpis %}{% if kpis.failed == 0 %}<span class="badge b-green">✓ All Passing</span>{% else %}<span class="badge b-red">✗ {{ kpis.failed }} Failed</span>{% endif %}<span class="badge b-blue">{{ kpis.total }} Tests</span>{% endif %}
  </div>
</header>

<main class="main">
  <div class="tabbar">
    <button class="tab {% if default_tab == 'overview' %}on{% endif %}" onclick="show('overview',this)">Overview</button>
    <button class="tab {% if default_tab == 'trace' %}on{% endif %}" onclick="show('trace',this)">Execution Trace</button>
    {% if diff_rows %}<button class="tab {% if default_tab == 'diffs' %}on{% endif %}" onclick="show('diffs',this)">Diffs</button>{% endif %}
    <button class="tab {% if default_tab == 'timeline' %}on{% endif %}" onclick="show('timeline',this)">Timeline</button>
    {% if compare %}<button class="tab" onclick="show('compare',this)">Compare Runs</button>{% endif %}
  </div>

  <!-- ═══════════ OVERVIEW ═══════════ -->
  <div id="p-overview" class="panel {% if default_tab == 'overview' %}on{% endif %}">
    {% if kpis %}

    <!-- KPI Strip -->
    <div class="kpi-strip">
      <div class="kpi-item">
        <span class="kpi-dot {% if kpis.pass_rate >= 80 %}green{% elif kpis.pass_rate >= 60 %}yellow{% else %}red{% endif %}"></span>
        <span class="kpi-val" style="color:{% if kpis.pass_rate >= 80 %}var(--green-bright){% elif kpis.pass_rate >= 60 %}var(--yellow-bright){% else %}var(--red-bright){% endif %}">{{ kpis.pass_rate }}% passed</span>
        <span class="kpi-label">({{ kpis.passed }}/{{ kpis.total }})</span>
      </div>
      <div class="kpi-item">
        <span class="kpi-val" style="color:{% if kpis.avg_score >= 80 %}var(--green-bright){% elif kpis.avg_score >= 60 %}var(--yellow-bright){% else %}var(--red-bright){% endif %}">Avg {{ kpis.avg_score }}/100</span>
      </div>
      <div class="kpi-item">
        <span class="kpi-val" style="color:var(--blue-bright)">${{ kpis.total_cost }}</span>
        <span class="kpi-label">total</span>
      </div>
      <div class="kpi-item">
        <span class="kpi-val">{{ kpis.avg_latency_ms|int }}ms</span>
        <span class="kpi-label">avg</span>
      </div>
      <div class="kpi-item">
        <span class="kpi-val">{{ kpis.models_display }}</span>
        {% if kpis.total_tokens %}<span class="kpi-label">({{ '{:,}'.format(kpis.total_tokens) }} tokens)</span>{% endif %}
      </div>
    </div>
    {% if dashboard %}
    <!-- Health Gauge + Trend Sparklines (or Score Per Test if no trends) -->
    <div class="meta-row">
      <div class="card" style="margin-bottom:0">
        <div class="card-title">Health Gauge</div>
        <div class="health-gauge">
          <div class="gauge-ring">
            <svg viewBox="0 0 36 36">
              <path d="M18 2.0845 a 15.9155 15.9155 0 0 1 0 31.831 a 15.9155 15.9155 0 0 1 0 -31.831" fill="none" stroke="rgba(255,255,255,.06)" stroke-width="3"/>
              <path d="M18 2.0845 a 15.9155 15.9155 0 0 1 0 31.831 a 15.9155 15.9155 0 0 1 0 -31.831" fill="none" stroke="{% if dashboard.health_pct >= 80 %}var(--green-bright){% elif dashboard.health_pct >= 50 %}var(--yellow-bright){% else %}var(--red-bright){% endif %}" stroke-width="3" stroke-dasharray="{{ dashboard.health_pct }}, 100" stroke-linecap="round"/>
            </svg>
            <span class="gauge-text" style="color:{% if dashboard.health_pct >= 80 %}var(--green-bright){% elif dashboard.health_pct >= 50 %}var(--yellow-bright){% else %}var(--red-bright){% endif %}">{{ dashboard.health_pct }}%</span>
          </div>
          <div class="gauge-stats">
            {% if dashboard.passed %}<div class="gauge-stat"><span class="gauge-dot" style="background:var(--green)"></span> {{ dashboard.passed }} passed</div>{% endif %}
            {% if dashboard.failed %}<div class="gauge-stat"><span class="gauge-dot" style="background:var(--red)"></span> {{ dashboard.failed }} failed</div>{% endif %}
            {% if dashboard.regressions %}<div class="gauge-stat"><span class="gauge-dot" style="background:var(--red)"></span> {{ dashboard.regressions }} regression{{ 's' if dashboard.regressions != 1 else '' }}</div>{% endif %}
            {% if dashboard.changed %}<div class="gauge-stat"><span class="gauge-dot" style="background:var(--yellow)"></span> {{ dashboard.changed }} diff{{ 's' if dashboard.changed != 1 else '' }} from baseline</div>{% endif %}
          </div>
        </div>
      </div>
      {% if dashboard.test_sparklines or dashboard.pass_trend %}
      <div class="card" style="margin-bottom:0">
        <div class="card-title">Score Trends</div>
        <div style="height:{{ [dashboard.test_sparklines|length * 28 + 50, 100]|max }}px;position:relative"><canvas id="trendChart"></canvas></div>
      </div>
      {% else %}
      <!-- No trend data yet — show Score Per Test next to Health Gauge -->
      <div class="card" style="margin-bottom:0">
        <div class="card-title">Score per Test</div>
        <div class="chart-wrap" style="height:{{ [kpis.scores|length * 40 + 24, 120]|max }}px"><canvas id="bars"></canvas></div>
      </div>
      {% endif %}
    </div>
    {% endif %}
    {% if healing %}
    <div class="healing-grid">
      <div class="card" style="margin-bottom:0">
        <div class="card-title">Healing Summary</div>
        <div style="display:flex;flex-wrap:wrap;gap:6px;margin-bottom:12px">
          {% if healing.total_healed %}<span class="badge b-green">⚡ {{ healing.total_healed }} healed</span>{% endif %}
          {% if healing.total_proposed %}<span class="badge b-cyan">◈ {{ healing.total_proposed }} proposed</span>{% endif %}
          {% if healing.total_review %}<span class="badge b-yellow">⚠ {{ healing.total_review }} review</span>{% endif %}
          {% if healing.total_blocked %}<span class="badge b-red">✗ {{ healing.total_blocked }} blocked</span>{% endif %}
          <span class="badge b-blue">{{ healing.policy_version }}</span>
        </div>
        <div class="healing-list">
          {% for hr in healing.results %}
          <div class="healing-row">
            <div>
              <div class="healing-name">{{ hr.test_name }}</div>
              <div class="healing-reason">{{ hr.diagnosis.reason }}</div>
            </div>
            <div>
              {% if hr.healed %}<span class="badge b-green">HEALED</span>
              {% elif hr.proposed %}<span class="badge b-cyan">PROPOSED</span>
              {% elif hr.diagnosis.action == 'blocked' %}<span class="badge b-red">BLOCKED</span>
              {% else %}<span class="badge b-yellow">REVIEW</span>{% endif %}
            </div>
          </div>
          {% endfor %}
        </div>
      </div>
      <div class="card" style="margin-bottom:0">
        <div class="card-title">Healing Policy</div>
        <div class="meta-label">Attempted</div>
        <div class="meta-value">{{ healing.attempted_count }}/{{ healing.failed_count }}</div>
        <div class="meta-sub">{{ healing.unresolved_count }} unresolved after healing</div>
        {% if healing.thresholds %}
        <div style="margin-top:12px;font-size:12px;color:var(--text-2);display:flex;flex-direction:column;gap:6px">
          {% if healing.thresholds.min_variant_score is defined %}<div>Variant score threshold: <b>{{ healing.thresholds.min_variant_score }}</b></div>{% endif %}
          {% if healing.thresholds.max_auto_variants is defined %}<div>Max auto variants: <b>{{ healing.thresholds.max_auto_variants|int }}</b></div>{% endif %}
          {% if healing.thresholds.max_cost_multiplier is defined %}<div>Cost guardrail: <b>{{ healing.thresholds.max_cost_multiplier }}x</b></div>{% endif %}
          {% if healing.thresholds.max_latency_multiplier is defined %}<div>Latency guardrail: <b>{{ healing.thresholds.max_latency_multiplier }}x</b></div>{% endif %}
        </div>
        {% endif %}
        {% if healing.audit_path %}
        <div style="margin-top:12px;font-size:11px;color:var(--text-3)">
          Audit log: <code style="background:rgba(255,255,255,.04);padding:2px 6px;border-radius:3px;font-family:var(--mono);font-size:11px;border:1px solid var(--border)">{{ healing.audit_path }}</code>
        </div>
        {% endif %}
      </div>
    </div>
    {% endif %}
    {% if model_runtime and model_runtime.classification != 'none' %}
    <div class="card">
      <div class="card-title">Model / Runtime Signal</div>
      <div style="display:flex;flex-wrap:wrap;gap:8px;margin-bottom:10px">
        <span class="badge {% if model_runtime.classification == 'declared' %}b-red{% else %}b-yellow{% endif %}">
          {% if model_runtime.classification == 'declared' %}Declared model change{% else %}Possible runtime update{% endif %}
        </span>
        <span class="badge b-blue">{{ model_runtime.confidence }} confidence</span>
        <span class="badge b-cyan">{{ model_runtime.affected_count }} affected</span>
      </div>
      {% if model_runtime.baseline_fingerprints and model_runtime.current_fingerprints %}
      <div style="font-size:12px;color:var(--text-2);margin-bottom:10px">
        <span style="color:var(--text-4)">Runtime fingerprint</span>
        <span class="mono" style="color:var(--text-3)"> {{ model_runtime.baseline_fingerprints[0] }}</span>
        <span style="color:var(--text-4)"> → </span>
        <span class="mono" style="color:var(--text)">{{ model_runtime.current_fingerprints[0] }}</span>
      </div>
      {% endif %}
      {% if model_runtime.evidence %}
      <div style="display:flex;flex-direction:column;gap:6px;font-size:12px;color:var(--text-2)">
        {% for item in model_runtime.evidence[:4] %}
        <div><span style="color:var(--text-4)">•</span> {{ item }}</div>
        {% endfor %}
      </div>
      {% endif %}
    </div>
    {% endif %}
    {% if not judge_usage or not judge_usage.call_count %}
    <div style="font-size:11px;color:var(--text-4);padding:0 4px 6px;line-height:1.4">
      No LLM judge was used — hallucination, safety, and PII checks were skipped. Run without <code style="background:rgba(255,255,255,.04);padding:1px 5px;border-radius:3px;font-family:var(--mono);font-size:10px;border:1px solid var(--border)">--no-judge</code> to enable them.
    </div>
    {% endif %}
    {% if baseline.latest_created_display != 'Unknown' or (judge_usage and judge_usage.call_count) %}
    <div class="meta-row" style="grid-template-columns:{% if baseline.latest_created_display != 'Unknown' and judge_usage and judge_usage.call_count %}1fr 1fr 1fr{% else %}1fr 1fr{% endif %}">
      {% if baseline.latest_created_display != 'Unknown' %}
      <div class="meta-card">
        <div class="meta-label">Baseline Snapshot</div>
        <div class="meta-value">{{ baseline.latest_created_display }}</div>
        <div class="meta-sub">{% if baseline.models_display != 'Unknown' %}Model: {{ baseline.models_display }}{% endif %}</div>
      </div>
      {% endif %}
      {% if judge_usage and judge_usage.call_count %}
      <div class="meta-card">
        <div class="meta-label">EvalView Judge{% if judge_usage.model %} ({{ judge_usage.model }}){% endif %}</div>
        <div class="meta-value">{% if judge_usage.total_cost > 0 %}${{ judge_usage.total_cost }}{% elif judge_usage.is_free %}FREE{% else %}$0{% endif %}</div>
        <div class="meta-sub">{{ '{:,}'.format(judge_usage.total_tokens) }} tokens across {{ judge_usage.call_count }} judge call{% if judge_usage.call_count != 1 %}s{% endif %}</div>
      </div>
      <div class="meta-card">
        <div class="meta-label">Token Breakdown</div>
        <div class="meta-value">in {{ '{:,}'.format(judge_usage.input_tokens) }} / out {{ '{:,}'.format(judge_usage.output_tokens) }}</div>
        <div class="meta-sub">{% if judge_usage.pricing %}{{ judge_usage.pricing }}{% else %}Separate from agent trace cost{% endif %}</div>
      </div>
      {% endif %}
    </div>
    {% endif %}

    <!-- Score chart (full width) — only show separately when trends exist (otherwise it's already in the meta-row above) -->
    {% if not dashboard or dashboard.test_sparklines or dashboard.pass_trend %}
    <div class="card">
      <div class="card-title">Score per Test</div>
      <div class="chart-wrap" style="height:{{ [kpis.scores|length * 40 + 24, 120]|max }}px"><canvas id="bars"></canvas></div>
    </div>
    {% endif %}

    <!-- Cost table -->
    <div class="card">
      <div class="card-title">Execution Cost per Query</div>
      <table class="ev-table">
        {% set has_tokens = traces | selectattr('tokens') | list | length > 0 %}
        <thead><tr><th>Test</th><th>Model</th><th>Trace Cost</th>{% if has_tokens %}<th>Tokens</th>{% endif %}<th>Latency</th><th>Score</th></tr></thead>
        <tbody>
          {% for t in traces %}<tr>
            <td style="font-weight:600">{{ t.name }}</td>
            <td class="mono" style="color:var(--text-4)">{{ t.model }}</td>
            <td class="mono num" style="color:{% if t.cost == '$0' %}var(--text-4){% else %}var(--blue-bright){% endif %}">{{ t.cost }}</td>
            {% if has_tokens %}<td class="mono" style="color:var(--text-3)">{{ t.tokens or '—' }}</td>{% endif %}
            <td style="color:var(--text-3)">{{ t.latency }}</td>
            <td class="num" style="color:{% if t.score >= 80 %}var(--green-bright){% elif t.score >= 60 %}var(--yellow-bright){% else %}var(--red-bright){% endif %}">{{ t.score }}</td>
          </tr>{% endfor %}
          <tr style="background:rgba(0,0,0,.08)">
            <td style="font-weight:800">Total</td><td style="color:var(--text-4)">—</td>
            <td class="mono num" style="color:var(--blue-bright)">${{ kpis.total_cost }}</td>
            <td colspan="{{ 3 if has_tokens else 2 }}" style="font-size:11px;color:var(--text-4)">avg ${{ '%.6f'|format(kpis.total_cost / kpis.total) if kpis.total else '0' }} per query</td>
          </tr>
        </tbody>
      </table>
      <div style="margin-top:10px;font-size:11px;color:var(--text-4);line-height:1.5">
        Trace cost comes from the agent execution trace only. Mock or non-metered tools will show <code style="background:rgba(255,255,255,.04);padding:2px 6px;border-radius:3px;font-family:var(--mono);font-size:11px;border:1px solid var(--border)">$0</code> even when EvalView used a separate judge or local model during evaluation.
        {% if judge_usage and judge_usage.call_count %} This check also used {{ judge_usage.call_count }} EvalView judge call{% if judge_usage.call_count != 1 %}s{% endif %} ({{ judge_usage.total_tokens }} tokens).{% endif %}
      </div>
    </div>
    {% else %}
    <div class="empty"><span class="empty-icon">📊</span>No results to display</div>
    {% endif %}
  </div>

  <!-- ═══════════ TRACE ═══════════ -->
  <div id="p-trace" class="panel {% if default_tab == 'trace' %}on{% endif %}">
    {% if traces %}{% for t in traces %}
      <div class="item">
        <div class="item-head" onclick="tog('tr{{ loop.index }}',this)">
          <span class="badge {% if t.passed %}b-green{% else %}b-red{% endif %}">{% if t.passed %}✓{% else %}✗{% endif %}</span>
          <span class="item-name">{{ t.name }}</span>
          <div class="item-meta">
            <span class="mc" style="color:{% if t.score >= 80 %}var(--green-bright){% elif t.score >= 60 %}var(--yellow-bright){% else %}var(--red-bright){% endif %}">{{ t.score }}/100</span>
            {% if t.cost != "$0" %}<span class="mc">💰 {{ t.cost }}</span>{% endif %}
            <span class="mc">⚡ {{ t.latency }}</span>
            {% if t.tokens %}<span class="mc">{{ t.tokens }}</span>{% endif %}
            <span class="mc">🧠 {{ t.model }}</span>
          </div>
          <span class="chevron">▾</span>
        </div>
        <div id="tr{{ loop.index }}" class="item-body" {% if traces|length > 4 and not loop.first %}style="display:none"{% endif %}>
          <div style="display:flex;flex-wrap:wrap;gap:4px;margin-bottom:12px">
            <span class="badge b-blue">Model: {{ t.model }}</span>
            {% if t.input_tokens or t.output_tokens %}<span class="badge b-blue">in {{ '{:,}'.format(t.input_tokens) }} / out {{ '{:,}'.format(t.output_tokens) }} tokens</span>{% if t.cost != "$0" %}<span class="badge b-blue">{{ t.cost }}</span>{% endif %}{% endif %}
            {% if not t.input_tokens and not t.output_tokens and t.cost != "$0" %}<span class="badge b-yellow">{{ t.cost }} (adapter-reported, no token data)</span>{% endif %}
            {% if t.baseline_created and t.baseline_created != 'Unknown' %}<span class="badge b-purple">Baseline: {{ t.baseline_created }}</span>{% endif %}
            {% if t.baseline_model and t.baseline_model != 'Unknown' %}<span class="badge b-yellow">Baseline model: {{ t.baseline_model }}</span>{% endif %}
          </div>
          {% if t.tool_accuracy is not none or t.output_quality is not none %}
          <div style="background:rgba(255,255,255,.02);border:1px solid var(--border);border-radius:var(--r-xs);padding:10px 14px;margin-bottom:12px;font-size:12px">
            <div style="font-size:10px;font-weight:700;text-transform:uppercase;letter-spacing:.06em;color:var(--text-4);margin-bottom:8px">Score Breakdown</div>
            <div style="display:flex;gap:16px;flex-wrap:wrap;align-items:center">
              {% if t.tool_accuracy is not none %}<div><span style="color:var(--text-4)">Tools</span> <span style="font-weight:700;color:{% if t.tool_accuracy >= 80 %}var(--green-bright){% elif t.tool_accuracy >= 50 %}var(--yellow-bright){% else %}var(--red-bright){% endif %}">{{ t.tool_accuracy }}%</span> <span style="color:var(--text-4);font-size:10px">× {{ t.w_tool }}%</span></div>{% endif %}
              {% if t.output_quality is not none %}<div><span style="color:var(--text-4)">Output</span> <span style="font-weight:700;color:{% if t.output_quality >= 80 %}var(--green-bright){% elif t.output_quality >= 50 %}var(--yellow-bright){% else %}var(--red-bright){% endif %}">{{ t.output_quality }}/100</span> <span style="color:var(--text-4);font-size:10px">× {{ t.w_output }}%</span></div>{% endif %}
              {% if t.sequence_correct is not none %}<div><span style="color:var(--text-4)">Sequence</span> <span style="font-weight:700;color:{% if t.sequence_correct %}var(--green-bright){% else %}var(--red-bright){% endif %}">{% if t.sequence_correct %}Correct{% else %}Wrong{% endif %}</span> <span style="color:var(--text-4);font-size:10px">× {{ t.w_seq }}%</span></div>{% endif %}
              <div style="border-left:1px solid var(--border);padding-left:16px"><span style="color:var(--text-4)">=</span> <span style="font-weight:800;font-size:14px;color:{% if t.score >= 80 %}var(--green-bright){% elif t.score >= 60 %}var(--yellow-bright){% else %}var(--red-bright){% endif %}">{{ t.score }}/100</span></div>
            </div>
            {% if t.output_rationale %}<div style="margin-top:8px;font-size:11px;color:var(--text-3);border-top:1px solid var(--border);padding-top:8px">{{ t.output_rationale }}</div>{% endif %}
          </div>{% endif %}
          {% if t.query %}
          <div style="background:rgba(37,99,235,.05);border:1px solid rgba(37,99,235,.12);border-radius:var(--r-xs);padding:9px 12px;margin-bottom:12px;font-size:13px;color:var(--text-2)">
            <span style="font-size:10px;font-weight:700;text-transform:uppercase;letter-spacing:.06em;color:var(--text-4);margin-right:6px">Query</span>{{ t.query }}
          </div>{% endif %}
          {% if t.failure_reasons %}
          <div style="background:rgba(239,68,68,.06);border:1px solid rgba(239,68,68,.18);border-radius:var(--r-xs);padding:10px 14px;margin-bottom:12px">
            <div style="font-size:10px;font-weight:700;text-transform:uppercase;letter-spacing:.06em;color:var(--red-bright);margin-bottom:6px">Why it failed</div>
            <ul style="margin:0;padding-left:18px;font-size:12px;color:var(--text-2)">{% for reason in t.failure_reasons %}<li style="margin-bottom:3px">{{ reason }}</li>{% endfor %}</ul>
          </div>{% endif %}
          {% if t.has_steps %}<div class="mermaid-box"><div class="mermaid">{{ t.diagram }}</div></div>
          {% else %}<div style="text-align:center;padding:18px 0;font-size:12px;color:var(--text-4)">◎ Direct response — no tools invoked</div>{% endif %}
          {% if t.turns %}
          <div class="chat-container">
            <div class="chat-header">Conversation Turns</div>
            <div class="chat-messages">
            {% for turn in t.turns %}
              <div class="chat-meta right">Turn {{ turn.index }}{% if turn.tools %} · {% for tool in turn.tools %}<span class="chat-tool-tag">{{ tool }}</span> {% endfor %}{% endif %} · ⚡ {{ turn.latency_ms|round(1) }}ms · 💰 ${{ '%.6f'|format(turn.cost) if turn.cost else '0' }}</div>
              <div class="chat-bubble user">{{ turn.query }}</div>
              {% if turn.output %}<div class="chat-bubble agent">{{ turn.output }}</div>{% endif %}
              {% if turn.evaluation %}
              <div class="chat-eval {% if turn.evaluation.passed %}pass{% else %}fail{% endif %}">
                <span style="font-weight:700">{% if turn.evaluation.passed %}✅ PASS{% else %}❌ FAIL{% endif %}</span>
                {% if turn.evaluation.tool_accuracy is not none %}<span style="margin-left:6px;opacity:.7">Tool accuracy: {{ (turn.evaluation.tool_accuracy * 100)|round(0) }}%</span>{% endif %}
                {% if turn.evaluation.forbidden_violations %}<span style="margin-left:6px;color:var(--red-bright)">Forbidden: {{ turn.evaluation.forbidden_violations|join(', ') }}</span>{% endif %}
                {% if turn.evaluation.contains_failed %}<span style="margin-left:6px;color:var(--red-bright)">Missing: {{ turn.evaluation.contains_failed|join(', ') }}</span>{% endif %}
                {% if turn.evaluation.not_contains_failed %}<span style="margin-left:6px;color:var(--red-bright)">Prohibited: {{ turn.evaluation.not_contains_failed|join(', ') }}</span>{% endif %}
              </div>{% endif %}
            {% endfor %}</div>
          </div>{% endif %}
          {% if t.hallucination or t.safety or t.pii or t.forbidden_tools %}
          <div style="display:flex;flex-wrap:wrap;gap:6px;margin-top:10px">
            {% if t.hallucination %}{% if t.hallucination.has_hallucination %}<span class="badge b-red" title="Extracts factual claims from the agent response, then verifies each claim against tool outputs. Score = supported claims / total claims.">🔮 Hallucination detected · {{ (t.hallucination.confidence * 100)|round(0)|int }}%{% if t.hallucination.details %} · {{ t.hallucination.details.split('\n')[0]|replace('Faithfulness: ', '') }}{% endif %}{% if judge_usage and judge_usage.model %} · {{ judge_usage.model }}{% endif %}</span>{% else %}<span class="badge b-green" title="Extracts factual claims from the agent response, then verifies each claim against tool outputs. Score = supported claims / total claims.">🔮 No hallucination{% if t.hallucination.details %} · {{ t.hallucination.details.split('\n')[0]|replace('Faithfulness: ', '') }}{% endif %}{% if judge_usage and judge_usage.model %} · {{ judge_usage.model }}{% endif %}</span>{% endif %}{% endif %}
            {% if t.safety %}{% if t.safety.is_safe %}<span class="badge b-green">🛡 Safe</span>{% else %}<span class="badge b-red">🛡 Unsafe: {{ t.safety.categories|join(', ') }}</span>{% endif %}{% endif %}
            {% if t.pii %}{% if t.pii.has_pii %}<span class="badge b-yellow">🔒 PII detected</span>{% else %}<span class="badge b-green">🔒 No PII</span>{% endif %}{% endif %}
            {% if t.forbidden_tools %}{% if t.forbidden_tools.violations %}<span class="badge b-red">⛔ Forbidden: {{ t.forbidden_tools.violations|join(', ') }}</span>{% else %}<span class="badge b-green">⛔ No violations</span>{% endif %}{% endif %}
          </div>
          {% if t.hallucination and t.hallucination.has_hallucination and t.hallucination.details %}<div style="background:rgba(168,85,247,.06);border:1px solid rgba(168,85,247,.15);border-radius:var(--r-xs);padding:9px 12px;margin-top:8px;font-size:11px;color:var(--text-3)"><span style="font-weight:600;color:var(--text-2)">Unsupported claims:</span> {{ t.hallucination.details[:500] }}{% if t.hallucination.details|length > 500 %}...{% endif %}</div>{% endif %}
          {% endif %}
          {% if t.output and not t.turns %}
          <div style="background:rgba(16,185,129,.04);border:1px solid rgba(16,185,129,.1);border-radius:var(--r-xs);padding:9px 12px;margin-top:12px;font-size:13px;color:var(--text-2)">
            <span style="font-size:10px;font-weight:700;text-transform:uppercase;letter-spacing:.06em;color:var(--text-4);margin-right:6px">Response</span>{{ t.output[:300] }}{% if t.output|length > 300 %}...{% endif %}
          </div>{% endif %}
        </div>
      </div>
    {% endfor %}{% else %}<div class="empty"><span class="empty-icon">🔍</span>No trace data available</div>{% endif %}
  </div>

  <!-- ═══════════ DIFFS ═══════════ -->
  {% if diff_rows %}
  <div id="p-diffs" class="panel {% if default_tab == 'diffs' %}on{% endif %}">
    {% for d in diff_rows %}
      <div class="diff-item">
        <div class="diff-head" style="cursor:pointer" onclick="tog('df{{ loop.index }}',this)">
          {% if d.status == 'regression' %}<span class="badge b-red">⬇ Regression</span>{% elif d.status == 'tools_changed' %}<span class="badge b-yellow">⚠ Tools Changed</span>{% elif d.status == 'output_changed' %}<span class="badge b-purple">~ Output Changed</span>{% else %}<span class="badge b-green">✓ Passed</span>{% endif %}
          {% if healing %}
            {% for hr in healing.results if hr.test_name == d.name %}
              {% if hr.healed %}<span class="badge b-green">⚡ Healed</span>
              {% elif hr.proposed %}<span class="badge b-cyan">◈ Proposed</span>
              {% elif hr.diagnosis.action == 'blocked' %}<span class="badge b-red">✗ Blocked</span>
              {% elif hr.diagnosis.action == 'flag_review' %}<span class="badge b-yellow">⚠ Review</span>{% endif %}
            {% endfor %}
          {% endif %}
          {% if d.model_changed %}<span class="badge b-red">Model ID changed</span>{% elif d.runtime_fingerprint_changed %}<span class="badge b-yellow">Runtime fingerprint changed</span>{% endif %}
          <span class="diff-name">{{ d.name }}</span>
          {% if d.actual_score is not none %}<span class="mc" title="Weighted score: tool accuracy (30%) + output quality (50%) + sequence correctness (20%). Baseline → Current." style="color:{% if d.actual_score >= 80 %}var(--green-bright){% elif d.actual_score >= 60 %}var(--yellow-bright){% else %}var(--red-bright){% endif %}">{{ d.baseline_score }} → {{ d.actual_score }}</span>{% endif %}
          {% if d.score_delta != 0 %}<span class="badge {% if d.score_delta > 0 %}b-green{% else %}b-red{% endif %}" title="Score change from baseline snapshot">{% if d.score_delta > 0 %}+{% endif %}{{ d.score_delta }}</span>{% endif %}
          <span class="sim" title="Exact word-for-word match between baseline and current output">lexical <span class="sim-track"><span class="sim-fill {% if d.similarity >= 80 %}hi{% elif d.similarity >= 50 %}mid{% else %}lo{% endif %}" style="width:{{ d.similarity }}%"></span></span> <b style="color:{% if d.similarity >= 80 %}var(--green-bright){% elif d.similarity >= 50 %}var(--yellow-bright){% else %}var(--red-bright){% endif %}">{{ d.similarity }}%</b></span>
          {% if d.semantic_similarity is not none %}<span class="sim" title="Meaning similarity — high means same intent even if wording changed">semantic <span class="sim-track"><span class="sim-fill {% if d.semantic_similarity >= 80 %}hi{% elif d.semantic_similarity >= 50 %}mid{% else %}lo{% endif %}" style="width:{{ d.semantic_similarity }}%"></span></span> <b style="color:{% if d.semantic_similarity >= 80 %}var(--green-bright){% elif d.semantic_similarity >= 50 %}var(--yellow-bright){% else %}var(--red-bright){% endif %}">{{ d.semantic_similarity }}%</b></span>{% endif %}
          {% if d.confidence_label and d.confidence_label != 'insufficient_history' %}<span class="confidence-badge conf-{{ d.confidence_label }}" title="Statistical confidence that this change is a real signal vs. normal LLM variance">{{ d.confidence_pct|int }}% confidence</span>{% elif d.confidence_label == 'insufficient_history' %}<span class="confidence-badge conf-insufficient">needs history</span>{% endif %}
          <span class="chevron">▾</span>
        </div>
        <div id="df{{ loop.index }}" {% if d.status == 'passed' %}style="display:none"{% endif %}>
        {% if d.golden_tools or d.actual_tools %}
        <div class="pipeline">
          <div class="pipeline-row"><span class="pipeline-label">Baseline</span>{% for t in d.golden_tools %}<span class="pipe-step {% if t not in d.actual_tools %}removed{% else %}match{% endif %}">{{ t }}</span>{% endfor %}{% if not d.golden_tools %}<span style="font-size:11px;color:var(--text-4);font-style:italic">No tools</span>{% endif %}</div>
          <div class="pipeline-row"><span class="pipeline-label">Current</span>{% for t in d.actual_tools %}<span class="pipe-step {% if t not in d.golden_tools %}added{% else %}match{% endif %}">{{ t }}</span>{% endfor %}{% if not d.actual_tools %}<span style="font-size:11px;color:var(--text-4);font-style:italic">No tools</span>{% endif %}</div>
        </div>{% endif %}
        {% if d.golden_runtime_fingerprint and d.actual_runtime_fingerprint and d.golden_runtime_fingerprint != d.actual_runtime_fingerprint %}
        <div style="padding:0 18px 12px;font-size:12px;color:var(--text-2)">
          <span style="color:var(--text-4)">Runtime fingerprint:</span>
          <span class="mono" style="color:var(--text-3)">{{ d.golden_runtime_fingerprint }}</span>
          <span style="color:var(--text-4)"> → </span>
          <span class="mono" style="color:var(--text)">{{ d.actual_runtime_fingerprint }}</span>
        </div>
        {% endif %}
        <div class="diff-cols">
          <div class="diff-col"><div class="col-title">Baseline Output</div><div class="outbox">{{ d.golden_out }}</div></div>
          <div class="diff-col"><div class="col-title">Current Output</div><div class="outbox">{{ d.actual_out }}</div>{% if d.diff_lines %}<div class="difflines">{% for line in d.diff_lines %}{% if line.startswith('+') %}<div class="a">{{ line }}</div>{% elif line.startswith('-') %}<div class="r">{{ line }}</div>{% else %}<div>{{ line }}</div>{% endif %}{% endfor %}</div>{% endif %}</div>
        </div>
        {% if d.param_diffs %}
        <div style="padding:12px 18px;border-top:1px solid var(--border)">
          <div class="col-title" style="margin-bottom:8px">Parameter Changes</div>
          <table class="param-table"><thead><tr><th>Step</th><th>Tool</th><th>Parameter</th><th>Baseline</th><th>Current</th><th style="text-align:center">Match</th></tr></thead>
            <tbody>{% for p in d.param_diffs %}<tr>
              <td style="color:var(--text-4)">{{ p.step }}</td><td style="font-family:var(--mono);color:var(--blue-bright)">{{ p.tool }}</td><td style="font-weight:600">{{ p.param }}</td>
              <td style="font-family:var(--mono);font-size:11px;{% if p.type == 'missing' %}color:var(--red-bright){% else %}color:var(--text-3){% endif %}">{{ p.golden or '—' }}</td>
              <td style="font-family:var(--mono);font-size:11px;{% if p.type == 'added' %}color:var(--green-bright){% else %}color:var(--text-3){% endif %}">{{ p.actual or '—' }}</td>
              <td style="text-align:center;font-weight:600;color:{% if p.type == 'added' %}var(--green-bright){% elif p.type == 'missing' %}var(--red-bright){% elif p.similarity is not none %}{% if p.similarity >= 80 %}var(--green-bright){% elif p.similarity >= 50 %}var(--yellow-bright){% else %}var(--red-bright){% endif %}{% else %}var(--yellow-bright){% endif %}">{% if p.type == 'added' %}+new{% elif p.type == 'missing' %}-gone{% elif p.similarity is not none %}{{ p.similarity }}%{% else %}~{% endif %}</td>
            </tr>{% endfor %}</tbody></table>
        </div>{% endif %}
        {% if d.golden_diagram or d.actual_diagram %}
        <div style="padding:10px 18px;border-top:1px solid var(--border)">
          <div style="cursor:pointer;display:flex;align-items:center;gap:8px;padding:4px 0" onclick="togTraj(this)">
            <span class="chevron">▾</span>
            <span style="font-size:12px;font-weight:700;color:var(--text-2);text-transform:uppercase;letter-spacing:.06em">Trajectory Comparison</span>
          </div>
          <div class="traj-grid" style="display:none" data-golden="{{ d.golden_diagram or 'sequenceDiagram\n    Note over Agent: No trace data' }}" data-actual="{{ d.actual_diagram or 'sequenceDiagram\n    Note over Agent: No trace data' }}">
            <div class="traj-col"><div class="col-title">Baseline</div><div class="mermaid-box" style="min-height:100px"><div class="mermaid-lazy"></div></div></div>
            <div class="traj-col"><div class="col-title">Current</div><div class="mermaid-box" style="min-height:100px"><div class="mermaid-lazy"></div></div></div>
          </div>
        </div>{% endif %}
        {% if d.accept_suggestion %}
        <div class="accept-box {% if not d.accept_suggestion.score_improved %}neutral{% endif %}">
          <div style="font-size:12px;font-weight:600;margin-bottom:6px;color:{% if d.accept_suggestion.score_improved %}var(--green-bright){% else %}var(--yellow-bright){% endif %}">
            💡 {% if d.accept_suggestion.score_improved %}Score improved — this looks intentional{% else %}Score is stable — this may be intentional{% endif %}
          </div>
          <div style="font-size:11px;color:var(--text-2);display:flex;flex-direction:column;gap:4px">
            <div>Accept: <code>{{ d.accept_suggestion.command }}</code></div>
            <div>Preview: <code>{{ d.accept_suggestion.preview_command }}</code></div>
          </div>
        </div>
        {% endif %}
        {% if healing %}
          {% for hr in healing.results if hr.test_name == d.name %}
          <div style="padding:12px 18px;border-top:1px solid var(--border);font-size:12px;color:var(--text-2)">
            <div class="col-title" style="margin-bottom:8px">Healing Decision</div>
            <div style="display:flex;flex-wrap:wrap;gap:6px;margin-bottom:8px">
              <span class="badge b-blue">{{ hr.diagnosis.trigger }}</span>
              {% if hr.attempted %}<span class="badge b-blue">retry attempted</span>{% else %}<span class="badge b-yellow">no retry attempted</span>{% endif %}
              {% if hr.retry_status %}<span class="badge b-blue">retry status: {{ hr.retry_status }}</span>{% endif %}
            </div>
            <div>{{ hr.diagnosis.reason }}</div>
            {% if hr.variant_saved %}
            <div style="margin-top:6px;color:var(--text-3)">Saved variant: <code style="background:rgba(255,255,255,.04);padding:2px 6px;border-radius:3px;font-family:var(--mono);font-size:11px;border:1px solid var(--border)">{{ hr.variant_saved }}</code></div>
            {% endif %}
          </div>
          {% endfor %}
        {% endif %}
        </div>
      </div>
    {% endfor %}
  </div>
  {% endif %}

  <!-- ═══════════ TIMELINE ═══════════ -->
  <div id="p-timeline" class="panel {% if default_tab == 'timeline' %}on{% endif %}">
    {% if timeline %}
    <!-- Timeline KPI strip -->
    <div class="kpi-strip" style="margin-bottom:12px">
      <div class="kpi-item">
        <span class="kpi-val">{{ timeline|length }}</span>
        <span class="kpi-label">steps</span>
      </div>
      <div class="kpi-item">
        <span class="kpi-val" style="color:var(--blue-bright)">{{ kpis.avg_latency_ms|int }}ms</span>
        <span class="kpi-label">avg latency</span>
      </div>
      <div class="kpi-item">
        <span class="kpi-val" style="color:var(--blue-bright)">${{ kpis.total_cost }}</span>
        <span class="kpi-label">total cost</span>
      </div>
      <div class="kpi-item">
        <span class="kpi-val">{{ kpis.total }}</span>
        <span class="kpi-label">tests</span>
      </div>
    </div>
    <div style="display:grid;grid-template-columns:1fr 1fr;gap:12px">
      <div class="card" style="margin-bottom:0">
        <div class="card-title">Step Latencies</div>
        <div style="position:relative;height:{{ [timeline|length * 36 + 40, 160]|max }}px"><canvas id="tlChart"></canvas></div>
      </div>
      <div class="card" style="margin-bottom:0">
        <div class="card-title">Step Cost</div>
        <div style="position:relative;height:{{ [timeline|length * 36 + 40, 160]|max }}px"><canvas id="tlCostChart"></canvas></div>
      </div>
    </div>
    {% else %}<div class="empty"><span class="empty-icon">⏱</span>No step timing data</div>{% endif %}
  </div>

  <!-- ═══════════ COMPARE ═══════════ -->
  {% if compare %}
  <div id="p-compare" class="panel">
    <div class="card"><div class="card-title">Pass Rate Across Runs</div><div class="chart-wrap" style="height:220px"><canvas id="cmpPassRate"></canvas></div></div>
    <div class="card"><div class="card-title">Avg Score Across Runs</div><div class="chart-wrap" style="height:220px"><canvas id="cmpScore"></canvas></div></div>
    <div class="card"><div class="card-title">Run Summary</div>
      <table class="ev-table"><thead><tr>{% for lbl in compare.labels %}<th>{{ lbl }}</th>{% endfor %}</tr></thead>
        <tbody><tr>{% for run in compare.runs %}<td>
          <div style="font-size:24px;font-weight:900;letter-spacing:-.04em;color:{% if run.pass_rate >= 80 %}var(--green-bright){% else %}var(--red-bright){% endif %}">{{ run.pass_rate }}%</div>
          <div style="font-size:11px;color:var(--text-4);margin-top:2px">{{ run.passed }}/{{ run.total }} · avg {{ run.avg_score }}/100</div>
        </td>{% endfor %}</tr></tbody></table>
    </div>
  </div>{% endif %}
</main>

<script>
mermaid.initialize({startOnLoad:true,theme:'dark',securityLevel:'loose',useMaxWidth:true,
  themeVariables:{darkMode:true,background:'transparent',primaryColor:'rgba(37,99,235,.1)',primaryTextColor:'#e2e8f0',primaryBorderColor:'rgba(37,99,235,.25)',lineColor:'rgba(100,116,139,.3)',secondaryColor:'rgba(16,185,129,.06)',tertiaryColor:'rgba(6,182,212,.06)',noteBkgColor:'rgba(37,99,235,.05)',noteTextColor:'#94a3b8',noteBorderColor:'rgba(37,99,235,.15)',actorBkg:'rgba(37,99,235,.08)',actorBorder:'rgba(37,99,235,.2)',actorTextColor:'#e2e8f0',signalColor:'#64748b',signalTextColor:'#cbd5e1'},
  sequence:{useMaxWidth:true,width:180,wrap:false,actorFontFamily:'Inter,sans-serif',noteFontFamily:'Inter,sans-serif',messageFontFamily:'Inter,sans-serif',actorFontSize:15,messageFontSize:14,noteFontSize:13,boxTextMargin:12,mirrorActors:false,messageAlign:'center',actorMargin:50,bottomMarginAdj:4,diagramMarginX:20,diagramMarginY:16}
});
function show(id,btn){document.querySelectorAll('.panel').forEach(p=>p.classList.remove('on'));document.querySelectorAll('.tab').forEach(t=>t.classList.remove('on'));document.getElementById('p-'+id).classList.add('on');btn.classList.add('on')}
function tog(id,head){const el=document.getElementById(id);const o=el.style.display!=='none';el.style.display=o?'none':'block';head.querySelector('.chevron').style.transform=o?'':'rotate(180deg)'}
function togTraj(trigger){const grid=trigger.nextElementSibling;const open=grid.style.display!=='none';grid.style.display=open?'none':'grid';trigger.querySelector('.chevron').style.transform=open?'':'rotate(180deg)';if(!open&&!grid.dataset.rendered){grid.dataset.rendered='1';const divs=grid.querySelectorAll('.mermaid-lazy');const src=[grid.dataset.golden,grid.dataset.actual];divs.forEach(function(d,i){if(src[i]){d.classList.add('mermaid');d.textContent=src[i];mermaid.init(undefined,d)}})}}

{% if kpis %}
(function(){
  const scores={{ kpis.scores|tojson }},names={{ kpis.test_names|tojson }};
  const tc='rgba(100,116,139,.6)',gc='rgba(255,255,255,.025)';
  const tt={backgroundColor:'rgba(6,11,24,.95)',borderColor:'rgba(51,65,85,.5)',borderWidth:1,titleFont:{family:'Inter',weight:'700',size:11},bodyFont:{family:'Inter',size:11},padding:8,cornerRadius:6};

  const sorted=names.map((n,i)=>({name:n,score:scores[i]})).sort((a,b)=>b.score-a.score);
  /* Warning stripes for low scores */
  const barBg=sorted.map(s=>{
    if(s.score>=80) return 'rgba(16,185,129,.35)';
    if(s.score>=60) return 'rgba(245,158,11,.35)';
    return 'rgba(239,68,68,.35)';
  });
  const barBorder=sorted.map(s=>{
    if(s.score>=80) return 'rgba(16,185,129,.55)';
    if(s.score>=60) return 'rgba(245,158,11,.55)';
    return 'rgba(239,68,68,.55)';
  });
  new Chart(document.getElementById('bars'),{type:'bar',
    data:{labels:sorted.map(s=>s.name),datasets:[{label:'Score',data:sorted.map(s=>s.score),backgroundColor:barBg,borderColor:barBorder,borderWidth:1,borderRadius:3,borderSkipped:false,barPercentage:.55,categoryPercentage:.8}]},
    options:{indexAxis:'y',responsive:true,maintainAspectRatio:false,
      scales:{x:{min:0,max:100,grid:{color:gc},ticks:{color:tc,font:{family:'Inter',size:9},stepSize:25},border:{display:false}},y:{grid:{display:false},ticks:{color:'rgba(203,213,225,.7)',font:{family:'Inter',size:11,weight:'600'},padding:4,mirror:false},border:{display:false},afterFit:function(axis){var maxLen=0;sorted.forEach(function(s){var w=s.name.length*7;if(w>maxLen)maxLen=w});axis.width=Math.min(Math.max(maxLen,140),280)}}},
      plugins:{legend:{display:false},tooltip:{...tt,callbacks:{label:ctx=>` Score: ${ctx.raw}/100`}}}}});
})();
{% endif %}

{% if timeline %}
(function(){
  const tl={{ timeline|tojson }};if(!tl.length)return;
  const labels=tl.map(r=>r.label||(r.test+' \u203a '+r.tool));const vals=tl.map(r=>r.latency||0);const costs=tl.map(r=>r.cost||0);
  const maxLat=Math.max(...vals,0);
  const tt={backgroundColor:'rgba(6,11,24,.95)',borderColor:'rgba(51,65,85,.5)',borderWidth:1,titleFont:{family:'Inter',weight:'700'},bodyFont:{family:'Inter'},padding:8,cornerRadius:6};
  /* Color palette per test — distinct hues */
  const palette=[
    {bg:'rgba(37,99,235,.4)',border:'rgba(37,99,235,.65)'},
    {bg:'rgba(16,185,129,.4)',border:'rgba(16,185,129,.65)'},
    {bg:'rgba(245,158,11,.4)',border:'rgba(245,158,11,.65)'},
    {bg:'rgba(168,85,247,.4)',border:'rgba(168,85,247,.65)'},
    {bg:'rgba(6,182,212,.4)',border:'rgba(6,182,212,.65)'},
    {bg:'rgba(239,68,68,.4)',border:'rgba(239,68,68,.65)'},
    {bg:'rgba(236,72,153,.4)',border:'rgba(236,72,153,.65)'},
    {bg:'rgba(132,204,22,.4)',border:'rgba(132,204,22,.65)'},
  ];
  const tests=[...new Set(tl.map(r=>r.test))];
  const testIdx=Object.fromEntries(tests.map((t,i)=>[t,i%palette.length]));
  const colors=tl.map(r=>r.success?palette[testIdx[r.test]].bg:'rgba(239,68,68,.45)');
  const borders=tl.map(r=>r.success?palette[testIdx[r.test]].border:'rgba(239,68,68,.65)');
  const chartOpts={indexAxis:'y',responsive:true,maintainAspectRatio:false,scales:{x:{suggestedMax:maxLat>0?maxLat*1.15:1,grid:{color:'rgba(255,255,255,.025)'},ticks:{color:'rgba(100,116,139,.5)',font:{family:'Inter',size:9},callback:v=>v+'ms'},border:{display:false}},y:{grid:{display:false},ticks:{color:'rgba(203,213,225,.6)',font:{family:'Inter',size:10,weight:'500'}},border:{display:false}}},plugins:{legend:{display:false},tooltip:{...tt,callbacks:{label:ctx=>` ${ctx.raw}ms`,afterLabel:ctx=>` Cost: $${(costs[ctx.dataIndex]||0).toFixed(6)}`,title:ctx=>ctx[0].label}}}};
  new Chart(document.getElementById('tlChart'),{type:'bar',data:{labels,datasets:[{label:'ms',data:vals,backgroundColor:colors,borderColor:borders,borderWidth:1,borderRadius:3,borderSkipped:false,barPercentage:.6}]},options:chartOpts});
  /* Cost chart */
  const maxCost=Math.max(...costs,0.000001);
  new Chart(document.getElementById('tlCostChart'),{type:'bar',data:{labels,datasets:[{label:'$',data:costs,backgroundColor:colors,borderColor:borders,borderWidth:1,borderRadius:3,borderSkipped:false,barPercentage:.6}]},options:{indexAxis:'y',responsive:true,maintainAspectRatio:false,scales:{x:{suggestedMax:maxCost>0?maxCost*1.15:0.001,grid:{color:'rgba(255,255,255,.025)'},ticks:{color:'rgba(100,116,139,.5)',font:{family:'Inter',size:9},callback:v=>'$'+v.toFixed(4)},border:{display:false}},y:{grid:{display:false},ticks:{color:'rgba(203,213,225,.6)',font:{family:'Inter',size:10,weight:'500'}},border:{display:false}}},plugins:{legend:{display:false},tooltip:{...tt,callbacks:{label:ctx=>` $${ctx.raw.toFixed(6)}`,title:ctx=>ctx[0].label}}}}});
})();
{% endif %}

{% if dashboard and dashboard.test_sparklines %}
(function(){
  const canvas=document.getElementById('trendChart');
  if(!canvas)return;
  const sparklines={{ dashboard.test_sparklines|tojson }};
  const passTrend={{ dashboard.pass_trend|tojson }};
  const palette=[
    {bg:'rgba(37,99,235,.15)',border:'rgba(37,99,235,.7)'},
    {bg:'rgba(16,185,129,.15)',border:'rgba(16,185,129,.7)'},
    {bg:'rgba(245,158,11,.15)',border:'rgba(245,158,11,.7)'},
    {bg:'rgba(168,85,247,.15)',border:'rgba(168,85,247,.7)'},
    {bg:'rgba(6,182,212,.15)',border:'rgba(6,182,212,.7)'},
    {bg:'rgba(239,68,68,.15)',border:'rgba(239,68,68,.7)'},
  ];
  const maxLen=Math.max(...sparklines.map(s=>s.values.length),passTrend.length);
  const labels=Array.from({length:maxLen},(_,i)=>''+(i+1));
  const datasets=sparklines.map(function(s,i){
    const c=palette[i%palette.length];
    return {label:s.name,data:s.values,borderColor:c.border,backgroundColor:c.bg,borderWidth:2,pointRadius:3,pointHoverRadius:5,tension:.3,fill:false};
  });
  if(passTrend.length>1){
    datasets.push({label:'Overall pass rate',data:passTrend,borderColor:'rgba(255,255,255,.4)',backgroundColor:'rgba(255,255,255,.05)',borderWidth:2,borderDash:[4,4],pointRadius:2,tension:.3,fill:false});
  }
  new Chart(canvas,{type:'line',data:{labels,datasets},options:{responsive:true,maintainAspectRatio:false,
    scales:{x:{display:true,grid:{color:'rgba(255,255,255,.025)'},ticks:{color:'rgba(100,116,139,.5)',font:{family:'Inter',size:9}},title:{display:true,text:'Check #',color:'rgba(100,116,139,.5)',font:{family:'Inter',size:10}},border:{display:false}},y:{min:0,max:100,grid:{color:'rgba(255,255,255,.025)'},ticks:{color:'rgba(100,116,139,.5)',font:{family:'Inter',size:9},callback:function(v){return v+'%'}},border:{display:false}}},
    plugins:{legend:{display:true,position:'bottom',labels:{color:'rgba(203,213,225,.7)',font:{family:'Inter',size:10},boxWidth:12,padding:10}},tooltip:{backgroundColor:'rgba(6,11,24,.95)',borderColor:'rgba(51,65,85,.5)',borderWidth:1,titleFont:{family:'Inter',weight:'700',size:11},bodyFont:{family:'Inter',size:11},padding:8,cornerRadius:6,callbacks:{label:function(ctx){return ' '+ctx.dataset.label+': '+ctx.raw+'%'}}}}}});
})();
{% endif %}

{% if compare %}
(function(){
  const labels={{ compare.labels|tojson }};const pr={{ compare.runs|map(attribute='pass_rate')|list|tojson }};const as={{ compare.runs|map(attribute='avg_score')|list|tojson }};
  const tc='rgba(100,116,139,.5)',gc='rgba(255,255,255,.025)';
  const c=['rgba(37,99,235,.45)','rgba(16,185,129,.45)','rgba(239,68,68,.45)','rgba(245,158,11,.45)','rgba(6,182,212,.45)'];
  const b=['rgba(37,99,235,.65)','rgba(16,185,129,.65)','rgba(239,68,68,.65)','rgba(245,158,11,.65)','rgba(6,182,212,.65)'];
  const o={responsive:true,maintainAspectRatio:false,scales:{y:{grid:{color:gc},ticks:{color:tc},border:{display:false}},x:{grid:{display:false},ticks:{color:tc,font:{size:10}},border:{display:false}}},plugins:{legend:{display:false}}};
  new Chart(document.getElementById('cmpPassRate'),{type:'bar',data:{labels,datasets:[{label:'Pass Rate %',data:pr,backgroundColor:c.slice(0,labels.length),borderColor:b.slice(0,labels.length),borderWidth:1,borderRadius:5,borderSkipped:false}]},options:{...o,scales:{...o.scales,y:{...o.scales.y,min:0,max:100}}}});
  new Chart(document.getElementById('cmpScore'),{type:'bar',data:{labels,datasets:[{label:'Avg Score',data:as,backgroundColor:c.slice(0,labels.length),borderColor:b.slice(0,labels.length),borderWidth:1,borderRadius:5,borderSkipped:false}]},options:{...o,scales:{...o.scales,y:{...o.scales.y,min:0,max:100}}}});
})();
{% endif %}
</script>

<!-- Share bar -->
<div style="position:fixed;bottom:0;left:0;right:0;z-index:100;background:rgba(6,11,24,.9);backdrop-filter:blur(16px);-webkit-backdrop-filter:blur(16px);border-top:1px solid var(--border);padding:8px 24px;display:flex;align-items:center;justify-content:space-between;font-family:var(--font);font-size:11px;color:var(--text-4)">
  <span>Built with <a href="https://github.com/hidai25/eval-view" target="_blank" rel="noopener" style="color:var(--blue-bright);text-decoration:none;font-weight:600">EvalView</a> <span style="opacity:.25;margin:0 5px">|</span> Agent testing &amp; regression detection</span>
  <span style="display:flex;align-items:center;gap:5px">
    <a href="https://twitter.com/intent/tweet?text=Testing%20my%20AI%20agent%20with%20EvalView%20%E2%80%94%20catches%20regressions%20before%20they%20ship.%20%F0%9F%9B%A1%EF%B8%8F&url=https%3A%2F%2Fgithub.com%2Fhidai25%2Feval-view" target="_blank" rel="noopener" style="display:inline-flex;align-items:center;gap:3px;padding:4px 10px;border-radius:5px;background:rgba(29,155,240,.08);color:#1d9bf0;text-decoration:none;font-weight:600;font-size:10px;border:1px solid rgba(29,155,240,.1)"><svg width="12" height="12" viewBox="0 0 24 24" fill="currentColor"><path d="M18.244 2.25h3.308l-7.227 8.26 8.502 11.24H16.17l-5.214-6.817L4.99 21.75H1.68l7.73-8.835L1.254 2.25H8.08l4.713 6.231zm-1.161 17.52h1.833L7.084 4.126H5.117z"/></svg>Share</a>
    <a href="https://github.com/hidai25/eval-view" target="_blank" rel="noopener" style="display:inline-flex;align-items:center;gap:3px;padding:4px 10px;border-radius:5px;background:rgba(255,255,255,.03);color:var(--text-2);text-decoration:none;font-weight:600;font-size:10px;border:1px solid var(--border)"><svg width="12" height="12" viewBox="0 0 16 16" fill="currentColor"><path d="M8 0c4.42 0 8 3.58 8 8a8.013 8.013 0 0 1-5.45 7.59c-.4.08-.55-.17-.55-.38 0-.27.01-1.13.01-2.2 0-.75-.25-1.23-.54-1.48 1.78-.2 3.65-.88 3.65-3.95 0-.88-.31-1.59-.82-2.15.08-.2.36-1.02-.08-2.12 0 0-.67-.22-2.2.82-.64-.18-1.32-.27-2-.27-.68 0-1.36.09-2 .27-1.53-1.03-2.2-.82-2.2-.82-.44 1.1-.16 1.92-.08 2.12-.51.56-.82 1.28-.82 2.15 0 3.06 1.86 3.75 3.64 3.95-.23.2-.44.55-.51 1.07-.46.21-1.61.55-2.33-.66-.15-.24-.6-.83-1.23-.82-.67.01-.27.38.01.53.34.19.73.9.82 1.13.16.45.68 1.31 2.69.94 0 .67.01 1.3.01 1.49 0 .21-.15.45-.55.38A7.995 7.995 0 0 1 0 8c0-4.42 3.58-8 8-8Z"/></svg>Star</a>
  </span>
</div>
<div style="height:40px"></div>

</body>
</html>"""
