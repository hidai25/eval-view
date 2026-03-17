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

        rows.append({
            "name": test_name,
            "status": status,
            "score_delta": round(score_delta, 1),
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
    )

    abs_path = os.path.abspath(output_path)
    with open(abs_path, "w", encoding="utf-8") as f:
        f.write(html)

    if auto_open:
        webbrowser.open(f"file://{abs_path}")

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
  --bg:#0a0f1e;--bg-card:rgba(15,23,42,.65);--bg-card-solid:#0f172a;
  --border:rgba(51,65,85,.5);--border-light:rgba(71,85,105,.5);
  --text:#f1f5f9;--text-2:#94a3b8;--text-3:#64748b;--text-4:#475569;
  --r:16px;--r-sm:12px;--r-xs:8px;
  --font:'Inter',-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;
  --mono:'JetBrains Mono','Fira Code','SF Mono',monospace;
}
html{scroll-behavior:smooth;-webkit-font-smoothing:antialiased;-moz-osx-font-smoothing:grayscale}
body{font-family:var(--font);font-size:14px;line-height:1.6;color:var(--text);min-height:100vh;overflow-x:hidden;
  background:var(--bg);
}
/* ── Background: visible blobs, not invisible ── */
.bg-blobs{position:fixed;inset:0;pointer-events:none;z-index:0;overflow:hidden}
.bg-blobs .b1{position:absolute;width:700px;height:700px;border-radius:50%;top:-250px;left:-100px;background:radial-gradient(circle,rgba(37,99,235,.18),transparent 70%);filter:blur(40px)}
.bg-blobs .b2{position:absolute;width:500px;height:500px;border-radius:50%;bottom:-150px;right:-80px;background:radial-gradient(circle,rgba(16,185,129,.12),transparent 70%);filter:blur(40px)}
.bg-blobs .b3{position:absolute;width:400px;height:400px;border-radius:50%;top:40%;left:50%;transform:translateX(-50%);background:radial-gradient(circle,rgba(6,182,212,.06),transparent 70%);filter:blur(50px)}

/* ── Header: minimal chrome ── */
.header{
  position:sticky;top:0;z-index:200;
  background:rgba(10,15,30,.8);
  border-bottom:1px solid var(--border);
  backdrop-filter:blur(20px) saturate(150%);-webkit-backdrop-filter:blur(20px) saturate(150%);
  padding:0 40px;height:56px;display:flex;align-items:center;justify-content:space-between;
}
.logo{display:flex;align-items:center;gap:10px}
.logo-icon{
  width:32px;height:32px;border-radius:8px;flex-shrink:0;
  background:linear-gradient(135deg,var(--blue-bright),var(--teal));
  display:flex;align-items:center;justify-content:center;font-size:14px;
  box-shadow:0 2px 12px rgba(37,99,235,.25);
}
.logo-text{font-size:15px;font-weight:700;letter-spacing:-.02em;color:var(--text)}
.logo-sub{font-size:11px;color:var(--text-3);font-weight:400}
.header-right{display:flex;align-items:center;gap:6px}

/* ── Badges ── */
.badge{display:inline-flex;align-items:center;gap:4px;padding:4px 12px;border-radius:20px;font-size:11px;font-weight:600;letter-spacing:-.01em;white-space:nowrap}
.b-green{background:rgba(16,185,129,.15);color:var(--green-bright);border:1px solid rgba(16,185,129,.3)}
.b-red{background:rgba(239,68,68,.15);color:var(--red-bright);border:1px solid rgba(239,68,68,.3)}
.b-yellow{background:rgba(245,158,11,.15);color:var(--yellow-bright);border:1px solid rgba(245,158,11,.3)}
.b-blue{background:rgba(37,99,235,.15);color:var(--blue-bright);border:1px solid rgba(37,99,235,.3)}
.b-purple{background:rgba(13,148,136,.15);color:var(--teal-bright);border:1px solid rgba(13,148,136,.3)}

/* ── Layout ── */
.main{max-width:1200px;margin:0 auto;padding:32px 40px 80px;position:relative;z-index:1}

/* ── Tabs: full-width bar, more presence ── */
.tabbar{
  display:flex;gap:0;
  background:rgba(15,23,42,.6);border:1px solid var(--border);
  border-radius:var(--r-sm);padding:3px;margin-bottom:36px;
  backdrop-filter:blur(12px);
}
.tab{
  flex:1;text-align:center;
  background:none;border:none;color:var(--text-3);cursor:pointer;
  font:600 13px/1 var(--font);padding:11px 16px;border-radius:9px;
  transition:all .15s;letter-spacing:-.01em;
}
.tab:hover{color:var(--text-2);background:rgba(255,255,255,.04)}
.tab.on{color:#fff;background:rgba(37,99,235,.2);border:1px solid rgba(37,99,235,.35);box-shadow:0 1px 8px rgba(37,99,235,.15)}
.panel{display:none}.panel.on{display:block}

/* ══════════════════════════════════════════
   HERO SECTION — the scoreboard
   ══════════════════════════════════════════ */
.hero{
  display:grid;grid-template-columns:1fr 1fr;gap:20px;
  margin-bottom:32px;
}
.hero-pass{
  background:var(--bg-card);border:1px solid var(--border);
  border-radius:var(--r);padding:36px 40px;
  position:relative;overflow:hidden;
}
/* Colored accent glow behind the card */
.hero-pass::after{
  content:'';position:absolute;top:-40px;right:-40px;width:200px;height:200px;border-radius:50%;
  pointer-events:none;filter:blur(50px);opacity:.5;
}
.hero-pass.is-green::after{background:rgba(16,185,129,.2)}
.hero-pass.is-red::after{background:rgba(239,68,68,.2)}
.hero-pass .hero-label{font-size:11px;font-weight:700;color:var(--text-3);text-transform:uppercase;letter-spacing:.1em;margin-bottom:12px}
.hero-pass .hero-num{font-size:72px;font-weight:900;letter-spacing:-.06em;line-height:1}
.hero-pass .hero-num.green{color:var(--green-bright)}
.hero-pass .hero-num.red{color:var(--red-bright)}
.hero-pass .hero-sub{font-size:14px;color:var(--text-3);margin-top:8px;font-weight:500}
.hero-pass .hero-ring{position:absolute;top:32px;right:36px;width:80px;height:80px}
.hero-pass .hero-ring svg{transform:rotate(-90deg)}
.hero-pass .hero-ring-label{position:absolute;inset:0;display:flex;align-items:center;justify-content:center;font-size:14px;font-weight:800;color:var(--text)}

.hero-right{display:grid;grid-template-columns:1fr 1fr;gap:12px}
.hero-stat{
  background:var(--bg-card);border:1px solid var(--border);
  border-radius:var(--r);padding:20px 22px;
}
.hero-stat .stat-label{font-size:10px;font-weight:700;color:var(--text-4);text-transform:uppercase;letter-spacing:.1em;margin-bottom:10px}
.hero-stat .stat-num{font-size:28px;font-weight:800;letter-spacing:-.04em;line-height:1;color:var(--text)}
.hero-stat .stat-num.blue{color:var(--blue-bright)}
.hero-stat .stat-sub{font-size:11px;color:var(--text-4);margin-top:6px;font-weight:500;line-height:1.4}

/* ── Card (for everything else) ── */
.card{
  background:var(--bg-card);border:1px solid var(--border);
  border-radius:var(--r);padding:22px 24px;margin-bottom:16px;
  position:relative;overflow:hidden;
}
.card-title{
  font-size:11px;font-weight:700;color:var(--text-3);
  text-transform:uppercase;letter-spacing:.08em;
  margin-bottom:16px;display:flex;align-items:center;gap:8px;
}
.card-title::before{content:'';width:3px;height:12px;border-radius:2px;background:var(--blue-bright)}

/* ── Meta row (compact) ── */
.meta-row{display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-bottom:16px}
@media(max-width:900px){.meta-row{grid-template-columns:1fr}}
.meta-card{
  background:var(--bg-card);border:1px solid var(--border);
  border-radius:var(--r-sm);padding:16px 20px;
}
.meta-label{font-size:10px;font-weight:700;color:var(--text-4);text-transform:uppercase;letter-spacing:.08em;margin-bottom:6px}
.meta-value{font-size:15px;font-weight:700;color:var(--text);letter-spacing:-.01em}
.meta-sub{font-size:12px;color:var(--text-4);margin-top:3px}

/* ── Charts ── */
.chart-row{display:grid;grid-template-columns:1fr 220px;gap:12px;margin-bottom:16px}
@media(max-width:900px){.chart-row{grid-template-columns:1fr}}
.chart-wrap{position:relative}

/* ── Trace items ── */
.item{
  background:var(--bg-card);border:1px solid var(--border);
  border-radius:var(--r);margin-bottom:10px;overflow:hidden;
  transition:border-color .15s;
}
.item:hover{border-color:var(--border-light)}
.item-head{padding:14px 20px;display:flex;align-items:center;gap:10px;cursor:pointer;transition:background .1s}
.item-head:hover{background:rgba(255,255,255,.02)}
.item-name{font-weight:700;font-size:14px;flex:1;letter-spacing:-.02em}
.item-meta{display:flex;align-items:center;gap:6px;flex-shrink:0;flex-wrap:wrap}
.meta-chip{
  display:inline-flex;align-items:center;gap:3px;
  padding:2px 8px;border-radius:5px;background:rgba(255,255,255,.04);
  font-size:11px;font-weight:500;color:var(--text-3);white-space:nowrap;
}
.chevron{color:var(--text-4);font-size:10px;transition:transform .2s;flex-shrink:0}
details[open] .turn-chevron{transform:rotate(90deg)}
.item-body{padding:20px;border-top:1px solid var(--border);background:rgba(0,0,0,.15)}
.mermaid-box{background:rgba(0,0,0,.2);border:1px solid rgba(51,65,85,.4);border-radius:var(--r-sm);padding:28px 20px;overflow-x:auto;min-height:200px}
.mermaid-box svg{min-width:560px;max-width:100%;height:auto;display:block;margin:0 auto}
.mermaid-box .mermaid{min-width:560px}

/* ── Chat turns ── */
.chat-container{margin-top:18px;padding:16px;background:rgba(0,0,0,.12);border:1px solid rgba(51,65,85,.3);border-radius:var(--r-sm)}
.chat-header{font-size:11px;font-weight:700;color:var(--text-3);text-transform:uppercase;letter-spacing:.06em;margin-bottom:14px;padding-bottom:10px;border-bottom:1px solid rgba(51,65,85,.3)}
.chat-messages{display:flex;flex-direction:column;gap:4px}
.chat-bubble{max-width:80%;padding:10px 14px;font-size:13px;line-height:1.55;border-radius:14px}
.chat-bubble.user{align-self:flex-end;background:rgba(37,99,235,.12);border:1px solid rgba(37,99,235,.2);color:var(--text);border-bottom-right-radius:4px}
.chat-bubble.agent{align-self:flex-start;background:rgba(255,255,255,.03);border:1px solid rgba(51,65,85,.4);color:var(--text-2);border-bottom-left-radius:4px}
.chat-meta{display:flex;align-items:center;gap:8px;padding:5px 2px;font-size:10px;color:var(--text-4);font-weight:500}
.chat-meta.right{justify-content:flex-end}
.chat-tool-tag{display:inline-flex;padding:1px 7px;border-radius:4px;background:rgba(37,99,235,.08);border:1px solid rgba(37,99,235,.15);font-size:10px;font-weight:600;color:var(--blue-bright);font-family:var(--mono)}
.chat-eval{padding:6px 10px;border-radius:8px;font-size:11px;font-weight:600;max-width:80%}
.chat-eval.pass{align-self:flex-start;background:rgba(16,185,129,.08);border:1px solid rgba(16,185,129,.2);color:var(--green-bright)}
.chat-eval.fail{align-self:flex-start;background:rgba(239,68,68,.08);border:1px solid rgba(239,68,68,.2);color:var(--red-bright)}

/* ── Diffs ── */
.diff-item{background:var(--bg-card);border:1px solid var(--border);border-radius:var(--r);margin-bottom:10px;overflow:hidden}
.diff-head{padding:14px 20px;display:flex;align-items:center;gap:8px;flex-wrap:wrap;border-bottom:1px solid var(--border)}
.diff-name{font-weight:700;font-size:14px;flex:1;letter-spacing:-.02em}
.diff-cols{display:grid;grid-template-columns:1fr 1fr}
.diff-col{padding:16px 20px}
.diff-col+.diff-col{border-left:1px solid var(--border)}
.col-title{font-size:10px;font-weight:700;color:var(--text-4);text-transform:uppercase;letter-spacing:.06em;margin-bottom:8px}
.tags{display:flex;flex-wrap:wrap;gap:4px;margin-bottom:8px}
.tag{background:rgba(255,255,255,.04);border:1px solid rgba(51,65,85,.5);border-radius:5px;padding:2px 8px;font-size:11px;font-family:var(--mono);font-weight:500}
.tag.add{border-color:rgba(16,185,129,.3);color:var(--green-bright);background:rgba(16,185,129,.06)}
.tag.rem{border-color:rgba(239,68,68,.3);color:var(--red-bright);background:rgba(239,68,68,.06);text-decoration:line-through}
.outbox{background:rgba(0,0,0,.2);border:1px solid rgba(51,65,85,.4);border-radius:var(--r-xs);padding:12px;font:12px/1.6 var(--mono);color:var(--text-3);white-space:pre-wrap;word-break:break-all;max-height:200px;overflow-y:auto}
.difflines{background:rgba(0,0,0,.2);border:1px solid rgba(51,65,85,.4);border-radius:var(--r-xs);padding:10px;font:11px/1.6 var(--mono);max-height:200px;overflow-y:auto;margin-top:8px}
.difflines .a{color:var(--green-bright);background:rgba(16,185,129,.05);display:block;padding:1px 4px;margin:0 -4px;border-radius:2px}
.difflines .r{color:var(--red-bright);background:rgba(239,68,68,.05);display:block;padding:1px 4px;margin:0 -4px;border-radius:2px}
.sim{display:inline-flex;align-items:center;gap:5px;font-size:11px;color:var(--text-3)}
.sim-track{width:44px;height:4px;background:rgba(255,255,255,.06);border-radius:2px;overflow:hidden;display:inline-block;vertical-align:middle}
.sim-fill{height:100%;border-radius:2px}
.sim-fill.hi{background:var(--green)}.sim-fill.mid{background:var(--yellow)}.sim-fill.lo{background:var(--red)}

/* Pipeline */
.pipeline{display:flex;flex-direction:column;gap:6px;padding:14px 20px;border-top:1px solid var(--border)}
.pipeline-row{display:flex;align-items:center;gap:4px;flex-wrap:wrap}
.pipeline-label{font-size:10px;font-weight:700;color:var(--text-4);text-transform:uppercase;letter-spacing:.06em;width:64px;flex-shrink:0}
.pipe-step{display:inline-flex;padding:4px 10px;border-radius:5px;font-size:11px;font-family:var(--mono);font-weight:600;background:rgba(255,255,255,.04);border:1px solid rgba(51,65,85,.5);color:var(--text-2);position:relative}
.pipe-step+.pipe-step{margin-left:6px}
.pipe-step+.pipe-step::before{content:'→';position:absolute;left:-13px;color:var(--text-4);font-size:9px;font-family:var(--font)}
.pipe-step.match{border-color:rgba(37,99,235,.25);background:rgba(37,99,235,.05)}
.pipe-step.added{border-color:rgba(16,185,129,.3);color:var(--green-bright);background:rgba(16,185,129,.06)}
.pipe-step.removed{border-color:rgba(239,68,68,.3);color:var(--red-bright);background:rgba(239,68,68,.06);text-decoration:line-through}
.traj-grid{display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-top:14px;padding-top:14px;border-top:1px solid var(--border)}
.traj-col .col-title{padding-bottom:8px}

/* ── Tables ── */
.ev-table{width:100%;border-collapse:collapse;font-size:13px}
.ev-table th{text-align:left;padding:8px 12px;color:var(--text-4);font-size:10px;font-weight:700;text-transform:uppercase;letter-spacing:.06em;border-bottom:1px solid var(--border)}
.ev-table td{padding:10px 12px;border-bottom:1px solid rgba(51,65,85,.3);transition:background .1s}
.ev-table tr:hover td{background:rgba(255,255,255,.015)}
.ev-table .mono{font-family:var(--mono);font-size:12px}
.ev-table .num{font-weight:700;font-variant-numeric:tabular-nums}
.param-table{width:100%;border-collapse:collapse;font-size:12px}
.param-table th{text-align:left;padding:6px 10px;color:var(--text-4);font-size:10px;font-weight:700;text-transform:uppercase;letter-spacing:.06em;border-bottom:1px solid var(--border)}
.param-table td{padding:6px 10px;border-bottom:1px solid rgba(51,65,85,.3)}
table td,table th{transition:background .1s}

/* ── Empty ── */
.empty{text-align:center;padding:72px 40px;color:var(--text-4)}
.empty-icon{font-size:36px;margin-bottom:12px;display:block;opacity:.3}
.empty code{background:rgba(255,255,255,.06);padding:2px 8px;border-radius:5px;font-family:var(--mono);font-size:12px;border:1px solid var(--border)}

/* ── Scrollbar ── */
::-webkit-scrollbar{width:4px;height:4px}::-webkit-scrollbar-track{background:transparent}::-webkit-scrollbar-thumb{background:rgba(255,255,255,.08);border-radius:4px}

/* ── Entrance animation (subtle) ── */
@keyframes fadeUp{from{opacity:0;transform:translateY(10px)}to{opacity:1;transform:none}}
.hero,.card,.item,.diff-item,.meta-card{animation:fadeUp .35s ease-out both}
</style>
</head>
<body>

<div class="bg-blobs"><div class="b1"></div><div class="b2"></div><div class="b3"></div></div>

<header class="header">
  <div class="logo">
    <div class="logo-icon">◈</div>
    <div>
      <div class="logo-text">{{ title }}</div>
      <div class="logo-sub">{{ generated_at }}{% if notes %} · {{ notes }}{% endif %}</div>
    </div>
  </div>
  <div class="header-right">
    {% if kpis %}
      {% if kpis.failed == 0 %}
        <span class="badge b-green">✓ All Passing</span>
      {% else %}
        <span class="badge b-red">✗ {{ kpis.failed }} Failed</span>
      {% endif %}
      <span class="badge b-blue">{{ kpis.total }} Tests</span>
    {% endif %}
  </div>
</header>

<main class="main">

  <div class="tabbar">
    <button class="tab {% if default_tab == 'overview' %}on{% endif %}" onclick="show('overview',this)">Overview</button>
    <button class="tab {% if default_tab == 'trace' %}on{% endif %}" onclick="show('trace',this)">Execution Trace</button>
    <button class="tab {% if default_tab == 'diffs' %}on{% endif %}" onclick="show('diffs',this)">Diffs</button>
    <button class="tab {% if default_tab == 'timeline' %}on{% endif %}" onclick="show('timeline',this)">Timeline</button>
    {% if compare %}<button class="tab" onclick="show('compare',this)">Compare Runs</button>{% endif %}
  </div>

  <!-- ═══════════ OVERVIEW ═══════════ -->
  <div id="p-overview" class="panel {% if default_tab == 'overview' %}on{% endif %}">
    {% if kpis %}

    <!-- HERO: The scoreboard -->
    <div class="hero">
      <div class="hero-pass {% if kpis.pass_rate >= 80 %}is-green{% else %}is-red{% endif %}">
        <div class="hero-label">Pass Rate</div>
        <div class="hero-num {% if kpis.pass_rate >= 80 %}green{% else %}red{% endif %}">{{ kpis.pass_rate }}%</div>
        <div class="hero-sub">{{ kpis.passed }} of {{ kpis.total }} tests passing</div>
        <div class="hero-ring">
          <svg width="80" height="80" viewBox="0 0 80 80">
            <circle cx="40" cy="40" r="34" fill="none" stroke="rgba(255,255,255,.06)" stroke-width="5"/>
            <circle cx="40" cy="40" r="34" fill="none"
              stroke="{% if kpis.pass_rate >= 80 %}var(--green-bright){% elif kpis.pass_rate >= 60 %}var(--yellow-bright){% else %}var(--red-bright){% endif %}"
              stroke-width="5" stroke-linecap="round"
              stroke-dasharray="{{ (kpis.pass_rate / 100 * 213.6)|round(1) }} 213.6"/>
          </svg>
          <div class="hero-ring-label">{{ kpis.passed }}/{{ kpis.total }}</div>
        </div>
      </div>
      <div class="hero-right">
        <div class="hero-stat">
          <div class="stat-label">Avg Score</div>
          <div class="stat-num" style="color:{% if kpis.avg_score >= 80 %}var(--green-bright){% elif kpis.avg_score >= 60 %}var(--yellow-bright){% else %}var(--red-bright){% endif %}">{{ kpis.avg_score }}<span style="font-size:14px;color:var(--text-4);font-weight:500">/100</span></div>
        </div>
        <div class="hero-stat">
          <div class="stat-label">Total Cost</div>
          <div class="stat-num blue">${{ kpis.total_cost }}</div>
          <div class="stat-sub">{% if kpis.total_tokens %}{{ '{:,}'.format(kpis.total_tokens) }} tokens{% elif kpis.total_cost > 0 %}reported by adapter (no token data){% else %}this run{% endif %}</div>
        </div>
        <div class="hero-stat">
          <div class="stat-label">Avg Latency</div>
          <div class="stat-num">{{ kpis.avg_latency_ms|int }}<span style="font-size:14px;color:var(--text-4);font-weight:500">ms</span></div>
          <div class="stat-sub">per test</div>
        </div>
        <div class="hero-stat">
          <div class="stat-label">Model</div>
          <div style="font-size:13px;font-weight:600;color:var(--text);margin-top:4px;line-height:1.4">{{ kpis.models_display }}</div>
          {% if kpis.total_input_tokens or kpis.total_output_tokens %}
          <div style="margin-top:6px;font-size:11px;color:var(--text-4);font-family:var(--mono)">in {{ '{:,}'.format(kpis.total_input_tokens) }} · out {{ '{:,}'.format(kpis.total_output_tokens) }}</div>
          {% endif %}
        </div>
      </div>
    </div>

    <!-- Agent Model + Token Usage -->
    <div class="meta-row">
      <div class="meta-card">
        <div class="meta-label">Agent Model</div>
        <div class="meta-value">{{ kpis.models_display }}</div>
        <div class="meta-sub">{{ kpis.total }} test{% if kpis.total != 1 %}s{% endif %} in this run</div>
      </div>
      {% if kpis.total_tokens %}
      <div class="meta-card">
        <div class="meta-label">Token Usage</div>
        <div class="meta-value">{{ '{:,}'.format(kpis.total_tokens) }} tokens</div>
        <div class="meta-sub">in {{ '{:,}'.format(kpis.total_input_tokens) }} / out {{ '{:,}'.format(kpis.total_output_tokens) }}</div>
      </div>
      {% elif kpis.total_cost > 0 %}
      <div class="meta-card">
        <div class="meta-label">Token Usage</div>
        <div class="meta-value" style="color:var(--yellow-bright)">Not available</div>
        <div class="meta-sub">Your adapter reports cost but not token counts. Cost cannot be independently verified.</div>
      </div>
      {% endif %}
    </div>
    {% if baseline.latest_created_display != 'Unknown' %}
    <div class="meta-row">
      <div class="meta-card">
        <div class="meta-label">Baseline Snapshot</div>
        <div class="meta-value">{{ baseline.latest_created_display }}</div>
        <div class="meta-sub">{% if baseline.models_display != 'Unknown' %}Model: {{ baseline.models_display }}{% endif %}</div>
      </div>
    </div>
    {% endif %}

    {% if judge_usage and judge_usage.call_count %}
    <div class="meta-row">
      <div class="meta-card">
        <div class="meta-label">EvalView Judge{% if judge_usage.model %} ({{ judge_usage.model }}){% endif %}</div>
        <div class="meta-value">{% if judge_usage.total_cost > 0 %}${{ judge_usage.total_cost }}{% elif judge_usage.is_free %}FREE{% else %}$0{% endif %}</div>
        <div class="meta-sub">{{ '{:,}'.format(judge_usage.total_tokens) }} tokens across {{ judge_usage.call_count }} judge call{% if judge_usage.call_count != 1 %}s{% endif %}</div>
      </div>
      <div class="meta-card">
        <div class="meta-label">Judge Token Breakdown</div>
        <div class="meta-value">in {{ '{:,}'.format(judge_usage.input_tokens) }} / out {{ '{:,}'.format(judge_usage.output_tokens) }}</div>
        <div class="meta-sub">{% if judge_usage.pricing %}Rate: {{ judge_usage.pricing }}{% else %}Separate from agent trace cost{% endif %}</div>
      </div>
    </div>
    {% endif %}

    <!-- Score bars + donut -->
    <div class="chart-row">
      <div class="card">
        <div class="card-title">Score per Test</div>
        <div class="chart-wrap" style="height:{{ [kpis.scores|length * 44 + 30, 180]|max }}px"><canvas id="bars"></canvas></div>
      </div>
      <div class="card">
        <div class="card-title">Distribution</div>
        <div class="chart-wrap" style="height:200px"><canvas id="donut"></canvas></div>
      </div>
    </div>

    <!-- Cost table -->
    <div class="card">
      <div class="card-title">Execution Cost per Query</div>
      <table class="ev-table">
        {% set has_tokens = traces | selectattr('tokens') | list | length > 0 %}
        <thead><tr><th>Test</th><th>Model</th><th>Trace Cost</th>{% if has_tokens %}<th>Tokens</th>{% endif %}<th>Latency</th><th>Score</th></tr></thead>
        <tbody>
          {% for t in traces %}
          <tr>
            <td style="font-weight:600">{{ t.name }}</td>
            <td class="mono" style="color:var(--text-4)">{{ t.model }}</td>
            <td class="mono num" style="color:{% if t.cost == '$0' %}var(--text-4){% else %}var(--blue-bright){% endif %}">{{ t.cost }}</td>
            {% if has_tokens %}<td class="mono" style="color:var(--text-3)">{{ t.tokens or '—' }}</td>{% endif %}
            <td style="color:var(--text-3)">{{ t.latency }}</td>
            <td class="num" style="color:{% if t.score >= 80 %}var(--green-bright){% elif t.score >= 60 %}var(--yellow-bright){% else %}var(--red-bright){% endif %}">{{ t.score }}</td>
          </tr>
          {% endfor %}
          <tr style="background:rgba(0,0,0,.1)">
            <td style="font-weight:800">Total</td><td style="color:var(--text-4)">—</td>
            <td class="mono num" style="color:var(--blue-bright)">${{ kpis.total_cost }}</td>
            <td colspan="{{ 3 if has_tokens else 2 }}" style="font-size:11px;color:var(--text-4)">avg ${{ '%.6f'|format(kpis.total_cost / kpis.total) if kpis.total else '0' }} per query</td>
          </tr>
        </tbody>
      </table>
      <div style="margin-top:12px;font-size:11px;color:var(--text-4);line-height:1.5">
        Trace cost comes from the agent execution trace only. Mock or non-metered tools will show <code style="background:rgba(255,255,255,.05);padding:2px 7px;border-radius:4px;font-family:var(--mono);font-size:11px;border:1px solid var(--border)">$0</code> even when EvalView used a separate judge or local model during evaluation.
        {% if judge_usage and judge_usage.call_count %} This check also used {{ judge_usage.call_count }} EvalView judge call{% if judge_usage.call_count != 1 %}s{% endif %} ({{ judge_usage.total_tokens }} tokens).{% endif %}
      </div>
    </div>
    {% else %}
    <div class="empty"><span class="empty-icon">📊</span>No results to display</div>
    {% endif %}
  </div>

  <!-- ═══════════ EXECUTION TRACE ═══════════ -->
  <div id="p-trace" class="panel {% if default_tab == 'trace' %}on{% endif %}">
    {% if traces %}
      {% for t in traces %}
      <div class="item">
        <div class="item-head" onclick="tog('tr{{ loop.index }}',this)">
          <span class="badge {% if t.passed %}b-green{% else %}b-red{% endif %}">{% if t.passed %}✓{% else %}✗{% endif %}</span>
          <span class="item-name">{{ t.name }}</span>
          <div class="item-meta">
            <span class="meta-chip" style="color:{% if t.score >= 80 %}var(--green-bright){% elif t.score >= 60 %}var(--yellow-bright){% else %}var(--red-bright){% endif %}">{{ t.score }}/100</span>
            {% if t.cost != "$0" %}<span class="meta-chip">💰 {{ t.cost }}</span>{% endif %}
            <span class="meta-chip">⚡ {{ t.latency }}</span>
            {% if t.tokens %}<span class="meta-chip">{{ t.tokens }}</span>{% endif %}
            <span class="meta-chip">🧠 {{ t.model }}</span>
          </div>
          <span class="chevron">▾</span>
        </div>
        <div id="tr{{ loop.index }}" class="item-body" {% if not loop.first %}style="display:none"{% endif %}>
          <div style="display:flex;flex-wrap:wrap;gap:5px;margin-bottom:14px">
            <span class="badge b-blue">Model: {{ t.model }}</span>
            {% if t.input_tokens or t.output_tokens %}<span class="badge b-blue">in {{ '{:,}'.format(t.input_tokens) }} / out {{ '{:,}'.format(t.output_tokens) }} tokens</span>{% if t.cost != "$0" %}<span class="badge b-blue">{{ t.cost }}</span>{% endif %}{% endif %}
            {% if not t.input_tokens and not t.output_tokens and t.cost != "$0" %}<span class="badge b-yellow">{{ t.cost }} (adapter-reported, no token data)</span>{% endif %}
            {% if t.baseline_created and t.baseline_created != 'Unknown' %}<span class="badge b-purple">Baseline: {{ t.baseline_created }}</span>{% endif %}
            {% if t.baseline_model and t.baseline_model != 'Unknown' %}<span class="badge b-yellow">Baseline model: {{ t.baseline_model }}</span>{% endif %}
          </div>
          {% if t.hallucination or t.safety or t.pii or t.forbidden_tools %}
          <div style="display:flex;flex-wrap:wrap;gap:6px;margin-bottom:14px">
            {% if t.hallucination %}
            <span class="badge {% if t.hallucination.passed %}b-green{% else %}b-red{% endif %}">
              {% if t.hallucination.passed %}No Hallucination{% else %}Hallucination Detected ({{ (t.hallucination.confidence * 100)|int }}% confidence){% endif %}
            </span>
            {% endif %}
            {% if t.safety %}
            <span class="badge {% if t.safety.passed %}b-green{% else %}b-red{% endif %}">
              {% if t.safety.passed %}Safe{% else %}Safety: {{ t.safety.severity }}{% if t.safety.categories %} — {{ t.safety.categories|join(', ') }}{% endif %}{% endif %}
            </span>
            {% endif %}
            {% if t.pii %}
            <span class="badge {% if t.pii.passed %}b-green{% else %}b-red{% endif %}">
              {% if t.pii.passed %}No PII{% else %}PII Detected: {{ t.pii.types|join(', ') }}{% endif %}
            </span>
            {% endif %}
            {% if t.forbidden_tools and t.forbidden_tools.violations %}
            <span class="badge b-red">Forbidden tools: {{ t.forbidden_tools.violations|join(', ') }}</span>
            {% endif %}
          </div>
          {% endif %}
          {% if t.query %}
          <div style="background:rgba(37,99,235,.06);border:1px solid rgba(37,99,235,.15);border-radius:var(--r-xs);padding:10px 14px;margin-bottom:14px;font-size:13px;color:var(--text-2)">
            <span style="font-size:10px;font-weight:700;text-transform:uppercase;letter-spacing:.06em;color:var(--text-4);margin-right:8px">Query</span>{{ t.query }}
          </div>
          {% endif %}
          {% if t.has_steps %}
          <div class="mermaid-box"><div class="mermaid">{{ t.diagram }}</div></div>
          {% else %}
          <div style="display:flex;align-items:center;justify-content:center;padding:20px 0 8px">
            <span style="display:inline-flex;align-items:center;gap:6px;background:rgba(255,255,255,.03);border:1px solid var(--border);border-radius:20px;padding:8px 18px;font-size:12px;color:var(--text-4)"><span style="opacity:.4">◎</span> Direct response — no tools invoked</span>
          </div>
          {% endif %}
          {% if t.turns %}
          <div class="chat-container">
            <div class="chat-header">Conversation Turns</div>
            <div class="chat-messages">
            {% for turn in t.turns %}
              <div class="chat-meta right">
                Turn {{ turn.index }}{% if turn.tools %} · {% for tool in turn.tools %}<span class="chat-tool-tag">{{ tool }}</span> {% endfor %}{% endif %} · ⚡ {{ turn.latency_ms|round(1) }}ms · 💰 ${{ '%.6f'|format(turn.cost) if turn.cost else '0' }}
              </div>
              <div class="chat-bubble user">{{ turn.query }}</div>
              {% if turn.output %}<div class="chat-bubble agent">{{ turn.output }}</div>{% endif %}
              {% if turn.evaluation %}
              <div class="chat-eval {% if turn.evaluation.passed %}pass{% else %}fail{% endif %}">
                <span style="font-weight:700">{% if turn.evaluation.passed %}✅ PASS{% else %}❌ FAIL{% endif %}</span>
                {% if turn.evaluation.tool_accuracy is not none %}<span style="margin-left:8px;opacity:.7">Tool accuracy: {{ (turn.evaluation.tool_accuracy * 100)|round(0) }}%</span>{% endif %}
                {% if turn.evaluation.forbidden_violations %}<span style="margin-left:8px;color:var(--red-bright)">Forbidden: {{ turn.evaluation.forbidden_violations|join(', ') }}</span>{% endif %}
                {% if turn.evaluation.contains_failed %}<span style="margin-left:8px;color:var(--red-bright)">Missing: {{ turn.evaluation.contains_failed|join(', ') }}</span>{% endif %}
                {% if turn.evaluation.not_contains_failed %}<span style="margin-left:8px;color:var(--red-bright)">Prohibited: {{ turn.evaluation.not_contains_failed|join(', ') }}</span>{% endif %}
              </div>
              {% endif %}
            {% endfor %}
            </div>
          </div>
          {% endif %}
          {% if t.output and not t.turns %}
          <div style="background:rgba(16,185,129,.05);border:1px solid rgba(16,185,129,.12);border-radius:var(--r-xs);padding:10px 14px;margin-top:14px;font-size:13px;color:var(--text-2)">
            <span style="font-size:10px;font-weight:700;text-transform:uppercase;letter-spacing:.06em;color:var(--text-4);margin-right:8px">Response</span>{{ t.output[:300] }}{% if t.output|length > 300 %}...{% endif %}
          </div>
          {% endif %}
        </div>
      </div>
      {% endfor %}
    {% else %}
      <div class="empty"><span class="empty-icon">🔍</span>No trace data available</div>
    {% endif %}
  </div>

  <!-- ═══════════ DIFFS ═══════════ -->
  <div id="p-diffs" class="panel {% if default_tab == 'diffs' %}on{% endif %}">
    {% if diff_rows %}
      {% for d in diff_rows %}
      <div class="diff-item">
        <div class="diff-head">
          {% if d.status == 'regression' %}<span class="badge b-red">⬇ Regression</span>{% elif d.status == 'tools_changed' %}<span class="badge b-yellow">⚠ Tools Changed</span>{% elif d.status == 'output_changed' %}<span class="badge b-purple">~ Output Changed</span>{% else %}<span class="badge b-green">✓ Passed</span>{% endif %}
          <span class="diff-name">{{ d.name }}</span>
          {% if d.score_delta != 0 %}<span class="badge {% if d.score_delta > 0 %}b-green{% else %}b-red{% endif %}">{% if d.score_delta > 0 %}+{% endif %}{{ d.score_delta }} pts</span>{% endif %}
          <span class="sim">lexical <span class="sim-track"><span class="sim-fill {% if d.similarity >= 80 %}hi{% elif d.similarity >= 50 %}mid{% else %}lo{% endif %}" style="width:{{ d.similarity }}%"></span></span> <b style="color:{% if d.similarity >= 80 %}var(--green-bright){% elif d.similarity >= 50 %}var(--yellow-bright){% else %}var(--red-bright){% endif %}">{{ d.similarity }}%</b></span>
          {% if d.semantic_similarity is not none %}<span class="sim">semantic <span class="sim-track"><span class="sim-fill {% if d.semantic_similarity >= 80 %}hi{% elif d.semantic_similarity >= 50 %}mid{% else %}lo{% endif %}" style="width:{{ d.semantic_similarity }}%"></span></span> <b style="color:{% if d.semantic_similarity >= 80 %}var(--green-bright){% elif d.semantic_similarity >= 50 %}var(--yellow-bright){% else %}var(--red-bright){% endif %}">{{ d.semantic_similarity }}%</b></span>{% endif %}
        </div>
        {% if d.golden_tools or d.actual_tools %}
        <div class="pipeline">
          <div class="pipeline-row"><span class="pipeline-label">Baseline</span>{% for t in d.golden_tools %}<span class="pipe-step {% if t not in d.actual_tools %}removed{% else %}match{% endif %}">{{ t }}</span>{% endfor %}{% if not d.golden_tools %}<span style="font-size:11px;color:var(--text-4);font-style:italic">No tools</span>{% endif %}</div>
          <div class="pipeline-row"><span class="pipeline-label">Current</span>{% for t in d.actual_tools %}<span class="pipe-step {% if t not in d.golden_tools %}added{% else %}match{% endif %}">{{ t }}</span>{% endfor %}{% if not d.actual_tools %}<span style="font-size:11px;color:var(--text-4);font-style:italic">No tools</span>{% endif %}</div>
        </div>
        {% endif %}
        <div class="diff-cols">
          <div class="diff-col"><div class="col-title">Baseline</div><div class="tags">{% for t in d.golden_tools %}<span class="tag {% if t not in d.actual_tools %}rem{% endif %}">{{ t }}</span>{% endfor %}</div><div class="outbox">{{ d.golden_out }}</div></div>
          <div class="diff-col"><div class="col-title">Current</div><div class="tags">{% for t in d.actual_tools %}<span class="tag {% if t not in d.golden_tools %}add{% endif %}">{{ t }}</span>{% endfor %}</div><div class="outbox">{{ d.actual_out }}</div>{% if d.diff_lines %}<div class="difflines">{% for line in d.diff_lines %}{% if line.startswith('+') %}<div class="a">{{ line }}</div>{% elif line.startswith('-') %}<div class="r">{{ line }}</div>{% else %}<div>{{ line }}</div>{% endif %}{% endfor %}</div>{% endif %}</div>
        </div>
        {% if d.param_diffs %}
        <div style="padding:14px 20px;border-top:1px solid var(--border)">
          <div class="col-title" style="margin-bottom:10px">Parameter Changes</div>
          <table class="param-table">
            <thead><tr><th>Step</th><th>Tool</th><th>Parameter</th><th>Baseline</th><th>Current</th><th style="text-align:center">Match</th></tr></thead>
            <tbody>{% for p in d.param_diffs %}<tr>
              <td style="color:var(--text-4)">{{ p.step }}</td>
              <td style="font-family:var(--mono);color:var(--blue-bright)">{{ p.tool }}</td>
              <td style="font-weight:600">{{ p.param }}</td>
              <td style="font-family:var(--mono);font-size:11px;{% if p.type == 'missing' %}color:var(--red-bright){% else %}color:var(--text-3){% endif %}">{{ p.golden or '—' }}</td>
              <td style="font-family:var(--mono);font-size:11px;{% if p.type == 'added' %}color:var(--green-bright){% else %}color:var(--text-3){% endif %}">{{ p.actual or '—' }}</td>
              <td style="text-align:center;font-weight:600;color:{% if p.type == 'added' %}var(--green-bright){% elif p.type == 'missing' %}var(--red-bright){% elif p.similarity is not none %}{% if p.similarity >= 80 %}var(--green-bright){% elif p.similarity >= 50 %}var(--yellow-bright){% else %}var(--red-bright){% endif %}{% else %}var(--yellow-bright){% endif %}">{% if p.type == 'added' %}+new{% elif p.type == 'missing' %}-gone{% elif p.similarity is not none %}{{ p.similarity }}%{% else %}~{% endif %}</td>
            </tr>{% endfor %}</tbody>
          </table>
        </div>
        {% endif %}
        {% if d.golden_diagram or d.actual_diagram %}
        <div class="traj-grid">
          <div class="traj-col"><div class="col-title">Baseline Trajectory</div><div class="mermaid-box" style="min-height:140px"><div class="mermaid">{{ d.golden_diagram or "sequenceDiagram\n    Note over Agent: No trace data" }}</div></div></div>
          <div class="traj-col"><div class="col-title">Current Trajectory</div><div class="mermaid-box" style="min-height:140px"><div class="mermaid">{{ d.actual_diagram or "sequenceDiagram\n    Note over Agent: No trace data" }}</div></div></div>
        </div>
        {% endif %}
      </div>
      {% endfor %}
    {% else %}
      <div class="empty"><span class="empty-icon">✨</span>No diffs yet — run <code>evalview check</code> to compare against a baseline</div>
    {% endif %}
  </div>

  <!-- ═══════════ TIMELINE ═══════════ -->
  <div id="p-timeline" class="panel {% if default_tab == 'timeline' %}on{% endif %}">
    {% if timeline %}
      <div class="card">
        <div class="card-title">Step Latencies</div>
        <div style="position:relative;height:{{ [timeline|length * 40 + 60, 200]|max }}px"><canvas id="tlChart"></canvas></div>
      </div>
    {% else %}
      <div class="empty"><span class="empty-icon">⏱</span>No step timing data</div>
    {% endif %}
  </div>

  <!-- ═══════════ COMPARE ═══════════ -->
  {% if compare %}
  <div id="p-compare" class="panel">
    <div class="card"><div class="card-title">Pass Rate Across Runs</div><div class="chart-wrap" style="height:240px"><canvas id="cmpPassRate"></canvas></div></div>
    <div class="card"><div class="card-title">Avg Score Across Runs</div><div class="chart-wrap" style="height:240px"><canvas id="cmpScore"></canvas></div></div>
    <div class="card">
      <div class="card-title">Run Summary</div>
      <table class="ev-table">
        <thead><tr>{% for lbl in compare.labels %}<th>{{ lbl }}</th>{% endfor %}</tr></thead>
        <tbody><tr>{% for run in compare.runs %}<td>
          <div style="font-size:26px;font-weight:900;letter-spacing:-.04em;color:{% if run.pass_rate >= 80 %}var(--green-bright){% else %}var(--red-bright){% endif %}">{{ run.pass_rate }}%</div>
          <div style="font-size:11px;color:var(--text-4);margin-top:3px">{{ run.passed }}/{{ run.total }} · avg {{ run.avg_score }}/100</div>
        </td>{% endfor %}</tr></tbody>
      </table>
    </div>
  </div>
  {% endif %}
</main>

<script>
mermaid.initialize({startOnLoad:true,theme:'dark',securityLevel:'loose',useMaxWidth:true,
  themeVariables:{darkMode:true,background:'transparent',primaryColor:'rgba(37,99,235,.12)',primaryTextColor:'#e2e8f0',primaryBorderColor:'rgba(37,99,235,.3)',lineColor:'rgba(100,116,139,.35)',secondaryColor:'rgba(16,185,129,.08)',tertiaryColor:'rgba(6,182,212,.08)',noteBkgColor:'rgba(37,99,235,.06)',noteTextColor:'#94a3b8',noteBorderColor:'rgba(37,99,235,.2)',actorBkg:'rgba(37,99,235,.1)',actorBorder:'rgba(37,99,235,.25)',actorTextColor:'#e2e8f0',signalColor:'#64748b',signalTextColor:'#cbd5e1'},
  sequence:{useMaxWidth:true,width:180,wrap:false,actorFontFamily:'Inter,sans-serif',noteFontFamily:'Inter,sans-serif',messageFontFamily:'Inter,sans-serif',actorFontSize:12,messageFontSize:11,noteFontSize:10,boxTextMargin:8,mirrorActors:false,messageAlign:'center',actorMargin:30,bottomMarginAdj:4}
});
function show(id,btn){document.querySelectorAll('.panel').forEach(p=>p.classList.remove('on'));document.querySelectorAll('.tab').forEach(t=>t.classList.remove('on'));document.getElementById('p-'+id).classList.add('on');btn.classList.add('on')}
function tog(id,head){const el=document.getElementById(id);const open=el.style.display!=='none';el.style.display=open?'none':'block';head.querySelector('.chevron').style.transform=open?'':'rotate(180deg)'}

{% if kpis %}
(function(){
  const passed={{ kpis.passed }},failed={{ kpis.failed }};
  const scores={{ kpis.scores|tojson }},names={{ kpis.test_names|tojson }};
  const tc='rgba(100,116,139,.7)',gc='rgba(255,255,255,.03)';
  const tt={backgroundColor:'rgba(10,15,30,.95)',borderColor:'rgba(51,65,85,.6)',borderWidth:1,titleFont:{family:'Inter',weight:'700',size:12},bodyFont:{family:'Inter',size:12},padding:10,cornerRadius:8};

  new Chart(document.getElementById('donut'),{type:'doughnut',data:{labels:['Passed','Failed'],datasets:[{data:[passed,failed],backgroundColor:['rgba(16,185,129,.65)','rgba(239,68,68,.65)'],borderColor:['rgba(16,185,129,.1)','rgba(239,68,68,.1)'],borderWidth:2,hoverOffset:6}]},options:{responsive:true,maintainAspectRatio:false,cutout:'76%',plugins:{legend:{position:'bottom',labels:{color:tc,font:{family:'Inter',size:11,weight:'500'},padding:16,boxWidth:8,boxHeight:8,usePointStyle:true,pointStyle:'circle'}},tooltip:{...tt,callbacks:{label:ctx=>` ${ctx.label}: ${ctx.raw}`}}}}});

  const sorted=names.map((n,i)=>({name:n,score:scores[i]})).sort((a,b)=>b.score-a.score);
  new Chart(document.getElementById('bars'),{type:'bar',
    data:{labels:sorted.map(s=>s.name),datasets:[{label:'Score',data:sorted.map(s=>s.score),
      backgroundColor:sorted.map(s=>s.score>=80?'rgba(16,185,129,.4)':s.score>=60?'rgba(245,158,11,.4)':'rgba(239,68,68,.4)'),
      borderColor:sorted.map(s=>s.score>=80?'rgba(16,185,129,.6)':s.score>=60?'rgba(245,158,11,.6)':'rgba(239,68,68,.6)'),
      borderWidth:1,borderRadius:4,borderSkipped:false,barPercentage:.6,categoryPercentage:.8}]},
    options:{indexAxis:'y',responsive:true,maintainAspectRatio:false,
      scales:{x:{min:0,max:100,grid:{color:gc},ticks:{color:tc,font:{family:'Inter',size:10},stepSize:25},border:{display:false}},y:{grid:{display:false},ticks:{color:'rgba(203,213,225,.8)',font:{family:'Inter',size:11,weight:'600'},padding:6},border:{display:false}}},
      plugins:{legend:{display:false},tooltip:{...tt,callbacks:{label:ctx=>` Score: ${ctx.raw}/100`}}}}});
})();
{% endif %}

{% if timeline %}
(function(){
  const tl={{ timeline|tojson }};if(!tl.length)return;
  const labels=tl.map(r=>r.label||(r.test+' \u203a '+r.tool));const vals=tl.map(r=>r.latency||0);const costs=tl.map(r=>r.cost||0);
  const maxLat=Math.max(...vals,0),maxCost=Math.max(...costs,0.000001);
  const colors=tl.map((r,i)=>r.success?`rgba(37,99,235,${(0.3+0.4*(costs[i]/maxCost)).toFixed(2)})`:'rgba(239,68,68,.5)');
  const borders=tl.map(r=>r.success?'rgba(37,99,235,.6)':'rgba(239,68,68,.6)');
  const tt={backgroundColor:'rgba(10,15,30,.95)',borderColor:'rgba(51,65,85,.6)',borderWidth:1,titleFont:{family:'Inter',weight:'700'},bodyFont:{family:'Inter'},padding:10,cornerRadius:8};
  new Chart(document.getElementById('tlChart'),{type:'bar',data:{labels,datasets:[{label:'ms',data:vals,backgroundColor:colors,borderColor:borders,borderWidth:1,borderRadius:4,borderSkipped:false,barPercentage:.6}]},options:{indexAxis:'y',responsive:true,maintainAspectRatio:false,scales:{x:{suggestedMax:maxLat>0?maxLat*1.15:1,grid:{color:'rgba(255,255,255,.03)'},ticks:{color:'rgba(100,116,139,.6)',font:{family:'Inter',size:10},callback:v=>v+'ms'},border:{display:false}},y:{grid:{display:false},ticks:{color:'rgba(203,213,225,.6)',font:{family:'Inter',size:11}},border:{display:false}}},plugins:{legend:{display:false},tooltip:{...tt,callbacks:{label:ctx=>` ${ctx.raw}ms`,afterLabel:ctx=>` Cost: $${(costs[ctx.dataIndex]||0).toFixed(6)}`,title:ctx=>ctx[0].label}}}}});
})();
{% endif %}

{% if compare %}
(function(){
  const labels={{ compare.labels|tojson }};const pr={{ compare.runs|map(attribute='pass_rate')|list|tojson }};const as={{ compare.runs|map(attribute='avg_score')|list|tojson }};
  const tc='rgba(100,116,139,.6)',gc='rgba(255,255,255,.03)';
  const c=['rgba(37,99,235,.5)','rgba(16,185,129,.5)','rgba(239,68,68,.5)','rgba(245,158,11,.5)','rgba(6,182,212,.5)'];
  const b=['rgba(37,99,235,.7)','rgba(16,185,129,.7)','rgba(239,68,68,.7)','rgba(245,158,11,.7)','rgba(6,182,212,.7)'];
  const o={responsive:true,maintainAspectRatio:false,scales:{y:{grid:{color:gc},ticks:{color:tc},border:{display:false}},x:{grid:{display:false},ticks:{color:tc,font:{size:11}},border:{display:false}}},plugins:{legend:{display:false}}};
  new Chart(document.getElementById('cmpPassRate'),{type:'bar',data:{labels,datasets:[{label:'Pass Rate %',data:pr,backgroundColor:c.slice(0,labels.length),borderColor:b.slice(0,labels.length),borderWidth:1,borderRadius:6,borderSkipped:false}]},options:{...o,scales:{...o.scales,y:{...o.scales.y,min:0,max:100}}}});
  new Chart(document.getElementById('cmpScore'),{type:'bar',data:{labels,datasets:[{label:'Avg Score',data:as,backgroundColor:c.slice(0,labels.length),borderColor:b.slice(0,labels.length),borderWidth:1,borderRadius:6,borderSkipped:false}]},options:{...o,scales:{...o.scales,y:{...o.scales.y,min:0,max:100}}}});
})();
{% endif %}
</script>

<!-- Share bar -->
<div style="position:fixed;bottom:0;left:0;right:0;z-index:100;background:rgba(10,15,30,.9);backdrop-filter:blur(16px);-webkit-backdrop-filter:blur(16px);border-top:1px solid var(--border);padding:10px 24px;display:flex;align-items:center;justify-content:space-between;font-family:var(--font);font-size:12px;color:var(--text-4)">
  <span>Built with <a href="https://github.com/hidai25/eval-view" target="_blank" rel="noopener" style="color:var(--blue-bright);text-decoration:none;font-weight:600">EvalView</a> <span style="opacity:.3;margin:0 6px">|</span> Agent testing &amp; regression detection</span>
  <span style="display:flex;align-items:center;gap:6px">
    <a href="https://twitter.com/intent/tweet?text=Testing%20my%20AI%20agent%20with%20EvalView%20%E2%80%94%20catches%20regressions%20before%20they%20ship.%20%F0%9F%9B%A1%EF%B8%8F&url=https%3A%2F%2Fgithub.com%2Fhidai25%2Feval-view" target="_blank" rel="noopener" style="display:inline-flex;align-items:center;gap:4px;padding:5px 12px;border-radius:6px;background:rgba(29,155,240,.1);color:#1d9bf0;text-decoration:none;font-weight:600;font-size:11px;border:1px solid rgba(29,155,240,.12)" onmouseover="this.style.background='rgba(29,155,240,.18)'" onmouseout="this.style.background='rgba(29,155,240,.1)'"><svg width="13" height="13" viewBox="0 0 24 24" fill="currentColor"><path d="M18.244 2.25h3.308l-7.227 8.26 8.502 11.24H16.17l-5.214-6.817L4.99 21.75H1.68l7.73-8.835L1.254 2.25H8.08l4.713 6.231zm-1.161 17.52h1.833L7.084 4.126H5.117z"/></svg>Share</a>
    <a href="https://github.com/hidai25/eval-view" target="_blank" rel="noopener" style="display:inline-flex;align-items:center;gap:4px;padding:5px 12px;border-radius:6px;background:rgba(255,255,255,.04);color:var(--text-2);text-decoration:none;font-weight:600;font-size:11px;border:1px solid var(--border)" onmouseover="this.style.background='rgba(255,255,255,.07)'" onmouseout="this.style.background='rgba(255,255,255,.04)'"><svg width="13" height="13" viewBox="0 0 16 16" fill="currentColor"><path d="M8 0c4.42 0 8 3.58 8 8a8.013 8.013 0 0 1-5.45 7.59c-.4.08-.55-.17-.55-.38 0-.27.01-1.13.01-2.2 0-.75-.25-1.23-.54-1.48 1.78-.2 3.65-.88 3.65-3.95 0-.88-.31-1.59-.82-2.15.08-.2.36-1.02-.08-2.12 0 0-.67-.22-2.2.82-.64-.18-1.32-.27-2-.27-.68 0-1.36.09-2 .27-1.53-1.03-2.2-.82-2.2-.82-.44 1.1-.16 1.92-.08 2.12-.51.56-.82 1.28-.82 2.15 0 3.06 1.86 3.75 3.64 3.95-.23.2-.44.55-.51 1.07-.46.21-1.61.55-2.33-.66-.15-.24-.6-.83-1.23-.82-.67.01-.27.38.01.53.34.19.73.9.82 1.13.16.45.68 1.31 2.69.94 0 .67.01 1.3.01 1.49 0 .21-.15.45-.55.38A7.995 7.995 0 0 1 0 8c0-4.42 3.58-8 8-8Z"/></svg>Star</a>
  </span>
</div>
<div style="height:44px"></div>

</body>
</html>"""

