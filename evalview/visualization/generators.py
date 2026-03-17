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
    for r in results:
        try:
            costs.append(r.trace.metrics.total_cost or 0)
            latencies.append(r.trace.metrics.total_latency or 0)
        except AttributeError:
            pass
    models = _collect_models(results)
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

    trace_context = getattr(trace, "trace_context", None)
    if trace_context:
        for span in trace_context.spans:
            if span.llm and span.llm.model and span.llm.model not in seen_ids:
                seen_ids.add(span.llm.model)
                provider = span.llm.provider or model_provider
                labels.append(_clean_model_name(span.llm.model, provider))

    return labels


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
            if r.trace.metrics.total_tokens:
                tokens = r.trace.metrics.total_tokens.input_tokens + r.trace.metrics.total_tokens.output_tokens
        except AttributeError:
            cost, latency, tokens = 0.0, 0.0, None
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
            "score": round(r.score, 1),
            "model": ", ".join(models) if models else "Unknown",
            "baseline_created": baseline_created or "Unknown",
            "baseline_model": baseline_model,
            "query": getattr(r, "input_query", "") or "",
            "output": _strip_markdown(getattr(r, "actual_output", "") or ""),
            "turns": turn_list,
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
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap" rel="stylesheet">
<script src="https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4/dist/chart.umd.min.js"></script>
<style>
*{box-sizing:border-box;margin:0;padding:0}
:root{
  --green:#22d3a5;--red:#ff6b8a;--yellow:#fbbf24;--blue:#7c95ff;--purple:#c084fc;--pink:#f472b6;
  --g1:rgba(34,211,165,.2);--g2:rgba(124,149,255,.2);--g3:rgba(248,107,138,.2);
  --glass:rgba(255,255,255,.04);--glass2:rgba(255,255,255,.07);
  --border:rgba(255,255,255,.09);--border2:rgba(255,255,255,.15);
  --text:#f8fafc;--muted:#7a8fa6;--r:14px;
  --font:'Inter',-apple-system,sans-serif;
}
html{scroll-behavior:smooth}
body{
  font-family:var(--font);font-size:14px;line-height:1.6;
  color:var(--text);min-height:100vh;overflow-x:hidden;
  background:
    radial-gradient(ellipse 90% 60% at 15% 0%,rgba(124,149,255,.18),transparent 60%),
    radial-gradient(ellipse 70% 50% at 85% 100%,rgba(34,211,165,.15),transparent 60%),
    radial-gradient(ellipse 50% 40% at 50% 50%,rgba(248,107,138,.08),transparent 60%),
    #050a14;
}
/* Animated orbs */
body::before,body::after{
  content:'';position:fixed;border-radius:50%;filter:blur(80px);
  pointer-events:none;z-index:0;animation:float 12s ease-in-out infinite;
}
body::before{width:500px;height:500px;background:rgba(124,149,255,.07);top:-150px;right:-100px}
body::after{width:400px;height:400px;background:rgba(34,211,165,.07);bottom:-100px;left:-80px;animation-delay:-6s}
@keyframes float{0%,100%{transform:translateY(0)}50%{transform:translateY(-20px)}}
/* Header */
.header{
  position:sticky;top:0;z-index:200;
  background:rgba(5,10,20,.8);
  border-bottom:1px solid var(--border);
  backdrop-filter:blur(24px) saturate(180%);
  -webkit-backdrop-filter:blur(24px) saturate(180%);
  padding:0 40px;height:62px;
  display:flex;align-items:center;justify-content:space-between;
}
.logo{
  display:flex;align-items:center;gap:12px;
}
.logo-icon{
  width:34px;height:34px;border-radius:10px;flex-shrink:0;
  background:linear-gradient(135deg,#7c95ff,#c084fc);
  box-shadow:0 0 0 1px rgba(124,149,255,.4),0 4px 20px rgba(124,149,255,.35);
  display:flex;align-items:center;justify-content:center;font-size:16px;
}
.logo-text{font-size:15px;font-weight:700;letter-spacing:-.02em}
.logo-sub{font-size:11px;color:var(--muted);font-weight:400}
.header-right{display:flex;align-items:center;gap:8px}
/* Badges */
.badge{
  display:inline-flex;align-items:center;gap:5px;
  padding:4px 12px;border-radius:20px;font-size:11px;font-weight:600;
}
.b-green{background:rgba(34,211,165,.12);color:var(--green);border:1px solid rgba(34,211,165,.25)}
.b-red{background:rgba(255,107,138,.12);color:var(--red);border:1px solid rgba(255,107,138,.25)}
.b-yellow{background:rgba(251,191,36,.12);color:var(--yellow);border:1px solid rgba(251,191,36,.25)}
.b-blue{background:rgba(124,149,255,.12);color:var(--blue);border:1px solid rgba(124,149,255,.25)}
.b-purple{background:rgba(192,132,252,.12);color:var(--purple);border:1px solid rgba(192,132,252,.25)}
/* Main */
.main{max-width:1200px;margin:0 auto;padding:32px 40px;position:relative;z-index:1}
/* Tab bar */
.tabbar{
  display:flex;gap:2px;
  background:rgba(255,255,255,.03);border:1px solid var(--border);
  border-radius:12px;padding:3px;margin-bottom:32px;width:fit-content;
}
.tab{
  background:none;border:none;color:var(--muted);cursor:pointer;
  font:500 13px/1 var(--font);padding:9px 20px;border-radius:9px;
  transition:all .18s;
}
.tab:hover{color:var(--text);background:rgba(255,255,255,.05)}
.tab.on{
  color:#fff;
  background:linear-gradient(135deg,rgba(124,149,255,.3),rgba(192,132,252,.2));
  border:1px solid rgba(124,149,255,.35);
  box-shadow:0 2px 16px rgba(124,149,255,.2),inset 0 1px 0 rgba(255,255,255,.1);
}
.panel{display:none}.panel.on{display:block}
/* KPI row */
.kpi-row{display:grid;grid-template-columns:repeat(4,1fr);gap:16px;margin-bottom:20px}
.kpi{
  background:var(--glass);border:1px solid var(--border);
  border-radius:var(--r);padding:22px 20px;
  backdrop-filter:blur(16px);-webkit-backdrop-filter:blur(16px);
  position:relative;overflow:hidden;
  transition:transform .2s,border-color .2s,box-shadow .2s;cursor:default;
}
.kpi::after{
  content:'';position:absolute;inset:0;pointer-events:none;border-radius:var(--r);
  background:linear-gradient(135deg,rgba(255,255,255,.05) 0%,transparent 60%);
}
.kpi:hover{transform:translateY(-3px)}
.kpi.kpi-pass{border-color:rgba(34,211,165,.2)}
.kpi.kpi-pass:hover{box-shadow:0 12px 40px rgba(34,211,165,.15);border-color:rgba(34,211,165,.4)}
.kpi.kpi-fail{border-color:rgba(255,107,138,.2)}
.kpi.kpi-fail:hover{box-shadow:0 12px 40px rgba(255,107,138,.15);border-color:rgba(255,107,138,.4)}
.kpi.kpi-blue{border-color:rgba(124,149,255,.2)}
.kpi.kpi-blue:hover{box-shadow:0 12px 40px rgba(124,149,255,.15);border-color:rgba(124,149,255,.4)}
.kpi-icon{font-size:20px;margin-bottom:14px;display:block}
.kpi-label{font-size:10px;font-weight:700;color:var(--muted);text-transform:uppercase;letter-spacing:.1em;margin-bottom:8px}
.kpi-num{font-size:36px;font-weight:800;letter-spacing:-.03em;line-height:1}
.kpi-num.c-green{color:var(--green);text-shadow:0 0 30px rgba(34,211,165,.4)}
.kpi-num.c-red{color:var(--red);text-shadow:0 0 30px rgba(255,107,138,.4)}
.kpi-num.c-yellow{color:var(--yellow)}
.kpi-num.c-blue{color:var(--blue);text-shadow:0 0 30px rgba(124,149,255,.4)}
.kpi-sub{font-size:12px;color:var(--muted);margin-top:6px}
/* Score bar on KPI */
.kpi-bar{margin-top:14px;height:3px;background:rgba(255,255,255,.08);border-radius:2px;overflow:hidden}
.kpi-bar-fill{height:100%;border-radius:2px;transition:width 1s cubic-bezier(.4,0,.2,1)}
.kpi-bar-fill.green{background:linear-gradient(90deg,#22d3a5,#7c95ff)}
.kpi-bar-fill.red{background:linear-gradient(90deg,#ff6b8a,#fbbf24)}
.kpi-bar-fill.blue{background:linear-gradient(90deg,#7c95ff,#c084fc)}
.meta-row{display:grid;grid-template-columns:1.2fr 1fr;gap:16px;margin-bottom:20px}
.meta-card{
  background:var(--glass);border:1px solid var(--border);
  border-radius:var(--r);padding:18px 20px;
  backdrop-filter:blur(16px);-webkit-backdrop-filter:blur(16px);
}
.meta-label{font-size:10px;font-weight:700;color:var(--muted);text-transform:uppercase;letter-spacing:.1em;margin-bottom:8px}
.meta-value{font-size:15px;font-weight:700;color:var(--text)}
.meta-sub{font-size:12px;color:var(--muted);margin-top:4px}
/* Charts */
.chart-row{display:grid;grid-template-columns:260px 1fr;gap:16px;margin-bottom:20px}
.card{
  background:var(--glass);border:1px solid var(--border);
  border-radius:var(--r);padding:22px;
  backdrop-filter:blur(16px);-webkit-backdrop-filter:blur(16px);
  position:relative;overflow:hidden;
}
.card::after{
  content:'';position:absolute;inset:0;pointer-events:none;border-radius:var(--r);
  background:linear-gradient(135deg,rgba(255,255,255,.04) 0%,transparent 50%);
}
.card-title{font-size:11px;font-weight:700;color:var(--muted);text-transform:uppercase;letter-spacing:.08em;margin-bottom:18px;display:flex;align-items:center;gap:8px}
.card-title::before{content:'';width:3px;height:14px;border-radius:2px;background:linear-gradient(to bottom,#7c95ff,#c084fc)}
.chart-wrap{position:relative;height:200px}
/* Trace */
.item{
  background:var(--glass);border:1px solid var(--border);
  border-radius:var(--r);margin-bottom:10px;overflow:hidden;
  backdrop-filter:blur(12px);transition:border-color .2s;
}
.item:hover{border-color:var(--border2)}
.item-head{
  padding:15px 20px;display:flex;align-items:center;gap:12px;
  cursor:pointer;transition:background .15s;
}
.item-head:hover{background:rgba(255,255,255,.03)}
.item-name{font-weight:600;font-size:13px;flex:1;letter-spacing:-.01em}
.chevron{color:var(--muted);font-size:11px;transition:transform .2s}
details[open] .turn-chevron{transform:rotate(90deg)}
.item-body{
  padding:20px;border-top:1px solid var(--border);
  background:rgba(0,0,0,.25);
}
.mermaid-box{
  background:rgba(0,0,0,.4);border:1px solid var(--border);
  border-radius:10px;padding:32px 24px;overflow-x:auto;
  min-height:220px;
}
.mermaid-box svg{
  min-width:560px;max-width:100%;height:auto;display:block;margin:0 auto;
}
.mermaid-box .mermaid{min-width:560px}
/* Diff */
.diff-item{
  background:var(--glass);border:1px solid var(--border);
  border-radius:var(--r);margin-bottom:10px;overflow:hidden;
  backdrop-filter:blur(12px);
}
.diff-head{padding:15px 20px;display:flex;align-items:center;gap:10px;flex-wrap:wrap;border-bottom:1px solid var(--border)}
.diff-name{font-weight:600;font-size:13px;flex:1}
.diff-cols{display:grid;grid-template-columns:1fr 1fr}
.diff-col{padding:16px 20px}
.diff-col+.diff-col{border-left:1px solid var(--border)}
.col-title{font-size:10px;font-weight:700;color:var(--muted);text-transform:uppercase;letter-spacing:.08em;margin-bottom:10px}
.tags{display:flex;flex-wrap:wrap;gap:5px;margin-bottom:10px}
.tag{
  background:rgba(255,255,255,.05);border:1px solid var(--border);
  border-radius:5px;padding:2px 9px;font-size:11px;font-family:monospace;
}
.tag.add{border-color:rgba(34,211,165,.3);color:var(--green);background:rgba(34,211,165,.08)}
.tag.rem{border-color:rgba(255,107,138,.3);color:var(--red);background:rgba(255,107,138,.08)}
.outbox{
  background:rgba(0,0,0,.3);border:1px solid var(--border);border-radius:8px;
  padding:12px;font:12px/1.5 monospace;color:var(--muted);
  white-space:pre-wrap;word-break:break-all;max-height:200px;overflow-y:auto;
}
.difflines{
  background:rgba(0,0,0,.3);border:1px solid var(--border);border-radius:8px;
  padding:10px;font:11px/1.5 monospace;max-height:160px;overflow-y:auto;margin-top:8px;
}
.difflines .a{color:var(--green)}.difflines .r{color:var(--red)}
/* Timeline */
.tl-row{display:flex;align-items:center;gap:12px;margin-bottom:8px}
.tl-label{font-size:11px;color:var(--muted);width:210px;flex-shrink:0;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.tl-track{flex:1;background:rgba(255,255,255,.04);border:1px solid var(--border);border-radius:4px;height:24px;overflow:hidden;position:relative}
.tl-fill{height:100%;border-radius:4px;transition:width .7s cubic-bezier(.4,0,.2,1);position:relative}
.tl-fill.ok{background:linear-gradient(90deg,rgba(34,211,165,.7),rgba(124,149,255,.4))}
.tl-fill.err{background:linear-gradient(90deg,rgba(255,107,138,.7),rgba(251,191,36,.4))}
.tl-ms{font-size:10px;color:var(--muted);width:65px;text-align:right;flex-shrink:0}
/* Sim */
.sim{display:inline-flex;align-items:center;gap:6px;font-size:12px;color:var(--muted)}
/* Empty */
.empty{text-align:center;padding:72px 40px;color:var(--muted)}
.empty-icon{font-size:40px;margin-bottom:14px;display:block;filter:grayscale(1);opacity:.5}
/* Compare table */
table td,table th{transition:background .15s}
table tr:hover td{background:rgba(255,255,255,.02)}
/* Trajectory grid — side-by-side Mermaid in Diffs tab */
.traj-grid{display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-top:16px;padding-top:16px;border-top:1px solid var(--border)}
.traj-col .col-title{padding-bottom:10px}
/* Scrollbar */
::-webkit-scrollbar{width:4px;height:4px}
::-webkit-scrollbar-track{background:transparent}
::-webkit-scrollbar-thumb{background:rgba(255,255,255,.12);border-radius:4px}
</style>
</head>
<body>

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
    <button class="tab on" onclick="show('overview',this)">Overview</button>
    <button class="tab" onclick="show('trace',this)">Execution Trace</button>
    <button class="tab" onclick="show('diffs',this)">Diffs</button>
    <button class="tab" onclick="show('timeline',this)">Timeline</button>
    {% if compare %}<button class="tab" onclick="show('compare',this)">Compare Runs</button>{% endif %}
  </div>

  <!-- OVERVIEW -->
  <div id="p-overview" class="panel on">
    {% if kpis %}
    <div class="kpi-row">
      <div class="kpi {% if kpis.pass_rate >= 80 %}kpi-pass{% else %}kpi-fail{% endif %}">
        <span class="kpi-icon">{% if kpis.pass_rate == 100 %}🟢{% elif kpis.pass_rate >= 80 %}✅{% else %}⚠️{% endif %}</span>
        <div class="kpi-label">Pass Rate</div>
        <div class="kpi-num {% if kpis.pass_rate >= 80 %}c-green{% elif kpis.pass_rate >= 60 %}c-yellow{% else %}c-red{% endif %}">{{ kpis.pass_rate }}%</div>
        <div class="kpi-sub">{{ kpis.passed }} of {{ kpis.total }} tests</div>
        <div class="kpi-bar"><div class="kpi-bar-fill {% if kpis.pass_rate >= 80 %}green{% else %}red{% endif %}" style="width:{{ kpis.pass_rate }}%"></div></div>
      </div>
      <div class="kpi {% if kpis.avg_score >= 80 %}kpi-pass{% else %}kpi-blue{% endif %}">
        <span class="kpi-icon">📊</span>
        <div class="kpi-label">Avg Score</div>
        <div class="kpi-num {% if kpis.avg_score >= 80 %}c-green{% elif kpis.avg_score >= 60 %}c-yellow{% else %}c-red{% endif %}">{{ kpis.avg_score }}</div>
        <div class="kpi-sub">out of 100</div>
        <div class="kpi-bar"><div class="kpi-bar-fill {% if kpis.avg_score >= 80 %}green{% else %}red{% endif %}" style="width:{{ kpis.avg_score }}%"></div></div>
      </div>
      <div class="kpi kpi-blue">
        <span class="kpi-icon">💰</span>
        <div class="kpi-label">Total Cost</div>
        <div class="kpi-num c-blue">${{ kpis.total_cost }}</div>
        <div class="kpi-sub">this run</div>
        <div class="kpi-bar"><div class="kpi-bar-fill blue" style="width:30%"></div></div>
      </div>
      <div class="kpi kpi-blue">
        <span class="kpi-icon">⚡</span>
        <div class="kpi-label">Avg Latency</div>
        <div class="kpi-num c-blue">{{ kpis.avg_latency_ms|int }}<span style="font-size:16px;font-weight:500">ms</span></div>
        <div class="kpi-sub">per test</div>
        <div class="kpi-bar"><div class="kpi-bar-fill blue" style="width:45%"></div></div>
      </div>
    </div>

    <div class="meta-row">
      <div class="meta-card">
        <div class="meta-label">Models Used In This Check</div>
        <div class="meta-value">{{ kpis.models_display }}</div>
        <div class="meta-sub">{{ kpis.total }} test{% if kpis.total != 1 %}s{% endif %} in this run</div>
      </div>
      <div class="meta-card">
        <div class="meta-label">Latest Baseline Snapshot</div>
        <div class="meta-value">{{ baseline.latest_created_display }}</div>
        <div class="meta-sub">Baseline model: {{ baseline.models_display }}</div>
      </div>
    </div>

    {% if judge_usage and judge_usage.call_count %}
    <div class="meta-row">
      <div class="meta-card">
        <div class="meta-label">EvalView Judge Usage</div>
        <div class="meta-value">
          {% if judge_usage.total_cost > 0 %}
            ${{ judge_usage.total_cost }}
          {% elif judge_usage.is_free %}
            FREE
          {% else %}
            $0
          {% endif %}
        </div>
        <div class="meta-sub">
          {{ judge_usage.total_tokens }} total tokens across {{ judge_usage.call_count }} judge call{% if judge_usage.call_count != 1 %}s{% endif %}
        </div>
      </div>
      <div class="meta-card">
        <div class="meta-label">Judge Token Breakdown</div>
        <div class="meta-value">in {{ judge_usage.input_tokens }} / out {{ judge_usage.output_tokens }}</div>
        <div class="meta-sub">Separate from agent trace cost</div>
      </div>
    </div>
    {% endif %}

    <div class="chart-row">
      <div class="card">
        <div class="card-title">Distribution</div>
        <div class="chart-wrap"><canvas id="donut"></canvas></div>
      </div>
      <div class="card">
        <div class="card-title">Score per Test</div>
        <div class="chart-wrap"><canvas id="bars"></canvas></div>
      </div>
    </div>

    <!-- Execution cost breakdown -->
    <div class="card">
      <div class="card-title">Execution Cost per Query</div>
      <table style="width:100%;border-collapse:collapse;font-size:13px">
        {% set has_tokens = traces | selectattr('tokens') | list | length > 0 %}
        <thead>
          <tr>
            <th style="text-align:left;padding:8px 12px;color:var(--muted);font-size:10px;font-weight:700;text-transform:uppercase;letter-spacing:.06em;border-bottom:1px solid var(--border)">Test</th>
            <th style="text-align:left;padding:8px 12px;color:var(--muted);font-size:10px;font-weight:700;text-transform:uppercase;letter-spacing:.06em;border-bottom:1px solid var(--border)">Model</th>
            <th style="text-align:left;padding:8px 12px;color:var(--muted);font-size:10px;font-weight:700;text-transform:uppercase;letter-spacing:.06em;border-bottom:1px solid var(--border)">Trace Cost</th>
            {% if has_tokens %}<th style="text-align:left;padding:8px 12px;color:var(--muted);font-size:10px;font-weight:700;text-transform:uppercase;letter-spacing:.06em;border-bottom:1px solid var(--border)">Tokens</th>{% endif %}
            <th style="text-align:left;padding:8px 12px;color:var(--muted);font-size:10px;font-weight:700;text-transform:uppercase;letter-spacing:.06em;border-bottom:1px solid var(--border)">Latency</th>
            <th style="text-align:left;padding:8px 12px;color:var(--muted);font-size:10px;font-weight:700;text-transform:uppercase;letter-spacing:.06em;border-bottom:1px solid var(--border)">Score</th>
          </tr>
        </thead>
        <tbody>
          {% for t in traces %}
          <tr>
            <td style="padding:10px 12px;border-bottom:1px solid var(--border);font-weight:500">{{ t.name }}</td>
            <td style="padding:10px 12px;border-bottom:1px solid var(--border);color:var(--muted)">{{ t.model }}</td>
            <td style="padding:10px 12px;border-bottom:1px solid var(--border);font-family:monospace;color:{% if t.cost == '$0' %}var(--muted){% else %}var(--blue){% endif %};font-weight:600">{{ t.cost }}</td>
            {% if has_tokens %}<td style="padding:10px 12px;border-bottom:1px solid var(--border);color:var(--muted)">{{ t.tokens or '—' }}</td>{% endif %}
            <td style="padding:10px 12px;border-bottom:1px solid var(--border);color:var(--muted)">{{ t.latency }}</td>
            <td style="padding:10px 12px;border-bottom:1px solid var(--border);font-weight:700;color:{% if t.score >= 80 %}var(--green){% elif t.score >= 60 %}var(--yellow){% else %}var(--red){% endif %}">{{ t.score }}</td>
          </tr>
          {% endfor %}
          <tr style="background:rgba(255,255,255,.02)">
            <td style="padding:10px 12px;font-weight:700;color:var(--text)">Total</td>
            <td style="padding:10px 12px;color:var(--muted)">—</td>
            <td style="padding:10px 12px;font-family:monospace;font-weight:700;color:var(--blue)">${{ kpis.total_cost }}</td>
            <td colspan="{{ 3 if has_tokens else 2 }}" style="padding:10px 12px;font-size:11px;color:var(--muted)">avg ${{ '%.6f'|format(kpis.total_cost / kpis.total) if kpis.total else '0' }} per query</td>
          </tr>
        </tbody>
      </table>
      <div style="margin-top:12px;font-size:11px;color:var(--muted)">
        Trace cost comes from the agent execution trace only. Mock or non-metered tools will show <code style="background:rgba(255,255,255,.08);padding:2px 6px;border-radius:4px">$0</code> even when EvalView used a separate judge or local model during evaluation.
        {% if judge_usage and judge_usage.call_count %} This check also used {{ judge_usage.call_count }} EvalView judge call{% if judge_usage.call_count != 1 %}s{% endif %} ({{ judge_usage.total_tokens }} tokens).{% endif %}
      </div>
    </div>

    {% else %}
    <div class="empty"><span class="empty-icon">📊</span>No results to display</div>
    {% endif %}
  </div>

  <!-- TRACE -->
  <div id="p-trace" class="panel">
    {% if traces %}
      {% for t in traces %}
      <div class="item">
        <div class="item-head" onclick="tog('tr{{ loop.index }}',this)">
          <span class="badge {% if t.passed %}b-green{% else %}b-red{% endif %}">{% if t.passed %}✓{% else %}✗{% endif %}</span>
          <span class="item-name">{{ t.name }}</span>
          <span style="display:flex;align-items:center;gap:12px;font-size:11px;color:var(--muted)">
            <span style="color:{% if t.score >= 80 %}var(--green){% elif t.score >= 60 %}var(--yellow){% else %}var(--red){% endif %}">● {{ t.score }}/100</span>
            {% if t.cost != "$0" %}<span>💰 {{ t.cost }}</span>{% endif %}
            <span>⚡ {{ t.latency }}</span>
            {% if t.tokens %}<span>🔤 {{ t.tokens }}</span>{% endif %}
            <span>🧠 {{ t.model }}</span>
          </span>
          <span class="chevron" style="display:inline-flex;align-items:center;gap:4px">▾ <span style="font-size:10px;font-weight:400">details</span></span>
        </div>
        <div id="tr{{ loop.index }}" class="item-body" {% if not loop.first %}style="display:none"{% endif %}>
          <div style="display:flex;flex-wrap:wrap;gap:8px;margin-bottom:14px">
            <span class="badge b-blue">Model: {{ t.model }}</span>
            {% if t.baseline_created and t.baseline_created != 'Unknown' %}
            <span class="badge b-purple">Baseline: {{ t.baseline_created }}</span>
            {% endif %}
            {% if t.baseline_model and t.baseline_model != 'Unknown' %}
            <span class="badge b-yellow">Baseline model: {{ t.baseline_model }}</span>
            {% endif %}
          </div>
          {% if t.query %}
          <div style="background:rgba(124,149,255,.06);border:1px solid rgba(124,149,255,.2);border-radius:8px;padding:10px 14px;margin-bottom:14px;font-size:12px;color:var(--muted)">
            <span style="font-size:10px;font-weight:700;text-transform:uppercase;letter-spacing:.06em;color:rgba(124,149,255,.7);margin-right:8px">Query</span>{{ t.query }}
          </div>
          {% endif %}
          {% if t.has_steps %}
          <div class="mermaid-box"><div class="mermaid">{{ t.diagram }}</div></div>
          {% else %}
          <div style="display:flex;align-items:center;justify-content:center;padding:20px 0 8px">
            <span style="display:inline-flex;align-items:center;gap:8px;background:rgba(255,255,255,.04);border:1px solid rgba(255,255,255,.1);border-radius:20px;padding:8px 18px;font-size:12px;color:var(--muted)">
              <span style="opacity:.5">◎</span> Direct response — no tools invoked
            </span>
          </div>
          {% endif %}
          {% if t.turns %}
          <div style="margin-top: 16px;">
            <div style="font-size:11px;font-weight:700;color:var(--muted);text-transform:uppercase;letter-spacing:.08em;margin-bottom:8px">Conversation Turns</div>

            {% for turn in t.turns %}
            <details style="background:rgba(255,255,255,.03);border:1px solid var(--border);border-radius:6px;margin-bottom:6px;overflow:hidden;" {% if loop.first %}open{% endif %}>
              <summary style="padding:10px 14px;cursor:pointer;font-size:12px;font-weight:600;display:flex;align-items:center;justify-content:space-between;color:var(--blue);">
                <span>Turn {{ turn.index }}{% if turn.tools %} · {{ turn.tools|join(', ') }}{% endif %}</span>
                <span style="font-size:10px;color:var(--muted);font-weight:400;display:inline-flex;align-items:center;gap:3px"><span style="transition:transform .2s;display:inline-block" class="turn-chevron">▶</span> details</span>
              </summary>

              <div style="padding:10px 14px;border-top:1px solid var(--border);background:rgba(0,0,0,.2);font-family:monospace;font-size:11px;color:var(--muted);">

                <div style="margin-bottom: 10px; color: var(--text); font-family: var(--font); line-height: 1.5;">
                  <span style="color:var(--muted); font-size: 10px; font-weight: 700; text-transform: uppercase; margin-right: 8px;">Query</span>{{ turn.query }}
                </div>

                {% if turn.output %}
                <div style="margin-bottom: 10px; color: var(--text); font-family: var(--font); line-height: 1.5;">
                  <span style="color:var(--muted); font-size: 10px; font-weight: 700; text-transform: uppercase; margin-right: 8px;">Response</span>{{ turn.output }}
                </div>
                {% endif %}

                <div style="display: flex; align-items: center; gap: 8px;">
                  <span style="color:var(--muted); font-size: 10px; font-weight: 700; text-transform: uppercase;">Tools called</span>
                  <div style="display: flex; flex-wrap: wrap; gap: 4px;">
                    {% if turn.tools %}
                      {% for tool in turn.tools %}<span style="background:rgba(255,255,255,.08);padding:2px 6px;border-radius:4px;color:var(--muted);">{{ tool }}</span>{% endfor %}
                    {% else %}
                      <span style="opacity: 0.5;">None</span>
                    {% endif %}
                  </div>
                </div>
                <div style="display:flex;align-items:center;gap:12px;margin-top:10px;color:var(--muted);font-size:11px;font-family:var(--font)">
                  <span>⚡ {{ turn.latency_ms|round(1) }}ms</span>
                  <span>💰 ${{ '%.6f'|format(turn.cost) if turn.cost else '0' }}</span>
                </div>

                {% if turn.evaluation %}
                <div style="margin-top:10px;padding:8px 12px;border-radius:6px;font-family:var(--font);font-size:11px;{% if turn.evaluation.passed %}background:rgba(34,211,165,.08);border:1px solid rgba(34,211,165,.2);{% else %}background:rgba(255,68,68,.08);border:1px solid rgba(255,68,68,.2);{% endif %}">
                  <span style="font-weight:700;{% if turn.evaluation.passed %}color:var(--green);{% else %}color:var(--red);{% endif %}">
                    {% if turn.evaluation.passed %}✅ PASS{% else %}❌ FAIL{% endif %}
                  </span>
                  {% if turn.evaluation.tool_accuracy is not none %}
                  <span style="margin-left:8px;color:var(--muted)">Tool accuracy: {{ (turn.evaluation.tool_accuracy * 100)|round(0) }}%</span>
                  {% endif %}
                  {% if turn.evaluation.forbidden_violations %}
                  <span style="margin-left:8px;color:var(--red)">Forbidden: {{ turn.evaluation.forbidden_violations|join(', ') }}</span>
                  {% endif %}
                  {% if turn.evaluation.contains_failed %}
                  <span style="margin-left:8px;color:var(--red)">Missing: {{ turn.evaluation.contains_failed|join(', ') }}</span>
                  {% endif %}
                  {% if turn.evaluation.not_contains_failed %}
                  <span style="margin-left:8px;color:var(--red)">Prohibited: {{ turn.evaluation.not_contains_failed|join(', ') }}</span>
                  {% endif %}
                </div>
                {% endif %}

              </div>
            </details>
            {% endfor %}
          </div>
          {% endif %}
          {% if t.output and not t.turns %}
          <div style="background:rgba(34,211,165,.04);border:1px solid rgba(34,211,165,.15);border-radius:8px;padding:10px 14px;margin-top:14px;font-size:12px;color:var(--muted)">
            <span style="font-size:10px;font-weight:700;text-transform:uppercase;letter-spacing:.06em;color:rgba(34,211,165,.7);margin-right:8px">Response</span>{{ t.output[:300] }}{% if t.output|length > 300 %}...{% endif %}
          </div>
          {% endif %}
        </div>
      </div>
      {% endfor %}
    {% else %}
      <div class="empty"><span class="empty-icon">🔍</span>No trace data available</div>
    {% endif %}
  </div>

  <!-- DIFFS -->
  <div id="p-diffs" class="panel">
    {% if diff_rows %}
      {% for d in diff_rows %}
      <div class="diff-item">
        <div class="diff-head">
          {% if d.status == 'regression' %}<span class="badge b-red">⬇ Regression</span>
          {% elif d.status == 'tools_changed' %}<span class="badge b-yellow">⚠ Tools Changed</span>
          {% elif d.status == 'output_changed' %}<span class="badge b-purple">~ Output Changed</span>
          {% else %}<span class="badge b-green">✓ Passed</span>{% endif %}
          <span class="diff-name">{{ d.name }}</span>
          {% if d.score_delta != 0 %}
            <span class="badge {% if d.score_delta > 0 %}b-green{% else %}b-red{% endif %}">{% if d.score_delta > 0 %}+{% endif %}{{ d.score_delta }} pts</span>
          {% endif %}
          <span class="sim">lexical <b style="color:{% if d.similarity >= 80 %}var(--green){% elif d.similarity >= 50 %}var(--yellow){% else %}var(--red){% endif %}">{{ d.similarity }}%</b></span>
          {% if d.semantic_similarity is not none %}
          <span class="sim">semantic <b style="color:{% if d.semantic_similarity >= 80 %}var(--green){% elif d.semantic_similarity >= 50 %}var(--yellow){% else %}var(--red){% endif %}">{{ d.semantic_similarity }}%</b></span>
          {% endif %}
        </div>
        <div class="diff-cols">
          <div class="diff-col">
            <div class="col-title">Baseline</div>
            <div class="tags">{% for t in d.golden_tools %}<span class="tag {% if t not in d.actual_tools %}rem{% endif %}">{{ t }}</span>{% endfor %}</div>
            <div class="outbox">{{ d.golden_out }}</div>
          </div>
          <div class="diff-col">
            <div class="col-title">Current</div>
            <div class="tags">{% for t in d.actual_tools %}<span class="tag {% if t not in d.golden_tools %}add{% endif %}">{{ t }}</span>{% endfor %}</div>
            <div class="outbox">{{ d.actual_out }}</div>
            {% if d.diff_lines %}
            <div class="difflines">{% for line in d.diff_lines %}{% if line.startswith('+') %}<div class="a">{{ line }}</div>{% elif line.startswith('-') %}<div class="r">{{ line }}</div>{% else %}<div>{{ line }}</div>{% endif %}{% endfor %}</div>
            {% endif %}
          </div>
        </div>
        {% if d.param_diffs %}
        <div style="padding:16px 20px;border-top:1px solid var(--border)">
          <div class="col-title" style="margin-bottom:12px">Parameter Changes</div>
          <table style="width:100%;border-collapse:collapse;font-size:12px">
            <thead>
              <tr>
                <th style="text-align:left;padding:6px 10px;color:var(--muted);font-size:10px;font-weight:700;text-transform:uppercase;letter-spacing:.06em;border-bottom:1px solid var(--border)">Step</th>
                <th style="text-align:left;padding:6px 10px;color:var(--muted);font-size:10px;font-weight:700;text-transform:uppercase;letter-spacing:.06em;border-bottom:1px solid var(--border)">Tool</th>
                <th style="text-align:left;padding:6px 10px;color:var(--muted);font-size:10px;font-weight:700;text-transform:uppercase;letter-spacing:.06em;border-bottom:1px solid var(--border)">Parameter</th>
                <th style="text-align:left;padding:6px 10px;color:var(--muted);font-size:10px;font-weight:700;text-transform:uppercase;letter-spacing:.06em;border-bottom:1px solid var(--border)">Baseline</th>
                <th style="text-align:left;padding:6px 10px;color:var(--muted);font-size:10px;font-weight:700;text-transform:uppercase;letter-spacing:.06em;border-bottom:1px solid var(--border)">Current</th>
                <th style="text-align:center;padding:6px 10px;color:var(--muted);font-size:10px;font-weight:700;text-transform:uppercase;letter-spacing:.06em;border-bottom:1px solid var(--border)">Match</th>
              </tr>
            </thead>
            <tbody>
              {% for p in d.param_diffs %}
              <tr>
                <td style="padding:6px 10px;border-bottom:1px solid var(--border);color:var(--muted)">{{ p.step }}</td>
                <td style="padding:6px 10px;border-bottom:1px solid var(--border);font-family:monospace;color:var(--blue)">{{ p.tool }}</td>
                <td style="padding:6px 10px;border-bottom:1px solid var(--border);font-weight:600">{{ p.param }}</td>
                <td style="padding:6px 10px;border-bottom:1px solid var(--border);font-family:monospace;font-size:11px;{% if p.type == 'missing' %}color:var(--red){% else %}color:var(--muted){% endif %}">{{ p.golden or '—' }}</td>
                <td style="padding:6px 10px;border-bottom:1px solid var(--border);font-family:monospace;font-size:11px;{% if p.type == 'added' %}color:var(--green){% else %}color:var(--muted){% endif %}">{{ p.actual or '—' }}</td>
                <td style="padding:6px 10px;border-bottom:1px solid var(--border);text-align:center;font-weight:600;color:{% if p.type == 'added' %}var(--green){% elif p.type == 'missing' %}var(--red){% elif p.similarity is not none %}{% if p.similarity >= 80 %}var(--green){% elif p.similarity >= 50 %}var(--yellow){% else %}var(--red){% endif %}{% else %}var(--yellow){% endif %}">{% if p.type == 'added' %}+new{% elif p.type == 'missing' %}-gone{% elif p.similarity is not none %}{{ p.similarity }}%{% else %}~{% endif %}</td>
              </tr>
              {% endfor %}
            </tbody>
          </table>
        </div>
        {% endif %}
        {% if d.golden_diagram or d.actual_diagram %}
        <div class="traj-grid">
          <div class="traj-col">
            <div class="col-title">Baseline Trajectory</div>
            <div class="mermaid-box" style="min-height:140px"><div class="mermaid">{{ d.golden_diagram or "sequenceDiagram\n    Note over Agent: No trace data" }}</div></div>
          </div>
          <div class="traj-col">
            <div class="col-title">Current Trajectory</div>
            <div class="mermaid-box" style="min-height:140px"><div class="mermaid">{{ d.actual_diagram or "sequenceDiagram\n    Note over Agent: No trace data" }}</div></div>
          </div>
        </div>
        {% endif %}
      </div>
      {% endfor %}
    {% else %}
      <div class="empty"><span class="empty-icon">✨</span>No diffs yet — run <code style="background:rgba(255,255,255,.08);padding:2px 6px;border-radius:4px">evalview check</code> to compare against a baseline</div>
    {% endif %}
  </div>

  <!-- TIMELINE -->
  <div id="p-timeline" class="panel">
    {% if timeline %}
      <div class="card">
        <div class="card-title">Step Latencies</div>
        <div style="position:relative;height:{{ [timeline|length * 38 + 80, 200]|max }}px">
          <canvas id="tlChart"></canvas>
        </div>
      </div>
    {% else %}
      <div class="empty"><span class="empty-icon">⏱</span>No step timing data</div>
    {% endif %}
  </div>

  <!-- COMPARE -->
  {% if compare %}
  <div id="p-compare" class="panel">
    <div class="card" style="margin-bottom:16px">
      <div class="card-title">Pass Rate Across Runs</div>
      <div class="chart-wrap" style="height:240px"><canvas id="cmpPassRate"></canvas></div>
    </div>
    <div class="card" style="margin-bottom:16px">
      <div class="card-title">Avg Score Across Runs</div>
      <div class="chart-wrap" style="height:240px"><canvas id="cmpScore"></canvas></div>
    </div>
    <div class="card">
      <div class="card-title">Run Summary</div>
      <table style="width:100%;border-collapse:collapse;font-size:13px">
        <thead>
          <tr>
            {% for lbl in compare.labels %}<th style="text-align:left;padding:10px 12px;color:var(--muted);font-size:11px;font-weight:700;text-transform:uppercase;letter-spacing:.06em;border-bottom:1px solid var(--border)">{{ lbl }}</th>{% endfor %}
          </tr>
        </thead>
        <tbody>
          <tr>
            {% for run in compare.runs %}
            <td style="padding:12px;border-bottom:1px solid var(--border)">
              <div style="font-size:22px;font-weight:800;color:{% if run.pass_rate >= 80 %}var(--green){% else %}var(--red){% endif %}">{{ run.pass_rate }}%</div>
              <div style="font-size:11px;color:var(--muted);margin-top:2px">{{ run.passed }}/{{ run.total }} · avg {{ run.avg_score }}/100</div>
            </td>
            {% endfor %}
          </tr>
        </tbody>
      </table>
    </div>
  </div>
  {% endif %}

</main>

<script>
mermaid.initialize({
  startOnLoad:true,theme:'dark',securityLevel:'loose',
  useMaxWidth:true,
  sequence:{
    useMaxWidth:true,
    width:180,
    wrap:false,
    actorFontFamily:'Inter,sans-serif',
    noteFontFamily:'Inter,sans-serif',
    messageFontFamily:'Inter,sans-serif',
    actorFontSize:13,
    messageFontSize:11,
    noteFontSize:11,
    boxTextMargin:6,
    mirrorActors:false,
    messageAlign:'center'
  }
});

function show(id,btn){
  document.querySelectorAll('.panel').forEach(p=>p.classList.remove('on'));
  document.querySelectorAll('.tab').forEach(t=>t.classList.remove('on'));
  document.getElementById('p-'+id).classList.add('on');
  btn.classList.add('on');
}
function tog(id,head){
  const el=document.getElementById(id);
  const open=el.style.display!=='none';
  el.style.display=open?'none':'block';
  head.querySelector('.chevron').style.transform=open?'':'rotate(180deg)';
}

{% if kpis %}
(function(){
  const passed={{ kpis.passed }},failed={{ kpis.failed }};
  const scores={{ kpis.scores|tojson }},names={{ kpis.test_names|tojson }};
  const tc='rgba(122,143,166,.8)',gc='rgba(255,255,255,.05)';

  new Chart(document.getElementById('donut'),{
    type:'doughnut',
    data:{labels:['Passed','Failed'],datasets:[{
      data:[passed,failed],
      backgroundColor:['rgba(34,211,165,.75)','rgba(255,107,138,.75)'],
      borderColor:['rgba(34,211,165,.2)','rgba(255,107,138,.2)'],
      borderWidth:1,hoverOffset:8
    }]},
    options:{responsive:true,maintainAspectRatio:false,cutout:'74%',
      plugins:{legend:{labels:{color:tc,font:{size:12},padding:20,boxWidth:10,boxHeight:10}},
      tooltip:{callbacks:{label:ctx=>` ${ctx.label}: ${ctx.raw}`}}}}
  });

  new Chart(document.getElementById('bars'),{
    type:'bar',
    data:{labels:names,datasets:[{
      label:'Score',data:scores,
      backgroundColor:scores.map(s=>s>=80?'rgba(34,211,165,.65)':s>=60?'rgba(251,191,36,.65)':'rgba(255,107,138,.65)'),
      borderColor:scores.map(s=>s>=80?'rgba(34,211,165,.9)':s>=60?'rgba(251,191,36,.9)':'rgba(255,107,138,.9)'),
      borderWidth:1,borderRadius:8,borderSkipped:false
    }]},
    options:{responsive:true,maintainAspectRatio:false,
      scales:{
        y:{min:0,max:100,grid:{color:gc},ticks:{color:tc,callback:v=>v+''},border:{display:false}},
        x:{grid:{display:false},ticks:{color:tc,font:{size:11}},border:{display:false}}
      },
      plugins:{legend:{display:false},tooltip:{callbacks:{label:ctx=>` Score: ${ctx.raw}/100`}}}}
  });
})();
{% endif %}

{% if timeline %}
(function(){
  const tl={{ timeline|tojson }};
  if(!tl.length) return;
  const labels=tl.map(r=>r.label||(r.test+' \u203a '+r.tool));
  const vals=tl.map(r=>r.latency||0);
  const costs=tl.map(r=>r.cost||0);
  const colors=tl.map(r=>r.success?'rgba(124,149,255,.65)':'rgba(255,107,138,.65)');
  const borders=tl.map(r=>r.success?'rgba(124,149,255,.9)':'rgba(255,107,138,.9)');
  const maxLatency=Math.max(...vals, 0);
  new Chart(document.getElementById('tlChart'),{
    type:'bar',
    data:{labels,datasets:[{label:'ms',data:vals,backgroundColor:colors,borderColor:borders,borderWidth:1,borderRadius:4,borderSkipped:false}]},
    options:{
      indexAxis:'y',responsive:true,maintainAspectRatio:false,
      scales:{
        x:{
          suggestedMax:maxLatency > 0 ? maxLatency * 1.15 : 1,
          grid:{color:'rgba(255,255,255,.04)'},
          ticks:{color:'rgba(122,143,166,.8)',callback:v=>v+'ms'},
          border:{display:false}
        },
        y:{grid:{display:false},ticks:{color:'rgba(122,143,166,.8)',font:{size:11}},border:{display:false}}
      },
      plugins:{legend:{display:false},tooltip:{callbacks:{
        label:ctx=>` ${ctx.raw}ms`,
        afterLabel:ctx=>` Cost: $${(costs[ctx.dataIndex] || 0).toFixed(6)}`,
        title:ctx=>ctx[0].label
      }}}
    }
  });
})();
{% endif %}

{% if compare %}
(function(){
  const labels={{ compare.labels|tojson }};
  const passRates={{ compare.runs|map(attribute='pass_rate')|list|tojson }};
  const avgScores={{ compare.runs|map(attribute='avg_score')|list|tojson }};
  const tc='rgba(122,143,166,.8)',gc='rgba(255,255,255,.05)';
  const colors=['rgba(124,149,255,.7)','rgba(34,211,165,.7)','rgba(248,107,138,.7)','rgba(251,191,36,.7)','rgba(192,132,252,.7)'];
  const borders=['rgba(124,149,255,.9)','rgba(34,211,165,.9)','rgba(248,107,138,.9)','rgba(251,191,36,.9)','rgba(192,132,252,.9)'];
  const opts={responsive:true,maintainAspectRatio:false,
    scales:{y:{grid:{color:gc},ticks:{color:tc},border:{display:false}},x:{grid:{display:false},ticks:{color:tc,font:{size:11}},border:{display:false}}},
    plugins:{legend:{display:false}}};
  new Chart(document.getElementById('cmpPassRate'),{type:'bar',
    data:{labels,datasets:[{label:'Pass Rate %',data:passRates,
      backgroundColor:colors.slice(0,labels.length),borderColor:borders.slice(0,labels.length),
      borderWidth:1,borderRadius:8,borderSkipped:false}]},
    options:{...opts,scales:{...opts.scales,y:{...opts.scales.y,min:0,max:100}}}});
  new Chart(document.getElementById('cmpScore'),{type:'bar',
    data:{labels,datasets:[{label:'Avg Score',data:avgScores,
      backgroundColor:colors.slice(0,labels.length),borderColor:borders.slice(0,labels.length),
      borderWidth:1,borderRadius:8,borderSkipped:false}]},
    options:{...opts,scales:{...opts.scales,y:{...opts.scales.y,min:0,max:100}}}});
})();
{% endif %}
</script>
</body>
</html>"""
