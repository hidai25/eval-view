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
from datetime import datetime
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from evalview.core.types import EvaluationResult
    from evalview.core.diff import TraceDiff


# ── Mermaid helpers ────────────────────────────────────────────────────────────

def _mermaid_trace(result: "EvaluationResult") -> str:
    """Convert an ExecutionTrace into a Mermaid sequence diagram."""
    steps = []
    try:
        steps = result.trace.steps or []
    except AttributeError:
        pass

    if not steps:
        return "sequenceDiagram\n    Note over Agent: No tool calls"

    lines = ["sequenceDiagram"]
    lines.append("    participant User")
    lines.append("    participant Agent")

    seen_tools: Dict[str, str] = {}
    for step in steps:
        tool = getattr(step, "tool_name", None) or getattr(step, "step_name", "unknown")
        if tool not in seen_tools:
            alias = f"T{len(seen_tools)}"
            seen_tools[tool] = alias
            short = tool[:20]
            lines.append(f"    participant {alias} as {short}")

    # Input
    query = getattr(result, "input_query", "") or ""
    short_query = (query[:40] + "…") if len(query) > 40 else query
    lines.append(f"    User->>Agent: {_safe_mermaid(short_query)}")

    for step in steps:
        tool = getattr(step, "tool_name", None) or getattr(step, "step_name", "unknown")
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

    output = getattr(result, "actual_output", "") or ""
    short_out = (output[:40] + "…") if len(output) > 40 else output
    lines.append(f"    Agent-->>User: {_safe_mermaid(short_out)}")

    return "\n".join(lines)


def _safe_mermaid(s: str) -> str:
    """Strip everything except safe alphanumeric + basic punctuation for Mermaid labels."""
    import re
    s = s.replace("\n", " ").replace("\r", "")
    # Keep only safe chars — anything else breaks Mermaid parser
    s = re.sub(r'[^\w\s\.\-_/]', '', s)
    return s[:40].strip() or "..."


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
    }


# ── Diff helpers ───────────────────────────────────────────────────────────────

def _diff_rows(diffs: List["TraceDiff"]) -> List[Dict[str, Any]]:
    rows = []
    for d in diffs:
        status = str(getattr(d, "overall_severity", "passed")).lower().replace("diffstatus.", "")
        golden_tools = getattr(d, "golden_tools", []) or []
        actual_tools = getattr(d, "actual_tools", []) or []
        output_diff = getattr(d, "output_diff", None)
        similarity = round(getattr(output_diff, "similarity", 1.0) * 100, 1) if output_diff else 100.0
        golden_out = getattr(output_diff, "golden_preview", "") if output_diff else ""
        actual_out = getattr(output_diff, "actual_preview", "") if output_diff else ""
        diff_lines = getattr(output_diff, "diff_lines", []) if output_diff else []
        score_delta = getattr(d, "score_diff", 0.0) or 0.0
        rows.append({
            "name": d.test_name,
            "status": status,
            "score_delta": round(score_delta, 1),
            "similarity": similarity,
            "golden_tools": golden_tools,
            "actual_tools": actual_tools,
            "golden_out": golden_out[:600],
            "actual_out": actual_out[:600],
            "diff_lines": diff_lines[:50],
        })
    return rows


# ── Timeline helpers ───────────────────────────────────────────────────────────

def _timeline_data(results: List["EvaluationResult"]) -> List[Dict[str, Any]]:
    rows = []
    for r in results:
        try:
            steps = r.trace.steps or []
            for step in steps:
                lat = getattr(step.metrics, "latency", 0) if hasattr(step, "metrics") else 0
                rows.append({
                    "test": r.test_case[:20],
                    "tool": getattr(step, "tool_name", "unknown")[:20],
                    "latency": round(lat, 1),
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
) -> str:
    """Generate a self-contained visual HTML report.

    Args:
        results: List of EvaluationResult objects.
        diffs: Optional list of TraceDiff objects for diff tab.
        output_path: Where to write the HTML (default: .evalview/reports/<timestamp>.html).
        auto_open: If True, open the report in the default browser.
        title: Report title shown in the header.
        notes: Optional free-text note shown in the header.

    Returns:
        Absolute path to the generated HTML file.
    """
    if output_path is None:
        os.makedirs(".evalview/reports", exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f".evalview/reports/{ts}.html"

    kpis = _kpis(results)
    traces = [
        {"name": r.test_case, "diagram": _mermaid_trace(r), "passed": r.passed}
        for r in results
    ]
    diff_rows = _diff_rows(diffs or [])
    timeline = _timeline_data(results)

    html = _render_template(
        title=title,
        notes=notes or "",
        generated_at=datetime.now().strftime("%Y-%m-%d %H:%M"),
        kpis=kpis,
        traces=traces,
        diff_rows=diff_rows,
        timeline=timeline,
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
.item-body{
  padding:20px;border-top:1px solid var(--border);
  background:rgba(0,0,0,.25);
}
.mermaid-box{
  background:rgba(0,0,0,.4);border:1px solid var(--border);
  border-radius:10px;padding:24px;overflow-x:auto;text-align:center;
  min-height:80px;display:flex;align-items:center;justify-content:center;
}
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
          <span class="chevron">▾</span>
        </div>
        <div id="tr{{ loop.index }}" class="item-body" {% if not loop.first %}style="display:none"{% endif %}>
          <div class="mermaid-box"><div class="mermaid">{{ t.diagram }}</div></div>
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
          <span class="sim">similarity <b style="color:{% if d.similarity >= 80 %}var(--green){% elif d.similarity >= 50 %}var(--yellow){% else %}var(--red){% endif %}">{{ d.similarity }}%</b></span>
        </div>
        <div class="diff-cols">
          <div class="diff-col">
            <div class="col-title">Golden</div>
            <div class="tags">{% for t in d.golden_tools %}<span class="tag {% if t not in d.actual_tools %}rem{% endif %}">{{ t }}</span>{% endfor %}</div>
            <div class="outbox">{{ d.golden_out }}</div>
          </div>
          <div class="diff-col">
            <div class="col-title">Actual</div>
            <div class="tags">{% for t in d.actual_tools %}<span class="tag {% if t not in d.golden_tools %}add{% endif %}">{{ t }}</span>{% endfor %}</div>
            <div class="outbox">{{ d.actual_out }}</div>
            {% if d.diff_lines %}
            <div class="difflines">{% for line in d.diff_lines %}{% if line.startswith('+') %}<div class="a">{{ line }}</div>{% elif line.startswith('-') %}<div class="r">{{ line }}</div>{% else %}<div>{{ line }}</div>{% endif %}{% endfor %}</div>
            {% endif %}
          </div>
        </div>
      </div>
      {% endfor %}
    {% else %}
      <div class="empty"><span class="empty-icon">✨</span>No diffs yet — run <code style="background:rgba(255,255,255,.08);padding:2px 6px;border-radius:4px">evalview check</code> to compare against a baseline</div>
    {% endif %}
  </div>

  <!-- TIMELINE -->
  <div id="p-timeline" class="panel">
    {% if timeline %}
      {% set mx = namespace(v=1) %}
      {% for row in timeline %}{% if row.latency > mx.v %}{% set mx.v = row.latency %}{% endif %}{% endfor %}
      <div class="card">
        <div class="card-title">Step Latencies</div>
        {% for row in timeline %}
        <div class="tl-row">
          <div class="tl-label" title="{{ row.test }} › {{ row.tool }}">{{ row.test }} › {{ row.tool }}</div>
          <div class="tl-track"><div class="tl-fill {% if row.success %}ok{% else %}err{% endif %}" style="width:{{ [(row.latency / mx.v * 100)|round|int, 2]|max }}%"></div></div>
          <div class="tl-ms">{{ row.latency }}ms</div>
        </div>
        {% endfor %}
      </div>
    {% else %}
      <div class="empty"><span class="empty-icon">⏱</span>No step timing data</div>
    {% endif %}
  </div>

</main>

<script>
mermaid.initialize({startOnLoad:true,theme:'dark',securityLevel:'loose',
  sequence:{actorFontFamily:'Inter,sans-serif',noteFontFamily:'Inter,sans-serif',messageFontFamily:'Inter,sans-serif'}});

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
</script>
</body>
</html>"""
