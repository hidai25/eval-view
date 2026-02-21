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
    """Escape characters that break Mermaid labels."""
    return s.replace('"', "'").replace("\n", " ").replace(":", ";").replace("<", "").replace(">", "")


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


_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en" data-theme="dark">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{{ title }}</title>
<script src="https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4/dist/chart.umd.min.js"></script>
<style>
:root{--bg:#0d1117;--bg2:#161b22;--bg3:#21262d;--border:#30363d;--text:#e6edf3;--muted:#8b949e;--green:#3fb950;--red:#f85149;--yellow:#d29922;--blue:#58a6ff;--purple:#a371f7;--radius:8px;--font:'Inter',-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif}
*{box-sizing:border-box;margin:0;padding:0}
body{background:var(--bg);color:var(--text);font-family:var(--font);font-size:14px;line-height:1.6;min-height:100vh}
a{color:var(--blue);text-decoration:none}
/* Layout */
.header{background:var(--bg2);border-bottom:1px solid var(--border);padding:20px 32px;display:flex;align-items:center;justify-content:space-between;position:sticky;top:0;z-index:100}
.header-left h1{font-size:18px;font-weight:600;display:flex;align-items:center;gap:10px}
.header-left h1 .logo{color:var(--blue)}
.header-meta{font-size:12px;color:var(--muted);margin-top:2px}
.badge{display:inline-flex;align-items:center;gap:4px;padding:2px 8px;border-radius:12px;font-size:11px;font-weight:600}
.badge-green{background:rgba(63,185,80,.15);color:var(--green)}
.badge-red{background:rgba(248,81,73,.15);color:var(--red)}
.badge-yellow{background:rgba(210,153,34,.15);color:var(--yellow)}
.badge-blue{background:rgba(88,166,255,.15);color:var(--blue)}
.badge-purple{background:rgba(163,113,247,.15);color:var(--purple)}
.container{max-width:1200px;margin:0 auto;padding:24px 32px}
/* Tabs */
.tabs{display:flex;gap:2px;border-bottom:1px solid var(--border);margin-bottom:28px}
.tab{background:none;border:none;color:var(--muted);cursor:pointer;font-size:13px;font-weight:500;padding:10px 18px;border-bottom:2px solid transparent;transition:all .15s;font-family:var(--font)}
.tab:hover{color:var(--text)}
.tab.active{color:var(--text);border-bottom-color:var(--blue)}
.tab-panel{display:none}
.tab-panel.active{display:block}
/* KPI Cards */
.kpi-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:16px;margin-bottom:28px}
.kpi-card{background:var(--bg2);border:1px solid var(--border);border-radius:var(--radius);padding:20px;transition:border-color .15s}
.kpi-card:hover{border-color:var(--blue)}
.kpi-label{font-size:11px;font-weight:600;color:var(--muted);text-transform:uppercase;letter-spacing:.05em;margin-bottom:8px}
.kpi-value{font-size:28px;font-weight:700;line-height:1}
.kpi-value.green{color:var(--green)}
.kpi-value.red{color:var(--red)}
.kpi-value.yellow{color:var(--yellow)}
.kpi-value.blue{color:var(--blue)}
.kpi-sub{font-size:11px;color:var(--muted);margin-top:4px}
/* Charts row */
.charts-row{display:grid;grid-template-columns:1fr 2fr;gap:16px;margin-bottom:28px}
.chart-card{background:var(--bg2);border:1px solid var(--border);border-radius:var(--radius);padding:20px}
.chart-card h3{font-size:13px;font-weight:600;color:var(--muted);margin-bottom:16px}
.chart-wrap{position:relative;height:180px}
/* Trace cards */
.trace-card{background:var(--bg2);border:1px solid var(--border);border-radius:var(--radius);margin-bottom:16px;overflow:hidden}
.trace-header{padding:14px 20px;display:flex;align-items:center;gap:10px;cursor:pointer;user-select:none}
.trace-header:hover{background:var(--bg3)}
.trace-name{font-weight:600;font-size:13px;flex:1}
.trace-body{padding:20px;border-top:1px solid var(--border);background:var(--bg)}
.mermaid-wrap{background:var(--bg2);border-radius:6px;padding:20px;overflow-x:auto;text-align:center}
/* Diff cards */
.diff-card{background:var(--bg2);border:1px solid var(--border);border-radius:var(--radius);margin-bottom:16px;overflow:hidden}
.diff-header{padding:14px 20px;display:flex;align-items:center;gap:10px}
.diff-name{font-weight:600;font-size:13px;flex:1}
.diff-body{display:grid;grid-template-columns:1fr 1fr;border-top:1px solid var(--border)}
.diff-col{padding:16px 20px}
.diff-col:first-child{border-right:1px solid var(--border)}
.diff-col-title{font-size:11px;font-weight:700;color:var(--muted);text-transform:uppercase;letter-spacing:.05em;margin-bottom:10px}
.tool-list{display:flex;flex-wrap:wrap;gap:6px;margin-bottom:10px}
.tool-tag{background:var(--bg3);border:1px solid var(--border);border-radius:4px;padding:2px 8px;font-size:11px;font-family:'SF Mono','Fira Code',monospace}
.tool-tag.added{border-color:var(--green);color:var(--green);background:rgba(63,185,80,.1)}
.tool-tag.removed{border-color:var(--red);color:var(--red);background:rgba(248,81,73,.1)}
.output-box{background:var(--bg);border:1px solid var(--border);border-radius:4px;padding:10px 12px;font-size:12px;font-family:'SF Mono','Fira Code',monospace;white-space:pre-wrap;word-break:break-all;max-height:200px;overflow-y:auto;line-height:1.5;color:var(--muted)}
.diff-lines{background:var(--bg);border:1px solid var(--border);border-radius:4px;padding:8px;font-size:11px;font-family:'SF Mono','Fira Code',monospace;max-height:160px;overflow-y:auto;margin-top:8px}
.diff-lines .add{color:var(--green)}
.diff-lines .rem{color:var(--red)}
/* Timeline */
.timeline-bars{display:flex;flex-direction:column;gap:6px}
.timeline-row{display:flex;align-items:center;gap:10px}
.timeline-label{font-size:11px;color:var(--muted);width:200px;flex-shrink:0;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.timeline-bar-wrap{flex:1;background:var(--bg3);border-radius:3px;height:20px;position:relative;overflow:hidden}
.timeline-bar{height:100%;border-radius:3px;transition:width .5s ease;display:flex;align-items:center;padding-left:6px;font-size:10px;font-weight:600;color:#000}
.timeline-bar.ok{background:var(--green)}
.timeline-bar.err{background:var(--red)}
.timeline-ms{font-size:10px;color:var(--muted);width:60px;text-align:right;flex-shrink:0}
/* Similarity ring */
.sim-ring{display:inline-flex;align-items:center;gap:6px;font-size:12px}
/* Empty state */
.empty{text-align:center;padding:48px;color:var(--muted)}
.empty .icon{font-size:32px;margin-bottom:12px}
/* Scrollbar */
::-webkit-scrollbar{width:6px;height:6px}
::-webkit-scrollbar-track{background:var(--bg)}
::-webkit-scrollbar-thumb{background:var(--border);border-radius:3px}
</style>
</head>
<body>

<div class="header">
  <div class="header-left">
    <h1><span class="logo">◈</span> {{ title }}</h1>
    <div class="header-meta">Generated {{ generated_at }}{% if notes %} · {{ notes }}{% endif %}</div>
  </div>
  <div style="display:flex;gap:8px;align-items:center">
    {% if kpis %}
      {% if kpis.failed == 0 %}
        <span class="badge badge-green">✓ All Passing</span>
      {% else %}
        <span class="badge badge-red">{{ kpis.failed }} Failed</span>
      {% endif %}
      <span class="badge badge-blue">{{ kpis.total }} Tests</span>
    {% endif %}
  </div>
</div>

<div class="container">

  <div class="tabs">
    <button class="tab active" onclick="showTab('overview')">Overview</button>
    <button class="tab" onclick="showTab('trace')">Execution Trace</button>
    <button class="tab" onclick="showTab('diffs')">Diffs</button>
    <button class="tab" onclick="showTab('timeline')">Timeline</button>
  </div>

  <!-- ─── OVERVIEW ─── -->
  <div id="tab-overview" class="tab-panel active">
    {% if kpis %}
    <div class="kpi-grid">
      <div class="kpi-card">
        <div class="kpi-label">Pass Rate</div>
        <div class="kpi-value {% if kpis.pass_rate >= 80 %}green{% elif kpis.pass_rate >= 60 %}yellow{% else %}red{% endif %}">{{ kpis.pass_rate }}%</div>
        <div class="kpi-sub">{{ kpis.passed }}/{{ kpis.total }} tests</div>
      </div>
      <div class="kpi-card">
        <div class="kpi-label">Avg Score</div>
        <div class="kpi-value {% if kpis.avg_score >= 80 %}green{% elif kpis.avg_score >= 60 %}yellow{% else %}red{% endif %}">{{ kpis.avg_score }}</div>
        <div class="kpi-sub">out of 100</div>
      </div>
      <div class="kpi-card">
        <div class="kpi-label">Total Cost</div>
        <div class="kpi-value blue">${{ kpis.total_cost }}</div>
        <div class="kpi-sub">this run</div>
      </div>
      <div class="kpi-card">
        <div class="kpi-label">Avg Latency</div>
        <div class="kpi-value blue">{{ kpis.avg_latency_ms|int }}ms</div>
        <div class="kpi-sub">per test</div>
      </div>
    </div>

    <div class="charts-row">
      <div class="chart-card">
        <h3>Pass / Fail</h3>
        <div class="chart-wrap"><canvas id="donutChart"></canvas></div>
      </div>
      <div class="chart-card">
        <h3>Score per Test</h3>
        <div class="chart-wrap"><canvas id="barChart"></canvas></div>
      </div>
    </div>
    {% else %}
    <div class="empty"><div class="icon">📊</div>No results to display</div>
    {% endif %}
  </div>

  <!-- ─── TRACE ─── -->
  <div id="tab-trace" class="tab-panel">
    {% if traces %}
      {% for t in traces %}
      <div class="trace-card">
        <div class="trace-header" onclick="toggleSection('trace-{{ loop.index }}')">
          <span class="badge {% if t.passed %}badge-green{% else %}badge-red{% endif %}">
            {% if t.passed %}✓{% else %}✗{% endif %}
          </span>
          <span class="trace-name">{{ t.name }}</span>
          <span style="color:var(--muted);font-size:12px">▾</span>
        </div>
        <div id="trace-{{ loop.index }}" class="trace-body" {% if not loop.first %}style="display:none"{% endif %}>
          <div class="mermaid-wrap">
            <div class="mermaid">{{ t.diagram }}</div>
          </div>
        </div>
      </div>
      {% endfor %}
    {% else %}
      <div class="empty"><div class="icon">🔍</div>No trace data available</div>
    {% endif %}
  </div>

  <!-- ─── DIFFS ─── -->
  <div id="tab-diffs" class="tab-panel">
    {% if diff_rows %}
      {% for d in diff_rows %}
      <div class="diff-card">
        <div class="diff-header">
          {% if d.status == 'regression' %}
            <span class="badge badge-red">⬇ Regression</span>
          {% elif d.status == 'tools_changed' %}
            <span class="badge badge-yellow">⚠ Tools Changed</span>
          {% elif d.status == 'output_changed' %}
            <span class="badge badge-purple">~ Output Changed</span>
          {% else %}
            <span class="badge badge-green">✓ Passed</span>
          {% endif %}
          <span class="diff-name">{{ d.name }}</span>
          {% if d.score_delta != 0 %}
            <span class="badge {% if d.score_delta > 0 %}badge-green{% else %}badge-red{% endif %}">
              {% if d.score_delta > 0 %}+{% endif %}{{ d.score_delta }} pts
            </span>
          {% endif %}
          <span class="sim-ring" style="color:var(--muted)">
            similarity: <b style="color:{% if d.similarity >= 80 %}var(--green){% elif d.similarity >= 50 %}var(--yellow){% else %}var(--red){% endif %}">{{ d.similarity }}%</b>
          </span>
        </div>
        <div class="diff-body">
          <div class="diff-col">
            <div class="diff-col-title">Golden</div>
            <div class="tool-list">
              {% for t in d.golden_tools %}
                <span class="tool-tag {% if t not in d.actual_tools %}removed{% endif %}">{{ t }}</span>
              {% endfor %}
            </div>
            <div class="output-box">{{ d.golden_out }}</div>
          </div>
          <div class="diff-col">
            <div class="diff-col-title">Actual</div>
            <div class="tool-list">
              {% for t in d.actual_tools %}
                <span class="tool-tag {% if t not in d.golden_tools %}added{% endif %}">{{ t }}</span>
              {% endfor %}
            </div>
            <div class="output-box">{{ d.actual_out }}</div>
            {% if d.diff_lines %}
            <div class="diff-lines">
              {% for line in d.diff_lines %}
                {% if line.startswith('+') %}<div class="add">{{ line }}</div>
                {% elif line.startswith('-') %}<div class="rem">{{ line }}</div>
                {% else %}<div>{{ line }}</div>{% endif %}
              {% endfor %}
            </div>
            {% endif %}
          </div>
        </div>
      </div>
      {% endfor %}
    {% else %}
      <div class="empty"><div class="icon">✨</div>No diffs — run <code>evalview check</code> to compare against a baseline</div>
    {% endif %}
  </div>

  <!-- ─── TIMELINE ─── -->
  <div id="tab-timeline" class="tab-panel">
    {% if timeline %}
      {% set max_lat = namespace(v=1) %}
      {% for row in timeline %}{% if row.latency > max_lat.v %}{% set max_lat.v = row.latency %}{% endif %}{% endfor %}
      <div class="chart-card" style="margin-bottom:20px">
        <h3>Step Latencies (ms)</h3>
        <div class="timeline-bars">
          {% for row in timeline %}
          <div class="timeline-row">
            <div class="timeline-label" title="{{ row.test }} › {{ row.tool }}">{{ row.test }} › {{ row.tool }}</div>
            <div class="timeline-bar-wrap">
              <div class="timeline-bar {% if row.success %}ok{% else %}err{% endif %}"
                   style="width:{{ [(row.latency / max_lat.v * 100)|round|int, 2]|max }}%">
              </div>
            </div>
            <div class="timeline-ms">{{ row.latency }}ms</div>
          </div>
          {% endfor %}
        </div>
      </div>
    {% else %}
      <div class="empty"><div class="icon">⏱</div>No step timing data available</div>
    {% endif %}
  </div>

</div><!-- /container -->

<script>
mermaid.initialize({startOnLoad:true,theme:'dark',securityLevel:'loose'});

function showTab(name){
  document.querySelectorAll('.tab,.tab-panel').forEach(el=>el.classList.remove('active'));
  document.querySelector('[onclick="showTab(\''+name+'\')"]').classList.add('active');
  document.getElementById('tab-'+name).classList.add('active');
}

function toggleSection(id){
  const el=document.getElementById(id);
  el.style.display=el.style.display==='none'?'block':'none';
}

// Charts
{% if kpis %}
(function(){
  const passed={{ kpis.passed }}, failed={{ kpis.failed }};
  const scores={{ kpis.scores|tojson }};
  const names={{ kpis.test_names|tojson }};

  new Chart(document.getElementById('donutChart'),{
    type:'doughnut',
    data:{
      labels:['Passed','Failed'],
      datasets:[{data:[passed,failed],backgroundColor:['#3fb950','#f85149'],borderWidth:0}]
    },
    options:{responsive:true,maintainAspectRatio:false,plugins:{legend:{labels:{color:'#8b949e',font:{size:11}}}}}
  });

  new Chart(document.getElementById('barChart'),{
    type:'bar',
    data:{
      labels:names,
      datasets:[{
        label:'Score',
        data:scores,
        backgroundColor:scores.map(s=>s>=80?'#3fb950':s>=60?'#d29922':'#f85149'),
        borderRadius:4
      }]
    },
    options:{
      responsive:true,maintainAspectRatio:false,
      scales:{y:{min:0,max:100,grid:{color:'#21262d'},ticks:{color:'#8b949e'}},x:{grid:{display:false},ticks:{color:'#8b949e',font:{size:10}}}},
      plugins:{legend:{display:false}}
    }
  });
})();
{% endif %}
</script>
</body>
</html>"""
