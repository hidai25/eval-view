"""HTML/Jinja2 template for the visual report.

Kept in its own module so visualization/generators.py stays focused on
data-shaping and rendering logic. Edits here change the report layout.
"""

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
    {% if adapter_compare and adapter_compare.enabled %}<button class="tab {% if default_tab == 'adapter-compare' %}on{% endif %}" onclick="show('adapter-compare',this)">Compare <span style="font-size:10px;opacity:.7;margin-left:3px">{{ adapter_compare.adapters|length }} models</span></button>{% endif %}
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
    {% if active_tags %}
    <div style="display:flex;flex-wrap:wrap;gap:6px;margin:0 0 14px 2px">
      <span class="badge b-blue">Filtered by tags</span>
      {% for tag in active_tags %}<span class="badge b-cyan">{{ tag }}</span>{% endfor %}
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
    {% if behavior_summary %}
    <div class="card">
      <div class="card-title">Behavior Summary</div>
      <table class="ev-table">
        <thead><tr><th>Behavior</th><th>Total</th><th>Passed</th><th>Changed</th><th>Regressions</th><th>Healed</th></tr></thead>
        <tbody>
          {% for row in behavior_summary %}
          <tr>
            <td style="font-weight:700">{{ row.tag }}</td>
            <td class="mono num">{{ row.total }}</td>
            <td class="mono num" style="color:var(--green-bright)">{{ row.passed }}</td>
            <td class="mono num" style="color:var(--yellow-bright)">{{ row.changed }}</td>
            <td class="mono num" style="color:var(--red-bright)">{{ row.regressions }}</td>
            <td class="mono num" style="color:var(--blue-bright)">{{ row.healed }}</td>
          </tr>
          {% endfor %}
        </tbody>
      </table>
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
            {% for tag in t.tags %}<span class="badge b-blue">{{ tag }}</span>{% endfor %}
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
          {% if t.hallucination or t.safety or t.pii or t.forbidden_tools or t.anomaly_report or t.trust_report or t.coherence_report %}
          <div style="display:flex;flex-wrap:wrap;gap:6px;margin-top:10px">
            {% if t.hallucination %}{% if t.hallucination.has_hallucination %}<span class="badge b-red" title="Extracts factual claims from the agent response, then verifies each claim against tool outputs. Score = supported claims / total claims.">🔮 Hallucination detected · {{ (t.hallucination.confidence * 100)|round(0)|int }}%{% if t.hallucination.details %} · {{ t.hallucination.details.split('\n')[0]|replace('Faithfulness: ', '') }}{% endif %}{% if judge_usage and judge_usage.model %} · {{ judge_usage.model }}{% endif %}</span>{% else %}<span class="badge b-green" title="Extracts factual claims from the agent response, then verifies each claim against tool outputs. Score = supported claims / total claims.">🔮 No hallucination{% if t.hallucination.details %} · {{ t.hallucination.details.split('\n')[0]|replace('Faithfulness: ', '') }}{% endif %}{% if judge_usage and judge_usage.model %} · {{ judge_usage.model }}{% endif %}</span>{% endif %}{% endif %}
            {% if t.safety %}{% if t.safety.is_safe %}<span class="badge b-green">🛡 Safe</span>{% else %}<span class="badge b-red">🛡 Unsafe: {{ t.safety.categories|join(', ') }}</span>{% endif %}{% endif %}
            {% if t.pii %}{% if t.pii.has_pii %}<span class="badge b-yellow">🔒 PII detected</span>{% else %}<span class="badge b-green">🔒 No PII</span>{% endif %}{% endif %}
            {% if t.forbidden_tools %}{% if t.forbidden_tools.violations %}<span class="badge b-red">⛔ Forbidden: {{ t.forbidden_tools.violations|join(', ') }}</span>{% else %}<span class="badge b-green">⛔ No violations</span>{% endif %}{% endif %}
            {% if t.anomaly_report %}{% if t.anomaly_report.anomalies %}<span class="badge b-red">🔄 {{ t.anomaly_report.anomalies|length }} anomal{{ 'y' if t.anomaly_report.anomalies|length == 1 else 'ies' }}</span>{% else %}<span class="badge b-green">🔄 No anomalies</span>{% endif %}{% endif %}
            {% if t.trust_report %}<span class="badge {% if t.trust_report.trust_score < 0.5 %}b-red{% elif t.trust_report.trust_score < 0.8 %}b-yellow{% else %}b-green{% endif %}">🔐 Trust: {{ (t.trust_report.trust_score * 100)|round|int }}%</span>{% endif %}
            {% if t.coherence_report %}{% if t.coherence_report.issues %}<span class="badge b-yellow">🔗 {{ t.coherence_report.issues|length }} coherence issue{{ 's' if t.coherence_report.issues|length != 1 }}</span>{% else %}<span class="badge b-green">🔗 Coherent</span>{% endif %}{% endif %}
          </div>
          {% if t.hallucination and t.hallucination.has_hallucination and t.hallucination.details %}<div style="background:rgba(168,85,247,.06);border:1px solid rgba(168,85,247,.15);border-radius:var(--r-xs);padding:9px 12px;margin-top:8px;font-size:11px;color:var(--text-3)"><span style="font-weight:600;color:var(--text-2)">Unsupported claims:</span> {{ t.hallucination.details[:500] }}{% if t.hallucination.details|length > 500 %}...{% endif %}</div>{% endif %}
          {% if t.anomaly_report and t.anomaly_report.anomalies %}<div style="background:rgba(239,68,68,.06);border:1px solid rgba(239,68,68,.15);border-radius:var(--r-xs);padding:9px 12px;margin-top:8px;font-size:11px;color:var(--text-3)"><span style="font-weight:600;color:var(--text-2)">Behavioral anomalies:</span>{% for a in t.anomaly_report.anomalies[:5] %}<br>• <b>{{ a.pattern }}</b>: {{ a.description[:150] }}{% endfor %}</div>{% endif %}
          {% if t.trust_report and t.trust_report.trust_score < 1.0 %}<div style="background:rgba(245,158,11,.06);border:1px solid rgba(245,158,11,.15);border-radius:var(--r-xs);padding:9px 12px;margin-top:8px;font-size:11px;color:var(--text-3)"><span style="font-weight:600;color:var(--text-2)">Trust:</span> {{ (t.trust_report.trust_score * 100)|round|int }}% — {{ t.trust_report.summary }}{% if t.trust_report.flags %}{% for f in t.trust_report.flags[:3] %}<br>• <b>{{ f.check }}</b> ({{ f.severity }}): {{ f.description[:120] }}{% endfor %}{% endif %}</div>{% endif %}
          {% if t.coherence_report and t.coherence_report.issues %}<div style="background:rgba(59,130,246,.06);border:1px solid rgba(59,130,246,.15);border-radius:var(--r-xs);padding:9px 12px;margin-top:8px;font-size:11px;color:var(--text-3)"><span style="font-weight:600;color:var(--text-2)">Coherence ({{ (t.coherence_report.coherence_score * 100)|round|int }}%):</span>{% for i in t.coherence_report.issues[:5] %}<br>• Turn {{ i.turn_index }}: <b>{{ i.category }}</b> — {{ i.description[:120] }}{% endfor %}</div>{% endif %}
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
          {% for tag in d.tags %}<span class="badge b-blue">{{ tag }}</span>{% endfor %}
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
        {% if d.root_cause_narrative %}
        <div style="padding:12px 18px;border-top:1px solid var(--border);background:rgba(6,182,212,0.06)">
          <div class="col-title" style="margin-bottom:6px;color:#06b6d4">🔍 Analysis</div>
          <div style="font-size:12px;color:var(--text-2);line-height:1.6">{{ d.root_cause_narrative }}</div>
        </div>
        {% endif %}
        {% if d.root_cause_summary %}
        <div style="padding:12px 18px;border-top:1px solid var(--border);font-size:12px;color:var(--text-2)">
          <div class="col-title" style="margin-bottom:6px">Why This Changed</div>
          <div style="display:flex;flex-wrap:wrap;gap:6px;margin-bottom:8px">
            {% if d.root_cause_category %}<span class="badge b-yellow">{{ d.root_cause_category }}</span>{% endif %}
          </div>
          <div>{{ d.root_cause_summary }}</div>
          {% if d.root_cause_fix %}<div style="margin-top:6px;color:var(--text-3)">Suggested fix: {{ d.root_cause_fix }}</div>{% endif %}
          {% if d.root_cause_ai %}<div style="margin-top:8px;padding:6px 8px;background:rgba(6,182,212,0.08);border-radius:4px"><span style="color:#06b6d4">🤖 AI:</span> {{ d.root_cause_ai }}</div>{% endif %}
        </div>
        {% endif %}
        {% if d.recommendations %}
        <div style="margin-top:8px;padding:8px 12px;background:rgba(59,130,246,0.08);border:1px solid rgba(59,130,246,0.2);border-radius:8px">
          <div style="font-size:12px;font-weight:600;color:#60a5fa;margin-bottom:4px">Suggested Fixes</div>
          {% for rec in d.recommendations %}
          <div style="margin-top:4px;font-size:12px">
            <span style="color:{% if rec.confidence == 'high' %}#34d399{% elif rec.confidence == 'medium' %}#fbbf24{% else %}#94a3b8{% endif %}">●</span>
            <strong>{{ rec.action }}</strong> <span style="color:#64748b">({{ rec.category }})</span>
            <div style="color:#94a3b8;margin-left:16px;font-size:11px">{{ rec.detail }}</div>
          </div>
          {% endfor %}
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

  <!-- ═══════════ ADAPTER COMPARE ═══════════ -->
  {% if adapter_compare and adapter_compare.enabled %}
  <div id="p-adapter-compare" class="panel {% if default_tab == 'adapter-compare' %}on{% endif %}">
    <div class="card">
      <div class="card-title">Model Comparison &mdash; {{ adapter_compare.adapters|length }} models &times; {{ adapter_compare.tasks|length }} tasks</div>
      <table class="ev-table">
        <thead>
          <tr>
            <th style="width:160px">Task</th>
            {% for adapter in adapter_compare.adapters %}
            <th colspan="3" style="text-align:center;border-left:1px solid var(--border);color:var(--text-2);padding:8px 10px">{{ adapter }}</th>
            {% endfor %}
          </tr>
          <tr>
            <th></th>
            {% for adapter in adapter_compare.adapters %}
            <th style="text-align:center;border-left:1px solid var(--border);font-size:9px">Score</th>
            <th style="text-align:center;font-size:9px">Latency</th>
            <th style="text-align:center;font-size:9px">Tools</th>
            {% endfor %}
          </tr>
        </thead>
        <tbody>
          {% for task in adapter_compare.tasks %}
          {% set row = adapter_compare.rows[task] %}
          <tr>
            <td class="mono">{{ task }}</td>
            {% for adapter in adapter_compare.adapters %}
            {% if adapter in row %}
            {% set cell = row[adapter] %}
            <td style="text-align:center;border-left:1px solid var(--border)">
              <span style="display:inline-block;padding:2px 8px;border-radius:12px;font-size:11px;font-weight:700;
                background:{% if cell.score >= 80 %}rgba(16,185,129,.18){% elif cell.score >= 60 %}rgba(245,158,11,.18){% else %}rgba(239,68,68,.18){% endif %};
                color:{% if cell.score >= 80 %}var(--green-bright){% elif cell.score >= 60 %}var(--yellow-bright){% else %}var(--red-bright){% endif %}">
                {{ cell.score }}{% if not cell.passed %} ✗{% endif %}
              </span>
            </td>
            <td style="text-align:center;color:var(--text-3);font-size:12px">{{ (cell.latency_ms / 1000)|round(1) }}s</td>
            <td style="text-align:center;color:var(--text-3);font-size:12px">{{ cell.tool_accuracy|round|int }}%</td>
            {% else %}
            <td colspan="3" style="text-align:center;color:var(--text-4);border-left:1px solid var(--border)">—</td>
            {% endif %}
            {% endfor %}
          </tr>
          {% endfor %}
        </tbody>
        <tfoot>
          <tr style="border-top:1px solid var(--border)">
            <td style="color:var(--text-3);font-size:11px;font-weight:600">Avg Score</td>
            {% for adapter in adapter_compare.adapters %}
            <td colspan="3" style="text-align:center;font-weight:700;font-size:13px;border-left:1px solid var(--border);
              color:{% set s = adapter_compare.totals[adapter].avg_score %}{% if s != '—' and s >= 80 %}var(--green-bright){% elif s != '—' and s >= 60 %}var(--yellow-bright){% elif s != '—' %}var(--red-bright){% else %}var(--text-3){% endif %}">
              {{ adapter_compare.totals[adapter].avg_score }}
            </td>
            {% endfor %}
          </tr>
          <tr>
            <td style="color:var(--text-4);font-size:11px">Avg Latency</td>
            {% for adapter in adapter_compare.adapters %}
            <td colspan="3" style="text-align:center;color:var(--text-3);font-size:12px;border-left:1px solid var(--border)">{{ adapter_compare.totals[adapter].avg_lat_s }}s</td>
            {% endfor %}
          </tr>
          <tr>
            <td style="color:var(--text-4);font-size:11px">Pass Rate</td>
            {% for adapter in adapter_compare.adapters %}
            <td colspan="3" style="text-align:center;color:var(--text-3);font-size:12px;border-left:1px solid var(--border)">{{ adapter_compare.totals[adapter].pass_rate }}</td>
            {% endfor %}
          </tr>
          <tr>
            <td style="color:var(--text-4);font-size:11px">Total Cost</td>
            {% for adapter in adapter_compare.adapters %}
            <td colspan="3" style="text-align:center;color:var(--text-3);font-size:12px;border-left:1px solid var(--border)">{{ adapter_compare.totals[adapter].cost_display }}</td>
            {% endfor %}
          </tr>
        </tfoot>
      </table>
    </div>
  </div>
  {% endif %}

  <!-- ═══════════ COMPARE RUNS ═══════════ -->
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
  </div>
  {% endif %}
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
