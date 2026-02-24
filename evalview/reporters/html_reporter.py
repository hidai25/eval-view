"""HTML report generator with interactive Plotly charts."""

import json
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any

try:
    from jinja2 import Environment, BaseLoader
    JINJA2_AVAILABLE = True
except ImportError:
    JINJA2_AVAILABLE = False

try:
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

from evalview.core.types import EvaluationResult


class HTMLReporter:
    """Generate interactive HTML reports from evaluation results."""

    def __init__(self):
        if not JINJA2_AVAILABLE:
            raise ImportError(
                "jinja2 is required for HTML reports. Install with: pip install jinja2"
            )

    def generate(
        self,
        results: List[EvaluationResult],
        output_path: str,
        title: str = "EvalView Test Report",
    ) -> str:
        """
        Generate an HTML report from evaluation results.

        Args:
            results: List of evaluation results
            output_path: Path to write the HTML file
            title: Report title

        Returns:
            Path to the generated report
        """
        summary = self._compute_summary(results)
        charts = self._generate_charts(results) if PLOTLY_AVAILABLE else {}

        html = self._render_template(
            results=results,
            summary=summary,
            charts=charts,
            title=title,
            timestamp=datetime.now().isoformat(),
            plotly_available=PLOTLY_AVAILABLE,
        )

        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(html)

        return str(output)

    def _compute_summary(self, results: List[EvaluationResult]) -> Dict[str, Any]:
        """Compute summary statistics from results."""
        if not results:
            return {
                "total": 0,
                "passed": 0,
                "failed": 0,
                "pass_rate": 0,
                "avg_score": 0,
                "total_cost": 0,
                "total_latency": 0,
                "avg_latency": 0,
            }

        passed = sum(1 for r in results if r.passed)
        failed = len(results) - passed
        scores = [r.score for r in results]
        costs = [r.trace.metrics.total_cost for r in results]
        latencies = [r.trace.metrics.total_latency for r in results]

        return {
            "total": len(results),
            "passed": passed,
            "failed": failed,
            "pass_rate": round(passed / len(results) * 100, 1) if results else 0,
            "avg_score": round(sum(scores) / len(scores), 1) if scores else 0,
            "min_score": round(min(scores), 1) if scores else 0,
            "max_score": round(max(scores), 1) if scores else 0,
            "total_cost": round(sum(costs), 4),
            "avg_cost": round(sum(costs) / len(costs), 4) if costs else 0,
            "total_latency": round(sum(latencies), 0),
            "avg_latency": round(sum(latencies) / len(latencies), 0) if latencies else 0,
        }

    def _generate_charts(self, results: List[EvaluationResult]) -> Dict[str, str]:
        """Generate Plotly charts as JSON strings."""
        if not results:
            return {}

        charts = {}

        # Pass/Fail pie chart
        passed = sum(1 for r in results if r.passed)
        failed = len(results) - passed
        fig = go.Figure(data=[go.Pie(
            labels=["Passed", "Failed"],
            values=[passed, failed],
            marker_colors=["#22c55e", "#ef4444"],
            hole=0.4,
        )])
        fig.update_layout(
            title="Test Results",
            showlegend=True,
            margin=dict(t=40, b=20, l=20, r=20),
            height=300,
        )
        charts["pass_fail"] = fig.to_json()

        # Score distribution histogram
        scores = [r.score for r in results]
        fig = go.Figure(data=[go.Histogram(
            x=scores,
            nbinsx=10,
            marker_color="#3b82f6",
        )])
        fig.update_layout(
            title="Score Distribution",
            xaxis_title="Score",
            yaxis_title="Count",
            margin=dict(t=40, b=40, l=40, r=20),
            height=300,
        )
        charts["score_distribution"] = fig.to_json()

        # Cost breakdown bar chart
        test_names = [r.test_case[:20] + "..." if len(r.test_case) > 20 else r.test_case for r in results]
        costs = [r.trace.metrics.total_cost for r in results]
        fig = go.Figure(data=[go.Bar(
            x=test_names,
            y=costs,
            marker_color="#8b5cf6",
        )])
        fig.update_layout(
            title="Cost by Test",
            xaxis_title="Test",
            yaxis_title="Cost ($)",
            margin=dict(t=40, b=80, l=40, r=20),
            height=300,
            xaxis_tickangle=-45,
        )
        charts["cost_breakdown"] = fig.to_json()

        # Latency scatter plot
        latencies = [r.trace.metrics.total_latency for r in results]
        colors = ["#22c55e" if r.passed else "#ef4444" for r in results]
        fig = go.Figure(data=[go.Scatter(
            x=list(range(1, len(results) + 1)),
            y=latencies,
            mode="markers",
            marker=dict(size=12, color=colors),
            text=[r.test_case for r in results],
            hovertemplate="<b>%{text}</b><br>Latency: %{y:.0f}ms<extra></extra>",
        )])
        fig.update_layout(
            title="Latency by Test",
            xaxis_title="Test #",
            yaxis_title="Latency (ms)",
            margin=dict(t=40, b=40, l=40, r=20),
            height=300,
        )
        charts["latency_scatter"] = fig.to_json()

        # Evaluation breakdown stacked bar
        tool_acc = [r.evaluations.tool_accuracy.accuracy * 100 for r in results]
        output_qual = [r.evaluations.output_quality.score for r in results]
        seq_correct = [100 if r.evaluations.sequence_correctness.correct else 0 for r in results]

        fig = go.Figure()
        fig.add_trace(go.Bar(name="Tool Accuracy", x=test_names, y=tool_acc, marker_color="#22c55e"))
        fig.add_trace(go.Bar(name="Output Quality", x=test_names, y=output_qual, marker_color="#3b82f6"))
        fig.add_trace(go.Bar(name="Sequence", x=test_names, y=seq_correct, marker_color="#f59e0b"))
        fig.update_layout(
            title="Evaluation Breakdown",
            barmode="group",
            xaxis_tickangle=-45,
            margin=dict(t=40, b=80, l=40, r=20),
            height=350,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        charts["eval_breakdown"] = fig.to_json()

        return charts

    def _serialize_spans(self, result: EvaluationResult) -> list:
        """Serialize span data for the trace replay timeline.

        Prefers the rich ``TraceContext`` (LLM + tool spans with full prompt/
        completion data). Falls back to the legacy ``StepTrace`` list when no
        ``trace_context`` is available so the replay works for every adapter.

        Args:
            result: Evaluation result whose trace we will serialise.

        Returns:
            A list of plain dicts, one per span, safe for JSON / Jinja2 rendering.
        """
        def _truncate(value, limit: int = 600) -> str:
            s = str(value) if value is not None else ""
            return s[:limit] + " …" if len(s) > limit else s

        def _json_preview(value, limit: int = 600) -> str:
            try:
                s = json.dumps(value, default=str, ensure_ascii=False, indent=2)
            except Exception:
                s = str(value)
            return s[:limit] + " …" if len(s) > limit else s

        trace = result.trace

        # ── Rich path: TraceContext is available ────────────────────────────
        if trace.trace_context and trace.trace_context.spans:
            spans = []
            for sp in sorted(trace.trace_context.spans, key=lambda s: s.start_time):
                entry: dict = {
                    "kind": sp.kind.value,         # "agent" | "llm" | "tool"
                    "name": sp.name,
                    "duration_ms": round(sp.duration_ms or 0, 1),
                    "status": sp.status,           # "ok" | "error" | "unset"
                    "cost": sp.cost,
                    "error": sp.error_message or "",
                    "llm": None,
                    "tool": None,
                }
                if sp.llm:
                    entry["llm"] = {
                        "model": sp.llm.model,
                        "provider": sp.llm.provider,
                        "prompt_tokens": sp.llm.prompt_tokens,
                        "completion_tokens": sp.llm.completion_tokens,
                        "finish_reason": sp.llm.finish_reason or "",
                        "prompt": _truncate(sp.llm.prompt),
                        "completion": _truncate(sp.llm.completion),
                    }
                if sp.tool:
                    entry["tool"] = {
                        "tool_name": sp.tool.tool_name,
                        "parameters": _json_preview(sp.tool.parameters),
                        "result": _truncate(sp.tool.result),
                    }
                spans.append(entry)
            return spans

        # ── Fallback path: convert StepTrace list ───────────────────────────
        return [
            {
                "kind": "tool",
                "name": step.tool_name,
                "duration_ms": round(step.metrics.latency, 1),
                "status": "ok" if step.success else "error",
                "cost": step.metrics.cost,
                "error": step.error or "",
                "llm": None,
                "tool": {
                    "tool_name": step.tool_name,
                    "parameters": _json_preview(step.parameters),
                    "result": _truncate(step.output),
                },
            }
            for step in trace.steps
        ]

    def _render_template(
        self,
        results: List[EvaluationResult],
        summary: Dict[str, Any],
        charts: Dict[str, str],
        title: str,
        timestamp: str,
        plotly_available: bool,
    ) -> str:
        """Render the HTML template."""
        env = Environment(loader=BaseLoader())
        template = env.from_string(HTML_TEMPLATE)

        # Convert results to serializable format
        results_data = []
        for r in results:
            results_data.append({
                "test_case": r.test_case,
                "passed": r.passed,
                "score": r.score,
                "input_query": r.input_query or "",
                "actual_output": (r.actual_output or "")[:500],
                "tool_accuracy": round(r.evaluations.tool_accuracy.accuracy * 100, 1),
                "correct_tools": r.evaluations.tool_accuracy.correct,
                "missing_tools": r.evaluations.tool_accuracy.missing,
                "unexpected_tools": r.evaluations.tool_accuracy.unexpected,
                "output_quality": r.evaluations.output_quality.score,
                "output_rationale": r.evaluations.output_quality.rationale,
                "sequence_correct": r.evaluations.sequence_correctness.correct,
                "expected_sequence": r.evaluations.sequence_correctness.expected_sequence,
                "actual_sequence": r.evaluations.sequence_correctness.actual_sequence,
                "cost": round(r.trace.metrics.total_cost, 4),
                "latency": round(r.trace.metrics.total_latency, 0),
                "steps": len(r.trace.steps),
                "adapter": r.adapter_name or "http",
                # Forbidden tool violations (empty list = no violations or not configured)
                "forbidden_violations": (
                    r.evaluations.forbidden_tools.violations
                    if r.evaluations.forbidden_tools and not r.evaluations.forbidden_tools.passed
                    else []
                ),
                # Trace replay: prefer rich TraceContext spans; fallback to StepTrace list
                "spans": self._serialize_spans(r),
            })

        return template.render(
            title=title,
            timestamp=timestamp,
            summary=summary,
            results=results_data,
            charts=charts,
            plotly_available=plotly_available,
        )


# Embedded HTML template (no external file needed)
# fmt: off
HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{{ title }}</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css" rel="stylesheet">
    {% if plotly_available %}
    <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
    {% endif %}
    <style>
        :root {
            --pass-color: #22c55e;
            --fail-color: #ef4444;
            --primary-color: #3b82f6;
            --span-agent: #8b5cf6;
            --span-llm: #3b82f6;
            --span-tool: #f59e0b;
        }
        body { background-color: #f8fafc; }
        .card { border: none; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }
        .stat-card { text-align: center; padding: 1.5rem; }
        .stat-value { font-size: 2.5rem; font-weight: 700; }
        .stat-label { color: #64748b; font-size: 0.875rem; text-transform: uppercase; }
        .pass { color: var(--pass-color); }
        .fail { color: var(--fail-color); }
        .badge-pass { background-color: var(--pass-color); }
        .badge-fail { background-color: var(--fail-color); }
        .test-card { margin-bottom: 1rem; }
        .test-card .card-header { cursor: pointer; user-select: none; }
        .test-card .card-header:hover { background-color: #f1f5f9; }
        .score-badge { font-size: 1.25rem; font-weight: 600; }
        .tool-list { display: flex; flex-wrap: wrap; gap: 0.5rem; }
        .tool-badge { font-size: 0.75rem; padding: 0.25rem 0.5rem; }
        .output-preview {
            max-height: 150px;
            overflow-y: auto;
            background: #f8fafc;
            padding: 0.75rem;
            border-radius: 0.375rem;
            font-family: monospace;
            font-size: 0.875rem;
            white-space: pre-wrap;
            word-break: break-word;
        }
        .chart-container { min-height: 300px; }
        pre { white-space: pre-wrap; word-wrap: break-word; }

        /* ── Forbidden tool alert ─────────────────────────────────────────── */
        .forbidden-alert {
            background: #fef2f2;
            border: 1px solid #fca5a5;
            border-left: 4px solid #ef4444;
            border-radius: 0.375rem;
            padding: 0.75rem 1rem;
            margin-bottom: 1rem;
        }
        .forbidden-alert .violation-chip {
            display: inline-block;
            background: #fee2e2;
            border: 1px solid #fca5a5;
            color: #b91c1c;
            font-family: monospace;
            font-size: 0.8rem;
            font-weight: 600;
            padding: 0.2rem 0.5rem;
            border-radius: 0.25rem;
            margin: 0.15rem;
        }

        /* ── Trace replay timeline ────────────────────────────────────────── */
        .trace-timeline { list-style: none; padding: 0; margin: 0; }
        .trace-timeline li { margin-bottom: 0.5rem; }
        .span-row {
            display: flex;
            align-items: center;
            gap: 0.6rem;
            padding: 0.45rem 0.75rem;
            border-radius: 0.375rem;
            background: #f8fafc;
            border: 1px solid #e2e8f0;
            cursor: pointer;
            transition: background 0.15s;
        }
        .span-row:hover { background: #f1f5f9; }
        .span-row.error-span { border-left: 3px solid #ef4444; }
        .span-kind {
            font-size: 0.65rem;
            font-weight: 700;
            letter-spacing: 0.05em;
            text-transform: uppercase;
            padding: 0.2rem 0.45rem;
            border-radius: 0.25rem;
            color: #fff;
            min-width: 42px;
            text-align: center;
            flex-shrink: 0;
        }
        .kind-agent { background: var(--span-agent); }
        .kind-llm   { background: var(--span-llm); }
        .kind-tool  { background: var(--span-tool); color: #1c1c1c; }
        .span-name  { flex: 1; font-family: monospace; font-size: 0.875rem; font-weight: 500; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
        .span-meta  { display: flex; gap: 0.75rem; font-size: 0.75rem; color: #64748b; flex-shrink: 0; }
        .span-status-ok   { color: var(--pass-color); font-weight: 700; }
        .span-status-err  { color: var(--fail-color); font-weight: 700; }
        .span-detail {
            background: #0f172a;
            color: #e2e8f0;
            border-radius: 0.375rem;
            padding: 0.75rem 1rem;
            margin-top: 0.25rem;
            font-family: monospace;
            font-size: 0.8rem;
            overflow-x: auto;
        }
        .span-detail .detail-label {
            color: #94a3b8;
            font-size: 0.7rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            margin-top: 0.5rem;
            margin-bottom: 0.15rem;
            display: block;
        }
        .span-detail .detail-value {
            white-space: pre-wrap;
            word-break: break-word;
            color: #f8fafc;
        }
        .span-detail .detail-value.prompt-text { color: #93c5fd; }
        .span-detail .detail-value.completion-text { color: #86efac; }
        .token-pill {
            display: inline-block;
            background: #1e293b;
            border: 1px solid #334155;
            border-radius: 0.25rem;
            padding: 0.1rem 0.4rem;
            font-size: 0.7rem;
            margin-right: 0.25rem;
            color: #94a3b8;
        }
        .trace-summary {
            display: flex;
            gap: 1rem;
            margin-bottom: 0.75rem;
            font-size: 0.8rem;
            color: #64748b;
        }
        .trace-summary span { font-weight: 600; color: #334155; }
    </style>
</head>
<body>
    <nav class="navbar navbar-dark bg-dark mb-4">
        <div class="container">
            <span class="navbar-brand mb-0 h1">EvalView Report</span>
            <span class="text-light">{{ timestamp[:19] }}</span>
        </div>
    </nav>

    <div class="container">
        <!-- Summary Cards -->
        <div class="row mb-4">
            <div class="col-md-2">
                <div class="card stat-card">
                    <div class="stat-value">{{ summary.total }}</div>
                    <div class="stat-label">Total Tests</div>
                </div>
            </div>
            <div class="col-md-2">
                <div class="card stat-card">
                    <div class="stat-value pass">{{ summary.passed }}</div>
                    <div class="stat-label">Passed</div>
                </div>
            </div>
            <div class="col-md-2">
                <div class="card stat-card">
                    <div class="stat-value fail">{{ summary.failed }}</div>
                    <div class="stat-label">Failed</div>
                </div>
            </div>
            <div class="col-md-2">
                <div class="card stat-card">
                    <div class="stat-value" style="color: var(--primary-color)">{{ summary.pass_rate }}%</div>
                    <div class="stat-label">Pass Rate</div>
                </div>
            </div>
            <div class="col-md-2">
                <div class="card stat-card">
                    <div class="stat-value">{{ summary.avg_score }}</div>
                    <div class="stat-label">Avg Score</div>
                </div>
            </div>
            <div class="col-md-2">
                <div class="card stat-card">
                    <div class="stat-value">${{ summary.total_cost }}</div>
                    <div class="stat-label">Total Cost</div>
                </div>
            </div>
        </div>

        {% if plotly_available and charts %}
        <!-- Charts -->
        <div class="row mb-4">
            <div class="col-md-4">
                <div class="card">
                    <div class="card-body">
                        <div id="chart-pass-fail" class="chart-container"></div>
                    </div>
                </div>
            </div>
            <div class="col-md-4">
                <div class="card">
                    <div class="card-body">
                        <div id="chart-score-dist" class="chart-container"></div>
                    </div>
                </div>
            </div>
            <div class="col-md-4">
                <div class="card">
                    <div class="card-body">
                        <div id="chart-latency" class="chart-container"></div>
                    </div>
                </div>
            </div>
        </div>
        <div class="row mb-4">
            <div class="col-md-6">
                <div class="card">
                    <div class="card-body">
                        <div id="chart-cost" class="chart-container"></div>
                    </div>
                </div>
            </div>
            <div class="col-md-6">
                <div class="card">
                    <div class="card-body">
                        <div id="chart-eval" class="chart-container"></div>
                    </div>
                </div>
            </div>
        </div>
        {% endif %}

        <!-- Test Results -->
        <h4 class="mb-3">Test Results</h4>
        <div id="testResults">
            {% for result in results %}
            {% set ri = loop.index %}
            <div class="card test-card">
                <!-- Card header: click to collapse/expand -->
                <div class="card-header d-flex justify-content-between align-items-center"
                     data-bs-toggle="collapse" data-bs-target="#test-{{ ri }}">
                    <div class="d-flex align-items-center gap-2">
                        <span class="badge {{ 'badge-pass' if result.passed else 'badge-fail' }}">
                            {{ 'PASS' if result.passed else 'FAIL' }}
                        </span>
                        <strong>{{ result.test_case }}</strong>
                        <span class="text-muted small">({{ result.adapter }})</span>
                        {% if result.forbidden_violations %}
                        <span class="badge bg-danger ms-1" title="Forbidden tool called">&#9888; FORBIDDEN TOOL</span>
                        {% endif %}
                    </div>
                    <span class="score-badge {{ 'pass' if result.score >= 80 else 'fail' if result.score < 60 else 'text-warning' }}">
                        {{ result.score }}
                    </span>
                </div>

                <div id="test-{{ ri }}" class="collapse">
                    <div class="card-body">

                        <!-- Forbidden tool alert banner -->
                        {% if result.forbidden_violations %}
                        <div class="forbidden-alert">
                            <strong style="color:#b91c1c;">&#9888; Forbidden Tool Contract Violated</strong>
                            <p class="mb-1 mt-1 text-muted small">
                                The following tools were declared forbidden but were called by the agent.
                                This test hard-fails regardless of output quality.
                            </p>
                            {% for v in result.forbidden_violations %}
                            <span class="violation-chip">{{ v }}</span>
                            {% endfor %}
                        </div>
                        {% endif %}

                        <!-- Tabs: Evaluation | Trace Replay -->
                        <ul class="nav nav-tabs mb-3" id="tabs-{{ ri }}">
                            <li class="nav-item">
                                <button class="nav-link active" data-bs-toggle="tab"
                                        data-bs-target="#tab-eval-{{ ri }}">Evaluation</button>
                            </li>
                            <li class="nav-item">
                                <button class="nav-link" data-bs-toggle="tab"
                                        data-bs-target="#tab-trace-{{ ri }}">
                                    Trace Replay
                                    <span class="badge bg-secondary ms-1">{{ result.spans | length }}</span>
                                </button>
                            </li>
                        </ul>

                        <div class="tab-content">
                            <!-- ── Tab 1: Evaluation ──────────────────────── -->
                            <div class="tab-pane fade show active" id="tab-eval-{{ ri }}">
                                <div class="row">
                                    <div class="col-md-6">
                                        <h6>Input Query</h6>
                                        <div class="output-preview">{{ result.input_query }}</div>

                                        <h6 class="mt-3">Agent Output</h6>
                                        <div class="output-preview">{{ result.actual_output }}</div>
                                    </div>
                                    <div class="col-md-6">
                                        <h6>Evaluation Scores</h6>
                                        <table class="table table-sm">
                                            <tr>
                                                <td>Tool Accuracy</td>
                                                <td><strong>{{ result.tool_accuracy }}%</strong></td>
                                            </tr>
                                            <tr>
                                                <td>Output Quality</td>
                                                <td><strong>{{ result.output_quality }}</strong></td>
                                            </tr>
                                            <tr>
                                                <td>Sequence Correct</td>
                                                <td>
                                                    {% if result.sequence_correct %}
                                                    <span class="badge bg-success">Yes</span>
                                                    {% else %}
                                                    <span class="badge bg-danger">No</span>
                                                    {% endif %}
                                                </td>
                                            </tr>
                                            <tr>
                                                <td>Cost</td>
                                                <td>${{ result.cost }}</td>
                                            </tr>
                                            <tr>
                                                <td>Latency</td>
                                                <td>{{ result.latency }}ms</td>
                                            </tr>
                                            <tr>
                                                <td>Steps</td>
                                                <td>{{ result.steps }}</td>
                                            </tr>
                                        </table>

                                        <h6>Tools</h6>
                                        <div class="tool-list mb-2">
                                            {% for tool in result.correct_tools %}
                                            <span class="badge bg-success tool-badge">&#10003; {{ tool }}</span>
                                            {% endfor %}
                                            {% for tool in result.missing_tools %}
                                            <span class="badge bg-warning text-dark tool-badge">Missing: {{ tool }}</span>
                                            {% endfor %}
                                            {% for tool in result.unexpected_tools %}
                                            <span class="badge bg-secondary tool-badge">Extra: {{ tool }}</span>
                                            {% endfor %}
                                        </div>

                                        <h6>Sequence</h6>
                                        <small class="text-muted">Expected: {{ result.expected_sequence | join(' → ') or 'Any' }}</small><br>
                                        <small>Actual: {{ result.actual_sequence | join(' → ') or 'None' }}</small>
                                    </div>
                                </div>

                                {% if result.output_rationale %}
                                <div class="mt-3">
                                    <h6>LLM Judge Rationale</h6>
                                    <div class="output-preview">{{ result.output_rationale }}</div>
                                </div>
                                {% endif %}
                            </div><!-- /tab-eval -->

                            <!-- ── Tab 2: Trace Replay ────────────────────── -->
                            <div class="tab-pane fade" id="tab-trace-{{ ri }}">
                                {% if result.spans %}
                                <!-- Trace summary stats -->
                                <div class="trace-summary">
                                    {% set llm_spans = result.spans | selectattr("kind", "equalto", "llm") | list %}
                                    {% set tool_spans = result.spans | selectattr("kind", "equalto", "tool") | list %}
                                    <div>LLM calls <span>{{ llm_spans | length }}</span></div>
                                    <div>Tool calls <span>{{ tool_spans | length }}</span></div>
                                    <div>Total cost <span>${{ result.cost }}</span></div>
                                    <div>Total latency <span>{{ result.latency }}ms</span></div>
                                </div>

                                <ul class="trace-timeline">
                                    {% for span in result.spans %}
                                    {% set sid = "span-" ~ ri ~ "-" ~ loop.index %}
                                    <li>
                                        <!-- Span header row (click to expand) -->
                                        <div class="span-row {{ 'error-span' if span.status == 'error' else '' }}"
                                             data-bs-toggle="collapse" data-bs-target="#{{ sid }}">
                                            <span class="span-kind kind-{{ span.kind }}">{{ span.kind }}</span>
                                            <span class="span-name" title="{{ span.name }}">{{ span.name }}</span>
                                            <div class="span-meta">
                                                {% if span.duration_ms %}
                                                <span>{{ span.duration_ms }}ms</span>
                                                {% endif %}
                                                {% if span.cost %}
                                                <span>${{ "%.4f" | format(span.cost) }}</span>
                                                {% endif %}
                                                {% if span.llm %}
                                                <span class="token-pill">↑{{ span.llm.prompt_tokens }} ↓{{ span.llm.completion_tokens }} tok</span>
                                                {% endif %}
                                                <span class="{{ 'span-status-ok' if span.status == 'ok' else 'span-status-err' }}">
                                                    {{ '✓' if span.status == 'ok' else '✗' }}
                                                </span>
                                            </div>
                                        </div>

                                        <!-- Span detail panel -->
                                        <div id="{{ sid }}" class="collapse">
                                            <div class="span-detail">
                                                {% if span.error %}
                                                <span class="detail-label">Error</span>
                                                <div class="detail-value" style="color:#fca5a5;">{{ span.error }}</div>
                                                {% endif %}

                                                {% if span.llm %}
                                                <span class="detail-label">Model</span>
                                                <div class="detail-value">{{ span.llm.model }} ({{ span.llm.provider }})</div>

                                                <span class="detail-label">Tokens</span>
                                                <div class="detail-value">
                                                    <span class="token-pill">input {{ span.llm.prompt_tokens }}</span>
                                                    <span class="token-pill">output {{ span.llm.completion_tokens }}</span>
                                                    {% if span.llm.finish_reason %}
                                                    <span class="token-pill">finish: {{ span.llm.finish_reason }}</span>
                                                    {% endif %}
                                                </div>

                                                {% if span.llm.prompt %}
                                                <span class="detail-label">Prompt sent to model</span>
                                                <div class="detail-value prompt-text">{{ span.llm.prompt }}</div>
                                                {% endif %}

                                                {% if span.llm.completion %}
                                                <span class="detail-label">Model completion</span>
                                                <div class="detail-value completion-text">{{ span.llm.completion }}</div>
                                                {% endif %}
                                                {% endif %}

                                                {% if span.tool %}
                                                <span class="detail-label">Tool</span>
                                                <div class="detail-value">{{ span.tool.tool_name }}</div>

                                                <span class="detail-label">Parameters</span>
                                                <div class="detail-value">{{ span.tool.parameters }}</div>

                                                {% if span.tool.result %}
                                                <span class="detail-label">Result</span>
                                                <div class="detail-value">{{ span.tool.result }}</div>
                                                {% endif %}
                                                {% endif %}

                                                {% if span.kind == 'agent' %}
                                                <span class="detail-label">Kind</span>
                                                <div class="detail-value">Agent execution span (root)</div>
                                                {% endif %}
                                            </div>
                                        </div>
                                    </li>
                                    {% endfor %}
                                </ul>
                                {% else %}
                                <p class="text-muted">
                                    No span data captured. To enable trace replay, your adapter must
                                    populate <code>ExecutionTrace.trace_context</code> using
                                    <code>evalview.core.tracing.Tracer</code>.
                                </p>
                                {% endif %}
                            </div><!-- /tab-trace -->
                        </div><!-- /tab-content -->

                    </div><!-- /card-body -->
                </div><!-- /collapse -->
            </div><!-- /card -->
            {% endfor %}
        </div><!-- /testResults -->

        <footer class="text-center text-muted py-4">
            Generated by <a href="https://github.com/hidai25/eval-view">EvalView</a>
        </footer>
    </div>

    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/js/bootstrap.bundle.min.js"></script>
    {% if plotly_available and charts %}
    <script>
        {% if charts.pass_fail %}
        Plotly.newPlot('chart-pass-fail', JSON.parse('{{ charts.pass_fail | safe }}').data, JSON.parse('{{ charts.pass_fail | safe }}').layout, {responsive: true});
        {% endif %}
        {% if charts.score_distribution %}
        Plotly.newPlot('chart-score-dist', JSON.parse('{{ charts.score_distribution | safe }}').data, JSON.parse('{{ charts.score_distribution | safe }}').layout, {responsive: true});
        {% endif %}
        {% if charts.latency_scatter %}
        Plotly.newPlot('chart-latency', JSON.parse('{{ charts.latency_scatter | safe }}').data, JSON.parse('{{ charts.latency_scatter | safe }}').layout, {responsive: true});
        {% endif %}
        {% if charts.cost_breakdown %}
        Plotly.newPlot('chart-cost', JSON.parse('{{ charts.cost_breakdown | safe }}').data, JSON.parse('{{ charts.cost_breakdown | safe }}').layout, {responsive: true});
        {% endif %}
        {% if charts.eval_breakdown %}
        Plotly.newPlot('chart-eval', JSON.parse('{{ charts.eval_breakdown | safe }}').data, JSON.parse('{{ charts.eval_breakdown | safe }}').layout, {responsive: true});
        {% endif %}
    </script>
    {% endif %}
</body>
</html>
"""

# Diff report template
DIFF_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{{ title }}</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>
        body { background: #0d1117; color: #c9d1d9; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; }
        .container { max-width: 1400px; }
        .card { background: #161b22; border: 1px solid #30363d; border-radius: 6px; margin-bottom: 1rem; }
        .card-header { background: #21262d; border-bottom: 1px solid #30363d; padding: 12px 16px; }
        .status-regression { border-left: 4px solid #f85149; }
        .status-tools_changed { border-left: 4px solid #d29922; }
        .status-output_changed { border-left: 4px solid #8b949e; }
        .status-passed { border-left: 4px solid #3fb950; }
        .badge-regression { background: #f85149; }
        .badge-tools_changed { background: #d29922; }
        .badge-output_changed { background: #8b949e; }
        .badge-passed { background: #3fb950; }
        .diff-container { display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; }
        .diff-panel { background: #0d1117; border: 1px solid #30363d; border-radius: 6px; padding: 1rem; }
        .diff-panel h6 { color: #8b949e; margin-bottom: 0.5rem; }
        .tool-sequence { font-family: monospace; }
        .tool-added { color: #3fb950; }
        .tool-removed { color: #f85149; text-decoration: line-through; }
        .tool-changed { color: #d29922; }
        .output-preview { background: #0d1117; padding: 1rem; border-radius: 4px; font-family: monospace; font-size: 12px; max-height: 300px; overflow-y: auto; white-space: pre-wrap; word-break: break-word; }
        .metric { display: inline-block; margin-right: 1rem; padding: 4px 8px; background: #21262d; border-radius: 4px; }
        .metric-label { color: #8b949e; font-size: 11px; }
        .metric-value { font-weight: 600; }
        .score-up { color: #3fb950; }
        .score-down { color: #f85149; }
        .summary-card { background: linear-gradient(135deg, #21262d 0%, #161b22 100%); }
        h1, h2, h3, h4, h5 { color: #c9d1d9; }
        a { color: #58a6ff; }
    </style>
</head>
<body>
    <div class="container py-4">
        <h1 class="mb-4">Golden Diff Report</h1>
        <p class="text-muted">Generated: {{ timestamp }}</p>

        <!-- Summary -->
        <div class="card summary-card mb-4">
            <div class="card-body">
                <div class="row text-center">
                    <div class="col-md-3">
                        <h3>{{ summary.total }}</h3>
                        <small class="text-muted">Tests Compared</small>
                    </div>
                    <div class="col-md-3">
                        <h3 class="text-danger">{{ summary.regressions }}</h3>
                        <small class="text-muted">Regressions</small>
                    </div>
                    <div class="col-md-3">
                        <h3 class="text-warning">{{ summary.tools_changed + summary.output_changed }}</h3>
                        <small class="text-muted">Changed</small>
                    </div>
                    <div class="col-md-3">
                        <h3 class="text-success">{{ summary.passed }}</h3>
                        <small class="text-muted">Passed</small>
                    </div>
                </div>
            </div>
        </div>

        <!-- Diff Cards -->
        {% for diff in diffs %}
        <div class="card status-{{ diff.status }}">
            <div class="card-header d-flex justify-content-between align-items-center">
                <div>
                    <span class="badge badge-{{ diff.status }} me-2">{{ diff.status | upper }}</span>
                    <strong>{{ diff.test_name }}</strong>
                </div>
                <div>
                    <span class="metric">
                        <span class="metric-label">Score</span>
                        <span class="metric-value {% if diff.score_diff > 0 %}score-up{% elif diff.score_diff < 0 %}score-down{% endif %}">
                            {{ diff.actual_score | round(1) }}
                            {% if diff.score_diff != 0 %}({{ '%+.1f' | format(diff.score_diff) }}){% endif %}
                        </span>
                    </span>
                    <span class="metric">
                        <span class="metric-label">Similarity</span>
                        <span class="metric-value">{{ (diff.similarity * 100) | round(0) }}%</span>
                    </span>
                </div>
            </div>
            <div class="card-body">
                <!-- Tool Comparison -->
                <h6>Tool Sequence</h6>
                <div class="diff-container mb-3">
                    <div class="diff-panel">
                        <h6>Golden (Baseline)</h6>
                        <div class="tool-sequence">
                            {% for tool in diff.golden_tools %}
                            <span class="badge bg-secondary me-1">{{ tool }}</span>
                            {% endfor %}
                        </div>
                    </div>
                    <div class="diff-panel">
                        <h6>Actual (Current)</h6>
                        <div class="tool-sequence">
                            {% for tool in diff.actual_tools %}
                            <span class="badge bg-primary me-1">{{ tool }}</span>
                            {% endfor %}
                        </div>
                    </div>
                </div>

                {% if diff.tool_changes %}
                <div class="mb-3">
                    <h6>Tool Changes</h6>
                    {% for change in diff.tool_changes %}
                    <div class="tool-{{ change.type }}">
                        {% if change.type == 'added' %}+ {{ change.tool }}{% endif %}
                        {% if change.type == 'removed' %}- {{ change.tool }}{% endif %}
                        {% if change.type == 'changed' %}~ {{ change.from }} → {{ change.to }}{% endif %}
                    </div>
                    {% endfor %}
                </div>
                {% endif %}

                <!-- Output Comparison -->
                <h6>Output Comparison</h6>
                <div class="diff-container">
                    <div class="diff-panel">
                        <h6>Golden Output</h6>
                        <div class="output-preview">{{ diff.golden_output }}</div>
                    </div>
                    <div class="diff-panel">
                        <h6>Actual Output</h6>
                        <div class="output-preview">{{ diff.actual_output }}</div>
                    </div>
                </div>
            </div>
        </div>
        {% endfor %}

        <footer class="text-center text-muted py-4">
            Generated by <a href="https://github.com/hidai25/eval-view">EvalView</a>
        </footer>
    </div>
</body>
</html>
"""


class DiffReporter:
    """Generate HTML diff reports comparing actual vs golden traces."""

    def __init__(self):
        if not JINJA2_AVAILABLE:
            raise ImportError(
                "jinja2 is required for HTML reports. Install with: pip install jinja2"
            )

    def generate(
        self,
        diffs: list,  # List of TraceDiff objects
        results: List[EvaluationResult],
        output_path: str,
        title: str = "Golden Diff Report",
    ) -> str:
        """
        Generate an HTML diff report.

        Args:
            diffs: List of TraceDiff objects from diff engine
            results: List of EvaluationResult for additional context
            output_path: Path to write the HTML file
            title: Report title

        Returns:
            Path to the generated report
        """
        from evalview.core.golden import GoldenStore
        from evalview.core.diff import DiffStatus

        store = GoldenStore()

        # Build diff data for template
        diff_data = []
        regressions = 0
        tools_changed = 0
        output_changed = 0
        passed = 0

        for trace_diff in diffs:
            # Get corresponding result
            result = next((r for r in results if r.test_case == trace_diff.test_name), None)
            golden = store.load_golden(trace_diff.test_name)

            if trace_diff.overall_severity == DiffStatus.REGRESSION:
                status = "regression"
                regressions += 1
            elif trace_diff.overall_severity == DiffStatus.TOOLS_CHANGED:
                status = "tools_changed"
                tools_changed += 1
            elif trace_diff.overall_severity == DiffStatus.OUTPUT_CHANGED:
                status = "output_changed"
                output_changed += 1
            else:
                status = "passed"
                passed += 1

            # Build tool changes list
            tool_changes = []
            for td in trace_diff.tool_diffs:
                if td.type == "added":
                    tool_changes.append({"type": "added", "tool": td.actual_tool})
                elif td.type == "removed":
                    tool_changes.append({"type": "removed", "tool": td.golden_tool})
                elif td.type == "changed":
                    tool_changes.append({"type": "changed", "from": td.golden_tool, "to": td.actual_tool})

            diff_data.append({
                "test_name": trace_diff.test_name,
                "status": status,
                "score_diff": trace_diff.score_diff,
                "actual_score": result.score if result else 0,
                "similarity": trace_diff.output_diff.similarity if trace_diff.output_diff else 1.0,
                "golden_tools": golden.tool_sequence if golden else [],
                "actual_tools": [s.tool_name for s in result.trace.steps] if result else [],
                "tool_changes": tool_changes,
                "golden_output": golden.trace.final_output[:1000] if golden else "",
                "actual_output": result.trace.final_output[:1000] if result else "",
            })

        # Render template
        env = Environment(loader=BaseLoader())
        template = env.from_string(DIFF_TEMPLATE)
        html = template.render(
            title=title,
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            summary={
                "total": len(diffs),
                "regressions": regressions,
                "tools_changed": tools_changed,
                "output_changed": output_changed,
                "passed": passed,
            },
            diffs=diff_data,
        )

        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(html)

        return str(output)

# fmt: on
