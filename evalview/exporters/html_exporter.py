"""HTML exporter for trace data."""

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

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

__all__ = ["TraceHTMLExporter"]


def _format_tokens(tokens: int) -> str:
    """Format token count with commas."""
    return f"{tokens:,}"


def _format_cost(cost: float) -> str:
    """Format cost for display."""
    if cost == 0:
        return "$0.00"
    elif cost < 0.01:
        return f"${cost:.4f}"
    return f"${cost:.2f}"


def _format_duration(ms: float) -> str:
    """Format duration for display."""
    if ms < 1000:
        return f"{ms:.0f}ms"
    return f"{ms / 1000:.1f}s"


class TraceHTMLExporter:
    """Export trace data to interactive HTML reports."""

    def __init__(self):
        if not JINJA2_AVAILABLE:
            raise ImportError(
                "jinja2 is required for HTML export. Install with: pip install evalview[reports]"
            )

    def export(
        self,
        trace: Dict[str, Any],
        spans: List[Dict[str, Any]],
        output_path: str,
    ) -> str:
        """Export a single trace to HTML.

        Args:
            trace: Trace record from database
            spans: List of span records
            output_path: Path to write HTML file

        Returns:
            Path to generated HTML file
        """
        # Filter LLM spans once, reuse in charts and template
        llm_spans = [s for s in spans if s.get("span_type") == "llm"]

        charts = self._generate_charts(llm_spans) if PLOTLY_AVAILABLE else {}

        html = self._render_template(
            trace=trace,
            llm_spans=llm_spans,
            charts=charts,
            plotly_available=PLOTLY_AVAILABLE,
        )

        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(html, encoding="utf-8")

        return str(output)

    def _generate_charts(self, llm_spans: List[Dict[str, Any]]) -> Dict[str, str]:
        """Generate Plotly charts as JSON strings.

        Args:
            llm_spans: Pre-filtered list of LLM span records
        """
        if not llm_spans:
            return {}

        charts = {}

        # Token breakdown pie chart
        total_input = sum(s.get("input_tokens", 0) for s in llm_spans)
        total_output = sum(s.get("output_tokens", 0) for s in llm_spans)
        if total_input > 0 or total_output > 0:
            fig = go.Figure(data=[go.Pie(
                labels=["Input Tokens", "Output Tokens"],
                values=[total_input, total_output],
                marker_colors=["#3b82f6", "#22c55e"],
                hole=0.4,
            )])
            fig.update_layout(
                title="Token Distribution",
                showlegend=True,
                margin=dict(t=40, b=20, l=20, r=20),
                height=300,
            )
            charts["token_pie"] = fig.to_json()

        # Cost by model bar chart
        cost_by_model: Dict[str, float] = {}
        for s in llm_spans:
            model = s.get("model", "unknown")
            cost_by_model[model] = cost_by_model.get(model, 0) + s.get("cost_usd", 0)

        if cost_by_model:
            models = list(cost_by_model.keys())
            costs = list(cost_by_model.values())
            fig = go.Figure(data=[go.Bar(
                x=models,
                y=costs,
                marker_color="#8b5cf6",
            )])
            fig.update_layout(
                title="Cost by Model",
                xaxis_title="Model",
                yaxis_title="Cost ($)",
                margin=dict(t=40, b=80, l=40, r=20),
                height=300,
                xaxis_tickangle=-45,
            )
            charts["cost_by_model"] = fig.to_json()

        # Duration timeline
        if len(llm_spans) > 1:
            durations = [s.get("duration_ms", 0) for s in llm_spans]
            models = [s.get("model", "unknown")[:20] for s in llm_spans]
            fig = go.Figure(data=[go.Bar(
                x=list(range(1, len(llm_spans) + 1)),
                y=durations,
                text=models,
                marker_color="#f59e0b",
            )])
            fig.update_layout(
                title="Call Duration Timeline",
                xaxis_title="Call #",
                yaxis_title="Duration (ms)",
                margin=dict(t=40, b=40, l=40, r=20),
                height=300,
            )
            charts["duration_timeline"] = fig.to_json()

        return charts

    def _render_template(
        self,
        trace: Dict[str, Any],
        llm_spans: List[Dict[str, Any]],
        charts: Dict[str, str],
        plotly_available: bool,
    ) -> str:
        """Render the HTML template.

        Args:
            trace: Trace record from database
            llm_spans: Pre-filtered list of LLM span records
            charts: Chart JSON strings
            plotly_available: Whether Plotly is available
        """
        env = Environment(loader=BaseLoader())
        template = env.from_string(TRACE_HTML_TEMPLATE)

        # Normalize span data for template
        spans_for_template = [
            {
                "model": s.get("model", "unknown"),
                "provider": s.get("provider", "unknown"),
                "input_tokens": s.get("input_tokens", 0),
                "output_tokens": s.get("output_tokens", 0),
                "duration_ms": s.get("duration_ms", 0),
                "cost_usd": s.get("cost_usd", 0),
                "status": s.get("status", "success"),
                "finish_reason": s.get("finish_reason", ""),
                "error_message": s.get("error_message", ""),
            }
            for s in llm_spans
        ]

        return template.render(
            trace=trace,
            spans=spans_for_template,
            charts=charts,
            plotly_available=plotly_available,
            timestamp=datetime.now().isoformat(),
            format_tokens=_format_tokens,
            format_cost=_format_cost,
            format_duration=_format_duration,
        )


TRACE_HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Trace {{ trace.run_id }} - EvalView</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css" rel="stylesheet">
    {% if plotly_available %}
    <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
    {% endif %}
    <style>
        :root {
            --primary-color: #3b82f6;
            --success-color: #22c55e;
            --warning-color: #f59e0b;
            --danger-color: #ef4444;
        }
        body { background-color: #f8fafc; }
        .card { border: none; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }
        .stat-card { text-align: center; padding: 1.5rem; }
        .stat-value { font-size: 2rem; font-weight: 700; }
        .stat-label { color: #64748b; font-size: 0.875rem; text-transform: uppercase; }
        .cost-low { color: var(--success-color); }
        .cost-med { color: var(--warning-color); }
        .cost-high { color: var(--danger-color); }
        .span-card { margin-bottom: 0.75rem; }
        .span-card .card-body { padding: 1rem; }
        .span-header { display: flex; justify-content: space-between; align-items: center; }
        .model-name { font-weight: 600; font-size: 1rem; }
        .provider-badge { font-size: 0.75rem; padding: 0.25rem 0.5rem; }
        .token-info { font-size: 0.875rem; color: #64748b; }
        .chart-container { min-height: 300px; }
        .status-success { border-left: 4px solid var(--success-color); }
        .status-error { border-left: 4px solid var(--danger-color); }
        table.metrics-table td { padding: 0.5rem 1rem; }
        table.metrics-table td:first-child { color: #64748b; }
        table.metrics-table td:last-child { font-weight: 600; text-align: right; }
    </style>
</head>
<body>
    <nav class="navbar navbar-dark bg-dark mb-4">
        <div class="container">
            <span class="navbar-brand mb-0 h1">EvalView Trace</span>
            <span class="text-light">{{ trace.run_id }}</span>
        </div>
    </nav>

    <div class="container">
        <!-- Header Info -->
        <div class="card mb-4">
            <div class="card-body">
                <div class="row">
                    <div class="col-md-6">
                        <h5>Trace Details</h5>
                        <table class="metrics-table">
                            <tr>
                                <td>Run ID</td>
                                <td><code>{{ trace.run_id }}</code></td>
                            </tr>
                            <tr>
                                <td>Source</td>
                                <td>{{ trace.source }}</td>
                            </tr>
                            {% if trace.script_name %}
                            <tr>
                                <td>Script</td>
                                <td>{{ trace.script_name }}</td>
                            </tr>
                            {% endif %}
                            <tr>
                                <td>Created</td>
                                <td>{{ trace.created_at[:19] }}</td>
                            </tr>
                            <tr>
                                <td>Status</td>
                                <td>
                                    <span class="badge {% if trace.status == 'completed' %}bg-success{% else %}bg-danger{% endif %}">
                                        {{ trace.status }}
                                    </span>
                                </td>
                            </tr>
                        </table>
                    </div>
                    <div class="col-md-6">
                        <h5>Summary</h5>
                        <table class="metrics-table">
                            <tr>
                                <td>Total LLM Calls</td>
                                <td>{{ trace.total_calls }}</td>
                            </tr>
                            <tr>
                                <td>Total Tokens</td>
                                <td>{{ format_tokens(trace.total_tokens) }}</td>
                            </tr>
                            <tr>
                                <td>Input / Output</td>
                                <td>{{ format_tokens(trace.total_input_tokens) }} / {{ format_tokens(trace.total_output_tokens) }}</td>
                            </tr>
                            <tr>
                                <td>Total Cost</td>
                                <td class="{% if trace.total_cost < 0.10 %}cost-low{% elif trace.total_cost < 1.0 %}cost-med{% else %}cost-high{% endif %}">
                                    {{ format_cost(trace.total_cost) }}
                                </td>
                            </tr>
                            <tr>
                                <td>Total Latency</td>
                                <td>{{ format_duration(trace.total_latency_ms) }}</td>
                            </tr>
                        </table>
                    </div>
                </div>
            </div>
        </div>

        <!-- Summary Cards -->
        <div class="row mb-4">
            <div class="col-md-3">
                <div class="card stat-card">
                    <div class="stat-value">{{ trace.total_calls }}</div>
                    <div class="stat-label">LLM Calls</div>
                </div>
            </div>
            <div class="col-md-3">
                <div class="card stat-card">
                    <div class="stat-value">{{ format_tokens(trace.total_tokens) }}</div>
                    <div class="stat-label">Total Tokens</div>
                </div>
            </div>
            <div class="col-md-3">
                <div class="card stat-card">
                    <div class="stat-value {% if trace.total_cost < 0.10 %}cost-low{% elif trace.total_cost < 1.0 %}cost-med{% else %}cost-high{% endif %}">
                        {{ format_cost(trace.total_cost) }}
                    </div>
                    <div class="stat-label">Total Cost</div>
                </div>
            </div>
            <div class="col-md-3">
                <div class="card stat-card">
                    <div class="stat-value">{{ format_duration(trace.total_latency_ms) }}</div>
                    <div class="stat-label">Total Latency</div>
                </div>
            </div>
        </div>

        {% if plotly_available and charts %}
        <!-- Charts -->
        <div class="row mb-4">
            {% if charts.token_pie %}
            <div class="col-md-4">
                <div class="card">
                    <div class="card-body">
                        <div id="chart-token-pie" class="chart-container"></div>
                    </div>
                </div>
            </div>
            {% endif %}
            {% if charts.cost_by_model %}
            <div class="col-md-4">
                <div class="card">
                    <div class="card-body">
                        <div id="chart-cost-model" class="chart-container"></div>
                    </div>
                </div>
            </div>
            {% endif %}
            {% if charts.duration_timeline %}
            <div class="col-md-4">
                <div class="card">
                    <div class="card-body">
                        <div id="chart-duration" class="chart-container"></div>
                    </div>
                </div>
            </div>
            {% endif %}
        </div>
        {% endif %}

        <!-- LLM Calls -->
        <h4 class="mb-3">LLM Calls ({{ spans | length }})</h4>
        {% for span in spans %}
        <div class="card span-card status-{{ span.status }}">
            <div class="card-body">
                <div class="span-header mb-2">
                    <div>
                        <span class="model-name">{{ span.model }}</span>
                        <span class="badge bg-secondary provider-badge ms-2">{{ span.provider }}</span>
                        {% if span.status == 'error' %}
                        <span class="badge bg-danger ms-2">Error</span>
                        {% endif %}
                    </div>
                    <div class="text-end">
                        <span class="badge bg-primary">{{ format_duration(span.duration_ms) }}</span>
                        <span class="badge {% if span.cost_usd < 0.01 %}bg-success{% elif span.cost_usd < 0.10 %}bg-warning{% else %}bg-danger{% endif %} ms-1">
                            {{ format_cost(span.cost_usd) }}
                        </span>
                    </div>
                </div>
                <div class="token-info">
                    <span class="me-3">
                        <strong>In:</strong> {{ format_tokens(span.input_tokens) }}
                    </span>
                    <span class="me-3">
                        <strong>Out:</strong> {{ format_tokens(span.output_tokens) }}
                    </span>
                    {% if span.finish_reason %}
                    <span class="text-muted">
                        Finish: {{ span.finish_reason }}
                    </span>
                    {% endif %}
                </div>
                {% if span.error_message %}
                <div class="alert alert-danger mt-2 mb-0 py-2">
                    {{ span.error_message }}
                </div>
                {% endif %}
            </div>
        </div>
        {% endfor %}

        <footer class="text-center text-muted py-4">
            Generated by <a href="https://github.com/hidai25/eval-view">EvalView</a> at {{ timestamp[:19] }}
        </footer>
    </div>

    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/js/bootstrap.bundle.min.js"></script>
    {% if plotly_available and charts %}
    <script>
        (function() {
            var chart;
            {% if charts.token_pie %}
            chart = JSON.parse('{{ charts.token_pie | safe }}');
            Plotly.newPlot('chart-token-pie', chart.data, chart.layout, {responsive: true});
            {% endif %}
            {% if charts.cost_by_model %}
            chart = JSON.parse('{{ charts.cost_by_model | safe }}');
            Plotly.newPlot('chart-cost-model', chart.data, chart.layout, {responsive: true});
            {% endif %}
            {% if charts.duration_timeline %}
            chart = JSON.parse('{{ charts.duration_timeline | safe }}');
            Plotly.newPlot('chart-duration', chart.data, chart.layout, {responsive: true});
            {% endif %}
        })();
    </script>
    {% endif %}
</body>
</html>
"""
