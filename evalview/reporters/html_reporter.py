"""HTML report generator with interactive Plotly charts."""

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
        .test-card .card-header { cursor: pointer; }
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
        }
        .chart-container { min-height: 300px; }
        pre { white-space: pre-wrap; word-wrap: break-word; }
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
        <div class="accordion" id="testResults">
            {% for result in results %}
            <div class="card test-card">
                <div class="card-header d-flex justify-content-between align-items-center"
                     data-bs-toggle="collapse" data-bs-target="#test-{{ loop.index }}">
                    <div>
                        <span class="badge {{ 'badge-pass' if result.passed else 'badge-fail' }} me-2">
                            {{ 'PASS' if result.passed else 'FAIL' }}
                        </span>
                        <strong>{{ result.test_case }}</strong>
                        <span class="text-muted ms-2">({{ result.adapter }})</span>
                    </div>
                    <div>
                        <span class="score-badge {{ 'pass' if result.score >= 80 else 'fail' if result.score < 60 else '' }}">
                            {{ result.score }}
                        </span>
                    </div>
                </div>
                <div id="test-{{ loop.index }}" class="collapse">
                    <div class="card-body">
                        <div class="row">
                            <div class="col-md-6">
                                <h6>Input Query</h6>
                                <div class="output-preview">{{ result.input_query }}</div>

                                <h6 class="mt-3">Output</h6>
                                <div class="output-preview">{{ result.actual_output }}</div>
                            </div>
                            <div class="col-md-6">
                                <h6>Evaluations</h6>
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
                                    <span class="badge bg-success tool-badge">{{ tool }}</span>
                                    {% endfor %}
                                    {% for tool in result.missing_tools %}
                                    <span class="badge bg-warning tool-badge">Missing: {{ tool }}</span>
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
                    </div>
                </div>
            </div>
            {% endfor %}
        </div>

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

# Diff report template - Designed for viral screenshots
DIFF_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{{ title }}</title>
    <style>
        *, *::before, *::after { box-sizing: border-box; }

        :root {
            --bg-deep: #0a0e14;
            --bg-card: #12171f;
            --bg-elevated: #1a2029;
            --border: #2d3748;
            --text-primary: #e2e8f0;
            --text-secondary: #8892a6;
            --green-glow: #10b981;
            --green-bg: rgba(16, 185, 129, 0.15);
            --green-border: rgba(16, 185, 129, 0.4);
            --red-glow: #ef4444;
            --red-bg: rgba(239, 68, 68, 0.15);
            --red-border: rgba(239, 68, 68, 0.4);
            --orange-glow: #f59e0b;
            --orange-bg: rgba(245, 158, 11, 0.15);
            --blue-accent: #3b82f6;
        }

        body {
            background: var(--bg-deep);
            color: var(--text-primary);
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Inter, Roboto, sans-serif;
            margin: 0;
            padding: 2rem;
            min-height: 100vh;
        }

        .container { max-width: 1200px; margin: 0 auto; }

        /* ===== HERO SECTION - THE SCREENSHOT MOMENT ===== */
        .hero {
            text-align: center;
            padding: 3rem 2rem;
            margin-bottom: 2rem;
            background: linear-gradient(180deg, var(--bg-elevated) 0%, var(--bg-deep) 100%);
            border-radius: 16px;
            border: 1px solid var(--border);
            position: relative;
            overflow: hidden;
        }

        .hero::before {
            content: '';
            position: absolute;
            top: 0;
            left: 50%;
            transform: translateX(-50%);
            width: 60%;
            height: 1px;
            background: linear-gradient(90deg, transparent, var(--green-glow), transparent);
        }

        .hero.has-regressions::before {
            background: linear-gradient(90deg, transparent, var(--red-glow), transparent);
        }

        .hero-title {
            font-size: 1rem;
            font-weight: 500;
            color: var(--text-secondary);
            text-transform: uppercase;
            letter-spacing: 0.1em;
            margin-bottom: 0.5rem;
        }

        .hero-subtitle {
            font-size: 0.875rem;
            color: var(--text-secondary);
            margin-bottom: 2rem;
        }

        /* Progress Ring */
        .progress-ring-container {
            display: flex;
            justify-content: center;
            align-items: center;
            margin-bottom: 2rem;
        }

        .progress-ring {
            position: relative;
            width: 180px;
            height: 180px;
        }

        .progress-ring svg {
            transform: rotate(-90deg);
            width: 180px;
            height: 180px;
        }

        .progress-ring circle {
            fill: none;
            stroke-width: 8;
            stroke-linecap: round;
        }

        .progress-ring .bg {
            stroke: var(--bg-card);
        }

        .progress-ring .progress {
            stroke: var(--green-glow);
            stroke-dasharray: 502;
            stroke-dashoffset: calc(502 - (502 * var(--percent)) / 100);
            transition: stroke-dashoffset 1s ease-out;
            filter: drop-shadow(0 0 8px var(--green-glow));
        }

        .progress-ring.has-regressions .progress {
            stroke: var(--red-glow);
            filter: drop-shadow(0 0 8px var(--red-glow));
        }

        .progress-ring .center-text {
            position: absolute;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            text-align: center;
        }

        .progress-percent {
            font-size: 3rem;
            font-weight: 700;
            line-height: 1;
            color: var(--text-primary);
        }

        .progress-label {
            font-size: 0.75rem;
            color: var(--text-secondary);
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }

        /* Stats Grid */
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 1rem;
            max-width: 700px;
            margin: 0 auto;
        }

        .stat-box {
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 1.25rem 1rem;
            text-align: center;
            transition: transform 0.2s, border-color 0.2s;
        }

        .stat-box:hover {
            transform: translateY(-2px);
        }

        .stat-box.passed {
            border-color: var(--green-border);
            background: var(--green-bg);
        }

        .stat-box.regression {
            border-color: var(--red-border);
            background: var(--red-bg);
        }

        .stat-box.changed {
            border-color: rgba(245, 158, 11, 0.3);
            background: var(--orange-bg);
        }

        .stat-number {
            font-size: 2.25rem;
            font-weight: 700;
            line-height: 1;
            margin-bottom: 0.25rem;
        }

        .stat-box.passed .stat-number { color: var(--green-glow); }
        .stat-box.regression .stat-number { color: var(--red-glow); }
        .stat-box.changed .stat-number { color: var(--orange-glow); }
        .stat-box.total .stat-number { color: var(--text-primary); }

        .stat-label {
            font-size: 0.7rem;
            color: var(--text-secondary);
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }

        /* ===== STATUS BANNER ===== */
        .status-banner {
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 0.75rem;
            padding: 1rem 2rem;
            border-radius: 50px;
            font-weight: 600;
            font-size: 1.1rem;
            max-width: fit-content;
            margin: 2rem auto 0;
        }

        .status-banner.all-passed {
            background: var(--green-bg);
            border: 1px solid var(--green-border);
            color: var(--green-glow);
            box-shadow: 0 0 30px rgba(16, 185, 129, 0.2);
        }

        .status-banner.has-issues {
            background: var(--red-bg);
            border: 1px solid var(--red-border);
            color: var(--red-glow);
            box-shadow: 0 0 30px rgba(239, 68, 68, 0.2);
        }

        .status-icon {
            font-size: 1.25rem;
        }

        /* ===== DIFF CARDS ===== */
        .section-title {
            font-size: 0.875rem;
            font-weight: 500;
            color: var(--text-secondary);
            text-transform: uppercase;
            letter-spacing: 0.1em;
            margin: 2.5rem 0 1rem;
            padding-bottom: 0.5rem;
            border-bottom: 1px solid var(--border);
        }

        .diff-card {
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: 12px;
            margin-bottom: 1rem;
            overflow: hidden;
            transition: border-color 0.2s;
        }

        .diff-card:hover {
            border-color: #4a5568;
        }

        .diff-card.status-passed { border-left: 4px solid var(--green-glow); }
        .diff-card.status-regression { border-left: 4px solid var(--red-glow); }
        .diff-card.status-tools_changed { border-left: 4px solid var(--orange-glow); }
        .diff-card.status-output_changed { border-left: 4px solid var(--text-secondary); }

        .diff-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 1rem 1.25rem;
            background: var(--bg-elevated);
            border-bottom: 1px solid var(--border);
        }

        .diff-header-left {
            display: flex;
            align-items: center;
            gap: 0.75rem;
        }

        .status-badge {
            display: inline-flex;
            align-items: center;
            gap: 0.35rem;
            padding: 0.25rem 0.75rem;
            border-radius: 50px;
            font-size: 0.7rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.03em;
        }

        .status-badge.passed { background: var(--green-bg); color: var(--green-glow); }
        .status-badge.regression { background: var(--red-bg); color: var(--red-glow); }
        .status-badge.tools_changed { background: var(--orange-bg); color: var(--orange-glow); }
        .status-badge.output_changed { background: rgba(139, 148, 158, 0.15); color: var(--text-secondary); }

        .test-name {
            font-weight: 600;
            font-size: 0.95rem;
        }

        .diff-metrics {
            display: flex;
            gap: 1rem;
        }

        .metric {
            display: flex;
            flex-direction: column;
            align-items: flex-end;
            padding: 0.35rem 0.75rem;
            background: var(--bg-card);
            border-radius: 6px;
        }

        .metric-label {
            font-size: 0.65rem;
            color: var(--text-secondary);
            text-transform: uppercase;
            letter-spacing: 0.03em;
        }

        .metric-value {
            font-size: 0.9rem;
            font-weight: 600;
        }

        .metric-value.positive { color: var(--green-glow); }
        .metric-value.negative { color: var(--red-glow); }

        .diff-body {
            padding: 1.25rem;
        }

        .diff-section-title {
            font-size: 0.7rem;
            font-weight: 600;
            color: var(--text-secondary);
            text-transform: uppercase;
            letter-spacing: 0.05em;
            margin-bottom: 0.75rem;
        }

        /* ===== TOOL SEQUENCE - BEFORE/AFTER ===== */
        .comparison-grid {
            display: grid;
            grid-template-columns: 1fr auto 1fr;
            gap: 1rem;
            align-items: start;
            margin-bottom: 1.5rem;
        }

        .comparison-panel {
            background: var(--bg-deep);
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 1rem;
        }

        .comparison-panel.golden {
            border-color: rgba(139, 148, 158, 0.3);
        }

        .comparison-panel.actual {
            border-color: var(--blue-accent);
        }

        .panel-label {
            display: flex;
            align-items: center;
            gap: 0.5rem;
            font-size: 0.7rem;
            font-weight: 500;
            color: var(--text-secondary);
            text-transform: uppercase;
            letter-spacing: 0.05em;
            margin-bottom: 0.75rem;
        }

        .panel-label .dot {
            width: 8px;
            height: 8px;
            border-radius: 50%;
        }

        .comparison-panel.golden .dot { background: var(--text-secondary); }
        .comparison-panel.actual .dot { background: var(--blue-accent); }

        .arrow-divider {
            display: flex;
            align-items: center;
            justify-content: center;
            color: var(--text-secondary);
            font-size: 1.5rem;
            padding-top: 2rem;
        }

        .tool-list {
            display: flex;
            flex-wrap: wrap;
            gap: 0.5rem;
        }

        .tool-badge {
            display: inline-flex;
            align-items: center;
            gap: 0.25rem;
            padding: 0.35rem 0.65rem;
            background: var(--bg-elevated);
            border: 1px solid var(--border);
            border-radius: 6px;
            font-family: 'SF Mono', 'Fira Code', Consolas, monospace;
            font-size: 0.75rem;
            color: var(--text-primary);
        }

        .tool-badge.added {
            background: var(--green-bg);
            border-color: var(--green-border);
            color: var(--green-glow);
        }

        .tool-badge.removed {
            background: var(--red-bg);
            border-color: var(--red-border);
            color: var(--red-glow);
            text-decoration: line-through;
        }

        .tool-badge.changed {
            background: var(--orange-bg);
            border-color: rgba(245, 158, 11, 0.4);
            color: var(--orange-glow);
        }

        /* ===== TOOL CHANGES INLINE DIFF ===== */
        .changes-list {
            background: var(--bg-deep);
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 0.75rem 1rem;
            margin-bottom: 1.5rem;
            font-family: 'SF Mono', 'Fira Code', Consolas, monospace;
            font-size: 0.8rem;
        }

        .change-line {
            padding: 0.25rem 0.5rem;
            border-radius: 4px;
            margin-bottom: 0.25rem;
        }

        .change-line:last-child { margin-bottom: 0; }

        .change-line.added {
            background: rgba(16, 185, 129, 0.1);
            color: var(--green-glow);
        }

        .change-line.removed {
            background: rgba(239, 68, 68, 0.1);
            color: var(--red-glow);
        }

        .change-line.changed {
            background: rgba(245, 158, 11, 0.1);
            color: var(--orange-glow);
        }

        .change-arrow {
            color: var(--text-secondary);
            margin: 0 0.5rem;
        }

        /* ===== OUTPUT COMPARISON ===== */
        .output-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 1rem;
        }

        .output-panel {
            background: var(--bg-deep);
            border: 1px solid var(--border);
            border-radius: 8px;
            overflow: hidden;
        }

        .output-header {
            display: flex;
            align-items: center;
            gap: 0.5rem;
            padding: 0.6rem 0.85rem;
            background: var(--bg-elevated);
            border-bottom: 1px solid var(--border);
            font-size: 0.7rem;
            font-weight: 500;
            color: var(--text-secondary);
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }

        .output-content {
            padding: 1rem;
            font-family: 'SF Mono', 'Fira Code', Consolas, monospace;
            font-size: 0.75rem;
            line-height: 1.6;
            max-height: 200px;
            overflow-y: auto;
            white-space: pre-wrap;
            word-break: break-word;
            color: var(--text-primary);
        }

        /* ===== FOOTER ===== */
        .footer {
            text-align: center;
            padding: 2rem 0;
            margin-top: 2rem;
            border-top: 1px solid var(--border);
            font-size: 0.8rem;
            color: var(--text-secondary);
        }

        .footer a {
            color: var(--blue-accent);
            text-decoration: none;
        }

        .footer a:hover {
            text-decoration: underline;
        }

        /* ===== ANIMATIONS ===== */
        @keyframes fadeInUp {
            from {
                opacity: 0;
                transform: translateY(10px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }

        .hero { animation: fadeInUp 0.5s ease-out; }
        .diff-card { animation: fadeInUp 0.4s ease-out backwards; }
        .diff-card:nth-child(1) { animation-delay: 0.1s; }
        .diff-card:nth-child(2) { animation-delay: 0.15s; }
        .diff-card:nth-child(3) { animation-delay: 0.2s; }
        .diff-card:nth-child(4) { animation-delay: 0.25s; }
        .diff-card:nth-child(5) { animation-delay: 0.3s; }

        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.7; }
        }

        .status-banner.all-passed {
            animation: pulse 2s ease-in-out infinite;
        }

        /* ===== RESPONSIVE ===== */
        @media (max-width: 768px) {
            .stats-grid { grid-template-columns: repeat(2, 1fr); }
            .comparison-grid { grid-template-columns: 1fr; }
            .arrow-divider { transform: rotate(90deg); padding: 0.5rem 0; }
            .output-grid { grid-template-columns: 1fr; }
        }
    </style>
</head>
<body>
    <div class="container">
        <!-- HERO SECTION - The Screenshot Moment -->
        <div class="hero {% if summary.regressions > 0 %}has-regressions{% endif %}">
            <div class="hero-title">Trace Diff Report</div>
            <div class="hero-subtitle">{{ timestamp }}</div>

            <!-- Progress Ring -->
            <div class="progress-ring-container">
                <div class="progress-ring {% if summary.regressions > 0 %}has-regressions{% endif %}" style="--percent: {{ ((summary.passed / summary.total) * 100) | round(0) if summary.total > 0 else 0 }}">
                    <svg viewBox="0 0 180 180">
                        <circle class="bg" cx="90" cy="90" r="80"/>
                        <circle class="progress" cx="90" cy="90" r="80"/>
                    </svg>
                    <div class="center-text">
                        <div class="progress-percent">{{ ((summary.passed / summary.total) * 100) | round(0) if summary.total > 0 else 0 }}%</div>
                        <div class="progress-label">Pass Rate</div>
                    </div>
                </div>
            </div>

            <!-- Stats Grid -->
            <div class="stats-grid">
                <div class="stat-box total">
                    <div class="stat-number">{{ summary.total }}</div>
                    <div class="stat-label">Total Tests</div>
                </div>
                <div class="stat-box passed">
                    <div class="stat-number">{{ summary.passed }}</div>
                    <div class="stat-label">Passed</div>
                </div>
                <div class="stat-box changed">
                    <div class="stat-number">{{ summary.tools_changed + summary.output_changed }}</div>
                    <div class="stat-label">Changed</div>
                </div>
                <div class="stat-box regression">
                    <div class="stat-number">{{ summary.regressions }}</div>
                    <div class="stat-label">Regressions</div>
                </div>
            </div>

            <!-- Status Banner -->
            {% if summary.regressions == 0 and summary.tools_changed == 0 and summary.output_changed == 0 %}
            <div class="status-banner all-passed">
                <span class="status-icon">&#10003;</span>
                All tests match golden baseline
            </div>
            {% elif summary.regressions > 0 %}
            <div class="status-banner has-issues">
                <span class="status-icon">!</span>
                {{ summary.regressions }} regression{{ 's' if summary.regressions > 1 else '' }} detected
            </div>
            {% else %}
            <div class="status-banner has-issues" style="background: var(--orange-bg); border-color: rgba(245, 158, 11, 0.4); color: var(--orange-glow); box-shadow: 0 0 30px rgba(245, 158, 11, 0.2);">
                <span class="status-icon">~</span>
                {{ summary.tools_changed + summary.output_changed }} test{{ 's' if (summary.tools_changed + summary.output_changed) > 1 else '' }} changed
            </div>
            {% endif %}
        </div>

        <!-- Diff Cards -->
        {% if diffs %}
        <div class="section-title">Test Results</div>

        {% for diff in diffs %}
        <div class="diff-card status-{{ diff.status }}">
            <div class="diff-header">
                <div class="diff-header-left">
                    <span class="status-badge {{ diff.status }}">
                        {% if diff.status == 'passed' %}&#10003;{% elif diff.status == 'regression' %}&#10007;{% else %}~{% endif %}
                        {{ diff.status | replace('_', ' ') }}
                    </span>
                    <span class="test-name">{{ diff.test_name }}</span>
                </div>
                <div class="diff-metrics">
                    <div class="metric">
                        <span class="metric-label">Score</span>
                        <span class="metric-value {% if diff.score_diff > 0 %}positive{% elif diff.score_diff < 0 %}negative{% endif %}">
                            {{ diff.actual_score | round(1) }}{% if diff.score_diff != 0 %} ({{ '%+.1f' | format(diff.score_diff) }}){% endif %}
                        </span>
                    </div>
                    <div class="metric">
                        <span class="metric-label">Similarity</span>
                        <span class="metric-value">{{ (diff.similarity * 100) | round(0) }}%</span>
                    </div>
                </div>
            </div>

            <div class="diff-body">
                <!-- Tool Sequence Comparison -->
                <div class="diff-section-title">Tool Sequence</div>
                <div class="comparison-grid">
                    <div class="comparison-panel golden">
                        <div class="panel-label">
                            <span class="dot"></span>
                            Golden (Baseline)
                        </div>
                        <div class="tool-list">
                            {% for tool in diff.golden_tools %}
                            <span class="tool-badge">{{ tool }}</span>
                            {% endfor %}
                            {% if not diff.golden_tools %}
                            <span style="color: var(--text-secondary); font-size: 0.8rem; font-style: italic;">No tools</span>
                            {% endif %}
                        </div>
                    </div>

                    <div class="arrow-divider">&rarr;</div>

                    <div class="comparison-panel actual">
                        <div class="panel-label">
                            <span class="dot"></span>
                            Actual (Current)
                        </div>
                        <div class="tool-list">
                            {% for tool in diff.actual_tools %}
                            <span class="tool-badge">{{ tool }}</span>
                            {% endfor %}
                            {% if not diff.actual_tools %}
                            <span style="color: var(--text-secondary); font-size: 0.8rem; font-style: italic;">No tools</span>
                            {% endif %}
                        </div>
                    </div>
                </div>

                {% if diff.tool_changes %}
                <div class="diff-section-title">Changes</div>
                <div class="changes-list">
                    {% for change in diff.tool_changes %}
                    <div class="change-line {{ change.type }}">
                        {% if change.type == 'added' %}+ {{ change.tool }}{% endif %}
                        {% if change.type == 'removed' %}&minus; {{ change.tool }}{% endif %}
                        {% if change.type == 'changed' %}~ {{ change.from }}<span class="change-arrow">&rarr;</span>{{ change.to }}{% endif %}
                    </div>
                    {% endfor %}
                </div>
                {% endif %}

                <!-- Output Comparison -->
                <div class="diff-section-title">Output</div>
                <div class="output-grid">
                    <div class="output-panel">
                        <div class="output-header">
                            <span class="dot" style="width: 6px; height: 6px; border-radius: 50%; background: var(--text-secondary);"></span>
                            Golden
                        </div>
                        <div class="output-content">{{ diff.golden_output or '(empty)' }}</div>
                    </div>
                    <div class="output-panel">
                        <div class="output-header">
                            <span class="dot" style="width: 6px; height: 6px; border-radius: 50%; background: var(--blue-accent);"></span>
                            Actual
                        </div>
                        <div class="output-content">{{ diff.actual_output or '(empty)' }}</div>
                    </div>
                </div>
            </div>
        </div>
        {% endfor %}
        {% endif %}

        <footer class="footer">
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
