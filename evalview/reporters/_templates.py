"""Embedded HTML templates for reporters.

`HTML_TEMPLATE` backs the legacy `_DeprecatedHTMLReporter` (kept only because
old result paths still reference it). `DIFF_TEMPLATE` backs `DiffReporter`,
which is what the `--diff-report` flag actually uses today.

Lifted out of `html_reporter.py` so the reporter module is readable without
scrolling past ~950 lines of inline HTML/CSS/JS.
"""


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
        .meta-card { padding: 1rem 1.25rem; }
        .meta-key { color: #64748b; font-size: 0.75rem; text-transform: uppercase; margin-bottom: 0.25rem; }
        .meta-value { font-weight: 600; color: #0f172a; }
        .pass { color: var(--pass-color); }
        .fail { color: var(--fail-color); }
        .badge-pass { background-color: var(--pass-color); }
        .badge-fail { background-color: var(--fail-color); }
        .test-card { margin-bottom: 1rem; }
        .test-card .card-header { cursor: pointer; user-select: none; transition: background 0.15s; }
        .test-card .card-header:hover { background-color: #e2e8f0; }
        .test-card .card-header::after {
            content: "▶ Click to expand";
            font-size: 0.7rem;
            color: #94a3b8;
            margin-left: auto;
            padding-left: 1rem;
        }
        .test-card .card-header[aria-expanded="true"]::after {
            content: "▼ Click to collapse";
        }
        .score-badge { font-size: 1.25rem; font-weight: 600; }
        .tool-list { display: flex; flex-wrap: wrap; gap: 0.5rem; }
        .tool-badge { font-size: 0.75rem; padding: 0.25rem 0.5rem; }
        .output-preview {
            max-height: 400px;
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
        .span-row:hover { background: #e2e8f0; }
        .span-row::after {
            content: "▶";
            font-size: 0.65rem;
            color: #94a3b8;
            transition: transform 0.15s;
        }
        .span-row[aria-expanded="true"]::after { transform: rotate(90deg); }
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
        /* ── top-level tab nav ── */
        .ev-tab-nav { border-bottom: 2px solid #e2e8f0; margin-bottom: 1.5rem; }
        .ev-tab-nav .nav-link {
            color: #64748b; font-weight: 500; border: none; border-bottom: 3px solid transparent;
            padding: 0.65rem 1.1rem; margin-bottom: -2px; border-radius: 0;
        }
        .ev-tab-nav .nav-link:hover { color: #334155; }
        .ev-tab-nav .nav-link.active { color: var(--primary-color); border-bottom-color: var(--primary-color); background: none; }
        /* ── compare table ── */
        .compare-table th, .compare-table td { vertical-align: middle; }
        .compare-table thead tr:first-child th { background: #1e293b; color: #f1f5f9; }
        .compare-table thead tr:last-child th { background: #0f172a; color: #94a3b8; font-size: 0.75rem; font-weight: 500; }
        .compare-table tbody tr:hover { background: #f8fafc; }
        .compare-table tfoot td { background: #f1f5f9; color: #475569; font-size: 0.8rem; }
        .score-chip {
            display: inline-block; padding: 0.15rem 0.5rem;
            border-radius: 9999px; font-weight: 700; font-size: 0.8rem; color: #fff;
        }
        .score-chip.pass  { background: #22c55e; }
        .score-chip.warn  { background: #f59e0b; }
        .score-chip.fail  { background: #ef4444; }
        .model-col-sep { border-left: 2px solid #cbd5e1 !important; }
    </style>
</head>
<body>
    <nav class="navbar navbar-dark bg-dark mb-0">
        <div class="container">
            <span class="navbar-brand mb-0 h1">EvalView Report</span>
            <span class="text-light">{{ timestamp[:19] }}</span>
        </div>
    </nav>

    <div class="container mt-4">

        <!-- ── Top-level tab navigation ── -->
        <ul class="nav ev-tab-nav" id="mainTabs" role="tablist">
            <li class="nav-item">
                <button class="nav-link active" data-bs-toggle="tab" data-bs-target="#tab-overview" type="button">
                    Overview
                </button>
            </li>
            {% if compare.enabled %}
            <li class="nav-item">
                <button class="nav-link" data-bs-toggle="tab" data-bs-target="#tab-compare" type="button">
                    Compare
                    <span class="badge ms-1" style="background:var(--primary-color); font-size:0.7rem;">{{ compare.models | length }} models</span>
                </button>
            </li>
            {% endif %}
            <li class="nav-item">
                <button class="nav-link" data-bs-toggle="tab" data-bs-target="#tab-tests" type="button">
                    Tests
                    <span class="badge bg-secondary ms-1" style="font-size:0.7rem;">{{ results | length }}</span>
                </button>
            </li>
        </ul>

        <div class="tab-content">

        <!-- ══ TAB 1: Overview ══════════════════════════════════════════════ -->
        <div class="tab-pane fade show active" id="tab-overview">
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
        <div class="row mb-4">
            <div class="col-md-6">
                <div class="card meta-card">
                    <div class="meta-key">Models Used</div>
                    <div class="meta-value">{{ summary.models_display }}</div>
                </div>
            </div>
            <div class="col-md-3">
                <div class="card meta-card">
                    <div class="meta-key">Total Tokens</div>
                    <div class="meta-value">{{ "{:,}".format(summary.total_tokens) if summary.total_tokens else "0" }}</div>
                </div>
            </div>
            <div class="col-md-3">
                <div class="card meta-card">
                    <div class="meta-key">Avg Latency</div>
                    <div class="meta-value">{{ summary.avg_latency }}ms</div>
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

        </div><!-- /tab-overview -->

        <!-- ══ TAB 2: Compare ══════════════════════════════════════════════ -->
        {% if compare.enabled %}
        <div class="tab-pane fade" id="tab-compare">
        <div class="card mb-4">
            <div class="card-header d-flex align-items-center gap-2">
                <strong>Model Comparison</strong>
                <span class="badge bg-secondary">{{ compare.models | length }} models &middot; {{ compare.tasks | length }} tasks</span>
            </div>
            <div class="card-body p-0">
                <table class="table table-sm table-hover mb-0 compare-table" style="font-size:0.875rem;">
                    <thead>
                        <tr>
                            <th class="px-3 py-2">Task</th>
                            {% for model in compare.models %}
                            <th class="px-3 py-2 text-center model-col-sep" colspan="3">{{ model }}</th>
                            {% endfor %}
                        </tr>
                        <tr>
                            <th class="px-3 py-1"></th>
                            {% for model in compare.models %}
                            <th class="px-2 py-1 text-center model-col-sep">Score</th>
                            <th class="px-2 py-1 text-center">Latency</th>
                            <th class="px-2 py-1 text-center">Tools</th>
                            {% endfor %}
                        </tr>
                    </thead>
                    <tbody>
                        {% for task in compare.tasks %}
                        {% set row = compare.rows[task] %}
                        <tr>
                            <td class="px-3 py-2 fw-semibold">{{ task }}</td>
                            {% for model in compare.models %}
                            {% if model in row %}
                            {% set cell = row[model] %}
                            {% set chip_class = 'pass' if cell.score >= 80 else 'fail' if cell.score < 60 else 'warn' %}
                            <td class="px-2 py-2 text-center model-col-sep">
                                <span class="score-chip {{ chip_class }}">{{ cell.score }}</span>
                                {% if not cell.passed %}&thinsp;<span title="Failed threshold" style="color:#ef4444;font-size:0.75rem;">✗</span>{% endif %}
                            </td>
                            <td class="px-2 py-2 text-center text-muted" style="font-size:0.8rem;">
                                {{ (cell.latency / 1000) | round(1) }}s
                            </td>
                            <td class="px-2 py-2 text-center text-muted" style="font-size:0.8rem;">
                                {{ cell.tool_accuracy | round }}%
                            </td>
                            {% else %}
                            <td class="px-2 py-2 text-center text-muted model-col-sep" colspan="3">—</td>
                            {% endif %}
                            {% endfor %}
                        </tr>
                        {% endfor %}
                    </tbody>
                    <tfoot>
                        <tr>
                            <td class="px-3 py-2 fw-semibold">Avg Score</td>
                            {% for model in compare.models %}
                            <td class="px-2 py-2 text-center fw-semibold model-col-sep" colspan="3">
                                {{ compare.totals[model].avg_score }}
                            </td>
                            {% endfor %}
                        </tr>
                        <tr>
                            <td class="px-3 py-1">Avg Latency</td>
                            {% for model in compare.models %}
                            <td class="px-2 py-1 text-center model-col-sep" colspan="3">
                                {{ compare.totals[model].avg_latency }}s
                            </td>
                            {% endfor %}
                        </tr>
                        <tr>
                            <td class="px-3 py-1">Total Cost</td>
                            {% for model in compare.models %}
                            <td class="px-2 py-1 text-center model-col-sep" colspan="3">
                                {{ compare.totals[model].cost_display }}
                            </td>
                            {% endfor %}
                        </tr>
                    </tfoot>
                </table>
            </div>
        </div>
        </div><!-- /tab-compare -->
        {% endif %}

        <!-- ══ TAB 3: Tests ════════════════════════════════════════════════ -->
        <div class="tab-pane fade" id="tab-tests">
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
                                                <td>Model</td>
                                                <td><strong>{{ result.model }}</strong></td>
                                            </tr>
                                            <tr>
                                                <td>Tokens</td>
                                                <td>
                                                    <strong>{{ "{:,}".format(result.tokens_total) }}</strong>
                                                    {% if result.tokens_total %}
                                                    <div class="small text-muted">
                                                        in {{ result.tokens_input }} / out {{ result.tokens_output }}{% if result.tokens_cached %} / cached {{ result.tokens_cached }}{% endif %}
                                                    </div>
                                                    {% endif %}
                                                </td>
                                            </tr>
                                            <tr>
                                                <td>Latency</td>
                                                <td>{{ result.latency }}ms</td>
                                            </tr>
                                            <tr>
                                                <td>Steps</td>
                                                <td>{{ result.steps }}</td>
                                            </tr>
                                            <tr>
                                                <td>Hallucination</td>
                                                <td>
                                                    {% if result.hallucination_detected %}
                                                    <span class="badge bg-warning text-dark">Detected ({{ result.hallucination_confidence }}%)</span>
                                                    {% else %}
                                                    <span class="badge bg-success">None detected</span>
                                                    {% endif %}
                                                </td>
                                            </tr>
                                            <tr>
                                                <td>Safety</td>
                                                <td>
                                                    {% if result.safety_safe %}
                                                    <span class="badge bg-success">Safe</span>
                                                    {% else %}
                                                    <span class="badge bg-danger">Unsafe</span>
                                                    {% if result.safety_categories %}<small class="text-muted ms-1">{{ result.safety_categories | join(', ') }}</small>{% endif %}
                                                    {% endif %}
                                                </td>
                                            </tr>
                                            {% if result.pii_detected %}
                                            <tr>
                                                <td>PII</td>
                                                <td><span class="badge bg-warning text-dark">Detected: {{ result.pii_types | join(', ') }}</span></td>
                                            </tr>
                                            {% endif %}
                                        </table>

                                        {% if result.failure_reasons %}
                                        <div class="alert alert-danger py-2 px-3" style="font-size: 0.85rem;">
                                            <strong>Failure reasons:</strong>
                                            <ul class="mb-0 ps-3">
                                            {% for reason in result.failure_reasons %}
                                                <li>{{ reason }}</li>
                                            {% endfor %}
                                            </ul>
                                        </div>
                                        {% endif %}

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
                                    <div>Model <span>{{ result.model }}</span></div>
                                    <div>Tokens <span>{{ "{:,}".format(result.tokens_total) }}</span></div>
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

                                {% if result.rationale_events %}
                                <h6 class="mt-4">Decision Rationale
                                    <span class="badge bg-secondary ms-1">{{ result.rationale_events | length }}</span>
                                </h6>
                                <p class="text-muted" style="font-size:12px;">
                                    Why the agent picked each option at each decision point. Grouped
                                    by <code>input_hash</code> in cloud analytics for cross-run drift detection.
                                </p>
                                <ul class="trace-timeline">
                                    {% for ev in result.rationale_events %}
                                    {% set rid = "rationale-" ~ ri ~ "-" ~ loop.index %}
                                    <li>
                                        <div class="span-row" data-bs-toggle="collapse" data-bs-target="#{{ rid }}">
                                            <span class="span-kind kind-tool">{{ ev.decision_type }}</span>
                                            <span class="span-name">{{ ev.chosen }}</span>
                                            <div class="span-meta">
                                                {% if ev.alternatives %}
                                                <span>{{ ev.alternatives | length }} alt{% if ev.alternatives | length != 1 %}s{% endif %}</span>
                                                {% endif %}
                                                {% if ev.confidence is not none %}
                                                <span class="token-pill">conf {{ ev.confidence }}%</span>
                                                {% endif %}
                                                {% if ev.truncated %}
                                                <span class="token-pill">truncated</span>
                                                {% endif %}
                                            </div>
                                        </div>
                                        <div id="{{ rid }}" class="collapse">
                                            <div class="span-detail">
                                                <span class="detail-label">Chosen</span>
                                                <div class="detail-value">{{ ev.chosen }}</div>
                                                {% if ev.alternatives %}
                                                <span class="detail-label">Alternatives considered</span>
                                                <div class="detail-value">{{ ev.alternatives | join(", ") }}</div>
                                                {% endif %}
                                                {% if ev.rationale_text %}
                                                <span class="detail-label">Reasoning</span>
                                                <div class="detail-value prompt-text">{{ ev.rationale_text }}</div>
                                                {% endif %}
                                            </div>
                                        </div>
                                    </li>
                                    {% endfor %}
                                </ul>
                                {% endif %}
                            </div><!-- /tab-trace -->
                        </div><!-- /tab-content -->

                    </div><!-- /card-body -->
                </div><!-- /collapse -->
            </div><!-- /card -->
            {% endfor %}
        </div><!-- /testResults -->
        </div><!-- /tab-tests -->

        </div><!-- /tab-content -->

    </div><!-- /container -->

    <footer class="text-center text-muted py-4">
        Generated by <a href="https://github.com/hidai25/eval-view">EvalView</a>
    </footer>

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


