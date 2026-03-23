<!-- mcp-name: io.github.hidai25/evalview-mcp -->
<!-- keywords: AI agent testing, regression detection, golden baselines -->

<p align="center">
  <img src="assets/logo.png" alt="EvalView" width="350">
  <br>
  <strong>Regression testing for AI agents.</strong><br>
  Snapshot behavior, detect regressions, block broken agents before production.
</p>

<p align="center">
  <a href="https://pypi.org/project/evalview/"><img src="https://img.shields.io/pypi/v/evalview.svg?label=release" alt="PyPI version"></a>
  <a href="https://pypi.org/project/evalview/"><img src="https://img.shields.io/pypi/dm/evalview.svg?label=downloads" alt="PyPI downloads"></a>
  <a href="https://github.com/hidai25/eval-view/stargazers"><img src="https://img.shields.io/github/stars/hidai25/eval-view?style=social" alt="GitHub stars"></a>
  <a href="https://github.com/hidai25/eval-view/actions/workflows/ci.yml"><img src="https://github.com/hidai25/eval-view/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="https://opensource.org/licenses/Apache-2.0"><img src="https://img.shields.io/badge/License-Apache_2.0-blue.svg" alt="License"></a>
  <a href="https://github.com/hidai25/eval-view/graphs/contributors"><img src="https://img.shields.io/github/contributors/hidai25/eval-view" alt="Contributors"></a>
</p>

---

EvalView sends test queries to your agent, records everything (tool calls, parameters, sequence, output, cost, latency), and diffs it against a golden baseline. When something changes, you know immediately.

```
  ✓ login-flow           PASSED
  ⚠ refund-request       TOOLS_CHANGED
      - lookup_order → check_policy → process_refund
      + lookup_order → check_policy → process_refund → escalate_to_human
  ✗ billing-dispute      REGRESSION  -30 pts
      Score: 85 → 55  Output similarity: 35%
```

Normal tests catch crashes. Tracing shows what happened after the fact. EvalView catches the harder class: the agent returns `200` but silently takes the wrong tool path, skips a clarification, or degrades output quality after a model update.

<p align="center">
  <img src="assets/hero.jpg" alt="EvalView — multi-turn execution trace with sequence diagram" width="860">
</p>

## Quick Start

```bash
pip install evalview
```

**Already have a local agent running?**

```bash
evalview init        # Detect agent, create starter suite
evalview snapshot    # Save current behavior as baseline
evalview check       # Catch regressions after every change
```

**No agent yet?**

```bash
evalview demo        # See regression detection live (~30 seconds, no API key)
```

**Want a real working agent?**

Starter repo: [evalview-support-automation-template](https://github.com/hidai25/evalview-support-automation-template)  
An LLM-backed support automation agent with built-in EvalView regression tests.

```bash
git clone https://github.com/hidai25/evalview-support-automation-template
cd evalview-support-automation-template
make run
```

**Other entry paths:**

```bash
# Generate tests from a live agent
evalview generate --agent http://localhost:8000

# Capture real user flows via proxy
evalview capture --agent http://localhost:8000/invoke

# Capture a multi-turn conversation as one test
evalview capture --agent http://localhost:8000/invoke --multi-turn

# Generate from existing logs
evalview generate --from-log traffic.jsonl
```

## How It Works

```
┌────────────┐      ┌──────────┐      ┌──────────────┐
│ Test Cases  │ ──→  │ EvalView │ ──→  │  Your Agent   │
│   (YAML)   │      │          │ ←──  │ local / cloud │
└────────────┘      └──────────┘      └──────────────┘
```

1. **`evalview init`** — detects your running agent, creates a starter test suite
2. **`evalview snapshot`** — runs tests, saves traces as baselines (picks judge model on first run)
3. **`evalview check`** — replays tests, diffs against baselines, opens HTML report with results
4. **`evalview monitor`** — runs checks continuously with optional Slack alerts

```bash
evalview snapshot list              # See all saved baselines
evalview snapshot show "my-test"    # Inspect a baseline
evalview snapshot delete "my-test"  # Remove a baseline
evalview snapshot --preview         # See what would change without saving
evalview snapshot --reset           # Clear all and start fresh
evalview replay                     # List tests, or: evalview replay "my-test"
```

**Your data stays local by default.** Nothing leaves your machine unless you opt in to cloud sync via `evalview login`.

## Two Modes, One CLI

EvalView has two complementary ways to test your agent:

### Regression Gating — *"Did my agent change?"*

Snapshot known-good behavior, then detect when something drifts.

```bash
evalview snapshot              # Capture current behavior as baseline
evalview check                 # Compare against baseline after every change
evalview check --judge opus    # Use a specific judge model (sonnet, gpt-5.4, deepseek...)
evalview monitor               # Continuous checks with Slack alerts
```

### Evaluation — *"How good is my agent?"*

Auto-generate tests and score your agent's quality right now.

```bash
evalview generate           # LLM generates realistic tests from your agent
evalview run                # Execute tests, score with LLM judge, get HTML report
```

Both modes start the same way: `evalview demo` → `evalview init` → then pick your path.

## What It Catches

| Status | Meaning | Action |
|--------|---------|--------|
| ✅ **PASSED** | Behavior matches baseline | Ship with confidence |
| ⚠️ **TOOLS_CHANGED** | Different tools called | Review the diff |
| ⚠️ **OUTPUT_CHANGED** | Same tools, output shifted | Review the diff |
| ❌ **REGRESSION** | Score dropped significantly | Fix before shipping |

## Four Scoring Layers

| Layer | What it checks | Needs API key? | Cost |
|-------|---------------|:--------------:|------|
| **Tool calls + sequence** | Exact tool names, order, parameters | No | Free |
| **Code-based checks** | Regex, JSON schema, contains/not_contains | No | Free |
| **Semantic similarity** | Output meaning via embeddings | `OPENAI_API_KEY` | ~$0.00004/test |
| **LLM-as-judge** | Output quality scored by LLM | Any provider key | ~$0.01/test |

The first two layers alone catch most regressions — fully offline, zero cost. The LLM judge supports GPT-5.4, Claude Opus/Sonnet, Gemini, DeepSeek, Grok, and Ollama (free local). EvalView asks which model to use on first run, or use `--judge sonnet`.

Every test shows a score breakdown so you know exactly what pulled the score down:

```
Score Breakdown
  Tools 100% ×30%    Output 42/100 ×50%    Sequence ✓ ×20%    = 54/100
  ↑ tools were fine   ↑ this is the problem
```

## Multi-Turn Testing

```yaml
name: refund-needs-order-number
turns:
  - query: "I want a refund"
    expected:
      output:
        contains: ["order number"]
  - query: "Order 4812"
    expected:
      tools: ["lookup_order", "check_policy"]
      forbidden_tools: ["delete_order"]
      output:
        contains: ["refund", "processed"]
        not_contains: ["error"]
thresholds:
  min_score: 70
```

Each turn is evaluated independently — tool usage, forbidden tools, output content, and **LLM judge scoring per turn** (not just the final response). The judge sees conversation history for context but scores each turn against its own query:

```
Turn 1 (92/100) ✓: Correctly processed refund, grounded in tool results
Turn 2 (90/100) ✓: Clear timeline, consistent with turn 1
Turn 3 (85/100) ✓: Appropriate escalation advice
```

No more false low scores from judging turn 3's answer against turn 1's question.

Capture multi-turn conversations from real traffic:

```bash
evalview capture --agent http://localhost:8000/invoke --multi-turn
```

## Key Features

| Feature | Description | Docs |
|---------|-------------|------|
| **Baseline diffing** | Tool call + parameter + output regression detection | [Docs](docs/GOLDEN_TRACES.md) |
| **Multi-turn testing** | Per-turn tool, forbidden_tools, and output checks | [Docs](#multi-turn-testing) |
| **Multi-turn capture** | `capture --multi-turn` records conversations as tests | [Docs](#multi-turn-testing) |
| **Multi-reference baselines** | Up to 5 variants for non-deterministic agents | [Docs](docs/GOLDEN_TRACES.md) |
| **`forbidden_tools`** | Safety contracts — hard-fail on any violation | [Docs](docs/YAML_SCHEMA.md) |
| **Semantic similarity** | Embedding-based output comparison | [Docs](docs/EVALUATION_METRICS.md) |
| **Production monitoring** | `evalview monitor` with Slack alerts and JSONL history | [Docs](#production-monitoring) |
| **A/B comparison** | `evalview compare --v1 <url> --v2 <url>` | [Docs](docs/CLI_REFERENCE.md) |
| **Test generation** | `evalview generate` — discovers your agent's domain, generates relevant tests | [Docs](docs/TEST_GENERATION.md) |
| **Per-turn judge scoring** | Multi-turn output quality scored per turn with conversation context | [Docs](#multi-turn-testing) |
| **Silent model detection** | Alerts when LLM provider updates the model version | [Docs](docs/GOLDEN_TRACES.md) |
| **Gradual drift detection** | Trend analysis across check history | [Docs](docs/GOLDEN_TRACES.md) |
| **Statistical mode (pass@k)** | Run N times, require a pass rate | [Docs](docs/STATISTICAL_MODE.md) |
| **HTML trace replay** | Auto-opens after check with full trace details | [Docs](docs/CLI_REFERENCE.md) |
| **Verified cost tracking** | Token breakdown (in/out) with model pricing rates | [Docs](docs/COST_TRACKING.md) |
| **Judge model picker** | Choose GPT, Claude, Gemini, DeepSeek, or Ollama (free) | [Docs](docs/EVALUATION_METRICS.md) |
| **Pytest plugin** | `evalview_check` fixture for standard pytest | [Docs](#pytest-plugin) |
| **PR comments + alerts** | Cost/latency spikes, model changes, collapsible diffs | [Docs](docs/CI_CD.md) |
| **GitHub Actions job summary** | Results visible in Actions UI, not just PR comments | [Docs](docs/CI_CD.md) |
| **Git hooks** | Pre-push regression blocking, zero CI config | [Docs](docs/CI_CD.md) |
| **LLM judge caching** | ~80% cost reduction in statistical mode | [Docs](docs/EVALUATION_METRICS.md) |
| **Python API** | `gate()` / `gate_async()` — programmatic regression checks | [Docs](#python-api) |
| **Quick mode** | `gate(quick=True)` — no judge, $0, sub-second | [Docs](#python-api) |
| **OpenClaw integration** | Regression gate skill + `gate_or_revert()` helpers | [Docs](#openclaw-integration) |
| **Terminal dashboard** | Scorecard, sparkline trends, confidence scoring | — |
| **Snapshot preview** | `evalview snapshot --preview` — dry-run before saving | — |
| **Skills testing** | E2E testing for Claude Code, Codex, OpenClaw | [Docs](docs/SKILLS_TESTING.md) |

## Supported Frameworks

Works with **LangGraph, CrewAI, OpenAI, Claude, Mistral, HuggingFace, Ollama, MCP, and any HTTP API**.

| Agent | E2E Testing | Trace Capture |
|-------|:-----------:|:-------------:|
| LangGraph | ✅ | ✅ |
| CrewAI | ✅ | ✅ |
| OpenAI Assistants | ✅ | ✅ |
| Claude Code | ✅ | ✅ |
| OpenClaw | ✅ | ✅ |
| Ollama | ✅ | ✅ |
| Any HTTP API | ✅ | ✅ |

[Framework details →](docs/FRAMEWORK_SUPPORT.md) | [Flagship starter →](https://github.com/hidai25/evalview-support-automation-template) | [Starter examples →](examples/)

## CI/CD Integration

Block broken agents in every PR. One step — PR comments, artifacts, and job summary are automatic.

```yaml
# .github/workflows/evalview.yml — copy this, add your secret, done
name: EvalView Agent Check
on: [pull_request, push]

jobs:
  agent-check:
    runs-on: ubuntu-latest
    permissions:
      pull-requests: write
    steps:
      - uses: actions/checkout@v4

      - name: Check for agent regressions
        uses: hidai25/eval-view@main
        with:
          openai-api-key: ${{ secrets.OPENAI_API_KEY }}
```

That's it. The action automatically:
- Runs `evalview check` against your golden baselines
- Posts/updates a PR comment with pass/fail, tool diffs, cost/latency alerts
- Writes a job summary visible in the Actions UI
- Uploads results + HTML report as artifacts

Common options: `strict: 'true'` | `fail-on: 'REGRESSION,TOOLS_CHANGED'` | `mode: 'run'` | `filter: 'my-test'` | `post-comment: 'false'`

**What lands on your PR:**

```
## ✅ EvalView: PASSED

| Metric | Value |
|--------|-------|
| Tests | 5/5 unchanged (100%) |

---
*Generated by EvalView*
```

When something breaks:

```
## ❌ EvalView: REGRESSION

> **Alerts**
> - 💸 Cost spike: $0.02 → $0.08 (+300%)
> - 🤖 Model changed: gpt-5.4 → gpt-5.4-mini

| Metric | Value |
|--------|-------|
| Tests | 3/5 unchanged (60%) |
| Regressions | 1 |
| Tools Changed | 1 |
| Model Changed | gpt-5.4 → gpt-5.4-mini |

### Changes from Baseline
- ❌ **search-flow**: score -15.0, 1 tool change(s)
- ⚠️ **create-flow**: 1 tool change(s)
```

Also works with pre-push hooks — zero CI config:

```bash
evalview install-hooks    # Pre-push regression blocking
```

[Full CI/CD guide →](docs/CI_CD.md)

## Production Monitoring

```bash
evalview monitor                                         # Check every 5 min
evalview monitor --interval 60                           # Every minute
evalview monitor --slack-webhook https://hooks.slack.com/services/...
evalview monitor --history monitor.jsonl                 # JSONL for dashboards
```

New regressions trigger Slack alerts. Recoveries send all-clear. No spam on persistent failures.

[Monitor config options →](docs/CLI_REFERENCE.md)

## Python API

Use EvalView as a library — no CLI, no subprocess, no output parsing.

```python
from evalview import gate, DiffStatus

result = gate(test_dir="tests/")

result.passed          # bool — True if no regressions
result.status          # DiffStatus.PASSED / REGRESSION / TOOLS_CHANGED
result.summary         # .total, .unchanged, .regressions, .tools_changed
result.diffs           # List[TestDiff] — per-test scores and tool diffs
```

**Quick mode** — skip the LLM judge for free, sub-second checks:

```python
result = gate(test_dir="tests/", quick=True)  # deterministic only, $0
```

**Async** — for agent frameworks already in an event loop:

```python
result = await gate_async(test_dir="tests/")
```

**Autonomous loops** — gate + auto-revert on regression:

```python
from evalview.openclaw import gate_or_revert

make_code_change()
if not gate_or_revert("tests/", quick=True):
    # Change was reverted — try a different approach
    try_alternative()
```

## OpenClaw Integration

Use EvalView as a regression gate in autonomous agent loops.

```bash
evalview openclaw install                    # Install gate skill into workspace
evalview openclaw check --path tests/        # Check and auto-revert on regression
```

Or programmatically:

```python
from evalview.openclaw import gate_or_revert

make_code_change()
if not gate_or_revert("tests/", quick=True):
    try_alternative()  # Change was reverted
```

## Pytest Plugin

```python
def test_weather_regression(evalview_check):
    diff = evalview_check("weather-lookup")
    assert diff.overall_severity.value != "regression", diff.summary()
```

```bash
pip install evalview    # Plugin registers automatically
pytest                  # Runs alongside your existing tests
```

## Claude Code (MCP)

```bash
claude mcp add --transport stdio evalview -- evalview mcp serve
```

8 tools: `create_test`, `run_snapshot`, `run_check`, `list_tests`, `validate_skill`, `generate_skill_tests`, `run_skill_test`, `generate_visual_report`

<details>
<summary><strong>MCP setup details</strong></summary>

```bash
# 1. Install
pip install evalview

# 2. Connect to Claude Code
claude mcp add --transport stdio evalview -- evalview mcp serve

# 3. Make Claude Code proactive
cp CLAUDE.md.example CLAUDE.md
```

Then just ask Claude: "did my refactor break anything?" and it runs `run_check` inline.

</details>

## Why EvalView?

|  | LangSmith | Braintrust | Promptfoo | **EvalView** |
|---|:---:|:---:|:---:|:---:|
| **Primary focus** | Observability | Scoring | Prompt comparison | **Regression detection** |
| Tool call + parameter diffing | — | — | — | **Yes** |
| Golden baseline regression | — | Manual | — | **Automatic** |
| PR comments with alerts | — | — | — | **Cost, latency, model change** |
| Works without API keys | No | No | Partial | **Yes** |
| Production monitoring | Tracing | — | — | **Check loop + Slack** |

[Detailed comparisons →](docs/COMPARISONS.md)

## Documentation

| Getting Started | Core Features | Integrations |
|---|---|---|
| [Getting Started](docs/GETTING_STARTED.md) | [Golden Traces](docs/GOLDEN_TRACES.md) | [CI/CD](docs/CI_CD.md) |
| [CLI Reference](docs/CLI_REFERENCE.md) | [Evaluation Metrics](docs/EVALUATION_METRICS.md) | [MCP Contracts](docs/MCP_CONTRACTS.md) |
| [FAQ](docs/FAQ.md) | [Test Generation](docs/TEST_GENERATION.md) | [Skills Testing](docs/SKILLS_TESTING.md) |
| [YAML Schema](docs/YAML_SCHEMA.md) | [Statistical Mode](docs/STATISTICAL_MODE.md) | [Chat Mode](docs/CHAT_MODE.md) |
| [Framework Support](docs/FRAMEWORK_SUPPORT.md) | [Behavior Coverage](docs/BEHAVIOR_COVERAGE.md) | [Debugging](docs/DEBUGGING.md) |

## Contributing

- **Bug or feature request?** Run `evalview feedback` or [open an issue](https://github.com/hidai25/eval-view/issues)
- **Questions?** [GitHub Discussions](https://github.com/hidai25/eval-view/discussions)
- **Setup help?** Email hidai@evalview.com
- **Contributing?** See [CONTRIBUTING.md](CONTRIBUTING.md)

**License:** Apache 2.0

---

### Star History

[![Star History Chart](https://api.star-history.com/svg?repos=hidai25/eval-view&type=Date)](https://star-history.com/#hidai25/eval-view&Date)
