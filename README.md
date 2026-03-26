<!-- mcp-name: io.github.hidai25/evalview-mcp -->
<!-- keywords: AI agent testing, regression detection, golden baselines -->

<p align="center">
  <img src="assets/logo.png" alt="EvalView" width="350">
  <br>
  <strong>The open-source regression gate for AI agents.</strong><br>
  Think Playwright, but for tool-calling and multi-turn AI agents.
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

Your agent returns `200` but silently takes the wrong tool path, skips a clarification, or degrades output quality after a model update. Normal tests don't catch this. **EvalView does.**

**Catches silent model and runtime updates.** EvalView looks for declared model swaps, runtime fingerprint changes, and coordinated drift across tests so you can tell "provider changed something" from "my code broke."

<p align="center">
  <video src="assets/evalview-demo.mp4" controls muted playsinline preload="metadata" width="1100">
    <a href="assets/evalview-demo.mp4">Watch the EvalView demo</a>
  </video>
  <br>
  <strong>30-second live demo.</strong> If the inline player does not render, open the <a href="assets/evalview-demo.mp4">MP4 directly</a>.
</p>

```
  ✓ login-flow           PASSED
  ⚠ refund-request       TOOLS_CHANGED
      - lookup_order → check_policy → process_refund
      + lookup_order → check_policy → process_refund → escalate_to_human
  ✗ billing-dispute      REGRESSION  -30 pts
      Score: 85 → 55  Output similarity: 35%
```

## Quick Start

```bash
pip install evalview
```

```bash
evalview init        # Detect agent, auto-configure profile + starter suite
evalview snapshot    # Save current behavior as baseline
evalview check       # Catch regressions after every change
```

That's it. Three commands to regression-test any AI agent. `init` auto-detects your agent type (chat, tool-use, multi-step, RAG, coding) and configures the right evaluators, thresholds, and assertions.

<details>
<summary><strong>Other install methods</strong></summary>

```bash
curl -fsSL https://raw.githubusercontent.com/hidai25/eval-view/main/install.sh | bash
```

</details>

<details>
<summary><strong>No agent yet? Try the demo</strong></summary>

```bash
evalview demo        # See regression detection live (~30 seconds, no API key)
```

Or clone a real working agent with built-in tests:

```bash
git clone https://github.com/hidai25/evalview-support-automation-template
cd evalview-support-automation-template
make run
```

</details>

<details>
<summary><strong>More entry paths</strong></summary>

```bash
evalview generate --agent http://localhost:8000           # Generate tests from a live agent
evalview capture --agent http://localhost:8000/invoke      # Capture real user flows (runs assertion wizard after)
evalview capture --agent http://localhost:8000/invoke --multi-turn  # Multi-turn conversation as one test
evalview generate --from-log traffic.jsonl                # Generate from existing logs
evalview init --profile rag                               # Override auto-detected agent profile
```

</details>

## Why EvalView?

Use LangSmith for observability. Use Braintrust for scoring. **Use EvalView for regression gating.**

|  | LangSmith | Braintrust | Promptfoo | **EvalView** |
|---|:---:|:---:|:---:|:---:|
| **Primary focus** | Observability | Scoring | Prompt comparison | **Regression detection** |
| Tool call + parameter diffing | — | — | — | **Yes** |
| Golden baseline regression | — | Manual | — | **Automatic** |
| Silent model change detection | — | — | — | **Yes** |
| Auto-heal (retry + variant proposal) | — | — | — | **Yes** |
| PR comments with alerts | — | — | — | **Cost, latency, model change** |
| Works without API keys | No | No | Partial | **Yes** |
| Production monitoring | Tracing | — | — | **Check loop + Slack** |

[Detailed comparisons →](docs/COMPARISONS.md)

## What It Catches

| Status | Meaning | Action |
|--------|---------|--------|
| ✅ **PASSED** | Behavior matches baseline | Ship with confidence |
| ⚠️ **TOOLS_CHANGED** | Different tools called | Review the diff |
| ⚠️ **OUTPUT_CHANGED** | Same tools, output shifted | Review the diff |
| ❌ **REGRESSION** | Score dropped significantly | Fix before shipping |

### Model / Runtime Change Detection

EvalView does more than compare `model_id`.

- **Declared model change**: adapter-reported model changed from baseline
- **Runtime fingerprint change**: observed model labels in the trace changed, even when the top-level model name is missing
- **Coordinated drift**: multiple tests shift together in the same check run, which often points to a silent provider rollout or runtime change

When detected, `evalview check` surfaces a run-level signal with a classification (`declared` or `suspected`), confidence level, and evidence from fingerprints, retries, and affected tests.

If the new behavior is correct, rerun `evalview snapshot` to accept the updated baseline.

**Four scoring layers** — the first two are free and offline:

| Layer | What it checks | Cost |
|-------|---------------|------|
| **Tool calls + sequence** | Exact tool names, order, parameters | Free |
| **Code-based checks** | Regex, JSON schema, contains/not_contains | Free |
| **Semantic similarity** | Output meaning via embeddings | ~$0.00004/test |
| **LLM-as-judge** | Output quality scored by LLM (GPT, Claude, Gemini, DeepSeek, Ollama) | ~$0.01/test |

```
Score Breakdown
  Tools 100% ×30%    Output 42/100 ×50%    Sequence ✓ ×20%    = 54/100
  ↑ tools were fine   ↑ this is the problem
```

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

<details>
<summary><strong>What lands on your PR</strong></summary>

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

### Changes from Baseline
- ❌ **search-flow**: score -15.0, 1 tool change(s)
- ⚠️ **create-flow**: 1 tool change(s)
```

</details>

Common options: `strict: 'true'` | `fail-on: 'REGRESSION,TOOLS_CHANGED'` | `mode: 'run'` | `filter: 'my-test'`

Also works with [pre-push hooks](docs/CI_CD.md) (`evalview install-hooks`) and [status badges](docs/CI_CD.md) (`evalview badge`).

[Full CI/CD guide →](docs/CI_CD.md)

## Watch Mode

Leave it running while you code. Every file save triggers a regression check.

```bash
evalview watch                          # Watch current dir, check on change
evalview watch --quick                  # No LLM judge — $0, sub-second
evalview watch --test "refund-flow"     # Only check one test
```

```
╭─────────────────────────── EvalView Watch ────────────────────────────╮
│   Watching   .                                                        │
│   Tests      all in tests/                                            │
│   Mode       quick (no judge, $0)                                     │
╰───────────────────────────────────────────────────────────────────────╯

14:32:07  Change detected: src/agent.py

╭──────────────────────────── Scorecard ────────────────────────────────╮
│ ████████████████████░░░░  4 passed · 1 tools changed · 0 regressions │
╰───────────────────────────────────────────────────────────────────────╯
  ⚠ TOOLS_CHANGED  refund-flow  1 tool change(s)

Watching for changes...
```

## Multi-Turn Testing

Most eval tools handle single-turn well. EvalView is built for multi-turn — clarification paths, follow-up handling, and tool use across conversations.

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

Each turn scored independently with conversation context. Per-turn judge scoring, not just final response.

## Smart DX

EvalView doesn't just run tests — it **understands your agent** and configures itself.

### Assertion Wizard — Tests From Real Traffic

Capture real interactions, get pre-configured tests. No YAML writing.

```bash
evalview capture --agent http://localhost:8000/invoke
# Use your agent normally, then Ctrl+C
```

```
Assertion Wizard — analyzing 8 captured interactions

  Agent type detected: multi-step
  Tools seen          search, extract, summarize
  Consistent sequence search -> extract -> summarize

  Suggested assertions:
    1. Lock tool sequence: search -> extract -> summarize  (recommended)
    2. Require tools: search, extract, summarize           (recommended)
    3. Max latency: 5000ms                                 (recommended)
    4. Minimum quality score: 70                           (recommended)

  Accept all recommended? [Y/n]: y
  Applied 4 assertions to 8 test files
```

### Auto-Variant Discovery — Solve Non-Determinism

Non-deterministic agents take different valid paths. Let EvalView discover and save them:

```bash
evalview check --statistical 10 --auto-variant
```

```
  search-flow  mean: 82.3, std: 8.1, flakiness: low_variance
    1. search -> extract -> summarize  (7/10 runs, avg score: 85.2)
    2. search -> summarize             (3/10 runs, avg score: 78.1)

    Save as golden variant? [Y/n]: y
    Saved variant 'auto-v1': search -> summarize
```

Run N times. Cluster the paths. Save the valid ones. Tests stop being flaky — automatically.

### Auto-Heal — Fix Flakes Without Leaving CI

Model got silently updated? Output drifted? `--heal` retries safe failures, proposes variants for borderline cases, and hard-escalates everything else. It also records when those retries were triggered by a likely model/runtime update.

```bash
evalview check --heal
```

```
  ⚠ Model update detected: gpt-5-2025-08-07 → gpt-5.1-2025-11-12 (3 tests affected)

  ✓ login-flow           PASSED
  ⚡ refund-request       HEALED   retried — non-deterministic drift
  ⚡ order-lookup         HEALED   retried — likely model/runtime update
  ◈ billing-dispute      PROPOSED saved candidate variant auto_heal_a1b2 (score 72)
  ⚠ search-flow          REVIEW   tool removed: web_search
  ✗ safety-check         BLOCKED  forbidden tool called — cannot heal

  3 resolved, 1 candidate variant saved, 1 needs review, 1 blocked.
  Model update: 2 of 3 affected tests healed via retry. Run `evalview snapshot` to rebase.
  Audit log: .evalview/healing/2026-03-25T14-30-00.json
```

**Decision policy:** Retry when tools match but output drifted (non-determinism or likely model/runtime update). Propose a variant when retry fails but score is acceptable. Never auto-resolve structural changes, forbidden tool violations, cost spikes, or score improvements. Full audit trail in `.evalview/healing/`.

**Exit code:** `0` only when every failure was resolved via retry. Proposed variants, reviews, and blocks always exit `1` — CI stays honest.

<details>
<summary><strong>Budget circuit breaker + Smart eval profiles</strong></summary>

**Budget circuit breaker** — enforced mid-execution, not post-hoc:

```bash
evalview check --budget 0.50
```

```
  $0.12 (24%) — search-flow
  $0.09 (18%) — refund-flow
  $0.31 (62%) — billing-dispute

  Budget circuit breaker tripped: $0.52 spent of $0.50 limit
  2 test(s) skipped to stay within budget
```

**Smart eval profiles** — `evalview init` detects your agent type and pre-configures evaluators:

Five profiles — `chat`, `tool-use`, `multi-step`, `rag`, `coding` — each with tailored thresholds, recommended checks, and actionable tips. Override with `--profile rag`.

</details>

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

## How It Works

```
┌────────────┐      ┌──────────┐      ┌──────────────┐
│ Test Cases  │ ──→  │ EvalView │ ──→  │  Your Agent   │
│   (YAML)   │      │          │ ←──  │ local / cloud │
└────────────┘      └──────────┘      └──────────────┘
```

1. **`evalview init`** — detects your running agent, creates a starter test suite
2. **`evalview snapshot`** — runs tests, saves traces as baselines
3. **`evalview check`** — replays tests, diffs against baselines, opens HTML report
4. **`evalview watch`** — re-runs checks on every file save
5. **`evalview monitor`** — continuous checks in production with Slack alerts

<details>
<summary><strong>Snapshot management</strong></summary>

```bash
evalview snapshot list              # See all saved baselines
evalview snapshot show "my-test"    # Inspect a baseline
evalview snapshot delete "my-test"  # Remove a baseline
evalview snapshot --preview         # See what would change without saving
evalview snapshot --reset           # Clear all and start fresh
evalview replay                     # List tests, or: evalview replay "my-test"
```

</details>

**Your data stays local by default.** Nothing leaves your machine unless you opt in to cloud sync via `evalview login`.

## Production Monitoring

```bash
evalview monitor                                         # Check every 5 min
evalview monitor --dashboard                             # Live terminal dashboard
evalview monitor --slack-webhook https://hooks.slack.com/services/...
evalview monitor --history monitor.jsonl                 # JSONL for dashboards
```

New regressions trigger Slack alerts. Recoveries send all-clear. No spam on persistent failures.

[Monitor config options →](docs/CLI_REFERENCE.md)

## Key Features

| Feature | Description | Docs |
|---------|-------------|------|
| **Assertion wizard** | Analyze captured traffic, suggest smart assertions automatically | [Above](#assertion-wizard--tests-from-real-traffic) |
| **Auto-variant discovery** | Run N times, cluster paths, save valid variants | [Above](#auto-variant-discovery--solve-non-determinism) |
| **Auto-heal** | Retry flakes, propose variants, escalate structural changes | [Above](#auto-heal--fix-flakes-without-leaving-ci) |
| **Budget circuit breaker** | Mid-execution budget enforcement with per-test cost breakdown | [Above](#smart-dx) |
| **Smart eval profiles** | Auto-detect agent type, pre-configure evaluators | [Above](#smart-dx) |
| **Baseline diffing** | Tool call + parameter + output regression detection | [Docs](docs/GOLDEN_TRACES.md) |
| **Multi-turn testing** | Per-turn tool, forbidden_tools, and output checks | [Docs](#multi-turn-testing) |
| **Multi-reference baselines** | Up to 5 variants for non-deterministic agents | [Docs](docs/GOLDEN_TRACES.md) |
| **`forbidden_tools`** | Safety contracts — hard-fail on any violation | [Docs](docs/YAML_SCHEMA.md) |
| **Watch mode** | `evalview watch` — re-run checks on file save, with dashboard | [Docs](#watch-mode) |
| **Python API** | `gate()` / `gate_async()` — programmatic regression checks | [Docs](#python-api) |
| **PR comments + alerts** | Cost/latency spikes, model changes, collapsible diffs | [Docs](docs/CI_CD.md) |
| **Terminal dashboard** | Scorecard, sparkline trends, confidence scoring | — |

<details>
<summary><strong>All features</strong></summary>

| Feature | Description | Docs |
|---------|-------------|------|
| **Multi-turn capture** | `capture --multi-turn` records conversations as tests | [Docs](#multi-turn-testing) |
| **Semantic similarity** | Embedding-based output comparison | [Docs](docs/EVALUATION_METRICS.md) |
| **Production monitoring** | `evalview monitor --dashboard` with Slack alerts and JSONL history | [Docs](#production-monitoring) |
| **A/B comparison** | `evalview compare --v1 <url> --v2 <url>` | [Docs](docs/CLI_REFERENCE.md) |
| **Test generation** | `evalview generate` — discovers your agent's domain, generates relevant tests | [Docs](docs/TEST_GENERATION.md) |
| **Per-turn judge scoring** | Multi-turn output quality scored per turn with conversation context | [Docs](#multi-turn-testing) |
| **Silent model detection** | Alerts when LLM provider updates the model version | [Docs](docs/GOLDEN_TRACES.md) |
| **Gradual drift detection** | Trend analysis across check history | [Docs](docs/GOLDEN_TRACES.md) |
| **Statistical mode (pass@k)** | Run N times, require a pass rate, auto-discover variants | [Docs](docs/STATISTICAL_MODE.md) |
| **HTML trace replay** | Auto-opens after check with full trace details | [Docs](docs/CLI_REFERENCE.md) |
| **Verified cost tracking** | Per-test cost breakdown with model pricing rates | [Docs](docs/COST_TRACKING.md) |
| **Judge model picker** | Choose GPT, Claude, Gemini, DeepSeek, or Ollama (free) | [Docs](docs/EVALUATION_METRICS.md) |
| **Pytest plugin** | `evalview_check` fixture for standard pytest | [Docs](#pytest-plugin) |
| **GitHub Actions job summary** | Results visible in Actions UI, not just PR comments | [Docs](docs/CI_CD.md) |
| **Git hooks** | Pre-push regression blocking, zero CI config | [Docs](docs/CI_CD.md) |
| **LLM judge caching** | ~80% cost reduction in statistical mode | [Docs](docs/EVALUATION_METRICS.md) |
| **Quick mode** | `gate(quick=True)` — no judge, $0, sub-second | [Docs](#python-api) |
| **OpenClaw integration** | Regression gate skill + `gate_or_revert()` helpers | [Docs](#openclaw-integration) |
| **Snapshot preview** | `evalview snapshot --preview` — dry-run before saving | — |
| **Skills testing** | E2E testing for Claude Code, Codex, OpenClaw | [Docs](docs/SKILLS_TESTING.md) |

</details>

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

<details>
<summary><strong>Quick mode, async, and autonomous loops</strong></summary>

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

</details>

## OpenClaw Integration

Use EvalView as a regression gate in autonomous agent loops.

```bash
evalview openclaw install                    # Install gate skill into workspace
evalview openclaw check --path tests/        # Check and auto-revert on regression
```

<details>
<summary><strong>Python API for autonomous loops</strong></summary>

```python
from evalview.openclaw import gate_or_revert

make_code_change()
if not gate_or_revert("tests/", quick=True):
    try_alternative()  # Change was reverted
```

</details>

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

## Agent-Friendly Docs

**Works with your coding agent out of the box.** Ask Cursor, Claude Code, or Copilot to add regression tests, build a new adapter, or debug a failing check — EvalView ships the architecture maps and task recipes they need to get it right on the first try.

- [AGENT_INSTRUCTIONS.md](AGENT_INSTRUCTIONS.md) — architecture map, contracts, invariants, verification commands
- [Agent Recipes](docs/agent-recipes/README.md) — task-specific playbooks for common extensions
- [Add an Adapter](docs/agent-recipes/add-adapter.md)
- [Add an Evaluator](docs/agent-recipes/add-evaluator.md)
- [Debug Check vs Snapshot Mismatch](docs/agent-recipes/debug-check-vs-snapshot-mismatch.md)
- [Extend the HTML Report](docs/agent-recipes/extend-html-report.md)
- [Integrate Ollama](docs/agent-recipes/integrate-ollama.md)

## Documentation

| Getting Started | Core Features | Integrations |
|---|---|---|
| [Getting Started](docs/GETTING_STARTED.md) | [Golden Traces](docs/GOLDEN_TRACES.md) | [CI/CD](docs/CI_CD.md) |
| [CLI Reference](docs/CLI_REFERENCE.md) | [Evaluation Metrics](docs/EVALUATION_METRICS.md) | [MCP Contracts](docs/MCP_CONTRACTS.md) |
| [Agent Instructions](AGENT_INSTRUCTIONS.md) | [Agent Recipes](docs/agent-recipes/README.md) | [Ollama Recipe](docs/agent-recipes/integrate-ollama.md) |
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
