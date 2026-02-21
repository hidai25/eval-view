# EvalView — Proof that your agent still works.

> You changed a prompt. Swapped a model. Updated a tool.
> Did anything break? **Run EvalView. Know for sure.**

<p align="center">
  <img src="assets/demo.gif" alt="EvalView Demo" width="700">
</p>

<p align="center">

```bash
pip install evalview && evalview demo   # Uses your configured API key
```

</p>

<p align="center">
  <a href="https://pypi.org/project/evalview/"><img src="https://img.shields.io/pypi/dm/evalview.svg?label=downloads" alt="PyPI downloads"></a>
  <a href="https://github.com/hidai25/eval-view/stargazers"><img src="https://img.shields.io/github/stars/hidai25/eval-view?style=social" alt="GitHub stars"></a>
  <a href="https://github.com/hidai25/eval-view/actions/workflows/ci.yml"><img src="https://github.com/hidai25/eval-view/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="https://opensource.org/licenses/Apache-2.0"><img src="https://img.shields.io/badge/License-Apache_2.0-blue.svg" alt="License"></a>
</p>

<p align="center">
  🌟 <strong>Like it?</strong> Give us a ⭐ — it helps more devs discover EvalView.
</p>

---

## 🔍 What EvalView Catches

| Status | What it means | What you do |
|--------|--------------|-------------|
| ✅ **PASSED** | Agent behavior matches baseline | Ship with confidence |
| ⚠️ **TOOLS_CHANGED** | Agent is calling different tools | Review the diff |
| ⚠️ **OUTPUT_CHANGED** | Same tools, output quality shifted | Review the diff |
| ❌ **REGRESSION** | Score dropped significantly | Fix before shipping |

---

## 🤔 How It Works

**Simple workflow (recommended):**

```bash
# 1. Your agent works correctly
evalview snapshot                 # 📸 Save current behavior as baseline

# 2. You change something (prompt, model, tools)
evalview check                    # 🔍 Detect regressions automatically

# 3. EvalView tells you exactly what changed
#    → ✅ All clean! No regressions detected.
#    → ⚠️ TOOLS_CHANGED: +web_search, -calculator
#    → ❌ REGRESSION: score 85 → 71
```

**Advanced workflow (more control):**

```bash
evalview run --save-golden        # Save specific result as baseline
evalview run --diff               # Compare with custom options
```

That's it. **Deterministic proof, no LLM-as-judge required, no API keys needed.**

### 🎯 New: Habit-Forming Regression Detection

EvalView now tracks your progress and celebrates wins:

```bash
evalview check
# 🔍 Comparing against your baseline...
# ✨ All clean! No regressions detected.
# 🎯 5 clean checks in a row! You're on a roll.
```

**Features:**
- 🔥 **Streak tracking** — Celebrate consecutive clean checks (3, 5, 10, 25+ milestones)
- 📊 **Health score** — See your project's stability at a glance
- 🔔 **Smart recaps** — "Since last time" summaries to stay in context
- 📈 **Progress visualization** — Track improvement over time

### 🎨 Multi-Reference Goldens (for non-deterministic agents)

Some agents produce valid variations. Save up to 5 golden variants per test:

```bash
# Save multiple acceptable behaviors
evalview snapshot --variant variant1
evalview snapshot --variant variant2

# EvalView compares against ALL variants, passes if ANY match
evalview check
# ✅ Matched variant 2/3
```

Perfect for LLM-based agents with creative variation.

---

## 🚀 Quick Start

1. **Install EvalView**

    ```bash
    pip install evalview
    ```

2. **Try the demo** (zero setup, no API key)

    ```bash
    evalview demo
    ```

3. **Set up a working example** in 2 minutes

    ```bash
    evalview quickstart
    ```

4. **Want LLM-as-judge scoring too?**

    ```bash
    export OPENAI_API_KEY='your-key'
    evalview run
    ```

5. **Prefer local/free evaluation?**

    ```bash
    evalview run --judge-provider ollama --judge-model llama3.2
    ```

[Full getting started guide →](docs/GETTING_STARTED.md)

---

## 🆕 New in v0.3: Visual Reports + Claude Code MCP

**Beautiful HTML reports** — one command, auto-opens in browser:

```bash
evalview inspect                          # Latest run → visual report
evalview inspect latest --notes "PR #42"  # With context
evalview visualize --compare run1.json --compare run2.json  # Side-by-side runs
```

The report includes tabbed **Overview** (KPI cards, score charts, cost-per-query table), **Execution Trace** (Mermaid sequence diagrams with full query/response), **Diffs** (golden vs actual), and **Timeline** (step latencies). Glassmorphism dark theme, auto-opens in browser, fully self-contained HTML.

**Claude Code MCP** — ask Claude inline without leaving your conversation:

```bash
claude mcp add --transport stdio evalview -- evalview mcp serve
cp CLAUDE.md.example CLAUDE.md
```

8 MCP tools: `create_test`, `run_snapshot`, `run_check`, `list_tests`, `validate_skill`, `generate_skill_tests`, `run_skill_test`, `generate_visual_report`

👉 Jump to [Claude Code Integration (MCP)](#-claude-code-integration-mcp)

---

## 💡 Why EvalView?

- 🔄 **Automatic regression detection** — Know instantly when your agent breaks
- 📸 **Golden baseline diffing** — Save known-good behavior, compare every change
- 🔑 **Works without API keys** — Deterministic scoring, no LLM-as-judge needed
- 💸 **Free & open source** — No vendor lock-in, no SaaS pricing
- 🏠 **Works offline** — Use Ollama for fully local evaluation

|  | Observability (LangSmith) | Benchmarks (Braintrust) | **EvalView** |
|---|:---:|:---:|:---:|
| **Answers** | "What did my agent do?" | "How good is my agent?" | **"Did my agent change?"** |
| Detects regressions | ❌ | ⚠️ Manual | ✅ Automatic |
| Golden baseline diffing | ❌ | ❌ | ✅ |
| Works without API keys | ❌ | ❌ | ✅ |
| Free & open source | ❌ | ❌ | ✅ |
| Works offline (Ollama) | ❌ | ⚠️ Some | ✅ |

**Use observability tools to see what happened. Use EvalView to prove it didn't break.**

---

## 🧭 Explore & Learn

### 💬 Interactive Chat

Talk to your tests. Debug failures. Compare runs.

```bash
evalview chat
```

```
You: run the calculator test
🤖 Running calculator test...
✅ Passed (score: 92.5)

You: compare to yesterday
🤖 Score: 92.5 → 87.2 (-5.3)
   Tools: +1 added (validator)
   Cost: $0.003 → $0.005 (+67%)
```

Slash commands: `/run`, `/test`, `/compare`, `/traces`, `/skill`, `/adapters`

[Chat mode docs →](docs/CHAT_MODE.md)

### 🏋️ EvalView Gym

Practice agent eval patterns with guided exercises.

```bash
evalview gym
```

---

## ⚡ Supported Agents & Frameworks

| Agent | E2E Testing | Trace Capture |
|-------|:-----------:|:-------------:|
| **Claude Code** | ✅ | ✅ |
| **OpenAI Codex** | ✅ | ✅ |
| **OpenClaw** | ✅ | ✅ |
| **LangGraph** | ✅ | ✅ |
| **CrewAI** | ✅ | ✅ |
| **OpenAI Assistants** | ✅ | ✅ |
| **Custom (any CLI/API)** | ✅ | ✅ |

Also works with: AutoGen • Dify • Ollama • HuggingFace • Any HTTP API

[Compatibility details →](docs/FRAMEWORK_SUPPORT.md)

---

## 🔧 Automate It

### GitHub Actions

```bash
evalview init --ci    # Generates workflow file
```

Or add manually:

```yaml
# .github/workflows/evalview.yml
name: Agent Health Check
on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: hidai25/eval-view@v0.2.5
        with:
          openai-api-key: ${{ secrets.OPENAI_API_KEY }}
          command: check                   # Use new check command
          fail-on: 'REGRESSION'            # Block PRs on regressions
          json: true                       # Structured output for CI
```

**Or use the CLI directly:**

```yaml
      - run: evalview check --fail-on REGRESSION --json
        env:
          OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
```

PRs with regressions get blocked. Add a PR comment showing exactly what changed:

```yaml
      - run: evalview ci comment
        env:
          GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
```

[Full CI/CD setup →](docs/CI_CD.md)

---

## 🤖 Claude Code Integration (MCP)

**Test your agent without leaving the conversation.** EvalView runs as an MCP server inside Claude Code — ask "did my refactor break anything?" and get the answer inline.

### Setup (3 steps, one-time)

```bash
# 1. Install
pip install evalview

# 2. Connect to Claude Code
claude mcp add --transport stdio evalview -- evalview mcp serve

# 3. Make Claude Code proactive (auto-checks after every edit)
cp CLAUDE.md.example CLAUDE.md
```

### What you get

7 tools Claude Code can call on your behalf:

**Agent regression testing:**

| Tool | What it does |
|------|-------------|
| `create_test` | Generate a test case from natural language — no YAML needed |
| `run_snapshot` | Capture current agent behavior as the golden baseline |
| `run_check` | Detect regressions vs baseline, returns structured JSON diff |
| `list_tests` | Show all golden baselines with scores and timestamps |

**Skills testing (full 3-phase workflow):**

| Tool | Phase | What it does |
|------|-------|-------------|
| `validate_skill` | Pre-test | Validate SKILL.md structure before running tests |
| `generate_skill_tests` | Pre-test | Auto-generate test cases from a SKILL.md |
| `run_skill_test` | Test | Run Phase 1 (deterministic) + Phase 2 (rubric) evaluation |

### How it works in practice

```
You: Add a test for my weather agent
Claude: [create_test] ✅ Created tests/weather-lookup.yaml
        [run_snapshot] 📸 Baseline captured — regression detection active.

You: Refactor the weather tool to use async
Claude: [makes code changes]
        [run_check] ✨ All clean! No regressions detected.

You: Switch to a different weather API
Claude: [makes code changes]
        [run_check] ⚠️ TOOLS_CHANGED: weather_api → open_meteo
                   Output similarity: 94% — review the diff?
```

No YAML. No terminal switching. No context loss.

**Skills testing example:**
```
You: I wrote a code-reviewer skill, test it
Claude: [validate_skill] ✅ SKILL.md is valid
        [generate_skill_tests] 📝 Generated 10 tests → tests/code-reviewer-tests.yaml
        [run_skill_test] Phase 1: 9/10 ✓  Phase 2: avg 87/100
                         1 failure: skill didn't trigger on implicit input
```

### Manual server start (advanced)

```bash
evalview mcp serve                        # Uses tests/ by default
evalview mcp serve --test-path my_tests/  # Custom test directory
```

---

## 📦 Features

| Feature | Description | Docs |
|---------|-------------|------|
| 📸 **Snapshot/Check Workflow** | Simple `snapshot` → `check` commands for regression detection | [→](docs/GOLDEN_TRACES.md) |
| 🎨 **Visual Reports** | `evalview inspect` — glassmorphism HTML with traces, diffs, cost-per-query | [↑](#-new-in-v03-visual-reports--claude-code-mcp) |
| 🤖 **Claude Code MCP** | 8 tools — run checks, generate tests, test skills, visual reports inline | [↑](#-claude-code-integration-mcp) |
| 🔥 **Streak Tracking** | Habit-forming celebrations for consecutive clean checks | [→](docs/GOLDEN_TRACES.md) |
| 🎨 **Multi-Reference Goldens** | Save up to 5 variants per test for non-deterministic agents | [→](docs/GOLDEN_TRACES.md) |
| 💬 **Chat Mode** | AI assistant: `/run`, `/test`, `/compare` | [→](docs/CHAT_MODE.md) |
| 🏷️ **Tool Categories** | Match by intent, not exact tool names | [→](docs/TOOL_CATEGORIES.md) |
| 📊 **Statistical Mode** | Handle flaky LLMs with `--runs N` and pass@k | [→](docs/STATISTICAL_MODE.md) |
| 💰 **Cost & Latency** | Automatic threshold enforcement | [→](docs/EVALUATION_METRICS.md) |
| 📈 **HTML Reports** | Interactive Plotly charts | [→](docs/CLI_REFERENCE.md) |
| 🧪 **Test Generation** | Generate 1000 tests from 1 | [→](docs/TEST_GENERATION.md) |
| 🏗️ **Suite Types** | Separate capability vs regression tests | [→](docs/SUITE_TYPES.md) |
| 🎯 **Difficulty Levels** | Filter by `--difficulty hard`, benchmark by tier | [→](docs/STATISTICAL_MODE.md) |
| 🔬 **Behavior Coverage** | Track tasks, tools, paths tested | [→](docs/BEHAVIOR_COVERAGE.md) |

---

## 🔬 Advanced: Skills Testing

Test that your agent's code actually works — not just that the output looks right.
Best for teams maintaining SKILL.md workflows for Claude Code, Codex, or OpenClaw.

```yaml
tests:
  - name: creates-working-api
    input: "Create an express server with /health endpoint"
    expected:
      files_created: ["index.js", "package.json"]
      build_must_pass:
        - "npm install"
        - "npm run lint"
      smoke_tests:
        - command: "node index.js"
          background: true
          health_check: "http://localhost:3000/health"
          expected_status: 200
          timeout: 10
      no_sudo: true
      git_clean: true
```

```bash
evalview skill test tests.yaml --agent claude-code
evalview skill test tests.yaml --agent codex
evalview skill test tests.yaml --agent openclaw
evalview skill test tests.yaml --agent langgraph
```

| Check | What it catches |
|-------|-----------------|
| `build_must_pass` | Code that doesn't compile, missing dependencies |
| `smoke_tests` | Runtime crashes, wrong ports, failed health checks |
| `git_clean` | Uncommitted files, dirty working directory |
| `no_sudo` | Privilege escalation attempts |
| `max_tokens` | Cost blowouts, verbose outputs |

[Skills testing docs →](docs/SKILLS_TESTING.md)

---

## 📚 Documentation

| | |
|---|---|
| [Getting Started](docs/GETTING_STARTED.md) | [CLI Reference](docs/CLI_REFERENCE.md) |
| [Golden Traces](docs/GOLDEN_TRACES.md) | [CI/CD Integration](docs/CI_CD.md) |
| [Tool Categories](docs/TOOL_CATEGORIES.md) | [Statistical Mode](docs/STATISTICAL_MODE.md) |
| [Chat Mode](docs/CHAT_MODE.md) | [Evaluation Metrics](docs/EVALUATION_METRICS.md) |
| [Skills Testing](docs/SKILLS_TESTING.md) | [Debugging](docs/DEBUGGING.md) |
| [FAQ](docs/FAQ.md) | |

**Guides:** [Testing LangGraph in CI](guides/pytest-for-ai-agents-langgraph-ci.md) • [Detecting Hallucinations](guides/detecting-llm-hallucinations-in-ci.md)

---

## 📂 Examples

| Framework | Link |
|-----------|------|
| Claude Code (E2E) | [examples/agent-test/](examples/agent-test/) |
| LangGraph | [examples/langgraph/](examples/langgraph/) |
| CrewAI | [examples/crewai/](examples/crewai/) |
| Anthropic Claude | [examples/anthropic/](examples/anthropic/) |
| Dify | [examples/dify/](examples/dify/) |
| Ollama (Local) | [examples/ollama/](examples/ollama/) |

**Node.js?** See [@evalview/node](sdks/node/)

---

## 🗺️ Roadmap

**Shipped:** Golden traces • **Snapshot/check workflow** • **Streak tracking & celebrations** • **Multi-reference goldens** • Tool categories • Statistical mode • Difficulty levels • Partial sequence credit • Skills validation • E2E agent testing • Build & smoke tests • Health checks • Safety guards (`no_sudo`, `git_clean`) • Claude Code & Codex adapters • **Opus 4.6 cost tracking** • MCP servers • HTML reports • Interactive chat mode • EvalView Gym

**Coming:** Agent Teams trace analysis • Multi-turn conversations • Grounded hallucination detection • Error compounding metrics • Container isolation

[Vote on features →](https://github.com/hidai25/eval-view/discussions)

---

## 🤝 Get Help & Contributing

- **Questions?** [GitHub Discussions](https://github.com/hidai25/eval-view/discussions)
- **Bugs?** [GitHub Issues](https://github.com/hidai25/eval-view/issues)
- **Want setup help?** Email hidai@evalview.com — happy to help configure your first tests
- **Contributing?** See [CONTRIBUTING.md](CONTRIBUTING.md)

**License:** Apache 2.0

---

### ⭐ Thank You for the Support!

[![Star History Chart](https://api.star-history.com/svg?repos=hidai25/eval-view&type=Date)](https://star-history.com/#hidai25/eval-view&Date)

🌟 **Don't miss out on future updates! Star the repo and be the first to know about new features.**

---

<p align="center">
  <b>Proof that your agent still works.</b><br>
  <a href="#-quick-start">Get started →</a>
</p>

---

*EvalView is an independent open-source project, not affiliated with LangGraph, CrewAI, OpenAI, Anthropic, or any other third party.*
