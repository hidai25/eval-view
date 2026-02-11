# EvalView — Proof that your agent still works.

> You changed a prompt. Swapped a model. Updated a tool.
> Did anything break? **Run EvalView. Know for sure.**

<p align="center">
  <img src="assets/demo.gif" alt="EvalView Demo" width="700">
</p>

```bash
pip install evalview && evalview demo   # No API key needed
```

[![PyPI downloads](https://img.shields.io/pypi/dm/evalview.svg?label=downloads)](https://pypi.org/project/evalview/)
[![GitHub stars](https://img.shields.io/github/stars/hidai25/eval-view?style=social)](https://github.com/hidai25/eval-view/stargazers)
[![CI](https://github.com/hidai25/eval-view/actions/workflows/ci.yml/badge.svg)](https://github.com/hidai25/eval-view/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

**Like it?** Give us a ⭐ — it helps more devs discover EvalView.

---

## What EvalView Catches

| Status | What it means | What you do |
|--------|--------------|-------------|
| ✅ **PASSED** | Agent behavior matches baseline | Ship with confidence |
| ⚠️ **TOOLS_CHANGED** | Agent is calling different tools | Review the diff |
| ⚠️ **OUTPUT_CHANGED** | Same tools, output quality shifted | Review the diff |
| ❌ **REGRESSION** | Score dropped significantly | Fix before shipping |

---

## How It Works

```
1. Your agent works correctly
   → evalview run --save-golden          # Save it as your baseline

2. You change something (prompt, model, tools)
   → evalview run --diff                  # Compare against baseline

3. EvalView tells you exactly what changed
   → REGRESSION: score 85 → 71
   → TOOLS_CHANGED: +web_search, -calculator
   → Agent healthy. No regressions detected.
```

That's it. **Deterministic proof, no LLM-as-judge required, no API keys needed.**

---

## Quick Start

```bash
pip install evalview
evalview quickstart                 # Working example in 2 minutes
```

Or try the demo first (zero setup):
```bash
evalview demo                       # See regression detection in action
```

**Want LLM-as-judge scoring too?**
```bash
export OPENAI_API_KEY='your-key'
evalview run                        # Adds output quality scoring
```

**Prefer local/free evaluation?**
```bash
evalview run --judge-provider ollama --judge-model llama3.2
```

[Full getting started guide →](docs/GETTING_STARTED.md)

---

## Why EvalView?

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

## Explore & Learn

### Interactive Chat

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

### EvalView Gym

Practice agent eval patterns with guided exercises.

```bash
evalview gym
```

---

## Automate It

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
          diff: true
          fail-on: 'REGRESSION'
```

PRs with regressions get blocked. Add a PR comment showing exactly what changed:

```yaml
      - run: evalview ci comment
        env:
          GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
```

[Full CI/CD setup →](docs/CI_CD.md)

---

## Supported Agents & Frameworks

| Agent | E2E Testing | Trace Capture |
|-------|:-----------:|:-------------:|
| **Claude Code** | ✅ | ✅ |
| **OpenAI Codex** | ✅ | ✅ |
| **LangGraph** | ✅ | ✅ |
| **CrewAI** | ✅ | ✅ |
| **OpenAI Assistants** | ✅ | ✅ |
| **Custom (any CLI/API)** | ✅ | ✅ |

Also works with: AutoGen • Dify • Ollama • HuggingFace • Any HTTP API

[Compatibility details →](docs/FRAMEWORK_SUPPORT.md)

---

## Features

| Feature | Description | Docs |
|---------|-------------|------|
| **Golden Traces** | Save baselines, detect regressions with `--diff` | [→](docs/GOLDEN_TRACES.md) |
| **Chat Mode** | AI assistant: `/run`, `/test`, `/compare` | [→](docs/CHAT_MODE.md) |
| **Tool Categories** | Match by intent, not exact tool names | [→](docs/TOOL_CATEGORIES.md) |
| **Statistical Mode** | Handle flaky LLMs with `--runs N` and pass@k | [→](docs/STATISTICAL_MODE.md) |
| **Cost & Latency** | Automatic threshold enforcement | [→](docs/EVALUATION_METRICS.md) |
| **HTML Reports** | Interactive Plotly charts | [→](docs/CLI_REFERENCE.md) |
| **Test Generation** | Generate 1000 tests from 1 | [→](docs/TEST_GENERATION.md) |
| **Suite Types** | Separate capability vs regression tests | [→](docs/SUITE_TYPES.md) |
| **Difficulty Levels** | Filter by `--difficulty hard`, benchmark by tier | [→](docs/STATISTICAL_MODE.md) |
| **Behavior Coverage** | Track tasks, tools, paths tested | [→](docs/BEHAVIOR_COVERAGE.md) |

---

## Advanced: Skills Testing

Test that your agent's code actually works — not just that the output looks right.
Best for teams maintaining SKILL.md workflows for Claude Code or Codex.

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

## Documentation

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

## Examples

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

## Get Help

- **Questions?** [GitHub Discussions](https://github.com/hidai25/eval-view/discussions)
- **Bugs?** [GitHub Issues](https://github.com/hidai25/eval-view/issues)
- **Want setup help?** Email hidai@evalview.com — happy to help configure your first tests

---

## Roadmap

**Shipped:** Golden traces • Tool categories • Statistical mode • Difficulty levels • Partial sequence credit • Skills validation • E2E agent testing • Build & smoke tests • Health checks • Safety guards (`no_sudo`, `git_clean`) • Claude Code & Codex adapters • **Opus 4.6 cost tracking** • MCP servers • HTML reports • Interactive chat mode • EvalView Gym

**Coming:** Agent Teams trace analysis • Multi-turn conversations • Grounded hallucination detection • Error compounding metrics • Container isolation

[Vote on features →](https://github.com/hidai25/eval-view/discussions)

---

## Contributing

Contributions welcome! See [CONTRIBUTING.md](CONTRIBUTING.md).

**License:** Apache 2.0

---

<p align="center">
  <b>Proof that your agent still works.</b><br>
  <a href="#quick-start">Get started →</a>
</p>

---

*EvalView is an independent open-source project, not affiliated with LangGraph, CrewAI, OpenAI, Anthropic, or any other third party.*
