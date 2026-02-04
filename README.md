# EvalView — Pytest for AI Agents

> Your agent worked yesterday. Today it's broken. EvalView catches why.

**CI/CD for agent behavior.** Detect tool changes, output drift, cost spikes, and latency regressions — before users complain.

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

**You changed a prompt.** Now your agent calls wrong tools, hallucinates, costs 3x more, or times out. You find out when users complain.

**EvalView catches this in CI — before you deploy.**

```bash
evalview golden save .evalview/results/xxx.json   # Save working run as baseline
evalview run --diff                                # Fail CI on regression
```

[Get started in 60 seconds →](#quick-start)

---

## Why EvalView?

|  | Observability Tools | Generic Eval Frameworks | **EvalView** |
|---|:---:|:---:|:---:|
| Blocks bad deploys in CI | ❌ | ⚠️ Manual | ✅ Built-in |
| Detects tool call changes | ❌ | ❌ | ✅ |
| Tracks cost/latency regressions | ⚠️ Alerts only | ❌ | ✅ Fails CI |
| Golden baseline diffing | ❌ | ❌ | ✅ |
| Free & open source | ❌ | ✅ | ✅ |
| Works offline (Ollama) | ❌ | ⚠️ Some | ✅ |

**Use observability tools to see what happened. Use EvalView to block it from shipping.**

---

## What EvalView Catches

| Status | Meaning | Action |
|--------|---------|--------|
| **REGRESSION** | Score dropped | Fix before deploy |
| **TOOLS_CHANGED** | Different tools called | Review before deploy |
| **OUTPUT_CHANGED** | Same tools, different output | Review before deploy |
| **PASSED** | Matches baseline | Ship it |

---

## Quick Start

```bash
pip install evalview

export OPENAI_API_KEY='your-key'   # For LLM-as-judge
evalview quickstart                 # Creates test + runs it
```

**Want free local evaluation?**
```bash
evalview run --judge-provider ollama --judge-model llama3.2
```

[Full getting started guide →](docs/GETTING_STARTED.md)

---

## Add to CI in 60 Seconds

```yaml
# .github/workflows/evalview.yml
name: Agent Tests
on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: hidai25/eval-view@v0.2.4
        with:
          openai-api-key: ${{ secrets.OPENAI_API_KEY }}
          diff: true
          fail-on: 'REGRESSION'
```

PRs with regressions get blocked. Add a PR comment showing exactly what changed:

```yaml
      - run: evalview ci comment  # Posts diff to PR
        env:
          GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
```

[Full CI/CD setup →](docs/CI_CD.md)

---

## Interactive Chat Mode

Talk to your tests. Debug failures. Compare runs.

```bash
evalview chat
```

```
You: run the calculator test
🤖 Running calculator test...
✅ Passed (score: 92.5)

You: what tools did it use?
🤖 The agent used: calculator, search

You: compare to yesterday
🤖 Score: 92.5 → 87.2 (-5.3)
   Tools: +1 added (validator)
   Cost: $0.003 → $0.005 (+67%)
```

Slash commands: `/run`, `/test`, `/compare`, `/traces`, `/skill`, `/adapters`

[Chat mode docs →](docs/CHAT_MODE.md)

---

## End-to-End Skills Testing

Test that your agent's code actually works — not just that the output looks right.

```yaml
tests:
  - name: creates-working-api
    input: "Create an express server with /health endpoint"
    expected:
      # Verify artifacts
      files_created: ["index.js", "package.json"]

      # Verify it compiles
      build_must_pass:
        - "npm install"
        - "npm run lint"

      # Verify it runs
      smoke_tests:
        - command: "node index.js"
          background: true
          health_check: "http://localhost:3000/health"
          expected_status: 200
          timeout: 10

      # Safety checks
      no_sudo: true
      git_clean: true
```

```bash
# Run with real Claude Code agent
evalview skill test tests.yaml --agent claude-code

# Other supported agents
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

## Features

| Feature | Description | Docs |
|---------|-------------|------|
| **Golden Traces** | Save baselines, detect regressions with `--diff` | [→](docs/GOLDEN_TRACES.md) |
| **Tool Categories** | Match by intent, not exact tool names | [→](docs/TOOL_CATEGORIES.md) |
| **Statistical Mode** | Handle flaky LLMs with `--runs N` and pass@k | [→](docs/STATISTICAL_MODE.md) |
| **Chat Mode** | AI assistant: `/run`, `/test`, `/compare` | [→](docs/CHAT_MODE.md) |
| **Skills Testing** | Validate Claude Code / OpenAI Codex skills | [→](docs/SKILLS_TESTING.md) |
| **Test Generation** | Generate 1000 tests from 1 | [→](docs/TEST_GENERATION.md) |
| **Suite Types** | Separate capability vs regression tests | [→](docs/SUITE_TYPES.md) |
| **Difficulty Levels** | Filter by `--difficulty hard`, benchmark by tier | [→](docs/STATISTICAL_MODE.md) |
| **Behavior Coverage** | Track tasks, tools, paths tested | [→](docs/BEHAVIOR_COVERAGE.md) |
| **Cost & Latency** | Automatic threshold enforcement | [→](docs/EVALUATION_METRICS.md) |
| **HTML Reports** | Interactive Plotly charts | [→](docs/CLI_REFERENCE.md) |

---

## Who Uses EvalView?

- **Teams shipping LangGraph / CrewAI agents** who need CI gates
- **Solo developers** tired of "it worked yesterday" bugs
- **Platform teams** building internal agent tooling

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

Also works with: AutoGen • Dify • Ollama • Any HTTP API

[Compatibility details →](docs/FRAMEWORK_SUPPORT.md)

---

## Documentation

| | |
|---|---|
| [Getting Started](docs/GETTING_STARTED.md) | [CLI Reference](docs/CLI_REFERENCE.md) |
| [Golden Traces](docs/GOLDEN_TRACES.md) | [CI/CD Integration](docs/CI_CD.md) |
| [Tool Categories](docs/TOOL_CATEGORIES.md) | [Statistical Mode](docs/STATISTICAL_MODE.md) |
| [Skills Testing](docs/SKILLS_TESTING.md) | [Evaluation Metrics](docs/EVALUATION_METRICS.md) |
| [FAQ](docs/FAQ.md) | [Debugging](docs/DEBUGGING.md) |

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

**Shipped:** Golden traces • Tool categories • Statistical mode • Difficulty levels • Partial sequence credit • Skills validation • E2E agent testing • Build & smoke tests • Health checks • Safety guards (`no_sudo`, `git_clean`) • Claude Code & Codex adapters • MCP servers • HTML reports

**Coming:** Multi-turn conversations • Grounded hallucination detection • Error compounding metrics • Container isolation

[Vote on features →](https://github.com/hidai25/eval-view/discussions)

---

## Contributing

Contributions welcome! See [CONTRIBUTING.md](CONTRIBUTING.md).

**License:** Apache 2.0

---

<p align="center">
  <b>Stop shipping regressions.</b><br>
  <a href="#quick-start">Get started in 60 seconds →</a>
</p>

---

*EvalView is an independent open-source project, not affiliated with LangGraph, CrewAI, OpenAI, Anthropic, or any other third party.*
