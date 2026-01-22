# EvalView — Regression Testing for AI Agents

**Catch agent regressions before they hit production.**

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

---

**Your agent worked yesterday. Today it's broken. What changed?**

You updated a prompt, swapped models, or changed a tool. Now your agent calls different tools, returns wrong outputs, costs 3x more, or takes 10x longer. You find out when users complain.

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
      - uses: hidai25/eval-view@v0.2.1
        with:
          openai-api-key: ${{ secrets.OPENAI_API_KEY }}
          diff: true
          fail-on: 'REGRESSION'
```

PRs with regressions get blocked. [Full CI/CD setup →](docs/CI_CD.md)

---

## Features

| Feature | Description | Docs |
|---------|-------------|------|
| **Golden Traces** | Save baselines, detect regressions with `--diff` | [→](docs/GOLDEN_TRACES.md) |
| **Tool Categories** | Match by intent, not exact tool names | [→](docs/TOOL_CATEGORIES.md) |
| **Statistical Mode** | Handle flaky LLMs with pass@k metrics | [→](docs/STATISTICAL_MODE.md) |
| **Chat Mode** | AI assistant: `/run`, `/test`, `/compare` | [→](docs/CHAT_MODE.md) |
| **Skills Testing** | Validate Claude Code / OpenAI Codex skills | [→](docs/SKILLS_TESTING.md) |
| **Test Generation** | Generate 1000 tests from 1 | [→](docs/TEST_GENERATION.md) |
| **Suite Types** | Separate capability vs regression tests | [→](docs/SUITE_TYPES.md) |
| **Behavior Coverage** | Track tasks, tools, paths tested | [→](docs/BEHAVIOR_COVERAGE.md) |
| **Cost & Latency** | Automatic threshold enforcement | [→](docs/EVALUATION_METRICS.md) |
| **HTML Reports** | Interactive Plotly charts | [→](docs/CLI_REFERENCE.md) |

---

## Who Uses EvalView?

- **Teams shipping LangGraph / CrewAI agents** who need CI gates
- **Solo developers** tired of "it worked yesterday" bugs
- **Platform teams** building internal agent tooling

---

## Supported Frameworks

LangGraph • CrewAI • OpenAI Assistants • Anthropic Claude • AutoGen • Dify • Ollama • Any HTTP API

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

**Shipped:** Golden traces • Tool categories • Statistical mode • Skills testing • MCP servers • HTML reports

**Coming:** Multi-turn conversations • Grounded hallucination detection • Error compounding metrics

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
