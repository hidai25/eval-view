# EvalView — Catch Agent Regressions Before You Ship

> Your agent worked yesterday. Today it's broken. What changed?

**EvalView catches agent regressions** — tool changes, output changes, cost spikes, and latency spikes — before they hit production.

```bash
pip install evalview
evalview demo          # Watch a regression get caught (no API key needed)
```

[![CI](https://github.com/hidai25/eval-view/actions/workflows/ci.yml/badge.svg)](https://github.com/hidai25/eval-view/actions/workflows/ci.yml)
[![PyPI version](https://img.shields.io/pypi/v/evalview.svg)](https://pypi.org/project/evalview/)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![GitHub stars](https://img.shields.io/github/stars/hidai25/eval-view?style=social)](https://github.com/hidai25/eval-view/stargazers)

[![Python downloads](https://img.shields.io/pypi/dm/evalview.svg?label=python%20downloads)](https://pypi.org/project/evalview/)
[![Node.js downloads](https://img.shields.io/npm/dm/@evalview/node.svg?label=node.js%20downloads)](https://www.npmjs.com/package/@evalview/node)

<p align="center">
  <img src="assets/demo.gif" alt="EvalView Demo" width="700">
</p>

---

## The Problem

You changed a prompt, swapped models, or updated a tool. Now your agent:

- Calls different tools than before
- Returns different outputs for the same input
- Costs 3x more than yesterday
- Takes 5 seconds instead of 500ms

You don't find out until users complain.

## The Solution

**EvalView detects these regressions in CI — before you deploy.**

```bash
evalview golden save .evalview/results/xxx.json   # Save a working run as baseline
evalview run --diff                                # Future runs compare against it
```

---

## What EvalView Catches

| Regression Type | What It Means | Status |
|-----------------|---------------|--------|
| **REGRESSION** | Score dropped — agent got worse | Fix before deploy |
| **TOOLS_CHANGED** | Agent uses different tools now | Review before deploy |
| **OUTPUT_CHANGED** | Same tools, different response | Review before deploy |
| **PASSED** | Matches baseline | Ship it |

---

## Quick Start

```bash
pip install evalview

# Set your OpenAI API key (for LLM-as-judge evaluation)
export OPENAI_API_KEY='your-key-here'

# Scaffold a test for YOUR agent
evalview quickstart
```

**Free local evaluation:** Don't want to pay for API calls? Use Ollama:
```bash
evalview run --judge-provider ollama --judge-model llama3.2
```

[Full Getting Started Guide →](docs/GETTING_STARTED.md)

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

Tests run on every PR, block merges on regression. [Full CI/CD docs →](docs/CI_CD.md)

---

## Key Features

- **Golden traces** — Save baselines, detect regressions with `--diff`. [Learn more →](docs/GOLDEN_TRACES.md)
- **Tool categories** — Flexible matching by intent, not exact names. [Learn more →](docs/TOOL_CATEGORIES.md)
- **Statistical mode** — Handle flaky LLMs with pass@k metrics. [Learn more →](docs/STATISTICAL_MODE.md)
- **Chat mode** — AI assistant with `/run`, `/test`, `/compare`. [Learn more →](docs/CHAT_MODE.md)
- **Skills testing** — Validate Claude Code / OpenAI Codex skills. [Learn more →](docs/SKILLS_TESTING.md)
- **Test generation** — Generate 1000 tests from 1. [Learn more →](docs/TEST_GENERATION.md)
- **Suite types** — Separate capability tests from regression tests. [Learn more →](docs/SUITE_TYPES.md)
- **Behavior coverage** — Track tasks, tools, and paths covered. [Learn more →](docs/BEHAVIOR_COVERAGE.md)
- **LLM-as-judge** — Automated output quality assessment
- **Cost & latency tracking** — Automatic threshold enforcement
- **Parallel execution** — 8x faster by default
- **HTML reports** — Interactive Plotly charts

---

## Who Is It For?

- **LangGraph / CrewAI teams** shipping agents to production
- **Solo devs** tired of "it worked yesterday" conversations
- **Platform teams** who need CI gates before agent deploys

Already using LangSmith or Langfuse? Good. Use them to *see* what happened. Use EvalView to **block it from shipping.**

---

## Supported Frameworks

LangGraph • CrewAI • OpenAI Assistants • Anthropic Claude • AutoGen • Dify • Ollama • Custom APIs

[Framework compatibility details →](docs/FRAMEWORK_SUPPORT.md)

---

## Documentation

| Guide | Description |
|-------|-------------|
| [Getting Started](docs/GETTING_STARTED.md) | 5-minute quickstart |
| [CLI Reference](docs/CLI_REFERENCE.md) | All commands and options |
| [Golden Traces](docs/GOLDEN_TRACES.md) | Regression detection setup |
| [CI/CD Integration](docs/CI_CD.md) | GitHub Actions, GitLab, CircleCI |
| [Tool Categories](docs/TOOL_CATEGORIES.md) | Flexible tool matching |
| [Statistical Mode](docs/STATISTICAL_MODE.md) | Handling flaky tests |
| [Skills Testing](docs/SKILLS_TESTING.md) | Claude Code / Codex skills |
| [Evaluation Metrics](docs/EVALUATION_METRICS.md) | Scoring and weights |
| [FAQ](docs/FAQ.md) | Common questions |

### Guides

| Guide | Description |
|-------|-------------|
| [Testing LangGraph Agents](guides/pytest-for-ai-agents-langgraph-ci.md) | Automated testing with GitHub Actions |
| [Detecting Hallucinations](guides/detecting-llm-hallucinations-in-ci.md) | Catch made-up facts before users see them |

---

## Examples

- [LangGraph Integration](examples/langgraph/) — Test LangGraph agents
- [CrewAI Integration](examples/crewai/) — Test CrewAI agents
- [Anthropic Claude](examples/anthropic/) — Test Claude API
- [Dify Workflows](examples/dify/) — Test Dify AI workflows
- [Ollama (Local)](examples/ollama/) — Free local testing

**Using Node.js?** See [@evalview/node](sdks/node/) for drop-in middleware.

---

## Early Adopter Program

**First 10 teams get white-glove setup.** Free.

I'll personally configure your YAML tests + CI integration.

- [Claim a spot →](https://github.com/hidai25/eval-view/discussions)
- Email: hidai@evalview.com

---

## Roadmap

**Shipped:** Golden traces, tool categories, flakiness detection, skills testing, MCP server testing, HTML diff reports

**Coming Soon:** Multi-turn conversations, grounded hallucination checking, error compounding metrics

[Vote on features →](https://github.com/hidai25/eval-view/discussions)

---

## Contributing

Contributions welcome! See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

Apache 2.0 — [LICENSE](LICENSE)

## Support

- [GitHub Issues](https://github.com/hidai25/eval-view/issues)
- [Discussions](https://github.com/hidai25/eval-view/discussions)

---

**Ship AI agents with confidence.** [Star us →](https://github.com/hidai25/eval-view)

---

*EvalView is an independent open-source project, not affiliated with LangGraph, CrewAI, OpenAI, Anthropic, or any other third party.*
