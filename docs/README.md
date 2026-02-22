# EvalView Documentation Index

> EvalView is an open-source, pytest-style testing and regression detection framework for AI agents. This page indexes all documentation.

## Quick Links

- **New to EvalView?** Start with [Getting Started](GETTING_STARTED.md)
- **Need a command reference?** See [CLI Reference](CLI_REFERENCE.md)
- **Have a question?** Check [FAQ](FAQ.md)
- **Something broken?** See [Troubleshooting](TROUBLESHOOTING.md)

---

## Getting Started

| Document | Description |
|----------|-------------|
| [Getting Started](GETTING_STARTED.md) | Install and run your first AI agent test in 5 minutes |
| [CLI Reference](CLI_REFERENCE.md) | Complete reference for all `evalview` commands |
| [FAQ](FAQ.md) | Frequently asked questions — comparisons, framework support, pricing |
| [YAML Test Case Schema](YAML_SCHEMA.md) | Complete schema reference for writing test cases |

## Core Concepts

| Document | Description |
|----------|-------------|
| [Golden Traces (Regression Detection)](GOLDEN_TRACES.md) | Automatic regression detection using golden baselines |
| [Evaluation Metrics](EVALUATION_METRICS.md) | 5-dimensional scoring: tool accuracy, output quality, sequence, cost, latency |
| [Statistical Mode (pass@k)](STATISTICAL_MODE.md) | Handle non-deterministic LLM outputs with statistical testing |
| [Tool Categories](TOOL_CATEGORIES.md) | Flexible tool matching by intent instead of exact name |
| [Suite Types](SUITE_TYPES.md) | Separate capability tests (expected failures) from regression tests (critical) |
| [Behavior Coverage](BEHAVIOR_COVERAGE.md) | Track which tasks, tools, and paths are tested |
| [Cost Tracking](COST_TRACKING.md) | Monitor token usage and spending per test |
| [Test Generation](TEST_GENERATION.md) | Auto-generate 100+ test variations from a single seed test |
| [Trace Specification](TRACE_SPEC.md) | Execution trace format used by all adapters |

## Framework Integration

| Document | Description |
|----------|-------------|
| [Framework Support](FRAMEWORK_SUPPORT.md) | Overview of all supported AI agent frameworks |
| [Adapters Guide](ADAPTERS.md) | How adapters work, built-in adapters, and building custom ones |
| [Backend Requirements](BACKEND_REQUIREMENTS.md) | What your AI agent backend needs to expose for testing |
| [Quick Start: LangGraph](QUICKSTART_LANGGRAPH.md) | Test LangGraph agents in 5 minutes |
| [Quick Start: HuggingFace](QUICKSTART_HUGGINGFACE.md) | Free, fully open-source testing with HuggingFace + Llama |
| [LangGraph Cloud](LANGGRAPH_CLOUD.md) | Testing LangGraph Cloud API agents |
| [LangGraph Example Setup](SETUP_LANGGRAPH_EXAMPLE.md) | Step-by-step LangGraph example walkthrough |
| [Database Setup](DATABASE_SETUP.md) | Test user configuration for agents that require user IDs |

## CI/CD & Integrations

| Document | Description |
|----------|-------------|
| [CI/CD Integration](CI_CD.md) | GitHub Actions, GitLab CI, CircleCI setup |
| [MCP Contract Testing](MCP_CONTRACTS.md) | Detect when external MCP servers change their interface |
| [Skills Testing](SKILLS_TESTING.md) | Validate and test SKILL.md for Claude Code, Codex, and OpenClaw |
| [Chat Mode](CHAT_MODE.md) | Interactive AI-powered CLI for agent testing |

## Learning & Troubleshooting

| Document | Description |
|----------|-------------|
| [Tutorials](TUTORIALS.md) | Step-by-step guides for advanced features |
| [Debugging Guide](DEBUGGING.md) | How to debug failing AI agent tests |
| [Troubleshooting](TROUBLESHOOTING.md) | Common issues and solutions |

## Guides (Long-form Articles)

| Guide | Description |
|-------|-------------|
| [Testing LangGraph Agents in CI](../guides/pytest-for-ai-agents-langgraph-ci.md) | Practical guide to CI testing for LangGraph agents |
| [Detecting LLM Hallucinations in CI](../guides/detecting-llm-hallucinations-in-ci.md) | How to catch AI agent hallucinations before production |
