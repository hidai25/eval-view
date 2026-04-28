# EvalView Docs

> EvalView is an open-source regression testing framework for AI agents. This page is the fastest way to find the right doc for your current task.

## Start Here

If you're new:

1. Read [Getting Started](GETTING_STARTED.md)
2. Skim [CLI Reference](CLI_REFERENCE.md)
3. Keep [FAQ](FAQ.md) and [Troubleshooting](TROUBLESHOOTING.md) nearby

## Choose Your Path

| I want to… | Read this first | Then |
|------------|-----------------|------|
| Get EvalView running quickly | [Getting Started](GETTING_STARTED.md) | [CLI Reference](CLI_REFERENCE.md) |
| Go from zero tests to a draft suite | [Test Generation](TEST_GENERATION.md) | [CI/CD Integration](CI_CD.md) |
| Compare EvalView to other tools | [Comparisons](COMPARISONS.md) | [AI Agent Testing in CI/CD](AI_AGENT_TESTING_CI.md) |
| Understand regression detection | [Golden Traces](GOLDEN_TRACES.md) | [Evaluation Metrics](EVALUATION_METRICS.md) |
| Test a specific framework | [Framework Support](FRAMEWORK_SUPPORT.md) | the matching quick start below |
| Set up CI/CD | [CI/CD Integration](CI_CD.md) | [Golden Traces](GOLDEN_TRACES.md) |
| Validate agent skills / `SKILL.md` | [Skills Testing](SKILLS_TESTING.md) | [Chat Mode](CHAT_MODE.md) |
| Debug a failure | [Debugging Guide](DEBUGGING.md) | [Troubleshooting](TROUBLESHOOTING.md) |

## Essentials

| Document | Description |
|----------|-------------|
| [Getting Started](GETTING_STARTED.md) | First run in about 5 minutes |
| [CLI Reference](CLI_REFERENCE.md) | Full command reference for `evalview` |
| [FAQ](FAQ.md) | Positioning, pricing, framework support, common questions |
| [YAML Test Case Schema](YAML_SCHEMA.md) | Complete schema for authoring test cases |
| [Comparisons](COMPARISONS.md) | EvalView vs LangSmith, Langfuse, Braintrust, and DeepEval |
| [Operating Model](OPERATING_MODEL.md) | How to run EvalView with frontier-lab rigor and startup-team practicality |
| [Internal Dogfooding](INTERNAL_DOGFOODING.md) | The lightweight internal ship gate for EvalView itself |

## Core Concepts

| Document | Description |
|----------|-------------|
| [Golden Traces](GOLDEN_TRACES.md) | Snapshot behavior and detect regressions |
| [Evaluation Metrics](EVALUATION_METRICS.md) | How tool, output, sequence, cost, and latency scoring work |
| [Statistical Mode](STATISTICAL_MODE.md) | Pass-rate based evaluation for non-deterministic agents |
| [Tool Categories](TOOL_CATEGORIES.md) | Match tools by intent instead of exact name |
| [Suite Types](SUITE_TYPES.md) | Separate capability tests from regression tests |
| [Behavior Coverage](BEHAVIOR_COVERAGE.md) | Track gaps in the behaviors you test |
| [Cost Tracking](COST_TRACKING.md) | Understand token and dollar usage |
| [Test Generation](TEST_GENERATION.md) | Generate a draft suite from an agent or logs |
| [Trace Specification](TRACE_SPEC.md) | Execution trace format used across adapters |
| [Decision Rationale](RATIONALE.md) | Structured "why" logging for agent decisions |

## Frameworks

| Document | Description |
|----------|-------------|
| [Framework Support](FRAMEWORK_SUPPORT.md) | Overview of supported agent stacks |
| [Adapters Guide](ADAPTERS.md) | Built-in adapters and custom adapter design |
| [Backend Requirements](BACKEND_REQUIREMENTS.md) | What your backend must expose for testing |
| [Quick Start: LangGraph](QUICKSTART_LANGGRAPH.md) | LangGraph setup |
| [Quick Start: HuggingFace](QUICKSTART_HUGGINGFACE.md) | Open-source local/hosted setup |
| [LangGraph Cloud](LANGGRAPH_CLOUD.md) | Testing LangGraph Cloud APIs |
| [LangGraph Example Setup](SETUP_LANGGRAPH_EXAMPLE.md) | End-to-end example walkthrough |
| [Database Setup](DATABASE_SETUP.md) | Test-user and stateful backend setup |

## Specialized Commands

| Document | Description |
|----------|-------------|
| [`evalview simulate`](SIMULATE.md) | Hermetic, mock-driven testing for CI |
| [`evalview model-check`](MODEL_CHECK.md) | Detect closed-model drift on a canary suite |

## CI, Integrations, and Operations

| Document | Description |
|----------|-------------|
| [CI/CD Integration](CI_CD.md) | GitHub Actions, GitLab CI, CircleCI |
| [AI Agent Testing in CI/CD](AI_AGENT_TESTING_CI.md) | Search-intent guide for regression testing agents in CI |
| [MCP Contract Testing](MCP_CONTRACTS.md) | Detect external MCP server interface drift |
| [Skills Testing](SKILLS_TESTING.md) | Test `SKILL.md` behavior with real agents |
| [Chat Mode](CHAT_MODE.md) | Interactive CLI guidance and exploration |
| [EvalView Cloud](CLOUD.md) | Optional team dashboard — what it stores, what it never runs |

## Contributing

| Document | Description |
|----------|-------------|
| [Repository Guidelines](AGENTS.md) | Project structure, conventions, and contributor guide |
| [Agent Recipes](agent-recipes/README.md) | Step-by-step recipes for common extension tasks |

## Website Guides

Use the website when you want the cleaner comparison and search-intent pages. Use the repo docs when you want command details and implementation guidance.

| Page | Best for |
|------|----------|
| [AI agent testing in CI/CD](https://www.evalview.com/ai-agent-testing-ci-cd) | high-level workflow and positioning |
| [AI agent regression testing](https://www.evalview.com/ai-agent-regression-testing) | explaining the core problem to a team |
| [MCP server testing](https://www.evalview.com/mcp-server-testing) | testing MCP servers and tool contracts |
| [LangGraph testing](https://www.evalview.com/langgraph-testing) | LangGraph-specific adoption |
| [Tool-calling agent testing](https://www.evalview.com/tool-calling-agent-testing) | tool-path and safety-focused testing |
| [EvalView vs LangSmith](https://www.evalview.com/vs/langsmith) | observability vs regression testing |
| [EvalView vs Langfuse](https://www.evalview.com/vs/langfuse) | tracing vs regression gating |
| [EvalView vs Braintrust](https://www.evalview.com/vs/braintrust) | broader evals vs baseline regression testing |
| [EvalView vs DeepEval](https://www.evalview.com/vs/deepeval) | metric-first evals vs trajectory diffs |

## Debugging and Learning

| Document | Description |
|----------|-------------|
| [Tutorials](TUTORIALS.md) | Longer step-by-step workflows |
| [Debugging Guide](DEBUGGING.md) | Triage failing tests and unexpected diffs |
| [Troubleshooting](TROUBLESHOOTING.md) | Common errors and fixes |

## Guides

| Guide | Description |
|-------|-------------|
| [Testing LangGraph Agents in CI](../guides/pytest-for-ai-agents-langgraph-ci.md) | Practical CI setup for LangGraph agents |
| [Detecting LLM Hallucinations in CI](../guides/detecting-llm-hallucinations-in-ci.md) | Catch hallucination regressions before production |
