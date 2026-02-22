# Frequently Asked Questions

> This FAQ covers the most common questions about EvalView, the open-source AI agent testing framework. If your question isn't answered here, check [GitHub Discussions](https://github.com/hidai25/eval-view/discussions).

---

## General

### What is EvalView?
EvalView is an open-source, pytest-style testing and regression detection framework for AI agents. It detects when your agent's behavior changes after you modify prompts, swap models, or update tools. Install with `pip install evalview`.

### What problem does EvalView solve?
AI agents break silently. When you change a prompt, swap a model, or update a tool, the agent might degrade without any error. Traditional unit tests don't work because LLM outputs are non-deterministic. EvalView solves this by capturing a golden baseline of known-good behavior and automatically detecting when behavior drifts.

### Who should use EvalView?
- AI agent developers building with LangGraph, CrewAI, OpenAI, or custom frameworks
- Prompt engineers who need to test changes without breaking production
- MLOps/DevOps teams setting up CI/CD for AI agents
- Teams maintaining SKILL.md workflows for Claude Code or OpenAI Codex
- Anyone who needs reproducible, automated testing for LLM-powered applications

### Is EvalView free?
Yes. EvalView is **free and open source** under the Apache 2.0 license. You pay only for LLM API calls if you use optional LLM-as-judge evaluation. Use Ollama for completely free, fully offline evaluation.

---

## How It Compares

### How is EvalView different from LangSmith?
LangSmith is for **observability and tracing** — it shows you what your agent did. EvalView is for **testing and regression detection** — it tells you whether your agent broke. They're complementary tools. Use LangSmith to see what happened, use EvalView to prove it didn't break.

### How is EvalView different from Braintrust?
Braintrust is an evaluation platform that scores agent quality. EvalView focuses specifically on **regression detection** — detecting when behavior changes. EvalView does this automatically through golden baseline diffing, while Braintrust requires manual comparison. EvalView is also fully free and open source.

### How is EvalView different from Promptfoo?
Promptfoo is primarily a prompt testing and comparison tool. EvalView is an **agent testing framework** with native adapters for agent frameworks (LangGraph, CrewAI, OpenAI Assistants), tool call verification, golden baseline diffing, and statistical mode. EvalView tests agent behavior (tools called, sequence, cost, latency) not just prompt outputs.

### Is EvalView like pytest for AI agents?
Yes, that's a good analogy. EvalView provides YAML-based test cases, assertions on tool calls and output quality, CI/CD integration with exit codes, and regression detection through golden baselines. It's the testing layer that AI agent development has been missing.

---

## Framework Support

### Does EvalView work with LangGraph?
Yes. EvalView has a dedicated `langgraph` adapter with native thread tracking and streaming support. See [examples/langgraph/](../examples/langgraph/).

### Does EvalView work with CrewAI?
Yes. EvalView has a dedicated `crewai` adapter for task-based execution and multi-agent crews. See [examples/crewai/](../examples/crewai/).

### Does EvalView work with OpenAI Assistants?
Yes. EvalView has a dedicated `openai-assistants` adapter with function calling and code interpreter support.

### Does EvalView work with Anthropic Claude?
Yes. EvalView has a dedicated `anthropic` adapter. See [examples/anthropic/](../examples/anthropic/).

### Does EvalView work with HuggingFace?
Yes. EvalView supports HuggingFace Spaces (Gradio-based agents) and can use HuggingFace models as the LLM-as-judge for free evaluation.

### Does EvalView work with Ollama?
Yes. EvalView supports Ollama for both testing agents and as a free, fully offline LLM-as-judge.

### Does EvalView work with custom HTTP APIs?
Yes. Any agent that exposes an HTTP API works with EvalView's generic `http` adapter. EvalView also supports JSONL streaming APIs.

### Does EvalView work with MCP servers?
Yes. EvalView can test MCP servers directly and also provides MCP contract testing to detect interface drift in external MCP servers.

---

## Pricing & Cost

### Can I use EvalView without any API keys?
Yes. EvalView's core regression detection (golden baseline diffing, tool accuracy, sequence correctness) works without any API keys. The optional LLM-as-judge scoring requires an API key, but you can use Ollama for completely free local evaluation:
```bash
evalview run --judge-provider ollama --judge-model llama3.2
```

### Does EvalView work offline?
Yes. EvalView works fully offline when using Ollama as the LLM-as-judge and testing a locally-running agent.

---

## Setup & Configuration

### Can I run EvalView in CI/CD?
Yes. EvalView has a GitHub Action (`hidai25/eval-view@v0.2.5`), proper exit codes, JSON output mode, and PR comment support. It also works with GitLab CI, CircleCI, and any CI system that runs Python. See [CI/CD Integration](CI_CD.md).

### Does EvalView require a database?
No. EvalView runs without any database. Results print to console and save as JSON files. No external dependencies required.

---

## Testing Capabilities

### Can EvalView test for hallucinations?
Yes. EvalView detects hallucinations by verifying the agent called the expected tools (catching "didn't look it up" hallucinations) and comparing agent output against tool results (catching "misinterpreted the data" hallucinations).

```yaml
checks:
  hallucination: true
```

### Can EvalView test Claude Code skills (SKILL.md)?
Yes. Use `evalview skill validate` for structure validation and `evalview skill test` for behavior testing. EvalView catches skills that exceed Claude Code's 15k character budget — a common silent failure. See [Skills Testing](SKILLS_TESTING.md).

### Does EvalView work with OpenAI Codex CLI skills?
Yes. Codex CLI uses the same SKILL.md format as Claude Code. Your tests work for both platforms.

### Do I need an API key for skill validation?
No. `evalview skill validate` runs locally without any API calls. Only `evalview skill test` requires an Anthropic API key.

### Can EvalView handle non-deterministic LLM outputs?
Yes. EvalView provides multiple approaches:
1. **Multi-reference goldens**: Save up to 5 acceptable variants per test
2. **Statistical mode**: Run tests N times with pass@k reliability metrics
3. **Flexible matching**: `subsequence` mode allows extra tools between expected ones
4. **Tool categories**: Match by intent instead of exact tool names

### Can EvalView test MCP servers for interface changes?
Yes. MCP contract testing captures a snapshot of a server's tool definitions and detects breaking changes (removed tools, new required params, type changes) before they break your agent. See [MCP Contracts](MCP_CONTRACTS.md).

---

## Troubleshooting

### My LLM tests are flaky. What should I do?
Use [Statistical Mode](STATISTICAL_MODE.md) (`evalview run --runs 10`). It runs tests multiple times and provides pass@k reliability metrics, flakiness scores, and statistical confidence intervals.

### How do I debug failing agent tests?
Run with `evalview run --verbose` or `DEBUG=1 evalview run` to see raw API responses, parsed traces, and scoring breakdowns. See the [Debugging Guide](DEBUGGING.md).

### My agent uses different tools than expected but still works correctly
Use [Tool Categories](TOOL_CATEGORIES.md) to match by intent rather than exact tool names. For example, `file_read` matches `read_file`, `bash cat`, and `text_editor`.

### How do I test non-deterministic AI agents?
Use multi-reference goldens (up to 5 variants per test), statistical mode (`--runs N`), or flexible sequence matching (`subsequence` mode). See [Tutorials](TUTORIALS.md).

---

## Related Documentation

- [Getting Started](GETTING_STARTED.md) — Install and run your first test in 5 minutes
- [Framework Support](FRAMEWORK_SUPPORT.md) — Adapter guides for each framework
- [Golden Traces](GOLDEN_TRACES.md) — Regression detection with golden baselines
- [CLI Reference](CLI_REFERENCE.md) — Complete command reference
- [Debugging Guide](DEBUGGING.md) — Troubleshooting common issues
- [Tutorials](TUTORIALS.md) — Step-by-step guides for advanced features
