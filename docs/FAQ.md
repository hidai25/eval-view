# Frequently Asked Questions

## Framework Support

### Does EvalView work with LangChain / LangGraph?
Yes. Use the `langgraph` adapter. See [examples/langgraph/](../examples/langgraph/).

### Does EvalView work with CrewAI?
Yes. Use the `crewai` adapter. See [examples/crewai/](../examples/crewai/).

### Does EvalView work with OpenAI Assistants?
Yes. Use the `openai-assistants` adapter.

### Does EvalView work with Anthropic Claude?
Yes. Use the `anthropic` adapter. See [examples/anthropic/](../examples/anthropic/).

---

## Pricing & Cost

### How much does EvalView cost?
EvalView is **free and open source**. You pay only for LLM API calls (for LLM-as-judge evaluation). Use Ollama for free local evaluation.

### Can I use it without an API key?
Yes. Use Ollama for free local LLM-as-judge:
```bash
evalview run --judge-provider ollama --judge-model llama3.2
```

---

## Setup & Configuration

### Can I run EvalView in CI/CD?
Yes. EvalView has a GitHub Action and proper exit codes. See [CI/CD Integration](CI_CD.md).

### Does EvalView require a database?
No. EvalView runs without any database by default. Results print to console and save as JSON.

If you later want history, dashboards, or analytics, you can plug in a database.

---

## Comparisons

### How is EvalView different from LangSmith?
LangSmith is for **tracing/observability**. EvalView is for **testing**.

Use both:
- **LangSmith** to see what happened
- **EvalView** to block bad behavior before prod

---

## Testing Capabilities

### Can I test for hallucinations?
Yes. EvalView has built-in hallucination detection that compares agent output against tool results.

```yaml
checks:
  hallucination: true
```

### Can I test Claude Code skills?
Yes. Use `evalview skill validate` for structure checks and `evalview skill test` for behavior tests. See [Skills Testing](SKILLS_TESTING.md).

### Does EvalView work with OpenAI Codex CLI skills?
Yes. Codex CLI uses the same SKILL.md format as Claude Code. Your tests work for both.

### Do I need an API key for skill validation?
No. `evalview skill validate` runs locally without any API calls. Only `evalview skill test` requires an Anthropic API key.

---

## Troubleshooting

### My tests are flaky. What should I do?
Use [Statistical Mode](STATISTICAL_MODE.md). It runs tests multiple times and uses statistical thresholds for pass/fail decisions.

### How do I debug failing tests?
See the [Debugging Guide](DEBUGGING.md) for troubleshooting common issues.

### My agent uses different tools than expected but still works correctly
Use [Tool Categories](TOOL_CATEGORIES.md) to match by intent rather than exact tool names.

---

## Related Documentation

- [Getting Started](GETTING_STARTED.md)
- [Framework Support](FRAMEWORK_SUPPORT.md)
- [Debugging Guide](DEBUGGING.md)
- [CLI Reference](CLI_REFERENCE.md)
