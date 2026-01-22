# Chat Mode — AI-Powered CLI

**Don't remember commands? Just ask.**

<p align="center">
  <img src="../assets/chat-demo.gif" alt="EvalView Chat Demo" width="700">
</p>

```bash
evalview chat
```

Chat mode understands natural language AND knows all EvalView commands:

- *"Run my stock analysis test"* → Suggests `/run stock-test.yaml`
- *"Compare yesterday's run with today"* → Runs `/compare` for you
- *"What adapters do I have?"* → Lists available adapters
- *"Show me the trace from my last test"* → Displays execution trace

---

## Slash Commands

| Command | Description |
|---------|-------------|
| `/run <file>` | Run a test case against its adapter |
| `/test <adapter> <query>` | Quick ad-hoc test against an adapter |
| `/compare <old> <new>` | Compare two test runs, detect regressions |
| `/adapters` | List available adapters |
| `/trace` | View execution trace from last run |
| `/help` | Show all commands |

---

## Natural Language Execution

When the LLM suggests a command, it asks if you want to run it:

```
You: How do I test my LangGraph agent?

Claude: To test your LangGraph agent, you can run:
  `/test langgraph "What's the weather?"`

Would you like me to run this command? [y/n]
```

---

## Provider Options

**Free & local** — powered by Ollama. No API key needed.

```bash
evalview chat                     # Auto-detects Ollama
evalview chat --provider openai   # Or use cloud models
evalview chat --provider anthropic
```

---

## Quick Comparison

Use the `/compare` command for side-by-side regression detection:

```bash
evalview chat
> /compare .evalview/results/old.json .evalview/results/new.json
```

Output:

```
┌─────────────────┬───────────┬───────────┬────────┬──────────┐
│ Test            │ Old Score │ New Score │ Δ      │ Status   │
├─────────────────┼───────────┼───────────┼────────┼──────────┤
│ stock-analysis  │ 92.5      │ 94.0      │ +1.5   │ ✅ OK    │
│ customer-support│ 88.0      │ 71.0      │ -17.0  │ REGR  │
└─────────────────┴───────────┴───────────┴────────┴──────────┘
```

---

## Related Documentation

- [CLI Reference](CLI_REFERENCE.md)
- [Golden Traces](GOLDEN_TRACES.md)
