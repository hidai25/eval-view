# CLI Reference — All EvalView Commands for AI Agent Testing

> Complete reference for all EvalView CLI commands, including `snapshot`, `check`, `run`, `chat`, `skill`, `mcp`, and more. EvalView is a command-line tool for testing and detecting regressions in AI agents.

Complete reference for all EvalView CLI commands.

## Installation

```bash
# Install (includes skills testing)
pip install evalview

# With HTML reports (Plotly charts)
pip install evalview[reports]

# With watch mode
pip install evalview[watch]

# All optional features
pip install evalview[all]
```

---

## `evalview quickstart`

The fastest way to try EvalView. Creates a demo agent, test case, and runs everything.

```bash
evalview quickstart
```

---

## `evalview run`

Run test cases.

```bash
evalview run [OPTIONS]

Options:
  --pattern TEXT         Test case file pattern (default: *.yaml)
  -t, --test TEXT        Run specific test(s) by name
  --diff                 Compare against golden traces, detect regressions
  --verbose              Enable verbose logging
  --sequential           Run tests one at a time (default: parallel)
  --max-workers N        Max parallel executions (default: 8)
  --max-retries N        Retry flaky tests N times (default: 0)
  --watch                Re-run tests on file changes
  --html-report PATH     Generate interactive HTML report
  --summary              Compact output with deltas vs last run + regression detection
  --coverage             Show behavior coverage: tasks, tools, paths, eval dimensions
  --judge-model TEXT     Model for LLM-as-judge (e.g., gpt-5, sonnet, llama-70b)
  --judge-provider TEXT  Provider for LLM-as-judge (openai, anthropic, huggingface, gemini, grok, ollama)
```

### Model Shortcuts

Use simple names, they auto-resolve:

| Shortcut | Full Model |
|----------|------------|
| `gpt-5` | `gpt-5` |
| `sonnet` | `claude-sonnet-4-5-20250929` |
| `opus` | `claude-opus-4-5-20251101` |
| `llama-70b` | `meta-llama/Llama-3.1-70B-Instruct` |
| `gemini` | `gemini-3.0` |
| `llama3.2` | `llama3.2` (Ollama) |

### Examples

```bash
# Basic run
evalview run

# Run specific tests
evalview run -t "stock-analysis" -t "customer-support"

# With regression detection
evalview run --diff

# Different judge models
evalview run --judge-model gpt-5 --judge-provider openai
evalview run --judge-model sonnet --judge-provider anthropic
evalview run --judge-model llama-70b --judge-provider huggingface  # Free!
evalview run --judge-model llama3.2 --judge-provider ollama  # Free & Local!
```

---

## `evalview expand`

Generate test variations from a seed test case.

```bash
evalview expand TEST_FILE [OPTIONS]

Options:
  --count N        Number of variations to generate (default: 10)
  --focus TEXT     Focus on specific scenarios (e.g., "edge cases, error scenarios")
```

### Examples

```bash
# Take 1 test, generate 100 variations
evalview expand tests/stock-test.yaml --count 100

# Focus on specific scenarios
evalview expand tests/stock-test.yaml --count 50 \
  --focus "different tickers, edge cases, error scenarios"
```

---

## `evalview record`

Record agent interactions and auto-generate test cases.

```bash
evalview record [OPTIONS]

Options:
  --interactive    Interactive recording mode
```

### Example

```bash
# Use your agent normally, auto-generate tests
evalview record --interactive
```

EvalView captures:
- Query → Tools called → Output
- Auto-generates test YAML
- Adds reasonable thresholds

---

## `evalview report`

Generate report from results.

```bash
evalview report RESULT_FILE [OPTIONS]

Options:
  --detailed       Include detailed metrics
  --html PATH      Generate HTML report
```

### Example

```bash
evalview report .evalview/results/20241118_004830.json --detailed --html report.html
```

---

## `evalview golden`

Manage golden traces for regression detection.

```bash
evalview golden <command> [OPTIONS]
```

### Commands

#### `evalview golden save`

Save a test result as the golden baseline.

```bash
evalview golden save RESULT_FILE [OPTIONS]

Options:
  --notes TEXT     Add notes to the golden trace
  --test TEXT      Save only a specific test from a multi-test result
```

Examples:
```bash
evalview golden save .evalview/results/xxx.json
evalview golden save result.json --notes "Post-refactor baseline"
evalview golden save result.json --test "specific-test-name"
```

#### `evalview golden list`

List all golden traces.

```bash
evalview golden list
```

#### `evalview golden show`

Show details of a golden trace.

```bash
evalview golden show TEST_NAME
```

#### `evalview golden delete`

Delete a golden trace.

```bash
evalview golden delete TEST_NAME [OPTIONS]

Options:
  --force    Skip confirmation prompt
```

---

## `evalview connect`

Auto-detect and connect to a running agent.

```bash
evalview connect
```

Supports 7+ frameworks with automatic detection:
LangGraph, CrewAI, OpenAI Assistants, Anthropic Claude, AutoGen, Dify, Custom APIs

---

## `evalview chat`

AI-powered CLI assistant.

```bash
evalview chat [OPTIONS]

Options:
  --provider TEXT    LLM provider (ollama, openai, anthropic)
```

See [Chat Mode](CHAT_MODE.md) for details.

---

## `evalview demo`

Run the interactive demo (no API key needed).

```bash
evalview demo
```

---

## `evalview skill` (Advanced)

Skills testing commands for Claude Code and OpenAI Codex SKILL.md workflows.

See [Advanced: Skills Testing](SKILLS_TESTING.md) for complete documentation.

```bash
evalview skill validate PATH      # Validate skill structure
evalview skill test TEST_FILE     # Run behavior tests
evalview skill doctor PATH        # Diagnose skill issues
```

Legacy `skill test` also supports explicit overrides:
```bash
evalview skill test tests.yaml --provider openai --base-url https://api.deepseek.com/v1 --model deepseek-chat
```

Legacy `evalview skill test` (system-prompt mode) supports Anthropic and OpenAI-compatible APIs via environment variables:
- `ANTHROPIC_API_KEY`
- `SKILL_TEST_PROVIDER` (`anthropic` or `openai`)
- `SKILL_TEST_API_KEY`, `SKILL_TEST_BASE_URL`
- Provider aliases: `OPENAI_*`, `DEEPSEEK_*`, `KIMI_*`, `MOONSHOT_*`

Note: non-OpenAI aliases (`DEEPSEEK_API_KEY`, `KIMI_API_KEY`, `MOONSHOT_API_KEY`) require a matching `*_BASE_URL` (or `SKILL_TEST_BASE_URL`).

---

## Exit Codes

| Scenario | Exit Code |
|----------|-----------|
| All tests pass, all PASSED | 0 |
| All tests pass, only warn-on statuses | 0 (with warnings) |
| Any test fails OR any fail-on status | 1 |
| Execution errors (network, timeout) | 2 |

---

## Environment Variables

| Variable | Description |
|----------|-------------|
| `OPENAI_API_KEY` | OpenAI API key for LLM-as-judge |
| `ANTHROPIC_API_KEY` | Anthropic API key |
| `HUGGINGFACE_API_KEY` | Hugging Face API key |

---

## Related Documentation

- [Getting Started](GETTING_STARTED.md)
- [Golden Traces](GOLDEN_TRACES.md)
- [Statistical Mode](STATISTICAL_MODE.md)
- [Chat Mode](CHAT_MODE.md)
