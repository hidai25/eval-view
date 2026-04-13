# CLI Reference — All EvalView Commands for AI Agent Testing

> Complete reference for all EvalView CLI commands, including `snapshot`, `check`, `run`, `chat`, `skill`, `mcp`, and more. EvalView is a command-line tool for testing and detecting regressions in AI agents.

Complete reference for all EvalView CLI commands.

## Installation

```bash
# Install (includes skills testing)
pip install evalview

# With interactive charts on top of the built-in HTML report
pip install evalview[reports]

# With watch mode
pip install evalview[watch]

# All optional features
pip install evalview[all]
```

---

## `evalview quickstart` (Deprecated)

Legacy compatibility command for the old demo bootstrap flow. Prefer:

```bash
evalview demo   # See EvalView catch a regression immediately
evalview init   # Connect your real agent and create a starter suite
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
  --judge-cache/--no-judge-cache  Cache LLM judge responses (on by default)
  --no-judge             Skip LLM-as-judge, use deterministic scoring only (free)
  --budget FLOAT         Maximum total budget in dollars. Warns if exceeded.
  --dry-run              Preview test plan and estimate cost without executing
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

# Cost control
evalview run --dry-run                     # Preview plan, no API calls
evalview run --budget 1.00                 # Cap spend at $1
evalview run --no-judge                    # Free — deterministic scoring only
evalview run --no-judge-cache              # Disable judge response caching
```

---

## `evalview snapshot`

Run tests and save passing results as baseline.

```bash
evalview snapshot [TEST_PATH] [OPTIONS]

Options:
  -t, --test TEXT     Snapshot only this specific test
  -n, --notes TEXT    Notes about this snapshot
  --variant TEXT      Save as named variant (max 5 per test)
  --approve-generated Approve generated draft tests before snapshotting
```

### Examples

```bash
evalview snapshot                           # Snapshot all passing tests
evalview snapshot --test "my-test"          # Snapshot one test
evalview snapshot --variant v2             # Save alternate acceptable behavior
evalview snapshot tests/generated --approve-generated
```

---

## `evalview check`

Check current behavior against snapshot baseline.

```bash
evalview check [TEST_PATH] [OPTIONS]

Options:
  -t, --test TEXT     Check only this specific test
  --json              Output JSON for CI
  --fail-on TEXT      Comma-separated statuses to fail on (default: REGRESSION)
  --strict            Fail on any change
  --report PATH       Generate HTML report
  --semantic-diff/--no-semantic-diff  Toggle embedding-based similarity
  --budget FLOAT      Maximum total budget in dollars
  --dry-run           Preview check plan without executing
```

### Examples

```bash
evalview check                              # Check all tests
evalview check --test "my-test"             # Check one test
evalview check --json --fail-on REGRESSION  # CI mode
evalview check --dry-run                    # Preview plan, no API calls
evalview check --budget 0.50               # Cap spend at $0.50
```

### Model / Runtime Detection

`evalview check` runs a layered detector during baseline comparison:

- **Declared model change** when the adapter reports a different `model_id`
- **Runtime fingerprint change** when observed model labels in the trace differ from baseline
- **Coordinated drift detection** when multiple tests shift together in the same run

The terminal output and HTML report surface the classification (`declared` or `suspected`), confidence, runtime fingerprint diff, and retry evidence from `evalview check --heal` when available.

---

## `evalview monitor`

Continuously run checks against a live agent and alert on regressions. Designed for production monitoring: `evalview monitor` runs `evalview check` in a loop, fires Slack/Discord alerts only for failures confirmed across two consecutive cycles, and records the full signal-to-noise trail in `.evalview/noise.jsonl` so `evalview slack-digest` can report a verifiable false-positive rate.

```bash
evalview monitor [TEST_PATH] [OPTIONS]

Options:
  -i, --interval N         Seconds between checks (default: 300, minimum: 10)
  --slack-webhook URL      Slack webhook URL for alerts
  --discord-webhook URL    Discord webhook URL for alerts
  --fail-on STATUSES       Comma-separated statuses that trigger alerts
                           (default: REGRESSION)
  --timeout FLOAT          Timeout per test in seconds (default: 30)
  -t, --test TEXT          Monitor only this specific test
  --history PATH           Append each cycle's results to a JSONL file
  --alert-cost-spike X     Alert when cost exceeds baseline by this
                           multiplier (e.g. 2.0)
  --alert-latency-spike X  Alert when latency exceeds baseline by this
                           multiplier (e.g. 3.0)
  --dashboard              Live-updating terminal dashboard instead of
                           scrolling logs
```

### Confirmation gate

Every alert is a promise. By default the monitor suppresses `n=1` failures: a test has to fail in two consecutive cycles before it pages a human, and a single blip self-resolves silently. Tests that must alert on the first failure (auth, payments, PII, refund paths) can opt out of the gate by setting `gate: strict` in their YAML; strict tests bypass confirmation and re-alert every cycle until they pass.

Self-resolved (suppressed) failures are never dropped: they are appended to `.evalview/noise.jsonl` with their test names, so `evalview slack-digest` can render a Noise section with an exact `N suppressed / M fired = Z% noise` false-positive rate.

### Prerequisites

`evalview monitor` requires existing baselines. Run `evalview snapshot` first to create them, otherwise the monitor exits immediately with the error `No baselines found. Run evalview snapshot first.`

### Configuration (`.evalview/config.yaml`)

```yaml
monitor:
  interval: 300
  slack_webhook: https://hooks.slack.com/services/...
  discord_webhook: https://discord.com/api/webhooks/...
  fail_on: [REGRESSION]
  cost_threshold: 2.0
  latency_threshold: 3.0
```

Webhook URLs can also be provided via environment variables as a fallback:

| Variable | Description |
|----------|-------------|
| `EVALVIEW_SLACK_WEBHOOK` | Slack webhook URL (fallback) |
| `EVALVIEW_DISCORD_WEBHOOK` | Discord webhook URL (fallback) |

### Examples

```bash
evalview monitor                                 # Check every 5 min
evalview monitor --interval 60                   # Check every minute
evalview monitor --dashboard                     # Live terminal dashboard
evalview monitor --slack-webhook https://...     # Alert to Slack
evalview monitor --discord-webhook https://...   # Alert to Discord
evalview monitor --test "weather-lookup"         # Monitor one test
evalview monitor --fail-on REGRESSION,TOOLS_CHANGED
evalview monitor --history monitor_log.jsonl     # Persist cycle history
evalview monitor --alert-cost-spike 2.0          # Alert if cost doubles
evalview monitor --alert-latency-spike 3.0       # Alert if latency triples
```

### Related files

| Path | Written by | Purpose |
|------|-----------|---------|
| `.evalview/noise.jsonl` | monitor loop | One line per cycle with `alerts_fired`, `suppressed`, and the names of self-resolved tests. Consumed by `evalview slack-digest` to report a false-positive rate. |
| `<--history PATH>` | `--history` | Optional per-cycle diagnostic log (total tests, pass/fail counts, cost, failing test names). Separate from the noise log. |

---

## `evalview model-check`

Detect silent drift in a closed-weight model (Claude, GPT, ...) against
a fixed canary suite. No agent required. No LLM judge. See
[MODEL_CHECK.md](MODEL_CHECK.md) for the full rationale, classification
table, and per-provider signal strength.

**v1 supports Anthropic.** OpenAI, Mistral, Cohere, and local providers
land in v1.1.

```bash
evalview model-check [OPTIONS]

Options:
  --model TEXT          Model id (required, e.g. claude-opus-4-5-20251101)
  --provider TEXT       Provider override (v1: anthropic). Auto-detected
                        from --model when omitted.
  --suite PATH          Custom canary YAML (default: bundled public canary)
  --runs INTEGER        Runs per prompt for variance (default: 3)
  --budget FLOAT        Hard cap on USD spend; refuses to run if estimate
                        exceeds (default: 2.00)
  --dry-run             Print a cost estimate and exit without API calls
  --pin                 Pin this run as the new reference for the model
  --reset-reference     Delete existing reference before saving this run
  --out PATH            Write full JSON snapshot+comparison to a file
  --no-save             Do not persist the snapshot to disk
  --json                Emit machine-readable JSON instead of human output
```

### Examples

```bash
# First run: saves baseline automatically (auto-pins as reference)
evalview model-check --model claude-opus-4-5-20251101

# Preview cost only — no API calls made
evalview model-check --model claude-opus-4-5-20251101 --dry-run

# Run with a larger budget cap
evalview model-check --model claude-opus-4-5-20251101 --budget 5.00

# Use your own custom canary suite
evalview model-check --model claude-opus-4-5-20251101 --suite ./my-canary.yaml

# CI wrapper: exit 0 = no drift, 1 = drift, 2 = error
evalview model-check --model claude-opus-4-5-20251101 --json > result.json
```

### Exit codes

- `0` — no drift detected (or first-time baseline saved)
- `1` — drift detected (any `MODEL` classification on any comparison)
- `2` — usage error (bad args, missing API key, suite error, cost over budget)

## `evalview generate`

Generate a draft regression suite from a live agent or existing traffic logs.

```bash
evalview generate [OPTIONS]

Options:
  --agent URL                  Agent endpoint URL
  --adapter TEXT               Adapter type (default: config or http)
  --budget N                   Maximum probe runs / imported entries
  --out DIR                    Output directory (default: tests/generated)
  --seed FILE                  Newline-delimited seed prompts
  --from-log PATH              Generate from a log file instead of live probing
  --log-format FORMAT          auto|jsonl|openai|evalview
  --include-tools TEXT         Comma-separated tool names to focus on
  --exclude-tools TEXT         Comma-separated tool names to avoid
  --allow-live-side-effects    Allow side-effecting prompts
  --timeout FLOAT              Probe timeout in seconds
  --dry-run                    Preview without writing files
```

### Examples

```bash
evalview generate --agent http://localhost:8000
evalview generate --from-log traffic.jsonl
evalview generate --agent http://localhost:8000 --include-tools search,calendar
evalview generate --dry-run
```

Generated suites are draft-only until approved:

```bash
evalview snapshot tests/generated --approve-generated
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

`skill test` supports explicit provider overrides:
```bash
evalview skill test tests.yaml --provider openai --base-url https://api.deepseek.com/v1 --model deepseek-chat
```

`evalview skill test` (system-prompt mode) supports Anthropic and OpenAI-compatible APIs via environment variables:
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
- [Monitor Config Options](#evalview-monitor)
