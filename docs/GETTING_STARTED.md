# Getting Started with EvalView

> **What is EvalView?** An open-source, pytest-style testing framework for AI agents that detects regressions when you change prompts, swap models, or update tools. Works with LangGraph, CrewAI, OpenAI Assistants, Anthropic Claude, HuggingFace, Ollama, and any HTTP API.

EvalView is lightweight, YAML-first, and gets you testing AI agents in under 5 minutes. No database required, no vendor lock-in, works offline with Ollama.

## Installation

```bash
pip install evalview
```

For HTML reports with interactive charts:
```bash
pip install evalview[reports]
# or
pip install evalview jinja2 plotly
```

For development (contributing):
```bash
# Clone the repo
git clone https://github.com/hidai25/eval-view.git
cd eval-view

# Option A: Using uv (faster)
uv sync --all-extras

# Option B: Using pip
pip install -e ".[dev]"
```

> **Note:** See [CONTRIBUTING.md](../CONTRIBUTING.md) for full development setup.

## Quick Setup (2 minutes)

### 1. Initialize your project

```bash
evalview init
```

This creates:
- `.evalview/config.yaml` - Configuration file
- `tests/test-cases/example.yaml` - Example test case

### 2. Configure your agent endpoint

Edit `.evalview/config.yaml`:

```yaml
adapter: http
endpoint: http://localhost:8000/api/chat
timeout: 30.0

# Optional: Custom scoring weights
scoring:
  weights:
    tool_accuracy: 0.3
    output_quality: 0.5
    sequence_correctness: 0.2

# Optional: Retry configuration
retry:
  max_retries: 2
  base_delay: 1.0
```

Or use auto-detection:
```bash
evalview connect
```

## Write Your First Test (2 minutes)

Create `tests/test-cases/my-test.yaml`:

```yaml
name: "Weather Query Test"
description: "Test that the agent can fetch weather information"

input:
  query: "What's the weather in San Francisco?"

expected:
  tools:
    - get_weather
  output:
    contains:
      - "San Francisco"
      - "temperature"
    not_contains:
      - "error"

thresholds:
  min_score: 80
  max_cost: 0.05
  max_latency: 5000
```

## Run Tests

### Basic run (parallel by default)
```bash
evalview run
```

### With options
```bash
# Run specific tests
evalview run --filter "weather*"

# Sequential mode
evalview run --sequential

# With retries for flaky tests
evalview run --max-retries 3

# Verbose output
evalview run --verbose

# Generate HTML report
evalview run --html-report report.html
```

### Watch mode (re-run on changes)
```bash
evalview run --watch
```

## View Results

### Console output
```bash
evalview report .evalview/results/latest.json
```

### HTML report
```bash
evalview report .evalview/results/latest.json --html report.html
```

Open `report.html` in your browser for interactive charts and detailed results.

## Test Case Reference

### Full test case structure

```yaml
name: "Test Name"
description: "Optional description"

input:
  query: "User message to send to agent"
  context:  # Optional context
    user_id: "123"
    session: "abc"

expected:
  tools:  # Expected tool calls
    - search
    - calculate
  tool_sequence:  # Expected order (optional)
    - search
    - calculate
  output:
    contains:
      - "expected phrase"
    not_contains:
      - "error"
      - "sorry"
  hallucination:
    check: true
    confidence_threshold: 0.8
  safety:
    check: true

thresholds:
  min_score: 80
  max_cost: 0.10
  max_latency: 10000

  # Optional: Override global weights for this test
  weights:
    tool_accuracy: 0.4
    output_quality: 0.4
    sequence_correctness: 0.2

# Optional: Use different adapter for this test
adapter: langgraph
endpoint: http://localhost:2024/runs
```

## Scoring

Default weights (configurable):
- **Tool Accuracy (30%)**: Did the agent use the expected tools?
- **Output Quality (50%)**: LLM-as-judge rating of the response
- **Sequence Correctness (20%)**: Were tools called in the right order?

A test **passes** if:
- Score >= `min_score`
- Cost <= `max_cost` (if specified)
- Latency <= `max_latency` (if specified)
- Hallucination check passes (if enabled)
- Safety check passes (if enabled)

## CI/CD Integration

Add to your GitHub Actions:

```yaml
- name: Run agent tests
  env:
    OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
  run: |
    evalview run --max-workers 4 --max-retries 2
```

See `.github/workflows/evalview.yml` for a complete example.

## Supported Adapters

- `http` - Generic REST API (default)
- `langgraph` - LangGraph / LangGraph Cloud
- `crewai` - CrewAI agents
- `openai-assistants` - OpenAI Assistants API
- `streaming` / `tapescope` - JSONL streaming APIs

## Configuring LLM-as-Judge

**CLI flags (recommended):**
```bash
evalview run --judge-model gpt-5 --judge-provider openai
evalview run --judge-model sonnet --judge-provider anthropic
evalview run --judge-model llama-70b --judge-provider huggingface  # Free!
```

**Model shortcuts:**
| Shortcut | Full Model |
|----------|------------|
| `gpt-5` | `gpt-5` |
| `sonnet` | `claude-sonnet-4-5-20250929` |
| `llama-70b` | `meta-llama/Llama-3.1-70B-Instruct` |

## Environment Variables

| Variable | Description |
|----------|-------------|
| `OPENAI_API_KEY` | OpenAI API key (for GPT judge) |
| `ANTHROPIC_API_KEY` | Anthropic API key (for Claude judge) |
| `HF_TOKEN` | HuggingFace token (for Llama judge - free!) |
| `EVAL_PROVIDER` | Force judge provider: `openai`, `anthropic`, `huggingface` |
| `EVAL_MODEL` | Override judge model (default: auto-detected)

## Next Steps

- [YAML Schema Reference](./YAML_SCHEMA.md)
- [Adapters & Agent Integration](./ADAPTERS.md)
- [Framework Support](./FRAMEWORK_SUPPORT.md)

## Regression Detection (Golden Baselines)

Once you have tests running, set up regression detection to catch behavioral drift:

```bash
# 1. Save current behavior as golden baseline
evalview snapshot

# 2. Make changes to your agent (prompt, model, tools)

# 3. Detect regressions automatically
evalview check
```

EvalView will report: PASSED, TOOLS_CHANGED, OUTPUT_CHANGED, or REGRESSION. See [Golden Traces](./GOLDEN_TRACES.md) for details.

## Need Help?

- [GitHub Issues](https://github.com/hidai25/eval-view/issues)
- [Discussions](https://github.com/hidai25/eval-view/discussions)
- [FAQ](./FAQ.md)
- [Troubleshooting](./TROUBLESHOOTING.md)
