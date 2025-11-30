# EvalView

[![CI](https://github.com/hidai25/EvalView/actions/workflows/ci.yml/badge.svg)](https://github.com/hidai25/EvalView/actions/workflows/ci.yml)
[![Python Version](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![GitHub stars](https://img.shields.io/github/stars/hidai25/EvalView?style=social)](https://github.com/hidai25/EvalView/stargazers)
[![GitHub forks](https://img.shields.io/github/forks/hidai25/EvalView?style=social)](https://github.com/hidai25/EvalView/network/members)

[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Checked with mypy](https://www.mypy-lang.org/static/mypy_badge.svg)](https://mypy-lang.org/)

<!-- Uncomment when published to PyPI:
[![PyPI version](https://img.shields.io/pypi/v/evalview.svg)](https://pypi.org/project/evalview/)
[![PyPI downloads](https://img.shields.io/pypi/dm/evalview.svg)](https://pypi.org/project/evalview/)
-->

**pytest for AI agents.** Catch hallucinations, regressions, and cost spikes before they reach production.

> 🔓 **100% Open Source** · Runs locally · Works with any agent framework · No SaaS required

---

## 🚀 Try it in 2 minutes

```bash
# Install
pip install evalview

# Set your OpenAI API key (for LLM-as-judge evaluation)
export OPENAI_API_KEY='your-key-here'

# Run the quickstart - creates demo agent, test case, runs everything!
evalview quickstart
```

**That's it!** You'll see a working test pass with tool accuracy, output quality, cost, and latency metrics.

<details>
<summary>📺 See example output</summary>

```
━━━ EvalView Quickstart ━━━

Step 1/4: Creating demo agent...
✅ Demo agent created

Step 2/4: Creating test case...
✅ Test case created

Step 3/4: Creating config...
✅ Config created

Step 4/4: Starting demo agent and running test...
✅ Demo agent running

Running test...

Test Case: Quickstart Test
Score: 95.0/100
Status: ✅ PASSED

Tool Accuracy: 100.0%
  ✅ Correct: calculator

Output Quality: 90.0/100

Performance:
  Cost: $0.0010
  Latency: 27ms

🎉 Quickstart complete!
```
</details>

---

## Why EvalView?

- **🔓 Fully Open Source** – No SaaS, no vendor lock-in, runs entirely on your machine
- **🔌 Framework Agnostic** – Works with LangGraph, CrewAI, OpenAI, or any HTTP API
- **🚀 Production Ready** – Parallel execution, CI/CD integration, security built-in
- **🧩 Extensible** – Plug in custom adapters and evaluators

---

## What it does

- **Write test cases in YAML** – Define expected tools, outputs, and thresholds
- **Automated evaluation** – Tool accuracy, output quality (LLM-as-judge), cost, and latency
- **CI/CD ready** – JSON reports and exit codes for automated testing

## Example test case

```yaml
# tests/test-cases/stock-analysis.yaml
name: "Stock Analysis Test"
input:
  query: "Analyze Apple stock performance"

expected:
  tools: [fetch_stock_data, analyze_metrics]
  output:
    contains: ["revenue", "earnings"]

thresholds:
  min_score: 80
  max_cost: 0.50
  max_latency: 5000
```

```bash
$ evalview run

✅ Stock Analysis Test - PASSED (score: 92.5)
   Cost: $0.0234 | Latency: 3.4s
```

---

## Connect to your agent

Already have an agent running? Use `evalview connect` to auto-detect it:

```bash
# Start your agent (LangGraph, CrewAI, whatever)
langgraph dev

# Auto-detect and connect
evalview connect  # Scans ports, detects framework, configures everything

# Run tests
evalview run
```

Supports 7+ frameworks with automatic detection:
✅ LangGraph • ✅ LangServe • ✅ CrewAI • ✅ OpenAI Assistants • ✅ TapeScope • ✅ Custom APIs

---

## Why this exists

**Agents hallucinate, regress, and silently break.**

Unlike deterministic code, AI agents can:
- Start using the wrong tools after a prompt change
- Generate plausible-but-wrong answers
- Suddenly cost 10x more due to a config change
- Get slower as context windows grow

Traditional testing doesn't catch this. EvalView lets you write repeatable tests and run them like CI – so you know *before* your users do.

---

## Features

- ✅ **YAML-based test cases** - Write readable, maintainable test definitions
- ⚡ **Parallel execution** - Run tests concurrently (8x faster by default)
- 📊 **Multiple evaluation metrics** - Tool accuracy, sequence correctness, output quality, cost, and latency
- 🤖 **LLM-as-judge** - Automated output quality assessment using GPT-4
- 💰 **Cost tracking** - Automatic cost calculation based on token usage with GPT-5 family pricing
- 🔌 **Universal adapters** - Works with any HTTP or streaming API
- 🎨 **Rich console output** - Beautiful, informative test results
- 📁 **JSON & HTML reports** - Interactive HTML reports with Plotly charts
- 🔄 **Retry logic** - Automatic retries with exponential backoff for flaky tests
- 👀 **Watch mode** - Re-run tests automatically on file changes
- ⚖️ **Configurable weights** - Customize scoring weights globally or per-test
- 🐛 **Verbose debugging** - Detailed logging to troubleshoot issues
- 🗄️ **Database-agnostic** - Works with PostgreSQL, MongoDB, MySQL, Firebase, and more

---

## ☁️ EvalView Cloud (Coming Soon)

We're building a hosted version:

- 📊 **Dashboard** - Visual test history, trends, and pass/fail rates
- 👥 **Teams** - Share results and collaborate on fixes
- 🔔 **Alerts** - Slack/Discord notifications on failures
- 📈 **Regression detection** - Automatic alerts when performance degrades
- ⚡ **Parallel runs** - Run hundreds of tests in seconds

👉 **[Join the waitlist](https://form.typeform.com/to/EQO2uqSa)** - be first to get access

---

## Quickstart

**Fastest way (recommended):**
```bash
pip install evalview
export OPENAI_API_KEY='your-key-here'
evalview quickstart
```

This creates a demo agent, test case, starts everything, and runs your first test in under 2 minutes.

**For existing agents:**
```bash
pip install evalview
evalview init --interactive  # Configure for your agent
evalview run
```

---

## Installation

**Stable Release (Recommended):**
```bash
pip install evalview
```

**Development Install:**
```bash
# Clone the repository
git clone https://github.com/hidai25/EvalView.git
cd EvalView

# Create virtual environment (optional)
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode
pip install -e .
```

## Detailed Setup

### 1. Initialize Project

```bash
evalview init --interactive
```

The interactive setup will guide you through:
1. **API Configuration** - Choose REST or Streaming API
2. **Endpoint URL** - Your agent's API endpoint
3. **Model Selection** - Which GPT model your agent uses (gpt-5, gpt-5-mini, gpt-5-nano, etc.)
4. **Pricing Configuration** - Confirm standard pricing or set custom rates

This creates:
- `.evalview/config.yaml` - Configuration for your agent endpoint and model pricing
- `tests/test-cases/` - Directory for test cases
- `tests/test-cases/example.yaml` - Example test case

### 2. Configure Your Agent

Edit `.evalview/config.yaml`:

**For standard REST APIs:**
```yaml
adapter: http
endpoint: http://localhost:3000/api/agent
timeout: 30.0
headers:
  Authorization: Bearer your-api-key
```

**For streaming JSONL APIs:**
```yaml
adapter: streaming  # Works with any JSONL streaming API
endpoint: http://localhost:3000/api/chat
timeout: 60.0
headers:
  Content-Type: application/json
```

See [docs/ADAPTERS.md](docs/ADAPTERS.md) for custom adapter development.

### 3. Write Test Cases

Create `tests/test-cases/stock-analysis.yaml`:

```yaml
name: "Stock Analysis Test"
description: "Test agent's ability to analyze stock data"

input:
  query: "Analyze Apple (AAPL) stock performance"
  context:
    symbol: "AAPL"

expected:
  tools:
    - fetch_stock_data
    - analyze_metrics
  tool_sequence:
    - fetch_stock_data
    - analyze_metrics
  output:
    contains:
      - "revenue"
      - "earnings"
      - "price"
    not_contains:
      - "error"

thresholds:
  min_score: 80
  max_cost: 0.50
  max_latency: 5000
```

### 4. Run Tests

```bash
# Set OpenAI API key for LLM-as-judge
export OPENAI_API_KEY=your-openai-api-key

# Run all tests
evalview run

# Run specific pattern
evalview run --pattern "stock-*.yaml"
```

### 5. View Results

Results are displayed in the console and saved to `.evalview/results/`:

```
📊 Evaluation Summary
┏━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━┳━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━━━┓
┃ Test Case                ┃ Score ┃ Status  ┃ Cost    ┃ Latency  ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━╇━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━━━┩
│ Stock Analysis Test      │ 92.5  │ ✅ PASSED│ $0.0234 │ 3456ms   │
└─────────────────────────┴───────┴─────────┴─────────┴──────────┘

✅ Passed: 1
❌ Failed: 0
📈 Success Rate: 100.0%
```

Generate detailed reports:

```bash
evalview report .evalview/results/20241118_004830.json --detailed
```

## Test Case Format

### Required Fields

```yaml
name: string                 # Test case name
input:
  query: string              # Query to send to agent
  context: dict              # Optional context data
expected:                    # Expected behavior
  tools: list[string]        # Expected tools (any order)
  tool_sequence: list[string]  # Exact tool order
  output:
    contains: list[string]   # Must contain these strings
    not_contains: list[string]  # Must NOT contain these
thresholds:
  min_score: float (0-100)   # Minimum passing score
  max_cost: float            # Maximum cost in dollars
  max_latency: float         # Maximum latency in ms
```

### Example: Multi-Tool Agent Test

```yaml
name: "Research and Report Generation"
description: "Test complex multi-step research workflow"

input:
  query: "Research the latest AI trends and create a summary report"

expected:
  tools:
    - web_search
    - extract_content
    - generate_report
  tool_sequence:
    - web_search
    - extract_content
    - extract_content
    - generate_report
  output:
    contains:
      - "machine learning"
      - "transformers"
      - "sources:"
    not_contains:
      - "error"
      - "failed to fetch"

thresholds:
  min_score: 85
  max_cost: 2.00
  max_latency: 15000
```

## Agent API Format

Your agent endpoint should return JSON with this structure:

```json
{
  "session_id": "session-123",
  "output": "Final agent response",
  "steps": [
    {
      "id": "step-1",
      "name": "Fetch data",
      "tool": "fetch_stock_data",
      "parameters": {"symbol": "AAPL"},
      "output": {"price": 150.25, "volume": 1000000},
      "success": true,
      "latency": 234,
      "cost": 0.001,
      "tokens": 150
    }
  ],
  "cost": 0.025,
  "tokens": 1250
}
```

## Evaluation Metrics

### 1. Tool Accuracy (30% weight)
- Checks if expected tools were called
- Reports missing and unexpected tools
- Score: `correct_tools / expected_tools`

### 2. Output Quality (50% weight)
- String contains/not-contains checks
- LLM-as-judge evaluation (GPT-4o-mini)
- Scored 0-100 with rationale

### 3. Sequence Correctness (20% weight)
- Validates exact tool call order
- Binary pass/fail
- Reports violations

### 4. Cost Threshold
- Automatic cost calculation based on token usage
- Supports GPT-5, GPT-5-mini, GPT-5-nano, and custom pricing
- Must stay under `max_cost`
- Provides detailed breakdown by step (input/output/cached tokens)
- Fails test if exceeded

### 5. Latency Threshold
- Must complete under `max_latency`
- Provides breakdown by step
- Fails test if exceeded

### Configurable Scoring Weights

Default weights can be customized globally in `config.yaml`:

```yaml
scoring:
  weights:
    tool_accuracy: 0.35        # 35%
    output_quality: 0.45       # 45%
    sequence_correctness: 0.20 # 20%
```

Or override per-test in individual test files:

```yaml
thresholds:
  min_score: 80
  weights:
    tool_accuracy: 0.4
    output_quality: 0.4
    sequence_correctness: 0.2
```

> **Note:** Weights must sum to 1.0

## Installation Options

```bash
# Basic installation
pip install evalview

# With HTML reports (Plotly charts)
pip install evalview[reports]

# With watch mode
pip install evalview[watch]

# All optional features
pip install evalview[all]

# Development (includes all features + testing tools)
pip install evalview[dev]
```

## CLI Reference

### `evalview init`

Initialize EvalView in current directory.

```bash
evalview init [--dir PATH]
```

### `evalview run`

Run test cases.

```bash
evalview run [OPTIONS]

Options:
  --pattern TEXT       Test case file pattern (default: *.yaml)
  -t, --test TEXT      Run specific test(s) by name (can repeat)
  -f, --filter TEXT    Filter tests by pattern (e.g., "weather*")
  --output PATH        Output directory for results (default: .evalview/results)
  --verbose            Enable verbose logging
  --debug              Show raw API responses and parsed traces

  # Execution options
  --sequential         Run tests one at a time (default: parallel)
  --max-workers N      Max parallel executions (default: 8)
  --max-retries N      Retry flaky tests N times (default: 0)
  --retry-delay SECS   Base delay between retries (default: 1.0)

  # Development options
  --watch              Re-run tests on file changes
  --html-report PATH   Generate interactive HTML report

  # Regression tracking
  --track              Track results for regression analysis
  --compare-baseline   Compare against baseline and show regressions
```

**Examples:**

```bash
# Run all tests in parallel (default)
evalview run

# Run specific tests
evalview run --filter "stock*" --verbose

# With retry for flaky tests
evalview run --max-retries 3

# Watch mode for development
evalview run --watch

# Generate HTML report
evalview run --html-report report.html
```

See [docs/DEBUGGING.md](docs/DEBUGGING.md) for troubleshooting guide.

### `evalview report`

Generate report from results.

```bash
evalview report RESULTS_FILE [OPTIONS]

Options:
  --detailed      Show detailed results for each test case
  --html PATH     Generate interactive HTML report with charts
```

**Example:**

```bash
# Console summary
evalview report .evalview/results/20241118_004830.json

# Detailed console output
evalview report .evalview/results/20241118_004830.json --detailed

# Interactive HTML report
evalview report .evalview/results/20241118_004830.json --html report.html
```

## Cost Tracking

EvalView automatically tracks costs based on token usage from your agent's API. This helps you:
- **Monitor expenses** - See exactly how much each test costs
- **Set budgets** - Use `max_cost` thresholds to prevent expensive queries
- **Optimize prompts** - Identify and optimize high-cost operations
- **Track trends** - Monitor cost changes across test runs

### Supported Models

Built-in pricing for:
- **gpt-5**: $1.25/1M input, $10/1M output
- **gpt-5-mini**: $0.25/1M input, $2/1M output (recommended)
- **gpt-5-nano**: $0.05/1M input, $0.40/1M output
- **gpt-4o, gpt-4o-mini** - Legacy models
- **Custom pricing** - Set your own rates

### Configuration

During `evalview init --interactive`, you'll select your model and pricing:

```yaml
# .evalview/config.yaml
model:
  name: gpt-5-mini
  # Uses standard OpenAI pricing by default
  # Override with custom pricing:
  # pricing:
  #   input_per_1m: 0.25
  #   output_per_1m: 2.0
  #   cached_per_1m: 0.025
```

### API Requirements

For cost tracking to work, your agent's API must emit token usage data:

**Streaming APIs:**
```json
{"type": "usage", "data": {
  "input_tokens": 1250,
  "output_tokens": 450,
  "cached_tokens": 800
}}
```

**REST APIs:**
```json
{
  "output": "Agent response...",
  "usage": {
    "input_tokens": 1250,
    "output_tokens": 450,
    "cached_tokens": 800
  }
}
```

### Cached Tokens

Cached tokens receive a **90% discount** (10% of input price). This applies when:
- Your agent reuses recent context (e.g., conversation history)
- The LLM provider supports prompt caching
- Tokens are explicitly marked as cached in the API response

### Example Output

```
📊 Evaluation Summary
┏━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━┳━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━━━━━━┳━━━━━━━━━┓
┃ Test Case            ┃ Score ┃ Status  ┃ Cost    ┃ Tokens      ┃ Latency ┃
┡━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━╇━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━━━━━━╇━━━━━━━━━┩
│ Stock Analysis       │  85.2 │ ✅ PASSED│ $0.0123 │ 12,450      │ 89,234ms│
│                      │       │         │         │ (3,200 cache)│         │
└──────────────────────┴───────┴─────────┴─────────┴─────────────┴─────────┘
```

See [docs/COST_TRACKING.md](docs/COST_TRACKING.md) for detailed implementation guide.

## Architecture

```
evalview/
├── core/
│   ├── types.py           # Pydantic models (ExecutionTrace, TokenUsage, etc.)
│   ├── loader.py          # Test case loader
│   ├── pricing.py         # Model pricing & cost calculation
│   ├── config.py          # Configuration models (ScoringWeights, RetryConfig)
│   ├── parallel.py        # Parallel test execution
│   ├── retry.py           # Retry logic with exponential backoff
│   └── watcher.py         # File watcher for watch mode
├── adapters/
│   ├── base.py            # AgentAdapter interface
│   ├── http_adapter.py    # Generic HTTP adapter
│   ├── langgraph_adapter.py  # LangGraph / LangGraph Cloud
│   ├── crewai_adapter.py  # CrewAI agents
│   └── tapescope_adapter.py  # Streaming JSONL adapter
├── evaluators/
│   ├── tool_call_evaluator.py
│   ├── sequence_evaluator.py
│   ├── output_evaluator.py
│   ├── hallucination_evaluator.py
│   ├── safety_evaluator.py
│   ├── cost_evaluator.py
│   ├── latency_evaluator.py
│   └── evaluator.py       # Main orchestrator
├── reporters/
│   ├── json_reporter.py   # JSON output
│   ├── console_reporter.py  # Console output
│   └── html_reporter.py   # Interactive HTML reports
├── tracking/
│   ├── database.py        # SQLite tracking database
│   └── regression.py      # Regression detection
└── cli.py                 # Click CLI
```

## Extending EvalView

### Custom Adapters

Create a custom adapter by subclassing `AgentAdapter`:

```python
from evalview.adapters.base import AgentAdapter
from evalview.core.types import ExecutionTrace

class MyCustomAdapter(AgentAdapter):
    @property
    def name(self) -> str:
        return "my-agent"

    async def execute(self, query: str, context=None) -> ExecutionTrace:
        # Your custom implementation
        pass
```

### Custom Evaluators

Add custom evaluation logic:

```python
from evalview.core.types import TestCase, ExecutionTrace

class CustomEvaluator:
    def evaluate(self, test_case: TestCase, trace: ExecutionTrace):
        # Your evaluation logic
        pass
```

## Environment Variables

- `OPENAI_API_KEY` - Required for LLM-as-judge evaluation
- `DEBUG=1` - Enable verbose logging (alternative to `--verbose` flag)

## Database Setup

Most agents require a valid user ID. Set up your test user:

```bash
# Interactive setup (recommended)
node scripts/setup-test-user.js

# Or see database-specific guides
```

Supported databases:
- PostgreSQL / Prisma
- MongoDB
- MySQL
- Firebase / Firestore
- Supabase
- Any other database system

See [docs/DATABASE_SETUP.md](docs/DATABASE_SETUP.md) for detailed guides.

## Troubleshooting

**Tests failing with "No response"?**
- Run with `--verbose` to see what your API is actually returning
- Check that your endpoint is running and accessible
- Verify the response format matches what the adapter expects

**Database errors about test user?**
- Run `node scripts/setup-test-user.js` to configure
- Or see [docs/DATABASE_SETUP.md](docs/DATABASE_SETUP.md)

**See [docs/DEBUGGING.md](docs/DEBUGGING.md) for detailed troubleshooting guide.**

## CI/CD Integration (Optional)

**Do I have to use EvalView in CI?** No. EvalView is a CLI-first tool.

You can:
- Run `evalview run` locally before deploying
- Add `make agent-tests` to your workflow
- Add it to CI **only if you want**

### Option 1: Local / Makefile (No CI)

```bash
# Run agent tests locally
make agent-tests

# Or directly
evalview run --pattern "tests/test-cases/*.yaml" --verbose
```

### Option 2: GitHub Actions (Optional)

If you want automated testing, create `.github/workflows/evalview.yml`:

```yaml
name: EvalView Agent Tests

on:
  push:
    branches: [ main ]
  pull_request:

jobs:
  evalview:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
      - name: Install EvalView
        run: pip install evalview
      - name: Run EvalView tests
        env:
          OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
        run: evalview run --pattern "tests/test-cases/*.yaml" --verbose
```

> **Note:** Add `OPENAI_API_KEY` to your repository secrets (Settings → Secrets → Actions).

See [.github/workflows/evalview-example.yml](.github/workflows/evalview-example.yml) for a manual-trigger example.

## Development

We use a Makefile for common development tasks. Here's how to get started:

```bash
# Install with dev dependencies
make dev-install

# Run all quality checks (format + lint + typecheck)
make check

# Run tests
make test

# Individual commands
make format      # Format code with black
make lint        # Lint with ruff
make typecheck   # Type check with mypy
make clean       # Clean build artifacts

# See all commands
make help
```

Or use the commands directly:

```bash
# Install development dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Format code
black evalview/

# Type checking
mypy evalview/

# Linting
ruff evalview/
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for detailed contribution guidelines.

## Built For

EvalView is designed for teams building:
- Financial analysis agents
- Customer support chatbots
- Research and data extraction agents
- Code generation tools
- Multi-agent systems

## Roadmap

**Recently Completed:**
- [x] Parallel test execution (8x faster by default)
- [x] Interactive HTML reports with Plotly charts
- [x] Retry logic with exponential backoff
- [x] Watch mode for development
- [x] Configurable scoring weights
- [x] GitHub Actions CI template

**Coming Soon:**
- [ ] Multi-run flakiness detection - Run tests N times, track variance, detect non-determinism
- [ ] Multi-turn conversation testing - Test full conversation flows with context persistence
- [ ] Grounded hallucination checking - Fact-check agent outputs against tool results
- [ ] Error compounding metrics - Track reliability decay over 20+ step workflows
- [ ] Memory/context influence tracking - Measure how agent memory affects behavior

**Want these?** [Vote in GitHub Discussions](https://github.com/hidai25/EvalView/discussions)

**Also Planned:**
- [ ] Test case templates library
- [ ] Custom metric plugins system
- [ ] Cloud-hosted test runner
- [ ] Slack/Discord notifications

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.

## Further Reading

| Topic | Description |
|-------|-------------|
| [Getting Started](docs/GETTING_STARTED.md) | 5-minute quickstart guide |
| [Framework Support](docs/FRAMEWORK_SUPPORT.md) | Supported frameworks and compatibility notes |
| [Cost Tracking](docs/COST_TRACKING.md) | Token usage and cost calculation details |
| [Debugging Guide](docs/DEBUGGING.md) | Troubleshooting common issues |
| [Adapters](docs/ADAPTERS.md) | Building custom adapters for your agent |
| [LangGraph Cloud](docs/LANGGRAPH_CLOUD.md) | LangGraph Cloud integration status |
| [Agent Testing](AGENT_TESTING.md) | Framework support matrix and testing plan |

**Internal docs:** [docs/internal/](docs/internal/) - Implementation notes and architecture decisions

## License

MIT License - see LICENSE file for details.

## Support

- Issues: https://github.com/hidai25/EvalView/issues
- Discussions: https://github.com/hidai25/EvalView/discussions

---

**Built for teams shipping AI agents to production** 🚀
