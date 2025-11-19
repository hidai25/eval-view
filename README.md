# AgentEval

**Professional Testing Framework for AI Agents** - Like Playwright, but for AI.

AgentEval is a production-ready Python CLI tool that helps teams systematically test, evaluate, and monitor AI agents through structured test cases, automated evaluations, and comprehensive reporting.

Perfect for:
- **AI Startups** - Ensure your agent works before shipping
- **Enterprise Teams** - Maintain quality across agent deployments
- **CI/CD Pipelines** - Automated testing for every commit
- **Research Labs** - Benchmark and compare agent performance

## Why AgentEval?

Traditional testing tools don't work for AI agents. Agents are:
- **Non-deterministic** - Same input, different outputs
- **Multi-step** - Complex workflows with tool calls
- **Context-dependent** - Behavior changes based on state

AgentEval solves this with:
- **Flexible assertions** - Test for content, not exact matches
- **Tool call tracking** - Verify correct tool usage and sequences
- **LLM-as-judge** - Automated quality scoring using GPT-4
- **Cost & latency monitoring** - Catch expensive or slow executions

## Features

- ✅ **YAML-based test cases** - Write readable, maintainable test definitions
- 📊 **Multiple evaluation metrics** - Tool accuracy, sequence correctness, output quality, cost, and latency
- 🤖 **LLM-as-judge** - Automated output quality assessment using GPT-4
- 💰 **Cost tracking** - Automatic cost calculation based on token usage with GPT-5 family pricing
- 🔌 **Universal adapters** - Works with any HTTP or streaming API
- 🎨 **Rich console output** - Beautiful, informative test results
- 📁 **JSON reports** - Structured results for CI/CD integration
- 🐛 **Verbose debugging** - Detailed logging to troubleshoot issues
- 🗄️ **Database-agnostic** - Works with PostgreSQL, MongoDB, MySQL, Firebase, and more

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/agent-eval.git
cd agent-eval

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode
pip install -e .
```

## Quick Start

### 1. Initialize Project

```bash
agent-eval init --interactive
```

The interactive setup will guide you through:
1. **API Configuration** - Choose REST or Streaming API
2. **Endpoint URL** - Your agent's API endpoint
3. **Model Selection** - Which GPT model your agent uses (gpt-5, gpt-5-mini, gpt-5-nano, etc.)
4. **Pricing Configuration** - Confirm standard pricing or set custom rates

This creates:
- `.agenteval/config.yaml` - Configuration for your agent endpoint and model pricing
- `tests/test-cases/` - Directory for test cases
- `tests/test-cases/example.yaml` - Example test case

### 2. Configure Your Agent

Edit `.agenteval/config.yaml`:

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
agent-eval run

# Run specific pattern
agent-eval run --pattern "stock-*.yaml"
```

### 5. View Results

Results are displayed in the console and saved to `.agenteval/results/`:

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
agent-eval report .agenteval/results/20241118_004830.json --detailed
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

## CLI Reference

### `agent-eval init`

Initialize AgentEval in current directory.

```bash
agent-eval init [--dir PATH]
```

### `agent-eval run`

Run test cases.

```bash
agent-eval run [OPTIONS]

Options:
  --pattern TEXT   Test case file pattern (default: *.yaml)
  --output PATH    Output directory for results (default: .agenteval/results)
  --verbose        Enable verbose logging (shows API requests/responses)
```

**Debugging**: Use `--verbose` to see detailed logs of what's happening:

```bash
# See exactly what the API is returning
agent-eval run --verbose

# Or use environment variable
DEBUG=1 agent-eval run
```

See [DEBUGGING.md](DEBUGGING.md) for troubleshooting guide.

### `agent-eval report`

Generate report from results.

```bash
agent-eval report RESULTS_FILE [OPTIONS]

Options:
  --detailed  Show detailed results for each test case
```

## Cost Tracking

AgentEval automatically tracks costs based on token usage from your agent's API. This helps you:
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

During `agent-eval init --interactive`, you'll select your model and pricing:

```yaml
# .agenteval/config.yaml
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

See [COST_TRACKING.md](COST_TRACKING.md) for detailed implementation guide.

## Architecture

```
agent_eval/
├── core/
│   ├── types.py           # Pydantic models (ExecutionTrace, TokenUsage, etc.)
│   ├── loader.py          # Test case loader
│   └── pricing.py         # Model pricing & cost calculation
├── adapters/
│   ├── base.py            # AgentAdapter interface
│   ├── http_adapter.py    # Generic HTTP adapter
│   └── tapescope_adapter.py  # Streaming JSONL adapter
├── evaluators/
│   ├── tool_call_evaluator.py
│   ├── sequence_evaluator.py
│   ├── output_evaluator.py
│   ├── cost_evaluator.py
│   ├── latency_evaluator.py
│   └── evaluator.py       # Main orchestrator
├── reporters/
│   ├── json_reporter.py   # JSON output
│   └── console_reporter.py  # Console output
└── cli.py                 # Click CLI
```

## Extending AgentEval

### Custom Adapters

Create a custom adapter by subclassing `AgentAdapter`:

```python
from agent_eval.adapters.base import AgentAdapter
from agent_eval.core.types import ExecutionTrace

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
from agent_eval.core.types import TestCase, ExecutionTrace

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

**See [DEBUGGING.md](DEBUGGING.md) for detailed troubleshooting guide.**

## Development

```bash
# Install development dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Format code
black agent_eval/

# Type checking
mypy agent_eval/

# Linting
ruff agent_eval/
```

## Who's Using AgentEval?

AgentEval is production-ready and used by teams building:
- Financial analysis agents
- Customer support chatbots
- Research and data extraction agents
- Code generation tools
- Multi-agent systems

## Roadmap

- [ ] HTML report generator with charts
- [ ] Parallel test execution for faster runs
- [ ] Test case templates library
- [ ] Native LangChain & CrewAI adapters
- [ ] Custom metric plugins system
- [ ] Test result diffing across runs
- [ ] Performance regression detection
- [ ] Cloud-hosted test runner
- [ ] Slack/Discord notifications

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.

## License

MIT License - see LICENSE file for details.

## Support

- Issues: https://github.com/yourusername/agent-eval/issues
- Discussions: https://github.com/yourusername/agent-eval/discussions

---

**Built for the AI agent community** 🤖
