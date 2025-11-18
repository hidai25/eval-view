# AgentEval

**Testing framework for multi-step AI agents** - Like Playwright, but for AI.

AgentEval is a Python CLI tool that helps you systematically test and evaluate AI agents through structured test cases, automated evaluations, and comprehensive reporting.

## Features

- **YAML-based test cases** - Write readable, maintainable test definitions
- **Multiple evaluation metrics** - Tool accuracy, sequence correctness, output quality, cost, and latency
- **LLM-as-judge** - Automated output quality assessment using GPT-4
- **Generic HTTP adapter** - Works with any REST API-based agent
- **Rich console output** - Beautiful, informative test results
- **JSON reports** - Structured results for CI/CD integration

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
agent-eval init
```

This creates:
- `.agenteval/config.yaml` - Configuration for your agent endpoint
- `tests/test-cases/` - Directory for test cases
- `tests/test-cases/example.yaml` - Example test case

### 2. Configure Your Agent

Edit `.agenteval/config.yaml`:

```yaml
adapter: http
endpoint: http://localhost:3000/api/agent
timeout: 30.0
headers:
  Authorization: Bearer your-api-key
```

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
- Must stay under `max_cost`
- Provides breakdown by step
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
```

### `agent-eval report`

Generate report from results.

```bash
agent-eval report RESULTS_FILE [OPTIONS]

Options:
  --detailed  Show detailed results for each test case
```

## Architecture

```
agent_eval/
├── core/
│   ├── types.py           # Pydantic models
│   └── loader.py          # Test case loader
├── adapters/
│   ├── base.py            # AgentAdapter interface
│   └── http_adapter.py    # Generic HTTP adapter
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

## Roadmap

- [ ] HTML report generator
- [ ] Parallel test execution
- [ ] Test case templates
- [ ] CI/CD integration guides
- [ ] Additional adapters (LangChain, CrewAI, etc.)
- [ ] Custom metric plugins
- [ ] Test result diffing
- [ ] Performance regression detection

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.

## License

MIT License - see LICENSE file for details.

## Support

- Issues: https://github.com/yourusername/agent-eval/issues
- Discussions: https://github.com/yourusername/agent-eval/discussions

---

**Built for the AI agent community** 🤖
