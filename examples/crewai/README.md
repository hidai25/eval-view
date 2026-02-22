# CrewAI Example — Testing CrewAI Multi-Agent Crews with EvalView

> Test CrewAI multi-agent crews with EvalView — track agent collaboration, verify tool usage across agents, measure costs, and detect regressions in multi-agent workflows.

## Example Output

![EvalView CrewAI Results](screenshot.png)

<details>
<summary>Text version</summary>

```
                               📊 Evaluation Summary
┏━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━┳━━━━━━━━━━┓
┃ Test Case           ┃ Backend  ┃ Score ┃ Status    ┃    Cost ┃ Tokens ┃  Latency ┃
┡━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━╇━━━━━━━━━━┩
│ Stock Analysis      │ Crewai   │  85.0 │ ✅ PASSED │ $0.0245 │  3,420 │  45230ms │
│ Content Team        │ Crewai   │  90.0 │ ✅ PASSED │ $0.0189 │  2,890 │  38100ms │
└─────────────────────┴──────────┴───────┴───────────┴─────────┴────────┴──────────┘

Execution Flow (4 steps)
├── Step 1: researcher_agent ✓  [12000ms | $0.0080]
│   └── → task: "Research AAPL stock performance"
├── Step 2: analyst_agent ✓  [15000ms | $0.0095]
│   └── → task: "Analyze market trends"
├── Step 3: writer_agent ✓  [10000ms | $0.0050]
│   └── → task: "Write analysis report"
└── Step 4: reviewer_agent ✓  [8000ms | $0.0020]
    └── → task: "Review and finalize"
```

</details>

## Quick Start

### 1. Install CrewAI

```bash
pip install crewai crewai-tools
```

### 2. Clone CrewAI Examples

```bash
git clone https://github.com/crewAIInc/crewAI-examples.git
cd crewAI-examples
```

### 3. Choose an Example

```bash
# Stock Analysis crew
cd crews/stock_analysis

# Or Content Creator flow
cd flows/content_creator_flow

# Or Trip Planner
cd crews/trip_planner
```

### 4. Set API Keys

```bash
export OPENAI_API_KEY=sk-...
# Some crews may need additional keys (e.g., SERPER_API_KEY for search)
```

> **Note:** Some examples use Ollama (local LLM). To use OpenAI instead, edit `crew.py` and replace:
> ```python
> from langchain.llms import Ollama
> llm = Ollama(model="llama3.1")
> ```
> with:
> ```python
> from langchain_openai import ChatOpenAI
> llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
> ```

### 5. Run the Crew with API Server

```bash
# Install dependencies
pip install -r requirements.txt

# Run with FastAPI server (if supported)
crewai run --serve

# Or check the example's README for specific instructions
```

Server typically runs at: `http://localhost:8000`

### 6. Run EvalView Tests

```bash
# From EvalView root
evalview run --pattern examples/crewai/
```

## Available CrewAI Examples

| Example | Path | Description |
|---------|------|-------------|
| Stock Analysis | `crews/stock_analysis` | Multi-agent stock research |
| Trip Planner | `crews/trip_planner` | Travel planning crew |
| Content Creator | `flows/content_creator_flow` | Content generation flow |
| Marketing Strategy | `crews/marketing_strategy` | Marketing research crew |
| Job Posting | `crews/job-posting` | Job description generator |

## Configuration

```yaml
# .evalview/config.yaml
adapter: crewai
endpoint: http://localhost:8000
timeout: 120  # CrewAI crews can take longer
```

## Writing Test Cases

```yaml
name: "My Crew Test"
adapter: crewai
endpoint: http://localhost:8000

input:
  query: "Analyze Tesla stock for investment potential"
  context:
    ticker: "TSLA"
    analysis_type: "comprehensive"

expected:
  tools:
    - search_tool
    - calculator_tool
  output:
    contains:
      - "Tesla"
      - "stock"
      - "recommendation"

thresholds:
  min_score: 70
  max_cost: 0.50
  max_latency: 120000  # 2 minutes
```

## Troubleshooting

**"Connection refused on port 8000"**
- Make sure the crew is running with `--serve` flag
- Check if the example supports API mode (not all do)

**"Crew takes too long"**
- Increase timeout in config: `timeout: 180`
- Multi-agent crews can take 1-2 minutes

**"Missing API keys"**
- Check the example's README for required keys
- Common: `OPENAI_API_KEY`, `SERPER_API_KEY`

## Links

- [CrewAI Docs](https://docs.crewai.com/)
- [CrewAI GitHub](https://github.com/crewAIInc/crewAI)
- [CrewAI Examples](https://github.com/crewAIInc/crewAI-examples)
- [EvalView Docs](../../docs/)
