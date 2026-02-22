# LangGraph Example — Testing LangGraph Agents with EvalView

> Test LangGraph agents with EvalView — capture tool calls, verify execution sequences, measure latency and cost, detect regressions with golden baselines, and run in CI.

## Example Output

![EvalView LangGraph Results](screenshot.png)

<details>
<summary>Text version</summary>

```
                               📊 Evaluation Summary
┏━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━┳━━━━━━━━━┓
┃ Test Case           ┃ Backend   ┃ Score ┃ Status    ┃    Cost ┃ Tokens ┃ Latency ┃
┡━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━╇━━━━━━━━━┩
│ Conversational Test │ Langgraph │  80.0 │ ✅ PASSED │ $0.0014 │    321 │  6533ms │
│ Search Test         │ Langgraph │  85.0 │ ✅ PASSED │ $0.0024 │    720 │  7244ms │
│ Multi-Step Research │ Langgraph │  90.0 │ ✅ PASSED │ $0.0089 │  2,450 │ 12340ms │
└─────────────────────┴───────────┴───────┴───────────┴─────────┴────────┴─────────┘

Execution Flow (3 steps)
├── Step 1: tavily_search [green]✓[/green]  [2100ms | $0.0020]
│   └── → params: {"query": "AI agents 2024 trends"}
├── Step 2: tavily_search [green]✓[/green]  [1800ms | $0.0020]
│   └── → params: {"query": "LangGraph vs AutoGPT comparison"}
└── Step 3: summarize [green]✓[/green]  [3200ms | $0.0049]
    └── → params: {"content": "Based on the search results..."}
```

</details>

## Quick Start

### 1. Install Dependencies

```bash
# Python 3.11+ required
pip install "langgraph-cli[inmem]" langchain-openai langchain-anthropic tavily-python
```

### 2. Set API Keys

```bash
export OPENAI_API_KEY=sk-...
export TAVILY_API_KEY=tvly-...  # Get free key at tavily.com
```

### 3. Start LangGraph Server

**Option A: Use the included example agent**

```bash
cd examples/langgraph/agent
langgraph dev
```

**Option B: Use your own LangGraph agent**

```bash
cd /path/to/your/langgraph/project
langgraph dev
```

Server runs at: `http://localhost:2024`

### 4. Run Tests

```bash
# From EvalView root
evalview run --pattern examples/langgraph/
```

## Test Cases

| Test | What it checks |
|------|---------------|
| `conversational.yaml` | Basic Q&A without tools |
| `search.yaml` | Web search tool usage |
| `multi-step.yaml` | Multi-tool research workflow |

## Configuration

EvalView auto-detects LangGraph Cloud API on port 2024. To configure manually:

```yaml
# .evalview/config.yaml
adapter: langgraph
endpoint: http://localhost:2024
assistant_id: agent  # Your graph name from langgraph.json
timeout: 90
```

## Writing Test Cases

```yaml
name: "My Test"
adapter: langgraph
endpoint: http://localhost:2024

input:
  query: "What are the latest AI trends?"
  context:
    assistant_id: agent  # Optional: override default assistant

expected:
  tools:
    - tavily_search  # Expected tools to be called
  output:
    contains:
      - "AI"
      - "trends"

thresholds:
  min_score: 70
  max_cost: 0.10
  max_latency: 30000
```

## Troubleshooting

**"Python 3.11+ required"**
```bash
# Use conda or pyenv
conda create -n langgraph python=3.12
conda activate langgraph
```

**"TAVILY_API_KEY not found"**
- Get a free key at [tavily.com](https://tavily.com)
- Or modify the agent to remove the search tool

**"Connection refused on port 2024"**
- Make sure `langgraph dev` is running
- Check for errors in the server terminal

## Links

- [LangGraph Docs](https://langchain-ai.github.io/langgraph/)
- [LangGraph GitHub](https://github.com/langchain-ai/langgraph)
- [EvalView Docs](../../docs/)
