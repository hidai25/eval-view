# LangGraph Example Agent

A simple LangGraph agent with search and calculator tools for testing with EvalView.

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Set up environment
cp .env.local.example .env.local
# Edit .env.local and add your OPENAI_API_KEY

# 3. Start the agent
langgraph dev
```

Server runs at: http://localhost:2024

## Test with EvalView

```bash
# From the EvalView root directory
evalview run examples/langgraph/
```

## Tools

| Tool | Description |
|------|-------------|
| `tavily_search_results_json` | Web search (uses mock data if no TAVILY_API_KEY) |
| `calculator` | Basic math operations |

## Notes

- Works without TAVILY_API_KEY (uses mock search results)
- Uses gpt-4o-mini by default (set OPENAI_MODEL to change)
- Standard LangGraph format - EvalView auto-detects it
