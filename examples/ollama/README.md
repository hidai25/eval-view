# Ollama Integration — Free, Offline AI Agent Testing with EvalView

> Use Ollama with EvalView for completely free, fully offline AI agent testing and evaluation. No API keys needed, no cloud dependencies.

Use Ollama with EvalView in two ways:

1. **Test agents powered by Ollama** - Test LangGraph/CrewAI agents using local Llama models
2. **Use Ollama as judge** - Free local LLM-as-judge evaluation (no API costs)

## Prerequisites

```bash
# Install Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# Pull a model
ollama pull llama3.2

# Start Ollama server
ollama serve
```

---

## Option 1: Test LangGraph Agents with Ollama

Test a LangGraph agent that uses Ollama as its LLM backend.

### Setup LangGraph + Ollama Agent

```python
# agent.py
from langgraph.graph import StateGraph
from langchain_ollama import ChatOllama

# Use local Ollama model
llm = ChatOllama(model="llama3.2")

# Build your agent with tools...
```

### Run Tests

```bash
# Start your LangGraph agent
cd your-agent/
langgraph dev

# Run EvalView tests
evalview run --pattern examples/ollama/
```

### Test Case Example

```yaml
# langgraph-ollama-test.yaml
name: "LangGraph + Ollama Agent Test"
description: "Test local Llama agent with tools"

adapter: langgraph
endpoint: http://localhost:2024

input:
  query: "What's the weather in Paris and convert 20 EUR to USD?"

expected:
  tools:
    - get_weather
    - currency_convert
  output:
    contains:
      - "Paris"
      - "USD"

thresholds:
  min_score: 70
  max_latency: 60000  # Local models are slower
```

---

## Option 2: Use Ollama as LLM-as-Judge (Free)

Evaluate your agents using Ollama instead of OpenAI/Anthropic - completely free!

### Run with Ollama Judge

```bash
# Make sure Ollama is running
ollama serve

# Run tests with Ollama as the judge
evalview run --judge-provider ollama --judge-model llama3.2

# Or set environment variable
export EVAL_PROVIDER=ollama
evalview run
```

### Supported Models

| Model | Best For |
|-------|----------|
| `llama3.2` | Fast, general purpose |
| `llama3.1:70b` | Higher quality (if you have RAM) |
| `mistral` | Good balance of speed/quality |
| `codellama` | Code-related evaluations |

### Example

```bash
# Test OpenAI agent, evaluate with free local Ollama
evalview run --judge-provider ollama --judge-model llama3.2

# Output:
# Using Ollama (Local) for LLM-as-judge
# ✅ Test passed (score: 85) - Cost: $0.00 (free!)
```

---

## Comparison

| Use Case | Ollama Role | API Costs |
|----------|-------------|-----------|
| Agent powered by Ollama | LLM brain | Free |
| Ollama as judge | Evaluator | Free |
| Cloud agent + Ollama judge | Both | Agent cost only |

---

## Troubleshooting

**"Connection refused"**
```bash
# Make sure Ollama is running
ollama serve
```

**"Model not found"**
```bash
# Pull the model first
ollama pull llama3.2
```

**Slow evaluation**
- Local models are slower than cloud APIs
- Consider smaller models for faster iteration
- Use `llama3.2` (3B params) for quick tests

---

## Links

- [Ollama](https://ollama.ai/)
- [LangChain Ollama](https://python.langchain.com/docs/integrations/llms/ollama)
- [LangGraph](https://langchain-ai.github.io/langgraph/)
