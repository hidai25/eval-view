# LangGraph Example

Test a LangGraph research agent with EvalView.

## Setup

### 1. Clone LangGraph Examples

```bash
# Option A: LangGraph quickstart
pip install langgraph langchain-openai

# Option B: Clone full examples repo
git clone https://github.com/langchain-ai/langgraph.git
cd langgraph/examples
```

### 2. Start the Agent

```bash
# Using LangGraph CLI
langgraph dev

# Or run the example server
cd langgraph/examples/chat_agent_executor
python server.py
```

Agent will be available at: `http://localhost:8123`

### 3. Run EvalView Test

```bash
# From EvalView root
evalview run --pattern examples/langgraph/test-case.yaml
```

## Links

- **Repo**: https://github.com/langchain-ai/langgraph
- **Quickstart**: https://langchain-ai.github.io/langgraph/tutorials/introduction/
- **Examples**: https://github.com/langchain-ai/langgraph/tree/main/examples
