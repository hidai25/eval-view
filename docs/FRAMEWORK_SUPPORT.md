# Framework Support Guide

EvalView supports multiple AI agent frameworks out of the box. Each framework has a dedicated adapter that handles its specific API format.

## Supported Frameworks

| Framework | Adapter | Auto-Detect | Default Port | Endpoint |
|-----------|---------|-------------|--------------|----------|
| **LangGraph** | `langgraph` | ✅ | 8000 | `/api/chat` or `/invoke` |
| **LangServe** | `http` or `streaming` | ✅ | 8000 | `/agent` or `/agent/stream` |
| **CrewAI** | `crewai` | ✅ | 8000 | `/crew` |
| **OpenAI Assistants** | `openai-assistants` | N/A | N/A | Uses OpenAI API |
| **TapeScope** | `streaming` | ✅ | 3000 | `/api/unifiedchat` |
| **Generic REST** | `http` | ✅ | Any | Any |
| **Generic Streaming** | `streaming` | ✅ | Any | Any |

## Quick Start

### Auto-Detection (Recommended)

```bash
# Start your agent server first
# Then let EvalView detect it automatically
evalview connect
```

The `connect` command will:
1. Try common endpoints
2. Detect which framework is running
3. Configure the correct adapter automatically
4. Update `.evalview/config.yaml`

### Manual Configuration

Edit `.evalview/config.yaml`:

```yaml
adapter: langgraph  # or crewai, http, streaming, etc.
endpoint: http://localhost:8000/api/chat
timeout: 30.0
```

## Framework-Specific Guides

### 1. LangGraph

**What it supports:**
- Standard invoke endpoint
- Streaming responses
- Message-based APIs
- Thread tracking

**Setup:**
```bash
# Start LangGraph agent
cd /path/to/langgraph-agent
python main.py
# or
uvicorn main:app --reload --port 8000

# Connect EvalView
evalview connect
```

**Config:**
```yaml
adapter: langgraph
endpoint: http://localhost:8000/api/chat
streaming: false  # Set to true for streaming endpoints
timeout: 30.0

model:
  name: gpt-4o-mini
```

**Test Case Example:**
```yaml
name: "LangGraph Test"
input:
  query: "What is the weather in SF?"
  context: {}

expected:
  tools: [tavily_search]  # Update with your actual tools
  output:
    contains: ["San Francisco", "weather"]

thresholds:
  min_score: 70
  max_cost: 0.50
  max_latency: 10000
```

**Response Format Expected:**
```json
{
  "messages": [
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "..."}
  ],
  "thread_id": "...",
  "intermediate_steps": [...]
}
```

---

### 2. CrewAI

**What it supports:**
- Task-based execution
- Multi-agent crews
- Usage metrics

**Setup:**
```bash
# Start CrewAI API
cd /path/to/crewai-agent
python api.py  # or however you serve it

# Connect
evalview connect
```

**Config:**
```yaml
adapter: crewai
endpoint: http://localhost:8000/crew
timeout: 120.0  # CrewAI can be slow
```

**Test Case Example:**
```yaml
name: "CrewAI Research Test"
input:
  query: "Research AI trends in 2025"
  context: {}

expected:
  tools: []  # CrewAI uses agents, not direct tools
  output:
    contains: ["AI", "trends", "2025"]

thresholds:
  min_score: 75
  max_cost: 2.00
  max_latency: 60000  # 60 seconds
```

**Response Format Expected:**
```json
{
  "result": "Final crew output",
  "tasks": [
    {
      "id": "task-1",
      "description": "Research task",
      "output": "...",
      "status": "completed"
    }
  ],
  "usage_metrics": {
    "total_tokens": 1500,
    "total_cost": 0.045
  }
}
```

---

### 3. OpenAI Assistants

**What it supports:**
- OpenAI Assistants API
- Function calling
- Code interpreter
- File search/retrieval

**Setup:**
```bash
# Set your OpenAI API key
export OPENAI_API_KEY=sk-...

# No server needed - uses OpenAI API directly
```

**Config:**
```yaml
adapter: openai-assistants
assistant_id: asst_xxxxxxxxxxxxx  # Your assistant ID
timeout: 120.0
```

**Test Case Example:**
```yaml
name: "OpenAI Assistant Test"
input:
  query: "Calculate the fibonacci sequence up to 10"
  context:
    assistant_id: asst_xxxxxxxxxxxxx  # Can override here too

expected:
  tools: [code_interpreter]
  output:
    contains: ["fibonacci", "0, 1, 1, 2, 3, 5, 8"]

thresholds:
  min_score: 80
  max_cost: 0.50
  max_latency: 30000
```

**Notes:**
- Requires `openai` Python package: `pip install openai`
- Uses threads and runs under the hood
- Automatically polls for completion

---

### 4. LangServe

**What it supports:**
- Standard REST endpoints
- Streaming via Server-Sent Events
- Batch processing

**Setup:**
```bash
# Start LangServe
cd /path/to/langserve-app
python server.py

# Connect
evalview connect
```

**Config (non-streaming):**
```yaml
adapter: http
endpoint: http://localhost:8000/agent/invoke
timeout: 30.0
```

**Config (streaming):**
```yaml
adapter: streaming
endpoint: http://localhost:8000/agent/stream
timeout: 60.0
```

---

### 5. Generic HTTP/REST

**For any custom REST API**

**Config:**
```yaml
adapter: http
endpoint: http://localhost:YOUR_PORT/YOUR_PATH
timeout: 30.0
headers:
  Authorization: Bearer YOUR_TOKEN
  Content-Type: application/json
```

**Expected Request Format:**
```json
{
  "query": "User query here",
  "context": {}
}
```

**Expected Response Format:**
```json
{
  "session_id": "...",
  "output": "Final response",
  "steps": [
    {
      "id": "step-1",
      "name": "Step name",
      "tool": "tool_name",
      "parameters": {...},
      "output": {...},
      "latency": 123,
      "cost": 0.001
    }
  ],
  "cost": 0.05,
  "tokens": 1000
}
```

---

## Creating Custom Adapters

If your framework isn't supported, create a custom adapter:

```python
# evalview/adapters/my_adapter.py
from evalview.adapters.base import AgentAdapter
from evalview.core.types import ExecutionTrace, StepTrace, StepMetrics, ExecutionMetrics
from datetime import datetime

class MyAdapter(AgentAdapter):
    @property
    def name(self) -> str:
        return "my-adapter"

    async def execute(self, query: str, context=None) -> ExecutionTrace:
        # 1. Call your agent API
        # 2. Parse response
        # 3. Extract steps and output
        # 4. Return ExecutionTrace
        pass
```

Register in `cli.py`:

```python
from evalview.adapters.my_adapter import MyAdapter

# In _run_async():
elif adapter_type == "my-adapter":
    adapter = MyAdapter(...)
```

See [ADAPTERS.md](docs/ADAPTERS.md) for full guide.

---

## Troubleshooting

### Connection Failed

```bash
# Test endpoint manually
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "test"}'

# Check if server is running
lsof -i :8000

# Try auto-detect
evalview connect
```

### Wrong Adapter Detected

Manually set in `.evalview/config.yaml`:

```yaml
adapter: langgraph  # Override auto-detection
```

### Response Format Mismatch

Run with verbose to see actual response:

```bash
evalview run --verbose
```

Then adjust your test case or create a custom adapter.

### Timeout Issues

Increase timeout:

```yaml
timeout: 120.0  # 2 minutes
```

---

## Framework Comparison

| Feature | LangGraph | CrewAI | OpenAI | LangServe |
|---------|-----------|--------|--------|-----------|
| Streaming | ✅ | ❌ | ❌ | ✅ |
| Multi-step | ✅ | ✅ | ✅ | ✅ |
| Self-hosted | ✅ | ✅ | ❌ | ✅ |
| Tool tracking | ✅ | Partial | ✅ | ✅ |
| Cost tracking | Manual | ✅ | ✅ | Manual |

---

## Best Practices

1. **Always use `evalview connect` first** - Let it auto-detect
2. **Start with verbose mode** - Understand API responses
3. **Check framework docs** - Verify endpoint paths
4. **Use framework-specific adapters** - Better parsing and metrics
5. **Monitor timeouts** - Some agents can be slow

---

## Need Help?

- Check [QUICKSTART_LANGGRAPH.md](QUICKSTART_LANGGRAPH.md) for LangGraph
- Check [SETUP_LANGGRAPH_EXAMPLE.md](SETUP_LANGGRAPH_EXAMPLE.md) for detailed setup
- Check [ADAPTERS.md](docs/ADAPTERS.md) for custom adapters
- Open an issue: https://github.com/hidai25/eval-view/issues

---

**Sources:**
- [LangGraph Platform API](https://langchain-ai.github.io/langgraph/cloud/reference/api/api_ref.html)
- [LangServe Documentation](https://python.langchain.com/docs/langserve/)
