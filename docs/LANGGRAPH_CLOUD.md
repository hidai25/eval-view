# LangGraph Cloud API Support — Testing LangGraph Cloud Agents with EvalView

## Quick Note

LangGraph has two API formats:

1. **Simple Invoke** - Direct `/invoke` or `/api/chat` endpoints (port 8000)
2. **LangGraph Cloud** - Thread-based API with `/threads/{thread_id}/runs` (port 2024)

## Currently Detected: LangGraph Cloud (Port 2024)

Your LangGraph is running the Cloud API format. This requires a different workflow.

### Temporary Workaround

Until full Cloud API support is added, you can:

1. **Use LangGraph's simple server mode:**
   ```bash
   # In your langgraph project
   # Create a simple FastAPI wrapper
   # See: https://github.com/langchain-ai/langgraph-example
   ```

2. **Or manually configure for Cloud API:**
   ```yaml
   # .evalview/config.yaml
   adapter: langgraph
   endpoint: http://127.0.0.1:2024
   assistant_id: agent  # Your assistant/graph ID
   ```

### Coming Soon

We're adding full LangGraph Cloud API support which will:
- Auto-create threads
- Stream runs properly
- Track all steps
- Handle the async nature of runs

### For Now - Use Simple Mode

The easiest solution:

```bash
# Instead of: langgraph dev
# Use a simple invoke endpoint:
python -c "
from langgraph import your_graph
from fastapi import FastAPI
import uvicorn

app = FastAPI()

@app.post('/api/chat')
def chat(query: str):
    result = your_graph.invoke({'messages': [{'role': 'user', 'content': query}]})
    return result

uvicorn.run(app, port=8000)
"
```

Then:
```bash
evalview connect  # Will find it on port 8000
evalview run
```

## Status

- ✅ LangGraph simple API (invoke) - Fully supported
- ⏳ LangGraph Cloud API (threads/runs) - Coming soon
- ✅ Auto-detection works for both
- ✅ Port scanning finds both

---

**Tracked in issue:** [Support for LangGraph Cloud API](#)
