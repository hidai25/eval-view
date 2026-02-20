# Backend Implementation Examples

Copy-paste examples for implementing each tier of EvalView support.

> **Key contract:** EvalView sends `POST /execute` with `{"query": "..."}` and expects `{"response": "..."}` back.
> The field is `query`, not `message`. The endpoint is `/execute`, not `/chat`.

---

## Level 1: Basic Agent (5 minutes)

Minimum required: accept `query`, return `response`.

### FastAPI (Python)
```python
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional

app = FastAPI()

class ExecuteRequest(BaseModel):
    query: str
    context: Optional[dict] = None

@app.post("/execute")
async def execute(req: ExecuteRequest):
    # Your agent logic here
    response = await your_agent.run(req.query)

    return {"response": response}
```

### Flask (Python)
```python
from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route("/execute", methods=["POST"])
def execute():
    query = request.json["query"]

    # Your agent logic here
    response = your_agent.run(query)

    return jsonify({"response": response})

if __name__ == "__main__":
    app.run(port=8080)
```

### Express.js (Node.js)
```javascript
const express = require('express');
const app = express();
app.use(express.json());

app.post('/execute', async (req, res) => {
  const { query } = req.body;

  // Your agent logic here
  const response = await yourAgent.run(query);

  res.json({ response });
});

app.listen(8080);
```

---

## Level 2: Agent with Tool Tracking (15 minutes)

Add `steps` so EvalView can evaluate which tools were called and in what order.

### FastAPI + Anthropic
```python
import anthropic
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional, List, Dict, Any

app = FastAPI()
client = anthropic.Anthropic()

class ExecuteRequest(BaseModel):
    query: str
    context: Optional[dict] = None

@app.post("/execute")
async def execute(req: ExecuteRequest):
    steps: List[Dict[str, Any]] = []

    response = client.messages.create(
        model="claude-opus-4-6",
        max_tokens=1024,
        messages=[{"role": "user", "content": req.query}],
        tools=[...],  # your tools
    )

    # Capture tool calls
    for block in response.content:
        if block.type == "tool_use":
            steps.append({
                "tool": block.name,
                "parameters": block.input,
                "output": None,  # fill in after tool execution
            })

    final_text = next(
        (b.text for b in response.content if hasattr(b, "text")), ""
    )

    return {
        "response": final_text,
        "steps": steps,
        "cost": (response.usage.input_tokens * 0.000015 +
                 response.usage.output_tokens * 0.000075),
        "tokens": {
            "input": response.usage.input_tokens,
            "output": response.usage.output_tokens,
        },
    }
```

### FastAPI + OpenAI
```python
from openai import AsyncOpenAI
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional, List, Dict, Any

app = FastAPI()
client = AsyncOpenAI()

class ExecuteRequest(BaseModel):
    query: str
    context: Optional[dict] = None

@app.post("/execute")
async def execute(req: ExecuteRequest):
    steps: List[Dict[str, Any]] = []

    response = await client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": req.query}],
        tools=[...],  # your tools
    )

    message = response.choices[0].message

    # Capture tool calls
    if message.tool_calls:
        for tc in message.tool_calls:
            steps.append({
                "tool": tc.function.name,
                "parameters": tc.function.arguments,
                "output": None,
            })

    usage = response.usage
    cost = (usage.prompt_tokens * 0.000005 +
            usage.completion_tokens * 0.000015)

    return {
        "response": message.content or "",
        "steps": steps,
        "cost": cost,
        "tokens": {
            "input": usage.prompt_tokens,
            "output": usage.completion_tokens,
            "cached": getattr(usage, "prompt_tokens_details", {}).get("cached_tokens", 0),
        },
    }
```

---

## Level 3: Streaming Agent (30 minutes)

Use `adapter: streaming` in your config. EvalView reads JSONL events line by line.

### FastAPI with JSONL Streaming
```python
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional
import json

app = FastAPI()

class ExecuteRequest(BaseModel):
    query: str
    context: Optional[dict] = None

@app.post("/execute")
async def execute(req: ExecuteRequest):
    async def event_stream():
        # Tool call
        yield json.dumps({
            "type": "tool_call",
            "data": {"name": "search", "args": {"query": req.query}}
        }) + "\n"

        # Tool result
        result = await your_agent.search(req.query)
        yield json.dumps({
            "type": "tool_result",
            "data": {"result": result, "success": True}
        }) + "\n"

        # Token usage
        yield json.dumps({
            "type": "usage",
            "data": {"input_tokens": 100, "output_tokens": 300, "cached_tokens": 0}
        }) + "\n"

        # Final response
        yield json.dumps({
            "type": "message_complete",
            "data": {"content": f"Here are results for: {req.query}"}
        }) + "\n"

    return StreamingResponse(event_stream(), media_type="application/x-ndjson")
```

---

## Testing Your Implementation

### 1. Verify basic connectivity
```bash
curl -X POST http://localhost:8080/execute \
  -H "Content-Type: application/json" \
  -d '{"query": "hello"}' | jq
```
Expected: `{"response": "..."}`

### 2. Verify tool tracking (Level 2)
```bash
curl -X POST http://localhost:8080/execute \
  -H "Content-Type: application/json" \
  -d '{"query": "search for EvalView"}' | jq '.steps'
```
Expected: `[{"tool": "search", "parameters": {...}, "output": "..."}]`

### 3. Point EvalView at it
```yaml
# .evalview/config.yaml
adapter: http
endpoint: http://localhost:8080/execute
timeout: 30.0
```

```bash
evalview run
```

---

## Common Mistakes

### ❌ Wrong field name
```python
query = request.json["message"]  # EvalView sends "query", not "message"
```
### ✅ Correct
```python
query = request.json["query"]
```

---

### ❌ Wrong endpoint path
```python
@app.route("/api/chat", methods=["POST"])  # EvalView expects /execute by default
```
### ✅ Correct
```python
@app.route("/execute", methods=["POST"])
# Or set a custom path in .evalview/config.yaml: endpoint: http://localhost:8080/api/chat
```

---

### ❌ Plain text response
```python
return "The answer is 42"  # EvalView can't parse this
```
### ✅ JSON with response key
```python
return {"response": "The answer is 42"}
```

---

## Questions?

- See `docs/BACKEND_REQUIREMENTS.md` for the full spec
- See `demo-agent/agent.py` for a complete working example
- File issues: https://github.com/hidai25/eval-view/issues
