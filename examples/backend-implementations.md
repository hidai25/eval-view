# Backend Implementation Examples

Copy-paste examples for implementing each tier of EvalView support.

## Level 1: Basic Agent (5 minutes)

### Express.js (Node.js)
```javascript
const express = require('express');
const app = express();
app.use(express.json());

app.post('/api/chat', async (req, res) => {
  const { message } = req.body;

  // Your agent logic here
  const response = await yourAgent.run(message);

  // Minimum response format
  res.json({
    response: response
  });
});

app.listen(3000);
```

### Flask (Python)
```python
from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/api/chat', methods=['POST'])
def chat():
    message = request.json['message']

    # Your agent logic here
    response = your_agent.run(message)

    # Minimum response format
    return jsonify({
        'response': response
    })

if __name__ == '__main__':
    app.run(port=3000)
```

### FastAPI (Python)
```python
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class ChatRequest(BaseModel):
    message: str

@app.post("/api/chat")
async def chat(req: ChatRequest):
    # Your agent logic here
    response = await your_agent.run(req.message)

    # Minimum response format
    return {"response": response}
```

---

## Level 2: Agent with Metadata (15 minutes)

### Express.js with Cost Tracking
```javascript
app.post('/api/chat', async (req, res) => {
  const { message } = req.body;
  const startTokens = /* track input tokens */;

  // Your agent logic here
  const result = await yourAgent.run(message);

  // Calculate costs
  const inputTokens = startTokens;
  const outputTokens = result.tokens;
  const cost = calculateCost(inputTokens, outputTokens);

  // Enhanced response format
  res.json({
    response: result.text,
    metadata: {
      cost: cost,
      tokens: {
        input: inputTokens,
        output: outputTokens
      },
      steps: result.toolsCalled || []  // Optional: list of tools used
    }
  });
});

function calculateCost(input, output) {
  const INPUT_PRICE = 0.01 / 1000;  // $0.01 per 1K tokens
  const OUTPUT_PRICE = 0.03 / 1000; // $0.03 per 1K tokens
  return (input * INPUT_PRICE) + (output * OUTPUT_PRICE);
}
```

### Python with OpenAI Integration
```python
import openai

@app.post("/api/chat")
async def chat(req: ChatRequest):
    # Track execution
    tools_called = []

    # Run agent with tools
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "user", "content": req.message}],
        tools=[...],  # Your tools
    )

    # Extract usage
    usage = response.usage
    cost = calculate_cost(
        usage.prompt_tokens,
        usage.completion_tokens
    )

    # Track which tools were called
    if response.choices[0].message.tool_calls:
        tools_called = [
            tc.function.name
            for tc in response.choices[0].message.tool_calls
        ]

    return {
        "response": response.choices[0].message.content,
        "metadata": {
            "cost": cost,
            "tokens": {
                "input": usage.prompt_tokens,
                "output": usage.completion_tokens,
                "cached": getattr(usage, 'cached_tokens', 0)
            },
            "steps": tools_called
        }
    }

def calculate_cost(input_tokens, output_tokens):
    # GPT-4 pricing (example)
    INPUT_PRICE = 0.03 / 1000
    OUTPUT_PRICE = 0.06 / 1000
    return (input_tokens * INPUT_PRICE) + (output_tokens * OUTPUT_PRICE)
```

---

## Level 3: Streaming Agent (30 minutes)

### Express.js with JSONL Streaming
```javascript
app.post('/api/chat', async (req, res) => {
  const { message } = req.body;

  // Set headers for streaming
  res.setHeader('Content-Type', 'application/x-ndjson');
  res.setHeader('Transfer-Encoding', 'chunked');

  const emit = (event) => {
    res.write(JSON.stringify(event) + '\n');
  };

  // Start event
  emit({ type: 'start', data: { message } });

  // Tool call
  emit({
    type: 'tool_call',
    data: {
      name: 'analyzeStock',
      args: { symbol: 'AAPL' }
    }
  });

  // Execute tool
  const toolResult = await yourAgent.tools.analyzeStock('AAPL');

  // Tool result
  emit({
    type: 'tool_result',
    data: {
      result: toolResult,
      success: true
    }
  });

  // Token usage
  emit({
    type: 'usage',
    data: {
      input_tokens: 100,
      output_tokens: 500,
      cached_tokens: 0
    }
  });

  // Final message
  emit({
    type: 'message_complete',
    data: {
      content: 'Final agent response...'
    }
  });

  res.end();
});
```

### Python FastAPI with Streaming
```python
from fastapi.responses import StreamingResponse
import json
import asyncio

@app.post("/api/chat")
async def chat_stream(req: ChatRequest):
    async def event_generator():
        # Start
        yield json.dumps({"type": "start", "data": {"message": req.message}}) + "\n"

        # Tool execution
        yield json.dumps({
            "type": "tool_call",
            "data": {"name": "analyzeStock", "args": {"symbol": "AAPL"}}
        }) + "\n"

        # Execute tool
        result = await your_agent.tools.analyze_stock("AAPL")

        yield json.dumps({
            "type": "tool_result",
            "data": {"result": result, "success": True}
        }) + "\n"

        # Usage tracking
        yield json.dumps({
            "type": "usage",
            "data": {
                "input_tokens": 100,
                "output_tokens": 500,
                "cached_tokens": 0
            }
        }) + "\n"

        # Final response
        yield json.dumps({
            "type": "message_complete",
            "data": {"content": "Final agent response..."}
        }) + "\n"

    return StreamingResponse(
        event_generator(),
        media_type="application/x-ndjson"
    )
```

---

## Testing Your Implementation

### 1. Test Basic Connectivity
```bash
curl -X POST http://localhost:3000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Test"}' | jq
```

Should see: `{"response": "..."}`

### 2. Test Metadata (Level 2)
```bash
curl -X POST http://localhost:3000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Test"}' | jq '.metadata'
```

Should see: `{"cost": 0.05, "tokens": {...}, "steps": [...]}`

### 3. Test Streaming (Level 3)
```bash
curl -X POST http://localhost:3000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Test"}' | jq -R 'fromjson? | .type'
```

Should see multiple event types: `start`, `tool_call`, `usage`, etc.

---

## Common Mistakes

### ❌ Don't: Return HTML or plain text
```javascript
res.send("Agent response");  // Wrong!
```

### ✅ Do: Always return JSON
```javascript
res.json({ response: "Agent response" });
```

---

### ❌ Don't: Forget to track ALL token usage
```javascript
// Only tracking final LLM call - missing tool calls!
const cost = calculateCost(finalResponse.tokens);
```

### ✅ Do: Sum tokens across all LLM calls
```javascript
let totalInput = 0;
let totalOutput = 0;

// Track each LLM call
for (const step of execution.steps) {
  totalInput += step.usage.input_tokens;
  totalOutput += step.usage.output_tokens;
}

const cost = calculateCost(totalInput, totalOutput);
```

---

### ❌ Don't: Emit events without newlines (streaming)
```javascript
res.write(JSON.stringify(event));  // Wrong - not JSONL!
```

### ✅ Do: Add newline after each event
```javascript
res.write(JSON.stringify(event) + '\n');  // Correct JSONL
```

---

## Next Steps

1. **Start with Level 1** - Get basic tests running (5 minutes)
2. **Add metadata** - Enable cost/token tracking (10 more minutes)
3. **Add streaming** (optional) - Full tool tracking (20 more minutes)
4. **Run tests**: `evalview run`
5. **Check results**: `.agenteval/results/TIMESTAMP.json`

## Questions?

- See `docs/BACKEND_REQUIREMENTS.md` for detailed specs
- See `tests/test-cases/` for example test cases
- File issues: https://github.com/hidai25/EvalView/issues
