# Testing LangGraph Example with EvalView — Step-by-Step Setup Guide

> Step-by-step guide to test the [LangGraph example](https://github.com/langchain-ai/langgraph-example) with EvalView. Covers setup, test case creation, adapter configuration, and debugging.

---

## Step 1: Set Up LangGraph Example

```bash
# Clone the LangGraph example
cd ~/Downloads  # or wherever you keep projects
git clone https://github.com/langchain-ai/langgraph-example.git
cd langgraph-example

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env  # If it exists

# Add your API keys to .env
# You'll need:
# - OPENAI_API_KEY
# - ANTHROPIC_API_KEY (if using Claude)
# - TAVILY_API_KEY (for web search, if needed)
```

## Step 2: Run the LangGraph Agent

The LangGraph example typically runs as a FastAPI server. Start it:

```bash
# Check the README for the exact command, but typically:
python main.py
# Or:
uvicorn main:app --reload --port 8000

# The agent should now be running at http://localhost:8000
```

**Note:** Check the repository's README for the exact startup command.

## Step 3: Test the Agent API Manually

Before using EvalView, verify the agent works:

```bash
# Test with curl
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the weather in San Francisco?"}'

# Or use their web UI if available
open http://localhost:8000
```

## Step 4: Understand the API Format

Look at the response format. You need to know:
1. **Endpoint URL** - e.g., `http://localhost:8000/api/chat`
2. **Request format** - What fields does it expect? (`query`, `message`, etc.)
3. **Response format** - Does it return `steps`, `output`, `tools_used`?
4. **Is it streaming?** - Does it return JSONL stream or complete JSON?

Example response you might see:
```json
{
  "output": "The weather in SF is...",
  "steps": [
    {
      "tool": "tavily_search",
      "parameters": {"query": "weather san francisco"},
      "output": {...}
    }
  ]
}
```

## Step 5: Set Up EvalView

In a new terminal (keep LangGraph running):

```bash
# Navigate to EvalView directory
cd ~/Downloads/EvalView
source venv/bin/activate

# Initialize EvalView for this agent
evalview init --interactive
```

During setup:
1. **API type**: Choose "Standard REST API" or "Streaming" based on what you found in Step 4
2. **Endpoint**: `http://localhost:8000/api/chat` (or whatever you found)
3. **Model**: Choose the model LangGraph is using (check their .env or code)
4. **Pricing**: Accept defaults or customize

## Step 6: Create a Test Case

Create `tests/test-cases/langgraph-weather.yaml`:

```yaml
name: "LangGraph Weather Test"
description: "Test agent's ability to fetch weather data"

input:
  query: "What is the weather in San Francisco?"
  context: {}

expected:
  tools:
    - tavily_search  # Or whatever tool they use
  tool_sequence:
    - tavily_search
  output:
    contains:
      - "San Francisco"
      - "weather"
    not_contains:
      - "error"
      - "failed"

thresholds:
  min_score: 70
  max_cost: 0.50
  max_latency: 10000
```

**Note:** You'll need to adjust:
- `tools` - Look at the API response to see what tool names they use
- `output.contains` - What should be in the response?

## Step 7: Configure the Adapter (If Needed)

If the LangGraph API format doesn't match EvalView's expectations, edit `.evalview/config.yaml`:

```yaml
adapter: http
endpoint: http://localhost:8000/api/chat
timeout: 30.0
headers:
  Content-Type: application/json

model:
  name: gpt-4o-mini
```

**If they use streaming JSONL:**
```yaml
adapter: streaming
endpoint: http://localhost:8000/api/chat/stream
timeout: 60.0
```

## Step 8: Run EvalView Tests

```bash
# Set your OpenAI API key (for LLM-as-judge)
export OPENAI_API_KEY=your-openai-key

# Run tests with verbose output to see what's happening
evalview run --verbose

# Or run specific test
evalview run --pattern "langgraph-*.yaml" --verbose
```

## Step 9: Debug if Needed

If tests fail, use verbose mode to see the API response:

```bash
evalview run --verbose
```

Common issues:

### Issue: "No response from agent"
**Fix:** Check that LangGraph is running and the endpoint URL is correct.

```bash
# Test manually
curl http://localhost:8000/api/chat -d '{"query": "test"}'
```

### Issue: "Tool names don't match"
**Fix:** Look at the actual API response in verbose output, update your test case with the correct tool names.

### Issue: "Response format not recognized"
**Fix:** You may need to create a custom adapter. See below.

## Step 10: Create Custom Adapter (Advanced)

If the LangGraph API format is very different, create a custom adapter:

```python
# evalview/adapters/langgraph_adapter.py
from evalview.adapters.http_adapter import HTTPAdapter
from evalview.core.types import ExecutionTrace, StepTrace
from typing import Dict, Any

class LangGraphAdapter(HTTPAdapter):
    """Custom adapter for LangGraph example."""

    def _parse_response(
        self, data: Dict[str, Any], start_time, end_time
    ) -> ExecutionTrace:
        # Parse LangGraph-specific response format
        # Example: if they use different field names
        steps = []
        for action in data.get("actions", []):
            steps.append(StepTrace(
                step_id=action["id"],
                tool_name=action["tool"],
                parameters=action["input"],
                output=action["result"],
            ))

        return ExecutionTrace(
            session_id=data.get("thread_id", "session-1"),
            steps=steps,
            final_output=data.get("final_answer", ""),
            # ...
        )
```

Then update `.evalview/config.yaml`:
```yaml
adapter: langgraph  # Use your custom adapter
```

And register it in `evalview/cli.py`:
```python
from evalview.adapters.langgraph_adapter import LangGraphAdapter

# In the run command:
if adapter_type == "langgraph":
    adapter = LangGraphAdapter(...)
```

## Example Test Cases

Here are some example test cases for common LangGraph patterns:

### Web Search Test
```yaml
name: "Web Search Test"
input:
  query: "What are the latest news about AI?"
expected:
  tools: [tavily_search]
  output:
    contains: ["AI", "news"]
thresholds:
  min_score: 75
  max_cost: 0.30
  max_latency: 8000
```

### Multi-Step Reasoning Test
```yaml
name: "Multi-Step Reasoning"
input:
  query: "Compare the populations of NYC and LA, then tell me the difference"
expected:
  tools: [web_search, calculator]
  tool_sequence:
    - web_search
    - web_search
    - calculator
  output:
    contains: ["difference", "million"]
thresholds:
  min_score: 80
  max_cost: 0.50
  max_latency: 15000
```

## Tips

1. **Start Simple**: Create one basic test case first, get it working, then add more

2. **Use Verbose Mode**: Always run with `--verbose` initially to understand the API responses

3. **Check Their Tests**: Look at LangGraph example's own tests to understand what the agent can do

4. **Iterate**: Start with loose thresholds, then tighten them as you understand performance

5. **Read Their Docs**: Check their README for API documentation

## Troubleshooting

**LangGraph won't start:**
- Check you have all required API keys in `.env`
- Check Python version (they might require 3.10+)
- Look at their logs for errors

**EvalView can't connect:**
- Verify LangGraph is running: `curl http://localhost:8000`
- Check firewall/network settings
- Try `http://127.0.0.1:8000` instead of `localhost`

**Tests always fail:**
- Run with `--verbose` to see actual vs expected
- Check if API response format matches your test case
- Verify tool names are correct
- Check token usage isn't exceeding cost thresholds

## Next Steps

Once you have basic tests working:

1. **Add More Test Cases** - Cover different scenarios
2. **Set Up CI** - Run tests on every commit
3. **Track Performance** - Monitor cost/latency trends
4. **Refine Thresholds** - Based on actual performance data

---

**Need help?** Open an issue or check the [DEBUGGING.md](DEBUGGING.md) guide.
