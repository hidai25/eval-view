# AgentEval Adapters

Adapters connect AgentEval to your AI agent's API. AgentEval includes adapters for common patterns and makes it easy to build custom ones.

## Built-in Adapters

### HTTP Adapter

For standard REST APIs that return a complete JSON response.

**Use when:**
- Your agent returns a full response in one request
- No streaming involved
- Response contains all steps/tools in a single payload

**Configuration:**
```yaml
# .agenteval/config.yaml
adapter: http
endpoint: http://localhost:3000/api/agent
timeout: 30.0
headers:
  Authorization: Bearer your-api-key
  Content-Type: application/json
```

**Expected Response Format:**
```json
{
  "session_id": "session-123",
  "output": "Final agent response",
  "steps": [
    {
      "id": "step-1",
      "name": "Tool name",
      "tool": "tool_identifier",
      "parameters": {"arg": "value"},
      "output": {"result": "data"},
      "success": true,
      "latency": 234,
      "cost": 0.001,
      "tokens": 150
    }
  ],
  "cost": 0.025,
  "tokens": 1250
}
```

### Streaming Adapter

For JSONL (JSON Lines) streaming APIs.

**Use when:**
- Your agent streams responses line-by-line
- Each line is a JSON object with event information
- Real-time updates during execution

**Configuration:**
```yaml
# .agenteval/config.yaml
adapter: streaming  # or 'jsonl' or 'tapescope'
endpoint: http://localhost:3000/api/chat
timeout: 60.0
headers:
  Content-Type: application/json
```

**Expected Event Format:**

The streaming adapter recognizes these event types:

```jsonl
{"type": "tool_call", "data": {"name": "search_web", "args": {"query": "..."}}}
{"type": "tool_result", "data": {"result": {"status": "success", "data": [...]}}}
{"type": "token", "data": {"token": "Hello"}}
{"type": "token", "data": {"token": " world"}}
{"type": "final_message", "data": {"text": "Complete response"}}
{"type": "error", "error": "Error message"}
```

**Supported Event Types:**

| Type | Purpose | Action |
|------|---------|--------|
| `tool_call` | Tool is being executed | Creates a new step trace |
| `tool_result` | Tool finished | Updates last step with result |
| `final_message` | Complete response | Sets final output |
| `token` | Streaming token (SSE) | Appends to output |
| `error` | Error occurred | Captures error message |
| `start`, `status`, `thinking` | Informational | Logged only |

**Fallback Behavior:**

If no recognized events are found, the adapter:
1. Treats each line as plain text
2. Accumulates all text as final output
3. Still captures timing and basic metrics

## Custom Adapters

Build a custom adapter for your specific agent implementation.

### Basic Template

```python
# agent_eval/adapters/my_adapter.py
from datetime import datetime
from typing import Any, Optional, Dict
from agent_eval.adapters.base import AgentAdapter
from agent_eval.core.types import (
    ExecutionTrace,
    StepTrace,
    StepMetrics,
    ExecutionMetrics,
)

class MyCustomAdapter(AgentAdapter):
    """Adapter for my custom agent."""

    def __init__(
        self,
        endpoint: str,
        headers: Optional[Dict[str, str]] = None,
        timeout: float = 30.0,
    ):
        self.endpoint = endpoint
        self.headers = headers or {}
        self.timeout = timeout

    @property
    def name(self) -> str:
        return "my-adapter"

    async def execute(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None
    ) -> ExecutionTrace:
        """Execute agent and capture trace."""
        start_time = datetime.now()

        # TODO: Call your agent API
        response = await self._call_agent(query, context)

        # TODO: Parse response and extract steps
        steps = self._parse_steps(response)

        # TODO: Get final output
        final_output = response.get("output", "")

        end_time = datetime.now()
        total_latency = (end_time - start_time).total_seconds() * 1000

        return ExecutionTrace(
            session_id=response.get("session_id", "custom-session"),
            start_time=start_time,
            end_time=end_time,
            steps=steps,
            final_output=final_output,
            metrics=ExecutionMetrics(
                total_cost=sum(s.metrics.cost for s in steps),
                total_latency=total_latency,
                total_tokens=sum(s.metrics.tokens or 0 for s in steps),
            ),
        )

    async def _call_agent(self, query: str, context: dict) -> dict:
        """Make API call to your agent."""
        import httpx

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(
                self.endpoint,
                json={"query": query, **context},
                headers=self.headers,
            )
            response.raise_for_status()
            return response.json()

    def _parse_steps(self, response: dict) -> list:
        """Convert your response format to StepTrace objects."""
        steps = []

        for step_data in response.get("steps", []):
            step = StepTrace(
                step_id=step_data["id"],
                step_name=step_data["name"],
                tool_name=step_data.get("tool"),
                parameters=step_data.get("parameters", {}),
                output=step_data.get("output"),
                success=step_data.get("success", True),
                error=step_data.get("error"),
                metrics=StepMetrics(
                    latency=step_data.get("latency", 0.0),
                    cost=step_data.get("cost", 0.0),
                    tokens=step_data.get("tokens"),
                ),
            )
            steps.append(step)

        return steps

    async def health_check(self) -> bool:
        """Check if agent endpoint is reachable."""
        try:
            import httpx
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(self.endpoint)
                return response.status_code == 200
        except Exception:
            return False
```

### Register Your Adapter

```python
# agent_eval/cli.py

from agent_eval.adapters.my_adapter import MyCustomAdapter

# In _run_async function:
adapter_type = config.get("adapter", "http")
if adapter_type == "my-adapter":
    adapter = MyCustomAdapter(
        endpoint=config["endpoint"],
        headers=config.get("headers", {}),
        timeout=config.get("timeout", 30.0),
    )
```

## Adapter Examples

### LangServe Streaming

```python
class LangServeAdapter(AgentAdapter):
    """Adapter for LangServe streaming endpoints."""

    async def execute(self, query: str, context=None) -> ExecutionTrace:
        start_time = datetime.now()
        steps = []
        final_output = ""

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            async with client.stream(
                "POST",
                f"{self.endpoint}/stream",
                json={"input": query},
            ) as response:
                async for line in response.aiter_lines():
                    if line.startswith("data: "):
                        data = json.loads(line[6:])
                        if "output" in data:
                            final_output = data["output"]

        end_time = datetime.now()
        # ... build ExecutionTrace
```

### OpenAI Assistants API

```python
class OpenAIAssistantsAdapter(AgentAdapter):
    """Adapter for OpenAI Assistants API."""

    async def execute(self, query: str, context=None) -> ExecutionTrace:
        import openai

        client = openai.AsyncOpenAI()
        start_time = datetime.now()

        # Create thread and run
        thread = await client.beta.threads.create()
        await client.beta.threads.messages.create(
            thread_id=thread.id,
            role="user",
            content=query
        )

        run = await client.beta.threads.runs.create(
            thread_id=thread.id,
            assistant_id=context.get("assistant_id")
        )

        # Poll for completion
        while run.status in ["queued", "in_progress"]:
            await asyncio.sleep(0.5)
            run = await client.beta.threads.runs.retrieve(
                thread_id=thread.id,
                run_id=run.id
            )

        # Extract steps
        steps = []
        for step in run.steps:
            if step.type == "tool_calls":
                for tool_call in step.step_details.tool_calls:
                    steps.append(StepTrace(
                        step_id=tool_call.id,
                        tool_name=tool_call.function.name,
                        parameters=json.loads(tool_call.function.arguments),
                        # ... more fields
                    ))

        # Get final message
        messages = await client.beta.threads.messages.list(thread_id=thread.id)
        final_output = messages.data[0].content[0].text.value

        end_time = datetime.now()
        # ... build ExecutionTrace
```

### CrewAI

```python
class CrewAIAdapter(AgentAdapter):
    """Adapter for CrewAI agents."""

    async def execute(self, query: str, context=None) -> ExecutionTrace:
        # CrewAI typically runs synchronously, wrap in thread
        from concurrent.futures import ThreadPoolExecutor
        import json

        def run_crew():
            # Import your crew
            from my_crew import MyCrew

            crew = MyCrew()
            result = crew.kickoff(inputs={"query": query})
            return result

        start_time = datetime.now()

        with ThreadPoolExecutor() as executor:
            future = executor.submit(run_crew)
            result = future.result(timeout=self.timeout)

        end_time = datetime.now()

        # Parse CrewAI result
        # ... convert to ExecutionTrace
```

## Testing Your Adapter

```python
# test_my_adapter.py
import asyncio
from agent_eval.adapters.my_adapter import MyCustomAdapter

async def test_adapter():
    adapter = MyCustomAdapter(
        endpoint="http://localhost:3000/api/agent"
    )

    # Test execution
    trace = await adapter.execute("Test query")

    print(f"Session: {trace.session_id}")
    print(f"Steps: {len(trace.steps)}")
    print(f"Output: {trace.final_output}")
    print(f"Latency: {trace.metrics.total_latency}ms")

asyncio.run(test_adapter())
```

## Best Practices

1. **Always capture timing** - Record start/end times accurately
2. **Handle errors gracefully** - Catch exceptions and log them
3. **Set reasonable timeouts** - Don't wait forever for responses
4. **Validate responses** - Check for required fields before parsing
5. **Log verbosely** - Use logger for debugging (respects DEBUG env var)
6. **Test with real data** - Use actual agent responses during development
7. **Document event formats** - Explain what your adapter expects

## Need Help?

- See `agent_eval/adapters/http_adapter.py` for simple example
- See `agent_eval/adapters/tapescope_adapter.py` for streaming example
- Open an issue on GitHub with your use case
- Check [DEBUGGING.md](../DEBUGGING.md) for troubleshooting

## Contributing Adapters

Have an adapter for a popular framework? We'd love to include it!

1. Create adapter in `agent_eval/adapters/your_adapter.py`
2. Add tests in `tests/adapters/test_your_adapter.py`
3. Document in this file
4. Submit PR with examples

Popular frameworks we'd love adapters for:
- AutoGPT
- BabyAGI
- SuperAGI
- LangGraph
- Haystack
