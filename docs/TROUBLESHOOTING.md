# Troubleshooting Guide

This guide covers common issues when using EvalView and how to resolve them.

## Quick Diagnostics

### Enable Verbose Output

```bash
# Run with verbose output
evalview run --verbose

# Or set environment variable
DEBUG=1 evalview run
```

### Test Adapter Connection

```bash
# Auto-detect and test adapter
evalview connect

# Validate specific endpoint
evalview validate-adapter --endpoint http://localhost:8000 --adapter http
```

---

## Common Type Errors

### `ValidationError: value is not a valid dict` (tokens)

**Symptom:** Error when adapter returns token count as integer instead of object.

**Cause:** Your adapter is returning `total_tokens=1500` (int) instead of a `TokenUsage` object.

**Solution:** EvalView v1.x+ auto-coerces integers to `TokenUsage`. If you're on an older version, update your adapter:

```python
# Before (causes error in older versions)
total_tokens = 1500

# After (always works)
from evalview.core.types import TokenUsage
total_tokens = TokenUsage(output_tokens=1500)
```

### `ValidationError: start_time / end_time`

**Symptom:** Error when passing datetime as string.

**Cause:** ExecutionTrace expects `datetime` objects, not strings.

**Solution:** EvalView v1.x+ auto-coerces ISO format strings. For older versions:

```python
from datetime import datetime

# Before (causes error in older versions)
start_time = "2025-01-15T10:30:00"

# After (always works)
start_time = datetime.fromisoformat("2025-01-15T10:30:00")
# Or
start_time = datetime.now()
```

### `StepMetrics missing latency/cost`

**Symptom:** Error creating StepTrace without metrics values.

**Cause:** StepMetrics previously required both `latency` and `cost`.

**Solution:** EvalView v1.x+ defaults these to 0.0. For older versions:

```python
# Before (causes error in older versions)
metrics = StepMetrics()

# After (always works)
metrics = StepMetrics(latency=0.0, cost=0.0)
```

---

## Connection Errors

### `Connection refused` / `ECONNREFUSED`

**Checklist:**
1. Is your agent server running?
2. Is it running on the correct port?
3. Is the endpoint URL correct in your test case or config?

```bash
# Check if server is running
curl http://localhost:8000/health

# Check what's listening on the port
lsof -i :8000
```

### `Request timed out`

**Cause:** Agent execution took longer than the configured timeout.

**Solutions:**

1. Increase timeout in test case:
   ```yaml
   adapter_config:
     timeout: 120  # seconds
   ```

2. Or in adapter initialization:
   ```python
   adapter = HTTPAdapter(endpoint="...", timeout=120.0)
   ```

**Framework-specific timeouts:**
- CrewAI: Often needs 120s+ for multi-agent workflows
- LangGraph: 30-60s typical
- OpenAI Assistants: 60-120s depending on tools

### `SSRFProtectionError: Hostname blocked`

**Cause:** EvalView's SSRF protection blocked a private/internal URL.

**Solution (development only):**
```python
adapter = HTTPAdapter(
    endpoint="http://localhost:8000",
    allow_private_urls=True  # Only for trusted dev environments!
)
```

---

## Framework-Specific Issues

### CrewAI

#### Response format varies
CrewAI can return either `tasks` or `agent_executions` format. EvalView handles both, but ensure your CrewAI version is compatible.

#### Long execution times
Multi-agent crews often take 60-120+ seconds. Set appropriate timeout:
```yaml
adapter_config:
  timeout: 120
```

#### Missing tool names
CrewAI tasks may not have explicit tool names. EvalView defaults to `"crew_task"` or `"agent_execution"`.

### LangGraph

#### Cloud API vs Self-Hosted
Different response formats - use appropriate adapter:
```yaml
# For LangGraph Cloud
adapter: langgraph
adapter_config:
  mode: cloud

# For self-hosted
adapter: langgraph
adapter_config:
  mode: standard
```

#### Streaming mode issues
If streaming fails, try standard mode:
```yaml
adapter_config:
  streaming: false
```

#### Token field name mismatches
LangGraph uses different field names depending on the underlying model:
- `input_tokens` vs `prompt_tokens`
- `output_tokens` vs `completion_tokens`

EvalView handles both automatically.

### OpenAI Assistants

#### Polling timeout
Assistants run asynchronously and need polling. Increase timeout if runs are timing out:
```yaml
adapter_config:
  timeout: 120
```

#### Missing assistant_id
```yaml
adapter_config:
  assistant_id: asst_xxxxx
```

---

## Evaluation Issues

### Score is 0 but API works

**Possible causes:**

1. **Tool mismatch:** Expected tools don't match actual tools used
   ```yaml
   expected:
     tools: ["search", "summarize"]  # Check tool names match exactly
   ```

2. **Output doesn't contain expected content:**
   ```yaml
   expected:
     output:
       contains: ["Paris"]  # Case-sensitive!
   ```

3. **LLM-as-judge failed:** Check OPENAI_API_KEY is set
   ```bash
   export OPENAI_API_KEY=sk-...
   ```

### Wrong tools detected

EvalView extracts tool names from the trace. Different frameworks expose tools differently:

- **Check actual tool names:** Run with `--verbose` to see extracted tool names
- **Use flexible matching:** Tool names are matched exactly - ensure YAML matches actual names

---

## Getting Raw API Response

For debugging, you can capture the raw response:

### Using --debug mode (v1.x+)
```bash
evalview run --debug
```

This shows:
- Raw API response JSON
- Parsed ExecutionTrace structure
- Type coercions performed

### Manual debugging
Add to your adapter:
```python
async def execute(self, query, context=None):
    response = await client.post(...)
    print(f"Raw response: {response.json()}")  # Debug
    ...
```

---

## Environment Issues

### Missing OPENAI_API_KEY

LLM-as-judge evaluation requires OpenAI API key:
```bash
export OPENAI_API_KEY=sk-...
```

### Python version issues

EvalView requires Python 3.9+. Check your version:
```bash
python --version
```

For LangGraph, Python 3.11+ may be required.

---

## Common Pitfalls

### Smaller LLM Judges Can Make Basic Errors

**Symptom:** Hallucination detector flags correct answers as wrong, or gives nonsensical feedback.

**Example:**
```
Hallucination (100% confidence):
"5 plus 7 equals 12" contradicts... the correct answer is 17.
```
*(5 + 7 = 12 is correct! The judge made a math error.)*

**Cause:** Smaller models (8B parameters) can make basic reasoning/math errors when acting as judge.

**Solution:** Use a larger model for evaluation:
```bash
# Option 1: CLI flags (recommended)
evalview run --judge-model gpt-5 --judge-provider openai
evalview run --judge-model llama-70b --judge-provider huggingface

# Option 2: Environment variables
export EVAL_MODEL=meta-llama/Llama-3.1-70B-Instruct
export EVAL_PROVIDER=huggingface

# Option 3: Unset to fall back to OpenAI
unset EVAL_PROVIDER
```

**Model quality comparison:**

| Model | Math/Logic | Speed | Cost | Best For |
|-------|------------|-------|------|----------|
| `meta-llama/Llama-3.1-8B-Instruct` | ⚠️ Can make errors | Fast | Free | Quick dev iteration |
| `meta-llama/Llama-3.1-70B-Instruct` | ✅ Reliable | Medium | Free | Production evals |
| `gpt-4o-mini` (OpenAI) | ✅ Reliable | Fast | ~$0.001/eval | Production evals |
| `gpt-4o` (OpenAI) | ✅ Best | Slow | ~$0.01/eval | Critical evals |

---

### Tool-Based Tests Require Tool-Enabled Agents

**Symptom:** Tests fail with "Missing tools: get_weather, calculator" even though the agent gave a correct answer.

**Example:**
```
Query: What's the weather in Tokyo?

Response: I don't have access to real-time weather information...

❌ Missing tools: get_weather
```

**Cause:** Your test expects the agent to call specific tools, but the agent doesn't have those tools configured.

**The key insight:** EvalView tests *how* an agent works, not just *what* it outputs. If you expect tool calls, the agent must have tools.

**Solutions:**

1. **Use an agent with tools:**
   ```yaml
   # OpenAI Assistants with code_interpreter
   adapter: openai-assistants

   # LangGraph with custom tools
   adapter: langgraph
   endpoint: http://localhost:2024
   ```

2. **Or remove tool expectations from test:**
   ```yaml
   expected:
     # Remove this if agent doesn't have tools:
     # tools: ["get_weather"]
     output:
       contains: ["weather", "Tokyo"]
   ```

3. **Or test output quality only:**
   ```yaml
   expected:
     output:
       min_score: 80
   # No tool expectations
   ```

---

### Adapter Capabilities

Not all adapters support the same features. Know what each can do:

| Adapter | Has Tools | Streaming | Use Case |
|---------|-----------|-----------|----------|
| `openai-assistants` | ✅ code_interpreter, file_search, custom | ❌ | OpenAI Assistants API |
| `anthropic` | ❌ Plain Claude | ❌ | Simple Claude API calls |
| `langgraph` | ✅ Custom tools | ✅ | LangGraph agents |
| `crewai` | ✅ Agent tools | ❌ | CrewAI multi-agent |
| `http` | ⚠️ Depends on backend | ⚠️ | Any REST API |
| `huggingface` | ❌ Gradio only | ❌ | HF Spaces chatbots |

**Matching tests to adapters:**

```yaml
# ✅ Good: Testing OpenAI Assistant with tool expectations
adapter: openai-assistants
expected:
  tools: ["code_interpreter"]

# ❌ Bad: Testing plain Anthropic with tool expectations
adapter: anthropic
expected:
  tools: ["calculator"]  # Will always fail - no tools!

# ✅ Good: Testing Anthropic with output-only expectations
adapter: anthropic
expected:
  output:
    contains: ["12"]
    min_score: 80
```

---

## Still Stuck?

1. **Check the examples:** See `examples/` directory for working configurations
2. **Enable maximum verbosity:** `DEBUG=1 evalview run --verbose`
3. **Open an issue:** https://github.com/hidai25/eval-view/issues

When reporting issues, include:
- EvalView version
- Python version
- Framework and version (LangGraph, CrewAI, etc.)
- Full error message
- Test case YAML (sanitized)
- Raw API response (if possible)
