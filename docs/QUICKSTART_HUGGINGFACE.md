# Quick Start: Testing with HuggingFace + EvalView (Free, Open Source)

> Use EvalView with HuggingFace for free, fully open-source AI agent testing — zero OpenAI dependency. Test any agent and use Llama as a free LLM-as-judge.

## Key Concept: Agent vs Judge

EvalView has two independent components:

| Component | What it does | Config |
|-----------|--------------|--------|
| **Agent Adapter** | The AI agent you're testing | `adapter:` in config.yaml |
| **LLM-as-Judge** | Scores the agent's outputs | `EVAL_PROVIDER` env var |

**You can mix and match!** Test any agent (OpenAI, Claude, LangGraph, etc.) while using Llama as the free judge.

---

This guide covers:
1. **Using Llama as LLM-as-judge** (free evaluation, no OpenAI needed)
2. **Testing HuggingFace Spaces agents** (Gradio-based agents)

---

## Prerequisites

Get your free HuggingFace token:
1. Go to https://huggingface.co/settings/tokens
2. Create a new token with "Read" permissions
3. Set it in your environment:

```bash
export HF_TOKEN="hf_your_token_here"
```

---

## Option 1: Use Llama as LLM-as-Judge

Skip OpenAI entirely - use Llama to evaluate your agent's outputs.

### Step 1: Set HuggingFace Token

```bash
export HF_TOKEN="hf_your_token_here"
```

### Step 2: Run Your Tests

**Option A: CLI flags (recommended)**
```bash
# Use Llama as judge with simple shortcut
evalview run --judge-model llama-70b --judge-provider huggingface

# Or use 8B for faster iteration
evalview run --judge-model llama --judge-provider huggingface
```

**Option B: Environment variables**
```bash
export EVAL_PROVIDER=huggingface
export EVAL_MODEL=meta-llama/Llama-3.1-70B-Instruct
evalview run
```

### Works With ANY Agent

The judge (Llama) is **independent** from the agent you're testing. You can use HuggingFace/Llama to evaluate any agent:

```yaml
# .evalview/config.yaml - Test OpenAI Assistants, judged by Llama
adapter: openai-assistants
model:
  name: gpt-4o

# .evalview/config.yaml - Test Anthropic Claude, judged by Llama
adapter: anthropic
model:
  name: claude-sonnet-4-5-20250929

# .evalview/config.yaml - Test LangGraph agent, judged by Llama
adapter: langgraph
endpoint: http://localhost:2024

# .evalview/config.yaml - Test any HTTP API, judged by Llama
adapter: http
endpoint: http://localhost:8000/api/agent
```

Just set `EVAL_PROVIDER=huggingface` in `.env.local` and your agent's outputs will be scored by Llama - **regardless of which agent you're testing**.

### Available Models

| Model | Speed | Quality | Best For |
|-------|-------|---------|----------|
| `meta-llama/Llama-3.1-8B-Instruct` | Fast | Good | Development, CI |
| `meta-llama/Llama-3.1-70B-Instruct` | Medium | Better | Production evals |
| `mistralai/Mixtral-8x7B-Instruct-v0.1` | Fast | Good | Alternative |

---

## Option 2: Test a HuggingFace Spaces Agent

Test any Gradio-based agent hosted on HuggingFace Spaces.

### Step 1: Find Your Space URL

Your Space URL can be in any of these formats:
- `username/my-agent`
- `https://huggingface.co/spaces/username/my-agent`
- `https://username-my-agent.hf.space`

### Step 2: Connect EvalView

```bash
# Connect to your Space (auto-detects everything)
evalview connect --adapter huggingface --endpoint username/my-agent

# Or use the full URL
evalview connect --adapter hf --endpoint https://huggingface.co/spaces/username/my-agent
```

### Step 3: Create a Test Case

Create `tests/test-cases/my-hf-agent.yaml`:

```yaml
name: "HuggingFace Agent Test"
description: "Test my Gradio chatbot"

input:
  query: "What is the capital of France?"

expected:
  output:
    contains:
      - "Paris"
    not_contains:
      - "error"
      - "sorry"

thresholds:
  min_score: 70
  max_latency: 30000  # 30s (Spaces can be slow on cold start)
```

### Step 4: Run Tests

```bash
evalview run --verbose
```

---

## Full Example: 100% Open Source Stack

Test a HuggingFace agent using Llama as the judge:

```bash
# 1. Set up environment
export HF_TOKEN="hf_your_token"
export EVAL_PROVIDER=huggingface

# 2. Initialize project
evalview init

# 3. Connect to a public chatbot Space (example)
evalview connect --adapter hf --endpoint HuggingFaceH4/zephyr-chat

# 4. Run tests
evalview run --verbose
```

**Cost: $0** - Uses HF free tier for both agent and evaluation.

---

## Configuration Reference

### config.yaml

```yaml
# .evalview/config.yaml
adapter: huggingface
endpoint: username/my-space

# Optional settings
timeout: 120  # Longer timeout for AI inference
verbose: true

# HuggingFace-specific
hf_token: ${HF_TOKEN}  # Uses environment variable
function_name: chat    # Auto-detected if not specified
```

### Environment Variables

| Variable | Description | Example |
|----------|-------------|---------|
| `HF_TOKEN` | HuggingFace API token | `hf_abc123...` |
| `EVAL_PROVIDER` | Force evaluation provider | `huggingface` |
| `EVAL_MODEL` | Override evaluation model | `meta-llama/Llama-3.1-70B-Instruct` |

---

## Handling Sleeping Spaces

Free-tier Spaces sleep after inactivity. EvalView handles this automatically, but the first request may take 30-60 seconds.

Tips:
- Set `max_latency: 60000` in test thresholds for cold starts
- Use `--verbose` to see wake-up progress
- Consider upgrading to a paid Space for production testing

---

## Troubleshooting

### "Not a Gradio Space or API not enabled"

```bash
# Check the Space is a Gradio app
curl https://username-space.hf.space/gradio_api/info

# If 404, the Space either:
# - Isn't using Gradio
# - Has API disabled
# - Is still sleeping (wait and retry)
```

### "No API endpoints found"

The Gradio app doesn't expose any functions. Check with the Space owner or use a different Space.

### "401 Unauthorized"

```bash
# Make sure HF_TOKEN is set
echo $HF_TOKEN

# Verify token works
curl -H "Authorization: Bearer $HF_TOKEN" \
  https://huggingface.co/api/whoami
```

### "Connection timeout"

Space is sleeping or slow. Try:
```bash
# Increase timeout
evalview run --timeout 120

# Or wake the Space manually first
curl https://username-space.hf.space
sleep 30
evalview run
```

---

## Popular Public Spaces to Test Against

| Space | Type | URL |
|-------|------|-----|
| HuggingFaceH4/zephyr-chat | Chatbot | `HuggingFaceH4/zephyr-chat` |
| microsoft/DialoGPT | Conversational | `microsoft/DialoGPT-large` |
| facebook/blenderbot | Chatbot | `facebook/blenderbot-400M-distill` |

---

## Next Steps

1. **Write more tests** - Add scenarios to `tests/test-cases/`
2. **Add to CI** - Run `evalview run` in GitHub Actions
3. **Deploy your own agent** - Create a Gradio Space and test it
4. **Compare models** - Switch `EVAL_MODEL` to compare judge quality

---

**Need help?** Check [TROUBLESHOOTING.md](TROUBLESHOOTING.md) or [open an issue](https://github.com/hidai25/eval-view/issues)!

---

## Related Documentation

- [Framework Support](FRAMEWORK_SUPPORT.md) — All supported frameworks including HuggingFace
- [Adapters](ADAPTERS.md) — HuggingFace adapter configuration
- [YAML Schema](YAML_SCHEMA.md) — Test case format reference
- [Evaluation Metrics](EVALUATION_METRICS.md) — How scores are calculated
- [Troubleshooting](TROUBLESHOOTING.md) — Common issues and fixes
