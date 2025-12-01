# Anthropic Examples

Test Anthropic Claude models and agents with EvalView.

## Two Ways to Test

| Example | Command | Use Case |
|---------|---------|----------|
| **Direct API** | `evalview run examples/anthropic` | Test Claude API with tool use |
| **Claude Agent SDK** | `evalview run examples/anthropic/claude-agent-sdk` | Test agents built with Claude Agent SDK |

---

## Option 1: Direct Anthropic API

Test Claude models directly using the Anthropic API with tool definitions.

### Quick Start

```bash
# 1. Setup (first time only)
cd /path/to/EvalView
python3 -m venv venv
source venv/bin/activate
pip install -e .
pip install anthropic

# 2. Set your API key
export ANTHROPIC_API_KEY=your-api-key

# 3. Run the test
evalview run examples/anthropic
```

### What It Tests

- Claude's ability to use tools (get_weather, convert_temperature)
- Response quality and accuracy
- Hallucination detection
- Cost and latency tracking

---

## Option 2: Claude Agent SDK

Test agents built with the [Claude Agent SDK](https://github.com/anthropics/claude-agent-sdk-python) - the same infrastructure that powers Claude Code.

### Quick Start

```bash
# 1. Setup
cd /path/to/EvalView
source venv/bin/activate
pip install claude-agent-sdk flask

# 2. Start the agent server (Terminal 1)
cd examples/anthropic/claude-agent-sdk
python server.py

# 3. Run tests (Terminal 2)
cd /path/to/EvalView
evalview run examples/anthropic/claude-agent-sdk
```

### Files

```
claude-agent-sdk/
├── agent.py          # Your agent with custom tools
├── server.py         # HTTP wrapper for EvalView
├── test-case.yaml    # Test definitions
└── .evalview/
    └── config.yaml   # EvalView config
```

### Custom Tools Example

```python
from claude_agent_sdk import custom_tool

@custom_tool
def get_weather(city: str) -> str:
    """Get weather for a city."""
    return f"Weather in {city}: 72°F, Sunny"

@custom_tool
def calculate(expression: str) -> str:
    """Evaluate a math expression."""
    return f"{expression} = {eval(expression)}"
```

---

## Environment Variables

```bash
# Required
export ANTHROPIC_API_KEY=your-api-key

# Optional: Choose LLM-as-judge provider
export EVAL_PROVIDER=anthropic  # or openai, gemini, grok
```

## Supported Models

| Model | API ID | Input/MTok | Output/MTok |
|-------|--------|------------|-------------|
| **Sonnet 4.5** | `claude-sonnet-4-5-20250929` | $3 | $15 |
| **Haiku 4.5** | `claude-haiku-4-5-20251001` | $1 | $5 |
| **Opus 4.5** | `claude-opus-4-5-20251101` | $5 | $25 |

## Links

- [Anthropic Docs](https://docs.anthropic.com/)
- [Tool Use Guide](https://docs.anthropic.com/en/docs/build-with-claude/tool-use)
- [Claude Agent SDK](https://github.com/anthropics/claude-agent-sdk-python)
- [Claude Agent SDK Demos](https://github.com/anthropics/claude-agent-sdk-demos)
