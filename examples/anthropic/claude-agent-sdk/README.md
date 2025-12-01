# Claude Agent SDK Example

Test agents built with the [Claude Agent SDK](https://github.com/anthropics/claude-agent-sdk-python).

## Quick Start

```bash
# 1. Install dependencies
cd /path/to/EvalView
source venv/bin/activate
pip install -e .
pip install claude-agent-sdk flask

# 2. Set your API key
export ANTHROPIC_API_KEY=your-api-key

# 3. Start the agent server (in one terminal)
cd examples/claude-agent-sdk
python server.py

# 4. Run tests (in another terminal)
cd /path/to/EvalView
source venv/bin/activate
evalview run examples/claude-agent-sdk
```

## Files

- `agent.py` - Example agent with custom tools (weather, temperature conversion)
- `server.py` - HTTP wrapper for EvalView testing
- `test-case.yaml` - Test case definition
- `.evalview/config.yaml` - EvalView configuration

## Custom Tools

The example agent includes these tools:

```python
@custom_tool
def get_weather(city: str) -> str:
    """Get current weather for a city."""
    ...

@custom_tool
def convert_temperature(value: float, from_unit: str, to_unit: str) -> str:
    """Convert temperature between Celsius and Fahrenheit."""
    ...

@custom_tool
def calculate(expression: str) -> str:
    """Evaluate a mathematical expression."""
    ...
```

## Testing Your Own Agent

1. Replace `agent.py` with your agent code
2. Update `server.py` to call your agent
3. Modify `test-case.yaml` with your expected behavior
4. Run `evalview run examples/claude-agent-sdk`

## Links

- [Claude Agent SDK Python](https://github.com/anthropics/claude-agent-sdk-python)
- [Claude Agent SDK Demos](https://github.com/anthropics/claude-agent-sdk-demos)
- [Building Agents Guide](https://www.anthropic.com/engineering/building-agents-with-the-claude-agent-sdk)
