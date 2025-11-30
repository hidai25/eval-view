# Reference Agent for EvalView Testing

## Overview

A simple FastAPI-based test agent with multiple tools to validate EvalView compatibility. This serves as:
1. **Validation** - Ensure EvalView core functionality works
2. **Template** - Reference implementation for testing other frameworks
3. **Debugging** - Isolated environment to test changes

## Features

### Available Tools

- **calculator** - Basic arithmetic (add, subtract, multiply, divide)
- **get_weather** - Weather lookup for major cities
- **search_web** - Web search simulation
- **convert_temperature** - Celsius/Fahrenheit conversion
- **get_stock_price** - Stock price lookup (AAPL, GOOGL, MSFT)

### Test Coverage

✅ Simple single-tool calls
✅ Multi-tool sequences
✅ Error handling
✅ Cost tracking
✅ Latency measurement

## Quick Start

### 1. Install Dependencies

```bash
cd agent-testing/reference-agent
pip install -r requirements.txt
```

### 2. Start the Agent

```bash
python agent.py
```

Expected output:
```
🚀 Starting Reference Test Agent on http://localhost:8000
📚 API docs available at http://localhost:8000/docs
🔧 Available tools: ['calculator', 'get_weather', 'search_web', 'convert_temperature', 'get_stock_price']
```

### 3. Test the Agent Manually (Optional)

```bash
# Health check
curl http://localhost:8000/health

# List tools
curl http://localhost:8000/tools

# Test execution
curl -X POST http://localhost:8000/execute \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "user", "content": "What is 5 plus 3?"}
    ]
  }'
```

### 4. Run EvalView Tests

In a new terminal:

```bash
# From project root
cd agent-testing/reference-agent
evalview run
```

## Test Cases

| Test Case | Description | Tools | Expected Score |
|-----------|-------------|-------|----------------|
| 01-simple-calculator | Basic addition | calculator | 80+ |
| 02-weather-query | Weather lookup | get_weather | 75+ |
| 03-multi-tool-sequence | Weather + temp conversion | get_weather, convert_temperature | 70+ |
| 04-error-handling | Invalid city handling | get_weather | 70+ |
| 05-stock-query | Stock price lookup | get_stock_price | 75+ |
| 06-multiplication | Multiplication | calculator | 80+ |

## Expected Results

All tests should **PASS** with:
- ✅ Correct tool calls detected
- ✅ Tool sequences validated
- ✅ Output contains expected strings
- ✅ Cost and latency within thresholds

### Sample Output

```
🧪 EvalView Test Results

✅ Simple Calculator - Addition (score: 95.0)
   Tools: calculator ✓
   Cost: $0.0005 | Latency: 45ms

✅ Weather Query - Single Tool (score: 92.5)
   Tools: get_weather ✓
   Cost: $0.001 | Latency: 52ms

✅ Multi-Tool Sequence - Weather & Conversion (score: 88.0)
   Tools: get_weather ✓, convert_temperature ✓
   Sequence: correct ✓
   Cost: $0.002 | Latency: 78ms

...

Overall: 6/6 tests passed
```

## API Reference

### POST /execute

Execute agent with user query.

**Request:**
```json
{
  "messages": [
    {"role": "user", "content": "Your query here"}
  ]
}
```

**Response:**
```json
{
  "output": "Agent response text",
  "tool_calls": [
    {
      "name": "calculator",
      "arguments": {"operation": "add", "a": 5, "b": 3},
      "result": 8
    }
  ],
  "cost": 0.0005,
  "latency": 45.2
}
```

### GET /health

Health check endpoint.

### GET /tools

List available tools and their descriptions.

## Agent Logic

The reference agent uses **simple rule-based logic** to determine tool calls:

- Weather keywords → `get_weather`
- Math keywords (add, plus, multiply, times) → `calculator`
- Stock keywords or symbols → `get_stock_price`
- Temperature conversion → `convert_temperature`
- Fallback → `search_web`

In production agents, this logic would be replaced by an LLM deciding which tools to use.

## Customization

### Adding New Tools

```python
def my_new_tool(param: str) -> Any:
    """Tool description."""
    return "result"

# Register tool
TOOLS["my_new_tool"] = my_new_tool

# Add to agent logic in simple_agent_logic()
```

### Modifying Agent Behavior

Edit `simple_agent_logic()` to change:
- Tool selection criteria
- Cost calculations
- Response formatting
- Error handling

## Troubleshooting

### Port Already in Use

```bash
# Kill process on port 8000
lsof -ti:8000 | xargs kill -9

# Or use a different port
python agent.py --port 8001
```

### EvalView Connection Error

- Ensure agent is running on http://localhost:8000
- Check `.evalview/config.yaml` has correct endpoint
- Verify firewall isn't blocking localhost

### Tests Failing

- Check agent logs for errors
- Run tests with `evalview run --verbose`
- Verify test case YAML syntax
- Check tool call names match exactly

## Next Steps

Once reference agent tests pass:

1. ✅ **Validate** - EvalView core functionality works
2. 📝 **Document** - Known issues or limitations
3. 🔄 **Template** - Use this structure for other frameworks
4. 🚀 **Test More** - Move on to LangChain, LangGraph, etc.

## Framework Testing Template

Use this structure for each framework:

```
agent-testing/{framework}/
├── README.md              # Setup instructions
├── agent.py              # Agent implementation
├── requirements.txt      # Framework dependencies
├── .evalview/
│   └── config.yaml      # Adapter configuration
└── test-cases/
    ├── 01-simple.yaml   # Basic test
    ├── 02-multi.yaml    # Complex test
    └── 03-error.yaml    # Error handling
```

## Contributing

Found issues testing with this reference agent?
1. Document the issue in AGENT_TESTING.md
2. Create a minimal reproduction
3. Open an issue with "[Testing]" prefix
