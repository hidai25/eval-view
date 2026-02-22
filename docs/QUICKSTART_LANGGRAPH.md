# Quick Start: Testing LangGraph Agents with EvalView

> This guide gets you testing your LangGraph agent with EvalView in 5 minutes. EvalView provides a dedicated LangGraph adapter with auto-detection, streaming support, and native thread tracking.

## Step 1: Start Your LangGraph Agent

```bash
# Navigate to your LangGraph project
cd /path/to/your/langgraph-agent

# Start the server (check your project's README for the exact command)
python main.py
# OR
uvicorn main:app --reload --port 8000

# Verify it's running:
curl http://localhost:8000
```

## Step 2: Connect EvalView to Your Agent

```bash
# Auto-detect and configure the endpoint
evalview connect

# Or manually specify your endpoint:
evalview connect --endpoint http://localhost:8000/api/chat
```

This will:
- ✅ Test the connection
- ✅ Auto-update `.evalview/config.yaml`
- ✅ Tell you the correct adapter type

## Step 3: Run Your Tests

```bash
# Run with verbose output to see what's happening
evalview run --verbose
```

## Step 4: Review Results

After running tests, you'll see:
- ✅ Pass/fail status for each test
- 📊 Scores, costs, and latency
- 💾 Results saved to `.evalview/results/`

View detailed report:
```bash
evalview report .evalview/results/LATEST.json --detailed
```

## Troubleshooting

### "Cannot connect to agent"
```bash
# Check if your agent is running
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "test"}'

# If not working, check:
# 1. Is the server running?
# 2. Is it on port 8000?
# 3. What's the correct endpoint path?
```

### "Wrong endpoint"
Update `.evalview/config.yaml`:
```yaml
endpoint: http://localhost:YOUR_PORT/YOUR_PATH
```

### "Tool names don't match"
Run with `--verbose` to see the actual API response, then update your test YAML files with the correct tool names.

## Next Steps

1. **Customize test cases** in `tests/test-cases/`
2. **Add more scenarios** - Copy and modify the example YAMLs
3. **Adjust thresholds** - Based on your agent's actual performance
4. **Run regularly** - Add to your CI/CD pipeline

## Example Test Case

Create `tests/test-cases/my-test.yaml`:

```yaml
name: "My Custom Test"
description: "Test my agent does X"

input:
  query: "Your test query here"
  context: {}

expected:
  tools: []  # Add tool names after seeing verbose output
  output:
    contains:
      - "expected keyword 1"
      - "expected keyword 2"
    not_contains:
      - "error"

thresholds:
  min_score: 70
  max_cost: 0.50
  max_latency: 10000
```

## Tips

- 🔍 Always use `--verbose` initially to understand API responses
- 📝 Start with loose thresholds, tighten based on actual performance
- 🧪 Test one scenario at a time when debugging
- 💰 Set `OPENAI_API_KEY` in `.env.local` for LLM-as-judge evaluation

---

**Need help?** Check [DEBUGGING.md](DEBUGGING.md) or open an issue!
