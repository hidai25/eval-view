# Debugging EvalView — How to Debug Failing AI Agent Tests

> This guide helps you troubleshoot issues when running EvalView tests, including "No response" errors, database issues, timeouts, and tool name mismatches.

## Common Issues

### 1. Tests Failing with "No response"

**Symptom**: Test results show `final_output: "No response"` and score 0

**Cause**: The adapter isn't parsing the API response correctly

**Solution**: Run with verbose mode to see what the API is actually returning:

```bash
# Option 1: Use --verbose flag
evalview run --verbose

# Option 2: Set DEBUG environment variable
DEBUG=1 evalview run
```

This will show you:
- 🚀 Request being sent
- 📤 Request payload
- ✅ Response status
- 📥 Each line of the streaming response
- 🔍 Parsed event types
- ⚠️ Unhandled events or errors

### 2. Database Foreign Key Errors

**Symptom**: Error message about `Foreign key constraint violated` for `userId`

**Cause**: The test user ID doesn't exist in your database

**Solutions**:

**Option A**: Create a test user in your database
```sql
-- For PostgreSQL (adjust for your database)
INSERT INTO users (id, email, name)
VALUES ('test-user', 'test@example.com', 'Test User');
```

**Option B**: Use a real user ID from your database
Edit your test cases to use a valid user:
```yaml
# tests/test-cases/example.yaml
input:
  query: "Your query here"
  context:
    userId: "your-real-user-id"  # Use an actual user ID
```

**Option C**: Update your API to handle non-existent test users gracefully

### 3. Tests Timing Out

**Symptom**: Tests exceed latency threshold (e.g., 76s instead of 10s)

**Possible causes**:
- API endpoint is slow
- Database queries are slow
- External API calls are timing out

**Solutions**:
1. Increase timeout in config:
```yaml
# .evalview/config.yaml
timeout: 120.0  # Increase from 60.0
```

2. Increase latency threshold in test case:
```yaml
# tests/test-cases/example.yaml
thresholds:
  max_latency: 120000  # 120 seconds instead of 10
```

3. Optimize your API endpoints

### 4. Wrong Tool Names

**Symptom**: Test shows "missing tools" even though your API is working

**Cause**: Your API uses different tool names than expected in the test

**Solution**: Update your test case to match your actual tool names. Run with `--verbose` to see what tools are actually being called.

## Understanding the Adapter

### TapeScopeAdapter Event Types

The adapter recognizes these event types from JSONL streaming:

| Event Type | Description | Action |
|------------|-------------|--------|
| `tool_call` | Tool is being executed | Creates a new step |
| `tool_result` | Tool execution completed | Updates last step with result |
| `final_message` | Final response from agent | Sets final_output |
| `token` | Streaming token (SSE) | Appends to final_output |
| `error` | Error occurred | Sets error message |
| `start`, `status`, `thinking`, `step_start`, `step_complete` | Informational | Logged only |

### Adding Support for Your Event Format

If your API uses different event types, you can:

1. **Check logs**: Run with `--verbose` to see what event types your API sends

2. **Extend the adapter**: Edit `evalview/adapters/tapescope_adapter.py` to handle your event types

3. **Use HTTPAdapter**: If your API returns a simple JSON response (non-streaming), use the HTTPAdapter instead:

```yaml
# .evalview/config.yaml
adapter: http  # Instead of tapescope
endpoint: http://localhost:3000/api/your-endpoint
```

## Verbose Output Example

```
🔍 Verbose mode enabled

Running test cases...

2025-11-19 22:30:00 - tapescope_adapter - INFO - 🚀 Executing request: Analyze AAPL stock performance...
2025-11-19 22:30:00 - tapescope_adapter - DEBUG - 📤 Payload: {
  "message": "Analyze AAPL stock performance",
  "route": "orchestrator",
  "userId": "test-user"
}
2025-11-19 22:30:00 - tapescope_adapter - INFO - ✅ Response status: 200
2025-11-19 22:30:01 - tapescope_adapter - DEBUG - 📥 Line 1: {"type":"start","data":{"planId":"plan_123"}}
2025-11-19 22:30:01 - tapescope_adapter - DEBUG - 🔍 Event type: 'start'
2025-11-19 22:30:01 - tapescope_adapter - DEBUG - ℹ️ Info event: start
...
```

## Getting Help

If you're still having issues:

1. Check the results JSON file: `.evalview/results/TIMESTAMP.json`
2. Look at the `trace` section to see what was captured
3. Compare `expected` vs `actual` in the evaluations
4. Open an issue with the verbose output and your test case

## Tips

- Start with simple test cases and gradually add complexity
- Use the conversational route for faster iteration during development
- Check that your API endpoint is actually running before running tests
- Verify your OPENAI_API_KEY is set correctly for LLM-as-judge evaluation
