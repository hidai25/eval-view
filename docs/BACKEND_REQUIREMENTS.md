# Backend Requirements — How to Make Your AI Agent Compatible with EvalView

> This document outlines what your AI agent backend needs to expose for EvalView testing. EvalView supports three tiers: basic (just text response), metadata (response + cost/tokens), and full streaming (JSONL event stream with tool calls).

## Quick Start (5 Minutes)

**Minimum to get started:**

1. Your agent must respond to POST requests:
   ```bash
   curl -X POST http://localhost:3000/api/chat \
     -H "Content-Type: application/json" \
     -d '{"message": "Test query"}'
   ```

2. Response must include the agent's answer:
   ```json
   {"response": "Agent answer here..."}
   ```

3. That's it! You can now test output quality and latency.

**To add cost tracking (10 more minutes):**

Add metadata to your response:
```json
{
  "response": "Agent answer...",
  "metadata": {
    "cost": 0.05,
    "tokens": {"input": 100, "output": 500}
  }
}
```

**For full tool tracking (20 more minutes):**

Stream JSONL events (see Level 3 below).

---

## Overview

EvalView is a **general-purpose testing framework** for AI agents. It works with any agent that:
- Accepts queries via HTTP API
- Returns responses (streaming or non-streaming)
- Can emit structured execution data (optional but recommended)

## Three-Tier Support Model

EvalView supports agents at different levels of sophistication:

| Level | What You Provide | What Gets Tested | Setup Time |
|-------|-----------------|-----------------|------------|
| **Level 1: Basic** | Just text response | Output quality, latency | 5 minutes |
| **Level 2: Metadata** | Response + cost/tokens | Everything except tool sequence | 15 minutes |
| **Level 3: Streaming** | Full event stream | Everything (tools, cost, sequence) | 30 minutes |

### Level 1: Basic Agent (Minimum Viable)

**What you need:**
```json
POST /api/chat
{"message": "What is AAPL stock price?"}

Response:
{"response": "AAPL is trading at $266.25..."}
```

**What gets tested:**
- ✅ Output quality (contains expected keywords)
- ✅ Latency (response time)
- ❌ Cost (will show $0 with warning)
- ❌ Tools called
- ❌ Sequence correctness

**Best for:** Quick start, proof of concept, simple agents

### Level 2: Agent with Metadata (Recommended)

**What you need:**
```json
POST /api/chat
{"message": "Analyze AAPL stock"}

Response:
{
  "response": "AAPL analysis...",
  "metadata": {
    "cost": 0.05,
    "tokens": {"input": 100, "output": 500},
    "steps": ["fetch_data", "analyze", "synthesize"]
  }
}
```

**What gets tested:**
- ✅ Output quality
- ✅ Latency
- ✅ Cost (from metadata)
- ✅ Tools called (basic list)
- ⚠️  Sequence (order only, not parameters)

**Best for:** Most production agents, good balance of effort vs. coverage

### Level 3: Streaming Agent (Full Featured)

**What you need:** JSONL event stream (see below)

**What gets tested:** Everything with full fidelity

**Best for:** Complex multi-step agents, orchestrators, production systems

---

## Core Requirements

### 1. HTTP API Endpoint

Your agent must expose an HTTP endpoint that:
- Accepts POST requests with JSON payload
- Returns a response (sync or streaming)
- Completes within reasonable time (30-120 seconds recommended)

**Example request:**
```json
POST /api/chat
{
  "message": "Analyze AAPL stock performance",
  "userId": "test-user"
}
```

### 2. Response Format

Two options supported:

#### Option A: JSONL Streaming (Recommended)
Stream JSON Lines events for rich execution tracking:

```jsonl
{"type": "tool_call", "data": {"name": "analyzeStock", "args": {"symbol": "AAPL"}}}
{"type": "tool_result", "data": {"result": "...", "success": true}}
{"type": "usage", "data": {"input_tokens": 1000, "output_tokens": 500}}
{"type": "message_complete", "data": {"content": "Final response..."}}
```

#### Option B: Simple JSON Response
Return complete response in single JSON:

```json
{
  "response": "Complete agent response text...",
  "metadata": {
    "cost": 0.05,
    "tokens": 1500
  }
}
```

## Recommended: Structured Event Streaming

For comprehensive test coverage, emit these event types:

### Tool Call Events
When your agent calls a tool/function:
```json
{"type": "tool_call", "data": {
  "name": "analyzeStock",
  "args": {"symbol": "AAPL"}
}}
```

### Tool Result Events
After tool execution:
```json
{"type": "tool_result", "data": {
  "result": "Stock analysis data...",
  "success": true,
  "error": null
}}
```

### Step Narration (Alternative)
If you have step descriptions:
```json
{"type": "step_narration", "data": {
  "text": "Analyzing stock fundamentals",
  "toolName": "analyzeStock"
}}
```

### Usage/Cost Events
For cost tracking:
```json
{"type": "usage", "data": {
  "input_tokens": 1000,
  "output_tokens": 500,
  "cached_tokens": 100
}}
```

### Final Message
Complete response:
```json
{"type": "message_complete", "data": {
  "content": "Complete agent response..."
}}
```

## Critical: Avoid Infinite Loops

**Problem:** Your backend refining indefinitely blocks tests.

**Example Bad Pattern:**
```javascript
// ❌ DON'T DO THIS
while (quality < threshold) {
  result = await refine(result);
  // No limit - can loop forever!
}
```

**Solution:** Add max iteration limits
```javascript
// ✅ DO THIS
const MAX_REFINEMENTS = 3;
let refinements = 0;

while (quality < threshold && refinements < MAX_REFINEMENTS) {
  result = await refine(result);
  refinements++;
}
```

### Recommended Refinement Strategy

1. **Set max iterations:** 3-5 refinements maximum
2. **Add time limits:** Stop if taking > 30 seconds
3. **Check for improvements:** Stop if quality isn't increasing
4. **Return partial results:** Better to return "good enough" than timeout

**Example:**
```javascript
async function executeWithRefinement(query, maxRefinements = 3) {
  let result = await initialExecution(query);
  let refinements = 0;
  const startTime = Date.now();
  const MAX_TIME = 30000; // 30 seconds

  while (
    shouldRefine(result) &&
    refinements < maxRefinements &&
    (Date.now() - startTime) < MAX_TIME
  ) {
    result = await refine(result);
    refinements++;

    // Emit progress
    await stream.write(JSON.stringify({
      type: "status",
      data: {text: `Refinement ${refinements}/${maxRefinements}`}
    }) + "\n");
  }

  return result;
}
```

## Response Time Guidelines

| Agent Type | Recommended Timeout |
|-----------|-------------------|
| Simple Q&A | 10-30 seconds |
| Multi-step analysis | 30-90 seconds |
| Complex orchestration | 60-120 seconds |

**If your agent needs more time:**
- Reduce complexity
- Limit refinement iterations
- Parallelize tool calls
- Cache expensive operations

## Cost & Token Tracking

To enable cost evaluation, emit usage events:

```json
{"type": "usage", "data": {
  "input_tokens": 1000,
  "output_tokens": 500,
  "cached_tokens": 100,
  "model": "gpt-4" // optional
}}
```

EvalView will:
- Sum tokens across all steps
- Calculate costs using built-in pricing
- Compare against test thresholds

## Testing Your Backend

Before running EvalView tests:

1. **Test response time:**
   ```bash
   time curl -X POST http://localhost:3000/api/chat \
     -H "Content-Type: application/json" \
     -d '{"message": "Test query"}'
   ```
   Should complete in < 60 seconds.

2. **Check event format:**
   ```bash
   curl -X POST http://localhost:3000/api/chat \
     -H "Content-Type: application/json" \
     -d '{"message": "Test query"}' \
     | jq -R 'fromjson? | .type'
   ```
   Should show event types.

3. **Monitor logs:**
   Check for infinite loops or stuck operations.

## Common Issues

### Issue: Tests timeout
**Cause:** Backend stuck in refinement loop
**Fix:** Add max refinement limit (3-5 iterations)

### Issue: No cost tracking
**Cause:** Not emitting usage events
**Fix:** Emit `{"type": "usage", ...}` after each LLM call

### Issue: No tool calls captured
**Cause:** Not emitting tool_call/tool_result events
**Fix:** Emit events when tools execute

### Issue: Empty or incomplete responses
**Cause:** Streaming not properly closed
**Fix:** Always emit final `message_complete` event

## TapeScope-Specific Notes

If you're using TapeScope backend:

1. **Fix refinement loop** in your orchestrator:
   - Limit "TinyLLM refinement decision" to max 3-5 iterations
   - Add timeout after 30 seconds
   - Return results even if quality < threshold

2. **Emit streaming events** in your API route:
   - When calling `analyzeStock`, emit `tool_call` event
   - After tool completes, emit `tool_result` event
   - At the end, emit `message_complete` event

3. **Track token usage**:
   - Sum tokens from all LLM calls
   - Emit `usage` event with totals

## Questions?

- See `examples/` directory for reference implementations
- Check `evalview/adapters/` for adapter code
- File issues at: https://github.com/hidai25/eval-view/issues
