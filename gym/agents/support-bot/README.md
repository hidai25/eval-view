# Support Bot Agent

A simple customer support agent for practicing eval patterns in the EvalView Gym.

## Tools

| Tool | Purpose |
|------|---------|
| `search_kb` | Search the knowledge base for help articles |
| `create_ticket` | Create a support ticket (high/medium/low priority) |
| `send_reply` | Send a reply to the customer |
| `check_order` | Check order status by order ID |

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Copy env file and add your OpenAI key
cp .env.local.example .env.local

# 3. Run the agent server
langgraph dev

# Agent runs at http://localhost:2024
```

## Direct Testing

```bash
# Basic query
python agent.py "I need a refund for order 12345"

# With chaos injection
python agent.py "What is your refund policy?" --chaos timeout
python agent.py "Check my order 99999" --chaos malformed
```

## Mock Mode

The agent uses deterministic mock responses by default (`EVALVIEW_MOCK_MODE=always`).

This means:
- No real API calls to external services
- Same input always produces same tool output
- Tests are fully reproducible

Set `EVALVIEW_MOCK_MODE=never` to use real APIs (requires valid API keys).

## Chaos Testing

The agent supports fault injection via the `context.chaos` field in test scenarios:

```yaml
input:
  query: "Search for refund policy"
  context:
    chaos:
      timeout: true      # Simulate tool timeout
      malformed: true    # Return malformed JSON
      rate_limit: true   # Simulate 429 error
      empty: true        # Return empty response
      error: "Custom error message"
```

See `gym/failure-modes/` for example scenarios.
