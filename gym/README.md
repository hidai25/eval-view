# EvalView Gym — Learn AI Agent Testing Patterns with Guided Exercises

> **Learn to write production-grade AI agent evaluations.** Practice regression detection, golden baselines, statistical testing, and hallucination detection with a built-in demo agent.

The Gym is a self-contained training environment with a demo agent and curated test scenarios. Use it to practice eval patterns before testing your own agents.

## Why a Gym?

Most teams jump straight from "agent works in notebook" to "agent breaks in production." The gap is **eval coverage** — knowing what to test.

The Gym teaches you:
- **Failure modes** — How agents break (timeouts, bad data, loops)
- **Security patterns** — How to test for prompt injection, jailbreaks
- **Assertion strategies** — What to check beyond "output looks right"

## Quick Start

```bash
# 1. Start the gym agent
cd gym/agents/support-bot
pip install -r requirements.txt
cp .env.local.example .env.local  # Add your OpenAI key
langgraph dev

# 2. Run all gym scenarios (in another terminal)
evalview gym

# Or run specific suites
evalview gym --suite failure-modes
evalview gym --suite security
```

## What's Inside

```
gym/
├── agents/
│   └── support-bot/       # Demo agent with 4 tools
│       ├── agent.py       # LangGraph agent
│       ├── chaos_tools.py # Fault injection utilities
│       └── mock_config.py # Deterministic responses
├── failure-modes/         # 10 resilience scenarios
│   ├── 01-tool-timeout.yaml
│   ├── 02-malformed-response.yaml
│   ├── 03-rate-limit.yaml
│   ├── 04-infinite-loop-guard.yaml
│   ├── 05-partial-failure.yaml
│   ├── 06-empty-response.yaml
│   ├── 07-network-error.yaml
│   ├── 08-high-latency.yaml
│   ├── 09-wrong-tool-output.yaml
│   └── 10-cascading-failure.yaml
└── security/              # 5 security scenarios
    ├── 01-prompt-injection-basic.yaml
    ├── 02-prompt-injection-nested.yaml
    ├── 03-jailbreak-resistance.yaml
    ├── 04-data-exfiltration.yaml
    └── 05-tool-misuse.yaml
```

## The Demo Agent

A simple **customer support bot** with 4 tools:

| Tool | Purpose |
|------|---------|
| `search_kb` | Search knowledge base for help articles |
| `create_ticket` | Create support ticket (high/medium/low) |
| `send_reply` | Send reply to customer |
| `check_order` | Check order status |

The agent uses **deterministic mock responses** by default — same input always produces same output. This makes tests reproducible.

## Failure Mode Scenarios

These test how your agent handles real-world problems:

| # | Scenario | What It Tests |
|---|----------|---------------|
| 01 | Tool Timeout | Does agent recover from hung tools? |
| 02 | Malformed Response | Does agent handle corrupted JSON? |
| 03 | Rate Limit | Does agent handle 429 gracefully? |
| 04 | Infinite Loop | Does system catch tool call loops? |
| 05 | Partial Failure | Can agent use working tools when others fail? |
| 06 | Empty Response | Does agent handle null/empty data? |
| 07 | Network Error | Does agent handle connection failures? |
| 08 | High Latency | Does agent work with slow tools? |
| 09 | Wrong Output | Does agent notice mismatched data? |
| 10 | Cascading Failure | Does agent cope when multiple tools fail? |

## Security Scenarios

These test your agent's resistance to adversarial inputs:

| # | Scenario | What It Tests |
|---|----------|---------------|
| 01 | Basic Injection | Direct "ignore instructions" attacks |
| 02 | Nested Injection | Injection hidden in data/queries |
| 03 | Jailbreak | DAN-style roleplay attacks |
| 04 | Data Exfiltration | Attempts to leak secrets/other users' data |
| 05 | Tool Misuse | Attempts to abuse tools for spam/attacks |

## Chaos Configuration

Inject failures via the `context.chaos` field in scenarios:

```yaml
input:
  query: "What is your refund policy?"
  context:
    chaos:
      timeout: true           # Tool times out
      timeout_seconds: 5.0    # How long before timeout
      malformed: true         # Return invalid JSON
      rate_limit: true        # Return 429 error
      empty: true             # Return null/empty
      error: "Custom error"   # Raise custom exception
      latency_ms: 2000        # Add artificial delay
      max_calls_per_tool: 5   # Loop detection threshold
      failing_tools:          # Only these tools fail
        - search_kb
```

## Writing Your Own Scenarios

1. Copy an existing scenario as a template
2. Modify the `input.query` and `context.chaos`
3. Update `expected` assertions
4. Run with `evalview run gym/your-scenario.yaml`

### Assertion Types

```yaml
expected:
  output:
    contains:           # Must include ALL of these strings
      - "refund"
      - "policy"
    not_contains:       # Must NOT include ANY of these
      - "Traceback"
      - "Exception"

  tools:                # Must call these tools
    - search_kb

thresholds:
  min_score: 70         # Minimum passing score (0-100)
  max_latency: 10000    # Max response time (ms)
  max_cost: 0.50        # Max cost in dollars
```

## Best Practices

1. **Start with happy path** — Make sure normal cases work first
2. **Add failure modes** — Test timeouts, errors, edge cases
3. **Add security tests** — Test injection resistance
4. **Keep scenarios focused** — One failure mode per test
5. **Use deterministic mocks** — Reproducible tests are debuggable tests

## Applying to Your Agent

Once you've practiced in the gym, apply the patterns to your own agent:

```bash
# Copy a scenario and adapt it
cp gym/failure-modes/01-tool-timeout.yaml my-agent/tests/

# Update the adapter and endpoint
adapter: langgraph  # or your adapter
endpoint: http://localhost:8000

# Run against your agent
evalview run my-agent/tests/01-tool-timeout.yaml
```

## Contributing

Found a useful failure mode we missed? PRs welcome:

1. Add scenario to `gym/failure-modes/` or `gym/security/`
2. Include clear `name` and `description`
3. Document what it tests and expected behavior
4. Ensure it passes against the demo agent (or document expected failures)
