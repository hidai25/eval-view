# Testing LangGraph Agents in CI: A Practical Guide Using EvalView

> **TL;DR:** EvalView provides pytest-style testing for LangGraph agents. Write YAML test cases that verify tool calls, output quality, cost, and latency. Run them in CI with `evalview run`. Catch regressions before they reach production.

I spent three months shipping a LangGraph agent to production without any tests. Every deploy was a coin flip. Sometimes the agent would start calling the wrong tools. Sometimes it would hallucinate prices. Once it told a customer their order was "probably fine" instead of looking it up.

The problem? Traditional testing doesn't work for agents. You can't `assert response == "exact string"` when the output is non-deterministic. And mocking the LLM defeats the purpose — you want to catch the weird stuff the model actually does.

This is how I test LangGraph agents now, using [EvalView](https://github.com/hidai25/eval-view).

---

## The core idea

Instead of testing outputs, test behaviors:

- Did the agent call the right tools?
- Did it call them in a sensible order?
- Is the response roughly correct (not exact match)?
- Did it stay under budget?
- Did it respond fast enough?

EvalView handles this with YAML test cases and an LLM-as-judge for output quality.

---

## Setup

```bash
pip install evalview
export OPENAI_API_KEY='sk-...'
```

If you have a LangGraph agent running locally:

```bash
langgraph dev  # in one terminal

evalview connect  # in another—auto-detects the agent
```

Or configure manually in `evalview.yaml`:

```yaml
adapter:
  type: langgraph
  endpoint: http://localhost:8123
```

---

## Writing test cases

Test cases are YAML files. Here's a minimal one:

```yaml
# tests/cases/basic.yaml
name: "capital city lookup"

input:
  query: "What's the capital of France?"

expected:
  output:
    contains:
      - "Paris"

thresholds:
  min_score: 70
```

Run it:

```bash
evalview run
```

Output:

```
✅ capital city lookup - PASSED (score: 82)
   Cost: $0.003 | Latency: 1.8s
```

The score comes from an LLM judge that evaluates whether the response adequately answers the query. It's not checking for exact matches—it's checking if the answer is reasonable.

---

## Testing tool calls

This is where it gets useful. Most agent bugs are tool bugs—calling the wrong tool, missing a required tool, or calling tools in a broken order.

```yaml
# tests/cases/order-lookup.yaml
name: "order status check"

input:
  query: "What's the status of order #12345?"

expected:
  tools:
    - lookup_order
  output:
    contains:
      - "12345"

thresholds:
  min_score: 75
  max_cost: 0.10
  max_latency: 5000
```

If the agent answers without calling `lookup_order`, this test fails. That's the behavior you want to catch—an agent that guesses instead of looking things up.

For strict ordering:

```yaml
expected:
  tools_sequence:
    - lookup_order
    - check_inventory
    - calculate_shipping
```

Now the test fails if tools are called out of order.

---

## Handling non-determinism

Agents don't give the same answer twice. A test that passes once might fail the next run. A few ways to handle this:

**1. Test behaviors, not exact outputs**

Don't do this:
```yaml
expected:
  output:
    equals: "The capital of France is Paris."
```

Do this:
```yaml
expected:
  output:
    contains:
      - "Paris"
```

**2. Set reasonable thresholds**

A `min_score` of 70-80 is usually right. Going higher leads to flaky tests.

**3. Use retries for flaky tests**

```bash
evalview run --max-retries 2
```

If a test fails, it'll retry twice before marking it as failed.

---

## Generating test variations

Writing tests by hand is slow. I usually write 3-5 seed tests, then expand them:

```bash
evalview expand tests/cases/order-lookup.yaml --count 50
```

This generates 50 variations:
- Different order IDs
- Different phrasings ("where's my order", "order status", "track order #X")
- Edge cases (invalid IDs, missing IDs)

You can focus the expansion:

```bash
evalview expand tests/cases/order-lookup.yaml --count 20 \
  --focus "angry customers, typos, multiple orders"
```

---

## CI setup (GitHub Actions)

```yaml
# .github/workflows/agent-tests.yml
name: Agent Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"

      - run: pip install evalview

      - name: Start agent
        run: |
          pip install -r requirements.txt
          langgraph dev &
          sleep 10
        env:
          OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}

      - name: Run tests
        run: evalview run
        env:
          OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
```

EvalView exits with code 1 on failures, so GitHub will block the PR.

---

## Catching regressions

The `--summary` flag compares against the last run:

```bash
evalview run --summary
```

```
━━━ Summary ━━━
Tests: 23 passed, 1 failed

Failures:
  ✗ order-lookup: invalid ID    tool accuracy 60% (was 100%)

Deltas vs last run:
  Cost:    +$0.04 ↑
  Latency: +340ms ↑
```

I run this on every PR. If costs spike or latency degrades, I want to know before merging.

---

## A real example

Here's a test from a support agent I work on:

```yaml
name: "refund request - eligible order"

input:
  query: "I want to refund order #99881"
  context:
    user_id: "user_abc"
    order_id: "99881"

expected:
  tools_sequence:
    - get_order
    - check_refund_eligibility
    - process_refund
  output:
    contains:
      - "refund"
      - "processed"

thresholds:
  min_score: 80
  max_cost: 0.15
  max_latency: 8000
```

This catches:
- Agent skipping the eligibility check (bad)
- Agent not actually processing the refund (really bad)
- Agent taking too long (annoying)
- Agent costing too much per request (expensive at scale)

---

## What this doesn't catch

No testing approach catches everything. This won't help with:

- **Subtle hallucinations**: If the agent makes up a plausible-sounding order status, the LLM judge might not catch it. You need grounded evaluation for that (coming soon to EvalView).
- **Long-tail failures**: Tests cover known scenarios. Production will always find new ones.
- **Model updates**: When OpenAI updates GPT-4, your agent might behave differently. Re-run your full test suite after model changes.

But it catches the obvious stuff—and in my experience, the obvious stuff causes 80% of production incidents.

---

## Getting started

```bash
pip install evalview
evalview quickstart  # creates a demo agent + test case
```

Then adapt the generated test case for your agent.

Full docs: [github.com/hidai25/eval-view](https://github.com/hidai25/eval-view)
