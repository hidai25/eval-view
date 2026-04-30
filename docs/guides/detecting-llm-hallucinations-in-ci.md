# How We Catch LLM Hallucinations Before They Reach Users (Using EvalView)

> **TL;DR:** EvalView detects AI agent hallucinations by verifying tool usage (did the agent actually look it up?), grounding checks (does the output match what the tool returned?), and regression tests from production incidents. Run in CI to catch hallucinations before they reach users.

Last year our support agent told a customer they were eligible for a $500 refund. They weren't. The agent had made up a policy that didn't exist.

We only found out when the customer called to ask why their refund hadn't arrived.

That incident cost us $500 plus a very awkward conversation. It also made us figure out how to catch hallucinations before they hit production. This is what we learned.

---

## What counts as a hallucination

For agents, hallucinations usually fall into three categories:

**1. Made-up facts**
The agent invents information that sounds plausible but isn't true. "Your order shipped yesterday" when it hasn't shipped. "The refund policy allows 60 days" when it's 30.

**2. Skipped verification**
The agent answers confidently without checking. "Your balance is $150" without calling the balance lookup tool.

**3. Wrong tool, wrong data**
The agent calls a tool but misinterprets the response. Tool returns `status: "pending"`, agent says "your order has been delivered."

Category 2 is the easiest to catch. Categories 1 and 3 are harder.

---

## The easy win: verify tool usage

Most hallucinations happen because the agent didn't use the tools it should have. Testing for tool calls catches this.

```yaml
name: "balance inquiry - must use lookup"

input:
  query: "What's my account balance?"
  context:
    user_id: "user_123"

expected:
  tools:
    - get_account_balance   # fail if this isn't called
  output:
    contains_pattern: "\\$[0-9]"   # should mention a dollar amount

thresholds:
  min_score: 80
```

If the agent answers the balance question without calling `get_account_balance`, this test fails. That's the most dangerous type of hallucination—confident answers with no data backing them.

Run it:

```bash
evalview run
```

---

## Testing for grounded responses

Tool verification catches "didn't look it up" hallucinations. But what about "looked it up wrong"?

For this, you need to verify the output against the tool response. EvalView's LLM judge can do basic grounding checks:

```yaml
name: "order status - grounded response"

input:
  query: "Where's my order #5544?"

expected:
  tools:
    - get_order_status
  grounding:
    tool: get_order_status
    response_must_reflect:
      - "status"
      - "tracking_number"

thresholds:
  min_score: 85
```

The `grounding` block tells the judge to verify that the agent's response actually reflects what the tool returned. If the tool says `status: "processing"` but the agent says "delivered," the test fails.

---

## The containment strategy

You can't catch every hallucination with tests. But you can reduce the blast radius.

**Layer 1: Tool call verification**
Catch agents that don't look things up at all.

```yaml
expected:
  tools:
    - lookup_policy
    - get_customer_tier
```

**Layer 2: Output patterns**
Catch agents that give answers in the wrong format or missing key info.

```yaml
expected:
  output:
    contains:
      - "order"
      - "#"
    min_length: 50
    max_length: 500
```

**Layer 3: Grounding checks**
Catch agents that misrepresent tool responses.

```yaml
expected:
  grounding:
    tool: get_order
    response_must_reflect:
      - "status"
```

**Layer 4: Human review triggers**
For high-stakes responses, flag for review instead of auto-sending.

This isn't an EvalView feature—it's an agent design pattern. But your tests can verify it:

```yaml
name: "large refund - requires approval"

input:
  query: "Process a refund for $800"

expected:
  tools:
    - calculate_refund
    - flag_for_review    # must flag, not auto-process
  output:
    contains:
      - "review"
      - "approval"
```

---

## Regression tests from production incidents

Every hallucination that reaches production becomes a test case. We keep a `tests/hallucinations/` directory:

```
tests/hallucinations/
├── 2024-01-fake-refund-policy.yaml
├── 2024-02-wrong-shipping-date.yaml
├── 2024-03-invented-discount.yaml
└── 2024-03-balance-without-lookup.yaml
```

Each file documents what happened and tests for it:

```yaml
# tests/hallucinations/2024-01-fake-refund-policy.yaml
name: "regression: fake refund policy"
description: |
  Agent told customer about a 60-day refund policy that doesn't exist.
  Root cause: didn't call policy lookup, made up answer.
  Incident: SUPPORT-1234

input:
  query: "What's your refund policy?"

expected:
  tools:
    - get_refund_policy    # must look it up
  output:
    contains:
      - "30 days"          # actual policy
    must_not_contain:
      - "60 days"          # the hallucinated policy

thresholds:
  min_score: 90            # high bar for policy questions
```

The `must_not_contain` field is useful for known hallucinations. If the agent ever says "60 days" for refund policy, we want to know immediately.

---

## Scaling up: generate adversarial tests

Hallucinations often happen on edge cases you didn't think to test. Generate variations that push the boundaries:

```bash
evalview expand tests/cases/refund-policy.yaml --count 30 \
  --focus "ambiguous questions, missing context, conflicting info"
```

This creates tests like:
- "What if I want to return something after 45 days?" (boundary)
- "My friend said you do 90-day returns?" (misinformation in query)
- "Refund policy for items bought during the sale?" (edge case)

The more weird inputs you throw at the agent, the more likely you'll find hallucination triggers before users do.

---

## Monitoring in CI

Add hallucination tests to your CI pipeline:

```yaml
# .github/workflows/agent-tests.yml
- name: Run hallucination tests
  run: evalview run --pattern "tests/hallucinations/*.yaml"
  env:
    OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
```

We run these on every PR. A failing hallucination test blocks the merge.

---

## What still gets through

Testing catches systematic hallucinations—the ones that happen reliably given certain inputs. It doesn't catch:

**One-off flukes**: LLMs occasionally produce garbage. If it only happens 1 in 1000 times, you might not catch it in testing.

**Novel situations**: Tests cover known scenarios. Users will always find new ones.

**Subtle inaccuracies**: "Your order will arrive in 3-5 days" when it's actually 5-7 days. The judge might not catch small discrepancies.

For high-stakes applications, testing is necessary but not sufficient. You also need:
- Confidence thresholds (don't answer if uncertain)
- Human review for sensitive topics
- Monitoring and alerting in production

---

## The 80/20

If you do nothing else:

1. **Require tool calls for factual questions**. No looking things up = automatic fail.

2. **Turn every production incident into a regression test**. Same hallucination shouldn't happen twice.

3. **Run tests on every PR**. Catch regressions before they merge.

This won't catch everything, but it'll catch the repeat offenders. And in my experience, hallucinations tend to cluster—fix the systematic ones and you eliminate most of the problem.

---

## Getting started

```bash
pip install evalview
```

Create your first hallucination test:

```yaml
# tests/hallucinations/must-verify.yaml
name: "factual question requires tool call"

input:
  query: "What's the status of order #12345?"

expected:
  tools:
    - get_order_status

thresholds:
  min_score: 75
```

Run it:

```bash
evalview run --pattern "tests/hallucinations/*.yaml"
```

Full docs: [github.com/hidai25/eval-view](https://github.com/hidai25/eval-view)
