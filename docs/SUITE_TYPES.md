# Suite Types — Capability vs Regression Tests for AI Agents

> **Problem:** Not all AI agent test failures are equal. A failure on a new experimental feature is expected. A failure on a core user flow is a critical regression. Without distinguishing them, you either ignore all failures or get overwhelmed by false alarms.
>
> **Solution:** EvalView's suite types let you tag tests as `capability` (expected failures, tracking progress) or `regression` (critical failures, block deploys). CI can be configured to fail only on regression failures.

Not all test failures are equal. Tag your tests to distinguish **expected** failures from **critical** regressions.

---

## Two Types of Tests

### Capability Tests

Measuring what the agent CAN do. Failures are **expected** — you're pushing boundaries.

```yaml
name: complex-multi-step-reasoning
suite_type: capability
thresholds:
  min_score: 70
```

### Regression Tests

Verifying it STILL works. Failures are **red alerts** — something broke.

```yaml
name: login-flow
suite_type: regression
thresholds:
  min_score: 90
```

---

## Console Output

Console output reflects the difference:

```
┌─ Test Results ─────────────────────────────────────┐
│ login-flow          regression  🚨 REGRESSION      │  ← Fix immediately
│ checkout-process    regression  ✅ PASSED          │
│ complex-reasoning   capability  ⚡ CLIMBING        │  ← Expected, keep improving
│ edge-case-handling  capability  ✅ PASSED          │
└────────────────────────────────────────────────────┘

By Suite Type:
  Regression:  1/2 (⚠️ 1 regressions!)
  Capability:  1/2 (hill climbing)
```

---

## Why This Matters

| Suite Type | Failure Meaning | Action |
|------------|-----------------|--------|
| **Regression** | Something broke | Block deploy, fix immediately |
| **Capability** | Not there yet | Track progress, keep improving |

**Regression failures block deploys. Capability failures track progress.**

---

## Usage in CI

Configure CI to treat them differently:

```yaml
# Only fail CI on regression test failures
evalview run --fail-on REGRESSION --suite-type regression

# Run capability tests for tracking, but don't fail CI
evalview run --suite-type capability --no-fail
```

---

## When to Use Each

### Use `regression` for:
- Core user flows (login, checkout, search)
- Features that previously worked
- Critical business logic
- Safety checks

### Use `capability` for:
- New features being developed
- Experimental capabilities
- Stretch goals
- Performance optimization targets

---

## Example Test File

```yaml
# tests/test-cases/core-flows.yaml
name: "User Authentication"
suite_type: regression
input:
  query: "Log me in with email test@example.com"
expected:
  tools:
    - authenticate_user
thresholds:
  min_score: 95  # High bar for regression tests

---
# tests/test-cases/experimental.yaml
name: "Complex Multi-Agent Reasoning"
suite_type: capability
input:
  query: "Coordinate three agents to solve this optimization problem"
expected:
  tools:
    - agent_coordinator
    - optimizer
thresholds:
  min_score: 60  # Lower bar, we're still improving
```

---

## Related Documentation

- [Behavior Coverage](BEHAVIOR_COVERAGE.md)
- [Golden Traces](GOLDEN_TRACES.md)
- [CI/CD Integration](CI_CD.md)
