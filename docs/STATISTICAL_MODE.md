# Statistical Mode (Variance Testing)

LLMs are non-deterministic. A test that passes once might fail the next run. Statistical mode addresses this by running tests multiple times and using statistical thresholds for pass/fail decisions.

---

## Quick Start: CLI Flag

The easiest way to enable statistical mode is with the `--runs` flag:

```bash
# Run each test 10 times
evalview run --runs 10

# Run with custom pass rate (70% must pass)
evalview run --runs 10 --pass-rate 0.7

# Filter by difficulty and run statistically
evalview run --difficulty hard --runs 5
```

This overrides any per-test variance configuration.

---

## Per-Test Configuration (YAML)

Add `variance` config to your test case:

```yaml
# tests/test-cases/my-test.yaml
name: "My Agent Test"
input:
  query: "Analyze the market trends"

expected:
  tools:
    - fetch_data
    - analyze

thresholds:
  min_score: 70

  # Statistical mode config
  variance:
    runs: 10           # Run test 10 times
    pass_rate: 0.8     # 80% of runs must pass
    min_mean_score: 70 # Average score must be >= 70
    max_std_dev: 15    # Score std dev must be <= 15
```

---

## What You Get

- **Pass rate** - Percentage of runs that passed
- **pass@k / pass^k** - Industry-standard reliability metrics (see below)
- **Score statistics** - Mean, std dev, min/max, percentiles, confidence intervals
- **Flakiness score** - 0 (stable) to 1 (flaky) with category labels
- **Contributing factors** - Why the test is flaky (score variance, tool inconsistency, etc.)

---

## Reliability Metrics: pass@k vs pass^k

Two metrics that tell you different things:

| Metric | Question it answers | High = |
|--------|---------------------|--------|
| **pass@k** | "Will it work if I give it a few tries?" | Usually finds a solution |
| **pass^k** | "Will it work reliably every time?" | Consistent and reliable |

```
Reliability Metrics:
  pass@10:       99.9% (usually finds a solution)
  pass^10:       2.8% (unreliable)
```

This example shows an agent that *eventually* works but isn't production-ready.

---

## Example Output

```
Statistical Evaluation: My Agent Test
PASSED

┌─ Run Summary ─────────────────────────┐
│  Total Runs:     10                   │
│  Passed:         8                    │
│  Failed:         2                    │
│  Pass Rate:      80% (required: 80%)  │
└───────────────────────────────────────┘

Score Statistics:
  Mean:      79.86    95% CI: [78.02, 81.70]
  Std Dev:   2.97     ▂▂▁▁▁ Low variance
  Min:       75.5
  Max:       84.5

┌─ Flakiness Assessment ────────────────┐
│  Flakiness Score: 0.12 ██░░░░░░░░     │
│  Category:        low_variance        │
│  Pass Rate:       80%                 │
└───────────────────────────────────────┘
```

---

## Variance Configuration Options

| Option | Description | Default |
|--------|-------------|---------|
| `runs` | Number of times to run the test | 10 |
| `pass_rate` | Minimum percentage of runs that must pass | 0.8 (80%) |
| `min_mean_score` | Minimum average score across all runs | Same as `min_score` |
| `max_std_dev` | Maximum allowed standard deviation | 15 |

---

## When to Use Statistical Mode

**Use it when:**
- Your agent's output varies between runs
- You're testing creative/generative tasks
- You need production-readiness confidence
- You're debugging flaky tests

**Skip it when:**
- Your agent is deterministic (temperature=0)
- You just need a quick sanity check
- Running in CI where time is critical

---

## Complete Example

See [examples/statistical-mode-example.yaml](../examples/statistical-mode-example.yaml) for a complete example.

---

## Related Documentation

- [Evaluation Metrics](EVALUATION_METRICS.md)
- [CLI Reference](CLI_REFERENCE.md)
