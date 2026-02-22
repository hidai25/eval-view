# Evaluation Metrics — How EvalView Scores AI Agents

> **Problem:** How do you score an AI agent's quality? Output quality alone isn't enough — you need to verify it called the right tools, in the right order, under budget, and within latency limits.
>
> **Solution:** EvalView uses 5-dimensional scoring: tool accuracy, output quality, sequence correctness, cost, and latency. Each is independently configurable with weights and thresholds.

EvalView evaluates agents across multiple dimensions to give you a complete picture of agent quality.

---

## Default Weights

| Metric | Weight | Description |
|--------|--------|-------------|
| **Tool Accuracy** | 30% | Checks if expected tools were called |
| **Output Quality** | 50% | LLM-as-judge evaluation |
| **Sequence Correctness** | 20% | Validates tool call order (flexible matching) |
| **Cost Threshold** | Pass/Fail | Must stay under `max_cost` |
| **Latency Threshold** | Pass/Fail | Must complete under `max_latency` |

Weights are configurable globally or per-test.

---

## Customizing Weights

### Global Configuration

```yaml
# .evalview/config.yaml
weights:
  tool_accuracy: 0.4
  output_quality: 0.4
  sequence_correctness: 0.2
```

### Per-Test Configuration

```yaml
# tests/test-cases/my-test.yaml
name: "My Test"
weights:
  tool_accuracy: 0.5
  output_quality: 0.3
  sequence_correctness: 0.2
```

---

## Sequence Matching Modes

By default, EvalView uses **flexible sequence matching** — your agent won't fail just because it used extra tools.

| Mode | Behavior | Use When |
|------|----------|----------|
| `subsequence` (default) | Expected tools in order, extras allowed | Most cases — agents can think/verify without penalty |
| `exact` | Exact match required | Strict compliance testing |
| `unordered` | Tools called, order doesn't matter | Order-independent workflows |

### Examples

**subsequence (default)**
```
Expected: [search, analyze]
Actual:   [search, think, analyze, verify]
Result:   ✓ PASS (search, analyze appear in order)
```

**exact**
```
Expected: [search, analyze]
Actual:   [search, think, analyze]
Result:   ✗ FAIL (extra tool: think)
```

**unordered**
```
Expected: [search, analyze]
Actual:   [analyze, search]
Result:   ✓ PASS (both tools called)
```

### Setting the Mode

```yaml
# Per-test override
adapter_config:
  sequence_mode: unordered
```

---

## Tool Accuracy

Measures whether the agent called the expected tools.

```yaml
expected:
  tools:
    - fetch_data
    - analyze
```

**Scoring:**
- All expected tools called: 100%
- Some missing: Proportional score
- No expected tools called: 0%

See [Tool Categories](TOOL_CATEGORIES.md) for flexible matching by intent.

---

## Output Quality (LLM-as-Judge)

Uses an LLM to evaluate the quality of the agent's output.

**Evaluation criteria:**
- Does the output answer the question?
- Is it accurate and factual?
- Is it well-structured?
- Does it follow instructions?

### Custom Evaluation Criteria

```yaml
expected:
  output:
    contains:
      - "revenue"
      - "earnings"
    not_contains:
      - "I don't know"
```

---

## Cost Threshold

Fail if the test exceeds a cost limit:

```yaml
thresholds:
  max_cost: 0.50  # Fail if cost > $0.50
```

---

## Latency Threshold

Fail if the test takes too long:

```yaml
thresholds:
  max_latency: 5000  # Fail if > 5 seconds (in ms)
```

---

## Combining Thresholds

```yaml
name: "Stock Analysis Test"
input:
  query: "Analyze Apple stock performance"

expected:
  tools:
    - fetch_stock_data
    - analyze_metrics
  output:
    contains:
      - "revenue"
      - "earnings"

thresholds:
  min_score: 80    # Overall score must be >= 80
  max_cost: 0.50   # Must cost less than $0.50
  max_latency: 5000  # Must complete in < 5 seconds
```

---

## Hallucination Detection

EvalView can detect when agents make things up:

```yaml
checks:
  hallucination: true
```

This compares the agent's output against the tool results to detect fabricated information.

---

## Example Test Output

```
✅ Stock Analysis Test - PASSED (score: 92.5)

Tool Accuracy:      100% (2/2 tools called)
Output Quality:     90/100 (LLM-as-judge)
Sequence:           100% (correct order)

Cost:    $0.0234 (limit: $0.50) ✓
Latency: 3.4s (limit: 5s) ✓
```

---

## Related Documentation

- [Tool Categories](TOOL_CATEGORIES.md)
- [Statistical Mode](STATISTICAL_MODE.md)
- [CLI Reference](CLI_REFERENCE.md)
