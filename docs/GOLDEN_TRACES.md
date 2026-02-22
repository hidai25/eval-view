# Golden Traces — Automatic Regression Detection for AI Agents

> **Problem:** You changed a prompt, swapped a model, or updated a tool in your AI agent. Did anything break? Without golden baselines, you can't know until users complain. Traditional unit tests don't work because LLM outputs are non-deterministic.
>
> **Solution:** EvalView's golden trace system captures a snapshot of known-good agent behavior and automatically detects when future runs deviate. Works without LLM-as-judge or API keys — pure deterministic tool-call and sequence comparison.

---

## How It Works

```bash
# 1. Run your tests
evalview run

# 2. Save a passing run as your golden baseline
evalview golden save .evalview/results/20241201_143022.json

# 3. On future runs, compare against golden
evalview run --diff
```

When you run with `--diff`, EvalView compares every test against its golden baseline and flags:

| Status | What It Means | Action |
|--------|---------------|--------|
| **PASSED** | Matches baseline | Ship it |
| **TOOLS_CHANGED** | Agent uses different tools | Review before deploy |
| **OUTPUT_CHANGED** | Same tools, different response | Review before deploy |
| **REGRESSION** | Score dropped significantly | Fix before deploy |

---

## Example Output

```
━━━ Golden Diff Report ━━━

✓ PASSED           test-stock-analysis
⚠ TOOLS_CHANGED    test-customer-support    added: web_search
~ OUTPUT_CHANGED   test-summarizer          similarity: 78%
✗ REGRESSION       test-code-review         score dropped 15 points

1 REGRESSION - fix before deploy
1 TOOLS_CHANGED - review before deploy
```

---

## Golden Commands

### Save a golden baseline

```bash
# Save a result as golden baseline
evalview golden save .evalview/results/xxx.json

# Save with notes
evalview golden save result.json --notes "Baseline after v2.0 refactor"

# Save only specific test from a multi-test result
evalview golden save result.json --test "stock-analysis"
```

### List golden traces

```bash
evalview golden list
```

### Show details

```bash
evalview golden show test-stock-analysis
```

### Delete a golden trace

```bash
evalview golden delete test-stock-analysis
evalview golden delete test-stock-analysis --force
```

---

## Quick Comparison in Chat

Don't want to memorize CLI flags? Use chat mode:

```bash
evalview chat
> /compare .evalview/results/old.json .evalview/results/new.json
```

Shows a side-by-side table with score deltas and regression detection:

```
┌─────────────────┬───────────┬───────────┬────────┬──────────┐
│ Test            │ Old Score │ New Score │ Δ      │ Status   │
├─────────────────┼───────────┼───────────┼────────┼──────────┤
│ stock-analysis  │ 92.5      │ 94.0      │ +1.5   │ ✅ OK    │
│ customer-support│ 88.0      │ 71.0      │ -17.0  │ REGR  │
└─────────────────┴───────────┴───────────┴────────┴──────────┘
```

---

## CI Integration

Add `evalview run --diff` to CI to block deploys when behavior regresses:

```yaml
- name: Run EvalView
  uses: hidai25/eval-view@v0.2.1
  with:
    openai-api-key: ${{ secrets.OPENAI_API_KEY }}
    diff: true
    fail-on: 'REGRESSION'
```

See [CI/CD Integration](CI_CD.md) for complete setup.

---

## Configurable Strictness

```bash
# Default: only fail on score drops
evalview run --diff --fail-on REGRESSION

# Stricter: also fail on tool changes
evalview run --diff --fail-on REGRESSION,TOOLS_CHANGED

# Strictest: fail on any change
evalview run --diff --strict
```

---

## Core Workflow

```bash
# 1. Run tests and capture a baseline
evalview run
evalview golden save .evalview/results/latest.json

# 2. Make changes to your agent (prompt, model, tools)

# 3. Run with diff to catch regressions
evalview run --diff

# 4. CI integration with configurable strictness
evalview run --diff --fail-on REGRESSION                    # Default: only fail on score drops
evalview run --diff --fail-on REGRESSION,TOOLS_CHANGED      # Stricter: also fail on tool changes
evalview run --diff --strict                                # Strictest: fail on any change
```

---

## Related Documentation

- [CLI Reference](CLI_REFERENCE.md)
- [CI/CD Integration](CI_CD.md)
- [Statistical Mode](STATISTICAL_MODE.md)
