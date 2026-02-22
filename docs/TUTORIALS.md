# EvalView Tutorials — Step-by-Step Guides for AI Agent Testing

> Hands-on guides for mastering EvalView's advanced features, including multi-reference goldens, CI/CD regression detection, diff status interpretation, parameter-level debugging, and migration from older workflows.

---

## Tutorial 1: Handling Non-Deterministic Agents with Multi-Reference Goldens

### The Problem

Your LLM-based agent produces valid but varied outputs:

```
Run 1: "AAPL stock is $150.23"
Run 2: "Apple Inc (AAPL) trades at $150.23"
Run 3: "Current AAPL price: $150.23"
```

All are correct, but `evalview check` fails because they don't match exactly.

### The Solution: Multi-Reference Goldens

Save multiple acceptable variants as golden baselines.

**Step 1: Create your first snapshot**

```bash
evalview snapshot
# ✅ Snapshotted: stock-price-check
```

**Step 2: Run again and save the variant**

```bash
evalview run --save-golden result.json --variant variant1
```

Or use the simpler workflow:

```bash
evalview snapshot --variant variant1
```

**Step 3: Repeat for variant 2**

```bash
evalview snapshot --variant variant2
```

**Step 4: Check matches ANY variant**

```bash
evalview check
# ✅ Matched variant 2/3
```

### How Multi-Reference Works

- **Compare against ALL variants** - Check runs your test against default + all variants
- **Return best match** - Passes if ANY variant matches (ranked by severity)
- **Up to 5 variants** - Prevents storage bloat
- **Severity ranking**: PASSED > OUTPUT_CHANGED > TOOLS_CHANGED > REGRESSION

### When to Use Variants

✅ **Good use cases:**
- LLM output with acceptable creative variation
- Tools called in different but valid orders
- Equivalent responses with different wording

❌ **Bad use cases:**
- Unstable/flaky agents (fix the agent instead)
- Widely different behaviors (split into separate tests)
- >5 variants needed (test is too broad)

### Managing Variants

```bash
# List all variants
evalview golden list

# Delete a variant
evalview golden delete stock-price-check --variant variant1

# Show variant count
evalview golden list  # Shows "stock-price-check (3 variants)"
```

---

## Tutorial 2: Setting Up Regression Detection in GitHub Actions

### Goal

Automatically catch agent regressions in CI before they reach production.

### Step 1: Create Baseline

On your main branch:

```bash
evalview snapshot
git add .evalview/golden/
git commit -m "Add golden baselines for agent tests"
git push
```

### Step 2: GitHub Actions Workflow

Create `.github/workflows/agent-regression-check.yml`:

```yaml
name: Agent Regression Check
on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  regression-check:
    runs-on: ubuntu-latest

    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.9'

      - name: Install dependencies
        run: |
          pip install evalview
          # Install your agent dependencies
          pip install -r requirements.txt

      - name: Run regression check
        run: evalview check --fail-on REGRESSION --json
        env:
          OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}

      - name: Upload results
        if: always()
        uses: actions/upload-artifact@v3
        with:
          name: regression-check-results
          path: .evalview/results/
```

### Step 3: Configure Exit Codes

Choose what failures should block PRs:

**Strict mode** (fail on any change):
```yaml
run: evalview check --strict
```

**Selective** (fail only on score regressions):
```yaml
run: evalview check --fail-on REGRESSION
```

**Permissive** (fail on regressions + tool changes):
```yaml
run: evalview check --fail-on REGRESSION,TOOLS_CHANGED
```

### Step 4: PR Comments (Optional)

Add diff summary to PR comments:

```yaml
      - name: Parse results
        if: failure()
        id: parse
        run: |
          SUMMARY=$(evalview check --json | jq -r '.summary')
          echo "summary=$SUMMARY" >> $GITHUB_OUTPUT

      - name: Comment PR
        if: failure() && github.event_name == 'pull_request'
        uses: actions/github-script@v6
        with:
          script: |
            github.rest.issues.createComment({
              issue_number: context.issue.number,
              owner: context.repo.owner,
              repo: context.repo.repo,
              body: '⚠️ **Agent Regression Detected**\n\n${{ steps.parse.outputs.summary }}'
            })
```

### Advanced: Parallel Test Matrix

Run multiple agent configurations:

```yaml
strategy:
  matrix:
    model: [gpt-4, claude-3-sonnet]
    agent_config: [default, verbose]

steps:
  - name: Run regression check
    run: evalview check --config ${{ matrix.model }}-${{ matrix.agent_config }}.yaml
```

---

## Tutorial 3: Understanding Diff Statuses

### The 4 Diff Statuses

EvalView uses structured diff statuses to categorize changes.

#### ✅ PASSED

**Meaning**: Behavior matches baseline exactly.

**What it shows:**
```
Diff Summary
  5/5 unchanged
```

**Action**: Ship with confidence.

---

#### ~ OUTPUT_CHANGED

**Meaning**: Tools and sequence correct, but output text differs.

**What it shows:**
```
⚠ OUTPUT_CHANGED: customer-support
  Output similarity: 82%
  Same tools, different wording
```

**Common causes:**
- LLM creativity/temperature
- Timestamp/date changes
- Minor wording variations

**Action**:
- Review output quality
- If acceptable: `evalview snapshot` to update baseline
- If unacceptable: Fix prompt and re-check

---

#### ⚠️ TOOLS_CHANGED

**Meaning**: Agent called different tools or different order.

**What it shows:**
```
⚠ TOOLS_CHANGED: stock-analysis
  + web_search (new tool used)
  - calculator (no longer used)
```

**Common causes:**
- Agent found alternative solution path
- Tool availability changed
- Prompt changes affected tool selection

**Action**:
- Verify new tools produce correct results
- Check if old tool was necessary
- Update baseline if change is intentional

---

#### ❌ REGRESSION

**Meaning**: Score dropped significantly (default: >5 points).

**What it shows:**
```
✗ REGRESSION: data-analysis
  Score: 88 → 71 (-17 points)
  Tool sequence differs
  Output quality degraded
```

**Common causes:**
- Broken tools/dependencies
- Model degradation
- Prompt engineering gone wrong
- Test data changed

**Action**:
- **Do not merge** - Fix the issue
- Investigate which component broke
- Run `evalview check` after each fix attempt

---

### Configuring Thresholds

Customize what triggers each status in `.evalview/config.yaml`:

```yaml
diff:
  tool_similarity_threshold: 0.8      # Lower = stricter tool matching
  output_similarity_threshold: 0.9    # Lower = stricter output matching
  score_regression_threshold: 5.0     # Points drop before REGRESSION
```

Or per-test:

```yaml
# tests/test-cases/my-test.yaml
name: flaky-llm-test
adapter_config:
  sequence_mode: "unordered"  # Don't care about tool order

expected:
  min_score: 75
  tools: ["search", "analyze"]

# Custom diff thresholds for this test only
diff_config:
  output_similarity_threshold: 0.85  # More lenient for this test
```

---

## Tutorial 4: Parameter-Level Debugging

### The Problem

Check says "TOOLS_CHANGED" but you don't know WHY.

Before parameter diffing:
```
⚠ TOOLS_CHANGED: search-analysis
  Tool: search (changed)
```

### The Solution: Parameter Diffing

After parameter diffing:
```
⚠ TOOLS_CHANGED: search-analysis
  Tool: search (changed)
    Parameter differences:
      ~ query:
        golden: "AAPL stock price"
        actual: "AAPL current price"
        similarity: 76%
      - max_results: 10
      + limit: 5
```

### Reading Parameter Diffs

**Symbol meanings:**

| Symbol | Meaning | Example |
|--------|---------|---------|
| `~` | Value changed | `query: "old" → "new"` |
| `-` | Parameter removed | `- max_results: 10` |
| `+` | Parameter added | `+ limit: 5` |
| (none) | Type changed | `id: "123" → 123` |

**Similarity percentage:**
- 90-100%: Nearly identical (typo, minor change)
- 70-89%: Similar (wording variation)
- 50-69%: Moderately different
- 0-49%: Substantially different

### Example: Debugging a Regression

```bash
$ evalview check

⚠ TOOLS_CHANGED: stock-search
  Tool: search (changed)
    Parameter differences:
      ~ query:
        golden: "AAPL stock price today"
        actual: ""
        similarity: 0%
```

**Analysis**: Query is empty! This is likely a bug.

**Fix**: Check your prompt or query construction logic.

---

## Tutorial 5: Customizing Celebration Milestones

### Default Milestones

Out of the box, EvalView celebrates:
- 3 checks: "You're on a roll"
- 5 checks: Panel with border
- 10 checks: ASCII art + "Reliability Champion"
- 25 checks: "Legendary" + shareable badge
- 50+ checks: "Incredible" milestone

### Milestone Constants

Defined in `evalview/core/celebrations.py`:

```python
STREAK_MILESTONE_START = 1
STREAK_MILESTONE_SMALL = 3
STREAK_MILESTONE_MEDIUM = 5
STREAK_MILESTONE_LARGE = 10
STREAK_MILESTONE_LEGENDARY = 25
STREAK_MILESTONE_INCREDIBLE = 50
```

### Viewing Your Progress

```bash
$ evalview check

🔍 Comparing against your baseline...
✨ All clean! No regressions detected.

🎯 5 clean checks in a row! You're on a roll.

🟢 Project Health: 100%
  Total checks: 5
  Clean: 5
  Regressions: 0
  Current streak: 5 🔥
  Best streak: 5
```

### Streak Tracking Data

Stored in `.evalview/state.json`:

```json
{
  "current_streak": 5,
  "longest_streak": 10,
  "total_checks": 25,
  "regression_count": 3,
  "milestones_hit": ["streak_3", "streak_5", "streak_10"]
}
```

**Note**: This file is personal (not committed to git).

---

## Tutorial 6: Migrating from `run --diff` to `snapshot/check`

### Old Workflow

```bash
evalview run --save-golden result.json
# ... make changes ...
evalview run --diff
```

**Pain points:**
- Have to remember result file path
- Verbose output
- Not memorable

### New Workflow

```bash
evalview snapshot
# ... make changes ...
evalview check
```

**Benefits:**
- No file paths to remember
- Concise, diff-focused output
- Habit-forming streak tracking
- Celebratory feedback

### Migration Guide

**Step 1**: If you have existing golden files, they still work:

```bash
# Your old goldens in .evalview/golden/ are compatible
evalview check  # Works immediately
```

**Step 2**: Update your CI:

```diff
- evalview run --diff --fail-on REGRESSION
+ evalview check --fail-on REGRESSION --json
```

**Step 3**: Update your docs/README:

```diff
- To create baseline: evalview run --save-golden
+ To create baseline: evalview snapshot

- To check for regressions: evalview run --diff
+ To check for regressions: evalview check
```

### Backward Compatibility

Both workflows coexist:

```bash
# Old way still works
evalview run --save-golden
evalview run --diff

# New way (recommended)
evalview snapshot
evalview check
```

---

## Tutorial 7: Interpreting Reason Codes

### What are Reason Codes?

Structured, machine-readable error codes with remediation guidance.

### Example

```bash
$ evalview check

Failure Reasons:
  ✗ TOOL_MISSING: Expected tool 'calculator' was not called
    → Fix: Ensure your agent has access to 'calculator' and the query triggers its use

  ⚠ PARAM_VALUE_CHANGED: search.query changed
    Details: {"expected": "AAPL stock", "actual": "AAPL"}
    → Fix: Update test case if this is intentional
```

### Common Reason Codes

| Code | Severity | Meaning | Fix |
|------|----------|---------|-----|
| `TOOL_MISSING` | error | Expected tool not called | Check agent tool access |
| `TOOL_UNEXPECTED` | info | Agent called extra tool | Add to test if correct |
| `TOOL_NAME_MISMATCH` | warning | Case/naming difference | Update test case name |
| `SEQUENCE_LENGTH_MISMATCH` | error | Wrong number of tools | Check agent logic |
| `SEQUENCE_ORDER_VIOLATION` | error | Tools out of order | Use `sequence_mode: unordered` if order doesn't matter |
| `PARAM_VALUE_CHANGED` | warning | Parameter value differs | Review and snapshot if OK |
| `PARAM_TYPE_MISMATCH` | error | Parameter type changed | Check serialization |

### Using Reason Codes in CI

```bash
evalview check --json | jq '.diffs[].reason_codes'
```

Output:
```json
[
  {
    "code": "TOOL_MISSING",
    "severity": "error",
    "message": "Expected tool 'search' was not called",
    "context": {
      "expected_tool": "search",
      "actual_tools": ["analyze"]
    },
    "remediation": "Ensure your agent has access to 'search'"
  }
]
```

---

## Next Steps

- **More examples**: See `examples/` directory
- **API Reference**: Coming soon
- **Community**: GitHub Discussions for questions
- **Contributing**: See CONTRIBUTING.md

---

**Questions?** Open an issue or discussion on GitHub!
