---
name: evalview-gate
description: Regression gate for autonomous agent development. After every code change, run EvalView checks to detect regressions. Revert changes that break tests. Snapshot passing behavior as new baselines.
version: "1.0.0"
author: "EvalView"
triggers:
  - "check for regressions"
  - "run evalview"
  - "gate check"
  - "regression test"
  - "snapshot baseline"
tools:
  - bash
  - read
  - write
---

# EvalView Regression Gate

You have access to EvalView, a regression testing tool for AI agents. Use it to verify that code changes don't break agent behavior.

## Core Workflow

After **every code change** that affects agent behavior, run the gate:

```bash
evalview check --json --path tests/
```

Parse the JSON output to determine the result:

- `"all_passed": true` — Safe to continue. The change didn't break anything.
- `"has_regressions": true` — **STOP. Revert the change.** A regression means the agent's score dropped significantly.
- `"has_tools_changed": true` — Review carefully. The agent is using different tools than before. This may be intentional or a bug.
- `"has_output_changed": true` — Minor change. The agent's output differs but tools are the same. Usually safe.

## Decision Rules

### On REGRESSION
1. Revert the change immediately with `git checkout -- .`
2. Analyze what broke by reading the JSON diff details
3. Try a different approach that preserves the existing behavior

### On TOOLS_CHANGED
1. Check if the tool change was intentional (part of the requested feature)
2. If intentional and score improved: accept with `evalview snapshot --test "<name>"`
3. If unintentional: revert and try again

### On ALL PASSED
1. Continue with the next task
2. If you made a significant improvement, snapshot it: `evalview snapshot --path tests/`

## Commands Reference

```bash
# Check for regressions (machine-readable output)
evalview check --json --path tests/

# Check a single test
evalview check --json --path tests/ --test "my-test"

# Preview what a snapshot would change (dry run)
evalview snapshot --path tests/ --preview

# Save current passing behavior as new baseline
evalview snapshot --path tests/

# Save only a specific test as baseline
evalview snapshot --path tests/ --test "my-test"

# Strict mode: fail on ANY change, not just regressions
evalview check --json --path tests/ --strict
```

## JSON Output Format

The `--json` flag returns structured data:

```json
{
  "summary": {
    "total_tests": 5,
    "unchanged": 4,
    "regressions": 1,
    "tools_changed": 0,
    "output_changed": 0
  },
  "diffs": [
    {
      "test_name": "weather-lookup",
      "status": "regression",
      "score_delta": -15.2,
      "has_tool_diffs": true,
      "output_similarity": 0.45
    }
  ]
}
```

## Python API (Alternative)

If you prefer Python over CLI:

```python
from evalview import gate, DiffStatus

result = gate(test_dir="tests/")

if not result.passed:
    # Revert and try again
    for d in result.diffs:
        if not d.passed:
            print(f"FAILED: {d.test_name} — {d.status.value} ({d.score_delta:+.1f})")
```

## Guidelines

- **Always check after changes.** Never skip the gate, even for "small" changes.
- **Revert fast.** Don't try to fix a regression in place — revert first, then iterate.
- **Snapshot improvements.** When you intentionally improve behavior, snapshot it so future checks use the new baseline.
- **Use `--strict` for critical paths.** In sensitive code, fail on any change, not just regressions.
- **One change at a time.** Make atomic changes so regressions are easy to identify.
