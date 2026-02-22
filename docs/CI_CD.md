# CI/CD Integration — Automated AI Agent Testing in GitHub Actions, GitLab, and CircleCI

> **Problem:** How do you prevent broken AI agents from reaching production? Manual testing doesn't scale, and LLM outputs are non-deterministic.
>
> **Solution:** EvalView integrates with CI/CD pipelines to automatically run agent regression tests on every PR and block merges when behavior degrades. It provides a GitHub Action, proper exit codes, JSON output, and PR comment support.

EvalView is CLI-first. You can run it locally or add to CI.

---

## GitHub Action (Recommended)

Use the official EvalView GitHub Action for the simplest setup:

```yaml
name: EvalView Agent Tests

on: [push, pull_request]

jobs:
  test-agents:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Run EvalView
        uses: hidai25/eval-view@v0.2.1
        with:
          openai-api-key: ${{ secrets.OPENAI_API_KEY }}
          diff: true
          fail-on: 'REGRESSION'
```

---

## Action Inputs

| Input | Description | Default |
|-------|-------------|---------|
| `openai-api-key` | OpenAI API key for LLM-as-judge | - |
| `anthropic-api-key` | Anthropic API key (optional) | - |
| `diff` | Compare against golden baselines | `false` |
| `fail-on` | Statuses that fail CI (REGRESSION, TOOLS_CHANGED, OUTPUT_CHANGED) | `REGRESSION` |
| `config-path` | Path to config file | `.evalview/config.yaml` |
| `filter` | Filter tests by name pattern | - |
| `max-workers` | Parallel workers | `4` |
| `max-retries` | Retry failed tests | `2` |
| `fail-on-error` | Fail workflow on test failure | `true` |
| `generate-report` | Generate HTML report | `true` |
| `python-version` | Python version | `3.11` |

---

## Action Outputs

| Output | Description |
|--------|-------------|
| `results-file` | Path to JSON results |
| `report-file` | Path to HTML report |
| `total-tests` | Total tests run |
| `passed-tests` | Passed count |
| `failed-tests` | Failed count |
| `pass-rate` | Pass rate percentage |

---

## Full Example with PR Comments

```yaml
name: EvalView Agent Tests

on:
  pull_request:
    branches: [main]

jobs:
  test-agents:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Run EvalView
        id: evalview
        uses: hidai25/eval-view@v0.2.1
        with:
          openai-api-key: ${{ secrets.OPENAI_API_KEY }}

      - name: Upload results
        uses: actions/upload-artifact@v4
        with:
          name: evalview-results
          path: |
            .evalview/results/*.json
            evalview-report.html

      - name: Comment on PR
        uses: actions/github-script@v7
        with:
          script: |
            github.rest.issues.createComment({
              issue_number: context.issue.number,
              owner: context.repo.owner,
              repo: context.repo.repo,
              body: `## EvalView Results\n\n✅ ${`${{ steps.evalview.outputs.passed-tests }}`}/${`${{ steps.evalview.outputs.total-tests }}`} tests passed (${`${{ steps.evalview.outputs.pass-rate }}`}%)`
            });
```

---

## Manual Setup (Alternative)

If you prefer manual setup:

```yaml
name: EvalView Agent Tests

on: [push, pull_request]

jobs:
  evalview:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
      - run: pip install evalview
      - run: evalview run --pattern "tests/test-cases/*.yaml"
        env:
          OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
```

---

## Configurable Strictness

Control what fails your CI:

```bash
# Default: only fail on score drops
evalview run --diff --fail-on REGRESSION

# Stricter: also fail on tool changes
evalview run --diff --fail-on REGRESSION,TOOLS_CHANGED

# Strictest: fail on any change
evalview run --diff --strict
```

---

## Exit Codes

| Scenario | Exit Code |
|----------|-----------|
| All tests pass, all PASSED | 0 |
| All tests pass, only warn-on statuses | 0 (with warnings) |
| Any test fails OR any fail-on status | 1 |
| Execution errors (network, timeout) | 2 |

---

## Regression Detection in CI

Block deploys when behavior regresses:

```yaml
- name: Run EvalView
  uses: hidai25/eval-view@v0.2.1
  with:
    openai-api-key: ${{ secrets.OPENAI_API_KEY }}
    diff: true              # Compare against golden baselines
    fail-on: 'REGRESSION'   # Block merge on regression
```

See [Golden Traces](GOLDEN_TRACES.md) for setting up baselines.

---

## GitLab CI

```yaml
evalview:
  image: python:3.11
  script:
    - pip install evalview
    - evalview run --pattern "tests/test-cases/*.yaml"
  variables:
    OPENAI_API_KEY: $OPENAI_API_KEY
```

---

## CircleCI

```yaml
version: 2.1
jobs:
  evalview:
    docker:
      - image: python:3.11
    steps:
      - checkout
      - run: pip install evalview
      - run: evalview run --pattern "tests/test-cases/*.yaml"
```

---

## Related Documentation

- [Golden Traces](GOLDEN_TRACES.md)
- [CLI Reference](CLI_REFERENCE.md)
- [Suite Types](SUITE_TYPES.md)
