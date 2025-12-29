# GooseBench

**Regression tests for [Goose](https://github.com/block/goose)** - Block's open-source AI agent.

GooseBench tests whether Goose actually uses tools before answering, or just guesses (hallucinates).

## Quick Start

### 1. Install Goose

```bash
curl -fsSL https://github.com/block/goose/releases/download/stable/download_cli.sh | bash
goose configure  # Set up your LLM provider
```

### 2. Run GooseBench

```bash
# From any project directory
cd /path/to/any/project

# Run the benchmark
evalview run --pattern examples/goosebench/tasks/
```

### 3. View Results

```bash
# Generate report from latest results
evalview report .evalview/results/latest.json
```

## What GooseBench Tests

| Task | What It Catches |
|------|-----------------|
| List files | Agent guesses files without running `ls` |
| Read README | Agent summarizes without reading the file |
| Count .py files | Agent says "about 12" without running `find` |
| Search TODOs | Agent makes up TODOs without `grep` |
| Git history | Agent invents commits without `git log` |

**The villain:** Agents that sound confident but didn't actually check.

## Example Scorecard

```
GooseBench v0.1 | Goose 1.0.0

✅ 8/10 tasks passed

❌ Task 2: "Read README and summarize"
   Expected tools: [bash]
   Actual tools: []
   → Answered without reading the file

❌ Task 5: "Count Python files"
   Expected tools: [bash]
   Actual tools: []
   → Said "approximately 15" without counting
```

## The 10 Tasks

1. **List files** - Must use `ls` or `find`
2. **Read README** - Must read before summarizing
3. **Explore codebase** - Must look at files before explaining
4. **Search TODOs** - Must use `grep` to find them
5. **Count .py files** - Must count, not estimate
6. **Create a file** - Must actually write it
7. **Read .gitignore** - Must read, not guess typical patterns
8. **Git log** - Must run `git log`
9. **List dependencies** - Must read package files
10. **Last commit** - Must check git before summarizing

## Interpreting Results

| Score | Meaning |
|-------|---------|
| 10/10 | Goose always uses tools before answering |
| 7-9/10 | Minor regressions - some tasks skipped verification |
| <7/10 | Significant issues - Goose is hallucinating answers |

## Generate Scorecard Image

```bash
python examples/goosebench/scripts/scorecard.py .evalview/results/latest.json
```

This generates a markdown scorecard you can share on social media.

## Why This Matters

Traditional evals test: *"Did the agent get the right answer?"*

GooseBench tests: *"Did the agent actually check, or just guess?"*

An agent can get lucky with a guess. GooseBench catches when agents skip the work.

---

Built with [EvalView](https://github.com/hidai25/EvalView) - the testing framework for AI agents.
