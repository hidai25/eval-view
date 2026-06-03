# EvalView vs Promptfoo

If you are comparing **EvalView vs Promptfoo**, the difference is:

- **Promptfoo** is strongest at prompt and model comparison — running a matrix of prompts/models against a set of assertions to pick the best combination.
- **EvalView** is strongest as a regression testing system for AI agents — snapshotting agent behavior and catching drift against a committed baseline.

## Choose Promptfoo when

- you are iterating on a single prompt or chain and want to A/B many variants
- you want a fast eval matrix across prompts × models with assertion scoring
- your unit of testing is mostly a prompt's input/output, not a multi-step trajectory

## Choose EvalView when

- you need **regression testing for agent behavior**, not just prompt comparison
- you care about **tool-call and sequence diffs**, not only final output assertions
- you want a **golden baseline** that fails CI when behavior drifts from a known-good run
- you want silent model-change detection, hermetic record/replay, and PR-comment diffs
- you want a fast zero-traffic onboarding story:

```bash
evalview generate --agent http://localhost:8000
```

## The core distinction

Promptfoo answers *"which prompt/model scores best on my assertions right now?"* — a forward-looking comparison.

EvalView answers *"did my agent's behavior change from the baseline I already approved?"* — a backward-looking regression gate. It diffs the whole trajectory (tools called, parameters, order, output, cost, latency), not just the final string, which is what catches a tool-sequence change that still produces a plausible answer.

## Best fit together

Use Promptfoo while you are choosing a prompt/model. Once you have a setup you trust, snapshot it with EvalView and let the regression gate protect it in CI from that point on.
