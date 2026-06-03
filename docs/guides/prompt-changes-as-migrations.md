# Treat Every Prompt Change Like a Database Migration (Using EvalView)

> **TL;DR:** A one-line prompt edit can silently change tool choice, skip a clarification, or degrade output quality days later — with no code change and no failing health check. Treat prompt and model changes the way you treat schema migrations: versioned, diffed against a known-good baseline, and gated by a CI check before they ship.

We changed one sentence in a system prompt. "Be concise" became "Be concise and decisive."

Nothing broke. CI was green. The agent still returned `200`. Three days later support noticed it had stopped asking customers to confirm the shipping address before creating orders — "decisive" had quietly nudged the model to skip the clarification step. By then a few dozen orders had gone to the wrong address.

The fix wasn't a better prompt. It was realizing that **a prompt edit is a behavior change, and behavior changes deserve the same discipline as schema changes.** You don't ship an `ALTER TABLE` without a migration and a review. A prompt edit is no different.

---

## Why prompts deserve migration discipline

A database migration has three properties that make it safe:

1. **It's versioned** — the change is a committed artifact, not an ambient edit.
2. **It's diffable** — you can see exactly what changed before it runs.
3. **It's gated** — it runs through review and CI before it touches production.

Prompt changes usually have none of these. They live in a string, get edited in a hurry, and the only "test" is that the app still boots. The behavior shift — different tool, skipped step, longer output, higher cost — is invisible until a user hits it.

EvalView gives a prompt edit the same three properties: a committed golden baseline (versioned), a behavioral diff (diffable), and a `check` exit code (gated).

---

## Step 1: Snapshot the current behavior as your baseline

Before you touch the prompt, capture what the agent does today. This is your "schema before the migration."

```bash
evalview snapshot
# ✅ Snapshotted: create-order, refund-request, address-confirm, ...
```

This records the full run for each test — tool calls, parameters, sequence, output, model ID, cost, and latency — into `.evalview/golden/`. Commit it:

```bash
git add .evalview/golden/
git commit -m "baseline: agent behavior before prompt v2"
```

Now the baseline is a versioned artifact in your repo, reviewable in a diff like any other migration.

---

## Step 2: Make the prompt change, then diff the behavior

Edit the prompt. Then run the migration's "dry run":

```bash
evalview check
```

Instead of a binary pass/fail, you get a behavioral diff against the baseline:

```
  ✓ create-order        PASSED
  ⚠ address-confirm     TOOLS_CHANGED
      - lookup_customer → confirm_address → create_order
      + lookup_customer → create_order
  ✗ refund-request      REGRESSION  -22 pts
      Score: 88 → 66   Output similarity: 41%
```

That `address-confirm` diff is exactly the regression that bit us — the `confirm_address` step disappeared. It would never show up in a health check, but it shows up here as a tool-sequence diff, before merge.

---

## Step 3: Classify the change — intended or regression?

Not every diff is a bug. Sometimes the new path is correct and the baseline is what's stale. The migration question is: *is this the change I meant to make?*

- **It's a regression** → fix the prompt and re-run `evalview check` until the diff is clean.
- **It's intended** → promote the new behavior to the baseline, the same way you'd commit a migration once you're happy with it:

```bash
evalview golden update address-confirm
git add .evalview/golden/
git commit -m "migration: prompt v2 — drop redundant confirm step (intended)"
```

The commit message records *why* the behavior changed. Six months later, `git blame` on the golden tells you which prompt version changed which behavior — the audit trail a migration gives you for free.

---

## Step 4: Handle non-determinism so the gate isn't flaky

A common objection: "My agent isn't deterministic, so a behavioral diff will be noise." Real concern — a flaky migration gate gets ignored, then disabled.

Run the same change a handful of times to separate real drift from sampling noise:

```bash
evalview check --statistical 5
```

If a path is non-deterministic but still valid, accept it as an additional variant instead of fighting it:

```bash
evalview check --statistical 10 --auto-variant
```

Now the baseline holds *several* acceptable paths, and the gate only fires when behavior leaves that envelope — not every time the model phrases an answer differently.

---

## Step 5: Gate it in CI so no prompt ships unreviewed

The last migration property is the gate. Add EvalView to CI and a behavioral regression blocks the PR the same way a failing migration blocks a deploy:

```yaml
# .github/workflows/evalview.yml
- name: Check for agent regressions
  uses: hidai25/eval-view@v0.8.0
  with:
    openai-api-key: ${{ secrets.OPENAI_API_KEY }}
    fail-on: REGRESSION
```

Now the workflow is:

1. Someone edits a prompt → opens a PR.
2. CI runs `evalview check` against the committed golden.
3. A regression or unexpected tool change posts a diff comment and fails the check.
4. The author either fixes the prompt or, if the change is intended, updates the golden in the same PR — so the behavior change and its baseline land together, reviewed.

A prompt edit can no longer reach production without someone seeing the behavioral diff first.

---

## The mental model

| Database migration | Prompt / model change with EvalView |
|---|---|
| Schema before | `evalview snapshot` → committed golden |
| Migration script | The prompt or model edit in the PR |
| `EXPLAIN` / dry run | `evalview check` behavioral diff |
| Reviewer approves | Diff reviewed in the PR comment |
| Migration runs in CI/CD | `fail-on: REGRESSION` gates the merge |
| Rollback | `git revert` the prompt + golden together |

The point isn't ceremony for its own sake. It's that the most load-bearing logic in an agent often lives in a prompt string, and right now most teams change that string with less rigor than they'd change a column type. EvalView closes that gap.

---

## Getting started

```bash
pip install evalview
evalview snapshot          # capture today's behavior
# ...make your prompt change...
evalview check             # diff the new behavior against the baseline
```

Full docs: [github.com/hidai25/eval-view](https://github.com/hidai25/eval-view)
