# `evalview model-check` — closed-model drift detection

> Detect when a closed LLM (Claude, GPT, ...) silently changes behavior
> on a fixed canary suite. No LLM judge. No calibration required.

Your product depends on a provider model like `claude-opus-4-5` or
`gpt-5.4`. One Tuesday the model responds differently to the same
prompt, nothing in your code changed, and your users notice before you
do. `evalview model-check` is built for that exact moment.

It runs a small, stable set of **structural** prompts — tool choice,
JSON schema, refusal behavior, exact-answer logic — directly against
the provider, then compares the results against snapshots from
previous runs. If the model's behavior drifted, you see it.

## Quick start

```bash
# First run — saves a baseline snapshot.
evalview model-check --model claude-opus-4-5-20251101

# A week later — detects drift from the baseline.
evalview model-check --model claude-opus-4-5-20251101
```

That's it. No config file. No agent to set up. One command, one
answer.

## What it checks

The bundled canary suite has **15 structural prompts** spread across
four categories:

| Category      | Prompts | Scored by                                         |
|---------------|---------|---------------------------------------------------|
| Tool choice   | 5       | Did the model call the expected tool?             |
| JSON schema   | 4       | Does the output parse and validate against a schema? |
| Refusal       | 3       | Did the model refuse (or comply) as expected?      |
| Exact match   | 3       | Does the output match a regex anchor?             |

**Why only structural scoring?** Because fuzzy scoring across two runs
of the same model drowns real drift in sampling noise. A structural
"did the tool call match?" is either true or false — no judge, no
gray area, no calibration problem.

## How drift is decided

Every `model-check` invocation produces a **snapshot**. On the second
run, EvalView compares the new snapshot against two anchors:

1. **Reference snapshot** — the first snapshot ever taken (or one you
   explicitly pinned). Never auto-updated. This is what lets you detect
   *gradual* drift: the reference stays fixed while the model drifts
   away from it.
2. **Latest prior snapshot** — the run right before this one. Shows
   day-over-day change.

Classification is based on how many prompts flipped direction (pass →
fail or fail → pass) and whether the provider gave us a fingerprint
change:

| Signal                                                    | Classification |
|-----------------------------------------------------------|-----------------|
| Provider fingerprint changed (OpenAI only)                | **STRONG**      |
| ≥ 2 prompts flipped direction                             | **MEDIUM**      |
| 1 prompt flipped, or pass-rate moved > 1%                 | **WEAK**        |
| Everything stable                                         | **NONE**        |

## Per-provider signal strength

Not all providers expose the same drift signal, and we label it
honestly in every output:

| Provider  | Signal source                          | Strength | Notes |
|-----------|----------------------------------------|----------|-------|
| OpenAI    | `system_fingerprint` per response       | **strong** | Changes when the serving model is updated |
| Anthropic | Requested model ID only                 | **weak**   | Drift must be inferred from behavior |
| Mistral   | Requested model ID only                 | **weak**   | Same as Anthropic |
| Cohere    | Requested model ID only                 | **weak**   | Same |
| Local (Ollama) | Model file hash                    | **strong** | Deterministic file hash |

If the provider gives us weak fingerprint signal, you'll see
`[weak — behavior-only]` in the CLI output. Take it seriously: it
means drift has to be inferred from canary results alone. You will
see fewer STRONG classifications on Anthropic than on OpenAI, and
that's an honest reflection of what the provider exposes.

## Cost control

Running the full canary once on Opus with the default 3 runs/prompt
is about 45 API calls. That lands around **$0.30–0.60 per invocation**
on Opus and roughly half that on GPT-5-mini.

Every invocation enforces a budget cap (default `$2.00`) before any
API call is made. If the estimated cost exceeds `--budget`, the
command refuses to run and tells you the estimate.

Use `--dry-run` to preview the cost without touching the API:

```bash
evalview model-check --model claude-opus-4-5-20251101 --dry-run
```

```
Would run: claude-opus-4-5-20251101
  Suite:           canary v1.public (15 prompts × 3 runs = 45 calls)
  Provider:        anthropic
  Estimated cost:  $0.4725
  Budget cap:      $2.00
```

## Flags you might care about

| Flag                    | Default     | Purpose |
|-------------------------|-------------|---------|
| `--model <id>`           | *(required)* | Model ID (e.g. `claude-opus-4-5-20251101`) |
| `--provider <name>`      | auto-detect  | `anthropic` or `openai` in v1 |
| `--suite <path>`         | bundled      | Custom canary YAML (recommended for teams) |
| `--runs <N>`             | `3`          | Runs per prompt for variance |
| `--budget <usd>`         | `2.00`       | Hard cap; refuse to run if estimate exceeds |
| `--dry-run`              | off          | Print cost estimate and exit |
| `--pin`                  | off          | Pin this run as the new reference |
| `--reset-reference`      | off          | Delete existing reference before the run |
| `--compare-to <spec>`    | n/a          | (Reserved for v1.1) |
| `--out <path>`           | n/a          | Write full JSON output to a file |
| `--no-save`              | off          | Do not persist the snapshot (one-off runs) |
| `--json`                 | off          | Emit machine-readable JSON instead of human output |

## Custom suites (recommended for teams)

The bundled canary is a good default, but the most valuable use of
`model-check` is running **your own** prompts over time. Drop a custom
suite in YAML:

```yaml
suite_name: acme_canary
version: v1.2026q2
prompts:
  - id: product_classification
    category: tool_choice
    prompt: |
      A customer writes: "The widget I ordered arrived broken."
      Tools: classify_intent, lookup_order, issue_refund.
      Call the right one first.
    scorer: tool_choice
    expected:
      tool: classify_intent
      position: 0
```

Run with `--suite ./acme_canary.yaml`. EvalView tracks drift for your
custom suite exactly like the bundled one, with its own separate
reference and history.

## Suite versioning

Canary suites are content-hashed. Any change to any prompt, scorer, or
expected block produces a new hash, and drift comparisons refuse to
compare across different hashes with a clear error:

```
Suite hash differs: current sha256:def456…, prior sha256:abc123…
The canary suite changed; old snapshots are not comparable.
Run with --reset-reference to start a new baseline.
```

This is intentional: if the suite changed, the old results mean
nothing. Reset the reference and start fresh.

## Exit codes

| Code | Meaning |
|------|---------|
| `0`  | No drift detected |
| `1`  | Drift detected (any `MODEL` classification) |
| `2`  | Usage error (bad args, missing API key, suite error, cost over budget) |

Suitable for cron. A reasonable wrapper:

```bash
#!/bin/bash
evalview model-check --model claude-opus-4-5-20251101 --json > /tmp/result.json
case $? in
  0) ;;  # no drift
  1) slack_notify "Claude drift detected" /tmp/result.json ;;
  *) slack_notify "model-check failed" /tmp/result.json ;;
esac
```

## Storage

Snapshots live under `.evalview/model_snapshots/<model-id>/`:

```
.evalview/model_snapshots/claude-opus-4-5-20251101/
├── 2026-04-01T14-03-11Z.json
├── 2026-04-02T14-18-44Z.json
├── 2026-04-09T09-22-17Z.json
└── reference.json              # the pinned baseline
```

Automatic pruning keeps the last 50 timestamped snapshots per model.
The reference is never pruned.

## What `model-check` is NOT

- **Not a leaderboard.** It compares a model against its own past
  behavior, not against other models.
- **Not a quality benchmark.** A model that never improves will still
  score PASSED on the canary. The point is *change detection*, not
  quality measurement.
- **Not a replacement for `evalview check`.** `check` validates your
  *agent*; `model-check` validates the *model underneath your agent*.
  Both are useful and independent.

## FAQ

**Q: Can I run this on a schedule?**
Yes. Cron it, wrap the exit code, pipe into Slack. A future v1.1 may
add a first-class `--watch` flag, but scheduled runs work fine today.

**Q: Why not use `evalview check` for this?**
`check` uses your full test suite against your agent and needs golden
baselines recorded from your agent. `model-check` uses a fixed canary
against the raw provider with no agent involved, so the signal is
about the model itself rather than your integration.

**Q: Will the public canary get overfit by labs?**
Possibly, over time. We mitigate two ways: (1) the bundled held-out
suite rotates quarterly, and (2) teams that care most should run
their own custom suite via `--suite`. If public and held-out scores
diverge for the same model, it's a sign the public suite is being
gamed and we'll rotate it.

**Q: Why no judge-scored prompts?**
Judge noise is larger than real model drift for any non-trivial
scoring task. Adding an uncalibrated judge here would produce
unreliable drift alerts. Judge-scored prompts can land in v1.1 after
judge calibration is in place.
