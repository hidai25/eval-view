# EvalView canary suites

The canary is a small, stable set of prompts that `evalview model-check`
runs against closed-weight LLMs to detect silent behavioral drift over time.

This directory holds the bundled suites that ship with EvalView itself.
Users can also pass a custom suite via `evalview model-check --suite
path/to/my-suite.yaml`; that path is usually more interesting for teams
who want drift detection on *their own* prompts, not ours.

## Why the suite is small and strict

The canary exists to answer one question: **"Is this model behaving the
same way it behaved last week?"** It is NOT a leaderboard, not a
comparative benchmark, and not a test of model quality. It is a single
model compared against its past self on a fixed set of structural tasks.

Every prompt in a canary suite must follow these rules:

1. **Structural scoring only.** Every prompt has exactly one of these
   scorers: `tool_choice`, `json_schema`, `refusal`, or `exact_match`.
   No LLM-judge-scored prompts. No free-form summarization. No "does
   this answer sound good" checks. Noise from judge variance swamps
   real drift signal and we do not add that risk to v1.
2. **Deterministic.** Runs are pinned to `temperature=0.0`,
   `top_p=1.0`. Any prompt whose answer depends on sampling variance
   does not belong here.
3. **No current events, no dates, no stateful context.** Answers must
   not change with the calendar or with knowledge cutoff.
4. **Prompts are versioned, never mutated.** Changing any prompt, any
   scorer, or any `expected` field is a new suite version (`suite.v2.*`).
   Old snapshots become incomparable and the CLI enforces this via a
   suite-hash check.
5. **Every expected outcome is verifiable in seconds by a human.** If a
   reviewer cannot tell at a glance whether a model passed or failed, the
   prompt is too ambiguous for drift detection.

## Files

- `suite.v1.public.yaml` — 15 prompts, public, stable. Run by default.
- `suite.v1.held-out.yaml` — 5 prompts, **rotated quarterly**. Exists as
  a sanity check against unintentional overfitting of the public suite
  into training data. Do not pin workflows to specific held-out prompts.

## File format

```yaml
suite_name: canary
version: v1.public
description: EvalView bundled canary suite v1
prompts:
  - id: tool_choice_refund            # unique, kebab_case
    category: tool_choice             # one of: tool_choice | json_schema | refusal | exact_match
    prompt: |
      A customer says: "I was charged twice for order #42. Please refund."
      Tools available: lookup_order, check_policy, process_refund.
      Use the right tool first.
    scorer: tool_choice
    expected:
      tool: lookup_order              # scorer-specific config (see below)
      position: 0                     # optional — require exact position
    notes: First step must always be lookup before refund.
```

### Scorer-specific `expected` fields

| Scorer        | Required                              | Optional     |
|---------------|----------------------------------------|--------------|
| `tool_choice` | `tool: <str>`                          | `position: <int>` |
| `json_schema` | `schema: <JSON Schema dict>`           | —            |
| `refusal`     | `should_refuse: <bool>`                | —            |
| `exact_match` | `pattern: <regex>`                     | —            |

## Adding or changing prompts

1. Open a discussion first. Canary changes move slowly on purpose —
   every change invalidates every old snapshot in the wild.
2. If you are adding a new prompt, prefer the held-out suite. The
   public suite grows slowly and deliberately.
3. If you are rotating the held-out suite, bump the suite version in
   the YAML header. Never silently replace prompts within the same
   version.
4. Run `evalview model-check --dry-run --suite <your-edited-suite>` to
   confirm the new suite parses and the cost estimate is sane.

## Why public suites get gamed — and how we mitigate

Any popular public benchmark eventually ends up as a row in some lab's
eval corpus, intentionally or not. The held-out rotating suite is our
mitigation: if public-suite scores and held-out scores diverge
significantly over time for the same model, it's a sign that the public
suite is being gamed and we should refresh it. EvalView maintainers
rotate the held-out prompts quarterly to make this check meaningful.
