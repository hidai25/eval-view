# Recipe: Add a Root-Cause Hint

## Goal

Teach EvalView to narrate a new cross-test failure pattern. A
"root-cause hint" turns a coordinated incident (3+ tests failing together)
into a one-paragraph diagnosis with concrete next-step commands, instead
of just *"5 tests shifted together — correlated batch failure (low
confidence)"*.

This is one of the highest-leverage contributor entry points in the
codebase: each hinter is a single pure function with a small,
well-bounded contract.

## Read These Files First

- `evalview/core/root_cause_hint.py` — the hinter registry and the
  shipped hinters (use one of them as a template).
- `evalview/core/noise_tracker.py` — where the hint gets attached to an
  `Incident` and consumed by the monitor / notifiers.
- `evalview/core/diff.py` — the `TraceDiff` fields available to your
  hinter (`tool_diffs`, `output_diff`, `score_diff`, `latency_diff`,
  `model_changed`, fingerprints, etc.).
- `tests/test_root_cause_hint.py` — testing patterns to copy.

## Requirements

- **Pure.** No I/O, no network, no LLM. Same `HintContext` → same
  `RootCauseHint`, always.
- **Conservative.** Only fire when the pattern is unambiguous. A
  false-positive narrative is worse than no narrative — it steers humans
  toward the wrong fix.
- **Cross-test.** A hinter is for *coordinated* failures (≥
  `ctx.min_affected` tests sharing a signal). Per-test root cause is
  already handled by `evalview.core.root_cause.analyze_root_cause`.
- **Actionable.** The first entry in `suggested_actions` should be a
  command the operator can copy-paste right now.

## Steps

1. **Pick a pattern from the roadmap** in
   `evalview/core/root_cause_hint.py` (`HINTERS_ROADMAP`) — or propose
   your own in an issue first. Currently open:
   - `coordinated_cost_spike`
   - `coordinated_latency_spike`
   - `coordinated_refusal`
   - `coordinated_parameter_drift`
   - `coordinated_decision_drift`
   - `coordinated_retrieval_drop`

2. **Write the hinter** as a function `hint_<name>(ctx: HintContext) ->
   Optional[RootCauseHint]` next to the existing ones. Mirror the
   structure of `hint_coordinated_tool_addition` — that one's the
   cleanest template.

3. **Choose a priority.**
   - `100`: structural, unambiguous signal (provider rollout).
   - `70–80`: strong indirect signal (fingerprint shift, tool changes).
   - `40–60`: heuristic that catches a common pattern but can be wrong.

4. **Register it** by appending the function to the `HINTERS` tuple. Do
   not reorder existing entries — the order is a deterministic
   tie-breaker and reordering can flip which hint wins on identical
   inputs.

5. **Remove the corresponding line from `HINTERS_ROADMAP`** if it was
   listed there.

6. **Add tests.** Mirror `tests/test_root_cause_hint.py`:
   - One test that the hinter fires on a clean positive case.
   - One test that it does *not* fire below `min_affected`.
   - One test that it does *not* fire on the obvious false-positive
     shape for your pattern.
   - One test that confirms the right `suggested_actions[0]`.

7. **Confirm selection logic.** If your hinter could fire alongside an
   existing one, add an `analyze_root_cause_hint` test that pins which
   one wins.

## Done Criteria

- New hinter is in `HINTERS` and removed from `HINTERS_ROADMAP`.
- `evidence` dict carries the raw signal so cloud/CI can group
  recurrences by `cause_id`.
- `suggested_actions[0]` is the operator's first move.
- 4+ new tests pass; full suite (`make test`) stays green.
- No new dependencies; module is still pure.

## Common Pitfalls

- **Firing too eagerly.** If your hinter would match on a random 3
  unrelated failures, tighten the predicate. The bar is "I'd be willing
  to put this narrative on a Slack alert."
- **Overlapping evidence keys.** Reusing an `evidence["signal"]` value
  from another hinter breaks cloud-side grouping. Use a unique string.
- **Mutating `ctx`.** `HintContext` is frozen; treat it as read-only.
  Building a counter from `ctx.failing` is fine; reassigning fields is
  not.
- **Hardcoding tool names or model IDs.** Hinters live in the open-
  source core — they have to work for any agent. If a heuristic only
  applies to one adapter, gate it on `getattr(diff, ...)` and skip
  silently when the attribute is absent.
- **Forgetting to test the negative case.** Hinters tend to slowly
  loosen their predicates over time; a "does not fire when X" test is
  the cheapest insurance.

## Where the Output Surfaces

- **Slack / Discord** — `slack_notifier.py` and `discord_notifier.py`
  render `incident.hint.narrative` and the top suggested action below
  the headline. Verify your narrative reads well in chat (no walls of
  text, no ANSI codes).
- **Monitor stdout** — appears in the alert log line when an incident is
  detected.
- **JSON** — `hint_to_dict()` is the canonical serialization for any
  `--json` consumer.

If you're unsure whether your pattern is worth a hinter, open a
discussion issue with three real-world traces that would have benefited
from it. That's almost always the right way in.
