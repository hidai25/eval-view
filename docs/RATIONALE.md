# Decision Rationale — structured "why" logging for agent decisions

> Record why your agent picked one option over another at every branch,
> so regressions surface before your users notice. No aggregation
> overhead locally, cloud-side analytics optional.

Agent observability has traced *what* happened for years (tool A, then
tool B, then a response). The April 2026 reports from the eval
community flagged the missing half: *why did the agent branch that
way?* EvalView captures that as structured data alongside every trace,
so:

- **Local HTML replay** shows the choice, the alternatives, and any
  model-reported reasoning inline with the timeline.
- **Cloud analytics** (optional) group decisions across runs by a
  stable `input_hash`, surfacing drift like "the agent used to pick
  `cached_search` 95% of the time on this prompt; today it's 40%."

No framework lock-in. Deterministic. Runs local-first.

## What gets captured

A `RationaleEvent` is emitted at every decision point:

```json
{
  "step_id": "tool-42",
  "turn": 3,
  "decision_type": "tool_choice",
  "chosen": "edit_file",
  "alternatives": ["read_file", "search", "grep"],
  "rationale_text": "User asked for an edit and I have the file read.",
  "input_hash": "a3f1...",
  "model_reported_confidence": 0.82,
  "truncated": false
}
```

`decision_type` is one of:

| Type | When it fires | Where |
|---|---|---|
| `tool_choice` | Any time the agent picks a tool to call | All supported adapters |
| `branch` | Multi-agent handoff / graph node transition | CrewAI |
| `refusal` | Model declined to act | Reserved for future |
| `retry` | Model retried after an error | Reserved for future |

## Adapter support

| Adapter | `tool_choice` | `branch` | Reasoning text |
|---|:---:|:---:|---|
| Anthropic | ✅ | — | `thinking` blocks (when enabled) |
| OpenAI Assistants | ✅ | — | Not exposed by Assistants API |
| LangGraph | ✅ | — | — |
| CrewAI (native) | ✅ | ✅ | — |
| Others | — | — | — |

Adding capture to an adapter is ~10 lines — construct a
`RationaleCollector`, call `capture_tool_choice(...)` at each tool
dispatch, attach `collector.events()` to the returned `ExecutionTrace`.

## Viewing rationales

### Local HTML replay

Any `evalview check --report report.html` or `evalview simulate`
report with captured rationales now includes a **Decision Rationale**
section inside the Trace Replay tab. Each event is collapsible and
shows the chosen option, alternatives considered, and — when the model
supplied it — the reasoning text and a confidence pill.

![rationale card — one line per decision, expand to see reasoning]

### Cloud analytics (optional)

When cloud is connected (`evalview login`), rationale events are sent
with every run and surface three views server-side:

1. **Decision-drift chart** — for any `input_hash`, shows the
   distribution of chosen options over time. Alerts fire when the
   distribution shifts materially (JS divergence threshold).
2. **Cross-run search** — "show every run where the agent chose
   `delete_file` in the last 30 days."
3. **Branch causal graph** — for multi-agent runs, renders the
   handoff graph (CrewAI agent A → agent B, with decision types and
   counts on edges).

Cloud never runs the agent or holds API keys — it only stores the
events the CLI already emits locally. See `docs/CLOUD.md`.

## `input_hash` — the grouping key

Cross-run grouping is the point. `input_hash` is a SHA-256 of
`prompt + normalized tool_state + extra`, with `tool_state` passed
through `json.dumps(..., sort_keys=True)` so key order doesn't matter.

Two runs with the same `input_hash` represent "same situation, let's
see if the agent made the same call." Different `chosen` values across
the same `input_hash` over time is the decision-drift signal.

The collector computes it automatically via `capture_tool_choice()` /
`capture_branch()`. If you call `capture()` directly, you pass the
hash in.

## Caps and wire size

Decision events can grow with agent complexity. The collector enforces
hard caps so a runaway agent can't blow up memory or payload size:

| Cap | Value | Source of truth |
|---|---|---|
| Events per run | 500 | `RATIONALE_MAX_EVENTS_PER_RUN` |
| `rationale_text` bytes | 4096 | `RATIONALE_MAX_TEXT_BYTES` |

When `rationale_text` is truncated, the event's `truncated` flag flips
to `true` so the UI can show it. Events past the per-run cap are
silently dropped after a one-shot warning — so logs stay quiet even
on pathological runs.

Cloud Zod validators mirror both caps. If you tune them in the OSS
types, coordinate a cloud deploy at the same time.

## Writing a custom adapter hook

Your adapter only needs to import the collector and call one method
per decision point:

```python
from evalview.core.rationale import RationaleCollector
from evalview.core.types import ExecutionTrace

async def execute(self, query, context=None):
    rationale = RationaleCollector()

    # ... your agent loop ...
    for tool_call in tool_calls:
        rationale.capture_tool_choice(
            step_id=tool_call.id,
            chosen_tool=tool_call.name,
            available_tools=[t["name"] for t in self.tools],
            prompt=query if first_turn else None,
            tool_state={"prior_tools": [s.tool_name for s in steps]},
            rationale_text=tool_call.thinking_block,  # optional
        )
        # ... execute the tool ...

    return ExecutionTrace(
        # ... existing fields ...
        rationale_events=rationale.events(),
    )
```

Full API in `evalview/core/rationale.py`. The collector is
single-threaded, deterministic, and has no network I/O.

## FAQ

**Does this slow down my tests?**
No. The collector is in-memory only; capture cost is a few
microseconds per event. HTML rendering is the same.

**What if my model doesn't emit reasoning text?**
Events are still useful without it — the chosen tool, alternatives
considered, and `input_hash` are the primary signal for drift
detection. `rationale_text` is a bonus when the model gives it.

**Can I opt out?**
Yes. The field defaults to `[]` for adapters that don't emit
rationales, and the capture call is a one-liner to comment out for
adapters you control. Cloud simply drops the field.

**How does this compare to LangSmith traces?**
LangSmith logs prompt/completion pairs and timings — same level as
EvalView's `trace_context`. Rationale is one level up: structured
decision records with cross-run grouping keys, designed for
*aggregation* across many runs rather than deep-dive on a single
session.

## Related

- `evalview/core/rationale.py` — collector API
- `docs/SIMULATE.md` — pairs naturally with simulation for what-if testing
- `evalview/reporters/html_reporter.py` — HTML rendering path
