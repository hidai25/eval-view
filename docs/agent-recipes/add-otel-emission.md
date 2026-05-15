# Recipe: Emit OpenTelemetry Spans for an Adapter

## Goal

Make an adapter (LangGraph, CrewAI, OpenAI Assistants, your custom one)
emit traces using EvalView's portable agent-layer OTel semantic
conventions. The result: traces from your adapter become consumable by
any tool that adopts the same vocabulary — not just EvalView.

This is the "make traces portable" complaint that's loud in the
community. The contribution here is *additive* — your adapter keeps
emitting whatever it already emits and adds the spec-defined attributes
on top.

## Read These Files First

- `evalview/core/otel_semconv.py` — the full spec. Constants are the
  *only* public contract; never hardcode strings.
- An existing adapter under `evalview/adapters/` — e.g. the LangGraph
  one — to see how trace data flows out.
- Upstream OTel `gen_ai.*` conventions
  (https://opentelemetry.io/docs/specs/semconv/gen-ai/) for the model-
  call layer your spans live alongside.

## Requirements

- **Additive only.** Don't remove or rename existing trace fields. The
  semconv attributes are extras.
- **Pin to the constants.** Import names like `SPAN_AGENT_TOOL_CHOICE`
  and `ATTR_TOOL_NAME` — don't write the strings inline. A spec rev
  will update the constants centrally.
- **Stamp the version.** Set `otel.semconv.version` to
  `OTEL_SEMCONV_VERSION` on the root span so consumers can branch on
  spec rev.
- **No new dependencies.** EvalView doesn't require the OTel Python
  SDK. If your adapter already has it, great — emit real spans. If
  not, attach the same attributes to the trace dict your adapter
  already produces.

## Steps

1. **Pick a span kind.** For each meaningful event your adapter
   already records, map it to one of the names in `SPAN_NAMES`:
   - Top-level invocation → `SPAN_AGENT_RUN`.
   - Each turn in a multi-turn conversation → `SPAN_AGENT_TURN`.
   - Model decision about which tool to call → `SPAN_AGENT_TOOL_CHOICE`.
   - Tool execution → `SPAN_AGENT_TOOL_CALL`.
   - Handoff to another agent or human → `SPAN_AGENT_HANDOFF`.
   - Vector / keyword retrieval → `SPAN_AGENT_RETRIEVAL`.
   - Memory store read/write → `SPAN_AGENT_MEMORY_READ` / `_WRITE`.
   - Human approval / edit / escalation → `SPAN_AGENT_INTERVENTION`.

2. **Look up the recommended attributes** with
   `attributes_for_span(span_name)`. Set every key in that set when you
   have the data. Skip keys you genuinely can't fill — partial coverage
   beats faking it.

3. **Always set the identity attributes** (`agent.id`, `agent.name`,
   `agent.framework`, `agent.run.id`, `agent.turn.index`). Consumers
   need them to join spans across processes.

4. **Fingerprint, don't dump.** For state, parameters, and goals,
   prefer a hash (`agent.*.fingerprint`) over the full payload unless
   you've already paid the cost to log the payload. Fingerprints make
   spans cheap and storage-friendly.

5. **Truncate text fields** to `RATIONALE_MAX_TEXT_BYTES` (defined in
   `evalview/core/types.py`) before emitting `agent.tool.choice.reason`,
   `agent.goal.text`, etc. Long fields are how token-cost spirals start.

6. **Add an adapter test.** Use `is_known_span()` and
   `is_known_attribute()` to assert that every span your adapter emits
   is in the spec and uses no off-spec attributes. This catches typos
   and prevents silent vocabulary drift.

## Done Criteria

- Every documented event in your adapter emits a span with a name from
  `SPAN_NAMES`.
- Identity attributes are present on every span.
- The root span carries `otel.semconv.version = OTEL_SEMCONV_VERSION`.
- The adapter's test suite includes a "no off-spec attributes" check.
- No hardcoded strings — everything goes through the constants.

## Common Pitfalls

- **Inventing new attribute names.** If you find yourself wanting a new
  key, add it to the spec via PR (and bump the spec version if it's
  breaking). Local extensions fragment the ecosystem.
- **Overloading a span kind.** `SPAN_AGENT_TOOL_CALL` is for *tool
  execution*, not for *the model's choice to call a tool*. Use both
  spans: the choice as `TOOL_CHOICE`, the execution as `TOOL_CALL`,
  with the call nested under the choice as a child span.
- **Logging full prompts as attributes.** OTel attributes have size
  limits — and even where they don't, your bill does. Hash, sample, or
  reference an external blob store.
- **Dropping span context across handoffs.** When agent A hands off to
  agent B, agent B's root span must set `agent.parent_run.id` to A's
  `agent.run.id`. Otherwise the trace tree fragments and consumers
  can't reconstruct the multi-agent flow.

## Where the Output Surfaces

- **Any OTel-aware backend** (Tempo, Jaeger, Honeycomb, Langfuse,
  Phoenix). Vendors that adopt the spec get a richer view automatically.
- **EvalView's own consumers.** `evalview fleet` and the rationale
  events use the same names; if you emit them on your adapter, the
  cross-cutting commands "just work" against your traces.

## Roadmap (Good First Issues)

These are concrete next steps that need a contributor. Each is a
self-contained PR.

- **Wire the conventions into the LangGraph adapter.** The trace data
  is already there; map it to the constants.
- **Add a `evalview validate-trace FILE` subcommand** that reads a
  JSONL of spans and reports off-spec attributes/spans.
- **Add a JSON-Schema export** for the spec so non-Python tools can
  validate.
- **Add a `verdict_at_step` extension attribute** so consumers can
  reconstruct verdict transitions over time without re-running checks.

If you're proposing a new attribute or span, open a discussion issue
first — the spec is governed by the same "don't fragment the
ecosystem" principle as upstream OTel.
