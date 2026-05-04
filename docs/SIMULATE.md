# `evalview simulate` — pre-deployment what-if testing

> Run your agent tests with declared mocks for tool calls, LLM
> responses, and HTTP. Deterministic, hermetic, free to run in CI.

## Why

The April 2026 agent-eval reports flagged one gap almost everyone
hit: **no way to run the full test suite before the change is live.**
Real tools cost money. Real LLM calls are nondeterministic. Real
HTTP APIs drift. Teams shipped, then found regressions in prod.

`evalview simulate` closes that gap. You declare mocks in the test
YAML, the simulator installs them at the adapter layer, and the
agent runs end-to-end as if the mocks were real. Cost: zero tool
calls, zero HTTP, and — if the agent itself is mocked via response
mocks — zero LLM tokens.

Pairs with [`evalview check`](CLI_REFERENCE.md#check) (golden-baseline
diffing) for a complete pre-flight:

```
evalview simulate tests/ --variants 5   # does the agent still behave right on synthetic scenarios?
evalview check    tests/                # is that behavior identical to the last known-good version?
```

## Quick start

```bash
# 1. Declare mocks in your test YAML (new mocks: section).
# 2. Run the simulator.
evalview simulate tests/my-test.yaml

# Fan-out: 5 deterministic replays with seed advancing per variant.
evalview simulate tests/ --variants 5 --seed 42

# CI-friendly JSON output.
evalview simulate tests/ --json > sim-results.json
```

## YAML reference

A new top-level `mocks:` section on any test case:

```yaml
name: flight-booking-sim
adapter: anthropic
input:
  query: Book the cheapest flight to Paris for next Tuesday.
expected:
  tools: [search_flights, book_flight]
thresholds:
  min_score: 70

mocks:
  # Deterministic RNG seed. Advanced by +1 per variant when --variants N is used.
  seed: 42

  # When true, any tool call / LLM response / HTTP call that doesn't match
  # a mock raises. Default false — unmatched calls fall through to the real
  # adapter so you can mix simulation with real dependencies.
  strict: false

  # ── Tool mocks ──────────────────────────────────────────────
  # Exact match on `tool` plus optional param subset matching.
  tool_mocks:
    - tool: search_flights
      match_params: { to: Paris }         # subset match; any call with to=Paris hits
      returns:                            # anything JSON-serializable
        - { id: FL123, price: 299, airline: Air France }
      latency_ms: 25                      # simulated latency (time.sleep)

    - tool: book_flight
      match_params: { flight_id: FL123 }
      returns: { confirmation: CONF-789, status: confirmed }

    - tool: send_email
      error: "SMTP server down"          # mock raises instead of returning

  # ── LLM response mocks ─────────────────────────────────────
  # Match on a prompt substring (or regex with regex: true).
  # Only consumed by adapters that opt in via install_mock_interceptor.
  response_mocks:
    - match_prompt: summarize
      returns: "Summary: one flight found at $299."

    - match_prompt: "^user:\\s+(\\w+)"
      regex: true
      returns: "Hello, user."

  # ── HTTP mocks ─────────────────────────────────────────────
  # Match outbound HTTP calls from tools or the agent runtime.
  http_mocks:
    - url_pattern: api.amadeus.com
      method: GET
      status: 200
      body: { results: [] }

    - url_pattern: flaky-service
      status: 503
```

## How it works

```
evalview simulate tests/t.yaml
         │
         ▼
  Loader (TestCaseLoader)  ───▶  TestCase with mocks: MockSpec
         │
         ▼
  Simulator(adapter, spec)
         │
         │  wraps:  adapter.tool_executor  →  MockedToolExecutor
         │  calls:  adapter.install_mock_interceptor(self)  (opt-in)
         │
         ▼
  await adapter.execute(query, context)
         │
         ▼
  ExecutionTrace + SimulationResult
         │
         ▼
  ┌─────────────────────────────────────────────┐
  │ Human output     │ --json                   │
  │ · Mocks applied  │ SimulationResult.model_  │
  │ · Variants       │   dump()                 │
  │ · Branches       │                          │
  └─────────────────────────────────────────────┘
```

### Matching rules

**Tool mocks**
1. `tool` must match the call tool name **exactly**.
2. If `match_params` is set, every key/value in it must equal the
   call's parameter (subset match — extra keys in the call are fine).
3. First matching mock wins, in declaration order.
4. On match: `latency_ms` sleeps, then `error` raises or `returns`
   is returned. The `AppliedMock` counter increments.

**Response mocks**
- Adapter must opt in via `install_mock_interceptor(simulator)` and
  call `simulator.response_mock_for(prompt)` before sending to the LLM.
- Default path: substring match. With `regex: true`, the pattern is
  passed to `re.search`.

**HTTP mocks**
- Same opt-in pattern. Adapter calls `simulator.http_mock_for(url, method)`.
- Substring match on URL by default; regex when `regex: true`.

### Fallthrough and strict mode

By default, any call that doesn't match a mock falls through to the
real thing. This is useful when you want to simulate just one flaky
dependency and run the rest live.

Set `strict: true` to make unmatched calls raise
`UnmatchedMockError`. Use strict mode in CI where you want a hermetic
run with no real outbound calls allowed.

## Variants

`--variants N` runs the test N times in sequence, advancing `seed` by
+1 per variant. Because the seed is deterministic, the same
`(test, seed)` pair always produces the same run. Use variants to:

- Stress-test nondeterministic logic (variant 1 might pick tool A,
  variant 2 tool B).
- Record a family of valid paths for golden-variant clustering.
- Measure cost / latency distribution over a deterministic sweep.

Aggregate output:

```
▶ flight-booking-sim  (seed=42, variants=3)
  Mocks applied:
    · tool:search_flights ×3
    · tool:book_flight ×2
  Variants:
    · #0 branch=b0 $0.0142 1800ms
    · #1 branch=b1 $0.0128 1650ms
    · #2 branch=b2 $0.0139 1720ms
  Branches:
    · b0: step-0:search_flights → step-1:book_flight
    · b1: step-0:search_flights → step-1:book_flight
    · b2: step-0:search_flights
```

## CLI reference

```
evalview simulate [TEST_PATH] [OPTIONS]

TEST_PATH        Directory (default: tests/) or single YAML file.

-t, --test       Run only this test by name.
    --seed       Override the seed declared in YAML.
    --variants   Run N deterministic replays (default: 1).
    --json       Emit JSON summary for CI.
    --record     Capture every real tool call into a cassette under
                 .evalview/cassettes/<test>.json.
    --replay     Serve tool calls from the cassette at the same path
                 (hermetic — never touches the network).
    --cassette-dir DIR
                 Override the cassette directory (default: .evalview/cassettes).
    --allow-live Suppress the warning when the adapter has no
                 interception seam. Hermetic modes still raise.
```

Exit code 1 on any test error; 0 on success.

## Adapter capability check

Before every run, the simulator probes the adapter for three
interception layers:

| Layer | Detection | Purpose |
|---|---|---|
| `tools` | `hasattr(adapter, "tool_executor")` | Tool-call swap (the path mocks/cassettes use). |
| `responses` | `callable(adapter.install_mock_interceptor)` | LLM-response pull hooks. |
| `http` | `getattr(adapter, "supports_http_mocks", False)` | Outbound HTTP intercept opt-in. |

The result lands on `SimulationResult.adapter_capability` and renders
under "Adapter capability:" in the human output.

**When the adapter has none of the three** (e.g. an HTTP/streaming
adapter where the agent runs server-side), the simulator escalates:

| Run mode | Behavior |
|---|---|
| Lenient (no `--record` / `--replay` / `mocks.strict`) | Logs a `WARNING`. Drops to `INFO` with `--allow-live`. |
| Hermetic (`--record`, `--replay`, or `mocks.strict=true`) | Raises `UninterceptableAdapterError` before any live call. |

This means a `--replay` against an unsupported adapter fails fast
instead of silently going live with an empty mock layer.

## Record / replay cassettes

Declarative `mocks:` are great when you know exactly what each tool
should return, but they're tedious for agents with dozens of tools or
variable params. Cassettes solve that — record once against the real
backend, replay forever:

```bash
# 1. Once, against a live backend (or a staging fixture):
evalview simulate tests/refund.yaml --record

# 2. Forever after, hermetically — no network, no LLM cost beyond the
#    agent's own model calls (which can be mocked separately):
evalview simulate tests/refund.yaml --replay
```

Cassettes live at `.evalview/cassettes/<test-name>.json` and look like:

```json
{
  "version": 1,
  "test_name": "refund",
  "recorded_at": "2026-05-04T12:34:56+00:00",
  "adapter": "anthropic",
  "interactions": [
    {"kind": "tool", "tool": "lookup_order", "params": {"id": 4812}, "returns": {"id": 4812, "total": 99.0}, "error": null},
    {"kind": "tool", "tool": "process_refund", "params": {"id": 4812}, "returns": {"status": "ok"}, "error": null}
  ]
}
```

**Matching rules.** Per-tool sequential: each tool name has its own
queue of recordings, consumed in declaration order. The agent can
legitimately reorder `lookup_order` vs `check_policy` between runs and
replay still works — only intra-tool sequence matters.

**Precedence.** If both a declarative `mocks:` block and a cassette
exist, declarative mocks win first (so you can override a single
recording without re-recording the whole run); cassette serves the
rest; live executor serves anything still unmatched (unless
`mocks.strict=true`, in which case unmatched calls raise).

**Recording is transparent.** The recorder runs the real executor
first, then captures the result. Errors are recorded too and re-raised
from replay, so the agent sees identical behavior either way.

**Synthetic mocks are not recorded.** Calls that hit a declarative
`tool_mocks` entry never reach the recorder — the cassette only
captures what the real layer actually returned.

## Adapter support matrix

| Adapter | Tool mocks | Response mocks | HTTP mocks |
|---|:---:|:---:|:---:|
| Anthropic | ✅ | via opt-in | — |
| OpenAI Assistants | ✅ | via opt-in | — |
| LangGraph (HTTP) | — | — | — ¹ |
| CrewAI native | ✅ | via opt-in | — |
| HTTP generic | — | — | — ¹ |

¹ HTTP-based adapters run the agent over the wire; tool calls happen
on the server side and aren't interceptable from the CLI. Use the
response/http mock hooks in the server-side adapter if you control it.

All adapters that read `self.tool_executor` (the Python convention)
get tool mocking for free — the simulator patches the attribute
without touching the adapter code.

## Cloud output

When cloud is connected, `simulate` runs are POSTed to the same
`/api/v1/results` endpoint as `check` with `run_type: "simulation"`
so they route to the simulation tab in the dashboard. Cloud never
runs simulations itself — it only renders the `mocks_applied`,
`branches_explored`, and `variant_outcomes` stored on the run.

## Known limitations

- LangGraph Cloud adapter: can intercept tool calls on the agent side
  only via `install_mock_interceptor` (not wired in v1). Use the
  Python-native LangGraph adapter for hermetic simulation.
- HTTP/streaming adapters: no default mock interception; you need to
  implement `install_mock_interceptor` on the server to read
  `simulator.http_mock_for(...)`.
- Simulation does not currently rerun failed variants for statistical
  mode — use `evalview check --statistical` for that.

## Related

- `evalview/core/simulation.py` — engine source
- `evalview/commands/simulate_cmd.py` — CLI
- `docs/RATIONALE.md` — pairs with simulation: record why each variant branched
- `examples/simulation/` — worked YAML examples
