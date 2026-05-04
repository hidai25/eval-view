# Simulation examples

Worked YAML files demonstrating `evalview simulate` — hermetic runs
against declared mocks (or recorded cassettes) for tool calls, LLM
responses, and HTTP.

| File | What it shows |
|---|---|
| `flight-booking.yaml` | Declarative `mocks:` block with subset param matching, a deliberate error-path mock, and expected-behavior assertions that fail if the agent picks the wrong flight. |
| `order-lookup-replay.yaml` + `cassettes/order-lookup-replay.json` | The cassette workflow: no `mocks:` block in the YAML, hermetic replay served entirely from a pre-recorded JSON cassette. |

## Cassette walkthrough

Cassettes capture real tool calls once and replay them deterministically
forever — no live services in CI.

```bash
# Replay hermetically using the bundled cassette:
evalview simulate examples/simulation/order-lookup-replay.yaml \
  --replay --cassette-dir examples/simulation/cassettes

# Re-record after the real backend changes (requires API keys):
evalview simulate examples/simulation/order-lookup-replay.yaml \
  --record --cassette-dir examples/simulation/cassettes
```

The cassette format is documented in
[`docs/SIMULATE.md`](../../docs/SIMULATE.md#record--replay-cassettes).
Per-tool sequential matching keeps replay robust even when the agent
reorders calls between runs; declarative `mocks:` (if present) take
precedence so a single recording can be overridden without
re-recording the whole run.

See `docs/SIMULATE.md` for the full YAML schema and CLI reference.
