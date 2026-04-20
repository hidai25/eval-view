# Simulation examples

Worked YAML files demonstrating `evalview simulate` — hermetic runs
against declared mocks for tool calls, LLM responses, and HTTP.

| File | What it shows |
|---|---|
| `flight-booking.yaml` | Tool mocks with subset param matching, a deliberate error-path mock, and expected-behavior assertions that fail if the agent picks the wrong flight. |

Run all examples:

```bash
evalview simulate examples/simulation/ --variants 3
```

See `docs/SIMULATE.md` for the full YAML schema and CLI reference.
