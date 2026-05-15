# Recipe: Add a Chaos Mode

## Goal

Add a new disruption mode to `evalview.core.chaos` so `evalview simulate
--chaos` can inject it. Chaos modes model the real-world failures that
quietly stale benchmarks ignore: tools flaking, users changing their
mind, contexts being corrupted, schemas drifting.

A good mode catches a class of agent failure no current test in the
suite would reveal.

## Read These Files First

- `evalview/core/chaos.py` — `ChaosDisruption`, `ChaosScenario`, the
  `SHIPPED_MODES` registry, the `CHAOS_MODES_ROADMAP` list of
  candidates.
- `evalview/commands/simulate_cmd.py` — the simulate command. The
  scenario plan is consumed here; new modes must integrate with the
  existing simulation loop.
- `tests/test_chaos.py` — testing patterns to mirror.

## Requirements

- **Deterministic.** Given the same `(scenario, seed, step_index)`,
  the disruption must produce the same effect every time. Randomness
  goes through `_seeded_choice(seed, ...)`, never `random.random()`.
- **One step, one disruption.** `build_scenario` rejects two
  disruptions targeting the same step. If your mode conceptually spans
  multiple steps (e.g. "intermittent failures"), model it as multiple
  disruptions.
- **Param-only state.** All mode-specific configuration goes in
  `disruption.params` so the scenario serializes to JSON cleanly.
- **No new dependencies.** The chaos module is intentionally pure.

## Steps

1. **Pick a mode from the roadmap** in `CHAOS_MODES_ROADMAP` (or
   propose a new one in an issue first):
   - `info_drift`
   - `rate_limit`
   - `partial_handoff`
   - `memory_corruption`
   - `schema_drift`
   - `user_typo`

2. **Add a mode constant** at the top of `chaos.py`:

   ```python
   MODE_INFO_DRIFT: str = "info_drift"
   ```

   And append it to `SHIPPED_MODES`. **Do not reorder existing
   entries** — a serialized scenario depends on the registry order
   for stable enum-like behavior.

3. **Add a builder function** following the `tool_failure` /
   `latency_spike` template. Take only the params your mode needs;
   return a `ChaosDisruption`. Defaults should be the most useful
   common case so callers usually pass nothing.

4. **Add a branch in `random_scenario`** so seeded scenario synthesis
   can pick your mode. Use `_seeded_choice(seed, "<unique_label>", i)`
   for any random parameter selection — never `random.random()`.

5. **Wire into `simulate_cmd.py`.** The simulator needs to know how
   to apply your disruption. Add a handler that takes the
   `ChaosDisruption` and modifies the simulated trajectory
   appropriately. Keep handlers small and side-effect-free; mutate
   the simulation state, return.

6. **Remove your mode from `CHAOS_MODES_ROADMAP`** — it shipped.

7. **Add tests.** Mirror `tests/test_chaos.py`:
   - Builder produces a frozen `ChaosDisruption` with your mode and
     params.
   - `random_scenario(seed=...)` is deterministic across two calls.
   - Two disruptions at the same step raise `ValueError`.
   - Simulator integration test: a small simulation that hits your
     mode and asserts the agent's behavior changes accordingly.

## Done Criteria

- New mode constant + builder function in `chaos.py`.
- `random_scenario` knows how to pick your mode.
- Simulator integration handler in `simulate_cmd.py`.
- Mode removed from `CHAOS_MODES_ROADMAP`.
- 4+ new tests passing; full suite green.

## Common Pitfalls

- **Using `random.random()`.** It breaks the determinism contract.
  Always go through `_seeded_choice`.
- **Mutating shared state.** Disruptions are frozen dataclasses; the
  simulator handler should consume them as read-only.
- **Catastrophic disruptions.** A mode that always crashes the agent
  doesn't test anything useful. Aim for disruptions an *adapted*
  agent could recover from.
- **Coupling to one adapter.** The chaos module is adapter-agnostic;
  keep it that way. Adapter-specific recovery logic lives in the
  adapter, not here.
- **Forgetting the OTel attribute.** When a disruption fires, set
  `agent.tool.result.status` (`error` / `timeout` / `rate_limited`
  per `evalview/core/otel_semconv.py`) on the affected span so
  downstream observability sees the chaos in the trace.

## Where the Output Surfaces

- **`evalview simulate`** — runs the scenario and produces traces
  exactly as a normal simulate run would, with the disruptions
  applied.
- **JSON scenario files** — `ChaosScenario.to_dict()` is the
  canonical serialization. Scenarios can be checked in for
  regression testing.

## Roadmap (Already Listed Above)

Each item in `CHAOS_MODES_ROADMAP` is one PR sized for a single
contributor. They roughly rank in usefulness as: `rate_limit` (most
production agents will hit this), `schema_drift` (catches a class of
brittle parsers), `memory_corruption` (pairs naturally with
`retrieval_lineage` stale-memory work), `info_drift`, `user_typo`,
`partial_handoff` (highest-leverage but needs multi-agent traces).

If you're proposing a brand-new mode not on the roadmap, open a
discussion issue with a real-world failure trace it would catch.
