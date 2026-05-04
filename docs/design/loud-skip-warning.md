# Design: loud failure when simulation can't actually intercept

**Status:** proposed
**Author:** EvalView reproducibility track
**Companion:** [SIMULATE.md](../SIMULATE.md), [`evalview/core/cassette.py`](../../evalview/core/cassette.py)

## Problem

The `Simulator` installs mocks/cassettes by **monkey-patching the
adapter's `tool_executor` attribute** ([`simulation.py:225`](../../evalview/core/simulation.py)):

```python
had_attr = hasattr(self._adapter, "tool_executor")
if had_attr:
    setattr(self._adapter, "tool_executor", mocked)
```

When `had_attr` is false — the adapter has no `tool_executor` attribute
to patch and no `install_mock_interceptor` either — the simulator
**silently skips installation**. The agent runs against live
dependencies. There is currently no warning, no error, and no
indication in the output.

This breaks the "hermetic CI" promise for at least these adapters
(based on `grep "tool_executor" evalview/adapters/`):

| Adapter | Has `tool_executor`? | Has `install_mock_interceptor`? |
|---|---|---|
| anthropic_adapter.py | ✅ | — |
| openai_assistants_adapter.py | unknown — needs audit | unknown |
| crewai_native_adapter.py | unknown | unknown |
| http_adapter.py | ❌ (calls happen server-side) | — |
| langgraph_adapter.py | likely ❌ for cloud, ✅ for native | — |
| ollama_adapter.py | unknown | unknown |
| mcp_adapter.py | unknown | unknown |
| (others) | unknown | unknown |

A user who runs `evalview simulate --strict` (or `--replay`) against
an HTTP/streaming adapter will see "Mocks applied: none matched" — the
same output they'd see if their declarative mocks just didn't match.
There is no way to tell whether the run was hermetic or live.

## Goal

Make it impossible to run `simulate`/`replay`/`record` against an
uninterceptable adapter without knowing it.

## Non-goals

- **Don't refuse to run.** Some users explicitly want `simulate` to
  fan out variants against a partially-live adapter. The signal is
  what changes; the run still goes.
- **Don't try to add interception** to remote adapters here. That's a
  separate, larger effort (e.g. server-side `install_mock_interceptor`,
  HTTP-transport wrapping). This design only fixes visibility.

## Proposal

### 1. Capability check on the adapter

Add a small helper in `evalview/core/simulation.py`:

```python
def adapter_simulation_capability(adapter) -> dict:
    """Report which simulation layers this adapter actually supports.

    Returns a dict with three booleans:
      tools:     can intercept tool calls (has tool_executor attr)
      responses: can intercept LLM responses (has install_mock_interceptor)
      http:      can intercept outbound HTTP (declares supports_http_mocks)
    """
    return {
        "tools": hasattr(adapter, "tool_executor"),
        "responses": callable(getattr(adapter, "install_mock_interceptor", None)),
        "http": bool(getattr(adapter, "supports_http_mocks", False)),
    }
```

### 2. Three escalation tiers

| Condition | Behavior | Where |
|---|---|---|
| Adapter supports tools, mocks declared | silent (today) | unchanged |
| Adapter supports tools, no mocks declared | DEBUG log | unchanged |
| Adapter has **none** of the three capabilities, run uses mocks/cassette/replay | **WARN log + stderr banner** | `Simulator.run` / `run_variants` entry |
| Same as above + `MockSpec.strict=True` OR `--replay` flag set | **raise `UninterceptableAdapterError`** | same |

The warning text:

```
⚠ Adapter 'http' cannot intercept tool calls — mocks/cassette declared
  but none will be installed. The run is hitting live services.
  See docs/SIMULATE.md#adapter-support-matrix.
```

The strict/replay error is fail-fast at the start of the run, before
any live call has been made.

### 3. Surface in the JSON / human output

Extend `SimulationResult` (in `types.py`) with one field:

```python
adapter_capability: Dict[str, bool] = Field(default_factory=dict)
```

The `simulate` CLI prints it under the "Mocks applied:" section so
users see the *capability*, not just the absence of matches:

```
▶ flight-search-sim  (seed=42, variants=1)
  Mocks applied: none matched
  Adapter capability: tools=✗ responses=✗ http=✗   ← new line, red when all-false
  Variants:
    · #0 branch=b0 $0.0000 0ms
```

### 4. CLI flag for opt-in to live runs

Add `--allow-live` to `evalview simulate` to suppress the warning when
the user has consciously decided that a partially-live run is fine.
Without the flag, the warning fires every run; with it, the warning
becomes a single-line `INFO`.

## Implementation sketch

Three small changes, all in already-touched files:

1. **`evalview/core/simulation.py`** (~20 lines)
   - Add `adapter_simulation_capability()` helper.
   - Add `UninterceptableAdapterError` class.
   - At the top of `Simulator.run` / `run_variants`, compute
     capability; if all-false AND mocks/cassette/replay are in play,
     either warn (lenient) or raise (strict/replay).
   - Stamp capability into `SimulationResult` for downstream.

2. **`evalview/core/types.py`** (~3 lines)
   - Add `adapter_capability: Dict[str, bool]` to `SimulationResult`.

3. **`evalview/commands/simulate_cmd.py`** (~10 lines)
   - Add `--allow-live` flag.
   - Render the capability line in `_format_human`.

4. **`evalview/adapters/*`** (audit + tag, ~1 line per adapter)
   - For each adapter, confirm whether `tool_executor` exists; if it
     doesn't but the adapter could support `install_mock_interceptor`,
     leave a TODO. (No code change today; just establishes ground
     truth for the matrix.)

## Tests

- `test_simulation.py::test_warns_when_adapter_uninterceptable` —
  fake adapter without `tool_executor`; assert WARN logged and
  stderr banner present.
- `test_simulation.py::test_raises_under_strict_when_uninterceptable`
  — same fake adapter, `MockSpec(strict=True)`, assert
  `UninterceptableAdapterError`.
- `test_simulation.py::test_raises_when_replay_uninterceptable` —
  pass a cassette to a non-tool-executor adapter, assert raise.
- `test_simulation.py::test_allow_live_suppresses_warning` — pass
  through CLI; assert `--allow-live` downgrades to INFO.
- `test_simulation.py::test_capability_recorded_in_result` —
  assert `SimulationResult.adapter_capability` reflects the truth
  for both an interceptable and uninterceptable adapter.

## Rollout

- One PR, no migration. The warning is purely additive; the strict
  raise only fires in modes (`strict=True`, `--replay`) where the
  user has already opted into hermetic semantics — they would *want*
  to be told the run is silently going live.
- Update `docs/SIMULATE.md`'s adapter matrix to use the same
  `tools/responses/http` rubric as the runtime check, so the doc and
  the CLI never disagree.

## Why not just refuse to run?

Mixed live/mock runs are legitimately useful — e.g. `simulate` against
a real LLM but with one flaky tool stubbed. The warning is the floor;
strict mode (which the user opts into) is the ceiling. This matches
the existing `MockSpec.strict` semantics and avoids breaking any
working workflow.

## Open questions

1. Should the `record` flag also raise on uninterceptable adapters?
   It would silently produce an empty cassette today, which is
   confusing. Recommendation: yes, raise — same as `--replay`.
2. Should the capability check distinguish "adapter has the attribute
   but it's `None`" (Anthropic before init) from "adapter doesn't
   support it"? Probably yes — `hasattr` returns true for both today.
3. The `supports_http_mocks` flag is new and will be missing from
   every adapter at first; default `False` is the safe answer but
   means the HTTP-mock column is "✗" everywhere until adapters
   explicitly opt in. That's fine — accurate is better than
   optimistic.
