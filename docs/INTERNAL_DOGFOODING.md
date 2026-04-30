# Internal Dogfooding

This is the smallest operational version of "EvalView evaluates EvalView."

The point is not to add process theater.
The point is to make the internal build loop repeatable:

1. write a short spec
2. implement with agent help
3. review hard
4. run the relevant dogfood slice
5. snapshot intentional changes when behavior changed
6. run `check`
7. ship

## Canonical Internal Slices

These are the default internal slices for EvalView development.

### `mcp`

Use when changing:

- `evalview/mcp_server.py`
- MCP contracts
- MCP tool schemas
- MCP command routing

Run:

```bash
make dogfood-mcp
```

### `healing`

Use when changing:

- `evalview/core/healing.py`
- healing policy
- healing audit/reporting behavior
- model-update recovery behavior

Run:

```bash
make dogfood-healing
```

### `snapshot`

Use when changing:

- `snapshot` command behavior
- golden storage behavior
- baseline creation / reset / variants

Run:

```bash
make dogfood-snapshot-core
```

### `check`

Use when changing:

- `check` command behavior
- diffing
- root-cause summaries
- tag filtering
- fail-on / strict semantics

Run:

```bash
make dogfood-check-core
```

### `reporting`

Use when changing:

- HTML report rendering
- CLI regression presentation
- PR/CI comment reporting
- diff explanation rendering

Run:

```bash
make dogfood-reporting
```

### `agent_docs`

Use when changing:

- `README.md`
- `AGENTS.md`
- `docs/agent-recipes/`
- agent-native guidance

This slice is partly manual today.

Run:

```bash
make dogfood-agent-docs
```

Then manually review:

- `README.md`
- `AGENTS.md`
- `docs/agent-recipes/README.md`

## Feature-to-Eval Matrix

Use this as the default ship gate.

| Change type | Must run |
|-------------|----------|
| MCP feature or contract change | `make dogfood-mcp` and `make dogfood-check-core` |
| Healing policy or audit change | `make dogfood-healing`, `make dogfood-check-core`, `make dogfood-reporting` |
| Snapshot / golden behavior change | `make dogfood-snapshot-core` and `make dogfood-check-core` |
| Check / diff / root-cause change | `make dogfood-check-core` and `make dogfood-reporting` |
| HTML / CLI report change | `make dogfood-reporting` |
| Agent docs / recipes / README change | `make dogfood-agent-docs` |
| Cross-cutting core change | `make dogfood-core` |

## Default Ship Loop

For most changes:

1. write a short spec
2. implement
3. review
4. run the slice from the matrix above
5. if behavior intentionally changed, update snapshot/baseline
6. run a broader `check` slice if the change touched core behavior
7. ship

## Bug-to-Test Rule

Every meaningful dogfood failure should become at least one of:

- a new test
- a stronger test
- a new tag
- a documented invariant

If a failure keeps happening and never becomes an eval asset, the loop is broken.

## Current Limits

This is intentionally lightweight.

It does **not** yet provide:

- automatic slice selection
- enforced ship gates in CI
- automatic bug-to-test conversion

Those can come later if they prove useful.

For now, the goal is simple:

**make the internal EvalView build loop explicit and repeatable.**
