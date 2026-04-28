# `demo-tests/`

Test cases for the `demo-agent/` FastAPI agent. Used by the `evalview demo`
flow as a quick "see EvalView catch a regression" experience.

## Running

Start the agent first (it listens on port 8002):

```bash
cd ../demo-agent && python agent.py &
```

Then from the repo root:

```bash
evalview run demo-tests/
# or, for the full snapshot/check loop:
evalview snapshot demo-tests/
evalview check demo-tests/
```

Configuration lives in `config.yaml` (adapter, endpoint, timeout).

## Related directories

- [`../demo-agent/`](../demo-agent/) — the agent these tests target.
- [`../demo/`](../demo/) — cross-model benchmark fixtures (separate scope).
