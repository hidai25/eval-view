# `demo-agent/`

A minimal FastAPI agent used by the `evalview demo` flow and by the
`demo-tests/` suite. Listens on `http://127.0.0.1:8002/execute` by default
and supports calculator and weather tools (with multi-tool sequences).

## Running it

```bash
cd demo-agent
pip install -r requirements.txt
python agent.py
```

## Related directories

- [`../demo-tests/`](../demo-tests/) — the YAML tests that target this agent.
  Start the agent first, then run `evalview run demo-tests/`.
- [`../demo/`](../demo/) — broader cross-model benchmark fixtures (Aider,
  Gemma, Qwen, Sonnet, local-deep-researcher), independent of this agent.
