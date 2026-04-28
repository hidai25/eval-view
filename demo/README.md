# `demo/`

Cross-model benchmark fixtures and run scripts for evaluating different
agents (Aider, Gemma, Qwen, Sonnet, local-deep-researcher) on the same
suite. This is **not** the same as the small `demo-agent/` used by
`evalview demo`.

## Layout

- `tests/` — YAML test suites grouped by target agent / model.
- `fixtures/` — input source files used by some of the tests
  (`buggy.py`, `messy.py`, `stub.py` for the Aider scenarios).
- `run_benchmark.sh`, `start_gemma.sh`, `start_qwen.sh` — orchestration
  scripts to spin up local model servers and run the suites.

## Related directories

- [`../demo-agent/`](../demo-agent/) — the simple FastAPI agent used by
  `evalview demo`. Different scope from this directory.
- [`../demo-tests/`](../demo-tests/) — the small smoke suite that targets
  `demo-agent/`. Different scope from this directory.
