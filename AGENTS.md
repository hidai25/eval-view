# Repository Guidelines

## Project Structure & Module Organization
- `evalview/`: Python package and CLI (`cli.py`). Modules: `adapters/` (framework integrations), `core/` (types, loader, pricing), `evaluators/` (cost, latency, output, sequence), `reporters/` (console, JSON).
- `tests/`: Pytest assets and YAML scenarios under `tests/test-cases/*.yaml`.
- `docs/`: Reference docs (adapters, database, backend setup). `examples/`: Example YAML and backend notes.
- Root tooling: `Makefile`, `pyproject.toml`, `requirements.txt`, optional Node workspace (`package.json`).

## Build, Test, and Development Commands
- Install: `make install` (editable) or `make dev-install` (adds dev deps).
- Quality: `make format` (black), `make lint` (ruff), `make typecheck` (mypy), `make check` (all).
- Tests: `make test` (pytest -v) or `make test-cov` (coverage HTML + terminal).
- Run example flow: `make run-example` (initializes `.evalview/` then runs sample case if present).
- CLI test runner: `evalview run [--pattern "*.yaml"] [--verbose]`.

## Coding Style & Naming Conventions
- Python 3.9+. Formatting via black (line length 100); lint with ruff; strict typing with mypy.
- Indentation: 4 spaces. Naming: modules/files `snake_case.py`, classes `PascalCase`, functions/vars `snake_case`.
- Keep public CLI/API stable; extend via new adapters/evaluators under existing folders.

## Testing Guidelines
- Framework: `pytest`. Place Python tests under `tests/`. Name files `test_*.py`.
- Scenario tests: add YAML cases to `tests/test-cases/` (see existing examples). Prefer precise `expected.output.contains` and realistic thresholds (`max_cost`, `max_latency`).
- Run locally with `pytest` or `evalview run --pattern "<case>.yaml" --verbose`.

## Commit & Pull Request Guidelines
- Commits: imperative, concise, and scoped (e.g., "Add LangGraph adapter config"). Group related changes; include rationale when behavior changes.
- PRs: clear description, linked issues, reproduction/testing steps, CLI output or screenshots when useful. Update docs/tests as needed.
- Checks: ensure `make check` and `make test` pass; avoid committing secrets or `.env.local`.

## Security & Configuration Tips
- Secrets: store in `.env.local` (gitignored). Common: `OPENAI_API_KEY`, service endpoints.
- Local config: `.evalview/config.yaml` controls defaults for runs. Avoid hard‑coding credentials in tests or examples.
