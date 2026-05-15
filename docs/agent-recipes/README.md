# Agent Recipes

These recipes are short, task-specific guides for coding agents working on EvalView.

Use them when the task is procedural and bounded.

## Available Recipes

- [Add an Adapter](add-adapter.md)
- [Add an Evaluator](add-evaluator.md)
- [Add a Root-Cause Hint](add-root-cause-hint.md) — narrate a new
  cross-test failure pattern (good first issue)
- [Emit OpenTelemetry Spans for an Adapter](add-otel-emission.md) —
  wire EvalView's portable agent semconv into your adapter
- [Add a Goal-Drift Judge](add-goal-drift-judge.md) — plug a smarter
  similarity metric into the goal-drift detector (good first issue)
- [Debug Check vs Snapshot Mismatch](debug-check-vs-snapshot-mismatch.md)
- [Extend the HTML Report](extend-html-report.md)
- [Integrate Ollama](integrate-ollama.md)

## How To Use

1. Read `AGENTS.md` first for architecture and invariants.
2. Pick the closest recipe for the task.
3. Follow the file references and done criteria.
4. Run the narrowest verification commands that cover the change.
