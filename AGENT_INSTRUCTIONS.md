# EvalView Agent Instructions

This file is written for coding agents working inside the EvalView repository.
Use it as the fastest way to understand the architecture, the important invariants, and the common extension paths.

## What EvalView Is

EvalView is a regression gate for AI agents.

Its core job is:

1. Load test cases.
2. Execute them against an agent backend.
3. Evaluate outputs, tool use, cost, latency, and safety-related checks.
4. Save baselines with `snapshot`.
5. Compare current behavior against baselines with `check`.
6. Render reviewable terminal and HTML diffs.

EvalView is not just a scorer. It is opinionated around baseline comparison, regression detection, and developer-facing review flows.

## Core Concepts

### `TestCase`

Path: `evalview/core/types.py`

`TestCase` is the declarative input spec. It includes:

- the input prompt or turns
- expected tools and sequences
- thresholds
- optional contains / not_contains / regex / schema checks
- optional per-test adapter and endpoint overrides

If you are changing test-loading or authoring behavior, inspect:

- `evalview/core/types.py`
- `evalview/core/loader.py`

### `EvaluationResult`

Path: `evalview/core/types.py`

`EvaluationResult` is the evaluated output of one executed test. It contains:

- `passed`
- `score`
- `evaluations`
- `trace`
- timestamps and metadata

This is the main object passed into reports, baselines, and diffing helpers.

### `GoldenStore`

Path: `evalview/core/golden.py`

`GoldenStore` persists and loads baselines and variants from `.evalview/golden/`.

Use it for:

- saving baselines during `snapshot`
- loading all variants during `check`
- counting and managing variants

### `DiffEngine`

Path: `evalview/core/diff.py`

`DiffEngine` compares current traces/results against saved goldens.

It produces `TraceDiff`, which drives:

- terminal status output
- HTML diffs
- root-cause analysis
- healing decisions

### Adapters

Primary paths:

- `evalview/adapters/base.py`
- `evalview/core/adapter_factory.py`
- `evalview/adapters/http_adapter.py`

Adapters convert an agent backend into an `ExecutionTrace`.

The adapter contract is strict:

- input: `query` and optional `context`
- output: `ExecutionTrace`
- health checks are optional but useful

### Evaluators

Primary paths:

- `evalview/evaluators/evaluator.py`
- `evalview/evaluators/tool_call_evaluator.py`
- `evalview/evaluators/output_evaluator.py`
- `evalview/evaluators/sequence_evaluator.py`

The top-level `Evaluator` orchestrates all evaluation components and computes the final score and pass/fail result.

Current overall score weighting is:

- tool accuracy: 30%
- output quality: 50%
- sequence correctness: 20%

### Commands

CLI entrypoint:

- `evalview/cli.py`

Common command modules:

- `evalview/commands/check_cmd.py`
- `evalview/commands/check_display.py`
- `evalview/commands/snapshot_cmd.py`
- `evalview/commands/shared.py`
- `evalview/commands/generate_cmd.py`
- `evalview/commands/watch_cmd.py`

`shared.py` contains key execution helpers used by both `snapshot` and `check`.

## System Architecture Map

Use this as the shortest repo map.

- CLI registration: `evalview/cli.py`
- command implementations: `evalview/commands/`
- shared execution helpers: `evalview/commands/shared.py`
- adapters: `evalview/adapters/`
- evaluator orchestration: `evalview/evaluators/evaluator.py`
- core domain types: `evalview/core/types.py`
- diffing: `evalview/core/diff.py`
- golden storage: `evalview/core/golden.py`
- healing engine: `evalview/core/healing.py`
- root-cause analysis: `evalview/core/root_cause.py`
- HTML reports: `evalview/visualization/generators.py`
- project state and streaks: `evalview/core/project_state.py`
- tests: `tests/`

## Runtime Contracts

### HTTP `/execute` Contract

Primary implementation:

- `evalview/adapters/http_adapter.py`

EvalView sends:

```json
{
  "query": "user input",
  "context": {},
  "enable_tracing": true
}
```

The generic HTTP adapter accepts several response shapes, but the safest format is:

```json
{
  "output": "final answer",
  "tool_calls": [
    {
      "name": "lookup_order",
      "arguments": {"order_id": "123"},
      "result": {"status": "found"},
      "latency": 12.5,
      "cost": 0.001
    }
  ],
  "cost": 0.002,
  "latency": 75.0,
  "tokens": {
    "input": 120,
    "output": 48
  }
}
```

Also accepted by the HTTP adapter:

- `response` instead of `output`
- `steps` instead of `tool_calls`
- nested token metadata in `metadata`

### Tracing Requirements

Adapters should preserve as much execution detail as possible.

Minimum useful fields:

- final output
- per-step tool names
- per-step parameters
- per-step latency and cost when available
- total metrics

The target object is `ExecutionTrace` in `evalview/core/types.py`.

### Multi-turn Expectations

Multi-turn execution is handled in:

- `evalview/commands/shared.py`

Important:

- each turn should still map into one combined `ExecutionTrace`
- per-turn outputs and tool paths matter
- `turn_index` and turn metadata should be preserved when available

If you touch multi-turn execution, inspect:

- `evalview/commands/shared.py`
- `evalview/core/types.py`
- `evalview/core/golden.py`
- `evalview/core/diff.py`
- `tests/test_multi_turn_evaluation.py`

## Invariants

Do not violate these without intentionally changing the product contract.

- forbidden tool violations are never auto-healed
- raw check results are preserved; presentation and healing must not silently rewrite source evidence
- healing decisions must be auditable
- diffing policy must remain deterministic
- adapters must return `ExecutionTrace`, not arbitrary JSON blobs
- snapshot/check compatibility must be preserved when changing core types
- report changes should not change underlying evaluation semantics

## Common Tasks For Agents

Use the linked recipes for procedures. These are the common task categories:

- add a new adapter
- add a new evaluator
- add a new report field
- add a CLI option
- integrate Ollama
- update snapshot/check flow

Recipes live in:

- `docs/agent-recipes/add-adapter.md`
- `docs/agent-recipes/add-evaluator.md`
- `docs/agent-recipes/debug-check-vs-snapshot-mismatch.md`
- `docs/agent-recipes/extend-html-report.md`
- `docs/agent-recipes/integrate-ollama.md`

## Change Impact Matrix

Use this before editing.

### If you edit adapters

Also inspect:

- `evalview/adapters/base.py`
- `evalview/core/adapter_factory.py`
- `evalview/core/types.py`
- `tests/test_adapters.py`

### If you edit evaluator behavior or scoring

Also inspect:

- `evalview/evaluators/evaluator.py`
- relevant evaluator module in `evalview/evaluators/`
- `evalview/core/types.py`
- `tests/test_evaluators.py`
- `tests/test_main_evaluator.py`

### If you edit diff semantics

Also inspect:

- `evalview/core/diff.py`
- `evalview/core/root_cause.py`
- `evalview/commands/check_display.py`
- `evalview/visualization/generators.py`
- `tests/test_diff_engine.py`
- `tests/test_root_cause.py`

### If you edit snapshot/check execution flow

Also inspect:

- `evalview/commands/shared.py`
- `evalview/commands/snapshot_cmd.py`
- `evalview/commands/check_cmd.py`
- `evalview/core/golden.py`
- `tests/test_check_cmd.py`
- `tests/test_snapshot_generated_workflow.py`

### If you edit healing

Also inspect:

- `evalview/core/healing.py`
- `evalview/commands/check_cmd.py`
- `evalview/commands/check_display.py`
- `evalview/visualization/generators.py`
- `tests/test_healing.py`

### If you edit HTML reports

Also inspect:

- `evalview/visualization/generators.py`
- `evalview/commands/check_cmd.py`
- `tests/test_visualization_generators.py`

## Verification Commands

Run the narrowest checks that cover your change, then widen if needed.

Typical commands:

```bash
pytest -q tests/test_check_cmd.py tests/test_healing.py tests/test_visualization_generators.py
python -m mypy evalview/commands/check_cmd.py evalview/core/healing.py evalview/visualization/generators.py
python -m evalview check tests/generated --heal --dry-run
python -m evalview check tests/generated --heal --strict
```

For adapter work:

```bash
pytest -q tests/test_adapters.py
```

For diff work:

```bash
pytest -q tests/test_diff_engine.py tests/test_root_cause.py
```

## Safe Change Guidance

Keep these patterns in mind.

- if you edit `evalview/core/types.py`, check loaders, adapters, goldens, and reports
- if you add a CLI flag, check command parsing, behavior, report wiring, and tests
- if you add a new diff field, wire it through terminal output, JSON output, and HTML if user-visible
- if you add a new evaluator, decide whether it affects score, pass/fail, reports, and CI semantics
- if you touch healing, preserve auditability and explicit failure semantics

## Agent Quickstart

If your task is: "set up regression tracking for our new Ollama models"

1. Read:
   - `AGENT_INSTRUCTIONS.md`
   - `docs/agent-recipes/integrate-ollama.md`
2. Inspect:
   - `evalview/adapters/ollama_adapter.py`
   - `evalview/core/adapter_factory.py`
   - `evalview/commands/shared.py`
   - `evalview/core/types.py`
3. Verify:
   - `pytest -q tests/test_adapters.py`
   - `python -m evalview check tests --dry-run`
4. Avoid:
   - bypassing `ExecutionTrace`
   - adding Ollama-specific logic to unrelated diff or report code unless needed

## If You Are An Autonomous Coding Agent Working On EvalView

Inspect these files first before making assumptions:

- `evalview/cli.py`
- `evalview/core/types.py`
- `evalview/commands/shared.py`
- `evalview/core/adapter_factory.py`
- `evalview/evaluators/evaluator.py`
- `evalview/core/diff.py`
- `evalview/core/golden.py`

Run these commands first when feasible:

```bash
rg --files evalview tests
pytest -q tests/test_check_cmd.py tests/test_healing.py tests/test_visualization_generators.py
python -m mypy evalview/commands/check_cmd.py evalview/core/healing.py evalview/visualization/generators.py
```

After every code change, run `pytest -q` on the relevant test files before reporting success.
Do not assume an edit is correct without verification.

Avoid these assumptions:

- do not assume all adapters return the same raw response shape
- do not assume score alone determines pass/fail
- do not assume `check` should mutate original diffs
- do not assume HTML reports are optional for user-facing review flows
- do not assume a green exit code means "no differences"; it only means current fail policy did not trigger

If the task is procedural, prefer using the linked recipes instead of rediscovering the path from scratch.
