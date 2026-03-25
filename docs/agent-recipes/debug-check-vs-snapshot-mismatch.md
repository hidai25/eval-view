# Recipe: Debug Check vs Snapshot Mismatch

## Goal

Debug cases where `evalview snapshot` and `evalview check` appear to disagree about the same test.

## Read These Files First

- `evalview/commands/shared.py`
- `evalview/commands/check_cmd.py`
- `evalview/commands/snapshot_cmd.py`
- `evalview/core/golden.py`
- `evalview/core/diff.py`
- `evalview/core/types.py`

## Typical Symptoms

- a test snapshots successfully but immediately shows as changed in `check`
- a tool path looks correct in one flow but different in the other
- baseline metadata or model info appears missing or inconsistent
- multi-turn behavior passes in one command path and drifts in the other

## Debug Flow

1. Confirm the same test file and test name are being used in both commands.
2. Confirm the same adapter and endpoint are being used.
3. Inspect whether `snapshot` and `check` are both routing through the same helper behavior in `evalview/commands/shared.py`.
4. Inspect what was persisted in `.evalview/golden/` via `GoldenStore`.
5. Inspect `TraceDiff` generation in `evalview/core/diff.py`.
6. Check whether model metadata, per-turn data, tool sequences, or parameter diffs are being preserved differently between flows.

## Useful Commands

```bash
python -m evalview snapshot --preview
python -m evalview check --dry-run
python -m evalview check --strict
pytest -q tests/test_check_cmd.py tests/test_snapshot_generated_workflow.py tests/test_diff_engine.py
```

## Done Criteria

- the mismatch cause is identified as either execution, persistence, or diffing
- the fix is covered by tests in the relevant command/core modules
- snapshot/check semantics remain consistent after the fix

## Common Pitfalls

- debugging terminal output instead of inspecting golden persistence and diff generation
- assuming the adapter returned identical trace structure in both paths
- fixing display code when the real bug is in `GoldenStore` or `DiffEngine`
