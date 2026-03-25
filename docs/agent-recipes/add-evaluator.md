# Recipe: Add an Evaluator

## Goal

Add a new evaluation component and wire it into the top-level `Evaluator`.

## Read These Files First

- `evalview/evaluators/evaluator.py`
- `evalview/core/types.py`
- an existing evaluator in `evalview/evaluators/`

## Requirements

- the new evaluator should return a typed result compatible with `Evaluations`
- decide whether it affects score, pass/fail, reporting, or only diagnostics
- keep deterministic vs LLM-backed behavior clear

## Steps

1. Add a new evaluator module in `evalview/evaluators/`.
2. Extend `Evaluations` or related result types in `evalview/core/types.py` if needed.
3. Instantiate and call the evaluator in `evalview/evaluators/evaluator.py`.
4. Update score computation and pass/fail logic only if intentionally part of the contract.
5. Expose the result in reports if it is user-visible.
6. Add tests.

## Done Criteria

- the evaluation result is present on `EvaluationResult`
- scoring/pass-fail changes are explicit and tested
- report output is updated if the feature is user-facing

## Common Pitfalls

- adding a field to `Evaluations` but not wiring reports
- changing pass/fail semantics accidentally
- mixing deterministic and LLM-backed checks without explicit gating
