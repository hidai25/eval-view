# Recipe: Add an Adapter

## Goal

Add a new EvalView adapter that executes a backend and returns `ExecutionTrace`.

## Read These Files First

- `evalview/adapters/base.py`
- `evalview/core/adapter_factory.py`
- `evalview/adapters/http_adapter.py`
- `evalview/core/types.py`

## Requirements

- implement `AgentAdapter`
- return `ExecutionTrace`
- preserve final output, steps, and metrics when available
- do not bypass SSRF or endpoint validation patterns unless the adapter does not use URLs

## Steps

1. Add the adapter module under `evalview/adapters/`.
2. Implement `name` and `execute()`.
3. Add optional `health_check()` if the backend has a lightweight health endpoint.
4. Register the adapter in `evalview/core/adapter_factory.py`.
5. Update validation or adapter listings if needed.
6. Add tests in `tests/test_adapters.py`.

## Done Criteria

- adapter can be created through factory code
- `execute()` returns a valid `ExecutionTrace`
- tests cover success and failure paths

## Common Pitfalls

- returning provider JSON instead of `ExecutionTrace`
- dropping tool-call parameters
- losing token/cost/latency metadata when it exists
- updating factory code but forgetting tests
