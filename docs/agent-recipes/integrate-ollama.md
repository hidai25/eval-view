# Recipe: Integrate Ollama

## Goal

Set up regression tracking for local Ollama models with minimal cloud dependency.

Decision rule:

- If your task involves changing how EvalView talks to Ollama, use Path A.
- If your task is setting up an existing project to use Ollama, use Path B.

## Read These Files First

- `evalview/adapters/ollama_adapter.py`
- `evalview/core/adapter_factory.py`
- `evalview/commands/shared.py`
- `README.md`

## Existing Support

EvalView already has an Ollama adapter at `evalview/adapters/ollama_adapter.py`.

It sends requests to:

- `http://localhost:11434/v1/chat/completions`

and converts the OpenAI-compatible response into `ExecutionTrace`.

## Typical Setup Paths

### Path A: direct adapter work

Use or extend `OllamaAdapter` if you need:

- new request parameters
- better token or trace capture
- health-check behavior

### Path B: project setup

Use EvalView against a project configured for Ollama by:

1. setting adapter/config values
2. creating or updating tests
3. running `snapshot`
4. running `check`

## Useful Commands

```bash
evalview check tests --dry-run
evalview snapshot
evalview check
```

If you need local model selection in eval flows, inspect:

- `evalview/commands/shared.py`
- `evalview/core/config.py`
- `evalview/core/llm_provider.py`

## Done Criteria

- EvalView can execute tests against an Ollama-backed agent
- traces preserve output and token/latency metadata where available
- `snapshot` and `check` run without adapter-specific hacks elsewhere

## Common Pitfalls

- confusing an Ollama-backed HTTP agent with the built-in Ollama adapter
- hardcoding model-specific logic into diffing or reports
- forgetting to test local health/failure behavior
