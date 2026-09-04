# Migrating the OpenAI adapter to Responses

OpenAI's Assistants API shut down on **August 26, 2026**. It is no longer
available. See the [official migration guide](https://developers.openai.com/api/docs/assistants/migration).

## Release status

As of September 5, 2026:

| Installation | OpenAI adapter behavior |
| --- | --- |
| PyPI / GitHub release **0.8.1** | Uses the removed Assistants API. Upgrading the OpenAI SDK alone does not fix it. |
| Current development source | Uses Responses. Requires `openai>=1.101.0` and an explicit configuration migration. |

The migration has **not been published to PyPI**. A plain `pip install -U evalview`
still installs 0.8.1. Check the [release notes](https://github.com/hidai25/eval-view/releases)
before assuming a packaged version includes it.

Until a release is available, use a reviewed source checkout containing the migration
and its follow-up fixes, in a separate environment:

```bash
# Run from that reviewed EvalView checkout.
python -m venv .venv-migration
source .venv-migration/bin/activate
python -m pip install .
git rev-parse HEAD  # Record the source revision alongside your eval results.
```

Development source may still report the previous package version. The source revision,
not that version string alone, identifies the migration. Maintainers must follow the
[release checklist](RELEASING.md) before publishing it.

## Preserve the agent you intended to test

The names `openai-assistants` and `openai` remain valid. **Assistant objects do not
migrate automatically.** An old `assistant_id` / `OPENAI_ASSISTANT_ID` cannot load
the old model, instructions, files, or tools. Legacy-only configuration now fails
with migration guidance instead of silently testing a default agent.

Choose one configuration source:

1. Copy the intended model, instructions, and tools into `.evalview/config.yaml`.
2. Create a reusable Prompt in the OpenAI dashboard and provide `prompt_id`
   (or `OPENAI_PROMPT_ID`). An explicit model setting overrides the Prompt's model;
   omit it to retain the Prompt's choice.

For a text-only agent:

```yaml
adapter: openai-assistants
timeout: 120
model:
  name: gpt-4o
instructions: "You are a support assistant. Ask for an order ID before looking up an order."
tools: []
```

For a dashboard-managed Prompt:

```yaml
adapter: openai-assistants
timeout: 120
prompt_id: pmpt_your_prompt_id
```

Set `OPENAI_API_KEY` in your environment; this SDK adapter needs no HTTP endpoint.
Remove obsolete assistant IDs after moving the configuration. The same settings
must reach `run`, `snapshot`, `check`, and the programmatic adapter factory.

## Tools and conversations

- Hosted `code_interpreter` runs through Responses. Its default container is automatic.
- `file_search` requires explicit vector stores, supplied through
  `OPENAI_VECTOR_STORE_IDS` (comma-separated) or a complete tool definition:

  ```yaml
  tools:
    - type: file_search
      vector_store_ids: [vs_your_vector_store_id]
  ```

- Omit `tools` for the adapter default; use `tools: []` to disable tools explicitly.
  A Prompt's tools are preserved unless overridden.
- Custom function calls are requests to execute your code, not proof of execution.
  These adapters do not execute arbitrary custom functions. Such a response fails
  clearly instead of claiming that a command ran or a file was created. To evaluate
  your own function execution loop, expose the agent through the
  [HTTP adapter](ADAPTERS.md), including the actual tool results in its trace.
- Multi-turn tests replay the preceding user/assistant messages supplied by EvalView.
  This does not migrate old `thread_id` histories or preserve hidden state from an
  old Assistant. The skills adapter is a separate execution path; do not assume it
  resumes an Assistants thread.
- Responses does not expose per-item latency. Do not interpret an even cost split
  or missing step timing as provider-measured tool metrics.

See the [OpenAI example](../examples/openai-assistants/README.md) for a starting suite.

## Verify before accepting new baselines

Keep a copy of your existing baselines and use a disposable project copy for the
first migration run. Confirm the configured model, instructions, hosted tools,
vector stores, and multi-turn behavior match the agent you intended to test.

```bash
evalview check --no-judge --strict --json
```

`--no-judge` avoids a separate judging request; executing the OpenAI agent still
requires API access and incurs provider costs. Review the raw trace and differences.
Only run `evalview snapshot` after the changed behavior is understood and intentionally
accepted. A new API transport is not a reason to silently bless changed agent behavior.
