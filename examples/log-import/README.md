# Log import examples

`evalview generate --from-log <path>` builds a draft test suite from existing
production logs without live probing. The format is auto-detected from the
file extension and contents; pass `--log-format` to force a specific parser.

## CSV (`traces.csv`)

CSV is the lowest-friction format: it imports cleanly from spreadsheets,
analytics exports, or any tool that can write a row per agent interaction.

The header row identifies the columns:

- **query** (required) — the user prompt. Aliases: `input`, `prompt`,
  `question`, `user_message`, `user_input`.
- **output** (optional) — the agent's response. Aliases: `response`,
  `answer`, `assistant_message`, `result`.
- **tools** (optional) — tool names invoked. Aliases: `tool_calls`,
  `tool_use`, `actions`. Cells may be JSON-list (`["a", "b"]`),
  comma-separated, semicolon-separated, or pipe-separated.

Rows missing a query are skipped with a warning printed to stderr; the
import continues.

```bash
evalview generate --from-log examples/log-import/traces.csv --log-format csv
```

The `--log-format csv` argument is optional when the file has a `.csv`
extension — auto-detection picks it up.

## Other formats

- **JSONL** — one JSON object per line, with `query`/`output`/`tool_calls`
  fields. See the `--from-log` tests for the expected shape.
- **OpenAI** — chat completion log format with `messages` and `choices`.
- **EvalView capture** — proxy capture format with `request` and `response`.

All four formats are auto-detected; you only need `--log-format` to override.
