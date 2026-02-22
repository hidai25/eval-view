# Tool Categories — Flexible Tool Matching by Intent for AI Agent Tests

> **Problem:** Different AI agents use different tool names for the same action. Your test expects `read_file`, but the agent uses `bash cat`. The test fails even though the behavior is correct.
>
> **Solution:** EvalView's tool categories let you test by intent instead of exact tool name. Define `file_read` and it matches `read_file`, `bash cat`, `text_editor`, and more.

**Problem:** Your test expects `read_file`. Agent uses `bash cat`. Test fails. Both are correct.

**Solution:** Test by *intent*, not exact tool name.

---

## Before (Brittle)

```yaml
expected:
  tools:
    - read_file      # Fails if agent uses bash, text_editor, etc.
```

## After (Flexible)

```yaml
expected:
  categories:
    - file_read      # Passes for read_file, bash cat, text_editor, etc.
```

---

## Built-in Categories

| Category | Matches |
|----------|---------|
| `file_read` | read_file, bash, text_editor, cat, view, str_replace_editor |
| `file_write` | write_file, bash, text_editor, edit_file, create_file |
| `file_list` | list_directory, bash, ls, find, directory_tree |
| `search` | grep, ripgrep, bash, search_files, code_search |
| `shell` | bash, shell, terminal, execute, run_command |
| `web` | web_search, browse, fetch_url, http_request, curl |
| `git` | git, bash, git_commit, git_push, github |
| `python` | python, bash, python_repl, execute_python, jupyter |

---

## Custom Categories

Add project-specific categories in `config.yaml`:

```yaml
# .evalview/config.yaml
tool_categories:
  database:
    - postgres_query
    - mysql_execute
    - sql_run
  my_custom_api:
    - internal_api_call
    - legacy_endpoint
```

---

## Why This Matters

Different agents use different tools for the same task. Categories let you test **behavior**, not **implementation**.

For example, all of these accomplish "read a file":
- Claude Code: `read_file`
- OpenAI: `bash` with `cat`
- Custom agent: `text_editor`

With categories, your test passes for all of them:

```yaml
expected:
  categories:
    - file_read  # All three approaches pass
```

---

## Combining Tools and Categories

You can mix exact tool names and categories:

```yaml
expected:
  tools:
    - my_specific_tool    # Must use this exact tool
  categories:
    - file_read           # Plus any file reading approach
```

---

## Related Documentation

- [Evaluation Metrics](EVALUATION_METRICS.md)
- [CLI Reference](CLI_REFERENCE.md)
