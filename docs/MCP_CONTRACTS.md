# MCP Contract Testing

Detect when external MCP servers change their interface before your agent breaks.

## The Problem

When you use MCP servers you don't own (Scenario 2), the server can change its
tool definitions at any time: rename parameters, remove tools, add required fields.
Your agent tests pass today and fail tomorrow — not because your code changed,
but because the server did.

## The Solution

MCP contract testing captures a snapshot of a server's tool definitions and diffs
against it on every CI run. If the interface changed, you know immediately — before
running your full test suite.

This mirrors EvalView's golden baseline system:
- **Golden traces** detect when your agent's *behavior* drifts
- **MCP contracts** detect when an external server's *interface* drifts

## Quick Start

### 1. Snapshot a server

```bash
evalview mcp snapshot "npx:@modelcontextprotocol/server-github" --name server-github
```

Output:
```
Snapshot saved: .evalview/contracts/server-github.contract.json
Tools discovered: 8
  - create_issue
  - list_issues
  - create_pull_request
  - ...
```

### 2. Check for drift

```bash
evalview mcp check server-github
```

If the server changed:
```
CONTRACT_DRIFT - 2 breaking change(s)
  REMOVED: create_pull_request - tool 'create_pull_request' no longer available
  CHANGED: list_issues - new required parameter 'owner'
```

### 3. Use in CI

```bash
evalview run tests/ --contracts --fail-on "REGRESSION,CONTRACT_DRIFT"
```

The `--contracts` flag checks all saved contracts *before* running tests.
If any contract drifted, the run aborts immediately — no point testing against
a broken interface.

## CLI Reference

### `evalview mcp snapshot`

Capture tool definitions from an MCP server.

```bash
evalview mcp snapshot <endpoint> --name <server-name> [--notes "..."] [--timeout 30]
```

| Argument | Description |
|----------|-------------|
| `endpoint` | MCP server endpoint (e.g., `npx:@modelcontextprotocol/server-github`) |
| `--name` | Human-readable identifier for this contract (required) |
| `--notes` | Optional notes about this snapshot |
| `--timeout` | Connection timeout in seconds (default: 30) |

Supports all MCP transport types:
- **stdio**: `"npx:@modelcontextprotocol/server-filesystem /tmp"`
- **HTTP**: `"http://localhost:8080"`
- **Command**: `"stdio:python my_server.py"`

### `evalview mcp check`

Compare current server interface against a saved contract.

```bash
evalview mcp check <name> [--endpoint <override>] [--timeout 30]
```

| Argument | Description |
|----------|-------------|
| `name` | Contract name (from `--name` in snapshot) |
| `--endpoint` | Override endpoint (default: use endpoint from snapshot) |

Exit codes:
- `0` — No breaking changes
- `1` — Breaking changes detected (CONTRACT_DRIFT)
- `2` — Could not connect to server

### `evalview mcp list`

List all saved contracts.

```bash
evalview mcp list
```

### `evalview mcp show`

Show full details of a contract including all tool schemas.

```bash
evalview mcp show <name>
```

### `evalview mcp delete`

Remove a contract.

```bash
evalview mcp delete <name> [--force]
```

## Integration with `evalview run`

The `--contracts` flag adds a pre-flight check to any test run:

```bash
evalview run tests/ --contracts
```

This checks all contracts in `.evalview/contracts/` before running tests.
Combine with `--fail-on CONTRACT_DRIFT` to fail CI on drift:

```bash
evalview run tests/ --contracts --fail-on "REGRESSION,CONTRACT_DRIFT"
```

Or use `--strict` (now includes CONTRACT_DRIFT):

```bash
evalview run tests/ --contracts --strict
```

## GitHub Actions

```yaml
- name: Run EvalView
  uses: hidai25/eval-view@v0.2.1
  with:
    diff: true
    contracts: true
    fail-on: 'REGRESSION,CONTRACT_DRIFT'
```

## What Gets Detected

### Breaking changes (trigger CONTRACT_DRIFT)

| Change | Example |
|--------|---------|
| Tool removed | `create_pull_request` no longer exists |
| Required parameter added | New required param `owner` on `list_issues` |
| Parameter removed | `repo` param no longer accepted |
| Parameter type changed | `limit` changed from `string` to `integer` |
| Parameter became required | `owner` was optional, now required |

### Informational changes (logged, don't fail)

| Change | Example |
|--------|---------|
| New tool added | `merge_pull_request` now available |
| Optional parameter added | New optional param `labels` on `create_issue` |
| Description changed | Tool description updated |

## Contract File Format

Contracts are stored as JSON in `.evalview/contracts/`:

```json
{
  "metadata": {
    "server_name": "server-github",
    "endpoint": "npx:@modelcontextprotocol/server-github",
    "snapshot_at": "2026-02-07T10:30:00",
    "protocol_version": "2024-11-05",
    "tool_count": 8,
    "schema_hash": "a1b2c3d4e5f67890"
  },
  "tools": [
    {
      "name": "create_issue",
      "description": "Create a new issue in a GitHub repository",
      "inputSchema": {
        "type": "object",
        "properties": {
          "repo": { "type": "string" },
          "title": { "type": "string" },
          "body": { "type": "string" }
        },
        "required": ["repo", "title"]
      }
    }
  ]
}
```

Commit these files to your repo so CI can use them.

## Best Practices

1. **Snapshot after verifying** — Run your tests first, confirm everything works,
   then snapshot. The contract represents a known-good interface.

2. **Refresh periodically** — If a contract is >30 days old, `evalview mcp check`
   will warn you. Re-snapshot to accept intentional changes.

3. **One contract per server** — Name contracts after the server, not the tools.
   `server-github` not `create-issue-tool`.

4. **Commit contracts** — Store `.evalview/contracts/` in git. They're small JSON
   files and CI needs them.

5. **Check before testing** — Use `--contracts` on `evalview run` so drift is
   caught before wasting time on tests that will fail anyway.
