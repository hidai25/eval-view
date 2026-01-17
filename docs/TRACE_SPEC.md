# EvalView Trace Specification v1.0

This document defines the trace contract that ALL adapters must follow. Every span, every export, and every report uses this specification.

## Version

```yaml
trace_spec_version: "1.0"
```

All traces include this version at the trace level (not per-span) to enable future evolution without breaking old traces/exports.

## Span Types

| Type | When Used | Description |
|------|-----------|-------------|
| `agent` | Root span | Top-level execution boundary |
| `llm` | Any LLM API call | Model inference (Claude, GPT, Ollama, etc.) |
| `tool` | Tool/function execution | Function calls, tool use |
| `mcp` | MCP server calls | Model Context Protocol operations |
| `http` | External HTTP requests | Non-LLM HTTP calls |
| `retrieval` | RAG/vector lookups | (Future) Embedding searches |

## Trace-Level Fields

Every trace MUST include:

```yaml
trace_id: "a1b2c3d4e5f6g7h8"  # UUID, 16 chars hex
run_id: "eval-20240115-143022"  # Optional, links related traces
source: "eval" | "trace_cmd" | "chat"  # How the trace was generated
trace_spec_version: "1.0"

# Execution context (optional but recommended)
command: "python my_agent.py"
cwd: "/path/to/project"
git_sha: "abc123def"  # Optional, for reproducibility

# Timing
started_at: "2026-01-15T14:30:22.123Z"  # ISO 8601
ended_at: "2026-01-15T14:30:26.456Z"

# Aggregates
total_cost_usd: 0.47
total_tokens: 15847
total_llm_calls: 12
total_tool_calls: 5
total_latency_ms: 4233

# Tags (flexible key-value for filtering)
tags:
  test_name: "booking_flow"
  suite: "regression"
  adapter: "anthropic"
  framework: "custom"
```

## Span-Level Fields

### Required Attributes (ALL spans)

```yaml
span_id: "a1b2c3d4"          # UUID, 8 chars hex
parent_span_id: null         # Nullable, null for root span
trace_id: "a1b2c3d4e5f6g7h8" # Links to parent trace
span_type: "llm"             # One of: agent, llm, tool, mcp, http
name: "claude-sonnet-4"       # Human-readable identifier

# Timing
start_time: "2026-01-15T14:30:22.123Z"
end_time: "2026-01-15T14:30:23.456Z"
latency_ms: 1333.0

# Status
status: "success" | "error"
error_message: null          # Populated if status == "error"
```

### LLM Span Attributes

For spans with `span_type: "llm"`:

```yaml
llm:
  provider: "anthropic" | "openai" | "ollama" | "google" | "grok" | "huggingface"
  model: "claude-sonnet-4-5-20250929"  # Exact model identifier

  # Token counts
  input_tokens: 1247
  output_tokens: 523
  cached_tokens: 0           # Anthropic cache, OpenAI cached prompts

  # Cost (null if unknown pricing)
  cost_usd: 0.02             # Calculated cost, null if unknown

  # Content sizes (ALWAYS stored, for debugging without leaking content)
  prompt_chars: 4521
  completion_chars: 1893

  # Content previews (ONLY with --trace-include-content flag)
  # These are OPT-IN to prevent accidental PII/secret leakage
  prompt_preview: "You are a helpful..."   # First 200 chars
  completion_preview: "I'll help you..."   # First 200 chars

  # Completion metadata
  finish_reason: "end_turn" | "tool_use" | "max_tokens" | "stop"

  # Streaming info (if applicable)
  streamed: false
  time_to_first_token_ms: null
```

### Tool Span Attributes

For spans with `span_type: "tool"`:

```yaml
tool:
  tool_name: "get_weather"

  # Argument sizes (ALWAYS stored)
  tool_args_bytes: 45
  tool_result_bytes: 1203

  # Success indicator
  tool_success: true

  # Content previews (ONLY with --trace-include-content flag)
  tool_args_preview: '{"city": "NYC"}'      # First 200 chars of JSON
  tool_result_preview: '{"temp": 72, ...}'  # First 500 chars
```

### MCP Span Attributes

For spans with `span_type: "mcp"`:

```yaml
mcp:
  server_name: "filesystem"
  tool_name: "read_file"

  # Same structure as tool spans
  tool_args_bytes: 32
  tool_result_bytes: 4096
  tool_success: true

  # MCP-specific
  protocol_version: "1.0"
```

## Privacy and Security

### Default Behavior (Safe)

By default, traces store:
- Sizes and counts (chars, bytes, tokens)
- Hashes for deduplication (optional)
- Metadata and timing

Traces do NOT store by default:
- Actual prompts or completions
- Tool arguments or results
- Any content that could contain secrets/PII

### Opt-in Content (`--trace-include-content`)

When explicitly enabled:
- Previews are captured (first N chars)
- A warning is printed once per session
- Sensitive keys are auto-sanitized

### Auto-Sanitization

The following keys are ALWAYS redacted from content previews:
- `api_key`, `apikey`, `api-key`
- `authorization`, `auth`
- `token`, `access_token`, `refresh_token`
- `secret`, `password`, `passwd`
- `cookie`, `session`
- `credential`, `credentials`

Redacted format: `"api_key": "[REDACTED]"`

## JSONL Export Format

When using `--trace-out <file>`, traces are written as newline-delimited JSON:

```jsonl
{"type": "trace_start", "trace_id": "...", "trace_spec_version": "1.0", "started_at": "..."}
{"type": "span", "span_id": "...", "span_type": "agent", "name": "...", ...}
{"type": "span", "span_id": "...", "span_type": "llm", "name": "claude-sonnet-4", ...}
{"type": "span", "span_id": "...", "span_type": "tool", "name": "get_weather", ...}
{"type": "trace_end", "trace_id": "...", "ended_at": "...", "total_cost_usd": 0.47, ...}
```

This format allows:
- Streaming writes (no need to buffer entire trace)
- Easy parsing with standard tools (`jq`, Python)
- Append-friendly (multiple traces in one file)

## Console Output Format

Live trace output follows this format:

```
━━━ Trace Started ━━━
[agent] Agent Execution

  [llm] claude-sonnet-4 → 1,247 in / 523 out → $0.02 (1.3s)
  [tool] get_weather → success (0.2s)
  [llm] claude-sonnet-4 → 892 in / 234 out → $0.01 (0.9s)
  [tool] book_flight → success (0.5s)

━━━ Trace Summary ━━━
💰 Total cost:    $0.03
⏱️  Total time:    2.9s
🔄 LLM calls:     2
🔧 Tool calls:    2

Slowest: claude-sonnet-4 (1.3s)
Most expensive: claude-sonnet-4 ($0.02)
```

### Color Coding

| Color | Meaning | Threshold |
|-------|---------|-----------|
| Green | Fast/Cheap | < 1s or < $0.01 |
| Yellow | Moderate | 1-3s or $0.01-$0.05 |
| Red | Slow/Expensive | > 3s or > $0.05 |

Errors are always red regardless of timing.

## Adapter Compliance

All adapters MUST:

1. **Emit spans following this spec** - Required attributes must be present
2. **Handle streaming** - Accumulate tokens for streaming responses
3. **Capture errors** - Set status and error_message on failures
4. **Track retries** - Include retry_count attribute when applicable
5. **Support nesting** - parent_span_id must correctly reference parent

Adapters SHOULD:

1. **Calculate costs** - Use pricing table, null if unknown
2. **Track time-to-first-token** - For streaming responses
3. **Include git_sha** - When running in a git repository

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | 2026-01 | Initial specification |
