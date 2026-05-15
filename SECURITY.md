# Security Policy

## Supported Versions

We release patches for security vulnerabilities in the following versions:

| Version | Supported          |
| ------- | ------------------ |
| 0.1.x   | :white_check_mark: |

## Reporting a Vulnerability

If you find a security vulnerability, **do not open a public issue.**

Report via [GitHub Security Advisory](https://github.com/hidai25/eval-view/security/advisories/new) or email security@evalview.com.

Include: vulnerability type, affected file paths, reproduction steps, and impact assessment. Proof-of-concept code helps if you have it.

**Response timeline:** initial response within 48 hours, assessment within 5 business days, patch typically within 30 days.

## Security Best Practices for Users

When using EvalView, please follow these security best practices:

### API Keys and Secrets

- **Never commit API keys**: Always use environment variables or `.env` files (which are gitignored)
- **Rotate keys regularly**: Rotate OpenAI API keys and other credentials periodically
- **Use least privilege**: Grant API keys only the minimum required permissions

### Test Case Security

- **Sanitize test data**: Avoid including sensitive data in test cases
- **Review before sharing**: Ensure test cases don't contain proprietary information
- **Validate inputs**: When writing custom adapters, validate and sanitize all inputs

### Agent Security

- **Isolate test environments**: Run agent tests in isolated/sandboxed environments
- **Monitor costs**: Set up billing alerts for API providers
- **Review agent actions**: Regularly audit tool calls and agent behaviors in traces

### Dependencies

- **Keep updated**: Regularly update EvalView and its dependencies
- **Review dependencies**: Use tools like `pip-audit` to check for known vulnerabilities
- **Lock versions**: Use `requirements.txt` or `poetry.lock` to pin dependency versions

## Built-in Security Features

### SSRF (Server-Side Request Forgery) Protection

EvalView includes built-in protection against SSRF attacks. By default in production mode, requests to the following destinations are blocked:

- **Private IP ranges**: 10.0.0.0/8, 172.16.0.0/12, 192.168.0.0/16
- **Loopback addresses**: localhost, 127.0.0.0/8
- **Cloud metadata endpoints**: 169.254.169.254 (AWS, GCP, Azure)
- **Link-local addresses**: 169.254.0.0/16
- **Internal hostnames**: kubernetes.default, metadata.google.internal

#### Configuration

For local development, SSRF protection allows private URLs by default. To enable strict mode in production:

```yaml
# .evalview/config.yaml
allow_private_urls: false  # Block private/internal networks (recommended for production)
```

#### Security Considerations

- When running EvalView in production environments, set `allow_private_urls: false`
- Be cautious when loading test cases from untrusted sources - they can specify arbitrary endpoints
- Review test case YAML files before running them in sensitive environments

### LLM Prompt Injection Mitigation

The LLM-as-judge feature includes protections against prompt injection attacks:

1. **Output Sanitization**: Agent outputs are sanitized before being sent to the LLM judge
   - Long outputs are truncated (default: 10,000 chars) to prevent token exhaustion
   - Control characters are removed
   - Common prompt delimiters are escaped (```, ###, ---, XML tags, etc.)

2. **Boundary Markers**: Untrusted content is wrapped in unique cryptographic boundary markers

3. **Security Instructions**: The judge prompt explicitly instructs the LLM to:
   - Ignore any instructions within the agent output
   - Only evaluate content quality, not meta-instructions
   - Not follow commands embedded in the evaluated content

#### Limitations

While these mitigations reduce risk, they cannot completely prevent sophisticated prompt injection attacks. Consider:

- Agent outputs could still influence LLM evaluation through subtle manipulation
- Very long outputs may be truncated, potentially hiding issues
- New prompt injection techniques may bypass current protections

For high-stakes evaluations, consider:
- Manual review of agent outputs
- Multiple evaluation models
- Structured evaluation criteria that are harder to manipulate

## Known Security Considerations

### LLM-as-Judge Evaluation

- EvalView uses OpenAI's API for output quality evaluation
- Test outputs and expected outputs are sent to OpenAI for comparison
- Agent outputs are sanitized to mitigate prompt injection, but no protection is 100% effective
- **Recommendation**: Don't include sensitive/proprietary data in test cases if using LLM-as-judge

### HTTP Adapters

- Custom HTTP adapters may expose your agent endpoints
- SSRF protection is enabled by default but can be bypassed with `allow_private_urls: true`
- **Recommendation**: Use authentication, HTTPS, and rate limiting on agent endpoints

### Trace Data

- Execution traces may contain sensitive information from agent responses
- **Recommendation**: Sanitize traces before sharing or storing long-term

### Verbose Mode

The `--verbose` flag may expose sensitive information in logs:
- API request/response payloads
- Query content and agent outputs
- **Recommendation**: Avoid using verbose mode in production or when processing sensitive data

## Threat Model: Untrusted Inputs

The May 2026 Langfuse RCE (a single OpenTelemetry trace request leading
to remote code execution via prototype pollution) sharpened a question
many AI-eval users are now asking: *"what does this tool parse, and how
safely?"* This section enumerates every external input EvalView accepts
and the parsing posture for each.

### Inputs EvalView consumes

| Input | Source | Parser | Sandbox | Notes |
|---|---|---|---|---|
| `tests/*.yaml`, `tests/*.toml` | Local file or `git clone` | `yaml.safe_load` (no Python objects); `tomllib` | None — runs in your shell | Test YAMLs can specify HTTP endpoints (see SSRF protection above) but cannot execute Python |
| `.evalview/incidents.jsonl` | Written by `evalview monitor` (local) | `json.loads` line-by-line; malformed lines skipped | None | Synthesized into deterministic regression-test YAML by `evalview/core/regression_synth.py` — pure function, no `eval`, no template injection |
| `.evalview/history/*.jsonl` (consumed by `evalview fleet`) | Written by `monitor`; may be merged from multiple machines | `json.loads`; unknown record shapes silently dropped | None | The fleet rollup never executes embedded code; it only counts/aggregates known fields |
| Cassette files `.evalview/cassettes/*.json` | `evalview simulate --record` | `json.loads`; per-tool sequential matching | None | Replayed values are returned to your agent verbatim — treat them like any test fixture |
| Production traces (cloud sync, when opted in) | Your machine → cloud via authenticated POST | Wire-format schema versioned (`SCHEMA_VERSION` in `evalview/core/types.py`); JSON-only | N/A (server side) | Local CLI never *receives* traces from the cloud; sync is one-way push |
| HTTP responses from agents under test | Your agent | JSON or text; never `pickle`, never `eval` | SSRF guard (see above) | Long bodies truncated before LLM judge to prevent prompt-injection token exhaustion |

### What EvalView does *not* do

- **No `pickle.load` on external data.** Anywhere. If you find one in a
  PR, that's a security review blocker.
- **No `eval` / `exec` on YAML or JSON content.** Test YAMLs are
  declarative; they cannot embed Python.
- **No prototype-style mutation of shared objects from parsed input.**
  Python's strong typing makes this less likely than in JS, but the
  parsers we use (`yaml.safe_load`, `json.loads`, `tomllib`) all return
  plain dicts/lists and never resolve user-controlled type tags.
- **No deserialization of arbitrary classes.** `pyyaml` is invoked
  exclusively via `safe_load` — a `pip-audit` finding on `yaml.load`
  with the unsafe loader is the canonical regression to watch for.
- **No code execution from cassettes.** Recorded tool responses are
  returned as data; if your agent then `exec`s them, that's your
  agent's threat model, not EvalView's.

### Cloud sync isolation

Cloud sync (`evalview login`) is fully opt-in. When disabled
(the default), no data leaves your machine. When enabled:

- Authentication uses short-lived OAuth tokens; refresh tokens are
  stored under `~/.config/evalview/` with `0600` permissions on
  POSIX systems.
- Wire-format schema is versioned and validated server-side; unknown
  fields are dropped, not echoed back.
- Per-project isolation is enforced server-side, not by the CLI.
  Trust the cloud's published threat model for that boundary; the CLI
  can only ensure it never sends data tagged for project A as project B.

### MCP server (`evalview mcp serve`)

The MCP server exposes 8 tools to a connected client (Claude Code,
Cursor, etc.). It runs over **stdio** by default, which means the
attack surface is whatever process spawned EvalView — usually your IDE.

If you operate the MCP over a network transport (not the default), you
are outside the supported configuration; review `evalview/commands/mcp_cmd.py`
yourself before doing so.

### Reporting parser bugs

If you find a way to make any of EvalView's parsers do something
beyond their documented behavior (e.g. a YAML construct that triggers
import, a JSONL line that escapes containment, a cassette that breaks
out of replay) — **please report via the GitHub Security Advisory
flow at the top of this file**, not as a public issue.

## Security Updates

We will disclose security vulnerabilities through:

1. **GitHub Security Advisories**: Primary notification channel
2. **Release Notes**: Documented in CHANGELOG.md
3. **GitHub Releases**: Tagged releases with security patch notes

---

Last updated: 2026-05-15
