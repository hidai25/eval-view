# Failure Mode Scenarios

Test how your agent handles real-world failures.

## Scenarios

| File | Scenario | Chaos Config |
|------|----------|--------------|
| `01-tool-timeout.yaml` | Tool takes too long | `timeout: true` |
| `02-malformed-response.yaml` | Tool returns invalid JSON | `malformed: true` |
| `03-rate-limit.yaml` | Tool returns 429 | `rate_limit: true` |
| `04-infinite-loop-guard.yaml` | Agent stuck in loop | `max_calls_per_tool: 5` |
| `05-partial-failure.yaml` | Some tools work, others don't | `failing_tools: [search_kb]` |
| `06-empty-response.yaml` | Tool returns null | `empty: true` |
| `07-network-error.yaml` | Connection error | `error: "Connection refused"` |
| `08-high-latency.yaml` | Slow but successful | `latency_ms: 3000` |
| `09-wrong-tool-output.yaml` | Semantically wrong data | `override_response: {...}` |
| `10-cascading-failure.yaml` | Multiple failures at once | `timeout: true, failing_tools: [...]` |

## Run All

```bash
evalview gym --suite failure-modes
```

## Run Individual

```bash
evalview run gym/failure-modes/01-tool-timeout.yaml
```

## What Makes a Good Failure Mode Test

1. **Clear failure trigger** — Use chaos config to inject specific failure
2. **Testable recovery** — Define what "graceful handling" looks like
3. **No false positives** — Assertions should pass when agent behaves correctly
4. **Realistic scenario** — Failures that actually happen in production
