# YAML Test Case Schema — EvalView Test Case Reference

> EvalView test cases are defined in YAML files. This document provides the complete schema reference for writing AI agent tests, including input configuration, expected behavior, scoring thresholds, and adapter settings.

This document provides the complete schema reference for EvalView test case YAML files.

## Minimal Example

```yaml
name: basic_search_test
input:
  query: "What is the capital of France?"
expected:
  tools:
    - search
thresholds:
  min_score: 70
```

## Complete Schema Reference

### Root Fields

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `name` | string | **Yes** | - | Unique identifier for the test case |
| `description` | string | No | `null` | Human-readable description |
| `input` | object | **Yes** | - | Test input configuration |
| `expected` | object | **Yes** | - | Expected behavior and output |
| `thresholds` | object | **Yes** | - | Pass/fail thresholds |
| `adapter` | string | No | config default | Override adapter type for this test |
| `endpoint` | string | No | config default | Override endpoint URL for this test |
| `adapter_config` | object | No | `{}` | Additional adapter-specific settings |

---

### `input` Object

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `query` | string | **Yes** | - | The query/prompt to send to the agent |
| `context` | object | No | `null` | Additional context passed to the agent |

**Example:**
```yaml
input:
  query: "Research the latest AI developments"
  context:
    max_results: 5
    time_range: "last_week"
```

---

### `expected` Object

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `tools` | list[string] | No | `null` | Expected tools to be used (any order) |
| `tool_sequence` | list[string] | No | `null` | Expected tools in specific order |
| `sequence` | list[string] | No | `null` | Alias for `tool_sequence` |
| `output` | object | No | `null` | Expected output criteria |
| `metrics` | object | No | `null` | Expected metric thresholds |
| `hallucination` | object | No | `null` | Hallucination detection config |
| `safety` | object | No | `null` | Safety evaluation config |

**Example:**
```yaml
expected:
  tools:
    - search
    - summarize
  tool_sequence:
    - search
    - summarize
  output:
    contains:
      - "Paris"
    not_contains:
      - "error"
```

---

### `expected.output` Object

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `contains` | list[string] | No | `null` | Strings that MUST appear in output |
| `not_contains` | list[string] | No | `null` | Strings that MUST NOT appear in output |
| `json_schema` | object | No | `null` | JSON schema the output must validate against |
| `must_acknowledge_uncertainty` | boolean | No | `null` | Output must acknowledge uncertainty |
| `no_pii` | boolean | No | `null` | Output must not contain PII |

**Example:**
```yaml
expected:
  output:
    contains:
      - "Paris"
      - "France"
    not_contains:
      - "error"
      - "I don't know"
    json_schema:
      type: object
      required:
        - answer
```

---

### `expected.hallucination` Object

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `check` | boolean | No | `false` | Enable hallucination detection |
| `allow` | boolean | No | `false` | Pass test even if hallucination detected |
| `confidence_threshold` | float | No | `0.8` | Confidence threshold (0-1) |

**Example:**
```yaml
expected:
  hallucination:
    check: true
    allow: false
    confidence_threshold: 0.9
```

---

### `expected.safety` Object

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `check` | boolean | No | `false` | Enable safety evaluation |
| `allow_harmful` | boolean | No | `false` | Pass test even if harmful content detected |
| `categories` | list[string] | No | `null` | Categories to check: `violence`, `hate_speech`, etc. |
| `severity_threshold` | string | No | `"medium"` | Threshold: `"low"`, `"medium"`, `"high"` |

**Example:**
```yaml
expected:
  safety:
    check: true
    allow_harmful: false
    categories:
      - violence
      - hate_speech
    severity_threshold: medium
```

---

### `expected.metrics` Object

Map of metric name to threshold configuration.

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `value` | float | **Yes** | Expected metric value |
| `tolerance` | float | **Yes** | Acceptable deviation |

**Example:**
```yaml
expected:
  metrics:
    latency:
      value: 1000
      tolerance: 200
    cost:
      value: 0.05
      tolerance: 0.01
```

---

### `thresholds` Object

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `min_score` | float | **Yes** | - | Minimum passing score (0-100) |
| `max_cost` | float | No | `null` | Maximum cost in dollars |
| `max_latency` | float | No | `null` | Maximum latency in milliseconds |

**Example:**
```yaml
thresholds:
  min_score: 70
  max_cost: 0.10
  max_latency: 5000
```

---

### `adapter_config` Object

Adapter-specific configuration. Common fields:

| Field | Type | Description | Adapters |
|-------|------|-------------|----------|
| `timeout` | float | Request timeout in seconds | All |
| `headers` | object | Custom HTTP headers | http, langgraph, crewai |
| `streaming` | boolean | Enable streaming mode | langgraph, tapescope |
| `mode` | string | Execution mode | langgraph (`standard`, `streaming`, `cloud`) |
| `assistant_id` | string | OpenAI Assistant ID | openai-assistants |
| `verbose` | boolean | Enable verbose logging | All |

**Example:**
```yaml
adapter: langgraph
endpoint: http://localhost:8000
adapter_config:
  timeout: 60
  streaming: true
  mode: cloud
  headers:
    Authorization: Bearer ${API_KEY}
```

---

## Available Adapters

| Adapter | Value | Description |
|---------|-------|-------------|
| HTTP (Generic) | `http` | Generic REST API adapter |
| LangGraph | `langgraph` | LangGraph agents (local or Cloud) |
| CrewAI | `crewai` | CrewAI multi-agent systems |
| OpenAI Assistants | `openai-assistants` | OpenAI Assistants API |
| TapeScope | `tapescope` | JSONL streaming APIs |
| Streaming | `streaming` | Generic SSE streaming adapter |

---

## Complete Example

```yaml
name: research_agent_test
description: Test the research agent's ability to search and summarize

input:
  query: "What are the latest developments in quantum computing?"
  context:
    max_sources: 3
    include_citations: true

expected:
  tools:
    - web_search
    - summarize
  tool_sequence:
    - web_search
    - summarize
  output:
    contains:
      - "quantum"
      - "computing"
    not_contains:
      - "error"
      - "I cannot"
  hallucination:
    check: true
    confidence_threshold: 0.85
  safety:
    check: true

thresholds:
  min_score: 75
  max_cost: 0.15
  max_latency: 10000

# Override global adapter settings for this test
adapter: langgraph
endpoint: http://localhost:8123
adapter_config:
  timeout: 90
  streaming: false
  headers:
    X-API-Key: ${RESEARCH_API_KEY}
```

---

## Environment Variables

You can use environment variables in YAML with `${VAR_NAME}` syntax:

```yaml
adapter_config:
  headers:
    Authorization: Bearer ${OPENAI_API_KEY}
endpoint: ${AGENT_ENDPOINT}
```

---

## Validation

Test cases are validated against Pydantic models on load. Common validation errors:

| Error | Cause | Fix |
|-------|-------|-----|
| `min_score must be >= 0` | Negative score | Use 0-100 range |
| `min_score must be <= 100` | Score > 100 | Use 0-100 range |
| `input is required` | Missing input block | Add `input:` with `query:` |
| `query is required` | Missing query field | Add `query:` under `input:` |

---

## Tips

1. **Start simple:** Begin with just `name`, `input`, `expected.tools`, and `thresholds`
2. **Use tool_sequence sparingly:** Only when order matters (most agents are non-deterministic)
3. **Set realistic thresholds:** Start with `min_score: 50` and increase as you refine
4. **Test timeouts:** Different frameworks have different latencies - adjust accordingly
5. **Use adapter overrides:** Test the same query against different backends

---

## Related Documentation

- [Getting Started](GETTING_STARTED.md) — Write and run your first test case
- [Evaluation Metrics](EVALUATION_METRICS.md) — How `min_score`, tool accuracy, and output quality are calculated
- [Tool Categories](TOOL_CATEGORIES.md) — Flexible tool name matching with categories
- [Cost Tracking](COST_TRACKING.md) — How `max_cost` thresholds work
- [Adapters](ADAPTERS.md) — Adapter-specific configuration options
- [CLI Reference](CLI_REFERENCE.md) — Running tests with `evalview run`
