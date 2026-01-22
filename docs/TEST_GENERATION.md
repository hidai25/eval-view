# Test Generation

**Problem:** Writing tests manually is slow. You need volume to catch regressions.

**Solution:** Auto-generate test variations.

---

## Option 1: Expand from Existing Tests

Take 1 test, generate 100 variations:

```bash
# Take 1 test, generate 100 variations
evalview expand tests/stock-test.yaml --count 100

# Focus on specific scenarios
evalview expand tests/stock-test.yaml --count 50 \
  --focus "different tickers, edge cases, error scenarios"
```

### What It Generates

Variations like:
- **Different inputs** (AAPL → MSFT, GOOGL, TSLA...)
- **Edge cases** (invalid tickers, empty input, malformed requests)
- **Boundary conditions** (very long queries, special characters)

### Example

Original test:
```yaml
name: "Stock Analysis"
input:
  query: "Analyze Apple stock"
expected:
  tools:
    - fetch_stock_data
```

Generated variations:
```yaml
name: "Stock Analysis - MSFT"
input:
  query: "Analyze Microsoft stock"
expected:
  tools:
    - fetch_stock_data

---
name: "Stock Analysis - Invalid Ticker"
input:
  query: "Analyze XXXXX stock"
expected:
  # Should handle gracefully

---
name: "Stock Analysis - Empty Query"
input:
  query: ""
expected:
  # Should return helpful error
```

---

## Option 2: Record from Live Interactions

Use your agent normally, auto-generate tests:

```bash
evalview record --interactive
```

### What It Captures

- Query → Tools called → Output
- Auto-generates test YAML
- Adds reasonable thresholds

### Example Session

```
$ evalview record --interactive

Recording session started. Use your agent normally.
Press Ctrl+C to stop recording.

> What's the weather in NYC?
[Agent calls: weather_api]
[Agent responds: "The weather in NYC is..."]

✓ Captured: weather-query-1

> Summarize this document
[Agent calls: read_file, summarize]
[Agent responds: "Here's a summary..."]

✓ Captured: document-summary-1

^C

Recording stopped. Generated 2 test cases:
  - tests/generated/weather-query-1.yaml
  - tests/generated/document-summary-1.yaml
```

---

## Result

Go from **5 manual tests → 500 comprehensive tests** in minutes.

---

## Best Practices

### Start with good seed tests

The quality of generated tests depends on your seed. Write a few high-quality tests first.

### Use focus for targeted expansion

```bash
# Focus on error handling
evalview expand tests/api-test.yaml --count 50 --focus "error scenarios, timeouts, invalid inputs"

# Focus on edge cases
evalview expand tests/parser-test.yaml --count 30 --focus "edge cases, unicode, empty values"
```

### Review generated tests

Always review generated tests before committing. Remove duplicates and irrelevant variations.

### Use recording for real-world scenarios

Recording captures actual usage patterns that you might not think to test manually.

---

## CLI Reference

### `evalview expand`

```bash
evalview expand TEST_FILE [OPTIONS]

Options:
  --count N        Number of variations to generate (default: 10)
  --focus TEXT     Focus on specific scenarios
  --output DIR     Output directory (default: tests/generated/)
```

### `evalview record`

```bash
evalview record [OPTIONS]

Options:
  --interactive    Interactive recording mode
  --output DIR     Output directory (default: tests/generated/)
```

---

## Related Documentation

- [CLI Reference](CLI_REFERENCE.md)
- [Getting Started](GETTING_STARTED.md)
