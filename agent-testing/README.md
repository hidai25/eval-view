# Agent Testing (Contributors Only)

This directory contains internal testing infrastructure for EvalView development.

## For New Users

**You don't need anything in this folder!**

Run `evalview quickstart` instead - it sets up everything you need automatically.

## For Contributors

The `internal/reference-agent/` contains a full-featured test agent used for:
- Testing EvalView features during development
- Validating changes across multiple tool types
- CI/CD integration testing

### Reference Agent Tools

- `calculator` - Basic math operations
- `get_weather` - Mock weather data
- `get_stock_price` - Mock stock prices
- `search_web` - Mock search results

### Running the Reference Agent

```bash
cd internal/reference-agent
pip install -r requirements.txt
python agent.py
```

Then run tests:
```bash
evalview run --pattern internal/reference-agent/.evalview/test-cases/
```
