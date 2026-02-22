# Cost Tracking — Monitor AI Agent Token Usage and Spending

> **Problem:** AI agent costs can spiral without visibility. A prompt change might double your token consumption, but you won't know until the bill arrives.
>
> **Solution:** EvalView tracks token usage and calculates costs per test run automatically. Set `max_cost` thresholds in test cases to catch budget overruns in CI before they reach production.

## Overview

EvalView includes comprehensive cost tracking based on model token usage. This feature automatically calculates costs per test run based on GPT-5 family pricing (or custom pricing) and provides detailed breakdowns.

## Features Implemented

### 1. **Token Usage Tracking** (`evalview/core/types.py`)
- Added `TokenUsage` class to track input, output, and cached tokens separately
- Updated `StepMetrics` to use `TokenUsage` instead of simple token count
- Updated `ExecutionMetrics` to track total token usage across all steps

### 2. **Pricing Module** (`evalview/core/pricing.py`)
- **Built-in pricing** for GPT-5 family models:
  - `gpt-5`: $1.25/1M input, $10/1M output, $0.125/1M cached
  - `gpt-5-mini`: $0.25/1M input, $2/1M output, $0.025/1M cached
  - `gpt-5-nano`: $0.05/1M input, $0.40/1M output, $0.005/1M cached
  - Also includes GPT-4o, GPT-4, GPT-3.5 for reference

- **Functions**:
  - `calculate_cost(model_name, input_tokens, output_tokens, cached_tokens)` - Calculate cost for token usage
  - `get_model_pricing_info(model_name)` - Get pricing details for a model

### 3. **Interactive Onboarding** (`evalview/cli.py`)
Enhanced `evalview init` command with:
- **Model selection**: Choose from gpt-5, gpt-5-mini, gpt-5-nano, gpt-4o-mini, or custom
- **Pricing display**: Shows pricing per 1M tokens before running tests
- **Custom pricing**: Allows users to set their own rates if they have special pricing
- **Config persistence**: Saves model config to `.evalview/config.yaml`

### 4. **Adapter Integration**
Both adapters now support cost tracking:

#### **TapeScopeAdapter** (`evalview/adapters/tapescope_adapter.py`)
- Listens for `usage` events in the streaming response
- Extracts `input_tokens`, `output_tokens`, and `cached_tokens` from API
- Calculates costs using pricing module
- Attaches costs to individual steps
- Logs token usage and costs in verbose mode

#### **HTTPAdapter** (`evalview/adapters/http_adapter.py`)
- Accepts model_config parameter
- Ready to parse token usage from REST API responses

### 5. **Enhanced Reporting** (`evalview/reporters/console_reporter.py`)
- **Summary table**: Added "Tokens" column showing total tokens used
- **Cached tokens**: Displays cached token count (90% discount) in summary
- **Detailed view**: Shows complete token breakdown:
  - Total tokens
  - Input tokens
  - Output tokens
  - Cached tokens (with note about 90% discount)

## Usage

### First-time Setup

```bash
evalview init --interactive
```

You'll be prompted for:
1. **API type**: REST or Streaming
2. **Endpoint**: Your agent's API URL
3. **Model**: Which GPT model your agent uses
4. **Pricing**: Confirm pricing or set custom rates

### Configuration File

`.evalview/config.yaml` example:

```yaml
# EvalView Configuration
adapter: streaming
endpoint: http://localhost:3000/api/unifiedchat
timeout: 60.0
headers: {}

# Model configuration
model:
  name: gpt-5-mini
  # Uses standard OpenAI pricing
  # Override with custom pricing if needed:
  # pricing:
  #   input_per_1m: 0.25
  #   output_per_1m: 2.0
  #   cached_per_1m: 0.025
```

### Custom Pricing

If you have enterprise pricing or custom rates, set them during init or edit the config:

```yaml
model:
  name: gpt-5
  pricing:
    input_per_1m: 1.00    # $1.00 per 1M input tokens
    output_per_1m: 8.00   # $8.00 per 1M output tokens
    cached_per_1m: 0.10   # $0.10 per 1M cached tokens
```

### Running Tests with Cost Tracking

```bash
# Run with verbose mode to see token usage in real-time
evalview run --verbose

# Results will show:
# - Cost per test case
# - Token usage breakdown (input/output/cached)
# - Total cost across all tests
```

### Example Output

```
📊 Evaluation Summary
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━━━━━┳━━━━━━━━━┓
┃ Test Case                 ┃ Score ┃ Status   ┃ Cost     ┃ Tokens       ┃ Latency ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━━━━━╇━━━━━━━━━┩
│ Stock Analysis Test       │  85.2 │ ✅ PASSED │ $0.0123  │ 12,450       │ 89,234ms│
│                           │       │          │          │ (3,200 cache)│         │
└───────────────────────────┴───────┴──────────┴──────────┴──────────────┴─────────┘
```

## How It Works

### 1. **API Event Flow** (for Streaming APIs)

When your agent processes a query, the streaming API sends events like:

```json
{"type": "step_narration", "data": {"text": "Analyzing stock data", "toolName": "analyzeStock"}}
{"type": "usage", "data": {"input_tokens": 1250, "output_tokens": 450, "cached_tokens": 800}}
{"type": "message_complete", "data": {"content": "Apple stock is..."}}
```

EvalView:
1. Captures the `step_narration` event → Creates a `StepTrace`
2. Captures the `usage` event → Calculates cost using pricing module
3. Attaches cost and token usage to the step
4. Accumulates totals for the entire execution

### 2. **Cost Calculation**

```python
# For gpt-5-mini with:
# - 1,250 input tokens
# - 450 output tokens
# - 800 cached tokens

cost = (1250 / 1_000_000) * 0.25 +    # Input: $0.0003125
       (450 / 1_000_000) * 2.0 +      # Output: $0.0009
       (800 / 1_000_000) * 0.025      # Cached: $0.00002
     = $0.00123 total
```

Cached tokens get a 90% discount (10% of normal input price).

### 3. **Integration Points**

The system integrates at multiple levels:

```
User runs test
     ↓
CLI loads model config from .evalview/config.yaml
     ↓
Adapter receives model_config parameter
     ↓
Adapter captures usage events from API
     ↓
Pricing module calculates cost
     ↓
Cost attached to ExecutionTrace
     ↓
Reporter displays costs and token breakdown
     ↓
Results saved to JSON with full cost data
```

## Benefits

1. **Cost Transparency**: See exactly what each test costs
2. **Budget Management**: Set `max_cost` thresholds in test cases
3. **Optimization**: Identify expensive queries and optimize prompts
4. **Accurate Billing**: Track costs across all test runs
5. **Custom Pricing**: Support for enterprise pricing agreements

## API Requirements

For cost tracking to work, your agent's API must:

1. **Emit token usage data** in one of these formats:
   - Streaming: `{"type": "usage", "data": {"input_tokens": N, "output_tokens": N, "cached_tokens": N}}`
   - REST: Include usage in response JSON

2. **Report token counts** after each LLM call or at the end of execution

If your API doesn't provide token counts yet, costs will show as $0.00 until you add this instrumentation.

## Future Enhancements

Potential future features:
- Cost budgets per test suite
- Cost trend analysis over time
- Cost optimization suggestions
- Support for other LLM providers (Anthropic, Gemini, etc.)
- Cost alerts when thresholds are exceeded

## Technical Details

### File Changes

- `evalview/core/types.py` - Added `TokenUsage` class
- `evalview/core/pricing.py` - **NEW** pricing module
- `evalview/adapters/tapescope_adapter.py` - Added usage event handling
- `evalview/adapters/http_adapter.py` - Added model_config parameter
- `evalview/cli.py` - Enhanced init with model selection
- `evalview/reporters/console_reporter.py` - Added token display

### Dependencies

No new dependencies required! Uses existing libraries.

### Backward Compatibility

✅ **Fully backward compatible**
- Old configs without `model` section use default gpt-5-mini pricing
- Tests without token usage still work (show $0.00 cost)
- No breaking changes to existing APIs
