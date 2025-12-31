# EvalView Test Suite

This directory contains comprehensive unit tests for the EvalView framework.

## Test Coverage

### Core Components

#### `test_types.py` - Core Type Definitions (219 lines)
Comprehensive tests for all Pydantic models:
- ✅ TestCase, TestInput, ExpectedBehavior validation
- ✅ Thresholds with Field constraints (ge=0, le=100)
- ✅ ExecutionTrace, StepTrace, StepMetrics
- ✅ TokenUsage with total_tokens property
- ✅ All Evaluation result types (Tool, Sequence, Output, Cost, Latency)
- ✅ EvaluationResult complete validation
- ✅ Edge cases: boundary values, missing fields, invalid constraints

**Test Classes:** 13 | **Test Methods:** 50+

#### `test_loader.py` - YAML Test Case Loader (328 lines)
Tests for loading test cases from YAML files:
- ✅ Valid YAML file loading (.yaml and .yml extensions)
- ✅ Invalid YAML syntax handling
- ✅ Invalid schema (missing required fields) handling
- ✅ File not found errors
- ✅ Directory loading with multiple files
- ✅ Empty files and null values
- ✅ Custom file patterns
- ✅ Field value preservation
- ✅ Adapter override configuration

**Test Classes:** 2 | **Test Methods:** 22

### Evaluators

#### `test_evaluators.py` - All Evaluator Modules (1040 lines)
Comprehensive tests for all evaluation logic:

**ToolCallEvaluator (8 tests):**
- ✅ Perfect accuracy (100%)
- ✅ Missing tools detection
- ✅ Unexpected tools detection
- ✅ No expected tools (empty list)
- ✅ Duplicate tool calls
- ✅ None vs empty list handling

**SequenceEvaluator (6 tests):**
- ✅ Correct sequence validation
- ✅ Incorrect order detection
- ✅ Length mismatch handling
- ✅ No expected sequence (None)
- ✅ Empty sequence
- ✅ Violation message generation

**OutputEvaluator (8 tests):**
- ✅ Contains checks (case-insensitive)
- ✅ Not-contains checks (case-insensitive)
- ✅ Partial match detection
- ✅ LLM-as-judge integration (mocked)
- ✅ Empty output handling
- ✅ No string checks specified

**CostEvaluator (5 tests):**
- ✅ Cost within threshold
- ✅ Cost exceeds threshold
- ✅ No cost threshold (infinity)
- ✅ Cost breakdown by step
- ✅ Zero cost warning

**LatencyEvaluator (5 tests):**
- ✅ Latency within threshold
- ✅ Latency exceeds threshold
- ✅ No latency threshold (infinity)
- ✅ Latency breakdown by step
- ✅ Exact threshold boundary

**Test Classes:** 5 | **Test Methods:** 32

#### `test_main_evaluator.py` - Main Orchestrator (414 lines)
Tests for the main Evaluator that orchestrates all sub-evaluators:
- ✅ Weighted scoring (30% tools, 50% output, 20% sequence)
- ✅ Pass/fail logic (score, cost, latency thresholds)
- ✅ All evaluations integration
- ✅ Perfect score calculation (100%)
- ✅ Zero score calculation
- ✅ Multiple failure detection
- ✅ Boundary score testing
- ✅ Score rounding (2 decimal places)
- ✅ No thresholds handling
- ✅ Weight validation (sum to 1.0)

**Test Classes:** 1 | **Test Methods:** 15

### Adapters

#### `test_adapters.py` - HTTP Adapter (768 lines)
Comprehensive tests for HTTP adapter response parsing:

**Response Parsing (20+ tests):**
- ✅ Flat response structure
- ✅ Nested metadata response
- ✅ Response with detailed steps
- ✅ Minimal response (only required fields)
- ✅ Output field priority (response > output > result > answer)
- ✅ No output field (empty string)
- ✅ Token formats (integer, dict, nested)
- ✅ Cost calculation from tokens
- ✅ Cost extraction (flat, metadata, steps)
- ✅ Session ID generation
- ✅ Latency calculation

**Step Parsing (6 tests):**
- ✅ Minimal step fields
- ✅ Tool name alternatives (tool, tool_name)
- ✅ Parameter alternatives (params, parameters)
- ✅ Output alternatives (output, result)
- ✅ Error information
- ✅ Empty steps list

**Execution (9 tests):**
- ✅ Basic execution
- ✅ Context parameter
- ✅ Custom headers
- ✅ Custom timeout
- ✅ HTTP error handling
- ✅ Health check success/failure
- ✅ Network exceptions
- ✅ Tracing enabled flag

**Test Classes:** 1 | **Test Methods:** 35

## Fixtures and Test Utilities

### `conftest.py` - Shared Fixtures (330 lines)
Comprehensive pytest fixtures for all tests:

**Test Data Fixtures:**
- `sample_test_case` - Complete test case with all fields
- `sample_execution_trace` - Execution trace with 2 steps
- `minimal_test_case` - Minimal valid test case
- `empty_trace` - Trace with no steps

**HTTP Response Fixtures:**
- `http_response_flat` - Flat structure
- `http_response_nested` - Nested metadata
- `http_response_with_steps` - Detailed steps
- `http_response_minimal` - Minimal fields
- `http_response_with_tokens_only` - For cost calculation

**Mock Fixtures:**
- `mock_openai_client` - Mocked OpenAI API client
- `mock_httpx_client` - Mocked HTTP client

**Temporary File Fixtures:**
- `temp_yaml_file` - Valid YAML test case
- `temp_invalid_yaml_file` - Malformed YAML
- `temp_invalid_schema_file` - Invalid schema
- `temp_yaml_directory` - Directory with multiple files

## Running the Tests

### Quick Start

**Install dependencies:**
```bash
# Option A: Using uv (faster)
uv sync --all-extras

# Option B: Using pip
pip install -e ".[dev]"
```

**Run tests:**
```bash
# Using uv
make test              # Run all tests
make test-cov          # Run with coverage

# Using pip
make pip-test          # Run all tests
make pip-test-cov      # Run with coverage
```

**Run specific tests:**
```bash
# With uv
uv run pytest tests/test_types.py -v
uv run pytest tests/test_evaluators.py::TestToolCallEvaluator -v

# With pip
pytest tests/test_types.py -v
pytest tests/test_evaluators.py::TestToolCallEvaluator -v
```

### Advanced Usage

```bash
# Run tests matching pattern
pytest tests/ -k "test_parse" -v

# Run tests with detailed output
pytest tests/ -vv

# Stop on first failure
pytest tests/ -x

# Show local variables on failure
pytest tests/ -l

# Run in parallel (requires pytest-xdist)
pytest tests/ -n auto

# Generate HTML coverage report
pytest tests/ --cov=evalview --cov-report=html
open htmlcov/index.html
```

### Test Markers

```bash
# Run only unit tests
pytest tests/ -m unit

# Run only integration tests
pytest tests/ -m integration

# Skip slow tests
pytest tests/ -m "not slow"

# Skip network tests
pytest tests/ -m "not network"
```

## Test Statistics

| Component | Test Files | Test Classes | Test Methods | Lines of Code |
|-----------|------------|--------------|--------------|---------------|
| Core Types | 1 | 13 | 50+ | 575 |
| YAML Loader | 1 | 2 | 22 | 328 |
| Evaluators | 1 | 5 | 32 | 1,040 |
| Main Evaluator | 1 | 1 | 15 | 414 |
| HTTP Adapter | 1 | 1 | 35 | 768 |
| **Total** | **5** | **22** | **154+** | **3,125+** |

Plus `conftest.py` with 330 lines of fixtures and utilities.

## Code Coverage Goals

Target coverage by module:
- ✅ `evalview/core/types.py` - 100%
- ✅ `evalview/core/loader.py` - 100%
- ✅ `evalview/evaluators/tool_call_evaluator.py` - 100%
- ✅ `evalview/evaluators/sequence_evaluator.py` - 100%
- ✅ `evalview/evaluators/output_evaluator.py` - 95%+
- ✅ `evalview/evaluators/cost_evaluator.py` - 100%
- ✅ `evalview/evaluators/latency_evaluator.py` - 100%
- ✅ `evalview/evaluators/evaluator.py` - 100%
- ✅ `evalview/adapters/http_adapter.py` - 95%+

## Test Design Principles

### 1. Comprehensive Coverage
- Test happy paths and edge cases
- Test boundary values (0, 100, infinity)
- Test missing/None values
- Test empty collections

### 2. Clear Test Names
- Use descriptive test method names
- Follow pattern: `test_<what>_<condition>`
- Examples:
  - `test_valid_test_case`
  - `test_thresholds_min_score_constraint_min`
  - `test_parse_response_with_steps`

### 3. Isolated Tests
- Each test is independent
- Use fixtures for shared setup
- Mock external dependencies (OpenAI API, HTTP calls)

### 4. Async Testing
- Use `@pytest.mark.asyncio` for async tests
- Test async/await patterns correctly
- Mock async clients properly

### 5. Validation Testing
- Test Pydantic validation errors
- Test field constraints
- Test required vs optional fields
- Test type coercion

## Common Test Patterns

### Testing Pydantic Models

```python
def test_valid_model():
    """Test creating a valid model."""
    model = MyModel(field1="value", field2=42)
    assert model.field1 == "value"
    assert model.field2 == 42

def test_invalid_constraint():
    """Test that field constraints are enforced."""
    with pytest.raises(ValidationError) as exc_info:
        MyModel(field1="value", field2=150)  # max=100
    assert "less than or equal to 100" in str(exc_info.value).lower()
```

### Testing Async Functions

```python
@pytest.mark.asyncio
async def test_async_function(mock_client):
    """Test an async function."""
    evaluator = MyEvaluator()
    evaluator.client = mock_client

    result = await evaluator.evaluate(test_case, trace)

    assert result.score > 0
    mock_client.some_method.assert_called_once()
```

### Testing File Operations

```python
def test_load_file(temp_yaml_file):
    """Test loading a file."""
    result = loader.load(temp_yaml_file)
    assert result.name == "expected_name"
```

## Contributing Tests

When adding new tests:
1. ✅ Use existing fixtures from `conftest.py`
2. ✅ Add new fixtures if needed (reusable test data)
3. ✅ Follow naming conventions
4. ✅ Add docstrings to test methods
5. ✅ Test both success and failure cases
6. ✅ Keep tests focused and atomic
7. ✅ Update this README with new test coverage

## Continuous Integration

These tests are designed to run in CI/CD pipelines:
- Fast execution (< 10 seconds for full suite)
- No external dependencies (all mocked)
- Deterministic results
- Clear failure messages

## Quality Metrics

- **Test Coverage:** Target 95%+
- **Test Execution Time:** < 10 seconds
- **Flaky Tests:** 0 (all deterministic)
- **Mock Coverage:** All external APIs mocked
- **Documentation:** Every test has a docstring

## Need Help?

- Read test docstrings for usage examples
- Check `conftest.py` for available fixtures
- Run `pytest --fixtures` to see all fixtures
- Run `pytest --markers` to see all markers
- See pytest docs: https://docs.pytest.org/
