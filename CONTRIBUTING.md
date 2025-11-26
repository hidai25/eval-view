# Contributing to EvalView

Thanks for your interest in contributing! This guide will help you get started.

## Development Setup

```bash
# Clone the repository
git clone https://github.com/hidai25/EvalView.git
cd EvalView

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode with dev dependencies
pip install -e ".[dev]"
```

## Development Workflow

### Running Tests

```bash
# Run all tests
make test

# Or manually
pytest

# Run with coverage
pytest --cov=evalview --cov-report=html
```

### Code Quality

```bash
# Format code
make format

# Run linter
make lint

# Type checking
make typecheck

# Run all checks (format + lint + typecheck)
make check
```

### Quick Commands

```bash
make install     # Install package in dev mode
make test        # Run tests
make format      # Format with black
make lint        # Lint with ruff
make typecheck   # Type check with mypy
make check       # Run all checks
make clean       # Clean build artifacts
```

## Project Structure

```
evalview/
├── core/           # Core types and utilities
│   ├── types.py    # Pydantic models
│   ├── loader.py   # Test case loader
│   └── pricing.py  # Cost calculation
├── adapters/       # Agent communication
│   ├── base.py     # Abstract adapter
│   ├── http_adapter.py
│   └── tapescope_adapter.py
├── evaluators/     # Evaluation logic
│   ├── evaluator.py           # Main orchestrator
│   ├── tool_call_evaluator.py
│   ├── sequence_evaluator.py
│   ├── output_evaluator.py
│   ├── cost_evaluator.py
│   └── latency_evaluator.py
├── reporters/      # Result formatting
│   ├── json_reporter.py
│   └── console_reporter.py
└── cli.py          # CLI entry point
```

## Adding a New Evaluator

Evaluators assess specific aspects of agent behavior. Here's how to add one:

### 1. Create the Evaluator

Create a new file in `evalview/evaluators/`:

```python
# evalview/evaluators/my_evaluator.py
from typing import Any, Dict
from evalview.core.types import TestCase, ExecutionTrace

class MyEvaluator:
    """Evaluates [what aspect] of agent execution."""

    def evaluate(
        self,
        test_case: TestCase,
        trace: ExecutionTrace
    ) -> Dict[str, Any]:
        """
        Evaluate the execution trace.

        Args:
            test_case: Test case with expected behavior
            trace: Actual execution trace from agent

        Returns:
            Dictionary with evaluation results
        """
        # Your evaluation logic here
        passed = True  # Your logic
        score = 100.0  # Your scoring

        return {
            "passed": passed,
            "score": score,
            "details": "Explanation of results",
        }
```

### 2. Add Result Type

Update `evalview/core/types.py` to include your result type:

```python
class MyEvaluationResult(BaseModel):
    """Result from my evaluator."""
    passed: bool
    score: float
    details: str
```

Add it to the `Evaluations` model:

```python
class Evaluations(BaseModel):
    tool_accuracy: ToolAccuracyResult
    sequence_correctness: SequenceCorrectnessResult
    output_quality: OutputQualityResult
    cost: CostResult
    latency: LatencyResult
    my_evaluation: MyEvaluationResult  # Add this
```

### 3. Integrate into Main Evaluator

Update `evalview/evaluators/evaluator.py`:

```python
from evalview.evaluators.my_evaluator import MyEvaluator

class Evaluator:
    def __init__(self, openai_api_key: Optional[str] = None):
        # ... existing evaluators ...
        self.my_evaluator = MyEvaluator()

    async def evaluate(
        self, test_case: TestCase, trace: ExecutionTrace
    ) -> EvaluationResult:
        evaluations = Evaluations(
            # ... existing evaluations ...
            my_evaluation=self.my_evaluator.evaluate(test_case, trace),
        )
        # ...
```

### 4. Update Scoring (Optional)

If your evaluator should affect the overall score, update `_compute_overall_score`:

```python
def _compute_overall_score(
    self, evaluations: Evaluations, test_case: TestCase
) -> float:
    weights = {
        "tool_accuracy": 0.25,      # Adjusted
        "output_quality": 0.45,     # Adjusted
        "sequence_correctness": 0.15,  # Adjusted
        "my_evaluation": 0.15,      # New
    }

    score = (
        # ... existing calculations ...
        + evaluations.my_evaluation.score * weights["my_evaluation"]
    )
    return round(score, 2)
```

### 5. Write Tests

Create tests in `tests/test_my_evaluator.py`:

```python
import pytest
from evalview.evaluators.my_evaluator import MyEvaluator
from evalview.core.types import TestCase, ExecutionTrace

def test_my_evaluator():
    evaluator = MyEvaluator()

    # Create test data
    test_case = TestCase(...)
    trace = ExecutionTrace(...)

    # Run evaluation
    result = evaluator.evaluate(test_case, trace)

    # Assert results
    assert result["passed"] is True
    assert result["score"] == 100.0
```

### 6. Update Documentation

- Add your evaluator to the "Evaluation Metrics" section in README.md
- Document any new test case fields it uses
- Add examples showing how to use it

## Creating a Custom Adapter

> **⚠️ EXPERIMENTAL**: The adapter registry system is experimental and may change in future versions.

Adapters handle communication with different AI agent frameworks. Here's how to create one:

### 1. Create the Adapter

Create a new file in `evalview/adapters/`:

```python
# evalview/adapters/my_adapter.py
from datetime import datetime
from typing import Any, Dict, List, Optional, Set
import httpx
import logging

from evalview.adapters.base import AgentAdapter
from evalview.core.types import (
    ExecutionTrace,
    StepTrace,
    StepMetrics,
    ExecutionMetrics,
    TokenUsage,
)

logger = logging.getLogger(__name__)


class MyAdapter(AgentAdapter):
    """Adapter for MyFramework agents.

    Supports:
    - Standard REST API
    - Your framework's specific response format

    Security Note:
        SSRF protection is enabled by default.
    """

    def __init__(
        self,
        endpoint: str,
        headers: Optional[Dict[str, str]] = None,
        timeout: float = 30.0,
        verbose: bool = False,
        model_config: Optional[Dict[str, Any]] = None,
        allow_private_urls: bool = False,
        allowed_hosts: Optional[Set[str]] = None,
    ):
        # Set SSRF protection settings BEFORE validation
        self.allow_private_urls = allow_private_urls
        self.allowed_hosts = allowed_hosts

        # Validate endpoint URL for SSRF protection
        self.endpoint = self.validate_endpoint(endpoint)

        self.headers = headers or {"Content-Type": "application/json"}
        self.timeout = timeout
        self.verbose = verbose
        self.model_config = model_config or {}
        self._last_raw_response = None  # For debug mode

    @property
    def name(self) -> str:
        return "my-adapter"

    async def execute(
        self, query: str, context: Optional[Dict[str, Any]] = None
    ) -> ExecutionTrace:
        """Execute agent and capture trace."""
        context = context or {}
        start_time = datetime.now()

        # Make API request
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(
                self.endpoint,
                json={"query": query, **context},
                headers=self.headers,
            )
            response.raise_for_status()
            data = response.json()

        end_time = datetime.now()

        # Store raw response for debug mode
        self._last_raw_response = data

        # Parse response into ExecutionTrace
        steps = self._parse_steps(data)
        final_output = self._extract_output(data)
        metrics = self._calculate_metrics(data, steps, start_time, end_time)

        return ExecutionTrace(
            session_id=data.get("session_id", f"my-{start_time.timestamp()}"),
            start_time=start_time,
            end_time=end_time,
            steps=steps,
            final_output=final_output,
            metrics=metrics,
        )

    def _parse_steps(self, data: Dict[str, Any]) -> List[StepTrace]:
        """Parse steps from API response."""
        steps = []

        for i, step_data in enumerate(data.get("steps", [])):
            step = StepTrace(
                step_id=step_data.get("id", f"step-{i}"),
                step_name=step_data.get("name", f"Step {i+1}"),
                tool_name=step_data.get("tool") or "unknown",  # Handle None
                parameters=step_data.get("params", {}),
                output=step_data.get("output", ""),
                success=step_data.get("success", True),
                error=step_data.get("error"),
                metrics=StepMetrics(
                    latency=step_data.get("latency", 0.0),  # Defaults to 0.0
                    cost=step_data.get("cost", 0.0),
                    tokens=step_data.get("tokens"),  # Can be int, dict, or None
                ),
            )
            steps.append(step)

        return steps

    def _extract_output(self, data: Dict[str, Any]) -> str:
        """Extract final output - try multiple field names."""
        return (
            data.get("response")
            or data.get("output")
            or data.get("result")
            or ""
        )

    def _calculate_metrics(
        self,
        data: Dict[str, Any],
        steps: List[StepTrace],
        start_time: datetime,
        end_time: datetime,
    ) -> ExecutionMetrics:
        """Calculate execution metrics."""
        total_latency = (end_time - start_time).total_seconds() * 1000

        # Get tokens - can be int or TokenUsage (validators handle coercion)
        total_tokens = data.get("total_tokens")

        # If not in response, aggregate from steps
        if not total_tokens:
            input_sum = sum(
                s.metrics.tokens.input_tokens
                for s in steps if s.metrics.tokens
            )
            output_sum = sum(
                s.metrics.tokens.output_tokens
                for s in steps if s.metrics.tokens
            )
            if input_sum + output_sum > 0:
                total_tokens = TokenUsage(
                    input_tokens=input_sum,
                    output_tokens=output_sum,
                )

        return ExecutionMetrics(
            total_cost=data.get("cost", 0.0),
            total_latency=total_latency,
            total_tokens=total_tokens,  # int, dict, TokenUsage, or None - all work
        )

    async def health_check(self) -> bool:
        """Check if endpoint is reachable."""
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(self.endpoint.replace("/invoke", "/health"))
                return response.status_code in [200, 201, 404]
        except Exception:
            return False
```

### 2. Common Pitfalls to Avoid

**Token type errors:**
```python
# WRONG: Returning int directly
return ExecutionMetrics(total_tokens=1500)

# RIGHT: EvalView auto-coerces, but explicit is better
return ExecutionMetrics(total_tokens=TokenUsage(output_tokens=1500))
```

**Missing defaults for StepMetrics:**
```python
# WRONG: Will fail if latency/cost missing
metrics=StepMetrics(
    latency=step_data.get("latency"),  # Could be None
    cost=step_data.get("cost"),
)

# RIGHT: Provide defaults
metrics=StepMetrics(
    latency=step_data.get("latency", 0.0),
    cost=step_data.get("cost", 0.0),
)
```

**Datetime handling:**
```python
# WRONG: Passing string
start_time="2025-01-15T10:30:00"

# RIGHT: Use datetime objects
start_time=datetime.now()
# Or let validators coerce ISO strings (v1.x+)
```

**SSRF protection order:**
```python
# WRONG: Setting allow_private_urls after validation
self.endpoint = self.validate_endpoint(endpoint)
self.allow_private_urls = allow_private_urls  # Too late!

# RIGHT: Set BEFORE calling validate_endpoint
self.allow_private_urls = allow_private_urls
self.endpoint = self.validate_endpoint(endpoint)
```

### 3. Register in CLI

Update `evalview/cli.py` to include your adapter:

```python
from evalview.adapters.my_adapter import MyAdapter

# In the run command, add to adapter selection:
elif adapter_type == "my-adapter":
    adapter = MyAdapter(
        endpoint=config["endpoint"],
        headers=config.get("headers", {}),
        timeout=config.get("timeout", 30.0),
        verbose=verbose,
        model_config=model_config,
        allow_private_urls=allow_private_urls,
    )
```

### 4. Write Tests

Create `tests/test_my_adapter.py`:

```python
import pytest
from datetime import datetime
from unittest.mock import AsyncMock, patch

from evalview.adapters.my_adapter import MyAdapter
from evalview.core.types import TokenUsage


class TestMyAdapter:
    """Tests for MyAdapter."""

    def test_name_property(self):
        adapter = MyAdapter(
            endpoint="http://localhost:8000",
            allow_private_urls=True,
        )
        assert adapter.name == "my-adapter"

    @pytest.mark.asyncio
    async def test_execute_basic(self):
        adapter = MyAdapter(
            endpoint="http://localhost:8000",
            allow_private_urls=True,
        )

        mock_response = {
            "session_id": "test-123",
            "steps": [
                {"tool": "search", "output": "results"},
            ],
            "response": "Final answer",
            "total_tokens": 1500,
        }

        with patch.object(adapter, '_make_request', return_value=mock_response):
            trace = await adapter.execute("test query")

            assert trace.session_id == "test-123"
            assert len(trace.steps) == 1
            assert trace.steps[0].tool_name == "search"
            assert trace.final_output == "Final answer"

    def test_token_coercion(self):
        """Test that integer tokens are coerced to TokenUsage."""
        adapter = MyAdapter(
            endpoint="http://localhost:8000",
            allow_private_urls=True,
        )

        data = {"total_tokens": 1500}
        metrics = adapter._calculate_metrics(
            data, [], datetime.now(), datetime.now()
        )

        # Should be coerced to TokenUsage by Pydantic validators
        assert isinstance(metrics.total_tokens, TokenUsage)
```

### 5. Validate Your Adapter

Use the built-in validator:

```bash
# Test your adapter
evalview validate-adapter --endpoint http://localhost:8000 --adapter my-adapter

# With custom query
evalview validate-adapter --endpoint http://localhost:8000 --adapter my-adapter --query "Hello"
```

## Testing Your Changes

### Manual Testing

```bash
# Initialize a test project
evalview init --dir /tmp/test-evalview

# Create a simple test case
cat > /tmp/test-evalview/tests/test-cases/simple.yaml <<EOF
name: "Simple Test"
input:
  query: "Hello"
expected:
  tools: []
  output:
    contains: ["hello"]
thresholds:
  min_score: 0
  max_cost: 1.0
  max_latency: 10000
EOF

# Run tests
cd /tmp/test-evalview
evalview run --verbose
```

### Automated Testing

```bash
# Run test suite
make test

# Run specific test file
pytest tests/test_my_evaluator.py

# Run with verbose output
pytest -v tests/test_my_evaluator.py
```

## Code Style

- **Formatting**: Use `black` with 100-character line length
- **Linting**: Use `ruff` for code quality
- **Type hints**: All functions should have type annotations
- **Docstrings**: Use Google-style docstrings for public APIs

Example:

```python
def my_function(arg1: str, arg2: int) -> Dict[str, Any]:
    """
    One-line summary of the function.

    Longer description if needed.

    Args:
        arg1: Description of arg1
        arg2: Description of arg2

    Returns:
        Description of return value

    Raises:
        ValueError: When something goes wrong
    """
    pass
```

## Commit Guidelines

- Use clear, descriptive commit messages
- Prefix commits with type: `feat:`, `fix:`, `docs:`, `refactor:`, `test:`
- Examples:
  - `feat: add token usage tracking to cost evaluator`
  - `fix: handle missing tool_name in HTTP adapter`
  - `docs: update CONTRIBUTING with evaluator guide`

## Pull Request Process

1. **Fork** the repository
2. **Create a branch**: `git checkout -b feat/my-feature`
3. **Make changes** and commit with clear messages
4. **Run checks**: `make check` (format, lint, typecheck)
5. **Run tests**: `make test`
6. **Push**: `git push origin feat/my-feature`
7. **Open PR** with description of changes

### PR Checklist

- [ ] Code follows style guidelines (`make check` passes)
- [ ] Tests pass (`make test` passes)
- [ ] New code has tests
- [ ] Documentation updated (README, docstrings)
- [ ] CHANGELOG updated (if applicable)

## Questions?

- Open an [issue](https://github.com/hidai25/EvalView/issues)
- Start a [discussion](https://github.com/hidai25/EvalView/discussions)

---

**Thank you for contributing to EvalView!** 🎉
