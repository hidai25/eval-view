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
