"""Pytest configuration and shared fixtures for EvalView tests."""

import json
from datetime import datetime
from typing import Dict, Any, List
from pathlib import Path
import pytest
from unittest.mock import AsyncMock, MagicMock

from evalview.core.types import (
    TestCase,
    TestInput,
    ExpectedBehavior,
    ExpectedOutput,
    Thresholds,
    ExecutionTrace,
    StepTrace,
    StepMetrics,
    ExecutionMetrics,
    TokenUsage,
)


# ============================================================================
# Test Data Fixtures
# ============================================================================


@pytest.fixture
def sample_test_case() -> TestCase:
    """Create a sample test case for testing."""
    return TestCase(
        name="test_search",
        description="Test search functionality",
        input=TestInput(
            query="What is the capital of France?",
            context={"language": "en"},
        ),
        expected=ExpectedBehavior(
            tools=["search", "summarize"],
            tool_sequence=["search", "summarize"],
            output=ExpectedOutput(
                contains=["Paris", "France"],
                not_contains=["London", "error"],
            ),
        ),
        thresholds=Thresholds(
            min_score=70.0,
            max_cost=0.50,
            max_latency=5000.0,
        ),
    )


@pytest.fixture
def sample_execution_trace() -> ExecutionTrace:
    """Create a sample execution trace for testing."""
    start_time = datetime(2025, 1, 1, 12, 0, 0)
    end_time = datetime(2025, 1, 1, 12, 0, 3)

    return ExecutionTrace(
        session_id="test-session-123",
        start_time=start_time,
        end_time=end_time,
        steps=[
            StepTrace(
                step_id="step-1",
                step_name="Search",
                tool_name="search",
                parameters={"query": "capital of France"},
                output={"results": ["Paris is the capital of France"]},
                success=True,
                error=None,
                metrics=StepMetrics(
                    latency=1500.0,
                    cost=0.02,
                    tokens=TokenUsage(input_tokens=50, output_tokens=100, cached_tokens=0),
                ),
            ),
            StepTrace(
                step_id="step-2",
                step_name="Summarize",
                tool_name="summarize",
                parameters={"text": "Paris is the capital of France"},
                output="Paris is the capital of France.",
                success=True,
                error=None,
                metrics=StepMetrics(
                    latency=1000.0,
                    cost=0.01,
                    tokens=TokenUsage(input_tokens=30, output_tokens=20, cached_tokens=0),
                ),
            ),
        ],
        final_output="The capital of France is Paris.",
        metrics=ExecutionMetrics(
            total_cost=0.03,
            total_latency=3000.0,
            total_tokens=TokenUsage(input_tokens=80, output_tokens=120, cached_tokens=0),
        ),
    )


@pytest.fixture
def minimal_test_case() -> TestCase:
    """Create a minimal test case with no optional fields."""
    return TestCase(
        name="minimal_test",
        input=TestInput(query="test query"),
        expected=ExpectedBehavior(),
        thresholds=Thresholds(min_score=0.0),
    )


@pytest.fixture
def empty_trace() -> ExecutionTrace:
    """Create an execution trace with no steps."""
    start_time = datetime(2025, 1, 1, 12, 0, 0)
    end_time = datetime(2025, 1, 1, 12, 0, 1)

    return ExecutionTrace(
        session_id="empty-session",
        start_time=start_time,
        end_time=end_time,
        steps=[],
        final_output="",
        metrics=ExecutionMetrics(
            total_cost=0.0,
            total_latency=1000.0,
            total_tokens=None,
        ),
    )


# ============================================================================
# HTTP Response Fixtures
# ============================================================================


@pytest.fixture
def http_response_flat() -> Dict[str, Any]:
    """HTTP response with flat structure."""
    return {
        "response": "Paris is the capital of France.",
        "cost": 0.05,
        "tokens": 150,
        "latency": 2500,
    }


@pytest.fixture
def http_response_nested() -> Dict[str, Any]:
    """HTTP response with nested metadata."""
    return {
        "output": "Paris is the capital of France.",
        "metadata": {
            "cost": 0.05,
            "tokens": {
                "input": 50,
                "output": 100,
                "cached": 0,
            },
            "latency": 2500,
        },
    }


@pytest.fixture
def http_response_with_steps() -> Dict[str, Any]:
    """HTTP response with detailed steps."""
    return {
        "session_id": "session-123",
        "response": "Paris is the capital of France.",
        "steps": [
            {
                "id": "step-1",
                "name": "Search",
                "tool": "search",
                "parameters": {"query": "capital of France"},
                "output": {"results": ["Paris"]},
                "success": True,
                "latency": 1500,
                "cost": 0.02,
            },
            {
                "id": "step-2",
                "name": "Summarize",
                "tool": "summarize",
                "parameters": {"text": "Paris"},
                "output": "Paris is the capital.",
                "success": True,
                "latency": 1000,
                "cost": 0.01,
            },
        ],
        "cost": 0.03,
        "tokens": {"input": 80, "output": 120},
    }


@pytest.fixture
def http_response_minimal() -> Dict[str, Any]:
    """HTTP response with minimal fields."""
    return {
        "result": "Test output",
    }


@pytest.fixture
def http_response_with_tokens_only() -> Dict[str, Any]:
    """HTTP response with tokens but no cost (should calculate)."""
    return {
        "response": "Test output",
        "tokens": {
            "input_tokens": 100,
            "output_tokens": 200,
            "cached_tokens": 50,
        },
    }


# ============================================================================
# Mock Fixtures
# ============================================================================


@pytest.fixture
def mock_openai_client() -> AsyncMock:
    """Mock OpenAI client for LLM-as-judge testing."""
    mock_client = AsyncMock()
    mock_response = MagicMock()
    mock_response.choices = [
        MagicMock(
            message=MagicMock(
                content=json.dumps(
                    {
                        "score": 85,
                        "rationale": "The output correctly answers the question.",
                    }
                )
            )
        )
    ]
    mock_client.chat.completions.create.return_value = mock_response
    return mock_client


@pytest.fixture
def mock_httpx_client() -> AsyncMock:
    """Mock httpx client for adapter testing."""
    mock_client = AsyncMock()
    mock_response = AsyncMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "response": "Test output",
        "cost": 0.05,
    }
    mock_client.post.return_value = mock_response
    mock_client.get.return_value = mock_response
    return mock_client


# ============================================================================
# Temporary File Fixtures
# ============================================================================


@pytest.fixture
def temp_yaml_file(tmp_path: Path) -> Path:
    """Create a temporary valid YAML test case file."""
    yaml_content = """
name: test_search
description: Test search functionality
input:
  query: What is the capital of France?
  context:
    language: en
expected:
  tools:
    - search
    - summarize
  tool_sequence:
    - search
    - summarize
  output:
    contains:
      - Paris
      - France
    not_contains:
      - London
      - error
thresholds:
  min_score: 70.0
  max_cost: 0.50
  max_latency: 5000.0
"""
    file_path = tmp_path / "test_case.yaml"
    file_path.write_text(yaml_content)
    return file_path


@pytest.fixture
def temp_invalid_yaml_file(tmp_path: Path) -> Path:
    """Create a temporary invalid YAML file (malformed)."""
    file_path = tmp_path / "invalid.yaml"
    file_path.write_text("invalid: yaml: content:\n  - missing\n  bracket")
    return file_path


@pytest.fixture
def temp_invalid_schema_file(tmp_path: Path) -> Path:
    """Create a YAML file with invalid schema (missing required fields)."""
    yaml_content = """
name: invalid_test
# Missing required 'input', 'expected', and 'thresholds' fields
description: This will fail validation
"""
    file_path = tmp_path / "invalid_schema.yaml"
    file_path.write_text(yaml_content)
    return file_path


@pytest.fixture
def temp_yaml_directory(tmp_path: Path) -> Path:
    """Create a directory with multiple YAML test files."""
    test_dir = tmp_path / "test_cases"
    test_dir.mkdir()

    # Create test1.yaml
    (test_dir / "test1.yaml").write_text(
        """
name: test1
input:
  query: test query 1
expected:
  tools: []
thresholds:
  min_score: 50.0
"""
    )

    # Create test2.yml (different extension)
    (test_dir / "test2.yml").write_text(
        """
name: test2
input:
  query: test query 2
expected:
  tools: []
thresholds:
  min_score: 60.0
"""
    )

    # Create a non-YAML file (should be ignored)
    (test_dir / "readme.txt").write_text("Not a YAML file")

    return test_dir


# ============================================================================
# Async Testing Utilities
# ============================================================================


@pytest.fixture
def event_loop():
    """Create an event loop for async tests."""
    import asyncio

    loop = asyncio.new_event_loop()
    yield loop
    loop.close()
