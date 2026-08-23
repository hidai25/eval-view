"""
Adapter test fixtures for EvalView.

Provides sample responses for different adapter types and a helper class
for common assertions.
"""

from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
from dataclasses import dataclass

from evalview.core.types import (
    ExecutionTrace,
    StepTrace,
    StepMetrics,
    ExecutionMetrics,
    TokenUsage,
)


# =============================================================================
# Sample API Responses
# =============================================================================


def langgraph_standard_response() -> Dict[str, Any]:
    """Sample LangGraph standard (non-cloud) API response."""
    return {
        "messages": [
            {
                "type": "human",
                "content": "What is 2+2?",
            },
            {
                "type": "ai",
                "content": "2+2 equals 4.",
                "tool_calls": [],
            },
        ],
        "output": "2+2 equals 4.",
    }


def langgraph_with_tools_response() -> Dict[str, Any]:
    """Sample LangGraph response with tool calls."""
    return {
        "messages": [
            {
                "type": "human",
                "content": "Search for the capital of France",
            },
            {
                "type": "ai",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_123",
                        "name": "search",
                        "args": {"query": "capital of France"},
                    }
                ],
            },
            {
                "type": "tool",
                "name": "search",
                "content": "Paris is the capital of France.",
                "tool_call_id": "call_123",
            },
            {
                "type": "ai",
                "content": "The capital of France is Paris.",
                "usage_metadata": {
                    "input_tokens": 50,
                    "output_tokens": 100,
                    "total_tokens": 150,
                },
            },
        ],
        "output": "The capital of France is Paris.",
    }


def langgraph_cloud_response() -> Dict[str, Any]:
    """Sample LangGraph Cloud API response (from thread/run)."""
    return {
        "thread_id": "thread_abc123",
        "run_id": "run_xyz789",
        "status": "success",
        "agent": {
            "messages": [
                {
                    "type": "ai",
                    "content": "I'll search for that information.",
                    "response_metadata": {
                        "token_usage": {
                            "prompt_tokens": 50,
                            "completion_tokens": 100,
                            "total_tokens": 150,
                        }
                    },
                }
            ]
        },
    }


def crewai_tasks_response() -> Dict[str, Any]:
    """Sample CrewAI response with 'tasks' format."""
    return {
        "crew_id": "crew_abc123",
        "tasks": [
            {
                "id": "task-1",
                "description": "Research the topic",
                "tool": "web_search",
                "inputs": {"query": "AI developments 2025"},
                "output": "Found 5 relevant articles",
                "status": "completed",
                "duration": 2500.0,
            },
            {
                "id": "task-2",
                "description": "Summarize findings",
                "tool": "summarize",
                "inputs": {"text": "..."},
                "output": "AI is advancing rapidly in 2025",
                "status": "completed",
                "duration": 1500.0,
            },
        ],
        "result": "AI developments in 2025 include...",
        "usage_metrics": {
            "total_tokens": 2500,
            "total_cost": 0.05,
        },
    }


def crewai_agent_executions_response() -> Dict[str, Any]:
    """Sample CrewAI response with 'agent_executions' format."""
    return {
        "crew_id": "crew_def456",
        "agent_executions": [
            {
                "agent_name": "Researcher",
                "tool_used": "web_search",
                "output": "Found relevant information",
            },
            {
                "agent_name": "Writer",
                "tool_used": "text_generator",
                "output": "Generated summary",
            },
        ],
        "final_output": "The research findings indicate...",
    }


def http_adapter_flat_response() -> Dict[str, Any]:
    """Sample HTTP adapter response - flat format."""
    return {
        "response": "The answer is 42.",
        "cost": 0.02,
        "tokens": 1500,
        "latency": 2500.0,
    }


def http_adapter_nested_response() -> Dict[str, Any]:
    """Sample HTTP adapter response - nested format with metadata."""
    return {
        "response": "The capital of France is Paris.",
        "metadata": {
            "cost": 0.03,
            "tokens": {
                "input": 50,
                "output": 100,
                "cached": 0,
            },
        },
        "steps": [
            {
                "tool_name": "search",
                "parameters": {"query": "capital France"},
                "output": "Paris",
                "latency": 1000.0,
                "cost": 0.01,
            },
            {
                "tool_name": "format",
                "parameters": {"template": "answer"},
                "output": "The capital of France is Paris.",
                "latency": 500.0,
                "cost": 0.02,
            },
        ],
    }


def tapescope_events() -> List[Dict[str, Any]]:
    """Sample TapeScope/JSONL streaming events."""
    return [
        {"event": "start", "timestamp": "2025-01-15T10:30:00Z"},
        {
            "event": "tool_call",
            "data": {
                "tool_name": "search",
                "args": {"query": "test"},
            },
        },
        {
            "event": "tool_result",
            "data": {
                "result": "Search results here",
                "success": True,
            },
        },
        {
            "event": "usage",
            "data": {
                "input_tokens": 100,
                "output_tokens": 200,
                "cost": 0.03,
            },
        },
        {
            "event": "message_complete",
            "data": {
                "content": "Based on the search results, I found...",
            },
        },
    ]


@dataclass
class MockOpenAIResponse:
    """Mock OpenAI Responses API response object."""

    id: str = "resp_abc123"
    status: str = "completed"
    model: str = "gpt-4o"
    output_text: str = "Done."

    @property
    def usage(self):
        """Mock usage object."""

        class Usage:
            total_tokens = 1500
            input_tokens = 500
            output_tokens = 1000

        return Usage()

    @property
    def output(self):
        return []

    @property
    def error(self):
        return None


# =============================================================================
# Helper Class
# =============================================================================


class AdapterTestHelper:
    """Helper class for common adapter test assertions."""

    @staticmethod
    def assert_valid_trace(trace: ExecutionTrace) -> None:
        """Validate ExecutionTrace structure."""
        assert trace is not None
        assert isinstance(trace.session_id, str)
        assert len(trace.session_id) > 0

        assert isinstance(trace.start_time, datetime)
        assert isinstance(trace.end_time, datetime)
        assert trace.end_time >= trace.start_time

        assert isinstance(trace.steps, list)
        assert isinstance(trace.final_output, str)

        # Validate metrics
        assert trace.metrics is not None
        assert isinstance(trace.metrics.total_cost, float)
        assert trace.metrics.total_cost >= 0
        assert isinstance(trace.metrics.total_latency, float)
        assert trace.metrics.total_latency >= 0

    @staticmethod
    def assert_valid_step(step: StepTrace) -> None:
        """Validate StepTrace structure."""
        assert step is not None
        assert isinstance(step.step_id, str)
        assert isinstance(step.step_name, str)
        assert isinstance(step.tool_name, str)
        assert isinstance(step.parameters, dict)
        assert isinstance(step.success, bool)

        # Validate step metrics
        assert step.metrics is not None
        assert isinstance(step.metrics.latency, float)
        assert step.metrics.latency >= 0
        assert isinstance(step.metrics.cost, float)
        assert step.metrics.cost >= 0

    @staticmethod
    def assert_tokens_valid(tokens: Optional[TokenUsage]) -> None:
        """Validate TokenUsage structure."""
        if tokens is None:
            return

        assert isinstance(tokens, TokenUsage)
        assert isinstance(tokens.input_tokens, int)
        assert tokens.input_tokens >= 0
        assert isinstance(tokens.output_tokens, int)
        assert tokens.output_tokens >= 0
        assert isinstance(tokens.cached_tokens, int)
        assert tokens.cached_tokens >= 0
        assert tokens.total_tokens == (
            tokens.input_tokens + tokens.output_tokens + tokens.cached_tokens
        )

    @staticmethod
    def create_sample_trace(
        session_id: str = "test-session",
        num_steps: int = 2,
        total_cost: float = 0.05,
        total_latency: float = 3000.0,
        total_tokens: Optional[int] = 1500,
    ) -> ExecutionTrace:
        """Create a sample ExecutionTrace for testing."""
        start_time = datetime.now()
        end_time = start_time + timedelta(milliseconds=total_latency)

        steps = []
        for i in range(num_steps):
            step = StepTrace(
                step_id=f"step-{i+1}",
                step_name=f"Step {i+1}",
                tool_name=f"tool_{i+1}",
                parameters={"param": f"value_{i+1}"},
                output=f"output_{i+1}",
                success=True,
                metrics=StepMetrics(
                    latency=total_latency / num_steps,
                    cost=total_cost / num_steps,
                    tokens=TokenUsage(
                        input_tokens=50 * (i + 1),
                        output_tokens=100 * (i + 1),
                    )
                    if total_tokens
                    else None,
                ),
            )
            steps.append(step)

        token_usage = TokenUsage(output_tokens=total_tokens) if total_tokens else None

        return ExecutionTrace(
            session_id=session_id,
            start_time=start_time,
            end_time=end_time,
            steps=steps,
            final_output="Sample final output",
            metrics=ExecutionMetrics(
                total_cost=total_cost,
                total_latency=total_latency,
                total_tokens=token_usage,
            ),
        )

    @staticmethod
    def assert_trace_matches_response(
        trace: ExecutionTrace,
        expected_tools: List[str],
        expected_output_contains: List[str],
    ) -> None:
        """Assert trace matches expected response characteristics."""
        # Check tools
        actual_tools = [step.tool_name for step in trace.steps]
        for tool in expected_tools:
            assert tool in actual_tools, f"Expected tool '{tool}' not found in {actual_tools}"

        # Check output contains expected strings
        for text in expected_output_contains:
            assert (
                text.lower() in trace.final_output.lower()
            ), f"Expected '{text}' not found in output: {trace.final_output[:200]}"


# =============================================================================
# Pytest Fixtures (for use with conftest.py)
# =============================================================================


def sample_langgraph_response():
    """Fixture: Sample LangGraph response."""
    return langgraph_with_tools_response()


def sample_crewai_response():
    """Fixture: Sample CrewAI response (tasks format)."""
    return crewai_tasks_response()


def sample_http_response():
    """Fixture: Sample HTTP adapter response."""
    return http_adapter_nested_response()


def sample_trace():
    """Fixture: Sample ExecutionTrace."""
    return AdapterTestHelper.create_sample_trace()
