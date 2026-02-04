"""Chaos/fault injection tools for testing agent resilience.

This module provides controlled failure injection for testing how agents
handle real-world problems: timeouts, malformed data, rate limits, etc.

Usage in test YAML:
    input:
      query: "Search for refund policy"
      context:
        chaos:
          timeout: true
          # or: malformed: true
          # or: rate_limit: true
          # or: empty: true
          # or: error: "Custom error message"
"""

import json
import time
from dataclasses import dataclass, field
from functools import wraps
from typing import Any, Callable, Dict, List, Optional


@dataclass
class ChaosConfig:
    """Configuration for chaos/fault injection.

    All probabilities are 0.0-1.0, but for deterministic gym tests,
    we typically use 1.0 (always trigger) or 0.0 (never trigger).
    """

    # Failure modes
    timeout: bool = False
    timeout_seconds: float = 5.0

    malformed: bool = False
    malformed_response: str = "{{invalid json}}"

    rate_limit: bool = False
    rate_limit_retry_after: int = 60

    empty: bool = False

    error: Optional[str] = None

    # Loop detection
    max_calls_per_tool: int = 10

    # Latency injection
    latency_ms: int = 0

    # Partial failure (only specific tools fail)
    failing_tools: List[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "ChaosConfig":
        """Create ChaosConfig from a dictionary (e.g., from YAML context)."""
        if not data:
            return cls()

        return cls(
            timeout=data.get("timeout", False),
            timeout_seconds=data.get("timeout_seconds", 5.0),
            malformed=data.get("malformed", False),
            malformed_response=data.get("malformed_response", "{{invalid json}}"),
            rate_limit=data.get("rate_limit", False),
            rate_limit_retry_after=data.get("rate_limit_retry_after", 60),
            empty=data.get("empty", False),
            error=data.get("error"),
            max_calls_per_tool=data.get("max_calls_per_tool", 10),
            latency_ms=data.get("latency_ms", 0),
            failing_tools=data.get("failing_tools", []),
        )


class ChaosRegistry:
    """Global registry for chaos state and tool call tracking.

    This singleton tracks:
    - Current chaos configuration
    - Number of calls per tool (for loop detection)
    """

    _instance: Optional["ChaosRegistry"] = None
    _config: ChaosConfig
    _call_counts: Dict[str, int]

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._config = ChaosConfig()
            cls._instance._call_counts = {}
        return cls._instance

    @classmethod
    def configure(cls, config: ChaosConfig) -> None:
        """Set the chaos configuration."""
        instance = cls()
        instance._config = config
        instance._call_counts = {}  # Reset counts on new config

    @classmethod
    def get_config(cls) -> ChaosConfig:
        """Get current chaos configuration."""
        return cls()._config

    @classmethod
    def reset(cls) -> None:
        """Reset chaos state (call between tests)."""
        instance = cls()
        instance._config = ChaosConfig()
        instance._call_counts = {}

    @classmethod
    def record_call(cls, tool_name: str) -> int:
        """Record a tool call and return the count."""
        instance = cls()
        instance._call_counts[tool_name] = instance._call_counts.get(tool_name, 0) + 1
        return instance._call_counts[tool_name]

    @classmethod
    def get_call_count(cls, tool_name: str) -> int:
        """Get the number of times a tool has been called."""
        return cls()._call_counts.get(tool_name, 0)


class ChaosError(Exception):
    """Base exception for chaos-induced failures."""

    pass


class ToolTimeoutError(ChaosError):
    """Simulated timeout error."""

    def __init__(self, tool_name: str, timeout_seconds: float):
        super().__init__(f"Tool '{tool_name}' timed out after {timeout_seconds}s")
        self.tool_name = tool_name
        self.timeout_seconds = timeout_seconds


class RateLimitError(ChaosError):
    """Simulated rate limit error (429)."""

    def __init__(self, tool_name: str, retry_after: int):
        super().__init__(f"Rate limited. Retry after {retry_after} seconds.")
        self.tool_name = tool_name
        self.retry_after = retry_after


class LoopDetectedError(ChaosError):
    """Tool called too many times - likely an infinite loop."""

    def __init__(self, tool_name: str, call_count: int, max_calls: int):
        super().__init__(
            f"Loop detected: '{tool_name}' called {call_count} times (max: {max_calls})"
        )
        self.tool_name = tool_name
        self.call_count = call_count
        self.max_calls = max_calls


def apply_chaos(tool_name: str) -> Optional[Any]:
    """Apply chaos effects and return a failure response if triggered.

    Args:
        tool_name: Name of the tool being called

    Returns:
        None if no chaos triggered, otherwise a failure response

    Raises:
        ToolTimeoutError: If timeout chaos is enabled
        RateLimitError: If rate limit chaos is enabled
        LoopDetectedError: If tool called too many times
        ChaosError: If generic error chaos is enabled
    """
    config = ChaosRegistry.get_config()

    # Check if this specific tool should fail (partial failure mode)
    if config.failing_tools and tool_name not in config.failing_tools:
        return None  # This tool is not configured to fail

    # Record call and check for loops
    call_count = ChaosRegistry.record_call(tool_name)
    if call_count > config.max_calls_per_tool:
        raise LoopDetectedError(tool_name, call_count, config.max_calls_per_tool)

    # Apply latency
    if config.latency_ms > 0:
        time.sleep(config.latency_ms / 1000.0)

    # Check failure modes (in order of severity)
    if config.timeout:
        # Simulate timeout by sleeping then raising
        time.sleep(min(config.timeout_seconds, 2.0))  # Cap at 2s for tests
        raise ToolTimeoutError(tool_name, config.timeout_seconds)

    if config.rate_limit:
        raise RateLimitError(tool_name, config.rate_limit_retry_after)

    if config.error:
        raise ChaosError(config.error)

    if config.malformed:
        return config.malformed_response

    if config.empty:
        return None

    return None  # No chaos triggered


def chaos_tool(func: Callable) -> Callable:
    """Decorator to add chaos behavior to a tool function.

    Usage:
        @tool
        @chaos_tool
        def my_tool(query: str) -> str:
            return "normal response"
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        tool_name = func.__name__

        # Apply chaos - may raise or return failure response
        chaos_result = apply_chaos(tool_name)

        if chaos_result is not None:
            # Return the chaos response instead of calling the real function
            return chaos_result

        # No chaos - call the real function
        return func(*args, **kwargs)

    return wrapper


def configure_chaos_from_context(context: Optional[Dict[str, Any]]) -> None:
    """Configure chaos from a test context dictionary.

    This is called at the start of each test to set up chaos behavior
    based on the YAML configuration.

    Args:
        context: The 'context' field from test input, may contain 'chaos' key
    """
    if not context:
        ChaosRegistry.reset()
        return

    chaos_data = context.get("chaos", {})
    config = ChaosConfig.from_dict(chaos_data)
    ChaosRegistry.configure(config)
