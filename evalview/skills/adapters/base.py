"""Base adapter for skill agent testing.

This module defines the abstract interface that all skill adapters must implement.
Unlike main AgentAdapter (which tests agents via REST APIs), SkillAgentAdapter:
1. Injects skills into agents as context
2. Captures detailed execution traces
3. Monitors file system and command execution
"""

from abc import ABC, abstractmethod
from typing import Any, Optional, Dict
import logging

from evalview.skills.agent_types import AgentConfig, SkillAgentTrace
from evalview.skills.types import Skill

logger = logging.getLogger(__name__)


class SkillAgentAdapterError(Exception):
    """Base exception for skill adapter errors.

    Attributes:
        message: Human-readable error description
        adapter_name: Which adapter raised the error
        recoverable: Whether retry might succeed
    """

    def __init__(
        self,
        message: str,
        adapter_name: str = "unknown",
        recoverable: bool = False,
    ):
        self.message = message
        self.adapter_name = adapter_name
        self.recoverable = recoverable
        super().__init__(f"[{adapter_name}] {message}")


class AgentNotFoundError(SkillAgentAdapterError):
    """Raised when agent CLI/binary is not found."""

    def __init__(self, adapter_name: str, install_hint: str):
        super().__init__(
            f"Agent not found. {install_hint}",
            adapter_name=adapter_name,
            recoverable=False,
        )
        self.install_hint = install_hint


class AgentTimeoutError(SkillAgentAdapterError):
    """Raised when agent execution times out."""

    def __init__(self, adapter_name: str, timeout: float):
        super().__init__(
            f"Execution timed out after {timeout}s",
            adapter_name=adapter_name,
            recoverable=True,
        )
        self.timeout = timeout


class SkillAgentAdapter(ABC):
    """Abstract adapter for executing skills through AI agents.

    Each adapter implementation handles:
    1. Building the command/request to invoke the agent
    2. Injecting the skill into agent context
    3. Executing the agent with timeout
    4. Parsing output to extract trace events
    5. Building SkillAgentTrace from execution

    Security considerations:
    - Working directories should be isolated
    - Commands should be logged for audit
    - Timeouts must be enforced
    - Environment variables should not leak secrets
    """

    def __init__(self, config: AgentConfig):
        """Initialize with agent configuration.

        Args:
            config: Agent configuration from test suite
        """
        self.config = config
        self._last_raw_output: Optional[str] = None

    @property
    @abstractmethod
    def name(self) -> str:
        """Adapter identifier (e.g., 'claude-code', 'codex')."""
        pass

    @abstractmethod
    async def execute(
        self,
        skill: Skill,
        query: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> SkillAgentTrace:
        """Execute a skill test and capture trace.

        Args:
            skill: The loaded skill to test
            query: User query to send to the agent
            context: Optional execution context (test_name, cwd override)

        Returns:
            SkillAgentTrace with all execution events

        Raises:
            AgentNotFoundError: If agent binary not found
            AgentTimeoutError: If execution exceeds timeout
            SkillAgentAdapterError: For other execution errors
        """
        pass

    async def health_check(self) -> bool:
        """Check if the agent is available and working.

        Returns:
            True if agent can be executed, False otherwise
        """
        return True

    def get_last_raw_output(self) -> Optional[str]:
        """Get raw output from last execution for debugging.

        Returns:
            Raw stdout/stderr from last execution, or None
        """
        return self._last_raw_output
