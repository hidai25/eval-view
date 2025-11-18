"""Base agent adapter interface."""

from abc import ABC, abstractmethod
from typing import Any, Optional, Dict
from agent_eval.core.types import ExecutionTrace


class AgentAdapter(ABC):
    """Abstract adapter for connecting to different agent frameworks."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Name of the adapter."""
        pass

    @abstractmethod
    async def execute(
        self, query: str, context: Optional[Dict[str, Any]] = None
    ) -> ExecutionTrace:
        """
        Execute agent with given input and capture trace.

        Args:
            query: The user query to send to the agent
            context: Optional context/metadata for the query

        Returns:
            ExecutionTrace containing the full execution history
        """
        pass

    async def health_check(self) -> bool:
        """
        Optional health check for agent availability.

        Returns:
            True if agent is healthy, False otherwise
        """
        return True
