"""Base agent adapter interface."""

from abc import ABC, abstractmethod
from typing import Any, Optional, Dict, Set
from evalview.core.types import ExecutionTrace
from evalview.core.security import validate_url


class AgentAdapter(ABC):
    """Abstract adapter for connecting to different agent frameworks.

    Security Note:
        All adapters include SSRF (Server-Side Request Forgery) protection by default.
        This prevents requests to internal networks, cloud metadata endpoints, and
        other potentially dangerous destinations. Set `allow_private_urls=True` only
        in trusted development environments.
    """

    # SSRF protection settings (can be overridden in subclasses or instances)
    allow_private_urls: bool = False
    allowed_hosts: Optional[Set[str]] = None
    blocked_hosts: Optional[Set[str]] = None

    @property
    @abstractmethod
    def name(self) -> str:
        """Name of the adapter."""
        pass

    @abstractmethod
    async def execute(self, query: str, context: Optional[Dict[str, Any]] = None) -> ExecutionTrace:
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

    def validate_endpoint(self, url: str) -> str:
        """
        Validate an endpoint URL for SSRF protection.

        Args:
            url: The URL to validate

        Returns:
            The validated URL

        Raises:
            SSRFProtectionError: If the URL fails security validation
        """
        return validate_url(
            url,
            allow_private=self.allow_private_urls,
            allowed_hosts=self.allowed_hosts,
            blocked_hosts=self.blocked_hosts,
        )
