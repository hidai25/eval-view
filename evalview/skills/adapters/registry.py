"""Registry for skill agent adapters.

Provides a centralized registry for adapter classes, allowing dynamic
adapter discovery and creation based on AgentType configuration.
"""

from typing import Dict, List, Optional, Type
import logging

from evalview.skills.agent_types import AgentConfig, AgentType
from evalview.skills.adapters.base import SkillAgentAdapter

logger = logging.getLogger(__name__)


class SkillAdapterRegistry:
    """Factory for skill agent adapters.

    Usage:
        adapter = SkillAdapterRegistry.create(config)
        trace = await adapter.execute(skill, query)
    """

    _adapters: Dict[str, Type[SkillAgentAdapter]] = {}
    _initialized: bool = False

    @classmethod
    def register(cls, name: str, adapter_class: Type[SkillAgentAdapter]) -> None:
        """Register an adapter class.

        Args:
            name: Unique identifier for the adapter (e.g., "claude-code")
            adapter_class: The adapter class to register
        """
        if name in cls._adapters:
            logger.warning(f"Overwriting existing adapter registration: {name}")

        cls._adapters[name] = adapter_class
        logger.debug(f"Registered skill adapter: {name} -> {adapter_class.__name__}")

    @classmethod
    def get(cls, name: str) -> Optional[Type[SkillAgentAdapter]]:
        """Get adapter class by name.

        Args:
            name: The adapter identifier

        Returns:
            The adapter class, or None if not found
        """
        cls._ensure_initialized()
        return cls._adapters.get(name)

    @classmethod
    def list_adapters(cls) -> Dict[str, Type[SkillAgentAdapter]]:
        """Get all registered adapters.

        Returns:
            Dictionary mapping adapter names to classes
        """
        cls._ensure_initialized()
        return cls._adapters.copy()

    @classmethod
    def list_names(cls) -> List[str]:
        """Get list of registered adapter names.

        Returns:
            List of adapter names
        """
        cls._ensure_initialized()
        return list(cls._adapters.keys())

    @classmethod
    def create(cls, config: AgentConfig) -> SkillAgentAdapter:
        """Create adapter instance from config.

        Args:
            config: Agent configuration containing type and settings

        Returns:
            Configured adapter instance

        Raises:
            ValueError: If adapter type is not registered
        """
        cls._ensure_initialized()

        adapter_name = config.type.value
        adapter_class = cls._adapters.get(adapter_name)

        if adapter_class is None:
            available = ", ".join(cls._adapters.keys())
            raise ValueError(
                f"Unknown skill adapter: '{adapter_name}'. "
                f"Available adapters: {available}"
            )

        return adapter_class(config)

    @classmethod
    def _ensure_initialized(cls) -> None:
        """Ensure built-in adapters are registered."""
        if cls._initialized:
            return

        cls._register_builtin_adapters()
        cls._initialized = True

    @classmethod
    def _register_builtin_adapters(cls) -> None:
        """Register built-in adapters with graceful fallback."""
        # Claude Code adapter
        try:
            from evalview.skills.adapters.claude_code_adapter import ClaudeCodeAdapter

            cls.register(AgentType.CLAUDE_CODE.value, ClaudeCodeAdapter)
        except ImportError as e:
            logger.debug(f"ClaudeCodeAdapter not available: {e}")

        # Custom adapter (script-based)
        try:
            from evalview.skills.adapters.custom_adapter import CustomAdapter

            cls.register(AgentType.CUSTOM.value, CustomAdapter)
        except ImportError as e:
            logger.debug(f"CustomAdapter not available: {e}")

        # Codex adapter (placeholder for future)
        try:
            from evalview.skills.adapters.codex_adapter import CodexAdapter

            cls.register(AgentType.CODEX.value, CodexAdapter)
        except ImportError as e:
            logger.debug(f"CodexAdapter not available: {e}")

    @classmethod
    def reset(cls) -> None:
        """Reset the registry (useful for testing).

        Clears all registrations and resets initialization flag.
        """
        cls._adapters.clear()
        cls._initialized = False


def get_skill_adapter(config: AgentConfig) -> SkillAgentAdapter:
    """Convenience function to create a skill adapter.

    Args:
        config: Agent configuration

    Returns:
        Configured adapter instance

    Example:
        >>> config = AgentConfig(type=AgentType.CLAUDE_CODE)
        >>> adapter = get_skill_adapter(config)
    """
    return SkillAdapterRegistry.create(config)
