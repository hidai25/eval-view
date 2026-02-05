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
        """Register built-in adapters with graceful fallback.

        Adapters are registered in order of expected usage frequency.
        Each adapter import is wrapped in try/except to handle
        missing optional dependencies gracefully.
        """
        # Claude Code adapter - primary adapter for Claude Code CLI
        try:
            from evalview.skills.adapters.claude_code_adapter import ClaudeCodeAdapter

            cls.register(AgentType.CLAUDE_CODE.value, ClaudeCodeAdapter)
        except ImportError as e:
            logger.debug(f"ClaudeCodeAdapter not available: {e}")

        # Claude Agent Teams adapter - multi-agent teams via Claude Code
        try:
            from evalview.skills.adapters.claude_agent_sdk_adapter import ClaudeAgentTeamsAdapter

            cls.register(AgentType.CLAUDE_AGENT_TEAMS.value, ClaudeAgentTeamsAdapter)
        except ImportError as e:
            logger.debug(f"ClaudeAgentTeamsAdapter not available: {e}")

        # Codex adapter - OpenAI Codex CLI
        try:
            from evalview.skills.adapters.codex_adapter import CodexAdapter

            cls.register(AgentType.CODEX.value, CodexAdapter)
        except ImportError as e:
            logger.debug(f"CodexAdapter not available: {e}")

        # LangGraph adapter - LangGraph SDK/Cloud
        try:
            from evalview.skills.adapters.langgraph_adapter import LangGraphSkillAdapter

            cls.register(AgentType.LANGGRAPH.value, LangGraphSkillAdapter)
        except ImportError as e:
            logger.debug(f"LangGraphSkillAdapter not available: {e}")

        # CrewAI adapter - CrewAI multi-agent framework
        try:
            from evalview.skills.adapters.crewai_adapter import CrewAISkillAdapter

            cls.register(AgentType.CREWAI.value, CrewAISkillAdapter)
        except ImportError as e:
            logger.debug(f"CrewAISkillAdapter not available: {e}")

        # OpenAI Assistants adapter - OpenAI Assistants API
        try:
            from evalview.skills.adapters.openai_assistants_adapter import (
                OpenAIAssistantsSkillAdapter,
            )

            cls.register(AgentType.OPENAI_ASSISTANTS.value, OpenAIAssistantsSkillAdapter)
        except ImportError as e:
            logger.debug(f"OpenAIAssistantsSkillAdapter not available: {e}")

        # Custom adapter - user-provided scripts (always last as fallback)
        try:
            from evalview.skills.adapters.custom_adapter import CustomAdapter

            cls.register(AgentType.CUSTOM.value, CustomAdapter)
        except ImportError as e:
            logger.debug(f"CustomAdapter not available: {e}")

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
