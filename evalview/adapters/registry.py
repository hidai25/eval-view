"""
Adapter registry for EvalView.

Provides a centralized registry for adapter classes, allowing dynamic
adapter discovery and creation without modifying CLI code.

.. warning::
    This module is **experimental** and may change in future versions.
    The API is not yet stable.
"""

from typing import Any, Dict, Optional, Set, Type, List
import logging

from evalview.adapters.base import AgentAdapter

logger = logging.getLogger(__name__)


class AdapterRegistry:
    """
    Registry for adapter classes.

    Allows registering and retrieving adapter classes by name,
    and creating adapter instances from configuration.

    .. warning::
        **EXPERIMENTAL**: This API is experimental and may change
        in future versions without notice.

    Example:
        >>> from evalview.adapters.registry import AdapterRegistry
        >>> from evalview.adapters.http_adapter import HTTPAdapter
        >>>
        >>> # Register adapter
        >>> AdapterRegistry.register("http", HTTPAdapter)
        >>>
        >>> # Create adapter from config
        >>> adapter = AdapterRegistry.create(
        ...     "http",
        ...     endpoint="http://localhost:8000",
        ...     timeout=30.0,
        ... )
    """

    _adapters: Dict[str, Type[AgentAdapter]] = {}
    _initialized: bool = False

    @classmethod
    def register(cls, name: str, adapter_class: Type[AgentAdapter]) -> None:
        """
        Register an adapter class.

        Args:
            name: Unique identifier for the adapter (e.g., "http", "langgraph")
            adapter_class: The adapter class to register

        Raises:
            ValueError: If name is already registered
        """
        if name in cls._adapters:
            logger.warning(f"Overwriting existing adapter registration: {name}")

        cls._adapters[name] = adapter_class
        logger.debug(f"Registered adapter: {name} -> {adapter_class.__name__}")

    @classmethod
    def get(cls, name: str) -> Optional[Type[AgentAdapter]]:
        """
        Get adapter class by name.

        Args:
            name: The adapter identifier

        Returns:
            The adapter class, or None if not found
        """
        cls._ensure_initialized()
        return cls._adapters.get(name)

    @classmethod
    def list_adapters(cls) -> Dict[str, Type[AgentAdapter]]:
        """
        Get all registered adapters.

        Returns:
            Dictionary mapping adapter names to classes
        """
        cls._ensure_initialized()
        return cls._adapters.copy()

    @classmethod
    def list_names(cls) -> List[str]:
        """
        Get list of registered adapter names.

        Returns:
            List of adapter names
        """
        cls._ensure_initialized()
        return list(cls._adapters.keys())

    @classmethod
    def create(
        cls,
        name: str,
        endpoint: str,
        headers: Optional[Dict[str, str]] = None,
        timeout: float = 30.0,
        verbose: bool = False,
        model_config: Optional[Dict[str, Any]] = None,
        allow_private_urls: bool = False,
        allowed_hosts: Optional[Set[str]] = None,
        **kwargs: Any,
    ) -> AgentAdapter:
        """
        Create an adapter instance from configuration.

        Args:
            name: The adapter identifier
            endpoint: API endpoint URL
            headers: Optional HTTP headers
            timeout: Request timeout in seconds
            verbose: Enable verbose logging
            model_config: Model configuration
            allow_private_urls: Allow private/internal URLs
            allowed_hosts: Set of allowed hostnames
            **kwargs: Additional adapter-specific arguments

        Returns:
            Configured adapter instance

        Raises:
            ValueError: If adapter name is not registered
        """
        cls._ensure_initialized()

        adapter_class = cls._adapters.get(name)
        if adapter_class is None:
            available = ", ".join(cls._adapters.keys())
            raise ValueError(
                f"Unknown adapter: '{name}'. Available adapters: {available}"
            )

        # Build kwargs based on what the adapter accepts
        init_kwargs = {
            "endpoint": endpoint,
            "timeout": timeout,
        }

        # Add optional common parameters
        if headers is not None:
            init_kwargs["headers"] = headers
        if verbose:
            init_kwargs["verbose"] = verbose
        if model_config is not None:
            init_kwargs["model_config"] = model_config
        if allow_private_urls:
            init_kwargs["allow_private_urls"] = allow_private_urls
        if allowed_hosts is not None:
            init_kwargs["allowed_hosts"] = allowed_hosts

        # Add any additional kwargs
        init_kwargs.update(kwargs)

        # Handle adapters that don't take endpoint (like OpenAI Assistants)
        try:
            return adapter_class(**init_kwargs)
        except TypeError as e:
            # Remove endpoint if not accepted
            if "endpoint" in str(e):
                del init_kwargs["endpoint"]
                return adapter_class(**init_kwargs)
            raise

    @classmethod
    def _ensure_initialized(cls) -> None:
        """Ensure built-in adapters are registered."""
        if cls._initialized:
            return

        # Import and register built-in adapters
        try:
            from evalview.adapters.http_adapter import HTTPAdapter

            cls.register("http", HTTPAdapter)
        except ImportError:
            logger.warning("HTTPAdapter not available")

        try:
            from evalview.adapters.langgraph_adapter import LangGraphAdapter

            cls.register("langgraph", LangGraphAdapter)
        except ImportError:
            logger.warning("LangGraphAdapter not available")

        try:
            from evalview.adapters.crewai_adapter import CrewAIAdapter

            cls.register("crewai", CrewAIAdapter)
        except ImportError:
            logger.warning("CrewAIAdapter not available")

        try:
            from evalview.adapters.openai_assistants_adapter import OpenAIAssistantsAdapter

            cls.register("openai-assistants", OpenAIAssistantsAdapter)
        except ImportError:
            logger.warning("OpenAIAssistantsAdapter not available")

        try:
            from evalview.adapters.tapescope_adapter import TapeScopeAdapter

            cls.register("tapescope", TapeScopeAdapter)
            cls.register("streaming", TapeScopeAdapter)  # Alias
            cls.register("jsonl", TapeScopeAdapter)  # Alias
        except ImportError:
            logger.warning("TapeScopeAdapter not available")

        try:
            from evalview.adapters.anthropic_adapter import AnthropicAdapter

            cls.register("anthropic", AnthropicAdapter)
            cls.register("claude", AnthropicAdapter)  # Alias
        except ImportError:
            logger.warning("AnthropicAdapter not available")

        try:
            from evalview.adapters.huggingface_adapter import HuggingFaceAdapter

            cls.register("huggingface", HuggingFaceAdapter)
            cls.register("hf", HuggingFaceAdapter)  # Alias
            cls.register("gradio", HuggingFaceAdapter)  # Alias
        except ImportError:
            logger.warning("HuggingFaceAdapter not available")

        try:
            from evalview.adapters.goose_adapter import GooseAdapter

            cls.register("goose", GooseAdapter)
        except ImportError:
            logger.warning("GooseAdapter not available")

        cls._initialized = True

    @classmethod
    def reset(cls) -> None:
        """
        Reset the registry (useful for testing).

        Clears all registrations and resets initialization flag.
        """
        cls._adapters.clear()
        cls._initialized = False


def get_adapter(
    name: str,
    endpoint: str,
    **kwargs: Any,
) -> AgentAdapter:
    """
    Convenience function to create an adapter.

    Args:
        name: Adapter type (http, langgraph, crewai, etc.)
        endpoint: API endpoint URL
        **kwargs: Additional adapter configuration

    Returns:
        Configured adapter instance

    Example:
        >>> adapter = get_adapter("langgraph", "http://localhost:2024")
    """
    return AdapterRegistry.create(name, endpoint, **kwargs)
