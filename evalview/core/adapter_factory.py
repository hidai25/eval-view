"""Shared adapter factory helpers used by CLI and programmatic runners."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from evalview.adapters.base import AgentAdapter
    from evalview.core.config import EvalViewConfig


def create_adapter(
    adapter_type: str,
    endpoint: str,
    timeout: float = 30.0,
    allow_private_urls: bool = True,
) -> "AgentAdapter":
    """Create an agent adapter from a normalized adapter type."""
    if adapter_type == "cohere":
        from evalview.adapters.cohere_adapter import CohereAdapter

        return CohereAdapter()

    if adapter_type == "mistral":
        from evalview.adapters.mistral_adapter import MistralAdapter

        return MistralAdapter()

    if adapter_type == "anthropic":
        # For Anthropic, `endpoint` is interpreted as the model ID since the
        # Anthropic SDK talks to a fixed API host. Falls back to the adapter's
        # default model when no endpoint is supplied.
        from evalview.adapters.anthropic_adapter import AnthropicAdapter

        if endpoint:
            return AnthropicAdapter(model=endpoint, timeout=timeout)
        return AnthropicAdapter(timeout=timeout)

    if adapter_type == "opencode":
        from evalview.adapters.opencode_adapter import OpenCodeAdapter

        return OpenCodeAdapter(timeout=timeout)

    from evalview.adapters.crewai_adapter import CrewAIAdapter
    from evalview.adapters.http_adapter import HTTPAdapter
    from evalview.adapters.langgraph_adapter import LangGraphAdapter
    from evalview.adapters.openai_assistants_adapter import OpenAIAssistantsAdapter
    from evalview.adapters.tapescope_adapter import TapeScopeAdapter

    adapter_map = {
        "http": HTTPAdapter,
        "langgraph": LangGraphAdapter,
        "tapescope": TapeScopeAdapter,
        "crewai": CrewAIAdapter,
        "openai": OpenAIAssistantsAdapter,
        "openai-assistants": OpenAIAssistantsAdapter,
    }

    adapter_class = adapter_map.get(adapter_type)
    if not adapter_class:
        raise ValueError(
            f"Unknown adapter type: '{adapter_type}'. "
            f"Supported: {', '.join(sorted(adapter_map.keys()))}, "
            f"anthropic, cohere, mistral"
        )

    if adapter_type == "http":
        return adapter_class(
            endpoint=endpoint,
            timeout=timeout,
            allow_private_urls=allow_private_urls,
        )

    return adapter_class(endpoint=endpoint, timeout=timeout)


def create_adapter_from_config(config: "EvalViewConfig") -> "AgentAdapter":
    """Create an adapter from EvalViewConfig."""
    return create_adapter(
        adapter_type=config.adapter,
        endpoint=config.endpoint,
        timeout=getattr(config, "timeout", 30.0),
        allow_private_urls=getattr(config, "allow_private_urls", True),
    )
