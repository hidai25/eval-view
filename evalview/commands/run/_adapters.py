"""Adapter factory for the run command.

Provides a single `build_adapter()` function that constructs any supported
AgentAdapter from a type string + configuration dict, plus `get_test_adapter()`
which resolves the right adapter for a specific test case (falling back to the
globally-configured adapter when the test has no adapter override).
"""
from __future__ import annotations

import os
import logging
from typing import Any, Dict, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from evalview.adapters.base import AgentAdapter

logger = logging.getLogger(__name__)


def build_adapter(
    adapter_type: str,
    endpoint: Optional[str],
    cfg: Dict[str, Any],
    model_config: Any,
    verbose: bool,
    allow_private_urls: bool,
) -> "AgentAdapter":
    """Construct an AgentAdapter from adapter type + configuration dict.

    Args:
        adapter_type: One of the supported adapter type strings.
        endpoint: Override endpoint (takes precedence over cfg["endpoint"]).
        cfg: Adapter configuration dict (from config.yaml or test-case adapter_config).
        model_config: Model configuration (str or dict).
        verbose: Whether to enable verbose adapter logging.
        allow_private_urls: SSRF guard — set False in production environments.

    Returns:
        A fully-configured AgentAdapter instance.

    Raises:
        ValueError: When required config keys are missing for the adapter type.
        ImportError: When an optional adapter dependency is not installed.
    """
    from evalview.adapters.http_adapter import HTTPAdapter
    from evalview.adapters.tapescope_adapter import TapeScopeAdapter
    from evalview.adapters.langgraph_adapter import LangGraphAdapter
    from evalview.adapters.crewai_adapter import CrewAIAdapter
    from evalview.adapters.openai_assistants_adapter import OpenAIAssistantsAdapter

    resolved_endpoint = endpoint or cfg.get("endpoint", "")

    if adapter_type == "langgraph":
        return LangGraphAdapter(
            endpoint=resolved_endpoint,
            headers=cfg.get("headers", {}),
            timeout=cfg.get("timeout", 30.0),
            streaming=cfg.get("streaming", False),
            verbose=verbose,
            model_config=model_config,
            assistant_id=cfg.get("assistant_id", "agent"),
            allow_private_urls=allow_private_urls,
        )

    if adapter_type == "crewai":
        return CrewAIAdapter(
            endpoint=resolved_endpoint,
            headers=cfg.get("headers", {}),
            timeout=cfg.get("timeout", 120.0),
            verbose=verbose,
            model_config=model_config,
            allow_private_urls=allow_private_urls,
        )

    if adapter_type == "openai-assistants":
        return OpenAIAssistantsAdapter(
            assistant_id=cfg.get("assistant_id"),
            timeout=cfg.get("timeout", 120.0),
            verbose=verbose,
            model_config=model_config,
        )

    if adapter_type in ("streaming", "tapescope", "jsonl"):
        return TapeScopeAdapter(
            endpoint=resolved_endpoint,
            headers=cfg.get("headers", {}),
            timeout=cfg.get("timeout", 60.0),
            verbose=verbose,
            model_config=model_config,
            allow_private_urls=allow_private_urls,
        )

    if adapter_type == "anthropic":
        if not os.getenv("ANTHROPIC_API_KEY"):
            raise ValueError(
                "ANTHROPIC_API_KEY not found. Set it in your .env.local file or "
                "export ANTHROPIC_API_KEY=sk-ant-..."
            )
        from evalview.adapters.anthropic_adapter import AnthropicAdapter

        anthropic_model = cfg.get("model", "claude-sonnet-4-5-20250929")
        if isinstance(anthropic_model, dict):
            anthropic_model = anthropic_model.get("name", "claude-sonnet-4-5-20250929")

        return AnthropicAdapter(
            model=anthropic_model,
            tools=cfg.get("tools", []),
            system_prompt=cfg.get("system_prompt"),
            max_tokens=cfg.get("max_tokens", 4096),
            timeout=cfg.get("timeout", 120.0),
            verbose=verbose,
        )

    if adapter_type in ("huggingface", "hf", "gradio"):
        from evalview.adapters.huggingface_adapter import HuggingFaceAdapter

        return HuggingFaceAdapter(
            endpoint=resolved_endpoint,
            headers=cfg.get("headers", {}),
            timeout=cfg.get("timeout", 120.0),
            hf_token=os.getenv("HF_TOKEN"),
            function_name=cfg.get("function_name"),
            verbose=verbose,
            model_config=model_config,
            allow_private_urls=allow_private_urls,
        )

    if adapter_type == "ollama":
        from evalview.adapters.ollama_adapter import OllamaAdapter

        ollama_model = cfg.get("model", "llama3.2")
        if isinstance(ollama_model, dict):
            ollama_model = ollama_model.get("name", "llama3.2")

        return OllamaAdapter(
            model=ollama_model,
            endpoint=cfg.get("endpoint", "http://localhost:11434"),
            timeout=cfg.get("timeout", 60.0),
            verbose=verbose,
            model_config=model_config,
        )

    if adapter_type == "goose":
        from evalview.adapters.goose_adapter import GooseAdapter

        return GooseAdapter(
            timeout=cfg.get("timeout", 300.0),
            cwd=cfg.get("cwd"),
            extensions=cfg.get("extensions", ["developer"]),
            provider=cfg.get("provider"),
            model=cfg.get("goose_model") or cfg.get("model"),
        )

    if adapter_type == "mcp":
        from evalview.adapters.mcp_adapter import MCPAdapter

        return MCPAdapter(
            endpoint=resolved_endpoint,
            timeout=cfg.get("timeout", 30.0),
        )

    if adapter_type == "cohere":
        from evalview.adapters.cohere_adapter import CohereAdapter

        cohere_model = model_config.get("name") if isinstance(model_config, dict) else model_config
        return CohereAdapter(model=cohere_model)

    # Default: generic HTTP adapter
    return HTTPAdapter(
        endpoint=resolved_endpoint,
        headers=cfg.get("headers", {}),
        timeout=cfg.get("timeout", 30.0),
        model_config=model_config,
        allow_private_urls=allow_private_urls,
    )


def get_test_adapter(
    test_case: Any,
    global_adapter: Optional["AgentAdapter"],
    model_config: Any,
    allow_private_urls: bool,
    verbose: bool,
    console: Any,
) -> "AgentAdapter":
    """Return the adapter for a specific test case.

    If the test case specifies its own adapter + endpoint (or is a no-endpoint
    adapter like openai-assistants / goose), a test-specific adapter is built.
    Otherwise the globally-configured adapter is returned.

    Raises:
        ValueError: When neither a test-specific nor global adapter is available.
    """
    API_ONLY_ADAPTERS = {"openai-assistants", "goose"}

    test_adapter_type = test_case.adapter
    test_endpoint = test_case.endpoint
    has_test_adapter = test_adapter_type and (test_endpoint or test_adapter_type in API_ONLY_ADAPTERS)

    if has_test_adapter:
        test_cfg: Dict[str, Any] = test_case.adapter_config or {}

        if verbose:
            console.print(
                f"[dim]  Using test-specific adapter: {test_adapter_type} @ {test_endpoint}[/dim]"
            )

        # crewai merges global model_config with test-level config
        if test_adapter_type == "crewai":
            merged_model = {**(model_config if isinstance(model_config, dict) else {}), **test_cfg}
            return build_adapter(test_adapter_type, test_endpoint, test_cfg, merged_model, verbose, allow_private_urls)

        # goose pulls cwd/extensions from the test's input context
        if test_adapter_type == "goose":
            from evalview.adapters.goose_adapter import GooseAdapter

            ctx = test_case.input.context or {}
            return GooseAdapter(
                timeout=test_cfg.get("timeout", 300.0),
                cwd=ctx.get("cwd"),
                extensions=ctx.get("extensions"),
                provider=test_cfg.get("provider"),
                model=test_cfg.get("model"),
            )

        return build_adapter(test_adapter_type, test_endpoint, test_cfg, model_config, verbose, allow_private_urls)

    # Fall back to global adapter
    if global_adapter is None:
        console.print(f"[red]❌ No adapter configured for test: {test_case.name}[/red]")
        console.print("[dim]Either add adapter/endpoint to the test case YAML, or create .evalview/config.yaml[/dim]")
        console.print("[dim]Example in test case:[/dim]")
        console.print("[dim]  adapter: http[/dim]")
        console.print("[dim]  endpoint: http://localhost:8000[/dim]")
        raise ValueError(f"No adapter for test: {test_case.name}")

    return global_adapter
