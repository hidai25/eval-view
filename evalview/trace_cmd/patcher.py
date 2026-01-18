"""SDK patcher for automatic instrumentation of OpenAI, Anthropic, and Ollama clients.

This module patches LLM SDKs to automatically capture call traces.
Designed to be imported early via PYTHONPATH/sitecustomize.

Patched methods:
    - openai.OpenAI.chat.completions.create
    - openai.AsyncOpenAI.chat.completions.create
    - anthropic.Anthropic.messages.create
    - anthropic.AsyncAnthropic.messages.create
    - ollama.chat / ollama.Client.chat
    - ollama.AsyncClient.chat

Safety:
    - Only patches specific methods, not entire modules
    - Prints what was patched for transparency
    - Fails gracefully on unknown SDK versions
    - Does not modify any request/response data
"""

from __future__ import annotations

import sys
import time
import functools
from typing import Any, Callable, List

__all__ = ["patch_sdks", "get_patched_sdks"]

# Track what we've patched
_patched_sdks: List[str] = []

# Pricing per 1M tokens (approximate, for common models)
PRICING = {
    # OpenAI
    "gpt-4o": {"input": 2.50, "output": 10.00},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-4-turbo": {"input": 10.00, "output": 30.00},
    "gpt-4": {"input": 30.00, "output": 60.00},
    "gpt-3.5-turbo": {"input": 0.50, "output": 1.50},
    "o1": {"input": 15.00, "output": 60.00},
    "o1-mini": {"input": 3.00, "output": 12.00},
    # Anthropic
    "claude-opus-4-5-20251101": {"input": 15.00, "output": 75.00},
    "claude-sonnet-4-5-20250929": {"input": 3.00, "output": 15.00},
    "claude-3-5-haiku": {"input": 0.80, "output": 4.00},
    "claude-3-opus": {"input": 15.00, "output": 75.00},
    "claude-3-sonnet": {"input": 3.00, "output": 15.00},
    "claude-3-haiku": {"input": 0.25, "output": 1.25},
    # Ollama (local, free)
    "llama3.2": {"input": 0.0, "output": 0.0},
    "llama3.1": {"input": 0.0, "output": 0.0},
    "llama3": {"input": 0.0, "output": 0.0},
    "mistral": {"input": 0.0, "output": 0.0},
    "codellama": {"input": 0.0, "output": 0.0},
    "phi3": {"input": 0.0, "output": 0.0},
    "gemma2": {"input": 0.0, "output": 0.0},
    "qwen2": {"input": 0.0, "output": 0.0},
}


def _estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """Estimate cost for a model call."""
    # Try exact match first
    if model in PRICING:
        pricing = PRICING[model]
    else:
        # Try prefix match (e.g., "gpt-4o-2024-08-06" -> "gpt-4o")
        pricing = None
        for key in PRICING:
            if model.startswith(key):
                pricing = PRICING[key]
                break
        if pricing is None:
            return 0.0

    input_cost = (input_tokens / 1_000_000) * pricing["input"]
    output_cost = (output_tokens / 1_000_000) * pricing["output"]
    return input_cost + output_cost


def _patch_openai_sync(original_create: Callable) -> Callable:
    """Wrap OpenAI sync chat.completions.create."""
    @functools.wraps(original_create)
    def wrapped(*args: Any, **kwargs: Any) -> Any:
        from evalview.trace_cmd.collector import get_collector

        collector = get_collector()
        start_time = time.time()
        error_msg = None
        response = None

        try:
            response = original_create(*args, **kwargs)
            return response
        except Exception as e:
            error_msg = str(e)
            raise
        finally:
            if collector:
                duration_ms = (time.time() - start_time) * 1000
                model = kwargs.get("model", "unknown")

                # Extract usage if available
                input_tokens = 0
                output_tokens = 0
                finish_reason = None

                if error_msg is None and response and hasattr(response, "usage") and response.usage:
                    input_tokens = response.usage.prompt_tokens or 0
                    output_tokens = response.usage.completion_tokens or 0

                if error_msg is None and response and hasattr(response, "choices") and response.choices:
                    finish_reason = response.choices[0].finish_reason

                cost = _estimate_cost(model, input_tokens, output_tokens)

                collector.record_llm_call(
                    provider="openai",
                    model=model,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    duration_ms=duration_ms,
                    cost=cost,
                    finish_reason=finish_reason,
                    error=error_msg,
                )

    return wrapped


def _patch_openai_async(original_create: Callable) -> Callable:
    """Wrap OpenAI async chat.completions.create."""
    @functools.wraps(original_create)
    async def wrapped(*args: Any, **kwargs: Any) -> Any:
        from evalview.trace_cmd.collector import get_collector

        collector = get_collector()
        start_time = time.time()
        error_msg = None
        response = None

        try:
            response = await original_create(*args, **kwargs)
            return response
        except Exception as e:
            error_msg = str(e)
            raise
        finally:
            if collector:
                duration_ms = (time.time() - start_time) * 1000
                model = kwargs.get("model", "unknown")

                input_tokens = 0
                output_tokens = 0
                finish_reason = None

                if error_msg is None and response and hasattr(response, "usage") and response.usage:
                    input_tokens = response.usage.prompt_tokens or 0
                    output_tokens = response.usage.completion_tokens or 0

                if error_msg is None and response and hasattr(response, "choices") and response.choices:
                    finish_reason = response.choices[0].finish_reason

                cost = _estimate_cost(model, input_tokens, output_tokens)

                collector.record_llm_call(
                    provider="openai",
                    model=model,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    duration_ms=duration_ms,
                    cost=cost,
                    finish_reason=finish_reason,
                    error=error_msg,
                )

    return wrapped


def _patch_anthropic_sync(original_create: Callable) -> Callable:
    """Wrap Anthropic sync messages.create."""
    @functools.wraps(original_create)
    def wrapped(*args: Any, **kwargs: Any) -> Any:
        from evalview.trace_cmd.collector import get_collector

        collector = get_collector()
        start_time = time.time()
        error_msg = None
        response = None

        try:
            response = original_create(*args, **kwargs)
            return response
        except Exception as e:
            error_msg = str(e)
            raise
        finally:
            if collector:
                duration_ms = (time.time() - start_time) * 1000
                model = kwargs.get("model", "unknown")

                input_tokens = 0
                output_tokens = 0
                finish_reason = None

                if error_msg is None and response and hasattr(response, "usage"):
                    input_tokens = response.usage.input_tokens or 0
                    output_tokens = response.usage.output_tokens or 0

                if error_msg is None and response and hasattr(response, "stop_reason"):
                    finish_reason = response.stop_reason

                cost = _estimate_cost(model, input_tokens, output_tokens)

                collector.record_llm_call(
                    provider="anthropic",
                    model=model,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    duration_ms=duration_ms,
                    cost=cost,
                    finish_reason=finish_reason,
                    error=error_msg,
                )

    return wrapped


def _patch_anthropic_async(original_create: Callable) -> Callable:
    """Wrap Anthropic async messages.create."""
    @functools.wraps(original_create)
    async def wrapped(*args: Any, **kwargs: Any) -> Any:
        from evalview.trace_cmd.collector import get_collector

        collector = get_collector()
        start_time = time.time()
        error_msg = None
        response = None

        try:
            response = await original_create(*args, **kwargs)
            return response
        except Exception as e:
            error_msg = str(e)
            raise
        finally:
            if collector:
                duration_ms = (time.time() - start_time) * 1000
                model = kwargs.get("model", "unknown")

                input_tokens = 0
                output_tokens = 0
                finish_reason = None

                if error_msg is None and response and hasattr(response, "usage"):
                    input_tokens = response.usage.input_tokens or 0
                    output_tokens = response.usage.output_tokens or 0

                if error_msg is None and response and hasattr(response, "stop_reason"):
                    finish_reason = response.stop_reason

                cost = _estimate_cost(model, input_tokens, output_tokens)

                collector.record_llm_call(
                    provider="anthropic",
                    model=model,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    duration_ms=duration_ms,
                    cost=cost,
                    finish_reason=finish_reason,
                    error=error_msg,
                )

    return wrapped


def _patch_ollama_sync(original_chat: Callable) -> Callable:
    """Wrap Ollama sync chat function."""
    @functools.wraps(original_chat)
    def wrapped(*args: Any, **kwargs: Any) -> Any:
        from evalview.trace_cmd.collector import get_collector

        collector = get_collector()
        start_time = time.time()
        error_msg = None
        response = None

        try:
            response = original_chat(*args, **kwargs)
            return response
        except Exception as e:
            error_msg = str(e)
            raise
        finally:
            if collector:
                duration_ms = (time.time() - start_time) * 1000
                model = kwargs.get("model", "unknown")

                # Ollama returns a dict with message and optional usage info
                input_tokens = 0
                output_tokens = 0
                finish_reason = None

                if error_msg is None and response:
                    if isinstance(response, dict):
                        input_tokens = response.get("prompt_eval_count", 0)
                        output_tokens = response.get("eval_count", 0)
                        finish_reason = response.get("done_reason")
                    elif hasattr(response, "prompt_eval_count"):
                        input_tokens = response.prompt_eval_count or 0
                        output_tokens = response.eval_count or 0
                        finish_reason = getattr(response, "done_reason", None)

                cost = _estimate_cost(model, input_tokens, output_tokens)

                collector.record_llm_call(
                    provider="ollama",
                    model=model,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    duration_ms=duration_ms,
                    cost=cost,
                    finish_reason=finish_reason,
                    error=error_msg,
                )

    return wrapped


def _patch_ollama_async(original_chat: Callable) -> Callable:
    """Wrap Ollama async chat function."""
    @functools.wraps(original_chat)
    async def wrapped(*args: Any, **kwargs: Any) -> Any:
        from evalview.trace_cmd.collector import get_collector

        collector = get_collector()
        start_time = time.time()
        error_msg = None
        response = None

        try:
            response = await original_chat(*args, **kwargs)
            return response
        except Exception as e:
            error_msg = str(e)
            raise
        finally:
            if collector:
                duration_ms = (time.time() - start_time) * 1000
                model = kwargs.get("model", "unknown")

                input_tokens = 0
                output_tokens = 0
                finish_reason = None

                if error_msg is None and response:
                    if isinstance(response, dict):
                        input_tokens = response.get("prompt_eval_count", 0)
                        output_tokens = response.get("eval_count", 0)
                        finish_reason = response.get("done_reason")
                    elif hasattr(response, "prompt_eval_count"):
                        input_tokens = response.prompt_eval_count or 0
                        output_tokens = response.eval_count or 0
                        finish_reason = getattr(response, "done_reason", None)

                cost = _estimate_cost(model, input_tokens, output_tokens)

                collector.record_llm_call(
                    provider="ollama",
                    model=model,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    duration_ms=duration_ms,
                    cost=cost,
                    finish_reason=finish_reason,
                    error=error_msg,
                )

    return wrapped


def patch_ollama() -> bool:
    """Patch Ollama SDK if available.

    Returns:
        True if patched successfully, False otherwise
    """
    try:
        import ollama

        # Patch module-level chat function
        if hasattr(ollama, "chat"):
            original_chat = ollama.chat
            ollama.chat = _patch_ollama_sync(original_chat)

        # Patch Client.chat
        if hasattr(ollama, "Client"):
            original_client_chat = ollama.Client.chat
            ollama.Client.chat = _patch_ollama_sync(original_client_chat)

        # Patch AsyncClient.chat
        if hasattr(ollama, "AsyncClient"):
            original_async_chat = ollama.AsyncClient.chat
            ollama.AsyncClient.chat = _patch_ollama_async(original_async_chat)

        _patched_sdks.append("ollama")
        return True

    except ImportError:
        return False
    except Exception as e:
        print(f"[evalview] Warning: Failed to patch ollama: {e}", file=sys.stderr)
        return False


def patch_openai() -> bool:
    """Patch OpenAI SDK if available.

    Returns:
        True if patched successfully, False otherwise
    """
    try:
        import openai

        # Patch sync client
        if hasattr(openai, "OpenAI"):
            original_sync = openai.resources.chat.completions.Completions.create
            openai.resources.chat.completions.Completions.create = _patch_openai_sync(original_sync)

        # Patch async client
        if hasattr(openai, "AsyncOpenAI"):
            original_async = openai.resources.chat.completions.AsyncCompletions.create
            openai.resources.chat.completions.AsyncCompletions.create = _patch_openai_async(original_async)

        _patched_sdks.append("openai")
        return True

    except ImportError:
        return False
    except Exception as e:
        print(f"[evalview] Warning: Failed to patch openai: {e}", file=sys.stderr)
        return False


def patch_anthropic() -> bool:
    """Patch Anthropic SDK if available.

    Returns:
        True if patched successfully, False otherwise
    """
    try:
        import anthropic

        # Patch sync client
        if hasattr(anthropic, "Anthropic"):
            original_sync = anthropic.resources.messages.Messages.create
            anthropic.resources.messages.Messages.create = _patch_anthropic_sync(original_sync)

        # Patch async client
        if hasattr(anthropic, "AsyncAnthropic"):
            original_async = anthropic.resources.messages.AsyncMessages.create
            anthropic.resources.messages.AsyncMessages.create = _patch_anthropic_async(original_async)

        _patched_sdks.append("anthropic")
        return True

    except ImportError:
        return False
    except Exception as e:
        print(f"[evalview] Warning: Failed to patch anthropic: {e}", file=sys.stderr)
        return False


def patch_sdks() -> List[str]:
    """Patch all available SDKs.

    Returns:
        List of successfully patched SDK names
    """
    patch_openai()
    patch_anthropic()
    patch_ollama()
    return _patched_sdks.copy()


def get_patched_sdks() -> List[str]:
    """Get list of SDKs that have been patched."""
    return _patched_sdks.copy()
