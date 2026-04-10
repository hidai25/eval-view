"""Provider call helper for `evalview model-check`.

The agentic adapter abstraction in ``evalview/adapters`` is built around
tool loops, golden traces, and ExecutionMetrics — none of which the
canary needs. The canary asks one question: "send this prompt to a
specific model with pinned sampling and give me back the text and the
token counts." This module is the smallest thing that does exactly that.

Each provider gets one branch. Adding a new provider is roughly:
1. Add a new ``_run_<provider>`` async function
2. Wire it into ``run_completion`` via the ``_RUNNERS`` dispatch
3. Add the provider key to ``SUPPORTED_PROVIDERS``

The helper is **completely independent** of the agent adapter layer.
That isolation matters: changes here cannot break the agent test path,
and changes to agent adapters cannot perturb the canary's stable signal.
"""
from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from typing import Awaitable, Callable, Dict, Optional

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Public types
# --------------------------------------------------------------------------- #


@dataclass
class CompletionResult:
    """One completion call's result, normalised across providers."""

    text: str
    input_tokens: int
    output_tokens: int
    latency_ms: float
    # Provider-supplied fingerprint, when available. OpenAI exposes a
    # ``system_fingerprint`` per call (strong drift signal); Anthropic
    # currently exposes only the requested model id (weaker, behavior-only).
    fingerprint: Optional[str]
    # Confidence band for the fingerprint signal. Surfaced in CLI output
    # so users know whether they're looking at ground truth or behavioral
    # inference. One of: "strong" | "weak" | "none".
    fingerprint_confidence: str


class ProviderError(RuntimeError):
    """Raised when a provider call fails for any reason.

    The CLI catches this and surfaces the message to the user. We use a
    single exception type so the dispatch loop has one ``except`` clause
    and so callers cannot accidentally swallow provider-specific errors.
    """


# --------------------------------------------------------------------------- #
# Anthropic
# --------------------------------------------------------------------------- #


# Cached client instance — reused across concurrent calls within one
# asyncio.run() to share the underlying httpx connection pool.
_anthropic_client: Optional["AsyncAnthropic"] = None  # type: ignore[name-defined]
_anthropic_timeout: Optional[float] = None


def _get_anthropic_client(timeout: float) -> "AsyncAnthropic":  # type: ignore[name-defined]
    """Return a shared AsyncAnthropic client, creating it on first use.

    If the timeout changes between calls (unlikely in practice — the
    canary pins it), a new client is created. This is cheaper than
    creating one per call but still respects config changes.
    """
    global _anthropic_client, _anthropic_timeout
    try:
        from anthropic import AsyncAnthropic
    except ImportError as exc:
        raise ProviderError(
            "anthropic package not installed. Run: pip install anthropic"
        ) from exc

    if _anthropic_client is None or _anthropic_timeout != timeout:
        _anthropic_client = AsyncAnthropic(timeout=timeout)
        _anthropic_timeout = timeout
    return _anthropic_client


async def _run_anthropic(
    model: str,
    prompt: str,
    *,
    temperature: float,
    top_p: float,
    max_tokens: int,
    timeout: float,
) -> CompletionResult:
    """Plain-text completion against the Anthropic Messages API."""
    try:
        from anthropic._exceptions import AnthropicError  # type: ignore
    except ImportError as exc:
        raise ProviderError(
            "anthropic package not installed. Run: pip install anthropic"
        ) from exc

    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise ProviderError(
            "ANTHROPIC_API_KEY environment variable is not set."
        )

    # NOTE on prompt caching: Anthropic requires a minimum of 1024 tokens
    # per cacheable block. Canary prompts are 15-73 tokens each — well
    # below the threshold. Adding a large system prompt just to enable
    # caching would change model behavior and invalidate existing snapshots.
    # If the minimum drops in the future or custom suites have larger
    # prompts, add cache_control={"type": "ephemeral"} to the content block.
    client = _get_anthropic_client(timeout)
    started = time.perf_counter()
    try:
        response = await client.messages.create(
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[{"role": "user", "content": prompt}],
        )
    except AnthropicError as exc:  # pragma: no cover - exercised manually
        raise ProviderError(f"Anthropic API error: {exc}") from exc
    latency_ms = (time.perf_counter() - started) * 1000.0

    # The Anthropic SDK returns content as a list of blocks; for plain-text
    # prompts (no tool use) we expect exactly one TextBlock. Defensive:
    # join any text blocks to handle SDK variations gracefully.
    text_parts = []
    for block in response.content:
        text = getattr(block, "text", None)
        if text:
            text_parts.append(text)
    text = "".join(text_parts)

    usage = response.usage
    return CompletionResult(
        text=text,
        input_tokens=getattr(usage, "input_tokens", 0) or 0,
        output_tokens=getattr(usage, "output_tokens", 0) or 0,
        latency_ms=latency_ms,
        # Anthropic does not currently expose a per-response fingerprint.
        # We record the requested model id so the snapshot still has *some*
        # signal — the CLI labels this as "weak — behavior-only".
        fingerprint=getattr(response, "model", None) or model,
        fingerprint_confidence="weak",
    )


# --------------------------------------------------------------------------- #
# Dispatch
# --------------------------------------------------------------------------- #


_RunnerFunc = Callable[..., Awaitable[CompletionResult]]

_RUNNERS: Dict[str, _RunnerFunc] = {
    "anthropic": _run_anthropic,
}

SUPPORTED_PROVIDERS = tuple(sorted(_RUNNERS))


def detect_provider(model: str) -> Optional[str]:
    """Best-effort provider inference from a model id.

    Returns None if the model id doesn't match a known prefix; the CLI
    then asks the user to pass ``--provider`` explicitly. We deliberately
    keep this list short and unambiguous — fuzzy matching here would
    silently route the wrong adapter at the wrong time.
    """
    lowered = model.lower()
    if lowered.startswith("claude"):
        return "anthropic"
    return None


async def run_completion(
    provider: str,
    model: str,
    prompt: str,
    *,
    temperature: float = 0.0,
    top_p: float = 1.0,
    max_tokens: int = 1024,
    timeout: float = 60.0,
) -> CompletionResult:
    """Send a single prompt to the given closed model.

    Always returns a CompletionResult or raises ProviderError. Never
    returns None and never silently substitutes a default — silent
    substitution would corrupt drift signals.
    """
    runner = _RUNNERS.get(provider)
    if runner is None:
        raise ProviderError(
            f"Unknown provider '{provider}'. Supported in v1: "
            f"{', '.join(SUPPORTED_PROVIDERS)}"
        )
    return await runner(
        model,
        prompt,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        timeout=timeout,
    )


__all__ = [
    "CompletionResult",
    "ProviderError",
    "SUPPORTED_PROVIDERS",
    "detect_provider",
    "run_completion",
]
