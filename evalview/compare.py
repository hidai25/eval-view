"""Model comparison API for EvalView.

Run the same query against multiple models side-by-side and score the results.
Designed for pytest parametrize workflows::

    import evalview
    import pytest

    @pytest.mark.parametrize("model", ["claude-opus-4-6", "gpt-4o", "claude-sonnet-4-6"])
    def test_summarize(model):
        result = evalview.run_eval(model, query="Summarize: AI is transforming software.")
        assert evalview.score(result) > 0.8

    def test_compare_all():
        results = evalview.compare_models(
            query="Summarize: AI is transforming software.",
            models=["claude-opus-4-6", "gpt-4o", "claude-sonnet-4-6"],
        )
        best = results[0]
        evalview.print_comparison_table(results)
"""
from __future__ import annotations

import asyncio
import difflib
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from evalview.core.pricing import calculate_cost


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

@dataclass
class ModelResult:
    """Result from a single model evaluation.

    Attributes:
        model: The model name/ID used (e.g., ``"claude-opus-4-6"``).
        query: The input query sent to the model.
        output: The model's text response.
        score: Quality score in [0.0, 1.0]. 1.0 = best.
        latency_ms: Wall-clock time for the API call.
        cost_usd: Estimated cost in USD.
        passed: True when ``score >= threshold``.
        error: Error message if the call failed, otherwise None.
        metadata: Provider-specific extras (tokens, stop_reason, etc.).
    """

    model: str
    query: str
    output: str
    score: float
    latency_ms: float
    cost_usd: float
    passed: bool
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        status = "✓" if self.passed else "✗"
        return (
            f"{status} {self.model}: score={self.score:.2f} "
            f"latency={self.latency_ms:.0f}ms cost=${self.cost_usd:.5f}"
        )


# ---------------------------------------------------------------------------
# Provider detection
# ---------------------------------------------------------------------------

def _detect_provider(model: str) -> str:
    """Return 'anthropic', 'openai', or 'unknown' based on model name."""
    m = model.lower()
    if m.startswith("claude"):
        return "anthropic"
    if m.startswith("gpt") or m.startswith("o1") or m.startswith("o3") or m.startswith("o4"):
        return "openai"
    if m.startswith("gemini"):
        return "google"
    if m.startswith("grok"):
        return "xai"
    if m.startswith("deepseek"):
        return "deepseek"
    return "openai"  # fallback — most providers offer openai-compatible APIs


# ---------------------------------------------------------------------------
# Async execution helpers
# ---------------------------------------------------------------------------

async def _call_anthropic(
    model: str,
    query: str,
    system_prompt: Optional[str],
    max_tokens: int,
    timeout: float,
) -> Dict[str, Any]:
    try:
        from anthropic import AsyncAnthropic
    except ImportError:
        raise ImportError("Install anthropic: pip install anthropic")

    client = AsyncAnthropic()
    kwargs: Dict[str, Any] = {
        "model": model,
        "max_tokens": max_tokens,
        "messages": [{"role": "user", "content": query}],
    }
    if system_prompt:
        kwargs["system"] = system_prompt

    t0 = time.perf_counter()
    response = await asyncio.wait_for(client.messages.create(**kwargs), timeout=timeout)
    latency_ms = (time.perf_counter() - t0) * 1000

    output = "".join(b.text for b in response.content if b.type == "text")
    input_tokens = response.usage.input_tokens if hasattr(response, "usage") else 0
    output_tokens = response.usage.output_tokens if hasattr(response, "usage") else 0
    cost = calculate_cost(model, input_tokens, output_tokens)

    return {
        "output": output,
        "latency_ms": latency_ms,
        "cost_usd": cost,
        "metadata": {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "stop_reason": getattr(response, "stop_reason", None),
            "resolved_model": getattr(response, "model", model),
        },
    }


async def _call_openai(
    model: str,
    query: str,
    system_prompt: Optional[str],
    max_tokens: int,
    timeout: float,
) -> Dict[str, Any]:
    try:
        from openai import AsyncOpenAI
    except ImportError:
        raise ImportError("Install openai: pip install openai")

    client = AsyncOpenAI()
    messages: List[Any] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": query})

    t0 = time.perf_counter()
    response = await asyncio.wait_for(
        client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
        ),
        timeout=timeout,
    )
    latency_ms = (time.perf_counter() - t0) * 1000

    output = response.choices[0].message.content or ""
    usage = response.usage
    input_tokens = usage.prompt_tokens if usage else 0
    output_tokens = usage.completion_tokens if usage else 0
    cost = calculate_cost(model, input_tokens, output_tokens)

    return {
        "output": output,
        "latency_ms": latency_ms,
        "cost_usd": cost,
        "metadata": {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "finish_reason": response.choices[0].finish_reason,
        },
    }


async def _call_model(
    model: str,
    query: str,
    system_prompt: Optional[str],
    max_tokens: int,
    timeout: float,
) -> Dict[str, Any]:
    provider = _detect_provider(model)
    if provider == "anthropic":
        return await _call_anthropic(model, query, system_prompt, max_tokens, timeout)
    else:
        return await _call_openai(model, query, system_prompt, max_tokens, timeout)


# ---------------------------------------------------------------------------
# Scoring helpers
# ---------------------------------------------------------------------------

def _compute_similarity(output: str, expected: str) -> float:
    """Rough token-overlap similarity (0–1). No external deps needed."""
    if not expected:
        return 1.0
    # Normalize
    a = output.lower().split()
    b = expected.lower().split()
    if not a or not b:
        return 0.0
    return difflib.SequenceMatcher(None, a, b).ratio()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_eval(
    model: str,
    query: str,
    system_prompt: Optional[str] = None,
    expected: Optional[str] = None,
    scorer: Optional[Callable[[str, Optional[str]], float]] = None,
    threshold: float = 0.8,
    max_tokens: int = 1024,
    timeout: float = 60.0,
) -> ModelResult:
    """Call a model and return a scored :class:`ModelResult`.

    Works with Anthropic and OpenAI models out of the box. Provider is
    inferred from the model name — no config needed.

    Args:
        model: Model identifier, e.g. ``"claude-opus-4-6"`` or ``"gpt-4o"``.
        query: The user message / prompt to evaluate.
        system_prompt: Optional system prompt.
        expected: Reference output for scoring. When provided, score is
            computed as the token-overlap similarity with the actual output.
        scorer: Custom scoring function ``(output, expected) -> float``.
            If given, overrides the built-in similarity scorer.
            Should return a float in [0.0, 1.0].
        threshold: Minimum score for ``result.passed`` to be True.
            Default: 0.8.
        max_tokens: Maximum output tokens.
        timeout: Per-call timeout in seconds.

    Returns:
        :class:`ModelResult` with output, score, latency, cost.

    Example::

        import evalview
        result = evalview.run_eval("claude-opus-4-6", query="What is 2+2?")
        assert evalview.score(result) > 0.8

        # With expected output for automatic scoring:
        result = evalview.run_eval(
            "gpt-4o",
            query="Summarize in one sentence: The sky is blue.",
            expected="The sky appears blue",
        )
    """
    try:
        data = asyncio.run(_call_model(model, query, system_prompt, max_tokens, timeout))
    except Exception as exc:
        return ModelResult(
            model=model,
            query=query,
            output="",
            score=0.0,
            latency_ms=0.0,
            cost_usd=0.0,
            passed=False,
            error=str(exc),
        )

    output = data["output"]

    # Compute score
    if scorer is not None:
        score_val = float(scorer(output, expected))
    elif expected is not None:
        score_val = _compute_similarity(output, expected)
    else:
        # No expected / no scorer: 1.0 if we got a non-empty response
        score_val = 1.0 if output.strip() else 0.0

    return ModelResult(
        model=model,
        query=query,
        output=output,
        score=score_val,
        latency_ms=data["latency_ms"],
        cost_usd=data["cost_usd"],
        passed=score_val >= threshold,
        metadata=data.get("metadata", {}),
    )


def score(result: ModelResult) -> float:
    """Return the quality score for a :class:`ModelResult`.

    Convenience accessor so test assertions read naturally::

        assert evalview.score(result) > 0.8

    Args:
        result: A result returned by :func:`run_eval`.

    Returns:
        Float in [0.0, 1.0].
    """
    return result.score


def compare_models(
    query: str,
    models: List[str],
    system_prompt: Optional[str] = None,
    expected: Optional[str] = None,
    scorer: Optional[Callable[[str, Optional[str]], float]] = None,
    threshold: float = 0.8,
    max_tokens: int = 1024,
    timeout: float = 60.0,
    parallel: bool = True,
) -> List[ModelResult]:
    """Evaluate multiple models on the same query and rank by score.

    Runs models in parallel by default.  Returns results sorted by score
    descending (best first).

    Args:
        query: The prompt to evaluate across all models.
        models: List of model identifiers.
        system_prompt: Optional shared system prompt.
        expected: Reference output for automatic scoring.
        scorer: Custom scoring function ``(output, expected) -> float``.
        threshold: Pass/fail threshold applied to each result.
        max_tokens: Maximum output tokens per call.
        timeout: Per-call timeout in seconds.
        parallel: Run all models concurrently. Set False to run sequentially
            (useful for debugging or rate-limit-sensitive setups).

    Returns:
        List of :class:`ModelResult` sorted by score descending.

    Example::

        results = evalview.compare_models(
            query="Explain quantum entanglement in one sentence.",
            models=["claude-opus-4-6", "gpt-4o", "claude-sonnet-4-6"],
        )
        print(results[0])  # best scoring model
    """
    if parallel:
        results = asyncio.run(_compare_parallel(
            query, models, system_prompt, expected, scorer, threshold, max_tokens, timeout
        ))
    else:
        results = [
            run_eval(
                model=m,
                query=query,
                system_prompt=system_prompt,
                expected=expected,
                scorer=scorer,
                threshold=threshold,
                max_tokens=max_tokens,
                timeout=timeout,
            )
            for m in models
        ]

    results.sort(key=lambda r: r.score, reverse=True)

    # Auto-push to cloud if token is configured (best-effort, never raises)
    try:
        import os
        if os.environ.get("EVALVIEW_API_TOKEN") or _has_cloud_config():
            from evalview.cloud.push import push_comparison
            push_comparison(results, query=query, threshold=threshold)
    except Exception:
        pass

    return results


def _has_cloud_config() -> bool:
    """Return True if a cloud API token exists in the config file."""
    try:
        from evalview.commands.shared import _load_config_if_exists
        config = _load_config_if_exists()
        return bool(getattr(getattr(config, "cloud", None), "api_token", None))
    except Exception:
        return False


async def _compare_parallel(
    query: str,
    models: List[str],
    system_prompt: Optional[str],
    expected: Optional[str],
    scorer: Optional[Callable[[str, Optional[str]], float]],
    threshold: float,
    max_tokens: int,
    timeout: float,
) -> List[ModelResult]:
    async def _single(model: str) -> ModelResult:
        try:
            data = await _call_model(model, query, system_prompt, max_tokens, timeout)
        except Exception as exc:
            return ModelResult(
                model=model,
                query=query,
                output="",
                score=0.0,
                latency_ms=0.0,
                cost_usd=0.0,
                passed=False,
                error=str(exc),
            )

        output = data["output"]
        if scorer is not None:
            score_val = float(scorer(output, expected))
        elif expected is not None:
            score_val = _compute_similarity(output, expected)
        else:
            score_val = 1.0 if output.strip() else 0.0

        return ModelResult(
            model=model,
            query=query,
            output=output,
            score=score_val,
            latency_ms=data["latency_ms"],
            cost_usd=data["cost_usd"],
            passed=score_val >= threshold,
            metadata=data.get("metadata", {}),
        )

    return list(await asyncio.gather(*[_single(m) for m in models]))


# ---------------------------------------------------------------------------
# Console table helper (optional — used by examples and CLI)
# ---------------------------------------------------------------------------

def print_comparison_table(results: List[ModelResult]) -> None:
    """Print a Rich table comparing model results.  Requires ``rich``."""
    try:
        from rich.console import Console
        from rich.table import Table
    except ImportError:
        # Fallback plain-text table
        print(f"\n{'Model':<40} {'Score':>6} {'Latency':>10} {'Cost':>10} {'Pass?':>6}")
        print("-" * 76)
        for r in results:
            status = "✓" if r.passed else "✗"
            print(
                f"{r.model:<40} {r.score:>6.2f} "
                f"{r.latency_ms:>9.0f}ms {r.cost_usd:>9.5f}$ {status:>6}"
            )
        return

    console = Console()
    table = Table(title="Model Comparison", show_header=True, header_style="bold blue")
    table.add_column("Model", style="cyan", no_wrap=True)
    table.add_column("Score", justify="right")
    table.add_column("Latency", justify="right")
    table.add_column("Cost", justify="right")
    table.add_column("Pass?", justify="center")

    for r in results:
        score_color = "green" if r.score >= 0.8 else ("yellow" if r.score >= 0.5 else "red")
        status = "[green]✓[/green]" if r.passed else "[red]✗[/red]"
        table.add_row(
            r.model,
            f"[{score_color}]{r.score:.2f}[/{score_color}]",
            f"{r.latency_ms:.0f}ms",
            f"${r.cost_usd:.5f}",
            status,
        )

    console.print(table)
