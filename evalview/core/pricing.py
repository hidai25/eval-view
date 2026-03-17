"""Model pricing configuration for cost tracking.

Single source of truth for all model pricing across EvalView.
Both adapter cost calculation and judge cost tracking use this table.

Pricing is per 1 million tokens. Update when providers change prices.
Last verified: March 2026.
"""

from typing import Dict, Optional, Tuple

# Format: (input_price_per_1M, output_price_per_1M, cached_input_price_per_1M)
#
# Sources:
#   OpenAI:    https://openai.com/api/pricing/
#   Anthropic: https://docs.anthropic.com/en/docs/about-claude/models
#   Google:    https://ai.google.dev/pricing
#   xAI:       https://docs.x.ai/docs
#   DeepSeek:  https://platform.deepseek.com/api-docs/pricing

MODEL_PRICING: Dict[str, Tuple[float, float, float]] = {
    # ── OpenAI GPT-5 family ──────────────────────────────────────────
    "gpt-5.4":           (2.00, 8.00, 0.50),
    "gpt-5.4-mini":      (0.10, 0.40, 0.025),
    "gpt-5.1":           (2.00, 8.00, 0.50),
    "gpt-5":             (1.25, 10.00, 0.125),
    "gpt-5-mini":        (0.25, 2.00, 0.025),
    "gpt-5-nano":        (0.05, 0.40, 0.005),
    # ── OpenAI GPT-4 family (legacy) ─────────────────────────────────
    "gpt-4o":            (2.50, 10.00, 1.25),
    "gpt-4o-mini":       (0.15, 0.60, 0.075),
    "gpt-4-turbo":       (10.00, 30.00, 5.00),
    "gpt-4":             (30.00, 60.00, 15.00),
    "gpt-3.5-turbo":     (0.50, 1.50, 0.25),
    # ── OpenAI o-series (reasoning) ──────────────────────────────────
    "o4-mini":           (1.10, 4.40, 0.275),
    "o3":                (2.00, 8.00, 0.50),
    "o3-mini":           (1.10, 4.40, 0.275),
    "o1":                (15.00, 60.00, 7.50),
    "o1-mini":           (1.10, 4.40, 0.275),
    # ── Anthropic Claude 4 family ────────────────────────────────────
    "claude-opus-4-6":               (15.00, 75.00, 1.50),
    "claude-opus-4-5-20251101":      (15.00, 75.00, 1.50),
    "claude-sonnet-4-6":             (3.00, 15.00, 0.30),
    "claude-sonnet-4-5-20250929":    (3.00, 15.00, 0.30),
    "claude-haiku-4-5-20251001":     (0.80, 4.00, 0.08),
    # ── Anthropic Claude 3.5 (legacy) ────────────────────────────────
    "claude-3-5-sonnet":  (3.00, 15.00, 0.30),
    "claude-3-5-haiku":   (0.80, 4.00, 0.08),
    # ── Google Gemini ────────────────────────────────────────────────
    "gemini-3.0":         (1.25, 5.00, 0.30),
    "gemini-2.5-pro":     (1.25, 10.00, 0.30),
    "gemini-2.5-flash":   (0.15, 0.60, 0.0375),
    "gemini-2.0-flash":   (0.10, 0.40, 0.025),
    "gemini-1.5-pro":     (1.25, 5.00, 0.30),
    # ── xAI Grok ─────────────────────────────────────────────────────
    "grok-3":             (3.00, 15.00, 0.75),
    "grok-3-mini":        (0.30, 0.50, 0.075),
    "grok-2":             (2.00, 10.00, 0.50),
    "grok-2-latest":      (2.00, 10.00, 0.50),
    # ── DeepSeek ─────────────────────────────────────────────────────
    "deepseek-chat":      (0.14, 0.28, 0.014),
    "deepseek-reasoner":  (0.55, 2.19, 0.055),
    # ── Meta Llama (HuggingFace hosted) ──────────────────────────────
    "meta-llama/Llama-3.1-8B-Instruct":  (0.05, 0.05, 0.0),
    "meta-llama/Llama-3.1-70B-Instruct": (0.35, 0.40, 0.0),
    # ── Cohere ───────────────────────────────────────────────────────
    "command-r-plus":     (2.50, 10.00, 0.0),
    "command-r":          (0.15, 0.60, 0.0),
    # ── Mistral ──────────────────────────────────────────────────────
    "mistral-large":      (2.00, 6.00, 0.0),
    "mistral-small":      (0.10, 0.30, 0.0),
    # ── Fallback ─────────────────────────────────────────────────────
    "default":            (0.25, 2.00, 0.025),
}


def _find_pricing(model_name: str) -> Tuple[float, float, float]:
    """Find pricing for a model by exact match, then partial match, then default."""
    model_name = model_name.lower().strip()

    # Exact match
    if model_name in MODEL_PRICING:
        return MODEL_PRICING[model_name]

    # Partial match (handles versioned names like claude-sonnet-4-5-20250929)
    for key in MODEL_PRICING:
        if key in model_name or model_name in key:
            return MODEL_PRICING[key]

    return MODEL_PRICING["default"]


def calculate_cost(
    model_name: str,
    input_tokens: int = 0,
    output_tokens: int = 0,
    cached_tokens: int = 0,
) -> float:
    """Calculate cost for model usage.

    Args:
        model_name: Name of the model (e.g., "gpt-5.4", "claude-opus-4-6")
        input_tokens: Number of input tokens
        output_tokens: Number of output tokens
        cached_tokens: Number of cached input tokens (discounted rate)

    Returns:
        Total cost in dollars
    """
    input_price, output_price, cached_price = _find_pricing(model_name)

    return (
        (input_tokens / 1_000_000) * input_price
        + (output_tokens / 1_000_000) * output_price
        + (cached_tokens / 1_000_000) * cached_price
    )


def get_model_pricing_info(model_name: str) -> Dict[str, float]:
    """Get pricing information for a model.

    Returns:
        Dictionary with input/output/cached prices per 1M and per token.
    """
    input_price, output_price, cached_price = _find_pricing(model_name)

    return {
        "input_price_per_1m": input_price,
        "output_price_per_1m": output_price,
        "cached_price_per_1m": cached_price,
        "input_price_per_token": input_price / 1_000_000,
        "output_price_per_token": output_price / 1_000_000,
        "cached_price_per_token": cached_price / 1_000_000,
    }


def format_pricing_line(model_name: str) -> Optional[str]:
    """Format a human-readable pricing summary for a model.

    Returns None if using default/fallback pricing.
    """
    model_lower = model_name.lower().strip()
    if model_lower not in MODEL_PRICING and not any(
        k in model_lower or model_lower in k for k in MODEL_PRICING if k != "default"
    ):
        return None

    input_price, output_price, _ = _find_pricing(model_name)
    return f"${input_price}/M input, ${output_price}/M output"
