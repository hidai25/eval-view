"""Model pricing configuration for cost tracking."""

from typing import Dict, Tuple

# Pricing per 1 million tokens (as of January 2025)
# Format: (input_price, output_price, cached_input_price)

MODEL_PRICING: Dict[str, Tuple[float, float, float]] = {
    # GPT-5 Family
    "gpt-5": (1.25, 10.0, 0.125),
    "gpt-5-mini": (0.25, 2.0, 0.025),
    "gpt-5-nano": (0.05, 0.40, 0.005),
    # GPT-4o Family (for reference)
    "gpt-4o": (2.50, 10.0, 0.25),
    "gpt-4o-mini": (0.15, 0.60, 0.015),
    # GPT-4 Turbo (legacy)
    "gpt-4-turbo": (10.0, 30.0, 1.0),
    "gpt-4": (30.0, 60.0, 3.0),
    # GPT-3.5 Turbo (legacy)
    "gpt-3.5-turbo": (0.50, 1.50, 0.05),
    # Default fallback (gpt-5-mini pricing)
    "default": (0.25, 2.0, 0.025),
}


def calculate_cost(
    model_name: str,
    input_tokens: int = 0,
    output_tokens: int = 0,
    cached_tokens: int = 0,
) -> float:
    """
    Calculate cost for model usage.

    Args:
        model_name: Name of the model (e.g., "gpt-5", "gpt-5-mini")
        input_tokens: Number of input tokens
        output_tokens: Number of output tokens
        cached_tokens: Number of cached input tokens (90% discount)

    Returns:
        Total cost in dollars
    """
    # Normalize model name (handle variations)
    model_name = model_name.lower().strip()

    # Try exact match first
    if model_name in MODEL_PRICING:
        input_price, output_price, cached_price = MODEL_PRICING[model_name]
    else:
        # Try partial match
        for key in MODEL_PRICING:
            if key in model_name or model_name in key:
                input_price, output_price, cached_price = MODEL_PRICING[key]
                break
        else:
            # Use default pricing
            input_price, output_price, cached_price = MODEL_PRICING["default"]

    # Calculate cost (prices are per 1M tokens)
    input_cost = (input_tokens / 1_000_000) * input_price
    output_cost = (output_tokens / 1_000_000) * output_price
    cached_cost = (cached_tokens / 1_000_000) * cached_price

    total_cost = input_cost + output_cost + cached_cost

    return total_cost


def get_model_pricing_info(model_name: str) -> Dict[str, float]:
    """
    Get pricing information for a model.

    Args:
        model_name: Name of the model

    Returns:
        Dictionary with pricing details
    """
    model_name = model_name.lower().strip()

    if model_name in MODEL_PRICING:
        input_price, output_price, cached_price = MODEL_PRICING[model_name]
    else:
        input_price, output_price, cached_price = MODEL_PRICING["default"]

    return {
        "input_price_per_1m": input_price,
        "output_price_per_1m": output_price,
        "cached_price_per_1m": cached_price,
        "input_price_per_token": input_price / 1_000_000,
        "output_price_per_token": output_price / 1_000_000,
        "cached_price_per_token": cached_price / 1_000_000,
    }


# Example usage
if __name__ == "__main__":
    # Test cost calculation
    cost = calculate_cost(
        model_name="gpt-5-mini",
        input_tokens=1000,
        output_tokens=500,
    )
    print(f"Cost for 1000 input + 500 output tokens (gpt-5-mini): ${cost:.6f}")

    cost = calculate_cost(
        model_name="gpt-5",
        input_tokens=10000,
        output_tokens=5000,
        cached_tokens=50000,
    )
    print(f"Cost for 10k input + 5k output + 50k cached (gpt-5): ${cost:.6f}")

    # Print all pricing
    print("\nCurrent Model Pricing:")
    for model, (inp, out, cache) in MODEL_PRICING.items():
        print(f"  {model:20s} → Input: ${inp}/1M, Output: ${out}/1M, Cached: ${cache}/1M")
