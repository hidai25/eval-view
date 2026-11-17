"""Model pricing lookups (evalview.core.pricing)."""

import pytest

from evalview.core.pricing import _find_pricing, calculate_cost, format_pricing_line


@pytest.mark.parametrize(
    "model, expected",
    [
        ("claude-opus-4-6", (5.00, 25.00, 0.50)),
        ("claude-opus-4-5-20251101", (5.00, 25.00, 0.50)),
        ("claude-haiku-4-5-20251001", (1.00, 5.00, 0.10)),
        ("gpt-5.4", (2.50, 15.00, 0.25)),
        ("gpt-5.4-mini", (0.75, 4.50, 0.075)),
        ("gpt-5.1", (1.25, 10.00, 0.125)),
        ("gemini-2.5-flash", (0.30, 2.50, 0.03)),
    ],
)
def test_list_prices(model, expected):
    assert _find_pricing(model) == pytest.approx(expected)


def test_newer_model_is_not_priced_as_an_older_one():
    # "gpt-5" used to match "gpt-5.5" by substring
    assert _find_pricing("gpt-5.5")[:2] == pytest.approx((5.00, 30.00))


def test_other_providers_keep_the_table():
    assert _find_pricing("command-r-plus") == (2.50, 10.00, 0.0)
    assert _find_pricing("some-unknown-model") == (0.25, 2.00, 0.025)


def test_calculate_cost_with_cache():
    assert calculate_cost("claude-sonnet-4-6", 1_000_000, 1_000_000, 1_000_000) == pytest.approx(18.30)


def test_format_pricing_line():
    assert format_pricing_line("claude-opus-4-6") == "$5.0/M input, $25.0/M output"
