"""Approximate per-model token prices ($ per 1M tokens) for eval cost estimates.

List prices only — good enough to compare providers and track cost per resolution.
"""
from __future__ import annotations

# (input_per_1M, output_per_1M) in USD.
PRICES: dict[str, tuple[float, float]] = {
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4o": (2.50, 10.00),
    "gpt-4.1": (2.00, 8.00),
    "claude-opus-5": (5.00, 25.00),
    "claude-opus-4-8": (5.00, 25.00),
    "claude-sonnet-5": (3.00, 15.00),
    "claude-haiku-4-5": (1.00, 5.00),
}


def price_for(model: str) -> tuple[float, float]:
    if not model:
        return (0.0, 0.0)
    for key, price in PRICES.items():
        if model.startswith(key) or key in model:
            return price
    return (0.0, 0.0)  # unknown / offline demo -> no cost


def cost_usd(model: str, input_tokens: int, output_tokens: int) -> float:
    p_in, p_out = price_for(model)
    return round(input_tokens / 1_000_000 * p_in + output_tokens / 1_000_000 * p_out, 6)
