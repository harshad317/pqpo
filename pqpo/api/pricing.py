"""Token pricing tables and dollar-cost computation.

Prices are USD per 1M tokens. Update the snapshot when a paper run is frozen and
record the snapshot in every artifact manifest (Sec 3.8). Numbers below are
illustrative defaults as of mid-2025; verify before any real billed run.
"""
from __future__ import annotations

# provider/model -> (input_per_million, output_per_million)
PRICING_USD_PER_M: dict[str, tuple[float, float]] = {
    "openai/gpt-4o-mini": (0.15, 0.60),
    "openai/gpt-4o": (2.50, 10.00),
    "openai/gpt-4.1-mini": (0.40, 1.60),
    "anthropic/claude-3-5-haiku": (0.80, 4.00),
    "anthropic/claude-3-5-sonnet": (3.00, 15.00),
    # The simulated target is free but we still account a synthetic price so that
    # cost-curve machinery is exercised end to end.
    "sim/sim-target": (0.15, 0.60),
}


def dollar_cost(model_key: str, input_tokens: int, output_tokens: int) -> float:
    """USD cost for a single call. Falls back to a conservative default price."""
    in_price, out_price = PRICING_USD_PER_M.get(model_key, (1.0, 3.0))
    return (input_tokens * in_price + output_tokens * out_price) / 1_000_000.0


def pricing_snapshot(model_key: str) -> dict:
    in_price, out_price = PRICING_USD_PER_M.get(model_key, (1.0, 3.0))
    return {
        "model_key": model_key,
        "input_per_million_tokens": in_price,
        "output_per_million_tokens": out_price,
    }
