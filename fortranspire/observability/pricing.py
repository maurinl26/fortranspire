"""Per-model token pricing — used to compute ``cost_usd`` on each Span.

Numbers are USD per million tokens, mirrored from Mistral La Plateforme
pricing as of 2026-06. Override per-deployment via
``FORTRANSPIRE_PRICING_FILE`` (JSON dict with the same shape).

When a model name isn't in the table, falls back to a conservative
``_default`` rate so cost tracking never silently zeroes out.
"""
from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path


# (input USD per 1M tokens, output USD per 1M tokens)
_BUILTIN_PRICES: dict[str, tuple[float, float]] = {
    "mistral-large-latest":           (2.0, 6.0),
    "mistral-large-2411":             (2.0, 6.0),
    "mistral-large-2407":             (2.0, 6.0),
    "codestral-latest":               (0.3, 0.9),
    "codestral-2501":                 (0.3, 0.9),
    "mistral-small-latest":           (0.2, 0.6),
    "mistral-nemo":                   (0.15, 0.15),
    "open-mistral-7b":                (0.25, 0.25),
    # OpenAI fallback rates — when fortranspire is pointed at an
    # OpenAI-compatible endpoint that isn't Mistral.
    "gpt-4o-mini":                    (0.15, 0.6),
    "gpt-4o":                         (2.5, 10.0),
    # Conservative default for unknown models — better to overestimate
    # cost than to silently report 0.
    "_default":                       (1.0, 3.0),
}


@lru_cache(maxsize=1)
def _load_prices() -> dict[str, tuple[float, float]]:
    """Merge built-in prices with the optional override file."""
    prices = dict(_BUILTIN_PRICES)
    override_path = os.getenv("FORTRANSPIRE_PRICING_FILE")
    if not override_path:
        return prices
    try:
        data = json.loads(Path(override_path).read_text(encoding="utf-8"))
        for key, value in data.items():
            if (isinstance(value, list) or isinstance(value, tuple)) and len(value) == 2:
                prices[key] = (float(value[0]), float(value[1]))
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        # Telemetry must never fail the pipeline — keep built-in defaults.
        pass
    return prices


def estimate_cost_usd(
    model: str | None,
    prompt_tokens: int,
    completion_tokens: int,
) -> float:
    """Return the USD cost of one LLM call given its model name + token counts."""
    prices = _load_prices()
    key = (model or "").lower() if model else "_default"
    input_price, output_price = prices.get(key, prices["_default"])
    return (prompt_tokens * input_price + completion_tokens * output_price) / 1_000_000


def clear_cache() -> None:
    """Wipe the cached price table — useful for tests that change env vars."""
    _load_prices.cache_clear()
