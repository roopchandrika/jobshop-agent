"""Token prices. Supplied by the operator (env vars or CLI flags), never hardcoded: they change,
differ per model, and a wrong built-in table would silently produce wrong cost numbers."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Prices:
    """USD per million tokens. Cached-token discounts are not modelled: this agent does not use prompt caching."""

    input_per_mtok: float
    output_per_mtok: float

    def __post_init__(self) -> None:
        if self.input_per_mtok < 0 or self.output_per_mtok < 0:
            raise ValueError("prices cannot be negative")

    def cost(self, input_tokens: int, output_tokens: int) -> float:
        return (input_tokens * self.input_per_mtok + output_tokens * self.output_per_mtok) / 1e6


def parse_prices(text: str) -> Prices:
    """'3,15' -> Prices(3.0, 15.0): input then output, USD per million tokens."""
    try:
        first, second = (float(part) for part in text.split(","))
        return Prices(first, second)
    except ValueError:
        raise ValueError(f"prices must look like '3,15' (input,output USD per million tokens), got '{text}'") from None
