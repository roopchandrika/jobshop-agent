"""Token prices. Supplied by the operator (env vars or CLI flags), never hardcoded: they change,
differ per model, and a wrong built-in table would silently produce wrong cost numbers."""

from __future__ import annotations

from dataclasses import dataclass

# How the API prices cached input relative to the plain input price (5-minute cache): a read costs a
# tenth, a write a quarter more. These are ratios of the operator's own input price, not a price table.
CACHE_READ_FACTOR = 0.1
CACHE_WRITE_FACTOR = 1.25


@dataclass(frozen=True)
class Prices:
    """USD per million tokens. Cache reads and writes default to the standard multiples of the input price."""

    input_per_mtok: float
    output_per_mtok: float
    cache_read_per_mtok: float | None = None
    cache_write_per_mtok: float | None = None

    def __post_init__(self) -> None:
        prices = (self.input_per_mtok, self.output_per_mtok, self.cache_read_per_mtok or 0, self.cache_write_per_mtok or 0)
        if any(p < 0 for p in prices):
            raise ValueError("prices cannot be negative")

    @property
    def cache_read(self) -> float:
        return self.input_per_mtok * CACHE_READ_FACTOR if self.cache_read_per_mtok is None else self.cache_read_per_mtok

    @property
    def cache_write(self) -> float:
        return self.input_per_mtok * CACHE_WRITE_FACTOR if self.cache_write_per_mtok is None else self.cache_write_per_mtok

    def cost(self, input_tokens: int, output_tokens: int, cache_read_tokens: int = 0, cache_write_tokens: int = 0) -> float:
        """``input_tokens`` excludes cached tokens, as in the API's usage report."""
        return (
            input_tokens * self.input_per_mtok
            + output_tokens * self.output_per_mtok
            + cache_read_tokens * self.cache_read
            + cache_write_tokens * self.cache_write
        ) / 1e6


def parse_prices(text: str) -> Prices:
    """'3,15' -> Prices(3.0, 15.0): input then output, USD per million tokens."""
    try:
        first, second = (float(part) for part in text.split(","))
        return Prices(first, second)
    except ValueError:
        raise ValueError(f"prices must look like '3,15' (input,output USD per million tokens), got '{text}'") from None
