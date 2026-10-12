"""
The tokens a completion consumed, which is what it costs.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TokenUsage:
    """
    The tokens one or more requests to a model consumed.
    """

    input_tokens: int = 0
    """
    The input tokens read without the cache.
    """

    output_tokens: int = 0
    """
    The tokens the model generated.
    """

    cache_creation_input_tokens: int = 0
    """
    The input tokens written to the cache.
    """

    cache_read_input_tokens: int = 0
    """
    The input tokens read from the cache.
    """

    def __add__(self, other: TokenUsage) -> TokenUsage:
        return TokenUsage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            cache_creation_input_tokens=self.cache_creation_input_tokens
            + other.cache_creation_input_tokens,
            cache_read_input_tokens=self.cache_read_input_tokens
            + other.cache_read_input_tokens,
        )


TOKENS_PER_PRICE = 1_000_000
"""
How many tokens a price is stated for.
"""


@dataclass(frozen=True)
class TokenPrices:
    """
    What a model charges, in dollars per million tokens of each kind.
    """

    input_tokens: float
    """
    For input tokens read without the cache.
    """

    output_tokens: float
    """
    For the tokens the model generates.
    """

    cache_creation_input_tokens: float
    """
    For input tokens written to the cache.
    """

    cache_read_input_tokens: float
    """
    For input tokens read from the cache.
    """

    def cost_of(self, usage: TokenUsage) -> float:
        """
        :return: What the usage cost, in dollars.
        """
        return (
            usage.input_tokens * self.input_tokens
            + usage.output_tokens * self.output_tokens
            + usage.cache_creation_input_tokens * self.cache_creation_input_tokens
            + usage.cache_read_input_tokens * self.cache_read_input_tokens
        ) / TOKENS_PER_PRICE


SONNET_5_5_PRICES = TokenPrices(
    input_tokens=2.0,
    output_tokens=10.0,
    cache_creation_input_tokens=2.5,
    cache_read_input_tokens=0.2,
)
"""
Claude Sonnet 5.5's prices.
"""
