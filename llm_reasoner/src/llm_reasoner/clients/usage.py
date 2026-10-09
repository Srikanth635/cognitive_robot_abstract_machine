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
