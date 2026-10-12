"""
Tests of what the tokens a model consumed cost.
"""

from __future__ import annotations

import pytest

from llm_reasoner.clients.usage import TokenPrices, TokenUsage


def test_each_kind_of_token_is_priced_at_its_own_rate():
    prices = TokenPrices(
        input_tokens=2.0,
        output_tokens=10.0,
        cache_creation_input_tokens=2.5,
        cache_read_input_tokens=0.2,
    )
    usage = TokenUsage(
        input_tokens=1_000_000,
        output_tokens=100_000,
        cache_creation_input_tokens=400_000,
        cache_read_input_tokens=2_000_000,
    )
    assert prices.cost_of(usage) == pytest.approx(2.0 + 1.0 + 1.0 + 0.4)
