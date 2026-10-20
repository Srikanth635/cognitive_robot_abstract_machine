"""
Root fixtures for static discovery.
"""

import pytest
from first.base import helper

# %% fixtures


@pytest.fixture(scope="session")
def seed() -> int:
    """
    Provide a source-backed input.
    """
    return helper(1)


@pytest.fixture(autouse=True)
def environment(seed: int) -> int:
    """
    Require the root input for every test.
    """
    return seed
