"""
A nearer fixture overrides the root declaration.
"""

import pytest as testing
from first.base import helper

# %% fixtures


@testing.fixture(scope="module")
def seed() -> int:
    """
    Provide the nearer source-backed input.
    """
    return helper(2)
