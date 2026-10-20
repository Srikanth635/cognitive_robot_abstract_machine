"""
Tests exercise direct calls, fixtures and dynamic selection.
"""

import pytest
from first.base import helper

# %% test declarations


def test_direct(seed: int) -> None:
    """
    Call the indexed function directly.
    """
    assert helper(seed) == seed


def test_fixture(seed: int) -> None:
    """
    Consume the function through a fixture.
    """
    assert seed == 2


@pytest.mark.parametrize("value", [1, 2])
def test_parameter(value: int) -> None:
    """
    Direct parametrization does not request a fixture.
    """
    assert helper(value) == value


@pytest.mark.parametrize("seed", [1], indirect=True)
def test_indirect(seed: int) -> None:
    """
    Indirect parametrization needs an explicit uncertainty signal.
    """
    assert seed == 2


def test_dynamic(request: object) -> None:
    """
    A runtime fixture name cannot be statically established.
    """
    assert request.getfixturevalue("seed") == 2
