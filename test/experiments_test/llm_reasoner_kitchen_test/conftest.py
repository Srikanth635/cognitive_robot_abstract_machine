"""
The breakfast scene each case is resolved in, set out afresh for every test, since the
symbol graph the resolver reads the world from is renewed for every test.
"""

from __future__ import annotations

from copy import deepcopy

import pytest

from experiments.llm_reasoner_kitchen.cases import KitchenQueries
from experiments.llm_reasoner_kitchen.scene import BreakfastScene


@pytest.fixture
def kitchen(pr2_apartment_world) -> KitchenQueries:
    """
    The breakfast scene, to write the cases' queries on.
    """
    return KitchenQueries(world=BreakfastScene().set_out(deepcopy(pr2_apartment_world)))


@pytest.fixture
def furnished_kitchen(pr2_apartment_world) -> KitchenQueries:
    """
    The breakfast scene with the apartment's furniture annotated.
    """
    return KitchenQueries(
        world=BreakfastScene(furnished=True).set_out(deepcopy(pr2_apartment_world))
    )
