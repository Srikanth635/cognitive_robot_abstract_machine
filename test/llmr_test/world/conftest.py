"""Conftest for world tests — provides simple world fixtures."""
from __future__ import annotations

from typing_extensions import Dict, Any
import pytest
from types import SimpleNamespace
from krrood.symbol_graph.symbol_graph import Symbol, SymbolGraph


@pytest.fixture
def simple_world() -> Dict[str, Any]:
    """Return a dict of mock world objects for grounding tests.

    Uses SimpleNamespace to simulate body objects with .name attributes.
    """
    milk = SimpleNamespace(name="milk")
    table = SimpleNamespace(name="table")
    fridge = SimpleNamespace(name="fridge")
    return {"milk": milk, "table": table, "fridge": fridge}


class WorldBody(Symbol):
    def __init__(self, name: str, parent: "WorldBody | None" = None):
        self.name = name
        self.parent_connection = (
            SimpleNamespace(parent=parent) if parent is not None else None
        )


class MilkAnnotation(Symbol):
    _synonyms = {"milk"}

    def __init__(self, *bodies: WorldBody):
        self.bodies = list(bodies)


@pytest.fixture
def symbol_world() -> Dict[str, Any]:
    """Populate SymbolGraph with small deterministic world objects."""
    graph = SymbolGraph()
    graph.clear()
    table = WorldBody("table")
    counter = WorldBody("counter")
    milk_on_table = WorldBody("milk_on_table", parent=table)
    milk_on_counter = WorldBody("milk_on_counter", parent=counter)
    red_cup = WorldBody("red_cup")
    blue_cup = WorldBody("blue_cup")
    structural = WorldBody("base_link")
    annotation = MilkAnnotation(milk_on_table)
    for instance in (
        table,
        counter,
        milk_on_table,
        milk_on_counter,
        red_cup,
        blue_cup,
        structural,
        annotation,
    ):
        graph.ensure_wrapped_instance(instance)
    return {
        "table": table,
        "counter": counter,
        "milk_on_table": milk_on_table,
        "milk_on_counter": milk_on_counter,
        "red_cup": red_cup,
        "blue_cup": blue_cup,
        "structural": structural,
        "annotation": annotation,
        "body_type": WorldBody,
        "annotation_type": MilkAnnotation,
    }
