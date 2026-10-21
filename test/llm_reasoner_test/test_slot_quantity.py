"""
Tests that a kind of quantity about the filler of a slot is one term of the action core:
the choices an action leaves open, the quantities deciding whether a relation of the
image schemas holds and the indicators a failure is noticed by are kinds of it, each
adding only what is its own.
"""

from __future__ import annotations

from dataclasses import fields

import pytest
from rdflib import RDFS, Graph, Namespace, URIRef
from typing_extensions import Optional, get_type_hints

from llm_reasoner.knowledge.ontologies import DUL_NAMESPACE, KNOWLEDGE_ONTOLOGY_MODULES
from llm_reasoner.knowledge.ontologies.action_perspectives import (
    action_core,
    failures,
    image_schemas,
)

DUL = Namespace(DUL_NAMESPACE)
"""
The terms of DOLCE+DnS Ultralite.
"""

KINDS_OF_QUANTITY = [
    (action_core.ParameterDimension, set()),
    (image_schemas.DecidingQuantity, {"quantity_aspect"}),
    (failures.Indicator, {"has_indicator_form", "noted_by"}),
]
"""
Each kind of quantity about a slot, with the fields only it has.
"""

# %% fixtures


@pytest.fixture(scope="module")
def graph() -> Graph:
    """
    Every module of both families in one graph.
    """
    union = Graph()
    for module in KNOWLEDGE_ONTOLOGY_MODULES:
        union.parse(module.path, format="turtle")
    return union


def field_names(cls: type) -> set[str]:
    """
    :return: The names of the fields of a generated class.
    """
    return {declared.name for declared in fields(cls)}


# %% the term


class TestTheTerm:
    def test_a_quantity_names_its_kind_its_slot_and_the_slot_it_is_measured_from(
        self,
    ):
        hints = get_type_hints(action_core.SlotQuantity)
        assert hints["has_quantity_kind"] == Optional[action_core.QuantityKind]
        assert hints["about_slot"] == Optional[action_core.FrameSlot]
        assert hints["relative_to_slot"] == Optional[action_core.FrameSlot]

    def test_it_is_a_dul_concept(self, graph):
        assert (
            URIRef(action_core.SlotQuantity.term.iri),
            RDFS.subClassOf,
            DUL.Concept,
        ) in graph

    @pytest.mark.parametrize("kind, own_fields", KINDS_OF_QUANTITY)
    def test_every_kind_specialises_it(self, kind, own_fields):
        assert issubclass(kind, action_core.SlotQuantity)

    @pytest.mark.parametrize("kind, own_fields", KINDS_OF_QUANTITY)
    def test_every_kind_adds_only_its_own_fields(self, kind, own_fields):
        assert field_names(kind) - field_names(action_core.SlotQuantity) == own_fields

    @pytest.mark.parametrize("kind, own_fields", KINDS_OF_QUANTITY)
    def test_no_kind_is_a_dul_description(self, graph, kind, own_fields):
        assert (
            URIRef(kind.term.iri),
            RDFS.subClassOf,
            DUL.Description,
        ) not in graph
