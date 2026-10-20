"""
Tests that a change becoming larger or smaller is one term of the action core, which the
action frame uses as it is and the failures' and the predictive manipulation core's
directions share beside the values only they have.
"""

from __future__ import annotations

import pytest
from rdflib import OWL, RDF, RDFS, Graph, Namespace, URIRef
from rdflib.collection import Collection
from typing_extensions import Optional, get_type_hints

from llm_reasoner.knowledge.ontologies import (
    DUL_NAMESPACE,
    KNOWLEDGE_ONTOLOGY_MODULES,
    ActionPerspectiveFile,
)
from llm_reasoner.knowledge.ontologies.action_perspectives import (
    action_core,
    action_frame,
    failures,
)
from llm_reasoner.knowledge.ontologies.predictive_manipulation import core, prehension
from llm_reasoner.knowledge.ontologies.terms import PropertyTerm
from llm_reasoner.knowledge.ontologies.vocabulary import Vocabulary
from llm_reasoner.knowledge.representation.rdf import GraphReader

DUL = Namespace(DUL_NAMESPACE)
"""
The terms of DOLCE+DnS Ultralite.
"""

SHARING_DIRECTIONS = [
    (
        failures.ParameterChangeDirection,
        {"TOWARDS_TARGET", "AWAY_FROM_TARGET", "TO_ALTERNATIVE"},
    ),
    (core.CorrectionDirection, {"HOLD_SETPOINT", "REVERSE_CHANGE", "STOP_CHANGE"}),
    (core.EstimateUpdateDirection, {"REPLACE_ESTIMATE"}),
]
"""
Each direction that shares the core's members, with the members only it has.
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


def class_iri(enumeration: type) -> URIRef:
    """
    :return: The IRI of the class an enumeration stands for: the namespace of its
        members' module with the enumeration's name.
    """
    module_file = Vocabulary().module_file_of(enumeration)
    return URIRef(module_file.namespace + enumeration.__name__)


def enumerated(graph: Graph, cls: URIRef) -> set[URIRef]:
    """
    :return: The individuals the graph enumerates as the class's members.
    """
    (enumeration,) = graph.objects(cls, OWL.equivalentClass)
    return set(Collection(graph, graph.value(enumeration, OWL.oneOf)))


CORE_MEMBERS = {URIRef(member.value) for member in action_core.ChangeDirection}
"""
The action core's members, larger and smaller.
"""


# %% the term


class TestTheTerm:
    def test_the_action_core_declares_it_with_two_members(self, graph):
        assert enumerated(graph, class_iri(action_core.ChangeDirection)) == (
            CORE_MEMBERS
        )
        assert [member.name for member in action_core.ChangeDirection] == [
            "INCREASES",
            "DECREASES",
        ]

    def test_the_action_frame_uses_it_as_it_is(self):
        assert "ChangeDirection" not in vars(action_frame)
        assert (
            get_type_hints(action_frame.AdditionalConsequence)["changes_in_direction"]
            == Optional[action_core.ChangeDirection]
        )

    @pytest.mark.parametrize("direction, own_members", SHARING_DIRECTIONS)
    def test_a_direction_has_the_core_members_and_only_its_own_others(
        self, graph, direction, own_members
    ):
        assert {member.name for member in direction} == {
            *(member.name for member in action_core.ChangeDirection),
            *own_members,
        }
        assert CORE_MEMBERS <= enumerated(graph, class_iri(direction))

    @pytest.mark.parametrize("direction, own_members", SHARING_DIRECTIONS)
    def test_a_shared_member_is_the_core_member(self, direction, own_members):
        for member in action_core.ChangeDirection:
            assert direction[member.name].value == member.value

    @pytest.mark.parametrize(
        "direction",
        [action_core.ChangeDirection, *(shared for shared, _ in SHARING_DIRECTIONS)],
    )
    def test_every_direction_is_a_dul_concept(self, graph, direction):
        assert (class_iri(direction), RDFS.subClassOf, DUL.Concept) in graph


# %% reading a shared member


class TestReadingASharedMember:
    def test_a_stated_member_is_read_as_the_fields_own(self):
        graph = Graph()
        individual = URIRef("http://example.org/hypothesis#correction")
        graph.add((individual, RDF.type, URIRef(core.CorrectiveAction.term.iri)))
        graph.add(
            (
                individual,
                URIRef(
                    PropertyTerm.of_field(
                        core.CorrectiveAction, "has_correction_direction"
                    ).iri
                ),
                URIRef(action_core.ChangeDirection.INCREASES.value),
            )
        )
        (correction,) = GraphReader().read(graph).fragments
        assert correction.has_correction_direction is (
            core.CorrectionDirection.INCREASES
        )

    def test_an_entailed_member_is_the_fields_own(self):
        reduction = prehension.LoadRateReduction()
        Vocabulary().fill_entailed_values(reduction)
        assert reduction.has_correction_direction is core.CorrectionDirection.DECREASES
