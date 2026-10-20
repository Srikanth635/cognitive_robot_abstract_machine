"""
Tests that whether a condition holds is one term of the action core, used by the
predictive manipulation core's requirements, the action frame's slot conditions and the
failures' observed conditions alike.
"""

from __future__ import annotations

import pytest
from rdflib import OWL, RDF, RDFS, Graph, URIRef
from rdflib.collection import Collection
from typing_extensions import Optional, get_type_hints

from llm_reasoner.knowledge.ontologies import (
    KNOWLEDGE_ONTOLOGY_MODULES,
    ActionPerspectiveFile,
    PredictiveManipulationOntologyFile,
)
from llm_reasoner.knowledge.ontologies.action_perspectives import (
    action_core,
    action_frame,
    failures,
)
from llm_reasoner.knowledge.ontologies.predictive_manipulation import core
from llm_reasoner.knowledge.ontologies.terms import PropertyTerm

POLARITY = URIRef(ActionPerspectiveFile.ACTION_CORE.namespace + "ConditionPolarity")
"""
The action core's class of whether a condition holds.
"""

POLARITY_FIELDS = [
    (core.ConditionRequirement, "has_condition_polarity"),
    (action_frame.SlotCondition, "has_polarity"),
    (failures.ObservableCondition, "failed_condition_presence"),
]
"""
The field of each module saying whether a condition holds.
"""

# %% fixtures


@pytest.fixture(scope="module")
def graphs() -> dict:
    """
    Every module of both families, parsed on its own.
    """
    return {
        module: Graph().parse(module.path, format="turtle")
        for module in KNOWLEDGE_ONTOLOGY_MODULES
    }


def polarity_members(graph: Graph) -> set[URIRef]:
    """
    :return: The individuals the graph enumerates as the action core's polarity.
    """
    (enumeration,) = graph.objects(POLARITY, OWL.equivalentClass)
    return set(Collection(graph, graph.value(enumeration, OWL.oneOf)))


# %% the term


class TestTheTerm:
    def test_the_action_core_declares_it_with_two_members(self, graphs):
        graph = graphs[ActionPerspectiveFile.ACTION_CORE]
        assert (POLARITY, RDF.type, OWL.Class) in graph
        assert polarity_members(graph) == {
            URIRef(member.value) for member in action_core.ConditionPolarity
        }
        assert len(action_core.ConditionPolarity) == 2

    @pytest.mark.parametrize("owner, field_name", POLARITY_FIELDS)
    def test_every_polarity_property_ranges_over_it(self, graphs, owner, field_name):
        prop = URIRef(PropertyTerm.of_field(owner, field_name).iri)
        ranges = {
            value_range
            for graph in graphs.values()
            for value_range in graph.objects(prop, RDFS.range)
        }
        assert ranges == {POLARITY}

    @pytest.mark.parametrize("owner, field_name", POLARITY_FIELDS)
    def test_every_polarity_field_holds_a_member_of_it(self, owner, field_name):
        assert (
            get_type_hints(owner)[field_name] == Optional[action_core.ConditionPolarity]
        )

    def test_no_module_keeps_a_polarity_of_its_own(self):
        assert "ConditionPolarity" not in vars(core)
        assert "SlotConditionPolarity" not in vars(action_frame)
        assert "StatePresence" not in vars(failures)


# %% the predictive manipulation core


class TestThePredictiveManipulationCore:
    def test_it_imports_the_action_core(self, graphs):
        graph = graphs[PredictiveManipulationOntologyFile.CORE]
        assert URIRef(ActionPerspectiveFile.ACTION_CORE.namespace.rstrip("#")) in set(
            graph.objects(None, OWL.imports)
        )

    def test_its_requirements_name_only_members_of_the_term(self, graphs):
        graph = graphs[PredictiveManipulationOntologyFile.CORE]
        prop = URIRef(
            PropertyTerm.of_field(
                core.ConditionRequirement, "has_condition_polarity"
            ).iri
        )
        named = {
            graph.value(restriction, OWL.hasValue)
            for restriction in graph.subjects(OWL.onProperty, prop)
            if graph.value(restriction, OWL.hasValue) is not None
        }
        assert named == {
            URIRef(member.value) for member in action_core.ConditionPolarity
        }
