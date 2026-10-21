"""
Tests for the class-level graph of the ontology behind a hypothesis, on the checked-in
sample hypothesis.
"""

from collections import Counter
from dataclasses import fields

import pytest

krrood = pytest.importorskip("krrood", reason="EQL requires krrood")

from llm_reasoner.knowledge.access import (  # noqa: E402
    ConceptReference,
    core,
    CRAM_AICOR_NAMESPACE,
    described,
    DescriptionEntity,
    pouring,
    prehension,
    PropertyTerm,
    Vocabulary,
)

from cramera_resolution import hypothesis_knowledge  # noqa: E402
from cramera_resolution.hypothesis_knowledge.ontology_backing import OntologyBacking  # noqa: E402
from cramera_resolution.hypothesis_knowledge.ontology_class_graph import (  # noqa: E402
    OntologyClassGraph,
    OntologyEdgeKind,
    OntologyNodeCategory,
)

# %% helpers


@pytest.fixture
def view():
    return hypothesis_knowledge.sample_view()


@pytest.fixture
def graph(view):
    return OntologyClassGraph.of(view)


def node(graph, iri):
    (found,) = [node for node in graph.nodes if node.iri == iri]
    return found


def edges(graph, kind):
    return [edge for edge in graph.edges if edge.kind is kind]


def classes_of(graph):
    vocabulary = Vocabulary()
    return [
        vocabulary.definition_named(node.iri)
        for node in graph.nodes
        if vocabulary.definition_named(node.iri) is not None
    ]


# %% classes


class TestClasses:
    def test_every_class_the_hypothesis_instantiates_is_a_used_node(self, graph, view):
        counts = Counter(type(fragment).term.iri for fragment in view.fragments)
        used = {
            node.iri: node.instance_count
            for node in graph.nodes
            if node.category is OntologyNodeCategory.USED
        }
        assert used == dict(counts)

    def test_superclass_of_a_used_class_is_a_microworld_node(self, graph):
        superclass = node(graph, core.ActionPhase.term.iri)
        assert superclass.category in (
            OntologyNodeCategory.USED,
            OntologyNodeCategory.MICROWORLD,
        )

    def test_linked_cram_aicor_term_is_a_linked_node(self, graph, view):
        linked = next(
            term
            for fragment in view.fragments
            for cls in type(fragment).__mro__
            if issubclass(cls, DescriptionEntity) and "term" in vars(cls)
            for term in cls.term.linked_terms
            if term.startswith(CRAM_AICOR_NAMESPACE)
        )
        assert node(graph, linked).category is OntologyNodeCategory.LINKED

    def test_concept_the_hypothesis_introduces_is_an_introduced_node(self, graph, view):
        introduced = {
            value.iri
            for fragment in view.fragments
            for declared in fields(fragment)
            for value in [getattr(fragment, declared.name)]
            if isinstance(value, ConceptReference)
            and OntologyBacking().source_of(value.iri).name == "HYPOTHESIS"
        }
        assert introduced
        assert {
            node.iri
            for node in graph.nodes
            if node.category is OntologyNodeCategory.INTRODUCED
        } >= introduced

    def test_class_carries_its_description_requirements_and_claims(self, graph):
        lift = node(graph, prehension.LiftPhase.term.iri)
        assert lift.description
        assert [
            described(restriction, False)
            for restriction in prehension.LiftPhase.term.restrictions
        ] == list(lift.requirements)
        stability = node(graph, prehension.GraspStability.term.iri)
        assert {claim.iri for claim in stability.claims} == set(
            prehension.GraspStability.term.claims
        )
        assert all(claim.text for claim in stability.claims)


# %% links


class TestLinks:
    def test_class_is_a_its_direct_superclasses_and_linked_terms(self, graph):
        lift = prehension.LiftPhase
        direct = {
            base.term.iri
            for base in lift.__bases__
            if issubclass(base, DescriptionEntity) and "term" in vars(base)
        }
        assert {
            edge.target
            for edge in edges(graph, OntologyEdgeKind.IS_A)
            if edge.source == lift.term.iri
        } == direct | set(lift.term.linked_terms)

    def test_linked_terms_are_reached_by_is_a_links(self, graph):
        targets = {edge.target for edge in edges(graph, OntologyEdgeKind.IS_A)}
        assert {
            node.iri
            for node in graph.nodes
            if node.category is OntologyNodeCategory.LINKED
            and not any(
                edge.target == node.iri for edge in edges(graph, OntologyEdgeKind.CITES)
            )
        } <= targets

    def test_structure_declares_its_phases_as_components(self, graph, view):
        defines_phase = PropertyTerm.of_field(type(view.root), "defines_phase").iri
        component = [
            edge
            for edge in edges(graph, OntologyEdgeKind.COMPONENT)
            if edge.source == type(view.root).term.iri
            and edge.property_iri == defines_phase
        ]
        assert {edge.target for edge in component} == {
            type(phase).term.iri for phase in view.root.defines_phase
        }

    def test_relation_counts_the_links_the_hypothesis_states(self, graph, view):
        involves_role = PropertyTerm.of_field(core.ActionPhase, "involves_role").iri
        lift = prehension.LiftPhase
        expected = Counter(
            type(role).term.iri
            for phase in view.root.defines_phase
            if type(phase) is lift
            for role in phase.involves_role
        )
        found = {
            edge.target: edge.count
            for edge in edges(graph, OntologyEdgeKind.RELATION)
            if edge.source == lift.term.iri and edge.property_iri == involves_role
        }
        assert found == dict(expected)

    def test_cited_concept_is_linked_by_the_property_citing_it(self, graph, view):
        realizes = PropertyTerm.of_field(
            core.ActionPhase, "realizes_action_category"
        ).iri
        assert any(
            edge.property_iri == realizes
            and edge.target == CRAM_AICOR_NAMESPACE + "Lifting"
            for edge in edges(graph, OntologyEdgeKind.CITES)
        )

    def test_disjoint_classes_are_linked(self, graph):
        assert any(
            {edge.source, edge.target}
            == {prehension.LiftPhase.term.iri, prehension.TransportPhase.term.iri}
            for edge in edges(graph, OntologyEdgeKind.DISJOINT)
        )

    def test_class_whose_instances_meet_a_definition_is_classified_by_it(
        self, graph, view
    ):
        lifts = [
            phase
            for phase in view.root.defines_phase
            if type(phase) is prehension.LiftPhase
        ]
        (classified,) = [
            edge
            for edge in edges(graph, OntologyEdgeKind.CLASSIFIED_AS)
            if edge.source == prehension.LiftPhase.term.iri
            and edge.target == core.CoupledPhase.term.iri
        ]
        assert classified.count == len(lifts)
        assert node(graph, core.CoupledPhase.term.iri).definition == described(
            core.CoupledPhase.term.definition, False
        )

    def test_class_whose_instances_miss_a_definition_is_not_classified_by_it(
        self, graph
    ):
        assert not any(
            edge.source == prehension.ReachPhase.term.iri
            and edge.target == core.CoupledPhase.term.iri
            for edge in edges(graph, OntologyEdgeKind.CLASSIFIED_AS)
        )

    def test_subclass_of_a_defined_class_is_linked_by_is_a_only(self, graph):
        assert not any(
            issubclass(
                next(cls for cls in classes_of(graph) if cls.term.iri == edge.source),
                next(cls for cls in classes_of(graph) if cls.term.iri == edge.target),
            )
            for edge in edges(graph, OntologyEdgeKind.CLASSIFIED_AS)
        )

    def test_every_link_joins_two_nodes_of_the_graph(self, graph):
        iris = {node.iri for node in graph.nodes}
        assert all(edge.source in iris and edge.target in iris for edge in graph.edges)

    def test_root_structure_class_is_used(self, graph, view):
        assert (
            node(graph, pouring.PouringPhaseStructure.term.iri).category
            is OntologyNodeCategory.USED
        )
