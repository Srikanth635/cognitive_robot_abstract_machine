"""
Tests for naming the ontology terms behind a projected hypothesis, on the checked-in
sample hypothesis.
"""

from pathlib import Path

import pytest

krrood = pytest.importorskip("krrood", reason="EQL requires krrood")

from llm_reasoner.hypotheses.graph import (  # noqa: E402
    HypothesisGraphNodeKind,
    HypothesisGraphPresentation,
    HypothesisGraphProjector,
)
from llm_reasoner.ontologies import (  # noqa: E402
    CRAM_AICOR_NAMESPACE,
    HYPOTHESIS_CONCEPT_NAMESPACE,
    PREDICTIVE_MANIPULATION_NAMESPACE_ROOT,
    PredictiveManipulationOntologyFile,
)
from llm_reasoner.ontologies.predictive_manipulation import core  # noqa: E402
from llm_reasoner.ontologies.terms import ConceptReference  # noqa: E402

from cram_viz import hypothesis_knowledge  # noqa: E402
from cram_viz.hypothesis_knowledge.ontology_backing import (  # noqa: E402
    DRAWN_LINK_NOTE,
    OntologyBacking,
    TermSource,
)
from cram_viz.hypothesis_knowledge.phase_hierarchy import (  # noqa: E402
    PhaseHierarchyOverlay,
)

# %% helpers


@pytest.fixture
def backing():
    return OntologyBacking()


@pytest.fixture
def view():
    return hypothesis_knowledge.sample_view()


@pytest.fixture
def projection(view):
    return PhaseHierarchyOverlay().apply(
        HypothesisGraphProjector().project(view, HypothesisGraphPresentation.COMPLETE),
        view,
    )


def node_of(projection, fragment):
    (node,) = [node for node in projection.nodes if node.source is fragment]
    return node


# %% prefixed names


class TestPrefixedNames:
    def test_every_microworld_module_has_a_namespace(self, backing):
        assert {
            PREDICTIVE_MANIPULATION_NAMESPACE_ROOT + Path(module).stem + "#"
            for module in PredictiveManipulationOntologyFile
        } <= {namespace.iri for namespace in backing.namespaces}

    def test_a_term_is_written_with_its_namespaces_prefix(self, backing):
        for namespace in backing.namespaces:
            assert backing.prefixed(namespace.iri + "Term") == (
                f"{namespace.prefix}:Term"
            )

    def test_a_term_outside_every_namespace_is_written_in_full(self, backing):
        iri = "http://example.org/elsewhere#Term"
        assert backing.prefixed(iri) == iri
        assert backing.source_of(iri) is TermSource.UNKNOWN

    @pytest.mark.parametrize(
        "namespace, source",
        [
            (CRAM_AICOR_NAMESPACE, TermSource.CRAM_AICOR),
            (HYPOTHESIS_CONCEPT_NAMESPACE, TermSource.HYPOTHESIS),
            (
                core.ActionPhase.term.iri.partition("#")[0] + "#",
                TermSource.PREDICTIVE_MANIPULATION,
            ),
        ],
    )
    def test_a_term_is_attributed_to_its_ontology(self, backing, namespace, source):
        assert backing.source_of(namespace + "Term") is source


# %% labels


class TestLabels:
    def test_a_fragment_node_shows_its_class_under_its_label(
        self, backing, projection, view
    ):
        phase = view.root.defines_phase[0]
        node = node_of(projection, phase)
        assert backing.node_label(node) == (
            f"{node.label}\n{backing.prefixed(type(phase).term.iri)}"
        )

    def test_a_node_without_a_class_keeps_its_label(self, backing, projection):
        group = next(
            node
            for node in projection.nodes
            if node.kind is HypothesisGraphNodeKind.GROUP
        )
        assert backing.node_label(group) == group.label

    def test_a_stated_link_shows_its_property_under_its_field(
        self, backing, projection
    ):
        relation = next(
            relation for relation in projection.relations if relation.property_iri
        )
        assert backing.relation_label(relation) == (
            f"{relation.predicate}\n{backing.prefixed(relation.property_iri)}"
        )

    def test_a_drawn_link_is_marked_as_drawn(self, backing, projection):
        relation = next(
            relation for relation in projection.relations if not relation.property_iri
        )
        assert backing.relation_label(relation) == (
            f"{relation.predicate}\n{DRAWN_LINK_NOTE}"
        )


# %% values


class TestBackedValues:
    def test_a_concept_is_backed_by_its_ontology(self, backing):
        concept = ConceptReference(iri=CRAM_AICOR_NAMESPACE + "Pouring")
        value = backing.backed_value(concept)
        assert value.text == backing.prefixed(concept.iri)
        assert value.source is TermSource.CRAM_AICOR

    def test_an_enumerated_value_is_backed_by_its_individual(self, backing):
        member = next(iter(core.ComparisonOutcome))
        value = backing.backed_value(member)
        assert value.text == backing.prefixed(member.value)
        assert value.source is TermSource.PREDICTIVE_MANIPULATION

    def test_a_fragment_is_backed_by_its_class(self, backing, view):
        phase = view.root.defines_phase[0]
        value = backing.backed_value(phase)
        assert backing.prefixed(type(phase).term.iri) in value.text
        assert value.source is TermSource.PREDICTIVE_MANIPULATION

    def test_a_number_is_a_literal(self, backing):
        assert backing.backed_value(600.0).source is TermSource.LITERAL


# %% details


class TestNodeLines:
    def test_lines_name_the_class_and_every_stated_property(self, backing, view):
        flat = HypothesisGraphProjector().project(
            view, HypothesisGraphPresentation.COMPLETE
        )
        phase = view.root.defines_phase[0]
        lines = backing.node_lines(node_of(flat, phase))
        assert type(phase).term.iri in "\n".join(lines)
        stated = [
            relation.property_iri
            for relation in flat.relations
            if relation.source.source is phase
        ]
        assert stated
        for property_iri in stated:
            assert any(backing.prefixed(property_iri) in line for line in lines)

    def test_a_node_without_a_class_says_the_viewer_drew_it(self, backing, projection):
        group = next(
            node
            for node in projection.nodes
            if node.kind is HypothesisGraphNodeKind.GROUP
        )
        assert DRAWN_LINK_NOTE in "\n".join(backing.node_lines(group))
