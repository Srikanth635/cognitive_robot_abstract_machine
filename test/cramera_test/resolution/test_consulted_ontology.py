"""
Tests of the ontology behind the hypotheses a resolution consulted: the action
perspective modules and the kinds of object, drawn, described and attributed like the
predictive manipulation modules.
"""

from __future__ import annotations

import pytest

pytest.importorskip("krrood", reason="EQL requires krrood")

from llm_reasoner.knowledge.access import (  # noqa: E402
    ActionPerspectiveFile,
    CRAM_AICOR_NAMESPACE,
    HAS_OPENING,
    OBJECT_KINDS_FILE,
    PredictiveManipulationOntologyFile,
)
from rdflib import RDFS, Graph, URIRef  # noqa: E402

from cramera.resolution import hypothesis_knowledge  # noqa: E402
from cramera.resolution.views import view_payload  # noqa: E402
from cramera.resolution.hypothesis_knowledge.ontology_backing import (  # noqa: E402
    OntologyBacking,
    TermSource,
)
from cramera.resolution.hypothesis_knowledge.ontology_class_graph import (  # noqa: E402
    OntologyClassGraph,
    OntologyText,
)

from .perspective_knowledge import consulted_knowledge  # noqa: E402

# %% the backing


class TestBackingOfTheActionPerspectives:
    @pytest.mark.parametrize("module", list(ActionPerspectiveFile))
    def test_a_term_of_each_module_is_attributed_to_the_action_perspectives(
        self, module
    ):
        assert OntologyBacking().source_of(module.namespace + "Anything") is (
            TermSource.ACTION_PERSPECTIVE
        )

    @pytest.mark.parametrize("module", list(ActionPerspectiveFile))
    def test_a_term_of_each_module_is_written_with_the_modules_name(self, module):
        assert OntologyBacking().prefixed(module.namespace + "Anything") == (
            f"{module.value.removesuffix('.ttl')}:Anything"
        )

    def test_a_term_of_the_predictive_manipulation_is_still_attributed_to_it(self):
        assert OntologyBacking().source_of(
            PredictiveManipulationOntologyFile.CORE.namespace + "Phase"
        ) is (TermSource.PREDICTIVE_MANIPULATION)

    def test_the_property_of_having_an_opening_is_attributed_to_the_object_kinds(
        self,
    ):
        assert OntologyBacking().source_of(str(HAS_OPENING)) is TermSource.OBJECT_KINDS

    def test_a_module_has_a_title_made_of_its_name(self):
        assert (
            OntologyBacking()
            .namespace_of(ActionPerspectiveFile.IMAGE_SCHEMAS.namespace + "x")
            .title
            == "Image Schemas"
        )


# %% the statements the descriptions are read from


class TestOntologyText:
    def test_the_description_of_an_action_perspective_class_is_its_comment_in_the_module(
        self,
    ):
        from llm_reasoner.knowledge.ontologies.action_perspectives import action_core

        iri = action_core.SourceSlot.term.iri
        assert OntologyText.shipped().description(iri) == str(
            Graph()
            .parse(ActionPerspectiveFile.ACTION_CORE.path, format="turtle")
            .value(URIRef(iri), RDFS.comment)
        )

    def test_the_description_of_the_opening_property_is_its_comment(self):
        assert OntologyText.shipped().description(str(HAS_OPENING)) == str(
            Graph()
            .parse(OBJECT_KINDS_FILE, format="turtle")
            .value(HAS_OPENING, RDFS.comment)
        )

    def test_the_container_requires_an_opening(self):
        label = OntologyText.shipped().graph.value(HAS_OPENING, RDFS.label)
        assert OntologyText.shipped().requirements_of(
            CRAM_AICOR_NAMESPACE + "Container"
        ) == (f"{label}: true",)

    def test_a_class_with_no_restriction_requires_nothing(self):
        assert (
            OntologyText.shipped().requirements_of(
                CRAM_AICOR_NAMESPACE + "ContainedState"
            )
            == ()
        )


# %% the graph of the consulted hypotheses


@pytest.fixture
def knowledge(tmp_path):
    return consulted_knowledge(tmp_path)


@pytest.fixture
def shown(knowledge):
    previous = hypothesis_knowledge.active()
    shown = hypothesis_knowledge.activate_resolution(
        knowledge.resolution(), knowledge.service
    )
    yield shown
    hypothesis_knowledge.activate(previous)


class TestClassGraphOfConsultedHypotheses:
    def test_every_class_a_consulted_fragment_instantiates_is_a_node(self, shown):
        graph = OntologyClassGraph.of(shown.view, *shown.consulted)
        iris = {node.iri for node in graph.nodes}
        for view in shown.consulted:
            for fragment in view.fragments:
                assert type(fragment).term.iri in iris

    def test_the_classes_of_each_module_are_attributed_to_the_action_perspectives(
        self, shown
    ):
        graph = OntologyClassGraph.of(shown.view, *shown.consulted)
        by_iri = {node.iri: node for node in graph.nodes}
        for view in shown.consulted:
            assert by_iri[type(view.root).term.iri].source is (
                TermSource.ACTION_PERSPECTIVE
            )

    def test_the_phase_structure_classes_are_still_drawn(self, shown):
        graph = OntologyClassGraph.of(shown.view, *shown.consulted)
        iris = {node.iri for node in graph.nodes}
        for fragment in shown.view.fragments:
            assert type(fragment).term.iri in iris

    def test_a_perspective_class_is_described_by_its_module(self, shown):
        graph = OntologyClassGraph.of(shown.view, *shown.consulted)
        root = type(shown.consulted[-1].root)
        (node,) = [node for node in graph.nodes if node.iri == root.term.iri]
        assert node.description

    def test_a_class_of_the_phase_structure_graph_is_unchanged_without_perspectives(
        self, shown
    ):
        alone = OntologyClassGraph.of(shown.view)
        together = OntologyClassGraph.of(shown.view, *shown.consulted)
        assert {node.iri for node in alone.nodes} <= {
            node.iri for node in together.nodes
        }

    def test_every_link_joins_two_nodes_of_the_graph(self, shown):
        graph = OntologyClassGraph.of(shown.view, *shown.consulted)
        iris = {node.iri for node in graph.nodes}
        assert all(edge.source in iris and edge.target in iris for edge in graph.edges)

    def test_the_instances_of_a_class_are_counted_across_the_hypotheses(self, shown):
        graph = OntologyClassGraph.of(shown.view, *shown.consulted)
        by_iri = {node.iri: node for node in graph.nodes}
        slot = type(shown.consulted[0].root.defines_slot[0])
        expected = sum(
            isinstance(fragment, slot) and type(fragment) is slot
            for view in (shown.view, *shown.consulted)
            for fragment in view.fragments
        )
        assert by_iri[slot.term.iri].instance_count == expected


class TestOntologyPayload:
    def test_the_container_cited_by_the_frame_requires_an_opening(self, shown):
        payload = view_payload("ontology", [])
        container = payload["details"][CRAM_AICOR_NAMESPACE + "Container"]
        assert container["requirements"] == list(
            OntologyText.shipped().requirements_of(CRAM_AICOR_NAMESPACE + "Container")
        )

    def test_the_payload_has_the_classes_of_each_consulted_hypothesis(self, shown):
        payload = view_payload("ontology", [])
        for view in shown.consulted:
            assert type(view.root).term.iri in payload["details"]

    def test_the_source_of_a_perspective_class_is_named_in_its_details(self, shown):
        payload = view_payload("ontology", [])
        root = type(shown.consulted[0].root)
        assert payload["details"][root.term.iri]["source"] == (
            TermSource.ACTION_PERSPECTIVE.value
        )
