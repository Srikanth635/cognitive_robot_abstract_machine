"""
Tests of reading admitted hypotheses: the questions a view of one stored hypothesis
answers, the service generating or finding hypotheses and resolving citations, and the
projection of a hypothesis into a graph.
"""

from __future__ import annotations

from dataclasses import fields

import pytest
import rustworkx as rx

from krrood.entity_query_language.factories import the

from llm_reasoner.knowledge.acceptance.rules import AcceptanceSuite
from llm_reasoner.knowledge.storage.citation import Citation, NotAHypothesisCitation
from llm_reasoner.knowledge.representation.rdf import HYPOTHESIS_NAMESPACE_ROOT
from llm_reasoner.knowledge.storage.store import HypothesisStore
from llm_reasoner.clients.base import ScriptedClient
from llm_reasoner.knowledge.generation.backend import HypothesisBackend
from llm_reasoner.knowledge.ontologies.terms import is_inferred
from llm_reasoner.knowledge.hypotheses.graph import (
    HypothesisGraph,
    HypothesisGraphNodeKind,
    HypothesisGraphPresentation,
    HypothesisGraphProjector,
    SyntheticHypothesisRelation,
)
from llm_reasoner.knowledge.hypotheses.service import KnowledgeService, StoreRequired
from llm_reasoner.knowledge.hypotheses.view import HypothesisView, UnknownCitation
from llm_reasoner.knowledge.ontologies import CRAM_AICOR_NAMESPACE
from llm_reasoner.knowledge.ontologies.predictive_manipulation import (
    core,
    pouring,
    prehension,
)
from llm_reasoner.knowledge.ontologies.terms import (
    ConceptReference,
    DescriptionEntity,
    PropertyTerm,
)

from .test_generation import GenerationCase

INSTRUCTION = "pour milk from the carton into the bowl"
"""
The instruction the scripted hypotheses answer.
"""

POURING = ConceptReference(iri=CRAM_AICOR_NAMESPACE + "Pouring")
"""
The task the requested structures decompose.
"""

# %% helpers


def pouring_query() -> object:
    """
    Return a query for one phase structure of the pouring task, all else open.
    """
    return the(core.ActionPhaseStructure)(decomposes_task=POURING)


def service_answering(
    store: HypothesisStore, *cases: GenerationCase
) -> KnowledgeService:
    """
    Return a service whose model gives the scripted answers in turn, admitted without
    acceptance rules.
    """
    return KnowledgeService(
        backend=HypothesisBackend(
            client=ScriptedClient(responses=[case.text for case in cases]),
            suite=AcceptanceSuite(rules=[]),
            store=store,
        )
    )


@pytest.fixture
def view(tmp_path) -> HypothesisView:
    """
    The stored view of the scripted lift-and-pour structure.
    """
    service = service_answering(
        HypothesisStore(directory=tmp_path), GenerationCase.LIFT_AND_POUR
    )
    return service.view_for(pouring_query(), INSTRUCTION)


def names(view: HypothesisView, fragments: list[DescriptionEntity]) -> list[str]:
    """
    Return the names the fragments have in their stored hypothesis.
    """
    return [view.citation_of(fragment).fragment_name for fragment in fragments]


def fragment_named(view: HypothesisView, name: str) -> DescriptionEntity:
    """
    Return the fragment with the given name in the stored hypothesis.
    """
    (fragment,) = [
        fragment
        for fragment in view.fragments
        if view.citation_of(fragment).fragment_name == name
    ]
    return fragment


# %% view


class TestHypothesisView:
    """
    A view answers questions that cross several fragments of one stored hypothesis.
    """

    def test_root_is_the_fragment_of_the_requested_type(self, view):
        assert type(view.root) is pouring.PouringPhaseStructure

    def test_phases_are_ordered_by_their_transitions(self, view):
        assert names(view, view.phases()) == ["load", "lift", "tilt"]

    def test_next_phase_is_the_target_of_a_transition_from_it(self, view):
        assert names(view, view.next_phases(fragment_named(view, "lift"))) == ["tilt"]

    def test_last_phase_has_no_next_phase(self, view):
        assert view.next_phases(fragment_named(view, "tilt")) == []

    def test_requirements_are_selected_by_their_mode(self, view):
        lift = fragment_named(view, "lift")
        assert names(
            view, view.requirements_of(lift, core.ConditionMode.THROUGHOUT)
        ) == ["carton_upright_throughout"]
        assert view.requirements_of(lift, core.ConditionMode.AT_ENTRY) == []

    def test_world_assumptions_are_the_requirements_assumed_of_the_world(self, view):
        assert names(view, view.world_assumptions()) == ["carton_upright_throughout"]

    def test_failure_events_are_the_failures_the_structure_defines(self, view):
        assert names(view, view.failure_events()) == ["slip"]

    def test_corrections_at_an_event_come_from_its_control_points(self, view):
        liftoff = fragment_named(view, "liftoff")
        assert names(view, view.corrections_at(liftoff)) == ["early_liftoff_correction"]

    def test_force_dynamics_name_the_forces_at_each_event(self, view):
        (dynamics,) = view.force_dynamics()
        assert dynamics.event is fragment_named(view, "liftoff")
        assert dynamics.driving_force.has_force_kind is core.ForceKind.LOAD_FORCE
        assert dynamics.resisting_force.has_force_kind is core.ForceKind.WEIGHT
        assert dynamics.ended_forces == [fragment_named(view, "support_reaction")]
        assert dynamics.started_forces == []

    def test_gaze_fixations_are_those_serving_the_phase(self, view):
        tilt, lift = fragment_named(view, "tilt"), fragment_named(view, "lift")
        assert names(view, view.gaze_fixations_for(tilt)) == ["watch_carton"]
        assert view.gaze_fixations_for(lift) == []

    def test_force_couplings_are_those_active_in_the_phase(self, view):
        lift, tilt = fragment_named(view, "lift"), fragment_named(view, "tilt")
        assert names(view, view.force_couplings_in(lift)) == ["lift_coupling"]
        assert view.force_couplings_in(tilt) == []

    def test_citation_is_the_fragment_iri_in_its_stored_hypothesis(self, view):
        citation = view.citation_of(view.root)
        assert citation.iri == (
            f"{HYPOTHESIS_NAMESPACE_ROOT}{view.record.record_id}#pour"
        )
        assert Citation.from_iri(citation.iri) == citation

    def test_iri_outside_stored_hypotheses_is_not_a_citation(self):
        with pytest.raises(NotAHypothesisCitation):
            Citation.from_iri(CRAM_AICOR_NAMESPACE + "Pouring")


# %% service


class TestKnowledgeService:
    """
    The service answers a request from the store when it can, and generates otherwise.
    """

    def test_a_request_is_served_by_the_identifier_of_its_stored_hypothesis(
        self, tmp_path
    ):
        store = HypothesisStore(directory=tmp_path)
        service = service_answering(store, GenerationCase.LIFT_AND_POUR)
        identifier = service.serve(pouring_query(), INSTRUCTION)
        assert [record.record_id for record in store.records()] == [identifier]

    def test_a_repeated_request_is_served_without_the_model(self, tmp_path):
        service = service_answering(
            HypothesisStore(directory=tmp_path), GenerationCase.LIFT_AND_POUR
        )
        first = service.serve(pouring_query())
        second = service.serve(pouring_query())
        assert second == first
        assert len(service.backend.client.calls) == 1

    def test_generated_hypothesis_is_stored(self, tmp_path):
        store = HypothesisStore(directory=tmp_path)
        service = service_answering(store, GenerationCase.LIFT_AND_POUR)
        view = service.view_for(pouring_query(), INSTRUCTION)
        assert store.records() == [view.record]

    def test_repeated_request_is_answered_from_the_store(self, tmp_path):
        service = service_answering(
            HypothesisStore(directory=tmp_path), GenerationCase.LIFT_AND_POUR
        )
        first = service.view_for(pouring_query(), INSTRUCTION)
        second = service.view_for(pouring_query(), INSTRUCTION)
        assert second.record == first.record

    def test_cited_fragment_is_found_in_its_stored_hypothesis(self, tmp_path):
        service = service_answering(
            HypothesisStore(directory=tmp_path), GenerationCase.LIFT_AND_POUR
        )
        view = service.view_for(pouring_query(), INSTRUCTION)
        citation = view.citation_of(fragment_named(view, "lift"))
        cited = service.fragment_cited(citation)
        assert type(cited) is prehension.LiftPhase

    def test_citation_of_a_missing_fragment_is_refused(self, tmp_path):
        service = service_answering(
            HypothesisStore(directory=tmp_path), GenerationCase.LIFT_AND_POUR
        )
        view = service.view_for(pouring_query(), INSTRUCTION)
        with pytest.raises(UnknownCitation):
            service.fragment_cited(
                Citation(record_id=view.record.record_id, fragment_name="absent")
            )

    def test_backend_without_a_store_is_refused(self):
        with pytest.raises(StoreRequired):
            KnowledgeService(
                backend=HypothesisBackend(
                    client=ScriptedClient(), suite=AcceptanceSuite(rules=[])
                )
            )


# %% graph projection


def projected(
    view: HypothesisView,
    presentation: HypothesisGraphPresentation = HypothesisGraphPresentation.GROUPED,
) -> HypothesisGraph:
    """
    Return the projection of a view.
    """
    return HypothesisGraphProjector().project(view, presentation)


class TestHypothesisGraphProjector:
    """
    A hypothesis is drawn around its instruction, with its fragments linked by the
    properties they state.
    """

    def test_instruction_describes_the_root(self, view):
        graph = projected(view)
        instruction = graph.graph[graph.instruction_index]
        assert instruction.kind is HypothesisGraphNodeKind.INSTRUCTION
        assert instruction.label == view.record.instruction
        assert (
            graph.graph.get_edge_data(
                graph.instruction_index, graph.root_index
            ).predicate
            == SyntheticHypothesisRelation.DESCRIBES
        )

    def test_graph_names_its_instruction_and_root_nodes(self, view):
        graph = projected(view)
        assert graph.instruction is graph.graph[graph.instruction_index]
        assert graph.root.source is view.root

    def test_fragment_node_is_identified_by_its_citation(self, view):
        graph = projected(view)
        assert (
            graph.graph[graph.root_index].identifier == view.citation_of(view.root).iri
        )

    def test_grouped_root_reaches_its_fragments_through_groups(self, view):
        graph = projected(view)
        groups = [
            graph.graph[index]
            for index in graph.graph.successor_indices(graph.root_index)
        ]
        assert {group.kind for group in groups} == {HypothesisGraphNodeKind.GROUP}
        assert f"Phases ({len(view.root.defines_phase)})" in {
            group.label for group in groups
        }

    def test_complete_root_links_its_fragments_directly(self, view):
        graph = projected(view, HypothesisGraphPresentation.COMPLETE)
        kinds = {
            graph.graph[index].kind
            for index in graph.graph.successor_indices(graph.root_index)
        }
        assert kinds == {HypothesisGraphNodeKind.FRAGMENT}

    def test_fragment_link_is_labelled_with_its_field(self, view):
        graph = projected(view)
        lift_to_tilt = view.citation_of(fragment_named(view, "lift_to_tilt")).iri
        tilt = view.citation_of(fragment_named(view, "tilt")).iri
        assert {
            relation.predicate
            for relation in graph.relations
            if (relation.source.identifier, relation.target.identifier)
            == (lift_to_tilt, tilt)
        } == {"to_phase"}

    def test_fragment_node_names_its_ontology_class(self, view):
        graph = projected(view)
        assert graph.root.ontology_class == type(view.root).term.iri
        assert graph.instruction.ontology_class == ""

    def test_link_names_its_ontology_property(self, view):
        graph = projected(view)
        (to_phase,) = [
            relation
            for relation in graph.relations
            if relation.predicate == "to_phase"
            and relation.source.identifier
            == view.citation_of(fragment_named(view, "lift_to_tilt")).iri
        ]
        assert to_phase.property_iri == (
            PropertyTerm.of_field(core.PhaseTransition, "to_phase").iri
        )
        (describes,) = [
            relation
            for relation in graph.relations
            if relation.predicate == SyntheticHypothesisRelation.DESCRIBES
        ]
        assert describes.property_iri == ""

    def test_inferred_super_property_is_not_drawn(self, view):
        inferred = {
            declared.name
            for declared in fields(view.root)
            if is_inferred(type(view.root), declared.name)
        }
        assert inferred
        assert not inferred & {
            relation.predicate for relation in projected(view).relations
        }

    def test_every_fragment_is_reachable_from_the_instruction(self, view):
        graph = projected(view)
        reachable = rx.descendants(graph.graph, graph.instruction_index)
        fragment_indices = {
            index
            for index in graph.graph.node_indices()
            if graph.graph[index].kind
            in (HypothesisGraphNodeKind.FRAGMENT, HypothesisGraphNodeKind.ROOT)
            and index != graph.root_index
        }
        unreachable = {
            graph.graph[index].identifier for index in fragment_indices - reachable
        }
        assert unreachable == set()
