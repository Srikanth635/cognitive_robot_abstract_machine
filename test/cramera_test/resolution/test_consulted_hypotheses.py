"""
Tests of the hypotheses a resolution consulted beside the phase structure: the action
core's frame and each perspective on it, shown in the Hypothesis Graph like the phase
structure.
"""

from __future__ import annotations

import json
from dataclasses import replace

import pytest

pytest.importorskip("krrood", reason="EQL requires krrood")

from llm_reasoner.knowledge.access import (  # noqa: E402
    HypothesisStore,
    SyntheticHypothesisRelation,
)

from llm_reasoner.resolver.grounds import Ground, GroundKey  # noqa: E402

from cramera.resolution import hypothesis_knowledge  # noqa: E402
from cramera.resolution.views import view_payload  # noqa: E402
from cramera.resolution.hypothesis_knowledge import build_projection  # noqa: E402
from cramera.resolution.hypothesis_knowledge.ontology_backing import OntologyBacking  # noqa: E402

from .perspective_knowledge import consulted_knowledge  # noqa: E402
from .resolutions import resolver_replaying  # noqa: E402

# %% helpers


@pytest.fixture
def knowledge(tmp_path):
    return consulted_knowledge(tmp_path)


@pytest.fixture
def shown(knowledge):
    """
    The viewer's knowledge after a resolution that consulted the core and perspectives.
    """
    previous = hypothesis_knowledge.active()
    shown = hypothesis_knowledge.activate_resolution(
        knowledge.resolution(), knowledge.service
    )
    yield shown
    hypothesis_knowledge.activate(previous)


def roots_of(knowledge):
    return [
        knowledge.service.view(record_id).root
        for record_id in knowledge.consulted_record_ids
    ]


# %% the projection


class TestConsultedProjection:
    def test_the_phase_structure_stays_the_one_the_queries_range_over(
        self, shown, knowledge
    ):
        assert shown.view.record.record_id == hypothesis_knowledge.SAMPLE_RECORD_ID
        assert shown.view.root is shown.projection.root.source

    def test_every_consulted_hypothesis_is_kept_in_order(self, shown, knowledge):
        assert [view.record.record_id for view in shown.consulted] == (
            knowledge.consulted_record_ids
        )

    def test_the_instruction_is_still_the_phase_structures(self, shown):
        assert shown.projection.instruction.label == shown.view.record.instruction

    def test_each_consulted_root_hangs_off_the_instruction(self, shown, knowledge):
        described = {
            relation.target.identifier
            for relation in shown.projection.relations
            if relation.source is shown.projection.instruction
            and relation.predicate == SyntheticHypothesisRelation.DESCRIBES
        }
        for view in shown.consulted:
            assert view.citation_of(view.root).iri in described

    def test_a_consulted_root_is_titled_by_the_module_it_is_written_in(self, shown):
        by_identifier = {node.identifier: node for node in shown.projection.nodes}
        titles = [
            by_identifier[view.citation_of(view.root).iri].label
            for view in shown.consulted
        ]
        assert titles == [
            OntologyBacking().namespace_of(type(view.root).term.iri).title
            for view in shown.consulted
        ]
        assert len(set(titles)) == len(titles)

    def test_every_consulted_fragment_has_a_node(self, shown):
        identifiers = {node.identifier for node in shown.projection.nodes}
        for view in shown.consulted:
            for fragment in view.fragments:
                assert view.citation_of(fragment).iri in identifiers

    def test_a_consulted_fragment_is_found_by_its_object(self, shown):
        view = shown.consulted[0]
        slot = view.root.defines_slot[0]
        assert shown.node_identifier_for(slot) == view.citation_of(slot).iri

    def test_a_perspective_links_the_slots_of_the_core_it_shares(self, shown):
        core_view, *perspective_views = shown.consulted
        slot_iris = {
            core_view.citation_of(slot).iri for slot in core_view.root.defines_slot
        }
        reaching_slots = {
            relation.source.identifier
            for relation in shown.projection.relations
            if relation.target.identifier in slot_iris
        }
        for view in perspective_views:
            assert reaching_slots & {
                view.citation_of(fragment).iri for fragment in view.fragments
            }

    def test_the_phase_structures_nodes_and_links_are_not_disturbed(self, shown):
        alone = build_projection(shown.view)
        together = shown.projection
        identifiers = {node.identifier for node in together.nodes}
        assert {node.identifier for node in alone.nodes} <= identifiers
        links = {
            (relation.source.identifier, relation.predicate, relation.target.identifier)
            for relation in together.relations
        }
        assert {
            (relation.source.identifier, relation.predicate, relation.target.identifier)
            for relation in alone.relations
        } <= links

    def test_without_perspectives_only_the_phase_structure_is_shown(self, knowledge):
        alone = hypothesis_knowledge.HypothesisKnowledge.from_view(
            knowledge.service.view(hypothesis_knowledge.SAMPLE_RECORD_ID)
        )
        assert alone.consulted == []


class TestActivatingAResolution:
    def test_a_resolution_without_perspectives_shows_none(self, knowledge):
        resolution = replace(knowledge.resolution(), perspective_record_ids=[])
        previous = hypothesis_knowledge.active()
        try:
            shown = hypothesis_knowledge.activate_resolution(
                resolution, knowledge.service
            )
            assert shown.consulted == []
        finally:
            hypothesis_knowledge.activate(previous)

    def test_without_the_phase_structure_the_core_is_shown_first(self, knowledge):
        resolution = replace(knowledge.resolution(), record_id=None)
        previous = hypothesis_knowledge.active()
        try:
            shown = hypothesis_knowledge.activate_resolution(
                resolution, knowledge.service
            )
            core_record, *perspective_records = resolution.perspective_record_ids
            assert shown.view.record.record_id == core_record
            assert [view.record.record_id for view in shown.consulted] == (
                perspective_records
            )
        finally:
            hypothesis_knowledge.activate(previous)

    def test_without_the_phase_structure_every_consulted_fragment_has_a_node(
        self, knowledge
    ):
        resolution = replace(knowledge.resolution(), record_id=None)
        previous = hypothesis_knowledge.active()
        try:
            shown = hypothesis_knowledge.activate_resolution(
                resolution, knowledge.service
            )
            identifiers = {node.identifier for node in shown.projection.nodes}
            for view in (shown.view, *shown.consulted):
                for fragment in view.fragments:
                    assert view.citation_of(fragment).iri in identifiers
        finally:
            hypothesis_knowledge.activate(previous)

    @pytest.mark.parametrize(
        "tab", ["hypothesis", "ontology", "designator"]
    )
    def test_without_the_phase_structure_every_tab_still_answers(self, knowledge, tab):
        resolution = replace(knowledge.resolution(), record_id=None)
        previous = hypothesis_knowledge.active()
        try:
            hypothesis_knowledge.activate_resolution(resolution, knowledge.service)
            assert view_payload(tab, [])["ok"] is True
        finally:
            hypothesis_knowledge.activate(previous)


# %% the payload of the tab


class TestHypothesisGraphPayload:
    def test_every_edge_joins_two_nodes(self, shown):
        payload = view_payload("hypothesis", [])
        identifiers = {node["id"] for node in payload["nodes"]}
        assert all(
            edge["from"] in identifiers and edge["to"] in identifiers
            for edge in payload["edges"]
        )

    def test_node_identifiers_are_unique(self, shown):
        identifiers = [node["id"] for node in view_payload("hypothesis", [])["nodes"]]
        assert len(identifiers) == len(set(identifiers))

    def test_the_payload_has_a_node_for_each_consulted_root_with_details(self, shown):
        payload = view_payload("hypothesis", [])
        identifiers = {node["id"] for node in payload["nodes"]}
        for view in shown.consulted:
            iri = view.citation_of(view.root).iri
            assert iri in identifiers
            assert payload["details"][iri]["lines"]

    def test_every_node_is_reachable_from_the_instruction(self, shown):
        payload = view_payload("hypothesis", [])
        children = {}
        for edge in payload["edges"]:
            children.setdefault(edge["from"], []).append(edge["to"])
        seen, pending = set(), [payload["root"]]
        while pending:
            current = pending.pop()
            if current in seen:
                continue
            seen.add(current)
            pending.extend(children.get(current, []))
        assert seen == {node["id"] for node in payload["nodes"]}


# %% the grounds of an answer


class TestGroundsInConsultedFragments:
    def test_a_ground_citing_a_consulted_fragment_is_named_by_its_fragment(
        self, knowledge, tmp_path
    ):
        resolver = resolver_replaying(
            "objects_given",
            HypothesisStore(directory=tmp_path / "store"),
            tmp_path / "runs",
        )
        for view in (knowledge.service.view(r) for r in knowledge.consulted_record_ids):
            citation = view.citation_of(view.root)
            shown_ground = resolver.shown(
                Ground(GroundKey.KNOWLEDGE, citation.iri), view=None
            )
            assert shown_ground.name == citation.fragment_name
            assert shown_ground.identifier == citation.iri

    def test_every_ground_of_the_resolution_is_a_node_of_the_hypothesis_graph(
        self, shown, knowledge
    ):
        nodes = {node["id"] for node in view_payload("hypothesis", [])["nodes"]}
        (action,) = shown.resolution.actions
        cited = [
            ground.identifier
            for ground in action.grounds
            if ground.source is GroundKey.KNOWLEDGE and ground.identifier
        ]
        assert len(cited) >= len(knowledge.consulted_record_ids)
        assert set(cited) <= nodes

    def test_every_ground_of_the_resolution_is_listed_in_the_designator(self, shown):
        designator = view_payload("designator", [])["designator"]
        (action,) = shown.resolution.actions
        text = json.dumps(designator)
        for ground in action.grounds:
            assert ground.name in text


class TestNodeDetails:
    def test_a_titled_root_still_gives_what_the_fragment_says(self, shown):
        view = shown.consulted[0]
        details = view_payload("hypothesis", [])["details"]
        lines = details[view.citation_of(view.root).iri]["lines"]
        assert f"says: {view.root.label}" in lines
