"""
Tests of projecting several stored hypotheses into one graph: a core hypothesis and the
perspectives on it, whose fragments link the core's slots and dimensions themselves.
"""

from __future__ import annotations

import pytest

from llm_reasoner.knowledge.hypotheses.graph import (
    HypothesisGraphNodeKind,
    HypothesisGraphPresentation,
    HypothesisGraphProjector,
    SyntheticHypothesisRelation,
)
from llm_reasoner.knowledge.storage.store import HypothesisStore

from .test_action_perspective_knowledge import (
    AnswerCase,
    answer_text as answer_in_cases,
    core_query,
    service_answering,
)
from .test_failures_knowledge import FailuresCase, failures_query_on


@pytest.fixture
def stored(tmp_path):
    """
    The views of the scripted core of pouring and of the failures on it.
    """
    service = service_answering(
        HypothesisStore(directory=tmp_path),
        answer_in_cases(AnswerCase.CORE),
        answer_in_cases(FailuresCase.FAILURES),
    )
    core_view = service.view(service.serve(core_query()))
    return core_view, service.view_for(failures_query_on(core_view))


class TestProjectingHypothesesTogether:
    def test_one_hypothesis_projects_as_it_does_alone(self, stored):
        core_view, _ = stored
        alone = HypothesisGraphProjector().project(core_view)
        together = HypothesisGraphProjector().project_together([core_view])
        assert [node.identifier for node in together.nodes] == [
            node.identifier for node in alone.nodes
        ]
        assert len(together.relations) == len(alone.relations)

    def test_the_first_hypothesis_gives_the_instruction(self, stored):
        core_view, failures_view = stored
        graph = HypothesisGraphProjector().project_together([core_view, failures_view])
        assert graph.instruction.label == core_view.record.instruction

    def test_every_fragment_of_every_hypothesis_is_a_node_once(self, stored):
        graph = HypothesisGraphProjector().project_together(list(stored))
        identifiers = [node.identifier for node in graph.nodes]
        assert len(identifiers) == len(set(identifiers))
        for view in stored:
            for fragment in view.fragments:
                assert view.citation_of(fragment).iri in identifiers

    def test_the_instruction_describes_the_root_of_each_hypothesis(self, stored):
        graph = HypothesisGraphProjector().project_together(list(stored))
        described = {
            relation.target.identifier
            for relation in graph.relations
            if relation.source is graph.instruction
            and relation.predicate == SyntheticHypothesisRelation.DESCRIBES
        }
        assert described == {view.citation_of(view.root).iri for view in stored}

    def test_a_root_is_a_root_node_in_its_own_hypothesis(self, stored):
        graph = HypothesisGraphProjector().project_together(list(stored))
        roots = {
            node.identifier
            for node in graph.nodes
            if node.kind is HypothesisGraphNodeKind.ROOT
        }
        assert roots == {view.citation_of(view.root).iri for view in stored}

    def test_a_perspective_fragment_links_the_core_slot_itself(self, stored):
        core_view, failures_view = stored
        graph = HypothesisGraphProjector().project_together(
            list(stored), HypothesisGraphPresentation.COMPLETE
        )
        slot_iris = {
            core_view.citation_of(slot).iri for slot in core_view.root.defines_slot
        }
        linked = {
            relation.target.identifier
            for relation in graph.relations
            if relation.predicate == "failure_concerns_slot"
        }
        assert linked and linked <= slot_iris

    def test_the_perspective_root_links_the_core_root(self, stored):
        core_view, failures_view = stored
        graph = HypothesisGraphProjector().project_together(
            list(stored), HypothesisGraphPresentation.COMPLETE
        )
        assert any(
            relation.source.identifier
            == failures_view.citation_of(failures_view.root).iri
            and relation.target.identifier == core_view.citation_of(core_view.root).iri
            for relation in graph.relations
        )
