"""
Tests of the system assembled in one place: the resolver reading a world that leaves out
the knowledge component's symbols, and consulting that component about every action.
"""

from __future__ import annotations

import pytest

from dataclasses import replace

from llm_reasoner.clients.base import ScriptedClient
from llm_reasoner.knowledge.access import (
    AcceptanceSuite,
    ActionPerspectiveFile,
    HypothesisBackend,
    HypothesisStore,
    KnowledgeService,
    pouring,
)
from llm_reasoner.observation.sinks import MemorySink
from llm_reasoner.reasoner import Reasoner
from llm_reasoner.resolver.context import TaskContextProvider

from .world_mimics import body_with_geometry

# %% helpers


def reasoner_storing_in(store: HypothesisStore) -> Reasoner:
    """
    :return: A reasoner whose knowledge component stores hypotheses in the store.
    """
    return Reasoner(
        client=ScriptedClient(),
        knowledge=KnowledgeService(
            backend=HypothesisBackend(
                client=ScriptedClient(),
                suite=AcceptanceSuite(rules=[]),
                store=store,
            )
        ),
    )


# %% the world


class TestWorld:
    def test_the_world_leaves_out_the_knowledge_components_symbols(self, tmp_path):
        reasoner = reasoner_storing_in(HypothesisStore(directory=tmp_path))
        structure = pouring.PouringPhaseStructure()
        assert all(
            individual is not structure
            for individual in reasoner.world.source.individuals()
        )

    def test_the_world_shows_the_symbols_of_the_world(self, tmp_path):
        reasoner = reasoner_storing_in(HypothesisStore(directory=tmp_path))
        body = body_with_geometry()
        assert any(
            individual is body for individual in reasoner.world.source.individuals()
        )


# %% the backend


class TestBackend:
    def test_the_backend_consults_the_knowledge_component(self, tmp_path):
        reasoner = reasoner_storing_in(HypothesisStore(directory=tmp_path))
        assert reasoner.backend().knowledge.service is reasoner.knowledge

    def test_the_backend_names_candidates_by_handles_in_the_world(self, tmp_path):
        reasoner = reasoner_storing_in(HypothesisStore(directory=tmp_path))
        assert reasoner.backend().referent_naming.world is reasoner.world

    def test_the_backend_takes_further_settings(self, tmp_path):
        reasoner = reasoner_storing_in(HypothesisStore(directory=tmp_path))
        task = TaskContextProvider(goal="Pour until the bowl is half full.")
        assert reasoner.backend(context_provider=task).context_provider is task


# %% the perspectives consulted


class TestPerspectives:
    def test_by_default_no_perspective_is_consulted(self, tmp_path):
        reasoner = reasoner_storing_in(HypothesisStore(directory=tmp_path))
        assert reasoner.backend().knowledge.perspectives == ()

    def test_the_perspectives_named_are_consulted(self, tmp_path):
        reasoner = replace(
            reasoner_storing_in(HypothesisStore(directory=tmp_path)),
            perspectives=(ActionPerspectiveFile.ACTION_FRAME,),
        )
        assert reasoner.backend().knowledge.perspectives == (
            ActionPerspectiveFile.ACTION_FRAME,
        )

    def test_by_default_the_phase_structure_is_consulted(self, tmp_path):
        reasoner = reasoner_storing_in(HypothesisStore(directory=tmp_path))
        assert reasoner.backend().knowledge.consults_phase_structure is True

    @pytest.mark.parametrize("reported", [False, True])
    def test_the_phase_structure_can_be_left_out(self, tmp_path, reported):
        reasoner = replace(
            reasoner_storing_in(HypothesisStore(directory=tmp_path)),
            consults_phase_structure=False,
            sink=MemorySink() if reported else None,
        )
        assert reasoner.backend().knowledge.consults_phase_structure is False

    def test_the_perspectives_named_are_consulted_when_reported_too(self, tmp_path):
        reasoner = replace(
            reasoner_storing_in(HypothesisStore(directory=tmp_path)),
            perspectives=(ActionPerspectiveFile.ACTION_FRAME,),
            sink=MemorySink(),
        )
        assert reasoner.backend().knowledge.perspectives == (
            ActionPerspectiveFile.ACTION_FRAME,
        )
