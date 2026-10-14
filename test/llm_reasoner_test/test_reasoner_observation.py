"""
Tests of the system assembled to be watched: the resolver, the world and the model
reporting to one sink, each as the component it is.
"""

from __future__ import annotations

import json

from krrood.entity_query_language.factories import the, variable

from llm_reasoner.clients.base import ScriptedClient
from llm_reasoner.clients.tools import ScriptedToolRequest
from llm_reasoner.knowledge.access import (
    AcceptanceSuite,
    HypothesisBackend,
    HypothesisStore,
    KnowledgeService,
)
from llm_reasoner.knowledge.hypotheses.tools import KnowledgeTool
from llm_reasoner.observation.events import (
    AttemptStarted,
    Component,
    ComponentCalled,
    ModelAnswered,
    ModelAsked,
    ResolutionFinished,
    ResolutionStarted,
    ToolCalled,
    ToolReturned,
)
from llm_reasoner.observation.sinks import MemorySink, NullSink
from llm_reasoner.reasoner import Reasoner
from llm_reasoner.world.access import WorldToolName, identifier_of
from llm_reasoner.world.tools import WorldTool
from llm_reasoner.world.view import WorldView

from .world_mimics import ActionWithBodyAndValueSlots, WorldBody, body_with_geometry

# %% helpers


def knowledge_storing_in(store: HypothesisStore) -> KnowledgeService:
    """
    :return: A knowledge component that stores hypotheses in the store.
    """
    return KnowledgeService(
        backend=HypothesisBackend(
            client=ScriptedClient(), suite=AcceptanceSuite(rules=[]), store=store
        )
    )


def tilt_query():
    """
    :return: A query for a tilting action whose body and angle are both left open.
    """
    return the(ActionWithBodyAndValueSlots)(source=variable(WorldBody), angle=...)


# %% assembling


class TestAssembly:
    def test_a_reasoner_without_a_sink_wraps_nothing(self, tmp_path):
        reasoner = Reasoner(
            client=ScriptedClient(),
            knowledge=knowledge_storing_in(HypothesisStore(directory=tmp_path)),
        )
        backend = reasoner.backend()
        assert backend.client is reasoner.client
        assert backend.referent_naming.world is reasoner.world
        assert backend.knowledge.service is reasoner.knowledge
        assert isinstance(backend.observer, NullSink)

    def test_a_reasoner_with_a_sink_has_the_backend_report_to_it(self, tmp_path):
        sink = MemorySink()
        reasoner = Reasoner(
            client=ScriptedClient(),
            knowledge=knowledge_storing_in(HypothesisStore(directory=tmp_path)),
            sink=sink,
        )
        assert reasoner.backend().observer is sink

    def test_the_tools_of_the_world_and_of_the_knowledge_are_told_apart(self, tmp_path):
        reasoner = Reasoner(
            client=ScriptedClient(),
            knowledge=knowledge_storing_in(HypothesisStore(directory=tmp_path)),
            sink=MemorySink(),
        )
        owners = {
            owner.tool_type: owner.component
            for owner in reasoner.backend().client.tool_owners
        }
        assert owners == {
            WorldTool: Component.WORLD_INJECTOR,
            KnowledgeTool: Component.KNOWLEDGE,
        }


# %% a resolution


class TestResolution:
    def test_every_component_of_a_resolution_reports_to_the_sink(self, tmp_path):
        body = body_with_geometry()
        view = WorldView()
        answer = json.dumps(
            [
                {
                    f"{ActionWithBodyAndValueSlots.__name__}.source": view.handle_of(
                        body
                    ),
                    f"{ActionWithBodyAndValueSlots.__name__}.angle": 1.2,
                }
            ]
        )
        sink = MemorySink()
        reasoner = Reasoner(
            client=ScriptedClient(
                tool_turns=[
                    [
                        ScriptedToolRequest(
                            WorldToolName.FIND_INDIVIDUALS,
                            {"kind": identifier_of(WorldBody)},
                        )
                    ],
                    answer,
                ]
            ),
            knowledge=knowledge_storing_in(HypothesisStore(directory=tmp_path)),
            world=view,
            sink=sink,
        )
        (action,) = tilt_query().evaluate(backend=reasoner.backend())
        assert action.source is body
        kinds = [type(event) for event in sink.events]
        assert kinds[-1] is ResolutionFinished
        assert kinds.index(ResolutionStarted) < kinds.index(AttemptStarted)
        assert kinds.index(AttemptStarted) < kinds.index(ModelAsked)
        assert kinds.index(ModelAnswered) < kinds.index(ResolutionFinished)
        (called,) = [e for e in sink.events if type(e) is ToolCalled]
        assert called.owner is Component.WORLD_INJECTOR
        assert called.tool == WorldToolName.FIND_INDIVIDUALS
        assert ToolReturned in kinds
        asked = next(e for e in sink.events if type(e) is ModelAsked)
        assert asked.source is Component.RESOLVER
        assert any(
            type(e) is ComponentCalled and e.target is Component.WORLD_INJECTOR
            for e in sink.events
        )
