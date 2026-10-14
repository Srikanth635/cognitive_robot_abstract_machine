"""
Tests of observing the calls between components from outside: the world and the
knowledge component report what they are asked and what they answer, without the
resolver asking them changing.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import pytest
from krrood.entity_query_language.factories import the, variable
from krrood.entity_query_language.query.match import Match
from typing_extensions import Any, List, Optional

from llm_reasoner.clients.tools import Tool
from llm_reasoner.knowledge.access import KnowledgeAccess
from llm_reasoner.observation.components import ObservedKnowledge, ObservedWorld
from llm_reasoner.observation.events import (
    Component,
    ComponentCalled,
    ComponentFailed,
    ComponentReturned,
)
from llm_reasoner.observation.sinks import MemorySink
from llm_reasoner.world.access import identifier_of
from llm_reasoner.world.exceptions import UnknownHandle
from llm_reasoner.world.view import WorldView

from .test_observed import ECHO, ToolAnsweringWithItsInput
from .world_mimics import ActionWithBodyAndValueSlots, WorldBody, body_with_geometry

# %% helpers

RECORD = "record-1"
"""
The identifier of the one record the knowledge mimic serves.
"""

CITATION = "https://example.org/fragment"
"""
What the knowledge mimic cites any fragment as.
"""


@dataclass
class KnowledgeServingOneRecord(KnowledgeAccess):
    """
    A knowledge component that serves one record for any request.
    """

    def serve(self, query: Match, instruction: str = "") -> str:
        return RECORD

    def tools(self, record_id: str) -> List[Tool]:
        return [ToolAnsweringWithItsInput()]

    def citation(self, record_id: str, name: str) -> Optional[str]:
        return CITATION

    def instances_of(self, kind: type) -> list:
        return [object(), object()]

    def is_vocabulary_class(self, kind: type) -> bool:
        return True


@dataclass
class KnowledgeThatFails(KnowledgeServingOneRecord):
    """
    A knowledge component that raises when asked to serve.
    """

    def serve(self, query: Match, instruction: str = "") -> str:
        raise RuntimeError("no hypothesis")


def tilt_query() -> Match:
    """
    :return: A query for a tilting action whose body and angle are left open.
    """
    return the(ActionWithBodyAndValueSlots)(source=variable(WorldBody), angle=...)


def watching(world: WorldView):
    """
    :return: The world observed by a memory sink, which is returned beside it.
    """
    sink = MemorySink()
    return ObservedWorld(world=world, sink=sink), sink


def informed(knowledge: KnowledgeAccess):
    """
    :return: The knowledge observed by a memory sink, which is returned beside it.
    """
    sink = MemorySink()
    return ObservedKnowledge(knowledge=knowledge, sink=sink), sink


# %% the world


class TestObservedWorld:
    def test_it_answers_as_the_world_it_wraps(self):
        view = WorldView()
        world, _ = watching(view)
        body = body_with_geometry()
        handle = world.handle_of(body)
        assert handle == view.handle_of(body)
        assert world.individual_of(handle) is body
        assert world.is_live(handle) is True
        assert world.holders(handle) == view.holders(handle)
        assert [tool.definition.name for tool in world.tools()] == [
            tool.definition.name for tool in view.tools()
        ]

    def test_it_fulfils_the_version_of_the_world_it_wraps(self):
        view = WorldView()
        world, _ = watching(view)
        assert world.version == view.version

    def test_a_call_is_reported_with_its_operation_and_what_it_returned(self):
        world, sink = watching(WorldView())
        body = body_with_geometry()
        handle = world.handle_of(body)
        called, returned = sink.events
        assert (type(called), type(returned)) == (ComponentCalled, ComponentReturned)
        assert called.operation == "handle_of"
        assert called.source is Component.RESOLVER
        assert called.target is Component.WORLD_INJECTOR
        assert called.arguments == {"individual": type(body).__name__}
        assert returned.call == called.call
        assert returned.result == handle

    def test_the_individual_a_handle_names_is_reported_by_its_kind(self):
        world, sink = watching(WorldView())
        body = body_with_geometry()
        handle = world.handle_of(body)
        sink.events.clear()
        world.individual_of(handle)
        assert sink.events[1].result == type(body).__name__

    def test_what_holds_an_individual_is_reported_as_data(self):
        world, sink = watching(WorldView())
        body = body_with_geometry()
        handle = world.handle_of(body)
        holders = world.holders(handle)
        assert sink.events[-1].result == [asdict(holder) for holder in holders]

    def test_the_tools_offered_are_reported_by_name(self):
        world, sink = watching(WorldView())
        tools = world.tools()
        assert sink.events[1].result == [tool.definition.name for tool in tools]

    def test_a_question_that_fails_is_reported_and_fails_as_before(self):
        world, sink = watching(WorldView())
        with pytest.raises(UnknownHandle):
            world.individual_of("WorldBody#999")
        called, failed = sink.events
        assert type(failed) is ComponentFailed
        assert failed.call == called.call
        assert failed.error_type == UnknownHandle.__name__


# %% the knowledge component


class TestObservedKnowledge:
    def test_it_answers_as_the_knowledge_it_wraps(self):
        knowledge, _ = informed(KnowledgeServingOneRecord())
        assert knowledge.serve(tilt_query(), "tilt") == RECORD
        assert [tool.definition.name for tool in knowledge.tools(RECORD)] == [ECHO]
        assert knowledge.citation(RECORD, "phase") == CITATION
        assert len(knowledge.instances_of(WorldBody)) == 2
        assert knowledge.is_vocabulary_class(WorldBody) is True

    def test_it_fulfils_the_version_of_the_knowledge_it_wraps(self):
        wrapped = KnowledgeServingOneRecord()
        knowledge, _ = informed(wrapped)
        assert knowledge.version == wrapped.version

    def test_serving_is_reported_with_the_kind_asked_for_and_the_record_served(self):
        knowledge, sink = informed(KnowledgeServingOneRecord())
        knowledge.serve(tilt_query(), "tilt")
        called, returned = sink.events
        assert called.operation == "serve"
        assert called.source is Component.RESOLVER
        assert called.target is Component.KNOWLEDGE
        assert called.arguments == {
            "query": identifier_of(ActionWithBodyAndValueSlots),
            "instruction": "tilt",
        }
        assert returned.result == RECORD

    def test_the_tools_offered_for_a_record_are_reported_by_name(self):
        knowledge, sink = informed(KnowledgeServingOneRecord())
        knowledge.tools(RECORD)
        called, returned = sink.events
        assert called.arguments == {"record": RECORD}
        assert returned.result == [ECHO]

    def test_a_citation_is_reported_with_the_fragment_and_what_cites_it(self):
        knowledge, sink = informed(KnowledgeServingOneRecord())
        knowledge.citation(RECORD, "phase")
        called, returned = sink.events
        assert called.arguments == {"record": RECORD, "name": "phase"}
        assert returned.result == CITATION

    def test_the_instances_of_a_class_are_reported_by_their_number(self):
        knowledge, sink = informed(KnowledgeServingOneRecord())
        knowledge.instances_of(WorldBody)
        called, returned = sink.events
        assert called.arguments == {"kind": identifier_of(WorldBody)}
        assert returned.result == 2

    def test_a_class_asked_about_is_reported_with_the_answer(self):
        knowledge, sink = informed(KnowledgeServingOneRecord())
        knowledge.is_vocabulary_class(WorldBody)
        called, returned = sink.events
        assert called.arguments == {"kind": identifier_of(WorldBody)}
        assert returned.result is True

    def test_serving_that_fails_is_reported_and_fails_as_before(self):
        knowledge, sink = informed(KnowledgeThatFails())
        with pytest.raises(RuntimeError, match="no hypothesis"):
            knowledge.serve(tilt_query())
        called, failed = sink.events
        assert type(failed) is ComponentFailed
        assert failed.call == called.call
        assert failed.error_type == RuntimeError.__name__
        assert failed.reason == "no hypothesis"
