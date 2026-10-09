"""
Tests of the resolver choosing world individuals through the world injector's tools:

candidates named by handle, listed when few and searched for when many, and a handle
naming no candidate sent back for repair.
"""

from __future__ import annotations

from pathlib import Path

from krrood.entity_query_language.factories import the, variable

from llm_reasoner.clients.base import ScriptedClient
from llm_reasoner.clients.tools import ScriptedToolRequest
from llm_reasoner.resolver.backend import LanguageModelBackend
from llm_reasoner.resolver.referents import (
    LISTED_CANDIDATES,
    IntegerTokens,
    WorldHandles,
)
from llm_reasoner.resolver.reflection import reflect_slots
from llm_reasoner.world.classes import identifier_of
from llm_reasoner.world.tools import WorldToolName
from llm_reasoner.world.view import WorldView

from .world_mimics import (
    ActionWithBodyAndValueSlots,
    VesselAnnotation,
    WorldBody,
    body_with_geometry,
)

CASES = Path(__file__).parent / "resolver_cases"
"""
The scripted model answers the tests replay.
"""

# %% helpers


def tilt_query():
    """
    :return: A query for a tilting action whose body and angle are both left open.
    """
    return the(ActionWithBodyAndValueSlots)(source=variable(WorldBody), angle=...)


def answer(name: str) -> str:
    return (CASES / name).read_text()


def search_for(kind: type) -> list:
    """
    :return: A scripted turn searching the world for individuals of a kind.
    """
    return [
        ScriptedToolRequest(
            WorldToolName.FIND_INDIVIDUALS, {"kind": identifier_of(kind)}
        )
    ]


def world_backend(view: WorldView, tool_turns: list) -> LanguageModelBackend:
    return LanguageModelBackend.with_world_tools(
        client=ScriptedClient(tool_turns=tool_turns), view=view
    )


# %% the prompt


class TestPrompt:
    def test_a_few_candidates_are_listed_by_handle_with_their_holders(self):
        view = WorldView()
        body = body_with_geometry()
        annotation = VesselAnnotation(body=body)
        prompt = world_backend(view, []).describe(tilt_query())
        assert WorldHandles(view=view).candidate_line(body) in prompt
        assert type(annotation).__name__ in WorldHandles(view=view).candidate_line(body)

    def test_many_candidates_are_counted_rather_than_listed(self):
        view = WorldView()
        bodies = [WorldBody(name=str(index)) for index in range(LISTED_CANDIDATES + 1)]
        query = tilt_query()
        prompt = world_backend(view, []).describe(query)
        (source,) = [
            slot for slot in reflect_slots(query) if slot.attribute_name == "source"
        ]
        assert WorldHandles(view=view).unlisted_line(source) in prompt
        assert not any(view.handles.handle_of(body) in prompt for body in bodies)

    def test_the_rules_say_how_world_individuals_are_answered(self):
        view = WorldView()
        assert (
            WorldHandles(view=view).answer_rule in world_backend(view, []).system_prompt
        )
        plain = LanguageModelBackend(client=ScriptedClient())
        assert IntegerTokens().answer_rule in plain.system_prompt
        assert WorldHandles(view=view).answer_rule not in plain.system_prompt


# %% resolution


class TestResolution:
    def test_a_body_found_with_the_tools_is_chosen_by_its_handle(self):
        view = WorldView()
        bodies = [WorldBody(name=str(index)) for index in range(LISTED_CANDIDATES + 1)]
        backend = world_backend(
            view, [search_for(WorldBody), answer("source_by_handle.json")]
        )
        (action,) = tilt_query().evaluate(backend=backend)
        assert action.source is view.handles.individual_of("WorldBody#1")
        assert any(action.source is body for body in bodies)
        (conversation,) = backend.conversations
        assert [call.name for call in conversation.calls] == [
            WorldToolName.FIND_INDIVIDUALS
        ]

    def test_a_handle_of_another_kind_is_sent_back_for_repair(self):
        view = WorldView()
        body = body_with_geometry()
        annotation = VesselAnnotation(body=body)
        backend = world_backend(
            view,
            [
                search_for(VesselAnnotation),
                answer("source_by_handle_of_an_annotation.json"),
                answer("source_by_handle.json"),
            ],
        )
        (action,) = tilt_query().evaluate(backend=backend)
        assert action.source is body
        assert view.handles.individual_of("VesselAnnotation#1") is annotation
        assert [exchange.attempt for exchange in backend.log] == [0, 1]

    def test_a_handle_never_given_out_is_sent_back_for_repair(self):
        view = WorldView()
        body = body_with_geometry()
        backend = world_backend(
            view,
            [
                answer("source_by_handle_never_given.json"),
                answer("source_by_handle.json"),
            ],
        )
        (action,) = tilt_query().evaluate(backend=backend)
        assert action.source is body
        assert len(backend.conversations) == 2
