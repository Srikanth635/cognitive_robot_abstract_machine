"""
Tests of the resolver choosing world individuals through the world injector's tools:

candidates named by handle, listed when few and searched for when many, and a handle
naming no candidate sent back for repair.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from krrood.entity_query_language.factories import the, variable

from llm_reasoner.clients.base import ScriptedClient
from llm_reasoner.clients.tools import ScriptedToolRequest
from llm_reasoner.resolver.backend import LanguageModelBackend
from llm_reasoner.resolver.context import SchemaOnlyContextProvider, TaskContextProvider
from llm_reasoner.resolver.exceptions import AmbiguousReferent, NoFittingReferent
from llm_reasoner.resolver.referents import (
    EQUALLY_FITTING_RULE,
    LISTED_CANDIDATES,
    NO_FITTING_CANDIDATE_RULE,
    IntegerTokens,
    ReferentAnswerKey,
    WorldHandles,
)
from llm_reasoner.resolver.reflection import reflect_slots
from llm_reasoner.resolver.prompt_format import (
    AcceptedType,
    AcceptKey,
    CandidateKey,
    PromptSection,
    SlotKey,
)
from llm_reasoner.world.classes import identifier_of
from llm_reasoner.world.naming import HANDLE_SEPARATOR
from llm_reasoner.world.tools import WorldToolName
from llm_reasoner.world.view import WorldView

from .prompt_sections import bodies, slot_entry
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
        entry = WorldHandles(view=view).candidate_entry(body)
        (source,) = [
            slot
            for slot in reflect_slots(tilt_query())
            if slot.attribute_name == "source"
        ]
        candidates = slot_entry(prompt, source.path)[SlotKey.CANDIDATES]
        assert candidates[CandidateKey.LISTED] == [entry]
        assert any(
            type(annotation).__name__ in holding
            for holding in entry[CandidateKey.HELD_AS]
        )

    def test_a_handle_is_accepted_as_a_json_string(self):
        view = WorldView()
        body = body_with_geometry()
        prompt = world_backend(view, []).describe(tilt_query())
        (source,) = [
            slot
            for slot in reflect_slots(tilt_query())
            if slot.attribute_name == "source"
        ]
        (accepted,) = slot_entry(prompt, source.path)[SlotKey.ACCEPTS]
        assert accepted == WorldHandles(view=view).accepted
        assert accepted[AcceptKey.TYPE] == AcceptedType.STRING
        assert view.handles.handle_of(body) in prompt

    def test_many_candidates_are_counted_rather_than_listed(self):
        view = WorldView()
        bodies = [WorldBody(name=str(index)) for index in range(LISTED_CANDIDATES + 1)]
        query = tilt_query()
        prompt = world_backend(view, []).describe(query)
        (source,) = [
            slot for slot in reflect_slots(query) if slot.attribute_name == "source"
        ]
        candidates = slot_entry(prompt, source.path)[SlotKey.CANDIDATES]
        assert candidates[CandidateKey.COUNT] == len(source.domain)
        assert CandidateKey.LISTED not in candidates
        assert not any(view.handles.handle_of(body) in prompt for body in bodies)

    def test_the_rules_say_how_world_individuals_are_answered(self):
        view = WorldView()
        assert WorldHandles(view=view).answer_rule in world_backend(view, []).rules
        plain = LanguageModelBackend(client=ScriptedClient())
        assert IntegerTokens().answer_rule in plain.rules
        assert WorldHandles(view=view).answer_rule not in plain.rules

    def test_the_rules_show_no_handle_a_world_could_give_out(self):
        handle = re.compile(rf"\w{re.escape(HANDLE_SEPARATOR)}\d")
        assert not handle.search(world_backend(WorldView(), []).system_prompt)

    def test_the_inputs_name_the_sections_the_providers_write(self):
        system_prompt = LanguageModelBackend(client=ScriptedClient()).system_prompt
        (inputs,) = bodies(system_prompt, PromptSection.INPUTS)
        for provider in (TaskContextProvider(), SchemaOnlyContextProvider()):
            assert f"<{provider.section}>" in inputs

    def test_in_a_conversation_with_tools_the_answer_is_written_as_text(self):
        view = WorldView()
        with_tools = world_backend(view, [])
        plain = LanguageModelBackend(client=ScriptedClient())
        (rule,) = with_tools.completion.answer_rules
        assert rule in with_tools.rules
        assert rule not in plain.rules


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


# %% no fitting and equally fitting candidates


def stated_reason(name: str) -> str:
    """
    :return: The reason a scripted answer gives for its verdict on the source slot.
    """
    (element,) = json.loads(answer(name))
    verdict = element["ActionWithBodyAndValueSlots.source"]
    return (
        verdict.get(ReferentAnswerKey.NO_FITTING_CANDIDATE)
        or verdict[ReferentAnswerKey.WHY]
    )


class TestVerdicts:
    def test_no_fitting_candidate_is_reported_and_nothing_is_built(self):
        view = WorldView()
        body = body_with_geometry()
        backend = world_backend(view, [answer("source_has_no_fitting_candidate.json")])
        with pytest.raises(NoFittingReferent) as raised:
            list(tilt_query().evaluate(backend=backend))
        assert raised.value.slot_path == "ActionWithBodyAndValueSlots.source"
        assert raised.value.reason == stated_reason(
            "source_has_no_fitting_candidate.json"
        )
        assert len(backend.log) == 1
        assert view.holders(view.handles.handle_of(body)) == []

    def test_equally_fitting_candidates_are_reported_as_individuals(self):
        view = WorldView()
        first, second = WorldBody(name="first"), WorldBody(name="second")
        backend = world_backend(view, [answer("source_equally_fitting_by_handle.json")])
        with pytest.raises(AmbiguousReferent) as raised:
            list(tilt_query().evaluate(backend=backend))
        assert [
            candidate is body
            for candidate, body in zip(raised.value.candidates, [first, second])
        ] == [True, True]
        assert raised.value.reason == stated_reason(
            "source_equally_fitting_by_handle.json"
        )

    def test_an_unknown_handle_among_equally_fitting_ones_is_sent_back_for_repair(self):
        view = WorldView()
        body = body_with_geometry()
        backend = world_backend(
            view,
            [
                answer("source_equally_fitting_with_a_handle_never_given.json"),
                answer("source_by_handle.json"),
            ],
        )
        (action,) = tilt_query().evaluate(backend=backend)
        assert action.source is body
        assert [exchange.attempt for exchange in backend.log] == [0, 1]

    def test_the_rules_say_how_to_report_no_fitting_or_equally_fitting_candidates(self):
        for backend in (
            world_backend(WorldView(), []),
            LanguageModelBackend(client=ScriptedClient()),
        ):
            assert NO_FITTING_CANDIDATE_RULE in backend.rules
            assert EQUALLY_FITTING_RULE in backend.rules
