"""
Tests of resolving an underspecified krrood query with a language model: the prompt
built from the query, and the answer turned back into a resolved instance.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from krrood.entity_query_language.factories import the, variable
from krrood.entity_query_language.verbalization.rendering.source_documentation import (
    _attribute_docstrings,
)

from llm_reasoner.clients.base import ScriptedClient
from llm_reasoner.resolver.backend import LanguageModelBackend
from llm_reasoner.resolver.context import (
    PromptContextProvider,
    SchemaOnlyContextProvider,
    StructuralCandidateRenderer,
    TaskContextProvider,
    context_sections,
)
from llm_reasoner.resolver.prompts import PromptBuilder
from llm_reasoner.resolver.reflection import reflect_slots

from .individual_world import ActionWithBodyAndValueSlots, WorldBody, body_with_geometry

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


def prompt_for(query, context_provider: PromptContextProvider) -> str:
    return PromptBuilder(renderer=StructuralCandidateRenderer()).build(
        query,
        reflect_slots(query),
        context_sections(context_provider, query),
        number_of_candidates=1,
    )


# %% the prompt


class TestPrompt:
    def test_every_open_slot_is_asked_for_with_its_docstring(self):
        body = body_with_geometry()
        prompt = prompt_for(tilt_query(), SchemaOnlyContextProvider())
        meanings = _attribute_docstrings(ActionWithBodyAndValueSlots)
        assert f'"{meanings["source"]}"' in prompt
        assert f'"{meanings["angle"]}"' in prompt
        assert body.name in prompt

    def test_the_task_is_stated_only_when_the_caller_gives_one(self):
        task = TaskContextProvider(goal="Pour slowly; the glass is nearly full.")
        assert task.goal in prompt_for(tilt_query(), task)
        assert f"## {task.title}" not in prompt_for(
            tilt_query(), SchemaOnlyContextProvider()
        )


# %% resolution


class TestResolution:
    def test_the_answer_chooses_the_body_and_sets_the_angle(self):
        body = body_with_geometry()
        (expected,) = json.loads(answer("body_and_angle_chosen.json"))
        backend = LanguageModelBackend(
            client=ScriptedClient(responses=[answer("body_and_angle_chosen.json")])
        )
        (action,) = tilt_query().evaluate(backend=backend)
        assert action.source is body
        assert action.angle == expected["ActionWithBodyAndValueSlots.angle"]

    def test_an_unusable_answer_is_sent_back_for_repair_once(self):
        body = body_with_geometry()
        backend = LanguageModelBackend(
            client=ScriptedClient(
                responses=[
                    answer("prose_instead_of_json.txt"),
                    answer("body_and_angle_chosen.json"),
                ]
            )
        )
        (action,) = tilt_query().evaluate(backend=backend)
        assert action.source is body
        assert [exchange.attempt for exchange in backend.log] == [0, 1]
