"""
Tests of resolving an underspecified krrood query with a language model: the prompt
built from the query, and the answer turned back into a resolved instance.
"""

from __future__ import annotations

import dataclasses
import inspect
import json
from pathlib import Path

import pytest

from krrood.entity_query_language.factories import the, variable
from krrood.entity_query_language.predicate import Symbol
from krrood.entity_query_language.verbalization.pipeline import verbalize_expression
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
from llm_reasoner.resolver.exceptions import (
    AmbiguousReferent,
    NoFittingReferent,
    NoValidCandidates,
    RepairBudgetExhausted,
    UnsatisfiableRequest,
)
from llm_reasoner.resolver.prompts import UNSATISFIABLE_RULE, PromptBuilder
from llm_reasoner.resolver.reflection import reflect_slots
from llm_reasoner.resolver.referents import NO_FITTING_CANDIDATE_RULE
from llm_reasoner.world.view import WorldView

from .world_mimics import (
    ActionInheritingItsPlan,
    ActionPlannedByItsParent,
    ActionWithADefaultChoice,
    ActionWithBodyAndValueSlots,
    Side,
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

    def test_a_default_is_shown_as_an_answer_would_write_it(self):
        body_with_geometry()
        query = the(ActionWithADefaultChoice)(source=variable(WorldBody), side=...)
        (side,) = [
            declared
            for declared in dataclasses.fields(ActionWithADefaultChoice)
            if declared.name == "side"
        ]
        prompt = prompt_for(query, SchemaOnlyContextProvider())
        assert json.dumps(side.default.name) in prompt
        assert repr(side.default) not in prompt

    def test_a_given_value_is_shown_as_an_answer_would_write_it(self):
        body_with_geometry()
        query = the(ActionWithADefaultChoice)(
            source=variable(WorldBody), side=Side.LEFT
        )
        prompt = prompt_for(query, SchemaOnlyContextProvider())
        assert json.dumps(Side.LEFT.name) in prompt
        assert repr(Side.LEFT) not in prompt

    def test_a_slot_whose_field_defaults_to_nothing_offers_null(self):
        body_with_geometry()
        query = the(ActionWithADefaultChoice)(source=variable(WorldBody), depth=...)
        (depth,) = [
            slot for slot in reflect_slots(query) if slot.path.endswith("depth")
        ]
        assert PromptBuilder.answer_form(None) in PromptBuilder.value_hint(depth)

    def test_the_refusal_rule_covers_what_the_task_asks_for(self):
        task = TaskContextProvider(goal="Pour the milk into the bowl.")
        assert f'"{task.title}"' in NO_FITTING_CANDIDATE_RULE


class TestSourceShown:
    def test_the_source_of_a_parent_in_the_same_module_is_shown_too(self):
        body_with_geometry()
        query = the(ActionInheritingItsPlan)(source=variable(WorldBody))
        prompt = LanguageModelBackend(client=ScriptedClient()).describe(query)
        assert inspect.getsource(ActionPlannedByItsParent).rstrip() in prompt
        assert inspect.getsource(Symbol).rstrip() not in prompt

    def test_the_backend_shows_the_full_source_of_the_action_by_default(self):
        body_with_geometry()
        prompt = LanguageModelBackend(client=ScriptedClient()).describe(tilt_query())
        assert inspect.getsource(ActionWithBodyAndValueSlots).rstrip() in prompt


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

    def test_slots_whose_fields_default_to_nothing_may_be_left_empty(self):
        body = body_with_geometry()
        backend = LanguageModelBackend(
            client=ScriptedClient(
                responses=[answer("depth_and_support_left_empty.json")]
            )
        )
        query = the(ActionWithADefaultChoice)(
            source=variable(WorldBody), depth=..., support=variable(WorldBody)
        )
        (action,) = query.evaluate(backend=backend)
        assert action.source is body
        assert (action.depth, action.support) == (None, None)
        assert [exchange.attempt for exchange in backend.log] == [0]

    def test_an_answer_breaking_a_condition_is_sent_back_naming_it(self):
        body = body_with_geometry()
        (expected,) = json.loads(answer("body_and_small_angle_chosen.json"))
        backend = LanguageModelBackend(
            client=ScriptedClient(
                responses=[
                    answer("body_and_angle_chosen.json"),
                    answer("body_and_small_angle_chosen.json"),
                ]
            )
        )
        query = tilt_query()
        condition = (
            query.variable.angle < expected["ActionWithBodyAndValueSlots.angle"] + 0.1
        )
        query.where(condition)
        (action,) = query.evaluate(backend=backend)
        assert action.source is body
        assert action.angle == expected["ActionWithBodyAndValueSlots.angle"]
        assert verbalize_expression(condition) in backend.log[1].user

    def test_answers_breaking_a_condition_every_time_end_the_resolution(self):
        body = body_with_geometry()
        backend = LanguageModelBackend(
            client=ScriptedClient(responses=[answer("body_and_angle_chosen.json")] * 2)
        )
        query = tilt_query()
        query.where(query.variable.angle < 0.0)
        with pytest.raises(RepairBudgetExhausted) as raised:
            list(query.evaluate(backend=backend))
        assert isinstance(raised.value.last_error, NoValidCandidates)
        assert body.name in backend.log[1].user

    def test_an_answer_may_say_the_conditions_cannot_all_hold(self):
        body = body_with_geometry()
        backend = LanguageModelBackend(
            client=ScriptedClient(responses=[answer("request_unsatisfiable.json")])
        )
        query = tilt_query()
        query.where(query.variable.angle < 1.0, query.variable.angle > 2.0)
        with pytest.raises(UnsatisfiableRequest):
            list(query.evaluate(backend=backend))
        assert [exchange.attempt for exchange in backend.log] == [0]
        assert body.name in backend.log[0].user
        assert UNSATISFIABLE_RULE in backend.system_prompt

    def test_a_token_answer_may_say_no_candidate_fits(self):
        body = body_with_geometry()
        backend = LanguageModelBackend(
            client=ScriptedClient(
                responses=[answer("source_has_no_fitting_candidate.json")]
            )
        )
        with pytest.raises(NoFittingReferent):
            list(tilt_query().evaluate(backend=backend))
        view = WorldView()
        assert view.holders(view.handles.handle_of(body)) == []

    def test_a_token_answer_may_name_equally_fitting_candidates(self):
        first, second = WorldBody(name="first"), WorldBody(name="second")
        backend = LanguageModelBackend(
            client=ScriptedClient(
                responses=[answer("source_equally_fitting_by_token.json")]
            )
        )
        with pytest.raises(AmbiguousReferent) as raised:
            list(tilt_query().evaluate(backend=backend))
        assert [
            candidate is body
            for candidate, body in zip(raised.value.candidates, [first, second])
        ] == [True, True]
