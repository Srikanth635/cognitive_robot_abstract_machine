"""
Tests of an answer the model did not finish: it is cut off at the token limit, the
resolver sends it back for repair as unusable, and gives up when every answer is cut off.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import pytest
from krrood.entity_query_language.factories import the, variable
from typing_extensions import List

from llm_reasoner.clients.base import LanguageModelClient, ScriptedClient
from llm_reasoner.clients.tools import ResponseTruncated
from llm_reasoner.resolver.backend import LanguageModelBackend
from llm_reasoner.resolver.exceptions import AnswerTruncated, RepairBudgetExhausted
from llm_reasoner.resolver.prompt_format import PromptSection

from .prompt_sections import bodies
from .world_mimics import ActionWithBodyAndValueSlots, WorldBody, body_with_geometry

CASES = Path(__file__).parent / "resolver_cases"
"""
The scripted model answers the tests replay.
"""

TOKEN_LIMIT = 1_000
"""
The most output tokens a request allowed in these tests.
"""

PARTIAL_ANSWER = "Fitting the two misses, the sideways error is about"
"""
What the model had written when it was cut off.
"""


@dataclass
class CutOffClient(LanguageModelClient):
    """
    Cuts off its first answers as the token limit does, then answers from a script.
    """

    cut_off: int
    """
    How many answers are cut off before the script answers.
    """

    script: ScriptedClient
    """
    Answers once the cut-offs are used up.
    """

    asked: List[str] = field(default_factory=list)
    """
    The prompts asked, in order.
    """

    def complete(self, system: str, user: str) -> str:
        self.asked.append(user)
        if len(self.asked) <= self.cut_off:
            raise ResponseTruncated(
                output_tokens=TOKEN_LIMIT,
                token_limit=TOKEN_LIMIT,
                partial_answer=PARTIAL_ANSWER,
            )
        return self.script.complete(system, user)


def tilt_query():
    """
    :return: A query for a tilting action whose body and angle are both left open.
    """
    return the(ActionWithBodyAndValueSlots)(source=variable(WorldBody), angle=...)


@pytest.fixture
def body():
    """
    A body of the world, kept alive for the test, for the query to choose.
    """
    return body_with_geometry()


def backend_cut_off(count: int, **settings) -> LanguageModelBackend:
    """
    :return: A backend whose first answers are cut off.
    """
    answer = (CASES / "body_and_angle_chosen.json").read_text()
    return LanguageModelBackend(
        client=CutOffClient(cut_off=count, script=ScriptedClient(responses=[answer])),
        **settings,
    )


class TestTruncatedAnswer:
    def test_an_answer_cut_off_is_sent_back_and_the_next_one_accepted(self, body):
        backend = backend_cut_off(1)
        (action,) = tilt_query().evaluate(backend=backend)
        assert action.source is body
        assert [exchange.attempt for exchange in backend.log] == [0, 1]

    def test_the_model_reads_what_it_had_written_and_that_it_was_cut_off(self, body):
        backend = backend_cut_off(1)
        list(tilt_query().evaluate(backend=backend))
        assert backend.log[0].raw_response == PARTIAL_ANSWER
        assert f'kind="{AnswerTruncated.__name__}"' in backend.log[1].user
        assert PARTIAL_ANSWER in backend.log[1].user

    def test_the_rejection_states_the_tokens_used_and_the_limit(self, body):
        backend = backend_cut_off(1)
        list(tilt_query().evaluate(backend=backend))
        (rejection,) = bodies(backend.log[1].user, PromptSection.REJECTION)
        assert json.loads(rejection) == {
            "output_tokens": TOKEN_LIMIT,
            "token_limit": TOKEN_LIMIT,
        }

    def test_answers_that_are_always_cut_off_exhaust_the_repairs(self, body):
        backend = backend_cut_off(10, max_repairs=1)
        with pytest.raises(RepairBudgetExhausted) as exhausted:
            list(tilt_query().evaluate(backend=backend))
        assert isinstance(exhausted.value.last_error, AnswerTruncated)
