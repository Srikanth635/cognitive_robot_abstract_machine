"""
Tests of slots an outcome check opens: the query decides which slots are asked for, a
check may hold back slots of the action the query leaves out, and a trial of an answer
may open them, after which the model answers for them too.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import pytest
from krrood.entity_query_language.factories import the, variable
from typing_extensions import Any, List, Sequence, Tuple

from llm_reasoner.clients.base import ScriptedClient
from llm_reasoner.resolver.backend import LanguageModelBackend
from llm_reasoner.resolver.exceptions import (
    MissingSlotInResponse,
    OutcomeFellShort,
    RepairBudgetExhausted,
    UnknownSlotInResponse,
)
from llm_reasoner.resolver.outcomes import OutcomeCheck
from llm_reasoner.resolver.prompt_format import PromptSection, SlotKey

from .prompt_sections import bodies
from .world_mimics import ActionWithAnOptionalOffset, WorldBody, body_with_geometry

CASES = Path(__file__).parent / "resolver_cases"
"""
The scripted model answers the tests replay.
"""

OFFSET = "offset"
"""
The field of the action the check holds back.
"""

# %% helpers


@dataclass
class SidewaysCheck(OutcomeCheck):
    """
    Holds back the offset of the action and, while it is still zero, refuses the answer
    saying the pouring ends beside its target and opening the offset.
    """

    withheld_slots: Tuple[str, ...] = (OFFSET,)
    """
    The fields of the action the check holds back.
    """

    opens_when_refusing: Tuple[str, ...] = (OFFSET,)
    """
    The fields a refusal opens.
    """

    tried: List[Any] = field(default_factory=list)
    """
    The actions the check was asked about, in order.
    """

    def check(self, instances: Sequence[Any]) -> None:
        (action,) = instances
        self.tried.append(action)
        if action.offset != 0.0:
            return
        raise OutcomeFellShort(
            outcome={"sideways_miss": 0.2}, opens=self.opens_when_refusing
        )


@dataclass
class AcceptingCheck(OutcomeCheck):
    """
    Holds back the offset of the action and accepts every answer.
    """

    withheld_slots: Tuple[str, ...] = (OFFSET,)
    """
    The fields of the action the check holds back.
    """

    def check(self, instances: Sequence[Any]) -> None:
        return


@pytest.fixture
def body() -> WorldBody:
    """
    The body the query can choose as the source, kept alive for the test.
    """
    return body_with_geometry()


def answer(name: str) -> str:
    return (CASES / name).read_text()


def offset_query():
    """
    :return: A query for the action whose body and angle are left open and whose offset
        it does not mention.
    """
    return the(ActionWithAnOptionalOffset)(source=variable(WorldBody), angle=...)


def backend_answering(
    check: OutcomeCheck, *answers: str, max_repairs: int = 3
) -> LanguageModelBackend:
    """
    :return: A backend answering with the scripted answers in turn, each tried by the
        check.
    """
    return LanguageModelBackend(
        client=ScriptedClient(responses=[answer(name) for name in answers]),
        outcome_check=check,
        max_repairs=max_repairs,
    )


def offered_paths(prompt: str) -> List[str]:
    """
    :return: The paths of the slots a prompt asks the model to answer for.
    """
    (slots,) = bodies(prompt, PromptSection.SLOTS)
    return [entry[SlotKey.PATH] for entry in json.loads(slots)]


# %% what is asked for


class TestOfferingHeldBackSlots:
    def test_the_first_request_offers_only_the_slots_the_query_leaves_open(self, body):
        backend = backend_answering(
            SidewaysCheck(), "offset_withheld_answer.json", "offset_opened_answer.json"
        )
        list(offset_query().evaluate(backend=backend))
        assert offered_paths(backend.log[0].user) == [
            "ActionWithAnOptionalOffset.source",
            "ActionWithAnOptionalOffset.angle",
        ]

    def test_a_held_back_slot_keeps_the_actions_default_until_it_is_opened(self, body):
        backend = backend_answering(AcceptingCheck(), "offset_withheld_answer.json")
        (action,) = offset_query().evaluate(backend=backend)
        assert (
            action.offset
            == ActionWithAnOptionalOffset.__dataclass_fields__[OFFSET].default
        )

    def test_a_slot_the_query_states_is_not_held_back(self, body):
        stated = 0.3
        backend = backend_answering(AcceptingCheck(), "offset_withheld_answer.json")
        (action,) = the(ActionWithAnOptionalOffset)(
            source=variable(WorldBody), angle=..., offset=stated
        ).evaluate(backend=backend)
        assert action.offset == stated


# %% opening a slot


class TestOpeningASlot:
    def test_the_rejection_offers_the_slot_it_opens(self, body):
        backend = backend_answering(
            SidewaysCheck(), "offset_withheld_answer.json", "offset_opened_answer.json"
        )
        list(offset_query().evaluate(backend=backend))
        (opened,) = bodies(backend.log[1].user, PromptSection.OPENED_SLOTS)
        (entry,) = json.loads(opened)
        assert entry[SlotKey.PATH] == "ActionWithAnOptionalOffset.offset"
        assert entry[SlotKey.DEFAULT] == 0.0

    def test_the_rejection_states_the_outcome_and_what_it_opens(self, body):
        backend = backend_answering(
            SidewaysCheck(), "offset_withheld_answer.json", "offset_opened_answer.json"
        )
        list(offset_query().evaluate(backend=backend))
        (rejection,) = bodies(backend.log[1].user, PromptSection.REJECTION)
        assert json.loads(rejection) == {
            "outcome": {"sideways_miss": 0.2},
            "opens": [OFFSET],
        }

    def test_the_answer_after_it_may_state_the_slot_and_the_action_carries_it(
        self, body
    ):
        check = SidewaysCheck()
        backend = backend_answering(
            check, "offset_withheld_answer.json", "offset_opened_answer.json"
        )
        (action,) = offset_query().evaluate(backend=backend)
        assert action.offset == -0.2
        assert check.tried[-1] is action

    def test_an_answer_that_leaves_an_opened_slot_out_is_sent_back(self, body):
        backend = backend_answering(
            SidewaysCheck(),
            "offset_withheld_answer.json",
            "offset_withheld_answer.json",
            "offset_opened_answer.json",
        )
        list(offset_query().evaluate(backend=backend))
        assert f'kind="{MissingSlotInResponse.__name__}"' in backend.log[2].user

    def test_an_answer_naming_a_slot_before_it_is_opened_is_sent_back(self, body):
        backend = backend_answering(
            AcceptingCheck(), "offset_opened_answer.json", "offset_withheld_answer.json"
        )
        list(offset_query().evaluate(backend=backend))
        assert f'kind="{UnknownSlotInResponse.__name__}"' in backend.log[1].user

    def test_a_refusal_opening_nothing_changes_what_is_asked(self, body):
        backend = backend_answering(
            SidewaysCheck(opens_when_refusing=()),
            "offset_withheld_answer.json",
            "offset_withheld_answer.json",
            "offset_withheld_answer.json",
            "offset_withheld_answer.json",
        )
        with pytest.raises(RepairBudgetExhausted):
            list(offset_query().evaluate(backend=backend))
        assert not bodies(backend.log[1].user, PromptSection.OPENED_SLOTS)
