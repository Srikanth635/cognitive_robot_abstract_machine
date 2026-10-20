"""
Tests of trying an answer before it is accepted: a check performs what the answer
constructs and refuses it when the outcome falls short, and the model reads that outcome
to answer again.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import pytest
from krrood.entity_query_language.factories import the, variable

from llm_reasoner.clients.base import ScriptedClient
from llm_reasoner.clients.tools import ScriptedToolRequest, Tool
from llm_reasoner.resolver.backend import LanguageModelBackend
from llm_reasoner.resolver.exceptions import OutcomeFellShort, RepairBudgetExhausted
from llm_reasoner.resolver.outcomes import OutcomeCheck, OutcomePrediction
from llm_reasoner.resolver.prompt_format import PromptRule, PromptSection, RuleName
from llm_reasoner.world.view import WorldView
from typing_extensions import Any, List, Sequence

from .prompt_sections import bodies
from .test_observed import ECHO, ToolAnsweringWithItsInput
from .world_mimics import ActionWithBodyAndValueSlots, WorldBody, body_with_geometry

CASES = Path(__file__).parent / "resolver_cases"
"""
The scripted model answers the tests replay.
"""

REQUIRED_ANGLE = 1.5
"""
The angle an action must tilt to for the check to accept it.
"""

# %% helpers


@dataclass
class MinimumAngleCheck(OutcomeCheck):
    """
    Accepts a tilting action only when it tilts at least to a required angle, stating
    the angle it reached otherwise.
    """

    required: float
    """
    The angle an action must reach.
    """

    tried: List[Any] = field(default_factory=list)
    """
    The actions the check was asked about, in order.
    """

    def check(self, instances: Sequence[Any]) -> None:
        self.tried.extend(instances)
        (action,) = instances
        if action.angle >= self.required:
            return
        raise OutcomeFellShort(
            outcome={"angle_reached": action.angle, "angle_required": self.required}
        )


def answer(name: str) -> str:
    return (CASES / name).read_text()


def tilt_query():
    """
    :return: A query for a tilting action whose body and angle are both left open.
    """
    return the(ActionWithBodyAndValueSlots)(source=variable(WorldBody), angle=...)


def checked_backend(check: OutcomeCheck, *answers: str) -> LanguageModelBackend:
    """
    :return: A backend answering with the scripted answers in turn, each tried by the
        check.
    """
    return LanguageModelBackend(
        client=ScriptedClient(responses=[answer(name) for name in answers]),
        outcome_check=check,
    )


# %% trying answers


class TestOutcomeCheck:
    def test_an_answer_meeting_the_check_is_accepted_at_once(self):
        body = body_with_geometry()
        check = MinimumAngleCheck(required=REQUIRED_ANGLE)
        backend = checked_backend(check, "body_and_steeper_angle_chosen.json")
        (action,) = tilt_query().evaluate(backend=backend)
        assert action.source is body
        assert action.angle >= REQUIRED_ANGLE
        assert [exchange.attempt for exchange in backend.log] == [0]

    def test_an_answer_falling_short_is_sent_back_with_its_outcome(self):
        body = body_with_geometry()
        check = MinimumAngleCheck(required=REQUIRED_ANGLE)
        backend = checked_backend(
            check, "body_and_angle_chosen.json", "body_and_steeper_angle_chosen.json"
        )
        (action,) = tilt_query().evaluate(backend=backend)
        (rejection,) = bodies(backend.log[1].user, PromptSection.REJECTION)
        first_tried = check.tried[0]
        assert json.loads(rejection) == {
            "outcome": {
                "angle_reached": first_tried.angle,
                "angle_required": REQUIRED_ANGLE,
            }
        }
        assert f'kind="{OutcomeFellShort.__name__}"' in backend.log[1].user
        assert action.source is body
        assert action.angle >= REQUIRED_ANGLE

    def test_the_situation_pass_is_never_checked(self):
        body_with_geometry()
        backend = checked_backend(MinimumAngleCheck(required=REQUIRED_ANGLE))
        assert backend.situation_backend_for(tilt_query()).outcome_check is None


# %% falling back to the best answer that fell short


@dataclass
class RankedAngleCheck(OutcomeCheck):
    """
    Accepts a tilting action only when it reaches a required angle; one that falls short
    but reaches a lesser angle may still be accepted as a last resort, the closer to the
    required angle the better.
    """

    required: float
    """
    The angle an action must reach.
    """

    acceptable: float
    """
    The least angle an action falling short may still be accepted at.
    """

    def check(self, instances: Sequence[Any]) -> None:
        (action,) = instances
        if action.angle >= self.required:
            return
        raise OutcomeFellShort(
            outcome={"angle_reached": action.angle, "angle_required": self.required},
            fallback_rank=(
                self.required - action.angle
                if action.angle >= self.acceptable
                else None
            ),
        )


def ranked_backend(check: OutcomeCheck, *answers: str) -> LanguageModelBackend:
    """
    :return: A backend answering with the scripted answers in turn, one repair per
        answer after the first, each tried by the check.
    """
    return LanguageModelBackend(
        client=ScriptedClient(responses=[answer(name) for name in answers]),
        outcome_check=check,
        max_repairs=len(answers) - 1,
    )


MODERATE_ANSWER = "body_and_angle_chosen_with_why.json"
"""
An answer tilting short of the required angle, but closer to it than the small one.
"""

SMALL_ANSWER = "body_and_small_angle_chosen.json"
"""
An answer tilting far short of the required angle.
"""


@pytest.fixture
def body():
    """
    A body of the world, kept alive for the test, for the query to choose.
    """
    return body_with_geometry()


class TestFallingBack:
    def test_when_every_answer_falls_short_the_best_acceptable_one_is_kept(self, body):
        check = RankedAngleCheck(required=REQUIRED_ANGLE, acceptable=0.0)
        backend = ranked_backend(check, MODERATE_ANSWER, SMALL_ANSWER)
        (action,) = tilt_query().evaluate(backend=backend)
        (moderate,) = json.loads(answer(MODERATE_ANSWER))
        assert action.source is body
        assert action.angle == moderate["ActionWithBodyAndValueSlots.angle"]

    def test_the_answer_kept_keeps_its_own_reason(self, body):
        check = RankedAngleCheck(required=REQUIRED_ANGLE, acceptable=0.0)
        backend = ranked_backend(check, MODERATE_ANSWER, SMALL_ANSWER)
        list(tilt_query().evaluate(backend=backend))
        (justification,) = backend.justifications
        (moderate,) = json.loads(answer(MODERATE_ANSWER))
        assert justification.why == moderate["why"]

    def test_an_answer_never_acceptable_is_never_kept(self, body):
        check = RankedAngleCheck(required=REQUIRED_ANGLE, acceptable=REQUIRED_ANGLE)
        backend = ranked_backend(check, MODERATE_ANSWER, SMALL_ANSWER)
        with pytest.raises(RepairBudgetExhausted):
            list(tilt_query().evaluate(backend=backend))

    def test_the_model_is_not_told_how_its_answer_ranks(self, body):
        check = RankedAngleCheck(required=REQUIRED_ANGLE, acceptable=0.0)
        backend = ranked_backend(check, MODERATE_ANSWER, SMALL_ANSWER)
        list(tilt_query().evaluate(backend=backend))
        (rejection,) = bodies(backend.log[1].user, PromptSection.REJECTION)
        assert set(json.loads(rejection)) == {"outcome"}


# %% predicting what an answer brings about


ECHO_PREDICTION_RULE = PromptRule(
    RuleName.OUTCOME_PREDICTION,
    f"{ECHO} predicts nothing; it answers with its input.",
)
"""
The rule of the prediction offering the echoing tool.
"""


@dataclass
class PredictionAnsweringWithItsInput(OutcomePrediction):
    """
    A prediction offering one tool, which answers with the input it was called with.
    """

    def tools(self) -> List[Tool]:
        return [ToolAnsweringWithItsInput()]

    @property
    def rule(self) -> PromptRule:
        return ECHO_PREDICTION_RULE


def predicting_backend(*turns: object) -> LanguageModelBackend:
    """
    :return: A backend asking with the world tools and the echoing prediction, its model
        taking the scripted turns in order.
    """
    return LanguageModelBackend.with_world_tools(
        client=ScriptedClient(tool_turns=list(turns)),
        world=WorldView(),
        outcome_prediction=PredictionAnsweringWithItsInput(),
    )


def tilt_of(body) -> object:
    """
    :return: A query for tilting the given body, its angle left open.
    """
    return the(ActionWithBodyAndValueSlots)(source=body, angle=...)


class TestOutcomePrediction:
    def test_the_prediction_is_offered_beside_the_world_tools(self):
        backend = predicting_backend(
            [ScriptedToolRequest(ECHO, {"angle": 1.2})],
            answer("angle_chosen.json"),
        )
        list(tilt_of(body_with_geometry()).evaluate(backend=backend))
        (call,) = backend.conversations[0].calls
        assert call.name == ECHO
        assert json.loads(call.outcome.content) == {"angle": 1.2}

    def test_the_rule_of_the_prediction_is_stated(self):
        backend = predicting_backend(answer("angle_chosen.json"))
        assert ECHO_PREDICTION_RULE in backend.rules

    def test_a_backend_asking_without_tools_is_offered_no_prediction(self):
        backend = LanguageModelBackend(
            client=ScriptedClient(responses=[answer("angle_chosen.json")]),
            outcome_prediction=PredictionAnsweringWithItsInput(),
        )
        assert ECHO_PREDICTION_RULE not in backend.rules

    def test_the_situation_pass_is_offered_no_prediction(self):
        backend = predicting_backend()
        assert backend.situation_backend_for(tilt_query()).outcome_prediction is None
