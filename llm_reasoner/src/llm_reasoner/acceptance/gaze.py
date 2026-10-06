"""
Acceptance rules of the gaze module: a fixation rests on a participant of the phase it
serves until that phase's goal is reached.
"""

from __future__ import annotations

from enum import StrEnum

from krrood.entity_query_language.factories import and_, contains, flat_variable, not_
from typing_extensions import Any

from llm_reasoner.acceptance.rules import FragmentRule
from llm_reasoner.ontologies.predictive_manipulation import gaze

# %% rule names


class GazeRule(StrEnum):
    """
    The acceptance rules of the gaze module.
    """

    FIXATION_HELD_UNTIL_PHASE_GOAL = "fixation_held_until_phase_goal"
    """
    Gaze stays at a landmark until about the time the goal of the phase it serves is
    reached, so a fixation is released at an event ending that phase.
    """

    FIXATION_ON_PHASE_PARTICIPANT = "fixation_on_phase_participant"
    """
    The landmark of a fixation is a participant of the phase the fixation serves.
    """


# %% rules


class FixationHeldUntilPhaseGoal(FragmentRule[gaze.GazeFixation]):
    """
    Every event a fixation is held until ends the phase the fixation serves.
    """

    @property
    def name(self) -> StrEnum:
        return GazeRule.FIXATION_HELD_UNTIL_PHASE_GOAL

    @property
    def reason(self) -> str:
        return "maintained_until names an event that does not end serves_phase"

    def violating(self, subject: Any) -> Any:
        release = flat_variable(subject.maintained_until)
        return and_(
            subject.serves_phase != None,
            not_(contains(subject.serves_phase.terminated_by, release)),
        )


class FixationOnPhaseParticipant(FragmentRule[gaze.GazeFixation]):
    """
    The participant a fixation rests on takes part in the phase the fixation serves.
    """

    @property
    def name(self) -> StrEnum:
        return GazeRule.FIXATION_ON_PHASE_PARTICIPANT

    @property
    def reason(self) -> str:
        return "fixates_role is not a participant of serves_phase"

    def violating(self, subject: Any) -> Any:
        return and_(
            subject.serves_phase != None,
            subject.fixates_role != None,
            not_(contains(subject.serves_phase.involves_role, subject.fixates_role)),
        )
