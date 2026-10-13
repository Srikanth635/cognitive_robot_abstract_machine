"""
Tests of trying a resolved pouring in the simulation before it is accepted: where the
source container's mouth ends against the target's opening, and the check telling the
resolver when a pour misses.
"""

from __future__ import annotations

import json
import sys
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pytest
from coraplex.datastructures.enums import Arms
from llm_reasoner.resolver.exceptions import OutcomeFellShort
from typing_extensions import Any, Dict, List

from cram_viz.resolution.trial import (
    TRIAL_MODULE,
    OutcomeKey,
    PouringTrial,
    SimulatedPouring,
)

from .resolutions import LANDS, MISSES, ScriptedTrials, objects_given

# %% helpers


def resolved_values() -> Dict[str, Any]:
    """
    :return: The slots of the resolution the trials perform, as the run keeps them.
    """
    (action,) = objects_given().actions
    return {slot.name: slot.serialized for slot in action.slots}


@dataclass
class PourAnswer:
    """
    What an answer constructs, as far as the check reads it.
    """

    arm: Arms
    """
    The arm holding the source container.
    """

    tilt_angle: float
    """
    The tilt angle, in radians.
    """


def checking(outcomes: List[Any], allowed_trials: int = 3) -> SimulatedPouring:
    """
    :return: A check whose trials report the outcomes in turn.
    """
    return SimulatedPouring(
        bundle=Path("pouring"),
        slots=["arm", "tilt_angle"],
        allowed_trials=allowed_trials,
        run=ScriptedTrials(outcomes=outcomes),
    )


# %% where the mouth ends


class TestOutcome:
    def test_the_default_pour_ends_beside_the_opening(self, pouring_kitchen):
        outcome = PouringTrial.outcome_in(pouring_kitchen, resolved_values())
        assert outcome[OutcomeKey.LANDS_IN_OPENING] is False
        assert np.isclose(
            outcome[OutcomeKey.DISTANCE_FROM_OPENING_CENTRE],
            np.hypot(
                outcome[OutcomeKey.AHEAD_OF_OPENING_CENTRE],
                outcome[OutcomeKey.RIGHT_OF_OPENING_CENTRE],
            ),
            atol=1e-3,
        )

    def test_shifting_the_pour_by_its_miss_lands_it(self, pouring_kitchen):
        missed = PouringTrial.outcome_in(deepcopy(pouring_kitchen), resolved_values())
        shifted = {
            **resolved_values(),
            "pour_side_offset": -missed[OutcomeKey.RIGHT_OF_OPENING_CENTRE],
        }
        outcome = PouringTrial.outcome_in(pouring_kitchen, shifted)
        assert outcome[OutcomeKey.LANDS_IN_OPENING] is True


# %% the check


class TestSimulatedPouring:
    def test_a_pour_landing_in_the_opening_is_accepted(self):
        check = checking([LANDS])
        check.check([PourAnswer(arm=Arms.RIGHT, tilt_angle=1.85)])
        assert [trial.outcome for trial in check.trials] == [LANDS]

    def test_a_pour_missing_the_opening_is_refused_with_its_outcome(self):
        check = checking([MISSES])
        with pytest.raises(OutcomeFellShort) as refusal:
            check.check([PourAnswer(arm=Arms.RIGHT, tilt_angle=1.85)])
        assert refusal.value.outcome == MISSES

    def test_a_failed_simulation_is_refused_saying_why(self):
        check = checking([None])
        with pytest.raises(OutcomeFellShort) as refusal:
            check.check([PourAnswer(arm=Arms.RIGHT, tilt_angle=1.85)])
        assert refusal.value.outcome[OutcomeKey.MOTION_FAILED] == (
            "MotionDidNotFinish: out of reach"
        )

    def test_the_last_allowed_trial_is_accepted_with_its_outcome(self):
        check = checking([MISSES, MISSES], allowed_trials=2)
        with pytest.raises(OutcomeFellShort):
            check.check([PourAnswer(arm=Arms.RIGHT, tilt_angle=1.85)])
        check.check([PourAnswer(arm=Arms.LEFT, tilt_angle=1.6)])
        assert [trial.values for trial in check.trials] == [
            {"arm": Arms.RIGHT.name, "tilt_angle": 1.85},
            {"arm": Arms.LEFT.name, "tilt_angle": 1.6},
        ]

    def test_a_trial_runs_in_a_process_of_its_own_on_the_scene(self):
        check = checking([LANDS])
        check.check([PourAnswer(arm=Arms.RIGHT, tilt_angle=1.85)])
        (command,) = check.run.commands
        assert command[:5] == [
            sys.executable,
            "-m",
            TRIAL_MODULE,
            str(check.bundle),
            json.dumps({"arm": Arms.RIGHT.name, "tilt_angle": 1.85}),
        ]
