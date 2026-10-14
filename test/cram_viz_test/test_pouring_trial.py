"""
Tests of trying a resolved pouring in the simulation before it is accepted: where the
source container's mouth ends against the target's opening, and the check telling the
resolver when a pour misses.
"""

from __future__ import annotations

import argparse
import json
import sys
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pytest
from coraplex.datastructures.enums import Arms
from llm_reasoner.observation.sinks import MemorySink
from llm_reasoner.resolver.exceptions import OutcomeFellShort
from llm_reasoner.observation.sinks import JsonLinesSink
from typing_extensions import Any, Dict, List, Optional

from cram_viz.events import NeemKind, TrialFinished, TrialStarted
from cram_viz.neem import (
    DatabaseNeemRecorder,
    IncompleteNeemOptions,
    NeemOption,
    NeemRecording,
    NoNeemRecorder,
    SqliteNeemStore,
)
from cram_viz.resolution.trial import (
    OUTCOME_FILE,
    TRIAL_MODULE,
    OutcomeKey,
    PouringTrial,
    SimulatedPouring,
)

from .resolutions import LANDS, MISSES, ScriptedTrials, objects_given
from .test_neem import PlanCollectingRecorder

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


def checking(
    outcomes: List[Any],
    allowed_trials: int = 3,
    observer: MemorySink | None = None,
    neems: Optional[NeemRecording] = None,
) -> SimulatedPouring:
    """
    :return: A check whose trials report the outcomes in turn, the trials to the
        observer when there is one, and keep their NEEMs as the recording says.
    """
    settings = {} if observer is None else {"observer": observer}
    return SimulatedPouring(
        neems=neems,
        bundle=Path("pouring"),
        slots=["arm", "tilt_angle"],
        allowed_trials=allowed_trials,
        run=ScriptedTrials(outcomes=outcomes),
        **settings,
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


# %% reporting the trials


class TestObservedTrials:
    def test_a_trial_is_reported_with_the_values_tried_and_what_they_brought_about(
        self,
    ):
        sink = MemorySink()
        check = checking([LANDS], observer=sink)
        check.check([PourAnswer(arm=Arms.RIGHT, tilt_angle=1.85)])
        started, finished = sink.events
        assert (type(started), type(finished)) == (TrialStarted, TrialFinished)
        assert started.values == {"arm": Arms.RIGHT.name, "tilt_angle": 1.85}
        assert finished.outcome == LANDS
        assert finished.trial == started.trial
        assert finished.seconds >= 0.0

    def test_a_failed_simulation_is_reported_as_a_trial_with_its_failure(self):
        sink = MemorySink()
        check = checking([None], observer=sink)
        with pytest.raises(OutcomeFellShort):
            check.check([PourAnswer(arm=Arms.RIGHT, tilt_angle=1.85)])
        assert OutcomeKey.MOTION_FAILED in sink.events[-1].outcome

    def test_every_trial_is_reported(self):
        sink = MemorySink()
        check = checking([MISSES, LANDS], observer=sink)
        with pytest.raises(OutcomeFellShort):
            check.check([PourAnswer(arm=Arms.RIGHT, tilt_angle=1.85)])
        check.check([PourAnswer(arm=Arms.LEFT, tilt_angle=1.6)])
        assert [type(event) for event in sink.events] == [
            TrialStarted,
            TrialFinished,
            TrialStarted,
            TrialFinished,
        ]


# %% keeping the NEEMs of the trials


def neem_options_of(command: List[str]) -> Dict[str, str]:
    """
    :return: What a trial's command line says about its NEEM, by option.
    """
    return {
        option: command[command.index(option) + 1]
        for option in NeemOption
        if option in command
    }


def trial_options(recording: NeemRecording, trial: str) -> argparse.Namespace:
    """
    :return: What a trial process reads from the options it is given for its NEEM.
    """
    parser = argparse.ArgumentParser()
    NeemRecording.add_options(parser)
    return parser.parse_args(recording.options(trial))


class TestKeepingTheNeemsOfTheTrials:
    def test_a_check_with_no_recording_asks_its_trials_to_keep_nothing(self):
        check = checking([LANDS])
        check.check([PourAnswer(arm=Arms.RIGHT, tilt_angle=1.85)])
        (command,) = check.run.commands
        assert neem_options_of(command) == {}

    def test_a_trial_is_told_where_to_keep_its_neem_and_which_trial_it_is(
        self, tmp_path
    ):
        recording = NeemRecording(
            database=tmp_path / "neems.sqlite", events=tmp_path / "events.jsonl"
        )
        sink = MemorySink()
        check = checking([LANDS], observer=sink, neems=recording)
        check.check([PourAnswer(arm=Arms.RIGHT, tilt_angle=1.85)])
        (command,) = check.run.commands
        started, _ = sink.events
        assert neem_options_of(command) == {
            NeemOption.DATABASE: str(recording.database),
            NeemOption.EVENTS: str(recording.events),
            NeemOption.EPISODE: started.trial,
        }

    def test_the_outcome_is_still_written_where_the_last_argument_says(self, tmp_path):
        recording = NeemRecording(
            database=tmp_path / "neems.sqlite", events=tmp_path / "events.jsonl"
        )
        check = checking([LANDS], neems=recording)
        check.check([PourAnswer(arm=Arms.RIGHT, tilt_angle=1.85)])
        (command,) = check.run.commands
        assert command[:5] == [
            sys.executable,
            "-m",
            TRIAL_MODULE,
            str(check.bundle),
            json.dumps({"arm": Arms.RIGHT.name, "tilt_angle": 1.85}),
        ]
        assert command[-1].endswith(OUTCOME_FILE)

    def test_a_trial_process_given_nothing_to_keep_keeps_nothing(self):
        parser = argparse.ArgumentParser()
        NeemRecording.add_options(parser)
        recorder = NeemRecording.recorder_from(parser.parse_args([]), NeemKind.TRIAL)
        assert isinstance(recorder, NoNeemRecorder)

    def test_a_trial_process_keeps_its_neem_where_it_is_told_and_reports_it_there(
        self, tmp_path
    ):
        recording = NeemRecording(
            database=tmp_path / "neems.sqlite", events=tmp_path / "events.jsonl"
        )
        recorder = NeemRecording.recorder_from(
            trial_options(recording, "a-trial"), NeemKind.TRIAL
        )
        assert recorder == DatabaseNeemRecorder(
            store=SqliteNeemStore(path=recording.database),
            kind=NeemKind.TRIAL,
            episode="a-trial",
            observer=JsonLinesSink(path=recording.events),
        )

    def test_options_that_name_a_database_but_not_the_rest_are_refused(self):
        parser = argparse.ArgumentParser()
        NeemRecording.add_options(parser)
        with pytest.raises(IncompleteNeemOptions):
            NeemRecording.recorder_from(
                parser.parse_args([NeemOption.DATABASE, "neems.sqlite"]),
                NeemKind.TRIAL,
            )

    def test_a_trial_performed_hands_its_plan_to_the_recorder(self, pouring_kitchen):
        recorder = PlanCollectingRecorder()
        PouringTrial.outcome_in(pouring_kitchen, resolved_values(), recorder)
        assert len(recorder.plans) == 1
