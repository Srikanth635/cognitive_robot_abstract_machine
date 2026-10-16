"""
Tests of predicting where a pour ends before it is answered: coraplex's pour poses for
the values, computed in the scene's world before anything is performed, carried with how
the gripper holds the source, and offered to the resolver as a tool.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pytest
from jsonschema import Draft202012Validator
from llm_reasoner.resolver.prompt_format import RuleName
from llm_reasoner.world.view import WorldView
from semantic_digital_twin.semantic_annotations.semantic_annotations import PouringCup

from cram_viz import paths
from cram_viz.resolution.execution import PouringExecution
from cram_viz.resolution.hold_measurement import MeasuredHolding
from cram_viz.resolution.prediction import (
    PREDICT_POUR_END,
    HeldSource,
    Holding,
    PourPrediction,
    PourPredictor,
    PredictPourEnd,
)
from cram_viz.resolution.scene_world import SceneWorldBuilder
from cram_viz.resolution.trial import OutcomeKey

RECORDED_POURS = (
    Path(__file__).parent / "resolution_cases" / "recorded_simple_pouring_pours.json"
)
"""
Pours performed in the simulation of the simple_pouring scene, with where the mouth
ended.
"""

POURING_VALUES = {
    "source_container": "Static_MilkPitcher.stl",
    "target_container": "bowl.stl",
    "arm": "RIGHT",
    "tilt_angle": 1.85,
}
"""
A pour from the pitcher into the bowl, its geometry left to the action's defaults.
"""

HELD_AT_THE_TOOL_FRAME = HeldSource(source_in_tool_frame=np.eye(4), robot_yaw=0.0)
"""
A source held with its frame at the gripper's tool frame, the robot facing along x.
"""

# %% helpers


@dataclass
class HoldingAtOnePose(Holding):
    """
    Holds every source at one pose in the gripper.
    """

    held_at: HeldSource
    """
    How every source is held.
    """

    def held(self, values) -> HeldSource:
        return self.held_at


@pytest.fixture
def predictor(pouring_kitchen) -> PourPredictor:
    """
    A predictor in the PR2's kitchen laid out as the simple_pouring scene, the source
    held at the tool frame.
    """
    return PourPredictor(
        world=pouring_kitchen, holding=HoldingAtOnePose(held_at=HELD_AT_THE_TOOL_FRAME)
    )


# %% the held source


class TestHeldSource:
    def test_a_held_source_is_kept_as_json_and_read_back(self):
        held = HeldSource(
            source_in_tool_frame=np.diag([1.0, 1.0, 1.0, 1.0]), robot_yaw=0.25
        )
        read = HeldSource.from_json(json.loads(json.dumps(held.to_json())))
        assert np.array_equal(read.source_in_tool_frame, held.source_in_tool_frame)
        assert read.robot_yaw == held.robot_yaw


# %% predicting


class TestPrediction:
    def test_a_prediction_states_what_a_simulated_pour_measures(self, predictor):
        outcome = predictor.predict(POURING_VALUES).as_data()
        assert set(outcome) >= {
            OutcomeKey.LANDS_IN_OPENING,
            OutcomeKey.MOUTH_POSITION,
            OutcomeKey.OPENING_CENTRE,
            OutcomeKey.DISTANCE_FROM_OPENING_CENTRE,
            OutcomeKey.RIGHT_OF_OPENING_CENTRE,
            OutcomeKey.AHEAD_OF_OPENING_CENTRE,
        }

    def test_a_lower_pour_lowers_the_mouth_by_as_much(self, predictor):
        high = predictor.predict({**POURING_VALUES, "pour_height": 0.13})
        low = predictor.predict({**POURING_VALUES, "pour_height": 0.09})
        assert high.mouth[2] - low.mouth[2] == pytest.approx(0.04)

    def test_a_side_offset_towards_the_left_moves_the_mouth_left(self, predictor):
        default = predictor.predict(POURING_VALUES)
        moved = predictor.predict({**POURING_VALUES, "pour_side_offset": -0.1})
        assert moved.right < default.right

    def test_predicting_performs_nothing_in_the_world(self, predictor):
        execution = PouringExecution(world=predictor.world, values=POURING_VALUES)
        source = execution.action().source_container.root
        before = source.global_pose.to_position().to_np()
        predictor.predict(POURING_VALUES)
        assert np.array_equal(source.global_pose.to_position().to_np(), before)


# %% the tool


class TestTool:
    def test_the_tool_predicts_a_pour_named_by_handles(self, predictor):
        view = WorldView()
        cup = predictor.world.get_semantic_annotations_by_type(PouringCup)[0]
        bowl = predictor.world.get_body_by_name("bowl.stl")
        tool = PredictPourEnd(predictor=predictor, world=view)
        outcome = json.loads(
            tool.run(
                {
                    "source_container": view.handle_of(cup),
                    "target_container": view.handle_of(bowl),
                    "arm": "RIGHT",
                    "tilt_angle": 1.85,
                }
            ).content
        )
        assert outcome == predictor.predict(POURING_VALUES).as_data()

    def test_the_tool_answers_in_the_shape_it_declares(self, predictor):
        view = WorldView()
        cup = predictor.world.get_semantic_annotations_by_type(PouringCup)[0]
        bowl = predictor.world.get_body_by_name("bowl.stl")
        tool = PredictPourEnd(predictor=predictor, world=view)
        Draft202012Validator(tool.definition.output_schema).validate(
            json.loads(
                tool.run(
                    {
                        "source_container": view.handle_of(cup),
                        "target_container": view.handle_of(bowl),
                        "arm": "RIGHT",
                        "tilt_angle": 1.85,
                    }
                ).content
            )
        )

    def test_the_prediction_offers_the_tool_with_its_rule(self, predictor):
        prediction = PourPrediction(predictor=predictor, world=WorldView())
        assert [tool.definition.name for tool in prediction.tools()] == [
            PREDICT_POUR_END
        ]
        assert prediction.rule.name is RuleName.OUTCOME_PREDICTION


# %% against the pours performed in the simulation


@pytest.mark.slow
class TestAgainstPerformedPours:
    def test_every_recorded_pour_is_predicted_within_a_centimetre(self):
        bundle = paths.scenes_dir() / "simple_pouring"
        if not bundle.is_dir() or os.environ.get("CRAM_VIZ_SCENES") is None:
            pytest.skip("needs the simple_pouring scene")
        held = HeldSource.measure_in(
            SceneWorldBuilder.robot_world(bundle), POURING_VALUES
        )
        predictor = PourPredictor(
            world=SceneWorldBuilder.robot_world(bundle),
            holding=HoldingAtOnePose(held_at=held),
        )
        for pour in json.loads(RECORDED_POURS.read_text()):
            values = {
                **POURING_VALUES,
                **{
                    name: value
                    for name, value in pour["values"].items()
                    if value is not None
                },
            }
            predicted = predictor.predict(values).mouth
            assert np.linalg.norm(predicted - np.array(pour["mouth"])) < 0.01, values


# %% measuring how the source is held


class TestMeasuredHolding:
    def test_the_hold_is_measured_once_per_source_and_arm(self, tmp_path):
        run = ProcessWritingTheHold(held=HELD_AT_THE_TOOL_FRAME)
        holding = MeasuredHolding(bundle=tmp_path, run=run)
        holding.held(POURING_VALUES)
        holding.held({**POURING_VALUES, "tilt_angle": 1.2})
        holding.held({**POURING_VALUES, "arm": "LEFT"})
        assert len(run.commands) == 2

    def test_the_hold_measured_is_what_the_process_wrote(self, tmp_path):
        held = HeldSource(source_in_tool_frame=np.eye(4), robot_yaw=0.5)
        holding = MeasuredHolding(bundle=tmp_path, run=ProcessWritingTheHold(held=held))
        assert holding.held(POURING_VALUES).robot_yaw == held.robot_yaw

    @pytest.mark.slow
    def test_the_hold_measured_in_a_process_of_its_own_is_read_back(self):
        bundle = paths.scenes_dir() / "simple_pouring"
        if not bundle.is_dir() or os.environ.get("CRAM_VIZ_SCENES") is None:
            pytest.skip("needs the simple_pouring scene")
        held = MeasuredHolding(bundle=bundle).held(POURING_VALUES)
        assert held.source_in_tool_frame.shape == (4, 4)


@dataclass
class ProcessWritingTheHold:
    """
    Runs no process: each call writes the hold where the measuring process would.
    """

    held: HeldSource
    """
    The hold written.
    """

    commands: list = field(default_factory=list)
    """
    The commands asked for, in order.
    """

    def __call__(self, command, **settings):
        self.commands.append(command)
        Path(command[-1]).write_text(json.dumps(self.held.to_json()))
