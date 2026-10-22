"""
Tests of predicting where a pour ends before it is answered: coraplex's pour poses for
the values, computed in the scene's world before anything is performed, carried with how
the gripper holds the source, and offered to the resolver as a tool.
"""

from __future__ import annotations

import json
from contextlib import suppress
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pytest
from coraplex.plans.failures import MotionMadeNoProgress
from scipy.spatial.transform import Rotation
from jsonschema import Draft202012Validator
from llm_reasoner.resolver.prompt_format import RuleName
from llm_reasoner.world.view import WorldView
from semantic_digital_twin.semantic_annotations.semantic_annotations import PouringCup

from typing_extensions import Dict

from cramera.resolution import paths
from cramera.resolution.resolver.execution import PouringExecution
from cramera.resolution.resolver.hold_measurement import MeasuredHolding
from cramera.resolution.resolver.prediction import (
    PREDICT_POUR_END,
    ApproachTolerances,
    HeldSource,
    Holding,
    PickUpFailed,
    PourPrediction,
    PourUnreachable,
    PourPredictor,
    PourReach,
    PredictPourEnd,
)
from cramera.resolution.resolver.scene_world import SceneWorldBuilder
from cramera.resolution.resolver.trial import OutcomeKey, PouringTrial

from .resolutions import FinishedTrial

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
    "arm": "PR2RightArm",
    "tilt_angle": 1.85,
}
"""
A pour from the pitcher into the bowl, its geometry left to the action's defaults.
"""

PICK_UP_FAILURE = "MotionMadeNoProgress: out of reach"
"""
Why a scripted pick-up fails, as the last line a failing process writes.
"""

HELD_AT_THE_TOOL_FRAME = HeldSource(
    source_in_tool_frame=np.eye(4), robot_yaw=0.0, joint_positions={}
)
"""
A source held with its frame at the gripper's tool frame, the robot facing along x, no
joint moved.
"""

TILT_GRID_SIZE = 9
"""
How many tilts, spread evenly from upright to upside down, the pours are tried at.
"""

TILT_GRID = tuple(float(tilt) for tilt in np.linspace(0.0, np.pi, TILT_GRID_SIZE))
"""
The tilts, in radians, the pours are tried at: which of them the arm can reach is read
from the prediction itself, not written here.
"""

PERFORMED_TILTS = (2.9,)
"""
Tilts, in radians, at which the simulation performs the pour from where the robot faces
the bowl although the arm only gets there by a path other than the interpolated one the
check follows.
"""

PERFORMED_WITHIN_THE_TOLERANCE = 2.0
"""
A tilt, in radians, at which the simulation also performs the pour, ending within the
motion's tolerance of the pour pose, but which no solve of the check reaches.
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


@pytest.fixture(scope="module")
def picked_up(pouring_kitchen_template) -> HeldSource:
    """
    How the right gripper holds the pitcher after a simulated pick-up in the PR2's
    kitchen laid out as the simple_pouring scene, measured once.
    """
    return HeldSource.measure_in(deepcopy(pouring_kitchen_template), POURING_VALUES)


@pytest.fixture
def predictor(pouring_kitchen, picked_up) -> PourPredictor:
    """
    A predictor in the PR2's kitchen laid out as the simple_pouring scene, the pitcher
    held as the pick-up leaves it.
    """
    return PourPredictor(
        world=pouring_kitchen, holding=HoldingAtOnePose(held_at=picked_up)
    )


# %% the held source


class TestHeldSource:
    def test_a_held_source_is_kept_as_json_and_read_back(self):
        held = HeldSource(
            source_in_tool_frame=np.diag([1.0, 1.0, 1.0, 1.0]),
            robot_yaw=0.25,
            joint_positions={"pr2/torso_lift_joint": 0.3},
        )
        read = HeldSource.from_json(json.loads(json.dumps(held.to_json())))
        assert np.array_equal(read.source_in_tool_frame, held.source_in_tool_frame)
        assert read.robot_yaw == held.robot_yaw
        assert read.joint_positions == held.joint_positions

    def test_measuring_keeps_the_joints_the_pick_up_moved(self, picked_up):
        assert picked_up.joint_positions["pr2/torso_lift_joint"] > 0.0


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
        before = source.global_pose.position.to_np()
        predictor.predict(POURING_VALUES)
        assert np.array_equal(source.global_pose.position.to_np(), before)


# %% reaching the pour pose


def reaches(predictor: PourPredictor, tilt_angle: float) -> bool:
    """
    :return: Whether the predictor finds the arm able to make the pour at the tilt.
    """
    reached = False
    with suppress(PourUnreachable):
        predictor.predict({**POURING_VALUES, "tilt_angle": tilt_angle})
        reached = True
    return reached


@pytest.fixture(scope="module")
def reach_by_tilt(pouring_kitchen_template, picked_up) -> Dict[float, bool]:
    """
    Whether the arm is predicted able to make the pour at each tilt of the grid, with
    the default tolerances.
    """
    predictor = PourPredictor(
        world=deepcopy(pouring_kitchen_template),
        holding=HoldingAtOnePose(held_at=picked_up),
    )
    return {tilt: reaches(predictor, tilt) for tilt in TILT_GRID}


@pytest.fixture
def unreachable_tilt(reach_by_tilt) -> float:
    """
    A tilt of the grid at which the arm is predicted unable to make the pour.
    """
    return next(tilt for tilt, reached in reach_by_tilt.items() if not reached)


class TestReachingThePourPose:
    def test_the_grid_holds_pours_the_arm_can_make_and_ones_it_cannot(
        self, reach_by_tilt
    ):
        assert set(reach_by_tilt.values()) == {True, False}

    def test_a_pour_the_arm_can_make_is_predicted(self, predictor, reach_by_tilt):
        tilt_angle, *_ = [tilt for tilt, reached in reach_by_tilt.items() if reached]
        outcome = predictor.predict({**POURING_VALUES, "tilt_angle": tilt_angle})
        assert outcome.mouth.shape == (3,)

    def test_a_pour_the_arm_cannot_make_is_refused_with_why(
        self, predictor, unreachable_tilt
    ):
        with pytest.raises(PourUnreachable) as unreachable:
            predictor.predict({**POURING_VALUES, "tilt_angle": unreachable_tilt})
        assert unreachable.value.reason

    @pytest.mark.parametrize("tilt_angle", PERFORMED_TILTS)
    def test_a_pour_the_simulation_performs_is_not_refused(self, predictor, tilt_angle):
        assert reaches(predictor, tilt_angle)

    @pytest.mark.xfail(
        strict=True,
        reason="no solve reaches the pour pose the simulation ends within tolerance of",
    )
    def test_a_pour_the_simulation_ends_within_tolerance_of_is_not_refused(
        self, predictor
    ):
        assert reaches(predictor, PERFORMED_WITHIN_THE_TOLERANCE)

    def test_asking_leaves_the_robot_where_it_was(self, predictor, unreachable_tilt):
        before = predictor.world.state.positions.copy()
        with pytest.raises(PourUnreachable):
            predictor.predict({**POURING_VALUES, "tilt_angle": unreachable_tilt})
        assert np.array_equal(predictor.world.state.positions, before)

    def test_the_tool_predicts_such_a_pour_as_a_failed_motion(
        self, predictor, unreachable_tilt
    ):
        view = WorldView()
        cup = predictor.world.get_semantic_annotations_by_type(PouringCup)[0]
        bowl = predictor.world.get_body_by_name("bowl.stl")
        tool = PredictPourEnd(predictor=predictor, world=view)
        outcome = json.loads(
            tool.run(
                {
                    "source_container": view.handle_of(cup),
                    "target_container": view.handle_of(bowl),
                    "arm": "PR2RightArm",
                    "tilt_angle": unreachable_tilt,
                }
            ).content
        )
        with pytest.raises(PourUnreachable) as unreachable:
            predictor.predict({**POURING_VALUES, "tilt_angle": unreachable_tilt})
        assert outcome == {
            OutcomeKey.LANDS_IN_OPENING: False,
            OutcomeKey.MOTION_FAILED: unreachable.value.reason,
        }
        Draft202012Validator(tool.definition.output_schema).validate(outcome)


# %% the steps of the approach

TURN = 1.0
"""
How far, in radians, the tool turns between the poses of these tests.
"""


def pose_turned(angle: float, lifted: float = 0.0) -> np.ndarray:
    """
    :return: A pose turned about the y axis by the angle and lifted by the height.
    """
    pose = np.eye(4)
    pose[:3, :3] = Rotation.from_euler("y", angle).as_matrix()
    pose[2, 3] = lifted
    return pose


def turn_between(first: np.ndarray, second: np.ndarray) -> float:
    """
    :return: The angle, in radians, that turns one pose's orientation into the other's.
    """
    return float(
        np.linalg.norm(
            Rotation.from_matrix(first[:3, :3].T @ second[:3, :3]).as_rotvec()
        )
    )


class TestApproachSteps:
    def test_a_turn_is_taken_in_steps_no_larger_than_the_tolerance(self):
        tolerance = ApproachTolerances(max_rotation_per_step=0.3)
        steps = PourReach.poses_between(pose_turned(0.0), pose_turned(TURN), tolerance)
        before = [pose_turned(0.0), *steps]
        assert (
            max(turn_between(first, second) for first, second in zip(before, steps))
            <= tolerance.max_rotation_per_step + 1e-9
        )

    def test_the_last_step_is_the_pose_asked_for(self):
        goal = pose_turned(TURN, lifted=0.2)
        steps = PourReach.poses_between(
            pose_turned(0.0), goal, ApproachTolerances(max_rotation_per_step=0.3)
        )
        assert np.allclose(steps[-1], goal)

    def test_a_larger_turn_takes_more_steps(self):
        tolerance = ApproachTolerances(max_rotation_per_step=0.3)
        small = PourReach.poses_between(pose_turned(0.0), pose_turned(0.5), tolerance)
        large = PourReach.poses_between(pose_turned(0.0), pose_turned(2.0), tolerance)
        assert len(large) > len(small)

    def test_a_move_without_a_turn_is_one_step(self):
        steps = PourReach.poses_between(
            pose_turned(0.0), pose_turned(0.0, lifted=0.3), ApproachTolerances()
        )
        assert len(steps) == 1


SWEPT_STEP_SIZES = (0.15, 0.5, 0.7)
"""
Largest turns per step, in radians, across which the same pours are predicted the same
way, the default among them.
"""


class TestApproachTolerances:
    @pytest.mark.parametrize("max_rotation_per_step", SWEPT_STEP_SIZES)
    def test_the_same_pours_are_predicted_across_the_step_sizes(
        self, pouring_kitchen, picked_up, reach_by_tilt, max_rotation_per_step
    ):
        predictor = PourPredictor(
            world=pouring_kitchen,
            holding=HoldingAtOnePose(held_at=picked_up),
            approach=ApproachTolerances(max_rotation_per_step=max_rotation_per_step),
        )
        assert {tilt: reaches(predictor, tilt) for tilt in TILT_GRID} == reach_by_tilt

    def test_the_default_step_size_is_one_of_the_swept(self):
        assert ApproachTolerances().max_rotation_per_step in SWEPT_STEP_SIZES


# %% against the pour performed in the simulation


@pytest.mark.slow
class TestAgainstThePourPerformed:
    @pytest.mark.parametrize(
        "tilt_angle", (*PERFORMED_TILTS, PERFORMED_WITHIN_THE_TOLERANCE)
    )
    def test_the_tilts_the_check_must_not_refuse_are_performed(
        self, pouring_kitchen, tilt_angle
    ):
        outcome = PouringTrial.outcome_in(
            deepcopy(pouring_kitchen), {**POURING_VALUES, "tilt_angle": tilt_angle}
        )
        assert OutcomeKey.MOTION_FAILED not in outcome

    def test_a_pour_predicted_out_of_reach_fails_in_the_simulation(
        self, pouring_kitchen, reach_by_tilt
    ):
        out_of_reach = [tilt for tilt, reached in reach_by_tilt.items() if not reached]
        assert out_of_reach
        for tilt_angle in out_of_reach:
            with pytest.raises(MotionMadeNoProgress):
                PouringTrial.outcome_in(
                    deepcopy(pouring_kitchen),
                    {**POURING_VALUES, "tilt_angle": tilt_angle},
                )


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
                    "arm": "PR2RightArm",
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
                        "arm": "PR2RightArm",
                        "tilt_angle": 1.85,
                    }
                ).content
            )
        )

    def test_a_pour_whose_pick_up_fails_is_predicted_as_a_failed_motion(
        self, pouring_kitchen
    ):
        view = WorldView()
        cup = pouring_kitchen.get_semantic_annotations_by_type(PouringCup)[0]
        bowl = pouring_kitchen.get_body_by_name("bowl.stl")
        tool = PredictPourEnd(
            predictor=PourPredictor(
                world=pouring_kitchen, holding=HoldingThatCannotPickUp()
            ),
            world=view,
        )
        outcome = json.loads(
            tool.run(
                {
                    "source_container": view.handle_of(cup),
                    "target_container": view.handle_of(bowl),
                    "arm": "PR2LeftArm",
                }
            ).content
        )
        assert outcome == {
            OutcomeKey.LANDS_IN_OPENING: False,
            OutcomeKey.MOTION_FAILED: PICK_UP_FAILURE,
        }
        Draft202012Validator(tool.definition.output_schema).validate(outcome)

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
        if not bundle.is_dir():
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
        holding.held({**POURING_VALUES, "arm": "PR2LeftArm"})
        assert len(run.commands) == 2

    def test_a_hold_the_process_cannot_pick_up_is_refused_with_its_reason(
        self, tmp_path
    ):
        holding = MeasuredHolding(bundle=tmp_path, run=ProcessFailingToPickUp())
        with pytest.raises(PickUpFailed) as failed:
            holding.held(POURING_VALUES)
        assert failed.value.reason == PICK_UP_FAILURE

    def test_the_hold_measured_is_what_the_process_wrote(self, tmp_path):
        held = HeldSource(
            source_in_tool_frame=np.eye(4), robot_yaw=0.5, joint_positions={}
        )
        holding = MeasuredHolding(bundle=tmp_path, run=ProcessWritingTheHold(held=held))
        assert holding.held(POURING_VALUES).robot_yaw == held.robot_yaw

    @pytest.mark.slow
    def test_the_hold_measured_in_a_process_of_its_own_is_read_back(self):
        bundle = paths.scenes_dir() / "simple_pouring"
        if not bundle.is_dir():
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
        return FinishedTrial(returncode=0)


@dataclass
class ProcessFailingToPickUp:
    """
    Runs no process: each call finishes as a measuring process whose pick-up failed.
    """

    def __call__(self, command, **settings):
        return FinishedTrial(
            returncode=1, stderr="Traceback\nMotionMadeNoProgress: out of reach\n"
        )


@dataclass
class HoldingThatCannotPickUp(Holding):
    """
    Fails to pick up every source.
    """

    def held(self, values) -> HeldSource:
        raise PickUpFailed(reason=PICK_UP_FAILURE)
