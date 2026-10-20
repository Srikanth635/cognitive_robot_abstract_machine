"""
Predicting where a pour ends, before it is answered.

coraplex's pouring computes its tool poses when its plan is built, before the source
container is picked up, so the tool pose for given values is the one computed in the
scene's world as it stands; carried with how the gripper holds the source after the
pick-up, it puts the source's mouth where the simulated pour measures it. How the
gripper holds the source, and where the pick-up leaves the robot, is measured once per
scene and arm, by a simulated pick-up in a process of its own. From there the arm is
asked whether it can move the tool to the pour pose at all.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from copy import deepcopy
import math
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path

import numpy as np
from coraplex.datastructures.enums import Arms
from coraplex.execution_environment import simulated_robot
from coraplex.plans.factories import sequential
from coraplex.view_manager import ViewManager
from krrood.adapters.json_serializer import SubclassJSONSerializer
from llm_reasoner.clients.tools import Tool, ToolDefinition, ToolOutcome
from llm_reasoner.resolver.outcomes import OutcomePrediction
from llm_reasoner.resolver.prompt_format import PromptRule, RuleName
from llm_reasoner.world.access import WorldAccess
from scipy.spatial.transform import Rotation, Slerp
from semantic_digital_twin.robots.robot_parts import AbstractRobot
from semantic_digital_twin.spatial_computations.ik_solver import IKSolverException
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix, Pose
from semantic_digital_twin.world import World
from typing_extensions import Any, Dict, List, Mapping, Self

from cram_viz.resolution.execution import PouringExecution
from cram_viz.resolution.pouring_slot import PouringSlot
from cram_viz.resolution.slot_data import SlotData
from cram_viz.resolution.trial import (
    OutcomeKey,
    PourOutcome,
    top_centre_in_own_frame,
)

PREDICT_POUR_END = "predict_pour_end"
"""
The name of the tool predicting where a pour ends.
"""

DEFINITION_FILE = (
    Path(__file__).parent / "tool_definitions" / f"{PREDICT_POUR_END}.json"
)
"""
The definition of that tool.
"""

PREDICTION_RULE = PromptRule(
    RuleName.OUTCOME_PREDICTION,
    f"{PREDICT_POUR_END} predicts where the source container's mouth ends for a pour's "
    "values, as the simulated pour will measure it. Before answering, predict the "
    "values you would answer; when the mouth does not land in the opening, change "
    "pour_side_offset, pour_approach_offset, pour_height or tilt_angle, reading how far "
    "ahead and right of the opening's centre it ends, and predict again.",
)
"""
The rule saying how the resolver uses the prediction.
"""


@dataclass
class PredictedMotionFailure(Exception):
    """
    A motion a pour needs is predicted to fail, so no outcome can be predicted for it.
    """

    reason: str
    """
    Why the motion fails.
    """


class PickUpFailed(PredictedMotionFailure):
    """
    The robot could not pick the source container up with the arm a pour names.
    """

    def __post_init__(self):
        super().__init__(f"the pick-up failed: {self.reason}")


class PourUnreachable(PredictedMotionFailure):
    """
    The arm cannot move the tool to the pose the pour needs.
    """

    def __post_init__(self):
        super().__init__(f"the pour pose is out of reach: {self.reason}")


class HeldSourceKey(StrEnum):
    """
    The keys of a held source's JSON form.
    """

    SOURCE_IN_TOOL_FRAME = "source_in_tool_frame"
    ROBOT_YAW = "robot_yaw"
    JOINT_POSITIONS = "joint_positions"


# %% how the gripper holds the source


@dataclass
class HeldSource(SubclassJSONSerializer):
    """
    How the gripper holds the source container after the pick-up, and where the robot
    faces then.
    """

    source_in_tool_frame: np.ndarray
    """
    The source container's pose in the frame of the tool holding it, a 4x4 transform.
    """

    robot_yaw: float
    """
    The heading of the robot's base after it faced the source, in radians.
    """

    joint_positions: Dict[str, float]
    """
    The position of every degree of freedom of the world after the pick-up, by name: where
    the base stands and how the torso and arms are held.
    """

    def to_json(self) -> Dict[str, Any]:
        return {
            **super().to_json(),
            HeldSourceKey.SOURCE_IN_TOOL_FRAME: self.source_in_tool_frame.tolist(),
            HeldSourceKey.ROBOT_YAW: self.robot_yaw,
            HeldSourceKey.JOINT_POSITIONS: self.joint_positions,
        }

    @classmethod
    def _from_json(cls, data: Dict[str, Any], **kwargs) -> Self:
        return cls(
            source_in_tool_frame=np.array(data[HeldSourceKey.SOURCE_IN_TOOL_FRAME]),
            robot_yaw=float(data[HeldSourceKey.ROBOT_YAW]),
            joint_positions=dict(data[HeldSourceKey.JOINT_POSITIONS]),
        )

    @classmethod
    def measure_in(cls, world: World, values: Mapping[str, Any]) -> HeldSource:
        """
        Perform the steps a trial performs before it pours, in the world, and measure
        how the gripper then holds the source.

        :param world: A world the robot can act in, its robot annotated; changed.
        :param values: The pour's values as data, by field name, naming the source, the
            target and the arm.
        :return: How the gripper holds the source, and where the robot faces.
        """
        execution = PouringExecution(world=world, values=dict(values))
        action = execution.action()
        with simulated_robot:
            sequential(
                execution.steps_before_pour(action), context=execution.context()
            ).perform()
        tool_frame = ViewManager.get_end_effector_view(
            action.arm, execution.robot
        ).tool_frame
        return cls(
            source_in_tool_frame=world.compute_forward_kinematics_np(
                tool_frame, action.source_container.root
            ),
            robot_yaw=float(
                execution.robot.root.global_pose.to_quaternion().to_rpy()[2]
            ),
            joint_positions={
                str(dof.name): float(world.state[dof.id].position)
                for dof in world.degrees_of_freedom
            },
        )


class Holding(ABC):
    """
    Says how the gripper holds the source container a pour names.
    """

    @abstractmethod
    def held(self, values: Mapping[str, Any]) -> HeldSource:
        """
        :param values: The pour's values as data, naming the source, the target and the
            arm.
        :return: How the arm's gripper holds the source after picking it up.
        :raises PickUpFailed: When the arm cannot pick the source up.
        """


# %% reaching the pour pose


@dataclass(frozen=True)
class ApproachTolerances:
    """
    How finely the tool's turn from the pre-pour pose to the pour pose is followed when
    asking whether the arm can make it. The same pours are predicted the same way across
    a range of these values; the defaults sit inside it.
    """

    max_rotation_per_step: float = 0.5
    """
    The most the tool's orientation turns, in radians, between two poses inverse
    kinematics is solved for. A single solve over a larger turn can stop short of poses
    the arm reaches.
    """


@dataclass
class PourReach:
    """
    Asks whether the arm can move the tool through the poses of a pour, from where the
    pick-up left the robot.
    """

    world: World
    """
    The scene's world; copied, never changed.
    """

    held: HeldSource
    """
    Where the pick-up leaves the robot.
    """

    tolerances: ApproachTolerances = field(default_factory=ApproachTolerances)
    """
    How finely the turn to the pour pose is followed.
    """

    def check(self, arm: Arms, pre_pour_pose: Pose, pour_pose: Pose) -> None:
        """
        :param arm: The arm holding the source.
        :param pre_pour_pose: The pose the tool starts the pour from.
        :param pour_pose: The tilted pose the tool pours at.
        :raises PourUnreachable: When inverse kinematics finds no arm configuration for
            a pose between them.
        """
        world = deepcopy(self.world)
        for dof in world.degrees_of_freedom:
            world.state[dof.id].position = self.held.joint_positions[str(dof.name)]
        world.notify_state_change()
        robot = world.get_semantic_annotations_by_type(AbstractRobot)[0]
        tool_frame = ViewManager.get_end_effector_view(arm, robot).tool_frame
        try:
            for pose in self.poses_between(
                pre_pour_pose.to_homogeneous_matrix().to_np(),
                pour_pose.to_homogeneous_matrix().to_np(),
                self.tolerances,
            ):
                solution = world.compute_inverse_kinematics(
                    robot.root,
                    tool_frame,
                    HomogeneousTransformationMatrix(pose, reference_frame=world.root),
                )
                for dof, position in solution.items():
                    world.state[dof.id].position = position
                world.notify_state_change()
        except IKSolverException as failed:
            raise PourUnreachable(
                reason=f"{type(failed).__name__}: {failed}"
            ) from failed

    @staticmethod
    def poses_between(
        start: np.ndarray, end: np.ndarray, tolerances: ApproachTolerances
    ) -> List[np.ndarray]:
        """
        :param start: The 4x4 pose the tool starts from.
        :param end: The 4x4 pose it ends at.
        :return: The poses from just after the start to the end, as 4x4 transforms: the
            position moving straight, the orientation turning in equal steps of at most
            :attr:`ApproachTolerances.max_rotation_per_step`, one step for a move
            without a turn.
        """
        turn = Slerp([0.0, 1.0], Rotation.from_matrix([start[:3, :3], end[:3, :3]]))
        angle = np.linalg.norm(
            Rotation.from_matrix(start[:3, :3].T @ end[:3, :3]).as_rotvec()
        )
        steps = max(1, math.ceil(angle / tolerances.max_rotation_per_step))
        poses = []
        for step in range(1, steps + 1):
            fraction = step / steps
            pose = np.eye(4)
            pose[:3, :3] = turn([fraction]).as_matrix()[0]
            pose[:3, 3] = (1 - fraction) * start[:3, 3] + fraction * end[:3, 3]
            poses.append(pose)
        return poses


# %% predicting


@dataclass
class PourPredictor:
    """
    Predicts where a pour ends, in a scene's world as it stands before anything is
    performed.
    """

    world: World
    """
    The scene's world, in which nothing is performed.
    """

    holding: Holding
    """
    Says how the gripper holds the source after the pick-up.
    """

    approach: ApproachTolerances = field(default_factory=ApproachTolerances)
    """
    How finely the tool's turn to the pour pose is followed when asking whether the arm
    can make it.
    """

    def predict(self, values: Mapping[str, Any]) -> PourOutcome:
        """
        :param values: The pour's values as data, by field name.
        :return: Where the source's mouth ends, against the target's opening, as the
            simulated pour measures it.
        """
        held = self.holding.held(values)
        execution = PouringExecution(world=self.world, values=dict(values))
        action = execution.action()
        sequential([action], context=execution.context())
        pre_pour_pose, pour_pose = action._pour_poses()
        PourReach(world=self.world, held=held, tolerances=self.approach).check(
            action.arm, pre_pour_pose, pour_pose
        )
        mouth = (
            pour_pose.to_homogeneous_matrix().to_np()
            @ held.source_in_tool_frame
            @ top_centre_in_own_frame(action.source_container.root)
        )[:3]
        return PourOutcome.at(mouth, action.target_container, held.robot_yaw)


# %% offered to the resolver


@dataclass
class PredictPourEnd(Tool):
    """
    Predicts where a pour named by its values ends, its containers by their handles.
    """

    predictor: PourPredictor
    """
    Predicts where a pour ends.
    """

    world: WorldAccess
    """
    The world the containers' handles name individuals of.
    """

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition.from_file(DEFINITION_FILE)

    def run(self, arguments: Mapping[str, Any]) -> ToolOutcome:
        values = {
            name: (
                SlotData.from_value(self.world.individual_of(value))
                if name in PouringSlot.containers()
                else value
            )
            for name, value in arguments.items()
        }
        return ToolOutcome(content=json.dumps(self.predicted_outcome(values)))

    def predicted_outcome(self, values: Mapping[str, Any]) -> Dict[str, Any]:
        """
        :return: The predicted outcome of the pour, as the simulated pour states one: a
            pick-up or a pour the arm cannot make is a failed motion, the resolver's to
            correct.
        """
        try:
            return self.predictor.predict(values).as_data()
        except PredictedMotionFailure as failed:
            return {
                OutcomeKey.LANDS_IN_OPENING: False,
                OutcomeKey.MOTION_FAILED: failed.reason,
            }


@dataclass
class PourPrediction(OutcomePrediction):
    """
    Offers the resolver the tool predicting where a pour ends.
    """

    predictor: PourPredictor
    """
    Predicts where a pour ends.
    """

    world: WorldAccess
    """
    The world the resolver names individuals of.
    """

    def tools(self) -> List[Tool]:
        return [PredictPourEnd(predictor=self.predictor, world=self.world)]

    @property
    def rule(self) -> PromptRule:
        return PREDICTION_RULE
