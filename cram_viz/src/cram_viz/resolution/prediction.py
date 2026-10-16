"""
Predicting where a pour ends, before it is answered.

coraplex's pouring computes its tool poses when its plan is built, before the source
container is picked up, so the tool pose for given values is the one computed in the
scene's world as it stands; carried with how the gripper holds the source after the
pick-up, it puts the source's mouth where the simulated pour measures it. How the
gripper holds the source is measured once per scene and arm, by a simulated pick-up in a
process of its own.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

import numpy as np
from coraplex.execution_environment import simulated_robot
from coraplex.plans.factories import sequential
from coraplex.view_manager import ViewManager
from krrood.adapters.json_serializer import SubclassJSONSerializer
from llm_reasoner.clients.tools import Tool, ToolDefinition, ToolOutcome
from llm_reasoner.resolver.outcomes import OutcomePrediction
from llm_reasoner.resolver.prompt_format import PromptRule, RuleName
from llm_reasoner.world.access import WorldAccess
from semantic_digital_twin.world import World
from typing_extensions import Any, Dict, List, Mapping, Self

from cram_viz.resolution.execution import PouringExecution
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

CONTAINER_SLOTS = ("source_container", "target_container")
"""
The slots of a pour the tool takes as world handles.
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
class PickUpFailed(Exception):
    """
    The robot could not pick the source container up with the arm a pour names, so no
    pour with it can be predicted.
    """

    reason: str
    """
    Why the pick-up failed, as the simulation said.
    """

    def __post_init__(self):
        super().__init__(f"the pick-up failed: {self.reason}")


class HeldSourceKey(StrEnum):
    """
    The keys of a held source's JSON form.
    """

    SOURCE_IN_TOOL_FRAME = "source_in_tool_frame"
    ROBOT_YAW = "robot_yaw"


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

    def to_json(self) -> Dict[str, Any]:
        return {
            **super().to_json(),
            HeldSourceKey.SOURCE_IN_TOOL_FRAME: self.source_in_tool_frame.tolist(),
            HeldSourceKey.ROBOT_YAW: self.robot_yaw,
        }

    @classmethod
    def _from_json(cls, data: Dict[str, Any], **kwargs) -> Self:
        return cls(
            source_in_tool_frame=np.array(data[HeldSourceKey.SOURCE_IN_TOOL_FRAME]),
            robot_yaw=float(data[HeldSourceKey.ROBOT_YAW]),
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
        _, pour_pose = action._pour_poses()
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
                if name in CONTAINER_SLOTS
                else value
            )
            for name, value in arguments.items()
        }
        return ToolOutcome(content=json.dumps(self.predicted_outcome(values)))

    def predicted_outcome(self, values: Mapping[str, Any]) -> Dict[str, Any]:
        """
        :return: The predicted outcome of the pour, as the simulated pour states one: a
            pick-up the arm cannot make is a failed motion, the resolver's to correct.
        """
        try:
            return self.predictor.predict(values).as_data()
        except PickUpFailed as failed:
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
