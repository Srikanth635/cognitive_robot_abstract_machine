"""
The actions the PR2 performs in the physics viewer, simple ones first.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum

from coraplex.datastructures.dataclasses import Context
from coraplex.plans.factories import sequential
from coraplex.plans.plan import Plan
from coraplex.robot_plans.actions.core.navigation import NavigateAction
from coraplex.robot_plans.actions.core.pick_up import PickUpAction
from coraplex.robot_plans.actions.core.robot_body import (
    MoveTorsoAction,
    SetGripperAction,
)
from semantic_digital_twin.datastructures.definitions import GripperState, TorsoState
from semantic_digital_twin.robots.robot_parts import AbstractRobot, Arm
from semantic_digital_twin.semantic_annotations.mixins import HasGraspCandidates
from semantic_digital_twin.spatial_types.spatial_types import Pose

from experiments.physics_viewer.exceptions import NoSuchBody
from experiments.physics_viewer.placement import Position


class ActionPlan(ABC):
    """
    How the plan of an action is built.
    """

    @abstractmethod
    def build(self, context: Context) -> Plan:
        """
        :param context: The world and the robot the plan is performed with.
        :return: The plan.
        """


@dataclass(frozen=True)
class TorsoPlan(ActionPlan):
    """
    Moves the torso to a height.
    """

    height: TorsoState
    """
    The height to move the torso to.
    """

    def build(self, context: Context) -> Plan:
        return sequential([MoveTorsoAction(self.height)], context=context).plan


@dataclass(frozen=True)
class GripperPlan(ActionPlan):
    """
    Sets the grippers of every arm.
    """

    state: GripperState
    """
    The state to set the grippers to.
    """

    def build(self, context: Context) -> Plan:
        return sequential(
            [
                SetGripperAction(arm.end_effector, self.state)
                for arm in context.robot.all_arms
            ],
            context=context,
        ).plan


@dataclass(frozen=True)
class NavigationPlan(ActionPlan):
    """
    Drives the robot's base to a spot on the floor.
    """

    target: Position
    """
    The spot to drive to.
    """

    def build(self, context: Context) -> Plan:
        return sequential([self.navigate(context)], context=context).plan

    def navigate(self, context: Context) -> NavigateAction:
        """
        :param context: The world and the robot the plan is performed with.
        :return: The action driving to the spot.
        """
        return NavigateAction(
            Pose.from_xyz_rpy(
                self.target.x,
                self.target.y,
                self.target.z,
                reference_frame=context.world.root,
            )
        )


class ArmSide(Enum):
    """
    Which arm of a robot performs an action.
    """

    LEFT = "left_arm"
    """
    The robot's left arm.
    """

    RIGHT = "right_arm"
    """
    The robot's right arm.
    """

    def arm_of(self, robot: AbstractRobot) -> Arm:
        """
        :param robot: The robot.
        :return: The arm on this side.
        """
        return robot.left_arm if self is ArmSide.LEFT else robot.right_arm


@dataclass(frozen=True)
class PickUpPlan(ActionPlan):
    """
    Picks a body up with one arm, after driving to where it can be reached from if the
    robot has to.
    """

    body: str
    """
    The name of the body to pick up, which the scene annotates as something to grasp.
    """

    arm: ArmSide
    """
    The arm that picks it up.
    """

    grasp_index: int
    """
    Which of the body's grasp candidates is taken.

    Measured, not chosen by a rule: the candidates the arm reaches are not all that lift
    the body under physics.
    """

    approach: NavigationPlan | None = None
    """
    Where the robot drives to first; ``None`` for a robot that does not move.
    """

    def build(self, context: Context) -> Plan:
        annotations = [
            annotation
            for annotation in context.world.get_semantic_annotations_by_type(
                HasGraspCandidates
            )
            if annotation.root.name.name == self.body
        ]
        if not annotations:
            raise NoSuchBody(self.body)
        steps = [] if self.approach is None else [self.approach.navigate(context)]
        steps.append(
            PickUpAction(
                annotations[0].grasp_candidates()[self.grasp_index],
                self.arm.arm_of(context.robot),
            )
        )
        return sequential(steps, context=context).plan


class PhysicsAction(Enum):
    """
    The actions the physics viewer has the robot perform.
    """

    RAISE_TORSO = TorsoPlan(TorsoState.HIGH)
    """
    The torso moves up to its highest position.
    """

    OPEN_GRIPPERS = GripperPlan(GripperState.OPEN)
    """
    The grippers of both arms open.
    """

    CLOSE_GRIPPERS = GripperPlan(GripperState.CLOSE)
    """
    The grippers of both arms close.
    """

    GO_TO_THE_COUNTER = NavigationPlan(Position(1.85, 2.2))
    """
    The base drives to the spot in front of the kitchen counter.
    """

    PICK_UP_MILK = PickUpPlan(
        "milk.stl", ArmSide.RIGHT, 1, NavigationPlan(Position(1.85, 2.2))
    )
    """
    The robot drives to the counter and picks up the milk carton with its right arm.
    """

    PICK_UP_CUBE = PickUpPlan("cube", ArmSide.LEFT, 6)
    """
    Tracy picks the cube up off its table with its left arm.
    """

    @classmethod
    def from_command_line(cls, name: str) -> PhysicsAction:
        """
        :param name: The action's name in lower case, as typed.
        :return: The action.
        :raises KeyError: When no action has the name.
        """
        return cls[name.upper()]
