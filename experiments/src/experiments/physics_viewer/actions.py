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
from semantic_digital_twin.semantic_annotations.semantic_annotations import Milk
from semantic_digital_twin.spatial_types.spatial_types import Pose

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


@dataclass(frozen=True)
class PickUpMilkPlan(ActionPlan):
    """
    Drives to a spot in front of the milk carton and picks it up with the right arm.
    """

    approach: NavigationPlan
    """
    Where the robot drives to first.
    """

    grasp_index: int
    """
    Which of the milk's grasp candidates is taken.

    The first is one the right arm does not reach from where the robot stands; the
    second is the first that it does.
    """

    def build(self, context: Context) -> Plan:
        [milk] = context.world.get_semantic_annotations_by_type(Milk)
        return sequential(
            [
                self.approach.navigate(context),
                PickUpAction(
                    milk.grasp_candidates()[self.grasp_index], context.robot.right_arm
                ),
            ],
            context=context,
        ).plan


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

    PICK_UP_MILK = PickUpMilkPlan(NavigationPlan(Position(1.85, 2.2)), grasp_index=1)
    """
    The robot drives to the counter and picks up the milk carton with its right arm.
    """

    @classmethod
    def from_command_line(cls, name: str) -> PhysicsAction:
        """
        :param name: The action's name in lower case, as typed.
        :return: The action.
        :raises KeyError: When no action has the name.
        """
        return cls[name.upper()]
