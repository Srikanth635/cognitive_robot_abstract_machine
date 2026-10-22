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
from coraplex.robot_plans.actions.core.robot_body import (
    MoveTorsoAction,
    SetGripperAction,
)
from semantic_digital_twin.datastructures.definitions import GripperState, TorsoState


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

    @classmethod
    def from_command_line(cls, name: str) -> PhysicsAction:
        """
        :param name: The action's name in lower case, as typed.
        :return: The action.
        :raises KeyError: When no action has the name.
        """
        return cls[name.upper()]
