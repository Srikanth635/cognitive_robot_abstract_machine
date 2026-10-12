"""
Performing a run's resolved pouring action in the simulation, from the environment the
run started from: the scene's world built anew, the robot readied with its arms parked
and its torso lowered, brought to face the source container, which it picks up with the
resolved arm, and the pouring performed as resolved.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np
from coraplex.datastructures.dataclasses import Context
from coraplex.datastructures.enums import ApproachDirection, Arms, VerticalAlignment
from coraplex.datastructures.grasp import GraspDescription
from coraplex.execution_environment import simulated_robot
from coraplex.language import SequentialNode
from coraplex.plans.factories import sequential
from coraplex.robot_plans.actions.composite.tool_based import PouringAction
from coraplex.robot_plans.actions.core.navigation import NavigateAction
from coraplex.robot_plans.actions.core.pick_up import PickUpAction
from coraplex.robot_plans.actions.core.robot_body import MoveTorsoAction, ParkArmsAction
from coraplex.view_manager import ViewManager
from semantic_digital_twin.datastructures.definitions import TorsoState
from semantic_digital_twin.robots.robot_parts import AbstractRobot
from semantic_digital_twin.semantic_annotations.mixins import HasRootBody
from semantic_digital_twin.spatial_types import Pose
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.world_entity import Body
from typing_extensions import Any

from cram_viz.designator_knowledge import resolved_field_type
from cram_viz.resolution.resolver import ResolvedDesignator
from cram_viz.resolution.runs import RunDirectory
from cram_viz.resolution.scene_world import SceneWorldBuilder

REACHING_TORSO = TorsoState.LOW
"""
The torso height the robot reaches for the source container from: raised, the PR2's
gripper cannot come down to an object at counter height.
"""

PICK_UP_DISTANCE = 0.63
"""
How far from the source container, in metres, the robot stands to pick it up: the
distance the recorded simple_pouring demo picked its pitcher up from.
"""

# %% what cannot be performed


@dataclass
class NotAPouring(TypeError):
    """
    The resolution is of an action other than pouring.
    """

    designator: str
    """
    The name of the resolved action's class.
    """

    def __post_init__(self):
        super().__init__(f"a {self.designator} is not a pouring action")


@dataclass
class NoRobotInWorld(LookupError):
    """
    The world holds no robot to perform the action.
    """

    def __post_init__(self):
        super().__init__("the world holds no robot to perform the action")


@dataclass
class NoObjectInWorld(LookupError):
    """
    The world holds no object of the kind the resolution names.
    """

    kind: type
    """
    The kind of object asked for.
    """

    name: str
    """
    The name of the object's body.
    """

    def __post_init__(self):
        super().__init__(f"the world holds no {self.kind.__name__} named {self.name}")


# %% where the robot stands


def standing_pose(world: World, base: Body, source: Body) -> Pose:
    """
    :param world: The world the robot stands in.
    :param base: The robot's base.
    :param source: The body to pick up.
    :return: The pose the base stands at to pick the source up: facing it,
        :data:`PICK_UP_DISTANCE` away along the line from where the base stands now.
    """
    world_P_base = base.global_pose.to_position().to_np()[:2]
    world_P_source = source.global_pose.to_position().to_np()[:2]
    heading = world_P_source - world_P_base
    heading = heading / np.linalg.norm(heading)
    world_P_stand = world_P_source - heading * PICK_UP_DISTANCE
    return Pose.from_xyz_rpy(
        x=float(world_P_stand[0]),
        y=float(world_P_stand[1]),
        yaw=float(np.arctan2(heading[1], heading[0])),
        reference_frame=world.root,
    )


# %% performing


@dataclass
class PouringExecution:
    """
    Performs a resolved pouring action in a world: the robot parks its arms, lowers its
    torso, faces the source container, picks it up with the resolved arm, and pours.
    """

    world: World
    """
    The world the action is performed in.
    """

    resolution: ResolvedDesignator
    """
    The resolution whose first action is performed.
    """

    def __post_init__(self):
        if self.resolution.designator != PouringAction.__name__:
            raise NotAPouring(designator=self.resolution.designator)

    @classmethod
    def of_run(cls, run: RunDirectory) -> PouringExecution:
        """
        :return: The execution of a run's resolution, in the world of the scene the run
            started from, built anew, its robot annotated as what it is.
        """
        builder = SceneWorldBuilder()
        bundle = run.environment().bundle
        world = builder.build(bundle)
        builder.annotate_robot(world, bundle)
        return cls(world=world, resolution=run.resolution())

    @property
    def robot(self) -> AbstractRobot:
        """
        :return: The robot performing the action.
        :raises NoRobotInWorld: When the world holds none.
        """
        robots = self.world.get_semantic_annotations_by_type(AbstractRobot)
        if not robots:
            raise NoRobotInWorld()
        return robots[0]

    def action(self) -> PouringAction:
        """
        :return: The pouring action the resolution resolved, its values found again in
            the world.
        """
        resolved, *_ = self.resolution.actions
        return PouringAction(
            **{
                slot.name: self.value_in_world(
                    resolved_field_type(PouringAction, slot.name), slot.serialized
                )
                for slot in resolved.slots
            }
        )

    def value_in_world(self, kind: Any, serialized: Any) -> Any:
        """
        :param kind: The type the slot declares.
        :param serialized: The slot's value as the run keeps it.
        :return: The value: a body by its name, an annotated object by its body's name,
            an enumeration member by its name, any other value as it is.
        :raises NoObjectInWorld: When the world holds no such object.
        """
        if not isinstance(kind, type):
            return serialized
        if issubclass(kind, Body):
            return self.world.get_body_by_name(serialized)
        if issubclass(kind, Enum):
            return kind[serialized]
        if not issubclass(kind, HasRootBody):
            return serialized
        found = [
            annotation
            for annotation in self.world.get_semantic_annotations_by_type(kind)
            if annotation.root.name.name == serialized
        ]
        if not found:
            raise NoObjectInWorld(kind=kind, name=serialized)
        return found[0]

    def plan(self) -> SequentialNode:
        """
        :return: The plan: park the arms, lower the torso, face the source container,
            pick it up with the resolved arm, pour.
        """
        action = self.action()
        robot = self.robot
        grasp = GraspDescription(
            approach_direction=ApproachDirection.FRONT,
            vertical_alignment=VerticalAlignment.NoAlignment,
            end_effector=ViewManager.get_end_effector_view(action.arm, robot),
        )
        return sequential(
            [
                ParkArmsAction(Arms.BOTH),
                MoveTorsoAction(REACHING_TORSO),
                NavigateAction(
                    standing_pose(self.world, robot.root, action.source_container.root)
                ),
                PickUpAction(action.source_container, action.arm, grasp),
                action,
            ],
            context=Context(world=self.world, robot=robot, evaluate_conditions=False),
        )

    def perform(self) -> None:
        """
        Perform the plan in the simulation.
        """
        with simulated_robot:
            self.plan().perform()
