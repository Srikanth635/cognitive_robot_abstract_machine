"""
Performing a run's resolved pouring action in the simulation, from the environment the
run started from: the scene's world built anew, the robot readied with its arms parked
and its torso lowered, brought to face the source container, which it picks up with the
resolved arm, then turned to face the target container, and the pouring performed as
resolved.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from coraplex.plans.context_extensions import RobotAccess, StatementGrounding
from coraplex.plans.executors import KinematicallySimulatedPlanExecutor
from coraplex.robot_plans.actions.base import Action
from coraplex.robot_plans.actions.composite.tool_based import PouringAction
from coraplex.robot_plans.actions.core.navigation import NavigateAction
from coraplex.robot_plans.actions.core.pick_up import PickUpAction
from coraplex.robot_plans.actions.core.robot_body import MoveTorsoAction, ParkArmsAction
from cramph.composites import Sequence
from cramph.context import StatechartContext
from cramph.statechart import Statechart
from cramph.visualization import WorldVisualization
from semantic_digital_twin.datastructures.definitions import TorsoState
from semantic_digital_twin.grasping.grasp_candidates import GraspCandidate
from semantic_digital_twin.robots.robot_parts import AbstractRobot
from semantic_digital_twin.semantic_annotations.mixins import HasGraspCandidates
from krrood_spatial_math.spatial_types import Pose
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.world_entity import Body
from typing_extensions import TYPE_CHECKING, Any, Dict, List, Optional

from cramera.resolution.designator_knowledge import resolved_field_type
from cramera.resolution.neem import NeemRecorder, NoNeemRecorder
from cramera.resolution.resolver.runs import RunDirectory
from cramera.resolution.resolver.scene_world import SceneWorldBuilder
from cramera.resolution.resolver.slot_data import SlotData

if TYPE_CHECKING:
    from cramera.resolution.resolver.resolver import ResolvedDesignator

REACHING_TORSO = TorsoState.LOW
"""
The torso height the robot reaches for the source container from: raised, the PR2's
gripper cannot come down to an object at counter height.
"""

POURING_TORSO = TorsoState.HIGH
"""
The torso height the robot pours from.

Raised before the pour, the arm tilts the source at once: left low, the torso rises
slowly while the arm waits, which the simulation's stall rule gives up on.
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


# %% where the robot stands


def standing_pose(world: World, base: Body, source: Body) -> Pose:
    """
    :param world: The world the robot stands in.
    :param base: The robot's base.
    :param source: The body to pick up.
    :return: The pose the base stands at to pick the source up: facing it,
        :data:`PICK_UP_DISTANCE` away along the line from where the base stands now.
    """
    world_P_base = base.global_pose.position.to_np()[:2]
    world_P_source = source.global_pose.position.to_np()[:2]
    heading = world_P_source - world_P_base
    heading = heading / np.linalg.norm(heading)
    world_P_stand = world_P_source - heading * PICK_UP_DISTANCE
    return Pose.from_xyz_rpy(
        x=float(world_P_stand[0]),
        y=float(world_P_stand[1]),
        yaw=float(np.arctan2(heading[1], heading[0])),
        reference_frame=world.root,
    )


def facing_pose(standing: Pose, target: Body, world: World) -> Pose:
    """
    :param standing: The pose the robot stands at.
    :param target: The body to face.
    :param world: The world the robot stands in.
    :return: The pose at the same place turned in place to face the target.
    """
    world_P_stand = standing.position.to_np()[:2]
    world_P_target = target.global_pose.position.to_np()[:2]
    heading = world_P_target - world_P_stand
    return Pose.from_xyz_rpy(
        x=float(world_P_stand[0]),
        y=float(world_P_stand[1]),
        yaw=float(np.arctan2(heading[1], heading[0])),
        reference_frame=world.root,
    )


def grasp_facing(graspable: HasGraspCandidates, standing: Pose) -> GraspCandidate:
    """
    :param graspable: The object to take hold of.
    :param standing: The pose the robot stands at, facing the object.
    :return: The grasp the object offers whose approach direction runs most nearly the
        way the robot faces, so the gripper comes at the object from the robot's side.
    """
    heading = standing.to_np()[:2, 0]
    return max(
        graspable.grasp_candidates(),
        key=lambda grasp: float(grasp.world_T_grasp.to_np()[:2, 0] @ heading),
    )


# %% performing


@dataclass
class PouringExecution:
    """
    Performs a resolved pouring action in a world: the robot parks its arms, lowers its
    torso, faces the source container, picks it up with the resolved arm, turns to face
    the target container, and pours.
    """

    world: World
    """
    The world the action is performed in.
    """

    values: Dict[str, Any]
    """
    The resolved slots' values as data, by field name.
    """

    recorder: NeemRecorder = field(default_factory=NoNeemRecorder)
    """
    Keeps the plan performed as a NEEM; nothing is kept by default.
    """

    visualization: Optional[WorldVisualization] = None
    """
    Shows the world and the plan while the plan is performed; nothing is shown by
    default.
    """

    @classmethod
    def for_resolution(
        cls,
        world: World,
        resolution: ResolvedDesignator,
        recorder: Optional[NeemRecorder] = None,
    ) -> PouringExecution:
        """
        :return: The execution of a resolution's first action in the world.
        :raises NotAPouring: When the resolution is of another action.
        """
        if resolution.designator != PouringAction.__name__:
            raise NotAPouring(designator=resolution.designator)
        resolved, *_ = resolution.actions
        return cls(
            world=world,
            values={slot.name: slot.serialized for slot in resolved.slots},
            recorder=recorder or NoNeemRecorder(),
        )

    @classmethod
    def of_run(
        cls, run: RunDirectory, recorder: Optional[NeemRecorder] = None
    ) -> PouringExecution:
        """
        :return: The execution of a run's resolution, in the world of the scene the run
            started from, built anew, its robot annotated as what it is.
        """
        return cls.for_resolution(
            SceneWorldBuilder.robot_world(run.environment().bundle),
            run.resolution(),
            recorder,
        )

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
        :return: The pouring action resolved, its values found again in the world.
        """
        return PouringAction(
            **{
                name: SlotData.to_value(
                    self.world, resolved_field_type(PouringAction, name), data
                )
                for name, data in self.values.items()
            }
        )

    def plan(self) -> Sequence:
        """
        :return: The plan: park the arms, lower the torso, face the source container,
            pick it up with the resolved arm, face the target container, pour.
        """
        action = self.action()
        return Sequence(nodes=[*self.steps_before_pour(action), action])

    def steps_before_pour(self, action: PouringAction) -> List[Action]:
        """
        :param action: The pouring the steps prepare.
        :return: Parking the arms, lowering the torso, facing the source container,
            picking it up with the action's arm, raising the torso and turning in place to
            face the target container, which brings the pour poses within the arm's reach.
        """
        robot = self.robot
        standing = standing_pose(self.world, robot.root, action.source_container.root)
        return [
            ParkArmsAction(arms=[robot.left_arm, robot.right_arm]),
            MoveTorsoAction(torso_state=REACHING_TORSO),
            NavigateAction(target_location=standing),
            PickUpAction(
                grasp=grasp_facing(action.source_container, standing), arm=action.arm
            ),
            MoveTorsoAction(torso_state=POURING_TORSO),
            NavigateAction(
                target_location=facing_pose(
                    standing, action.target_container, self.world
                )
            ),
        ]

    def executor(self) -> KinematicallySimulatedPlanExecutor:
        """
        :return: A new executor simulating the robot's plans in the world, shown by
            :attr:`visualization` if there is one.
        """
        return KinematicallySimulatedPlanExecutor(
            StatechartContext.with_extensions(
                self.world, [RobotAccess(self.robot), StatementGrounding()]
            ),
            extensions=[] if self.visualization is None else [self.visualization],
        )

    def statechart_holding(
        self, *, executor: KinematicallySimulatedPlanExecutor, plan: Action | Sequence
    ) -> Statechart:
        """
        :param executor: The executor the plan is built for.
        :param plan: The plan to build.
        :return: A statechart in the context of the executor holding the plan, which
            builds the plan's steps.
        """
        statechart = Statechart(context=executor.context)
        statechart.add_node(plan)
        return statechart

    def perform_plan(self, *, plan: Action | Sequence) -> Statechart:
        """
        Perform a plan in the simulation until it ends.

        :param plan: The plan to perform.
        :return: The statechart the plan was performed as.
        """
        executor = self.executor()
        statechart = self.statechart_holding(executor=executor, plan=plan)
        executor.compile(statechart)
        executor.execute()
        return statechart

    def perform(self) -> None:
        """
        Perform the plan in the simulation, and have the recorder keep the statechart it
        was performed as, however it ended.
        """
        executor = self.executor()
        statechart = self.statechart_holding(executor=executor, plan=self.plan())
        try:
            executor.compile(statechart)
            executor.execute()
        finally:
            self.recorder.record(statechart)
