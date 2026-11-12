"""
The actions the robots perform in the physics viewer: one demonstration of each kind of
action coraplex offers, each made of the actions it performs one after the other.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum, StrEnum

from coraplex.plans.context_extensions import RobotAccess
from coraplex.robot_plans.actions.base import Action
from coraplex.robot_plans.actions.composite.facing import FaceAndLookAtAction
from coraplex.robot_plans.actions.composite.tool_based import (
    MixingAction,
    PouringAction,
    WipingAction,
)
from coraplex.robot_plans.actions.composite.transporting import (
    MoveAndOpenAction,
    MoveAndPickUpAction,
    MoveAndPlaceAction,
    PickAndPlaceAction,
    TransportAction,
)
from coraplex.robot_plans.actions.core.container import CloseAction, OpenAction
from coraplex.robot_plans.actions.core.navigation import (
    FaceAtAction,
    LookAtAction,
    NavigateAction,
)
from coraplex.robot_plans.actions.core.pick_up import (
    GraspingAction,
    PickUpAction,
    ReachAction,
)
from coraplex.robot_plans.actions.core.placing import PlaceAction
from coraplex.robot_plans.actions.core.robot_body import (
    MoveManipulatorAction,
    MoveTorsoAction,
    ParkArmsAction,
    SetGripperAction,
)
from cramph.composites import Sequence
from cramph.context import StatechartContext
from semantic_digital_twin.datastructures.definitions import GripperState, TorsoState
from semantic_digital_twin.grasping.grasp_candidates import GraspCandidate
from semantic_digital_twin.robots.robot_parts import AbstractRobot, Arm
from semantic_digital_twin.semantic_annotations.mixins import HasGraspCandidates
from semantic_digital_twin.semantic_annotations.semantic_annotations import (
    Handle,
    Tool,
)
from krrood_spatial_math.spatial_types import Pose

from experiments.physics_viewer.exceptions import NoSuchBody, NoToolInHand
from experiments.physics_viewer.placement import Position

# %% how a plan is built


class ActionPlan(ABC):
    """
    How the plan of an action is built.
    """

    @abstractmethod
    def actions(self, *, context: StatechartContext) -> list[Action]:
        """
        :param context: The context the plan is performed in, holding the world and the
            :class:`~coraplex.plans.context_extensions.RobotAccess` of the robot.
        :return: The actions the plan performs, in order.
        """

    def build(self, *, context: StatechartContext) -> Sequence:
        """
        :param context: The context the plan is performed in, holding the world and the
            :class:`~coraplex.plans.context_extensions.RobotAccess` of the robot.
        :return: The plan performing :meth:`actions` one after the other.
        """
        return Sequence(nodes=self.actions(context=context))

    @staticmethod
    def performing_robot(*, context: StatechartContext) -> AbstractRobot:
        """
        :param context: The context the plan is performed in.
        :return: The robot performing the plan.
        """
        return context.require_extension(RobotAccess).robot


@dataclass(frozen=True, kw_only=True)
class PlanSequence(ActionPlan):
    """
    Several plans performed one after the other.
    """

    plans: tuple[ActionPlan, ...]
    """
    The plans, in the order they are performed.
    """

    def actions(self, *, context: StatechartContext) -> list[Action]:
        return [
            action for plan in self.plans for action in plan.actions(context=context)
        ]


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


@dataclass(frozen=True, kw_only=True)
class ArmPlan(ActionPlan, ABC):
    """
    An action one arm of the robot performs.
    """

    arm: ArmSide
    """
    The arm that performs the action.
    """

    def performing_arm(self, *, context: StatechartContext) -> Arm:
        """
        :param context: The context the plan is performed in.
        :return: The arm of the performing robot on :attr:`arm`'s side.
        """
        return self.arm.arm_of(self.performing_robot(context=context))


# %% the robot's body


@dataclass(frozen=True, kw_only=True)
class TorsoPlan(ActionPlan):
    """
    Moves the torso to a height.
    """

    height: TorsoState
    """
    The height to move the torso to.
    """

    def actions(self, *, context: StatechartContext) -> list[Action]:
        return [MoveTorsoAction(torso_state=self.height)]


@dataclass(frozen=True, kw_only=True)
class GripperPlan(ActionPlan):
    """
    Sets the grippers of every arm.
    """

    state: GripperState
    """
    The state to set the grippers to.
    """

    def actions(self, *, context: StatechartContext) -> list[Action]:
        return [
            SetGripperAction(
                configuration=arm.end_effector.default_configuration(self.state)
            )
            for arm in self.performing_robot(context=context).all_arms
        ]


@dataclass(frozen=True, kw_only=True)
class ParkArmsPlan(ActionPlan):
    """
    Folds every arm into its parking pose beside the body.
    """

    def actions(self, *, context: StatechartContext) -> list[Action]:
        return [ParkArmsAction(arms=self.performing_robot(context=context).all_arms)]


@dataclass(frozen=True, kw_only=True)
class ArmToPositionPlan(ArmPlan):
    """
    Moves the tool frame of an arm's hand to a position, the hand pointing along the
    world's x axis.
    """

    position: Position
    """
    Where the tool frame goes.
    """

    def actions(self, *, context: StatechartContext) -> list[Action]:
        return [
            MoveManipulatorAction(
                target_pose=self.position.pose_in(context.world),
                end_effector=self.performing_arm(context=context).end_effector,
                allow_gripper_collision=False,
            )
        ]


@dataclass(frozen=True, kw_only=True)
class NavigationPlan(ActionPlan):
    """
    Drives the robot's base to a spot on the floor.
    """

    target: Position
    """
    The spot to drive to.
    """

    heading: float = 0.0
    """
    The angle, in radians, the base ends turned by from the world's x axis.
    """

    def actions(self, *, context: StatechartContext) -> list[Action]:
        return [
            NavigateAction(
                target_location=self.target.pose_in(context.world, self.heading)
            )
        ]


# %% turning towards a body


@dataclass(frozen=True, kw_only=True)
class BodyPlan(ActionPlan, ABC):
    """
    An action aimed at a body of the scene.
    """

    body: str
    """
    The name of the body.
    """

    def body_pose(self, *, context: StatechartContext) -> Pose:
        """
        :param context: The context the plan is performed in.
        :return: Where the body is now.
        """
        return context.world.get_body_by_name(self.body).global_pose


@dataclass(frozen=True, kw_only=True)
class LookAtPlan(BodyPlan):
    """
    Turns the head to look at a body.
    """

    def actions(self, *, context: StatechartContext) -> list[Action]:
        return [LookAtAction(target=self.body_pose(context=context))]


@dataclass(frozen=True, kw_only=True)
class FaceAtPlan(BodyPlan):
    """
    Turns the base on the spot to face a body.
    """

    def actions(self, *, context: StatechartContext) -> list[Action]:
        return [FaceAtAction(target=self.body_pose(context=context))]


@dataclass(frozen=True, kw_only=True)
class FaceAndLookAtPlan(BodyPlan):
    """
    Turns the base to face a body and the head to look at it.
    """

    def actions(self, *, context: StatechartContext) -> list[Action]:
        pose = self.body_pose(context=context)
        return [
            FaceAndLookAtAction(
                face_at=FaceAtAction(target=pose), look_at=LookAtAction(target=pose)
            )
        ]


# %% grasping, carrying and putting down


@dataclass(frozen=True, kw_only=True)
class GraspablePlan(ActionPlan, ABC):
    """
    An action on a body the scene annotates as something to grasp.
    """

    body: str
    """
    The name of the body.
    """

    def graspable(self, *, context: StatechartContext) -> HasGraspCandidates:
        """
        :param context: The context the plan is performed in.
        :return: The body's annotation as something to grasp.
        :raises NoSuchBody: When the scene annotates no body of that name as something
            to grasp.
        """
        annotations = [
            annotation
            for annotation in context.world.get_semantic_annotations_by_type(
                HasGraspCandidates
            )
            if annotation.root.name.name == self.body
        ]
        if not annotations:
            raise NoSuchBody(self.body)
        return annotations[0]


@dataclass(frozen=True, kw_only=True)
class GraspPlan(GraspablePlan, ArmPlan, ABC):
    """
    An action that takes a body with one arm, by one of the body's grasp candidates.
    """

    grasp_index: int
    """
    Which of the body's grasp candidates is taken.

    Measured, not chosen by a rule: the candidates the arm reaches are not all that lift
    the body under physics.
    """

    def grasp(self, *, context: StatechartContext) -> GraspCandidate:
        """
        :param context: The context the plan is performed in.
        :return: The grasp candidate taken.
        """
        return self.graspable(context=context).grasp_candidates()[self.grasp_index]


@dataclass(frozen=True, kw_only=True)
class ReachPlan(GraspPlan):
    """
    Moves the hand around the body, as for a grasp, without closing it.
    """

    def actions(self, *, context: StatechartContext) -> list[Action]:
        return [
            ReachAction(
                arm=self.performing_arm(context=context),
                grasp=self.grasp(context=context),
            )
        ]


@dataclass(frozen=True, kw_only=True)
class GraspingPlan(GraspPlan):
    """
    Grasps the body where it stands, without lifting it.
    """

    def actions(self, *, context: StatechartContext) -> list[Action]:
        return [
            GraspingAction(
                grasp=self.grasp(context=context),
                arm=self.performing_arm(context=context),
            )
        ]


@dataclass(frozen=True, kw_only=True)
class PickUpPlan(GraspPlan):
    """
    Grasps the body and lifts it.
    """

    def actions(self, *, context: StatechartContext) -> list[Action]:
        return [
            PickUpAction(
                grasp=self.grasp(context=context),
                arm=self.performing_arm(context=context),
            )
        ]


@dataclass(frozen=True, kw_only=True)
class MoveAndPickUpPlan(GraspPlan):
    """
    Drives to a spot, turns towards the body and picks it up.
    """

    standing: Position
    """
    The spot on the floor the body is picked up from.
    """

    def actions(self, *, context: StatechartContext) -> list[Action]:
        return [
            MoveAndPickUpAction.from_standing_position(
                self.standing.pose_in(context.world),
                self.grasp(context=context),
                self.performing_arm(context=context),
            )
        ]


@dataclass(frozen=True, kw_only=True)
class PickAndPlacePlan(GraspPlan):
    """
    Picks the body up and puts it down elsewhere within reach.
    """

    target: Position
    """
    Where the body is put down.
    """

    def actions(self, *, context: StatechartContext) -> list[Action]:
        return [
            PickAndPlaceAction(
                pick_up=PickUpAction(
                    grasp=self.grasp(context=context),
                    arm=self.performing_arm(context=context),
                ),
                place=PlaceAction(
                    object_designator=self.graspable(context=context),
                    target_location=self.target.pose_in(context.world),
                ),
            )
        ]


@dataclass(frozen=True, kw_only=True)
class TransportPlan(GraspPlan):
    """
    Picks the body up at one spot, carries it with the arms parked and puts it down from
    another.
    """

    standing: Position
    """
    The spot on the floor the body is picked up from.
    """

    target: Position
    """
    Where the body is put down.
    """

    placing_standing: Position
    """
    The spot on the floor the body is put down from.
    """

    def actions(self, *, context: StatechartContext) -> list[Action]:
        arm = self.performing_arm(context=context)
        return [
            TransportAction(
                pick_up=MoveAndPickUpAction.from_standing_position(
                    self.standing.pose_in(context.world),
                    self.grasp(context=context),
                    arm,
                ),
                place=MoveAndPlaceAction.from_standing_position(
                    self.placing_standing.pose_in(context.world),
                    self.target.pose_in(context.world),
                    self.graspable(context=context),
                    arm,
                ),
            )
        ]


@dataclass(frozen=True, kw_only=True)
class PlacePlan(GraspablePlan):
    """
    Puts the held body down.
    """

    target: Position
    """
    Where the body is put down.
    """

    def actions(self, *, context: StatechartContext) -> list[Action]:
        return [
            PlaceAction(
                object_designator=self.graspable(context=context),
                target_location=self.target.pose_in(context.world),
            )
        ]


@dataclass(frozen=True, kw_only=True)
class MoveAndPlacePlan(GraspablePlan, ArmPlan):
    """
    Drives to a spot, turns towards the target and puts the held body down.
    """

    target: Position
    """
    Where the body is put down.
    """

    standing: Position
    """
    The spot on the floor the body is put down from.
    """

    def actions(self, *, context: StatechartContext) -> list[Action]:
        return [
            MoveAndPlaceAction.from_standing_position(
                self.standing.pose_in(context.world),
                self.target.pose_in(context.world),
                self.graspable(context=context),
                self.performing_arm(context=context),
            )
        ]


# %% drawers and doors


@dataclass(frozen=True, kw_only=True)
class HandlePlan(ArmPlan, ABC):
    """
    An action on a container, by its handle.
    """

    handle: str
    """
    The name of the handle's body.
    """

    def handle_grasp(self, *, context: StatechartContext) -> GraspCandidate:
        """
        :param context: The context the plan is performed in.
        :return: The grasp of the handle at its origin, annotating the body as a handle
            if the scene does not.
        """
        body = context.world.get_body_by_name(self.handle)
        handle = next(
            (
                handle
                for handle in context.world.get_semantic_annotations_by_type(Handle)
                if handle.root is body
            ),
            None,
        )
        if handle is None:
            handle = Handle(root=body)
            with context.world.modify_world():
                context.world.add_semantic_annotation_recursively(handle)
        return GraspCandidate.from_body_origin(handle)


@dataclass(frozen=True, kw_only=True)
class OpenPlan(HandlePlan):
    """
    Pulls the container open by its handle.
    """

    def actions(self, *, context: StatechartContext) -> list[Action]:
        return [
            OpenAction(
                grasp=self.handle_grasp(context=context),
                arm=self.performing_arm(context=context),
            )
        ]


@dataclass(frozen=True, kw_only=True)
class ClosePlan(HandlePlan):
    """
    Pushes the container shut by its handle.
    """

    def actions(self, *, context: StatechartContext) -> list[Action]:
        return [
            CloseAction(
                grasp=self.handle_grasp(context=context),
                arm=self.performing_arm(context=context),
            )
        ]


@dataclass(frozen=True, kw_only=True)
class MoveAndOpenPlan(HandlePlan):
    """
    Drives to a spot, turns towards the handle and pulls the container open.
    """

    standing: Position
    """
    The spot on the floor the container is opened from.
    """

    def actions(self, *, context: StatechartContext) -> list[Action]:
        return [
            MoveAndOpenAction.from_standing_position(
                self.standing.pose_in(context.world),
                self.handle_grasp(context=context),
                self.performing_arm(context=context),
            )
        ]


# %% working with a tool


@dataclass(frozen=True, kw_only=True)
class ToolPlan(ArmPlan, ABC):
    """
    An action performed with the tool an arm's hand holds.
    """

    def held_tool(self, *, context: StatechartContext) -> Tool:
        """
        :param context: The context the plan is performed in.
        :return: The tool fixed in the hand.
        :raises NoToolInHand: When the hand holds no tool.
        """
        tool_frame = self.performing_arm(context=context).end_effector.tool_frame
        for tool in context.world.get_semantic_annotations_by_type(Tool):
            if tool.root.parent_connection.parent is tool_frame:
                return tool
        raise NoToolInHand(self.arm.value)


@dataclass(frozen=True, kw_only=True)
class PouringPlan(ToolPlan):
    """
    Pours from the held container into another.
    """

    container: str
    """
    The name of the body poured into.
    """

    def actions(self, *, context: StatechartContext) -> list[Action]:
        return [
            PouringAction(
                target_container=context.world.get_body_by_name(self.container),
                source_container=self.held_tool(context=context),
                arm=self.performing_arm(context=context),
            )
        ]


@dataclass(frozen=True, kw_only=True)
class MixingPlan(ToolPlan):
    """
    Stirs inside a container with the held tool.
    """

    container: str
    """
    The name of the body stirred in.
    """

    def actions(self, *, context: StatechartContext) -> list[Action]:
        return [
            MixingAction(
                container=context.world.get_body_by_name(self.container),
                arm=self.performing_arm(context=context),
                tool=self.held_tool(context=context),
            )
        ]


@dataclass(frozen=True, kw_only=True)
class WipingPlan(ToolPlan):
    """
    Wipes a surface with the held tool.
    """

    target: Position
    """
    The middle of the stretch of surface wiped.
    """

    def actions(self, *, context: StatechartContext) -> list[Action]:
        return [
            WipingAction(
                arm=self.performing_arm(context=context),
                tool=self.held_tool(context=context),
                target_pose=self.target.pose_in(context.world),
            )
        ]


# %% where the demonstrations happen


class SceneBody(StrEnum):
    """
    The bodies the demonstrations act on.
    """

    MILK = "milk.stl"
    """
    The milk carton on the counter.
    """

    BOWL = "bowl.stl"
    """
    The bowl on the counter.
    """

    CUP = "jeroen_cup.stl"
    """
    The cup on the counter.
    """

    DRAWER_HANDLE = "handle_cab10_m"
    """
    The handle of the middle drawer of the cabinet below the counter.
    """

    CUBE = "cube"
    """
    The cube on Tracy's table.
    """


class KitchenSpot(Enum):
    """
    The spots in the kitchen the demonstrations stand at or put things down at.
    """

    THE_COUNTER = Position(1.85, 2.2)
    """
    On the floor in front of the counter, facing it.
    """

    IN_FRONT_OF_THE_MILK = Position(1.75, 2.2)
    """
    On the floor, close enough to the counter to take the milk head-on.
    """

    IN_FRONT_OF_THE_FREE_SPOT = Position(1.75, 2.4)
    """
    On the floor, close enough to the counter to put something down at
    :attr:`FREE_SPOT_ON_THE_COUNTER`.
    """

    FREE_SPOT_ON_THE_COUNTER = Position(2.37, 2.2, 1.0345)
    """
    A free spot on the counter between the milk and the bowl, at the milk's height.
    """

    IN_FRONT_OF_THE_DRAWER = Position(1.6, 1.9)
    """
    On the floor in front of the drawer below the counter.
    """

    BESIDE_THE_BOWL = Position(1.85, 2.55)
    """
    On the floor in front of the counter, just past the bowl, so the right hand works
    over the bowl from the side away from the milk.
    """

    IN_FRONT_OF_THE_CLEAR_COUNTER = Position(1.85, 3.0)
    """
    On the floor in front of the stretch of counter between the cup and the sink.
    """

    CLEAR_COUNTER = Position(2.45, 3.0, 1.0)
    """
    The middle of the stretch of counter between the cup and the sink, just above it.
    """

    IN_FRONT_OF_THE_RIGHT_SHOULDER = Position(1.65, 2.0, 1.1)
    """
    In the air in front of the right shoulder of the PR2 standing where it starts.
    """

    LOW_IN_FRONT_OF_THE_RIGHT_SHOULDER = Position(1.65, 2.0, 0.9)
    """
    In the air in front of the right shoulder of the PR2 standing where it starts, at
    the height of the counter's edge.
    """


# %% the demonstrations


class PhysicsAction(Enum):
    """
    The actions the physics viewer has the robot perform.
    """

    RAISE_TORSO = TorsoPlan(height=TorsoState.HIGH)
    """
    The torso moves up to its highest position.
    """

    OPEN_GRIPPERS = GripperPlan(state=GripperState.OPEN)
    """
    The grippers of both arms open.
    """

    CLOSE_GRIPPERS = GripperPlan(state=GripperState.CLOSE)
    """
    The grippers of both arms close.
    """

    PARK_ARMS = PlanSequence(
        plans=(
            ArmToPositionPlan(
                position=KitchenSpot.LOW_IN_FRONT_OF_THE_RIGHT_SHOULDER.value,
                arm=ArmSide.RIGHT,
            ),
            ParkArmsPlan(),
        )
    )
    """
    The right arm reaches out, then both arms fold into their parking pose.
    """

    MOVE_ARM = PlanSequence(
        plans=(
            TorsoPlan(height=TorsoState.HIGH),
            ArmToPositionPlan(
                position=KitchenSpot.IN_FRONT_OF_THE_RIGHT_SHOULDER.value,
                arm=ArmSide.RIGHT,
            ),
        )
    )
    """
    The torso rises and the right hand moves to a point in front of the robot.
    """

    LOOK_AT_MILK = LookAtPlan(body=SceneBody.MILK)
    """
    The head turns to look at the milk carton.
    """

    FACE_THE_CUP = FaceAtPlan(body=SceneBody.CUP)
    """
    The base turns on the spot to face the cup.
    """

    FACE_AND_LOOK_AT_THE_CUP = FaceAndLookAtPlan(body=SceneBody.CUP)
    """
    The base turns to face the cup and the head looks at it.
    """

    GO_TO_THE_COUNTER = NavigationPlan(target=KitchenSpot.THE_COUNTER.value)
    """
    The base drives to the spot in front of the kitchen counter.
    """

    REACH_FOR_MILK = PlanSequence(
        plans=(
            NavigationPlan(target=KitchenSpot.IN_FRONT_OF_THE_MILK.value),
            GripperPlan(state=GripperState.OPEN),
            ReachPlan(body=SceneBody.MILK, arm=ArmSide.RIGHT, grasp_index=0),
        )
    )
    """
    The robot drives to the counter, opens its grippers and moves its right hand around
    the milk carton.
    """

    GRASP_MILK = PlanSequence(
        plans=(
            NavigationPlan(target=KitchenSpot.IN_FRONT_OF_THE_MILK.value),
            GraspingPlan(body=SceneBody.MILK, arm=ArmSide.RIGHT, grasp_index=0),
        )
    )
    """
    The robot drives to the counter and grasps the milk carton where it stands.
    """

    PICK_UP_MILK = PlanSequence(
        plans=(
            NavigationPlan(target=KitchenSpot.IN_FRONT_OF_THE_MILK.value),
            PickUpPlan(body=SceneBody.MILK, arm=ArmSide.RIGHT, grasp_index=0),
        )
    )
    """
    The robot drives to the counter and picks up the milk carton with its right arm.
    """

    PLACE_MILK = PlanSequence(
        plans=(
            NavigationPlan(target=KitchenSpot.IN_FRONT_OF_THE_MILK.value),
            PickUpPlan(body=SceneBody.MILK, arm=ArmSide.RIGHT, grasp_index=0),
            PlacePlan(
                body=SceneBody.MILK, target=KitchenSpot.FREE_SPOT_ON_THE_COUNTER.value
            ),
        )
    )
    """
    The robot picks up the milk carton and puts it down beside where it stood.
    """

    PICK_AND_PLACE_MILK = PlanSequence(
        plans=(
            NavigationPlan(target=KitchenSpot.IN_FRONT_OF_THE_MILK.value),
            PickAndPlacePlan(
                body=SceneBody.MILK,
                arm=ArmSide.RIGHT,
                grasp_index=0,
                target=KitchenSpot.FREE_SPOT_ON_THE_COUNTER.value,
            ),
        )
    )
    """
    The robot moves the milk carton along the counter in one pick-and-place action.
    """

    MOVE_AND_PICK_UP_MILK = MoveAndPickUpPlan(
        body=SceneBody.MILK,
        arm=ArmSide.RIGHT,
        grasp_index=0,
        standing=KitchenSpot.IN_FRONT_OF_THE_MILK.value,
    )
    """
    The robot drives to the counter, turns towards the milk carton and picks it up.
    """

    MOVE_AND_PLACE_MILK = PlanSequence(
        plans=(
            MoveAndPickUpPlan(
                body=SceneBody.MILK,
                arm=ArmSide.RIGHT,
                grasp_index=0,
                standing=KitchenSpot.IN_FRONT_OF_THE_MILK.value,
            ),
            MoveAndPlacePlan(
                body=SceneBody.MILK,
                arm=ArmSide.RIGHT,
                target=KitchenSpot.FREE_SPOT_ON_THE_COUNTER.value,
                standing=KitchenSpot.IN_FRONT_OF_THE_FREE_SPOT.value,
            ),
        )
    )
    """
    The robot picks up the milk carton, drives a little along the counter and puts it
    down there.
    """

    TRANSPORT_MILK = TransportPlan(
        body=SceneBody.MILK,
        arm=ArmSide.RIGHT,
        grasp_index=0,
        standing=KitchenSpot.IN_FRONT_OF_THE_MILK.value,
        target=KitchenSpot.FREE_SPOT_ON_THE_COUNTER.value,
        placing_standing=KitchenSpot.IN_FRONT_OF_THE_FREE_SPOT.value,
    )
    """
    The robot carries the milk carton along the counter with its arms parked.
    """

    OPEN_DRAWER = PlanSequence(
        plans=(
            TorsoPlan(height=TorsoState.HIGH),
            ParkArmsPlan(),
            NavigationPlan(
                target=KitchenSpot.IN_FRONT_OF_THE_DRAWER.value, heading=0.58
            ),
            OpenPlan(handle=SceneBody.DRAWER_HANDLE, arm=ArmSide.LEFT),
        )
    )
    """
    The robot drives to the drawer below the counter and pulls it open.
    """

    CLOSE_DRAWER = PlanSequence(
        plans=(
            OPEN_DRAWER,
            ClosePlan(handle=SceneBody.DRAWER_HANDLE, arm=ArmSide.LEFT),
        )
    )
    """
    The robot pulls the drawer below the counter open and pushes it shut again.
    """

    MOVE_AND_OPEN_DRAWER = PlanSequence(
        plans=(
            TorsoPlan(height=TorsoState.HIGH),
            ParkArmsPlan(),
            MoveAndOpenPlan(
                handle=SceneBody.DRAWER_HANDLE,
                arm=ArmSide.LEFT,
                standing=KitchenSpot.IN_FRONT_OF_THE_DRAWER.value,
            ),
        )
    )
    """
    The robot drives to the drawer, turns towards its handle and pulls it open.
    """

    POUR_INTO_BOWL = PlanSequence(
        plans=(
            GripperPlan(state=GripperState.CLOSE),
            ParkArmsPlan(),
            TorsoPlan(height=TorsoState.HIGH),
            NavigationPlan(target=KitchenSpot.BESIDE_THE_BOWL.value),
            PouringPlan(arm=ArmSide.RIGHT, container=SceneBody.BOWL),
        )
    )
    """
    The robot pours from the cup in its hand into the bowl; the scene has no liquid, so
    only the cup moves.
    """

    MIX_IN_BOWL = PlanSequence(
        plans=(
            GripperPlan(state=GripperState.CLOSE),
            ParkArmsPlan(),
            TorsoPlan(height=TorsoState.HIGH),
            NavigationPlan(target=KitchenSpot.BESIDE_THE_BOWL.value),
            MixingPlan(arm=ArmSide.RIGHT, container=SceneBody.BOWL),
        )
    )
    """
    The robot stirs in the bowl with the whisk in its hand.
    """

    WIPE_COUNTER = PlanSequence(
        plans=(
            GripperPlan(state=GripperState.CLOSE),
            ParkArmsPlan(),
            TorsoPlan(height=TorsoState.HIGH),
            NavigationPlan(target=KitchenSpot.IN_FRONT_OF_THE_CLEAR_COUNTER.value),
            WipingPlan(arm=ArmSide.RIGHT, target=KitchenSpot.CLEAR_COUNTER.value),
        )
    )
    """
    The robot wipes a free stretch of the counter with the sponge in its hand.
    """

    PICK_UP_CUBE = PickUpPlan(body=SceneBody.CUBE, arm=ArmSide.LEFT, grasp_index=6)
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
