"""
The actions the robots perform in the physics viewer: one demonstration of each kind of
action coraplex offers, each made of the actions it performs one after the other.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum, StrEnum

from coraplex.datastructures.dataclasses import Context
from coraplex.plans.factories import sequential
from coraplex.plans.plan import Plan
from coraplex.robot_plans.actions.base import ActionDescription
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
from semantic_digital_twin.datastructures.definitions import GripperState, TorsoState
from semantic_digital_twin.grasping.grasp_candidates import GraspCandidate
from semantic_digital_twin.robots.robot_parts import AbstractRobot, Arm
from semantic_digital_twin.semantic_annotations.mixins import HasGraspCandidates
from semantic_digital_twin.semantic_annotations.semantic_annotations import (
    Handle,
    Tool,
)
from semantic_digital_twin.spatial_types.spatial_types import Pose

from experiments.physics_viewer.exceptions import NoSuchBody, NoToolInHand
from experiments.physics_viewer.placement import Position

# %% how a plan is built


class ActionPlan(ABC):
    """
    How the plan of an action is built.
    """

    @abstractmethod
    def actions(self, context: Context) -> list[ActionDescription]:
        """
        :param context: The world and the robot the plan is performed with.
        :return: The actions the plan performs, in order.
        """

    def build(self, context: Context) -> Plan:
        """
        :param context: The world and the robot the plan is performed with.
        :return: The plan performing :meth:`actions` one after the other.
        """
        return sequential(self.actions(context), context=context).plan


@dataclass(frozen=True)
class PlanSequence(ActionPlan):
    """
    Several plans performed one after the other.
    """

    plans: tuple[ActionPlan, ...]
    """
    The plans, in the order they are performed.
    """

    def actions(self, context: Context) -> list[ActionDescription]:
        return [action for plan in self.plans for action in plan.actions(context)]


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


# %% the robot's body


@dataclass(frozen=True)
class TorsoPlan(ActionPlan):
    """
    Moves the torso to a height.
    """

    height: TorsoState
    """
    The height to move the torso to.
    """

    def actions(self, context: Context) -> list[ActionDescription]:
        return [MoveTorsoAction(self.height)]


@dataclass(frozen=True)
class GripperPlan(ActionPlan):
    """
    Sets the grippers of every arm.
    """

    state: GripperState
    """
    The state to set the grippers to.
    """

    def actions(self, context: Context) -> list[ActionDescription]:
        return [
            SetGripperAction(arm.end_effector, self.state)
            for arm in context.robot.all_arms
        ]


@dataclass(frozen=True)
class ParkArmsPlan(ActionPlan):
    """
    Folds every arm into its parking pose beside the body.
    """

    def actions(self, context: Context) -> list[ActionDescription]:
        return [ParkArmsAction(context.robot.all_arms)]


@dataclass(frozen=True)
class ArmToPositionPlan(ActionPlan):
    """
    Moves the tool frame of an arm's hand to a position, the hand pointing along the
    world's x axis.
    """

    position: Position
    """
    Where the tool frame goes.
    """

    arm: ArmSide
    """
    The arm that moves.
    """

    def actions(self, context: Context) -> list[ActionDescription]:
        return [
            MoveManipulatorAction(
                self.position.pose_in(context.world),
                self.arm.arm_of(context.robot).end_effector,
                allow_gripper_collision=False,
            )
        ]


@dataclass(frozen=True)
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

    def actions(self, context: Context) -> list[ActionDescription]:
        return [NavigateAction(self.target.pose_in(context.world, self.heading))]


# %% turning towards a body


@dataclass(frozen=True)
class BodyPlan(ActionPlan, ABC):
    """
    An action aimed at a body of the scene.
    """

    body: str
    """
    The name of the body.
    """

    def body_pose(self, context: Context) -> Pose:
        """
        :param context: The world and the robot the plan is performed with.
        :return: Where the body is now.
        """
        return context.world.get_body_by_name(self.body).global_pose


@dataclass(frozen=True)
class LookAtPlan(BodyPlan):
    """
    Turns the head to look at a body.
    """

    def actions(self, context: Context) -> list[ActionDescription]:
        return [LookAtAction(self.body_pose(context))]


@dataclass(frozen=True)
class FaceAtPlan(BodyPlan):
    """
    Turns the base on the spot to face a body.
    """

    def actions(self, context: Context) -> list[ActionDescription]:
        return [FaceAtAction(self.body_pose(context))]


@dataclass(frozen=True)
class FaceAndLookAtPlan(BodyPlan):
    """
    Turns the base to face a body and the head to look at it.
    """

    def actions(self, context: Context) -> list[ActionDescription]:
        pose = self.body_pose(context)
        return [FaceAndLookAtAction(FaceAtAction(pose), LookAtAction(pose))]


# %% grasping, carrying and putting down


@dataclass(frozen=True)
class GraspablePlan(ActionPlan, ABC):
    """
    An action on a body the scene annotates as something to grasp.
    """

    body: str
    """
    The name of the body.
    """

    def graspable(self, context: Context) -> HasGraspCandidates:
        """
        :param context: The world and the robot the plan is performed with.
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


@dataclass(frozen=True)
class GraspPlan(GraspablePlan, ABC):
    """
    An action that takes a body with one arm, by one of the body's grasp candidates.
    """

    arm: ArmSide
    """
    The arm that takes the body.
    """

    grasp_index: int
    """
    Which of the body's grasp candidates is taken.

    Measured, not chosen by a rule: the candidates the arm reaches are not all that lift
    the body under physics.
    """

    def grasp(self, context: Context) -> GraspCandidate:
        """
        :param context: The world and the robot the plan is performed with.
        :return: The grasp candidate taken.
        """
        return self.graspable(context).grasp_candidates()[self.grasp_index]


@dataclass(frozen=True)
class ReachPlan(GraspPlan):
    """
    Moves the hand around the body, as for a grasp, without closing it.
    """

    def actions(self, context: Context) -> list[ActionDescription]:
        return [
            ReachAction(arm=self.arm.arm_of(context.robot), grasp=self.grasp(context))
        ]


@dataclass(frozen=True)
class GraspingPlan(GraspPlan):
    """
    Grasps the body where it stands, without lifting it.
    """

    def actions(self, context: Context) -> list[ActionDescription]:
        return [
            GraspingAction(
                grasp=self.grasp(context), arm=self.arm.arm_of(context.robot)
            )
        ]


@dataclass(frozen=True)
class PickUpPlan(GraspPlan):
    """
    Grasps the body and lifts it.
    """

    def actions(self, context: Context) -> list[ActionDescription]:
        return [PickUpAction(self.grasp(context), self.arm.arm_of(context.robot))]


@dataclass(frozen=True)
class MoveAndPickUpPlan(GraspPlan):
    """
    Drives to a spot, turns towards the body and picks it up.
    """

    standing: Position
    """
    The spot on the floor the body is picked up from.
    """

    def actions(self, context: Context) -> list[ActionDescription]:
        return [
            MoveAndPickUpAction.from_standing_position(
                self.standing.pose_in(context.world),
                self.grasp(context),
                self.arm.arm_of(context.robot),
            )
        ]


@dataclass(frozen=True)
class PickAndPlacePlan(GraspPlan):
    """
    Picks the body up and puts it down elsewhere within reach.
    """

    target: Position
    """
    Where the body is put down.
    """

    def actions(self, context: Context) -> list[ActionDescription]:
        return [
            PickAndPlaceAction(
                PickUpAction(self.grasp(context), self.arm.arm_of(context.robot)),
                PlaceAction(
                    self.graspable(context), self.target.pose_in(context.world)
                ),
            )
        ]


@dataclass(frozen=True)
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

    def actions(self, context: Context) -> list[ActionDescription]:
        return [
            TransportAction(
                MoveAndPickUpAction.from_standing_position(
                    self.standing.pose_in(context.world),
                    self.grasp(context),
                    self.arm.arm_of(context.robot),
                ),
                MoveAndPlaceAction.from_standing_position(
                    self.placing_standing.pose_in(context.world),
                    self.target.pose_in(context.world),
                    self.graspable(context),
                ),
            )
        ]


@dataclass(frozen=True)
class PlacePlan(GraspablePlan):
    """
    Puts the held body down.
    """

    target: Position
    """
    Where the body is put down.
    """

    def actions(self, context: Context) -> list[ActionDescription]:
        return [
            PlaceAction(self.graspable(context), self.target.pose_in(context.world))
        ]


@dataclass(frozen=True)
class MoveAndPlacePlan(GraspablePlan):
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

    def actions(self, context: Context) -> list[ActionDescription]:
        return [
            MoveAndPlaceAction.from_standing_position(
                self.standing.pose_in(context.world),
                self.target.pose_in(context.world),
                self.graspable(context),
            )
        ]


# %% drawers and doors


@dataclass(frozen=True)
class HandlePlan(ActionPlan, ABC):
    """
    An action on a container, by its handle.
    """

    handle: str
    """
    The name of the handle's body.
    """

    arm: ArmSide
    """
    The arm that takes the handle.
    """

    def handle_annotation(self, context: Context) -> Handle:
        """
        :param context: The world and the robot the plan is performed with.
        :return: The handle's annotation, annotating the body as a handle if the scene
            does not.
        """
        body = context.world.get_body_by_name(self.handle)
        for handle in context.world.get_semantic_annotations_by_type(Handle):
            if handle.root is body:
                return handle
        handle = Handle(root=body)
        with context.world.modify_world():
            context.world.add_semantic_annotation_recursively(handle)
        return handle


@dataclass(frozen=True)
class OpenPlan(HandlePlan):
    """
    Pulls the container open by its handle.
    """

    def actions(self, context: Context) -> list[ActionDescription]:
        return [
            OpenAction(self.handle_annotation(context), self.arm.arm_of(context.robot))
        ]


@dataclass(frozen=True)
class ClosePlan(HandlePlan):
    """
    Pushes the container shut by its handle.
    """

    def actions(self, context: Context) -> list[ActionDescription]:
        return [
            CloseAction(self.handle_annotation(context), self.arm.arm_of(context.robot))
        ]


@dataclass(frozen=True)
class MoveAndOpenPlan(HandlePlan):
    """
    Drives to a spot, turns towards the handle and pulls the container open.
    """

    standing: Position
    """
    The spot on the floor the container is opened from.
    """

    def actions(self, context: Context) -> list[ActionDescription]:
        return [
            MoveAndOpenAction.from_standing_position(
                self.standing.pose_in(context.world),
                self.handle_annotation(context),
                self.arm.arm_of(context.robot),
            )
        ]


# %% working with a tool


@dataclass(frozen=True)
class ToolPlan(ActionPlan, ABC):
    """
    An action performed with the tool an arm's hand holds.
    """

    arm: ArmSide
    """
    The arm whose hand holds the tool.
    """

    def held_tool(self, context: Context) -> Tool:
        """
        :param context: The world and the robot the plan is performed with.
        :return: The tool fixed in the hand.
        :raises NoToolInHand: When the hand holds no tool.
        """
        tool_frame = self.arm.arm_of(context.robot).end_effector.tool_frame
        for tool in context.world.get_semantic_annotations_by_type(Tool):
            if tool.root.parent_connection.parent is tool_frame:
                return tool
        raise NoToolInHand(self.arm.value)


@dataclass(frozen=True)
class PouringPlan(ToolPlan):
    """
    Pours from the held container into another.
    """

    container: str
    """
    The name of the body poured into.
    """

    def actions(self, context: Context) -> list[ActionDescription]:
        return [
            PouringAction(
                target_container=context.world.get_body_by_name(self.container),
                source_container=self.held_tool(context),
                arm=self.arm.arm_of(context.robot),
            )
        ]


@dataclass(frozen=True)
class MixingPlan(ToolPlan):
    """
    Stirs inside a container with the held tool.
    """

    container: str
    """
    The name of the body stirred in.
    """

    def actions(self, context: Context) -> list[ActionDescription]:
        return [
            MixingAction(
                container=context.world.get_body_by_name(self.container),
                arm=self.arm.arm_of(context.robot),
                tool=self.held_tool(context),
            )
        ]


@dataclass(frozen=True)
class WipingPlan(ToolPlan):
    """
    Wipes a surface with the held tool.
    """

    target: Position
    """
    The middle of the stretch of surface wiped.
    """

    def actions(self, context: Context) -> list[ActionDescription]:
        return [
            WipingAction(
                arm=self.arm.arm_of(context.robot),
                tool=self.held_tool(context),
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

    PARK_ARMS = PlanSequence(
        (
            ArmToPositionPlan(
                KitchenSpot.LOW_IN_FRONT_OF_THE_RIGHT_SHOULDER.value, ArmSide.RIGHT
            ),
            ParkArmsPlan(),
        )
    )
    """
    The right arm reaches out, then both arms fold into their parking pose.
    """

    MOVE_ARM = PlanSequence(
        (
            TorsoPlan(TorsoState.HIGH),
            ArmToPositionPlan(
                KitchenSpot.IN_FRONT_OF_THE_RIGHT_SHOULDER.value, ArmSide.RIGHT
            ),
        )
    )
    """
    The torso rises and the right hand moves to a point in front of the robot.
    """

    LOOK_AT_MILK = LookAtPlan(SceneBody.MILK)
    """
    The head turns to look at the milk carton.
    """

    FACE_THE_CUP = FaceAtPlan(SceneBody.CUP)
    """
    The base turns on the spot to face the cup.
    """

    FACE_AND_LOOK_AT_THE_CUP = FaceAndLookAtPlan(SceneBody.CUP)
    """
    The base turns to face the cup and the head looks at it.
    """

    GO_TO_THE_COUNTER = NavigationPlan(KitchenSpot.THE_COUNTER.value)
    """
    The base drives to the spot in front of the kitchen counter.
    """

    REACH_FOR_MILK = PlanSequence(
        (
            NavigationPlan(KitchenSpot.IN_FRONT_OF_THE_MILK.value),
            GripperPlan(GripperState.OPEN),
            ReachPlan(SceneBody.MILK, ArmSide.RIGHT, 0),
        )
    )
    """
    The robot drives to the counter, opens its grippers and moves its right hand around
    the milk carton.
    """

    GRASP_MILK = PlanSequence(
        (
            NavigationPlan(KitchenSpot.IN_FRONT_OF_THE_MILK.value),
            GraspingPlan(SceneBody.MILK, ArmSide.RIGHT, 0),
        )
    )
    """
    The robot drives to the counter and grasps the milk carton where it stands.
    """

    PICK_UP_MILK = PlanSequence(
        (
            NavigationPlan(KitchenSpot.IN_FRONT_OF_THE_MILK.value),
            PickUpPlan(SceneBody.MILK, ArmSide.RIGHT, 0),
        )
    )
    """
    The robot drives to the counter and picks up the milk carton with its right arm.
    """

    PLACE_MILK = PlanSequence(
        (
            NavigationPlan(KitchenSpot.IN_FRONT_OF_THE_MILK.value),
            PickUpPlan(SceneBody.MILK, ArmSide.RIGHT, 0),
            PlacePlan(SceneBody.MILK, KitchenSpot.FREE_SPOT_ON_THE_COUNTER.value),
        )
    )
    """
    The robot picks up the milk carton and puts it down beside where it stood.
    """

    PICK_AND_PLACE_MILK = PlanSequence(
        (
            NavigationPlan(KitchenSpot.IN_FRONT_OF_THE_MILK.value),
            PickAndPlacePlan(
                SceneBody.MILK,
                ArmSide.RIGHT,
                0,
                KitchenSpot.FREE_SPOT_ON_THE_COUNTER.value,
            ),
        )
    )
    """
    The robot moves the milk carton along the counter in one pick-and-place action.
    """

    MOVE_AND_PICK_UP_MILK = MoveAndPickUpPlan(
        SceneBody.MILK, ArmSide.RIGHT, 0, KitchenSpot.IN_FRONT_OF_THE_MILK.value
    )
    """
    The robot drives to the counter, turns towards the milk carton and picks it up.
    """

    MOVE_AND_PLACE_MILK = PlanSequence(
        (
            MoveAndPickUpPlan(
                SceneBody.MILK, ArmSide.RIGHT, 0, KitchenSpot.IN_FRONT_OF_THE_MILK.value
            ),
            MoveAndPlacePlan(
                SceneBody.MILK,
                KitchenSpot.FREE_SPOT_ON_THE_COUNTER.value,
                KitchenSpot.IN_FRONT_OF_THE_FREE_SPOT.value,
            ),
        )
    )
    """
    The robot picks up the milk carton, drives a little along the counter and puts it
    down there.
    """

    TRANSPORT_MILK = TransportPlan(
        SceneBody.MILK,
        ArmSide.RIGHT,
        0,
        KitchenSpot.IN_FRONT_OF_THE_MILK.value,
        KitchenSpot.FREE_SPOT_ON_THE_COUNTER.value,
        KitchenSpot.IN_FRONT_OF_THE_FREE_SPOT.value,
    )
    """
    The robot carries the milk carton along the counter with its arms parked.
    """

    OPEN_DRAWER = PlanSequence(
        (
            TorsoPlan(TorsoState.HIGH),
            ParkArmsPlan(),
            NavigationPlan(KitchenSpot.IN_FRONT_OF_THE_DRAWER.value, heading=0.58),
            OpenPlan(SceneBody.DRAWER_HANDLE, ArmSide.LEFT),
        )
    )
    """
    The robot drives to the drawer below the counter and pulls it open.
    """

    CLOSE_DRAWER = PlanSequence(
        (
            OPEN_DRAWER,
            ClosePlan(SceneBody.DRAWER_HANDLE, ArmSide.LEFT),
        )
    )
    """
    The robot pulls the drawer below the counter open and pushes it shut again.
    """

    MOVE_AND_OPEN_DRAWER = PlanSequence(
        (
            TorsoPlan(TorsoState.HIGH),
            ParkArmsPlan(),
            MoveAndOpenPlan(
                SceneBody.DRAWER_HANDLE,
                ArmSide.LEFT,
                KitchenSpot.IN_FRONT_OF_THE_DRAWER.value,
            ),
        )
    )
    """
    The robot drives to the drawer, turns towards its handle and pulls it open.
    """

    POUR_INTO_BOWL = PlanSequence(
        (
            GripperPlan(GripperState.CLOSE),
            ParkArmsPlan(),
            TorsoPlan(TorsoState.HIGH),
            NavigationPlan(KitchenSpot.BESIDE_THE_BOWL.value),
            PouringPlan(ArmSide.RIGHT, SceneBody.BOWL),
        )
    )
    """
    The robot pours from the cup in its hand into the bowl; the scene has no liquid, so
    only the cup moves.
    """

    MIX_IN_BOWL = PlanSequence(
        (
            GripperPlan(GripperState.CLOSE),
            ParkArmsPlan(),
            TorsoPlan(TorsoState.HIGH),
            NavigationPlan(KitchenSpot.BESIDE_THE_BOWL.value),
            MixingPlan(ArmSide.RIGHT, SceneBody.BOWL),
        )
    )
    """
    The robot stirs in the bowl with the whisk in its hand.
    """

    WIPE_COUNTER = PlanSequence(
        (
            GripperPlan(GripperState.CLOSE),
            ParkArmsPlan(),
            TorsoPlan(TorsoState.HIGH),
            NavigationPlan(KitchenSpot.IN_FRONT_OF_THE_CLEAR_COUNTER.value),
            WipingPlan(ArmSide.RIGHT, KitchenSpot.CLEAR_COUNTER.value),
        )
    )
    """
    The robot wipes a free stretch of the counter with the sponge in its hand.
    """

    PICK_UP_CUBE = PickUpPlan(SceneBody.CUBE, ArmSide.LEFT, 6)
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
