"""
Tests of performing a run's resolved pouring action in the simulation: the resolved
values found again in the world, the pose the robot picks the source up from, and the
plan performed from the robot's start in the kitchen the simple_pouring scene records.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
from coraplex.datastructures.enums import PouringSide
from coraplex.robot_plans.actions.core.navigation import NavigateAction
from coraplex.robot_plans.actions.core.pick_up import PickUpAction
from coraplex.robot_plans.actions.core.robot_body import MoveTorsoAction
from semantic_digital_twin.robots.robot_parts import Arm
from semantic_digital_twin.semantic_annotations.semantic_annotations import PouringCup

from cram_viz.resolution.execution import (
    PICK_UP_DISTANCE,
    POURING_TORSO,
    PouringExecution,
    facing_pose,
    grasp_facing,
    standing_pose,
)
from cram_viz.resolution.resolver import ResolvedSlot
from cram_viz.resolution.scene_world import SceneWorldBuilder
from cram_viz.resolution.slot_data import SlotData
from cram_viz.resolution.trial import OutcomeKey, PouringTrial

from .resolutions import arm_in, objects_given

FACING_TOLERANCE = 0.05
"""
How far, in radians, the robot may end from facing the target for it to count as facing
it.
"""

NEARLY_UPSIDE_DOWN = 2.9
"""
A tilt, in radians, at which the right arm finishes the pour only once the robot faces
the bowl.
"""

# %% helpers


def resolved_values() -> dict:
    """
    :return: The resolved slots of the resolution performed, as the run keeps them.
    """
    (action,) = objects_given().actions
    return {slot.name: slot.serialized for slot in action.slots}


# %% the action


class TestAction:
    def test_the_action_holds_the_resolved_values_found_in_the_world(
        self, pouring_bundle_with_arms
    ):
        world = SceneWorldBuilder.robot_world(pouring_bundle_with_arms)
        action = PouringExecution.for_resolution(world, objects_given()).action()
        values = resolved_values()
        assert action.source_container.root is world.get_body_by_name(
            values["source_container"]
        )
        assert isinstance(action.source_container, PouringCup)
        assert action.target_container is world.get_body_by_name(
            values["target_container"]
        )
        assert action.arm is arm_in(world, values["arm"])
        assert action.tilt_angle == values["tilt_angle"]

    @pytest.mark.parametrize("pour_side", [PouringSide.LEFT.name, None])
    def test_an_optional_slot_holds_its_member_or_nothing(
        self, pouring_bundle_with_arms, pour_side
    ):
        world = SceneWorldBuilder.robot_world(pouring_bundle_with_arms)
        resolution = objects_given()
        (resolved,) = resolution.actions
        with_side = replace(
            resolution,
            actions=[
                replace(
                    resolved,
                    slots=[
                        *resolved.slots,
                        ResolvedSlot(
                            "pour_side",
                            str(pour_side),
                            given=False,
                            serialized=pour_side,
                        ),
                    ],
                )
            ],
        )
        action = PouringExecution.for_resolution(world, with_side).action()
        assert action.pour_side is (
            None if pour_side is None else PouringSide[pour_side]
        )


# %% the arm and the grasp


class TestArmAsData:
    def test_the_two_arms_are_told_apart_though_they_share_a_root_body(
        self, pouring_kitchen
    ):
        left, right = (
            arm_in(pouring_kitchen, name) for name in ("PR2LeftArm", "PR2RightArm")
        )
        assert left.root is right.root
        assert SlotData.from_value(left) != SlotData.from_value(right)

    @pytest.mark.parametrize("name", ["PR2LeftArm", "PR2RightArm"])
    def test_an_arm_is_found_again_by_the_data_it_was_written_as(
        self, pouring_kitchen, name
    ):
        arm = arm_in(pouring_kitchen, name)
        assert SlotData.to_value(pouring_kitchen, Arm, SlotData.from_value(arm)) is arm


class TestGraspFacing:
    def test_the_grasp_comes_from_the_way_the_robot_faces(self, pouring_kitchen):
        execution = PouringExecution.for_resolution(pouring_kitchen, objects_given())
        action = execution.action()
        standing = standing_pose(
            pouring_kitchen, execution.robot.root, action.source_container.root
        )
        heading = standing.to_np()[:2, 0]
        chosen = grasp_facing(action.source_container, standing)
        approach = lambda grasp: float(grasp.world_T_grasp.to_np()[:2, 0] @ heading)
        assert approach(chosen) == max(
            approach(grasp) for grasp in action.source_container.grasp_candidates()
        )


# %% where the robot stands


class TestStandingPose:
    def test_the_robot_stands_facing_the_source_at_pick_up_distance(
        self, pouring_bundle
    ):
        world = SceneWorldBuilder().build(pouring_bundle)
        base = world.get_body_by_name("base_link")
        source = world.get_body_by_name("Static_MilkPitcher.stl")
        pose = standing_pose(world, base, source)
        stand = pose.position.to_np()[:2]
        target = source.global_pose.position.to_np()[:2]
        heading = target - stand
        assert np.isclose(np.linalg.norm(heading), PICK_UP_DISTANCE)
        assert np.isclose(
            float(pose.yaw),
            np.arctan2(heading[1], heading[0]),
        )


# %% facing the target


def heading_between(origin: np.ndarray, destination: np.ndarray) -> float:
    """
    :return: The direction, in radians, from one point on the floor to another.
    """
    offset = destination - origin
    return float(np.arctan2(offset[1], offset[0]))


class TestFacingPose:
    def test_the_robot_turns_in_place(self, pouring_bundle):
        world = SceneWorldBuilder().build(pouring_bundle)
        base = world.get_body_by_name("base_link")
        standing = standing_pose(
            world, base, world.get_body_by_name("Static_MilkPitcher.stl")
        )
        facing = facing_pose(standing, world.get_body_by_name("bowl.stl"), world)
        assert np.allclose(facing.position.to_np()[:2], standing.position.to_np()[:2])

    def test_the_robot_faces_the_target(self, pouring_bundle):
        world = SceneWorldBuilder().build(pouring_bundle)
        base = world.get_body_by_name("base_link")
        target = world.get_body_by_name("bowl.stl")
        standing = standing_pose(
            world, base, world.get_body_by_name("Static_MilkPitcher.stl")
        )
        facing = facing_pose(standing, target, world)
        assert np.isclose(
            float(facing.yaw),
            heading_between(
                standing.position.to_np()[:2],
                target.global_pose.position.to_np()[:2],
            ),
        )


# %% the steps before the pour


class TestStepsBeforeThePour:
    def steps(self, world):
        execution = PouringExecution.for_resolution(world, objects_given())
        action = execution.action()
        return execution, action, execution.steps_before_pour(action)

    def test_the_robot_raises_its_torso_after_picking_the_source_up(
        self, pouring_kitchen
    ):
        _, _, steps = self.steps(pouring_kitchen)
        *_, picking_up, raising, _ = steps
        assert isinstance(picking_up, PickUpAction)
        assert isinstance(raising, MoveTorsoAction)
        assert raising.torso_state is POURING_TORSO

    def test_the_robot_turns_to_face_the_target_after_picking_the_source_up(
        self, pouring_kitchen
    ):
        _, _, steps = self.steps(pouring_kitchen)
        *_, picking_up, _, turning = steps
        assert isinstance(picking_up, PickUpAction)
        assert isinstance(turning, NavigateAction)

    def test_it_turns_to_face_the_target_from_where_it_picked_the_source_up(
        self, pouring_kitchen
    ):
        execution, action, steps = self.steps(pouring_kitchen)
        standing = standing_pose(
            pouring_kitchen, execution.robot.root, action.source_container.root
        )
        wanted = facing_pose(standing, action.target_container, pouring_kitchen)
        planned = steps[-1].target_location
        assert np.allclose(planned.position.to_np()[:2], wanted.position.to_np()[:2])
        assert np.isclose(
            float(planned.yaw),
            float(wanted.yaw),
        )


# %% performing


class TestPerforming:
    def test_performing_leaves_the_source_in_the_resolved_hand(self, pouring_kitchen):
        execution = PouringExecution.for_resolution(pouring_kitchen, objects_given())
        execution.perform()
        tool_frame = arm_in(
            pouring_kitchen, resolved_values()["arm"]
        ).end_effector.tool_frame
        source = pouring_kitchen.get_body_by_name(resolved_values()["source_container"])
        assert source.parent_connection.parent is tool_frame

    def test_performing_leaves_the_robot_facing_the_target(self, pouring_kitchen):
        execution = PouringExecution.for_resolution(pouring_kitchen, objects_given())
        action = execution.action()
        execution.perform()
        base = execution.robot.root.global_pose
        wanted = heading_between(
            base.position.to_np()[:2],
            action.target_container.global_pose.position.to_np()[:2],
        )
        turned = float(base.yaw) - wanted
        assert abs(np.arctan2(np.sin(turned), np.cos(turned))) < FACING_TOLERANCE


class TestPouringAfterFacing:
    def test_a_pour_the_arm_cannot_make_from_where_it_picked_up_is_made(
        self, pouring_kitchen
    ):
        outcome = PouringTrial.outcome_in(
            pouring_kitchen, {**resolved_values(), "tilt_angle": NEARLY_UPSIDE_DOWN}
        )
        assert outcome[OutcomeKey.LANDS_IN_OPENING]
