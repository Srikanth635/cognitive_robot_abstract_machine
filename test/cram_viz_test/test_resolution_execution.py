"""
Tests of performing a run's resolved pouring action in the simulation: the resolved
values found again in the world, the pose the robot picks the source up from, and the
plan performed from the robot's start in the kitchen the simple_pouring scene records.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
from coraplex.datastructures.enums import Arms
from coraplex.view_manager import ViewManager
from semantic_digital_twin.robots.pr2 import PR2
from semantic_digital_twin.semantic_annotations.semantic_annotations import PouringCup

from cram_viz.resolution.execution import (
    PICK_UP_DISTANCE,
    PouringExecution,
    standing_pose,
)
from cram_viz.resolution.resolver import ResolvedSlot
from cram_viz.resolution.scene_world import SceneWorldBuilder

from .resolutions import objects_given

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
        self, pouring_bundle
    ):
        world = SceneWorldBuilder().build(pouring_bundle)
        action = PouringExecution.for_resolution(world, objects_given()).action()
        values = resolved_values()
        assert action.source_container.root is world.get_body_by_name(
            values["source_container"]
        )
        assert isinstance(action.source_container, PouringCup)
        assert action.target_container is world.get_body_by_name(
            values["target_container"]
        )
        assert action.arm is Arms[values["arm"]]
        assert action.tilt_angle == values["tilt_angle"]

    @pytest.mark.parametrize("pour_side", [Arms.LEFT.name, None])
    def test_an_optional_slot_holds_its_member_or_nothing(
        self, pouring_bundle, pour_side
    ):
        world = SceneWorldBuilder().build(pouring_bundle)
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
        assert action.pour_side is (None if pour_side is None else Arms[pour_side])


# %% where the robot stands


class TestStandingPose:
    def test_the_robot_stands_facing_the_source_at_pick_up_distance(
        self, pouring_bundle
    ):
        world = SceneWorldBuilder().build(pouring_bundle)
        base = world.get_body_by_name("base_link")
        source = world.get_body_by_name("Static_MilkPitcher.stl")
        pose = standing_pose(world, base, source)
        stand = pose.to_position().to_np()[:2]
        target = source.global_pose.to_position().to_np()[:2]
        heading = target - stand
        assert np.isclose(np.linalg.norm(heading), PICK_UP_DISTANCE)
        assert np.isclose(
            float(pose.to_quaternion().to_rpy()[2]),
            np.arctan2(heading[1], heading[0]),
        )


# %% performing


class TestPerforming:
    def test_performing_leaves_the_source_in_the_resolved_hand(self, pouring_kitchen):
        execution = PouringExecution.for_resolution(pouring_kitchen, objects_given())
        execution.perform()
        robot = pouring_kitchen.get_semantic_annotations_by_type(PR2)[0]
        tool_frame = ViewManager.get_end_effector_view(
            Arms[resolved_values()["arm"]], robot
        ).tool_frame
        source = pouring_kitchen.get_body_by_name(resolved_values()["source_container"])
        assert source.parent_connection.parent is tool_frame
