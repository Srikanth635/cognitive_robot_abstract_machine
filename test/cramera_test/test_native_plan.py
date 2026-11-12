"""
Native plan lifecycle values at the live and recording boundaries.
"""

from __future__ import annotations

import json

import pytest

from coraplex.robot_plans.actions.core.robot_body import (
    ParkArmsAction,
    SetGripperAction,
)
from cramph.composites import Sequence
from cramph.data_types import LifeCycleValues
from cramph.statechart import Statechart
from krrood.entity_query_language.factories import inference
from krrood.entity_query_language.verbalization.pipeline import verbalize_expression
from semantic_digital_twin.datastructures.definitions import GripperState
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.robots.pr2 import PR2
from semantic_digital_twin.semantic_annotations.semantic_annotations import Handle
from semantic_digital_twin.world_description.world_entity import Body

from cramera.live.bridge import Bridge
from cramera.recording_fields import SceneField

from .dataset.plan_statecharts import ActingOnAnAnnotation, MotionStep, plan_statechart

# %% plan observers


def test_every_plan_snapshot_is_handed_to_the_plan_observers(
    plan_statechart: Statechart,
) -> None:
    """
    An observer of the bridge's plan receives each snapshot the bridge takes, as taken.

    :param plan_statechart: An empty statechart to build the plan in.
    """
    node = MotionStep(name="moving")
    plan_statechart.add_node(node)
    bridge = Bridge()
    observed = []
    bridge.plan_observers.append(observed.append)

    bridge.begin_plan(statechart=plan_statechart)
    plan_statechart.life_cycle_state[node] = LifeCycleValues.RUNNING
    bridge.snapshot_plan()

    assert [snapshot.nodes[0].status for snapshot in observed] == [
        LifeCycleValues.NOT_STARTED,
        LifeCycleValues.RUNNING,
    ]
    assert observed[-1] == bridge.plan_state


# %% lifecycle publication


@pytest.mark.parametrize("status", list(LifeCycleValues))
def test_plan_snapshot_keeps_the_native_lifecycle(
    plan_statechart: Statechart, status: LifeCycleValues
) -> None:
    """
    Keep native states in memory and their names in serialized snapshots.

    :param plan_statechart: An empty statechart to build the plan in.
    :param status: The native lifecycle state to publish.
    """
    node = MotionStep(name="moving")
    plan_statechart.add_node(node)
    plan_statechart.life_cycle_state[node] = status
    bridge = Bridge()

    bridge.begin_plan(statechart=plan_statechart)

    assert bridge.plan_state.nodes[0].status is status
    assert (
        json.loads(json.dumps(bridge.get_plan()))["nodes"][0]["status"] == status.name
    )
    assert (
        json.loads(json.dumps(bridge.plan_state.recorded_trees()))[0]["status"]
        == status.name
    )


@pytest.mark.parametrize("status", list(LifeCycleValues))
def test_parent_lifecycle_is_independent_of_finished_children(
    plan_statechart: Statechart,
    status: LifeCycleValues,
) -> None:
    """
    A parent's lifecycle remains its own when children have completed.

    :param plan_statechart: An empty statechart to build the plan in.
    :param status: The parent's current lifecycle, including a reset.
    """
    child = MotionStep(name="moving")
    parent = Sequence(nodes=[child])
    plan_statechart.add_node(parent)
    plan_statechart.life_cycle_state[parent] = LifeCycleValues.RUNNING
    plan_statechart.life_cycle_state[child] = LifeCycleValues.SUCCEEDED
    bridge = Bridge()
    bridge.begin_plan(statechart=plan_statechart)

    plan_statechart.life_cycle_state[parent] = status
    bridge.snapshot_plan()

    assert bridge.plan_state.nodes[0].status is status
    assert bridge.plan_state.nodes[0].derived is False


# %% native designator metadata


@pytest.mark.parametrize("gripper_index", [0, 1])
def test_designator_description_uses_native_parameter_verbalization(
    plan_statechart: Statechart,
    gripper_index: int,
) -> None:
    """
    Publish native wording for the configuration the selected gripper is moved to.

    :param plan_statechart: An empty statechart performed by the PR2.
    :param gripper_index: The arm whose native end effector is described.
    """
    [robot] = plan_statechart.context.world.get_semantic_annotations_by_type(PR2)
    gripper = robot.all_arms[gripper_index].end_effector
    action = SetGripperAction(
        configuration=gripper.default_configuration(GripperState.CLOSE)
    )
    plan_statechart.add_node(action)
    bridge = Bridge()

    bridge.begin_plan(statechart=plan_statechart)

    entry = bridge.get_plan()["nodes"][0]
    assert entry[SceneField.DESCRIPTION] == verbalize_expression(
        inference(type(action))(**action.designator_parameter)
    )
    assert "arm" not in entry


@pytest.mark.parametrize("arm_indices", [(0,), (1,), (0, 1)])
def test_native_arm_selection_is_verbalized(
    plan_statechart: Statechart, arm_indices: tuple[int, ...]
) -> None:
    """
    Describe every native arm selection, including the left arm and both arms.

    :param plan_statechart: An empty statechart performed by the PR2.
    :param arm_indices: The indices of the native arms selected for parking.
    """
    [robot] = plan_statechart.context.world.get_semantic_annotations_by_type(PR2)
    action = ParkArmsAction(arms=[robot.all_arms[index] for index in arm_indices])
    plan_statechart.add_node(action)
    bridge = Bridge()

    bridge.begin_plan(statechart=plan_statechart)

    assert bridge.plan_state.nodes[0].description == verbalize_expression(
        inference(type(action))(**action.designator_parameter)
    )


def test_gripper_state_enum_is_not_mistaken_for_a_target_body(
    plan_statechart: Statechart,
) -> None:
    """
    An enum's name does not create an object reference with the same name.

    :param plan_statechart: An empty statechart performed by the PR2.
    """
    state = GripperState.CLOSE
    body = Body(name=PrefixedName(state.name))
    [robot] = plan_statechart.context.world.get_semantic_annotations_by_type(PR2)
    gripper = robot.all_arms[0].end_effector
    plan_statechart.add_node(
        SetGripperAction(configuration=gripper.default_configuration(state))
    )
    bridge = Bridge()
    bridge.publish_bodies({str(body.name): body})

    bridge.begin_plan(statechart=plan_statechart)

    assert bridge.plan_state.nodes[0].target is None


def test_native_annotation_resolves_its_published_body_name(
    plan_statechart: Statechart,
) -> None:
    """
    A semantic annotation retains the target match through its native name.

    :param plan_statechart: An empty statechart performed by the PR2.
    """
    body = Body(name=PrefixedName("handle", prefix="world"))
    annotation = Handle(root=body, name=body.name)
    plan_statechart.add_node(ActingOnAnAnnotation(target_annotation=annotation))
    bridge = Bridge()
    bridge.publish_bodies({body.name.name: body})

    bridge.begin_plan(statechart=plan_statechart)

    assert bridge.plan_state.nodes[0].target == body.name.name
