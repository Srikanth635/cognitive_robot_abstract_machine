"""
Tests of reading the steps of a plan out of the NEEM database: the tree the nodes of its
statechart form, how each ended, what each was asked to do, and the things and places it
refers to.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typing_extensions import Any, Dict, List

from llm_reasoner_viz.neem_plan import (
    MissingRow,
    Orientation,
    Parameter,
    PlanNode,
    Pose,
    Position,
    Reference,
)
from llm_reasoner_viz.neem_schema import JsonKey, LifeCycle, Table
from llm_reasoner_viz.neems import NeemDatabase, UnknownNeem

from .neem_files import (
    PAGE_FIXTURES,
    PAGE_FIXTURES_DIRECTORY,
    Change,
    KeptPlan,
    change_database,
    sent_by_the_server,
    write_neem_database,
)

TORSO_HIGH = "semantic_digital_twin.datastructures.definitions.TorsoState.HIGH"
APPROACH_FROM_THE_FRONT = (
    "semantic_digital_twin.datastructures.definitions.ApproachDirection.FRONT"
)

# %% helpers


@pytest.fixture
def database(tmp_path: Path) -> NeemDatabase:
    """
    :return: A database of the two plans of the tests.
    """
    return NeemDatabase(path=write_neem_database(tmp_path / "neems.sqlite"))


def node_of(nodes: List[PlanNode], identifier: int) -> PlanNode:
    return next(node for node in nodes if node.identifier == identifier)


def values_of(node: PlanNode) -> Dict[str, Any]:
    """
    :return: The values a node was asked for, by the name of the parameter.
    """
    return {parameter.name: parameter.value for parameter in node.parameters}


def reference_named(node: PlanNode, name: str) -> Reference:
    return next(reference for reference in node.references if reference.name == name)


def seconds_at(plan: KeptPlan, tick: int) -> float:
    """
    :return: When a tick of the plan was, in seconds after the plan began.
    """
    return tick * plan.recording["tick_duration"]


def stored_node(plan: KeptPlan, identifier: int) -> Dict[str, Any]:
    """
    :return: A node of the plan's statechart, as the fixture keeps it.
    """
    return plan.recording["structure"]["nodes"][identifier]


# %% the tree


class TestTheTree:
    def test_the_steps_come_in_the_order_of_the_tree(self, database):
        assert [node.identifier for node in database.nodes(1)] == [0, 1, 2, 3, 4, 5, 6]

    def test_a_step_knows_the_one_above_it_and_how_deep_it_is(self, database):
        nodes = database.nodes(1)
        assert [node.parent for node in nodes] == [None, 0, 1, 0, 3, None, None]
        assert [node.depth for node in nodes] == [0, 1, 2, 1, 2, 0, 0]

    def test_the_plan_comes_before_what_its_executor_added_to_watch_it(self, database):
        top = [node.type for node in database.nodes(1) if node.parent is None]
        assert top == ["Sequence", "EndMotion", "StillProgressing"]

    def test_the_steps_below_a_step_keep_the_order_of_the_statechart(self, database):
        assert [node.identifier for node in database.nodes(2)] == [0, 1, 2, 3, 4, 5]

    def test_a_plan_the_database_does_not_hold_is_unknown(self, database):
        with pytest.raises(UnknownNeem) as unknown:
            database.nodes(99)
        assert unknown.value.neem == 99

    def test_without_a_database_every_plan_is_unknown(self, tmp_path):
        with pytest.raises(UnknownNeem):
            NeemDatabase(path=tmp_path / "none.sqlite").nodes(1)


# %% what a step is


class TestWhatAStepIs:
    def test_a_step_is_of_its_class(self, database):
        assert [node.type for node in database.nodes(1)] == [
            "Sequence",
            "PickUpAction",
            "CartesianPose",
            "PouringAction",
            "JointPositionList",
            "EndMotion",
            "StillProgressing",
        ]

    def test_only_an_action_has_the_designator_it_was_asked_to_perform(self, database):
        assert [node.designator for node in database.nodes(1)] == [
            None,
            "PickUpAction",
            None,
            "PouringAction",
            None,
            None,
            None,
        ]

    def test_a_step_has_the_status_it_ended_in(self, database):
        nodes = database.nodes(1)
        assert node_of(nodes, 1).status == LifeCycle.SUCCEEDED.name
        assert node_of(nodes, 3).status == LifeCycle.FAILED.name

    def test_a_step_has_when_it_started_and_ended(self, database):
        pour = node_of(database.nodes(1), 3)
        assert pour.started == seconds_at(KeptPlan.POUR, 40)
        assert pour.ended == seconds_at(KeptPlan.POUR, 90)

    def test_a_step_that_has_not_ended_has_no_end(self, database):
        change_database(database.path, Change.PLAN_NOT_ENDED)
        navigation = node_of(database.nodes(2), 3)
        assert navigation.status == LifeCycle.RUNNING.name
        assert navigation.started == seconds_at(KeptPlan.RUN, 21)
        assert navigation.ended is None


# %% what a step was asked to do


class TestWhatAStepWasAskedToDo:
    def test_the_values_of_an_action_are_its_fields_that_are_values(self, database):
        pour = node_of(database.nodes(1), 3)
        assert values_of(pour) == {
            "tilt_angle": 1.85,
            "pour_side": None,
            "pour_height": 0.13,
        }

    def test_the_values_keep_the_order_of_the_fields(self, database):
        pick_up = node_of(database.nodes(1), 1)
        assert [parameter.name for parameter in pick_up.parameters] == [
            "approach_clearance",
            "grasp_closing_velocity",
            "tolerate_grasp_stall",
        ]

    def test_a_flag_is_true_or_false(self, database):
        pick_up = node_of(database.nodes(1), 1)
        assert values_of(pick_up)["tolerate_grasp_stall"] is False

    def test_a_choice_from_an_enumeration_is_the_path_of_its_member(self, database):
        torso = node_of(database.nodes(2), 2)
        assert values_of(torso) == {"torso_state": TORSO_HIGH}

    def test_what_places_a_step_in_the_statechart_is_no_value_of_it(self, database):
        names = {
            parameter.name
            for node in database.nodes(1)
            for parameter in node.parameters
        }
        assert not names & {key.value for key in JsonKey if key.places_a_node}

    def test_a_parameter_is_json(self):
        assert Parameter(name="tilt_angle", value=1.85).to_json() == {
            "name": "tilt_angle",
            "value": 1.85,
        }


# %% what a step refers to


class TestWhatAStepRefersTo:
    def test_a_thing_of_the_world_is_named_and_of_the_class_the_database_keeps(
        self, database
    ):
        pour = node_of(database.nodes(1), 3)
        bowl = stored_node(KeptPlan.POUR, 3)["target_container"]
        assert reference_named(pour, "target_container") == Reference(
            name="target_container",
            type="Bowl",
            identifier=bowl["id"]["value"],
            label=bowl["name"]["name"],
            pose=None,
            parameters=[],
        )

    def test_a_thing_the_database_does_not_keep_is_of_the_class_it_is_stored_as(
        self, database
    ):
        motion = node_of(database.nodes(1), 2)
        tool = reference_named(motion, "tip_link")
        assert (tool.type, tool.label) == (
            "ReferencedWorldEntity",
            "r_gripper_tool_frame",
        )

    def test_the_references_keep_the_order_of_the_fields(self, database):
        pour = node_of(database.nodes(1), 3)
        assert [reference.name for reference in pour.references] == [
            "target_container",
            "source_container",
            "arm",
        ]

    def test_each_thing_of_a_list_is_a_reference(self, database):
        park = node_of(database.nodes(2), 1)
        assert [(reference.name, reference.label) for reference in park.references] == [
            ("arms", "PR2LeftArm"),
            ("arms", "PR2RightArm"),
        ]

    def test_a_pose_has_its_position_orientation_and_frame(self, database):
        motion = node_of(database.nodes(1), 2)
        assert reference_named(motion, "goal_pose") == Reference(
            name="goal_pose",
            type="Pose",
            identifier=None,
            label=None,
            pose=Pose(
                frame="Static_MilkPitcher.stl",
                position=Position(x=-0.168, y=0.0, z=0.0),
                orientation=Orientation(x=0.0, y=0.0, z=0.0, w=1.0),
            ),
            parameters=[],
        )

    def test_a_reference_that_is_not_set_is_a_value_that_is_not_set(self, database):
        change_database(database.path, Change.SOURCE_CONTAINER_UNSET)
        pour = node_of(database.nodes(1), 3)
        assert [reference.name for reference in pour.references] == [
            "target_container",
            "arm",
        ]
        assert values_of(pour)["source_container"] is None

    def test_a_step_below_a_step_is_no_reference_of_it(self, database):
        navigation = node_of(database.nodes(2), 3)
        watching = node_of(database.nodes(1), 6)
        assert [reference.name for reference in navigation.references] == [
            "target_location"
        ]
        assert [reference.name for reference in watching.references] == ["timeout"]

    def test_an_object_that_is_only_values_carries_them(self, database):
        pick_up = node_of(database.nodes(1), 1)
        grasp = reference_named(pick_up, "grasp")
        assert (grasp.type, grasp.label, grasp.pose) == ("GraspCandidate", None, None)
        assert {parameter.name: parameter.value for parameter in grasp.parameters} == {
            "approach_direction": APPROACH_FROM_THE_FRONT,
            "manipulation_offset": 0.1,
        }

    def test_a_reference_is_json(self, database):
        pour = node_of(database.nodes(1), 3)
        bowl = reference_named(pour, "target_container")
        assert bowl.to_json() == {
            "name": "target_container",
            "type": "Bowl",
            "identifier": bowl.identifier,
            "label": bowl.label,
            "pose": None,
            "parameters": [],
        }

    def test_a_pose_is_json(self, database):
        navigation = node_of(database.nodes(2), 3)
        assert navigation.references[0].pose.to_json() == {
            "frame": "map",
            "position": {"x": 0.5, "y": -1.0, "z": 0.0},
            "orientation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
        }


# %% what a step keeps


class TestWhatAStepKeeps:
    def test_a_step_keeps_its_fields_without_how_it_is_drawn_or_the_steps_below(
        self, database
    ):
        navigation = node_of(database.nodes(2), 3)
        stored = stored_node(KeptPlan.RUN, 3)
        assert navigation.stored == {
            key: value
            for key, value in stored.items()
            if key not in (JsonKey.PLOT, "goal")
        }

    def test_a_step_is_json(self, database):
        motion = node_of(database.nodes(1), 4)
        assert motion.to_json() == {
            "id": 4,
            "parent": 3,
            "depth": 2,
            "type": "JointPositionList",
            "designator": None,
            "status": LifeCycle.FAILED.name,
            "started": motion.started,
            "ended": motion.ended,
            "parameters": [{"name": "threshold", "value": 0.01}],
            "references": [],
            "stored": motion.stored,
        }


# %% a database that is broken


class TestADatabaseThatIsBroken:
    def test_a_plan_whose_context_is_gone_is_reported_with_what_is_missing(
        self, database
    ):
        change_database(database.path, Change.CONTEXT_DELETED)
        with pytest.raises(MissingRow) as missing:
            database.nodes(1)
        assert missing.value.table == Table.CONTEXT
        assert missing.value.identifier == 1


# %% what the page is tested on


class TestThePagesFixtures:
    @pytest.mark.parametrize("fixture", PAGE_FIXTURES, ids=lambda fixture: fixture.file)
    def test_a_fixture_of_the_page_is_what_the_server_sends(self, fixture):
        """
        When this fails the server's answer has changed: call ``write_page_fixtures`` to
        write the fixtures again, and look at what the page's tests say of them.
        """
        written = json.loads((PAGE_FIXTURES_DIRECTORY / fixture.file).read_text())
        assert written == sent_by_the_server(fixture)
