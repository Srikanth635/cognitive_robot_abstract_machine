"""
Tests of reading the steps of a plan out of the NEEM database: the tree they form, what
each was asked to do, and the things and places it refers to.
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
    TableRow,
)
from llm_reasoner_viz.neem_schema import Table
from llm_reasoner_viz.neems import NeemDatabase, UnknownNeem

from .neem_files import (
    PAGE_FIXTURES,
    PAGE_FIXTURES_DIRECTORY,
    Change,
    change_database,
    sent_by_the_server,
    write_neem_database,
)

POURING_TEMPLATE = "giskardpy.motion_statechart.goals.templates.Sequence"
RIGHT_ARM = "coraplex.datastructures.enums.Arms.RIGHT"
BOTH_ARMS = "coraplex.datastructures.enums.Arms.BOTH"

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


# %% the tree


class TestTheTree:
    def test_the_steps_come_in_the_order_of_the_tree(self, database):
        assert [node.identifier for node in database.nodes(1)] == [1, 2, 3, 4, 5]

    def test_a_step_knows_the_one_above_it_and_how_deep_it_is(self, database):
        nodes = database.nodes(1)
        assert [node.parent for node in nodes] == [None, 1, 1, 3, 3]
        assert [node.depth for node in nodes] == [0, 1, 1, 2, 2]

    def test_the_steps_below_a_step_keep_the_order_of_their_edges(self, database):
        assert [node.identifier for node in database.nodes(2)] == [11, 12, 13, 14]

    def test_a_plan_the_database_does_not_hold_is_unknown(self, database):
        with pytest.raises(UnknownNeem) as unknown:
            database.nodes(99)
        assert unknown.value.neem == 99

    def test_without_a_database_every_plan_is_unknown(self, tmp_path):
        with pytest.raises(UnknownNeem):
            NeemDatabase(path=tmp_path / "none.sqlite").nodes(1)


# %% what a step is


class TestWhatAStepIs:
    def test_a_step_is_the_kind_of_node_it_is_stored_as(self, database):
        assert [node.type for node in database.nodes(1)] == [
            "SequentialNode",
            "ActionNode",
            "ActionNode",
            "ConditionNode",
            "MotionNode",
        ]

    def test_an_action_or_a_motion_has_the_designator_it_was_asked_to_perform(
        self, database
    ):
        assert [node.designator for node in database.nodes(1)] == [
            None,
            "ParkArmsAction",
            "PouringAction",
            None,
            "MoveToolCenterPointMotion",
        ]

    def test_a_sequence_says_which_template_runs_it(self, database):
        sequence = node_of(database.nodes(1), 1)
        assert values_of(sequence) == {"motion_state_chart_template": POURING_TEMPLATE}

    def test_a_condition_says_whether_it_is_checked_before(self, database):
        condition = node_of(database.nodes(1), 4)
        assert values_of(condition) == {"pre_condition": True}
        assert condition.references == []


# %% what a step was asked to do


class TestWhatAStepWasAskedToDo:
    def test_the_values_of_an_action_are_those_of_every_table_it_extends(
        self, database
    ):
        pour = node_of(database.nodes(1), 3)
        assert values_of(pour) == {
            "position_threshold": None,
            "orientation_threshold": None,
            "tilt_angle": 1.85,
            "pour_side_offset": 0.0,
            "pour_approach_offset": 0.0,
            "pour_height": 0.13,
            "arm": RIGHT_ARM,
            "pour_side": RIGHT_ARM,
        }

    def test_the_values_keep_the_order_of_the_tables(self, database):
        pour = node_of(database.nodes(1), 3)
        assert [parameter.name for parameter in pour.parameters][:4] == [
            "position_threshold",
            "orientation_threshold",
            "tilt_angle",
            "pour_side_offset",
        ]

    def test_a_flag_is_true_or_false(self, database):
        navigation = node_of(database.nodes(2), 13)
        assert values_of(navigation)["keep_joint_states"] is True

    def test_an_action_without_the_values_of_a_pour_has_only_its_own(self, database):
        park = node_of(database.nodes(1), 2)
        assert values_of(park) == {"max_joint_velocity": None, "arm": BOTH_ARMS}

    def test_a_parameter_is_json(self):
        assert Parameter(name="tilt_angle", value=1.85).to_json() == {
            "name": "tilt_angle",
            "value": 1.85,
        }


# %% what a step refers to


class TestWhatAStepRefersTo:
    def test_a_body_is_named_as_the_world_names_it(self, database):
        pour = node_of(database.nodes(1), 3)
        bowl = next(
            reference
            for reference in pour.references
            if reference.name == "target_container"
        )
        assert bowl == Reference(
            name="target_container",
            type="Body",
            identifier=101,
            label="bowl.stl",
            pose=None,
            parameters=[],
        )

    def test_the_references_keep_the_order_of_the_columns(self, database):
        pour = node_of(database.nodes(1), 3)
        assert [reference.name for reference in pour.references] == [
            "target_container",
            "source_container",
        ]

    def test_a_thing_is_of_the_class_its_row_says_it_really_is(self, database):
        pour = node_of(database.nodes(1), 3)
        source = pour.references[1]
        assert (source.type, source.label) == ("PouringCup", "PouringCup")

    def test_a_pose_has_its_position_and_the_frame_it_is_measured_from(self, database):
        motion = node_of(database.nodes(1), 5)
        (target,) = motion.references
        assert target == Reference(
            name="target",
            type="Pose",
            identifier=300,
            label=None,
            pose=Pose(
                frame="map",
                position=Position(x=1.5, y=2.25, z=0.0),
                orientation=None,
            ),
            parameters=[],
        )

    def test_a_pose_has_its_orientation_when_it_has_one(self, database):
        navigation = node_of(database.nodes(2), 13)
        (target,) = navigation.references
        assert target.pose == Pose(
            frame="map",
            position=Position(x=0.5, y=-1.0, z=0.0),
            orientation=Orientation(x=0.0, y=0.0, z=0.0, w=1.0),
        )

    def test_a_reference_that_is_not_set_is_left_out(self, database):
        change_database(database.path, Change.SOURCE_CONTAINER_UNSET)
        pour = node_of(database.nodes(1), 3)
        assert [reference.name for reference in pour.references] == ["target_container"]

    def test_a_step_that_moves_a_body_refers_to_the_body_and_its_new_parent(
        self, database
    ):
        change_database(database.path, Change.RE_ATTACH_STEP)
        move = node_of(database.nodes(1), 6)
        assert [(reference.name, reference.label) for reference in move.references] == [
            ("body", "bowl.stl"),
            ("new_parent", "map"),
        ]

    def test_what_the_execution_recorded_is_no_reference_of_the_step(self, database):
        change_database(database.path, Change.EXECUTION_RECORDED)
        pour = node_of(database.nodes(1), 3)
        assert [reference.name for reference in pour.references] == [
            "target_container",
            "source_container",
        ]

    def test_the_step_a_condition_belongs_to_is_no_reference_of_it(self, database):
        condition = node_of(database.nodes(1), 4)
        assert condition.references == []

    def test_an_object_that_is_only_values_carries_them(self, database):
        change_database(database.path, Change.PICK_UP_ACTION)
        pick_up = node_of(database.nodes(1), 3)
        (grasp,) = pick_up.references
        assert (grasp.name, grasp.type, grasp.label, grasp.pose) == (
            "grasp_description",
            "GraspDescription",
            None,
            None,
        )
        assert {parameter.name: parameter.value for parameter in grasp.parameters} == {
            "rotate_gripper": False,
            "manipulation_offset": 0.1,
            "approach_direction": "coraplex.datastructures.enums.ApproachDirection.FRONT",
            "vertical_alignment": "coraplex.datastructures.enums.VerticalAlignment.NoAlignment",
        }

    def test_a_thing_of_a_world_and_a_pose_carry_no_values_of_their_own(self, database):
        pour = node_of(database.nodes(1), 3)
        navigation = node_of(database.nodes(2), 13)
        assert [reference.parameters for reference in pour.references] == [[], []]
        assert navigation.references[0].parameters == []

    def test_a_reference_is_json(self, database):
        pour = node_of(database.nodes(1), 3)
        assert pour.references[0].to_json() == {
            "name": "target_container",
            "type": "Body",
            "identifier": 101,
            "label": "bowl.stl",
            "pose": None,
            "parameters": [],
        }

    def test_a_pose_is_json(self, database):
        navigation = node_of(database.nodes(2), 13)
        assert navigation.references[0].pose.to_json() == {
            "frame": "map",
            "position": {"x": 0.5, "y": -1.0, "z": 0.0},
            "orientation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
        }


# %% the rows a step is kept in


class TestTheRowsAStepIsKeptIn:
    def test_a_step_lists_the_row_of_each_table_it_is_spread_over(self, database):
        pour = node_of(database.nodes(1), 3)
        assert [row.table for row in pour.rows] == [
            Table.ENTITY,
            Table.NODE,
            "DesignatorNodeDAO",
            Table.ACTION_NODE,
            Table.DESIGNATOR,
            "ActionDescriptionDAO",
            "FullBodyControlledActionDAO",
            "PouringActionDAO",
        ]

    def test_a_row_holds_the_values_of_all_the_columns_as_stored(self, database):
        pour = node_of(database.nodes(1), 3)
        row = next(row for row in pour.rows if row.table == "PouringActionDAO")
        assert row.values["tilt_angle"] == 1.85
        assert row.values["_target_container_id"] == 101

    def test_a_column_of_bytes_is_left_out_of_the_row_and_the_parameters(
        self, database
    ):
        change_database(database.path, Change.DESIGNATOR_WITH_BYTES)
        scan = node_of(database.nodes(1), 3)
        assert scan.rows[-1] == TableRow(
            table="ScanActionDAO", values={"database_id": 11, "resolution": 2.5}
        )
        assert values_of(scan) == {"resolution": 2.5}

    def test_a_condition_is_json_with_its_rows(self, database):
        condition = node_of(database.nodes(1), 4)
        assert condition.to_json() == {
            "id": 4,
            "parent": 3,
            "depth": 2,
            "type": "ConditionNode",
            "designator": None,
            "parameters": [{"name": "pre_condition", "value": True}],
            "references": [],
            "rows": [
                TableRow(
                    table=Table.ENTITY,
                    values={
                        "database_id": 4,
                        "polymorphic_type": "ConditionNodeDAO",
                        "_plan_id": 1,
                    },
                ).to_json(),
                TableRow(
                    table=Table.NODE,
                    values={
                        "database_id": 4,
                        "start_time": "2026-10-14 16:48:50.000000",
                        "end_time": None,
                        "status": (
                            "giskardpy.motion_statechart.data_types."
                            "LifeCycleValues.NOT_STARTED"
                        ),
                        "_reason_id": None,
                    },
                ).to_json(),
                TableRow(
                    table="ConditionNodeDAO",
                    values={
                        "database_id": 4,
                        "pre_condition": True,
                        "_action_node_id": 3,
                    },
                ).to_json(),
            ],
        }


# %% a database that is broken


class TestADatabaseThatIsBroken:
    def test_a_step_whose_designator_is_gone_is_reported_with_what_is_missing(
        self, database
    ):
        change_database(database.path, Change.DESIGNATOR_DELETED)
        with pytest.raises(MissingRow) as missing:
            database.nodes(1)
        assert missing.value.table == Table.DESIGNATOR
        assert missing.value.identifier == 11


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
