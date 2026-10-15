"""
Tests of reading how the tables of the NEEM database extend one another: the ORM keeps a
class in one table per level, each pointing at the table of the class it extends.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest
from typing_extensions import Iterator

from llm_reasoner_viz.neem_schema import Column, Table
from llm_reasoner_viz.neem_tables import TableColumn, TableSchema

from .neem_files import write_neem_database

POURING = "PouringActionDAO"
"""
A table three levels below the root of the designators.
"""


@pytest.fixture
def schema(tmp_path: Path) -> Iterator[TableSchema]:
    """
    :return: The schema of a database of the tables a plan is kept in.
    """
    connection = sqlite3.connect(write_neem_database(tmp_path / "neems.sqlite"))
    yield TableSchema(connection=connection)
    connection.close()


def column_of(schema: TableSchema, table: str, name: str) -> TableColumn:
    return next(column for column in schema.columns(table) if column.name == name)


# %% what a table extends


class TestWhatATableExtends:
    def test_a_table_extends_the_one_its_identifier_refers_to(self, schema):
        assert schema.parent_of(POURING) == "FullBodyControlledActionDAO"

    def test_the_table_at_the_root_extends_none(self, schema):
        assert schema.parent_of(Table.DESIGNATOR) is None

    def test_a_chain_runs_from_the_root_to_the_table(self, schema):
        assert schema.chain(POURING) == [
            Table.DESIGNATOR,
            "ActionDescriptionDAO",
            "FullBodyControlledActionDAO",
            POURING,
        ]

    def test_the_chain_of_a_root_is_the_root_alone(self, schema):
        assert schema.chain(Table.DESIGNATOR) == [Table.DESIGNATOR]

    def test_a_table_is_known_when_the_database_has_it(self, schema):
        assert schema.has_table(POURING)
        assert not schema.has_table("NoSuchDAO")


# %% the columns of a table


class TestTheColumnsOfATable:
    def test_the_columns_come_in_the_order_of_the_table(self, schema):
        assert [column.name for column in schema.columns(POURING)][:4] == [
            Column.IDENTIFIER,
            "position_threshold",
            "orientation_threshold",
            "tilt_angle",
        ]

    def test_a_column_that_refers_to_a_table_says_which(self, schema):
        assert (
            column_of(schema, POURING, "_target_container_id").references == "BodyDAO"
        )

    def test_the_identifier_that_extends_a_table_is_no_reference(self, schema):
        assert column_of(schema, POURING, Column.IDENTIFIER).references is None

    def test_a_plain_column_refers_to_no_table(self, schema):
        assert column_of(schema, POURING, "tilt_angle").references is None

    def test_a_reference_is_named_for_what_it_holds(self, schema):
        assert (
            column_of(schema, POURING, "_target_container_id").attribute
            == "target_container"
        )

    def test_a_plain_column_is_named_as_it_is(self, schema):
        assert column_of(schema, POURING, "tilt_angle").attribute == "tilt_angle"

    def test_a_boolean_column_says_so(self, schema):
        assert column_of(schema, "ConditionNodeDAO", "pre_condition").is_boolean
        assert not column_of(schema, POURING, "tilt_angle").is_boolean

    def test_a_column_of_bytes_says_so(self, schema):
        assert column_of(
            schema, "ExecutionDataDAO", "execution_start_world_state"
        ).is_binary
        assert not column_of(schema, POURING, "tilt_angle").is_binary

    def test_the_columns_that_hold_the_parameters_leave_out_what_is_structure(
        self, schema
    ):
        assert [column.name for column in schema.parameter_columns(POURING)] == [
            "position_threshold",
            "orientation_threshold",
            "tilt_angle",
            "pour_side_offset",
            "pour_approach_offset",
            "pour_height",
            "arm",
            "pour_side",
        ]

    def test_the_columns_that_refer_to_other_things_are_apart(self, schema):
        assert [column.name for column in schema.reference_columns(POURING)] == [
            "_target_container_id",
            "_source_container_id",
        ]
