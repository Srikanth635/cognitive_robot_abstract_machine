"""
Tests of reading the database the NEEMs are kept in: which plans it holds and how each
went, read without ever changing the database, and refused when it is not one.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pytest

from llm_reasoner_viz import paths
from llm_reasoner_viz.neem_schema import Table
from llm_reasoner_viz.neems import (
    NeemDatabase,
    NeemDatabaseBusy,
    PlanSummary,
    UnknownNeemSchema,
)

from .neem_files import Change, being_written, change_database, write_neem_database

# %% where the database is


class TestWhereTheDatabaseIs:
    def test_by_default_it_is_the_database_of_cram_viz_in_its_data_directory(
        self, tmp_path, monkeypatch
    ):
        monkeypatch.delenv(paths.NEEM_DATABASE_ENVIRONMENT, raising=False)
        monkeypatch.setenv(paths.DATA_ENVIRONMENT, str(tmp_path))
        assert paths.neem_database_path() == tmp_path / paths.NEEM_DATABASE_FILE

    def test_the_environment_can_name_another_database(self, tmp_path, monkeypatch):
        monkeypatch.setenv(
            paths.NEEM_DATABASE_ENVIRONMENT, str(tmp_path / "elsewhere.sqlite")
        )
        assert paths.neem_database_path() == tmp_path / "elsewhere.sqlite"


# %% finding the database


class TestFindingTheDatabase:
    def test_a_database_that_is_not_there_does_not_exist_and_holds_no_plan(
        self, tmp_path
    ):
        database = NeemDatabase(path=tmp_path / "none.sqlite")
        assert not database.exists
        assert database.plans() == []

    def test_a_database_that_is_there_exists(self, tmp_path):
        assert NeemDatabase(path=write_neem_database(tmp_path / "neems.sqlite")).exists


# %% the plans


class TestPlans:
    @pytest.fixture()
    def database(self, tmp_path: Path) -> NeemDatabase:
        return NeemDatabase(path=write_neem_database(tmp_path / "neems.sqlite"))

    def test_every_plan_is_listed_in_the_order_of_its_identifiers(self, database):
        assert [plan.neem for plan in database.plans()] == [1, 2]

    def test_a_plan_is_how_its_root_ended(self, database):
        first, _ = database.plans()
        assert first.status == "SUCCEEDED"
        assert first.started == datetime(2026, 10, 14, 16, 48, 50)
        assert first.ended == datetime(2026, 10, 14, 16, 48, 52, 500000)
        assert first.seconds == 2.5

    def test_a_plan_whose_root_has_not_ended_has_no_duration(self, database):
        change_database(database.path, Change.END_TIME_CLEARED)
        first, _ = database.plans()
        assert first.ended is None
        assert first.seconds is None

    def test_a_plan_counts_its_nodes_the_root_included(self, database):
        assert [plan.nodes for plan in database.plans()] == [5, 4]

    def test_a_plan_counts_the_actions_directly_below_its_root_as_its_steps(
        self, database
    ):
        assert [plan.steps for plan in database.plans()] == [2, 3]

    def test_a_plan_is_written_as_data_for_the_page(self, database):
        first, _ = database.plans()
        assert first.to_json() == {
            "neem": 1,
            "status": "SUCCEEDED",
            "started": "2026-10-14T16:48:50",
            "ended": "2026-10-14T16:48:52.500000",
            "seconds": 2.5,
            "nodes": 5,
            "steps": 2,
        }

    def test_a_database_without_plans_lists_none(self, tmp_path):
        database = NeemDatabase(
            path=write_neem_database(tmp_path / "empty.sqlite", plans=False)
        )
        assert database.plans() == []


# %% reading safely


class TestReadingSafely:
    def test_reading_leaves_the_database_as_it_was(self, tmp_path):
        path = write_neem_database(tmp_path / "neems.sqlite")
        before = path.read_bytes()
        NeemDatabase(path=path).plans()
        assert path.read_bytes() == before

    def test_a_database_being_written_is_reported_as_busy(self, tmp_path):
        path = write_neem_database(tmp_path / "neems.sqlite")
        with being_written(path):
            with pytest.raises(NeemDatabaseBusy):
                NeemDatabase(path=path, timeout_seconds=0.1).plans()

    def test_a_database_that_is_not_one_of_neems_is_refused_with_the_table_it_lacks(
        self, tmp_path
    ):
        path = tmp_path / "other.sqlite"
        change_database(path, Change.UNRELATED_TABLE)
        with pytest.raises(UnknownNeemSchema) as refused:
            NeemDatabase(path=path).plans()
        assert refused.value.missing == Table.PLAN
