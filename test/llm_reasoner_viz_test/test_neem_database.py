"""
Tests of reading the database the NEEMs are kept in: which plans it holds and how each
went, read without ever changing the database, and refused when it is not one.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from cramera.resolution import paths as cramera_paths

from llm_reasoner_viz import paths
from llm_reasoner_viz.neem_schema import LifeCycle, Table
from llm_reasoner_viz.neems import (
    NeemDatabase,
    NeemDatabaseBusy,
    PlanSummary,
    UnknownNeemSchema,
)

from .neem_files import (
    Change,
    KeptPlan,
    being_written,
    change_database,
    write_neem_database,
)

# %% helpers


def tick_duration(plan: KeptPlan) -> float:
    """
    :return: How many seconds a tick of the plan lasted.
    """
    return plan.recording["tick_duration"]


def ended_tick(plan: KeptPlan) -> int:
    """
    :return: The first tick the node the plan was performed from had ended on.
    """
    return next(
        states["tick"]
        for states in plan.recording["ticks"]
        if LifeCycle(states["life_cycles"][0]).is_terminal
    )


# %% where the database is


class TestWhereTheDatabaseIs:
    def test_by_default_it_is_the_database_cramera_keeps_in_its_data_directory(
        self, tmp_path, monkeypatch
    ):
        monkeypatch.delenv(paths.EnvironmentVariable.NEEM_DATABASE, raising=False)
        monkeypatch.setenv(paths.EnvironmentVariable.CRAMERA_DATA, str(tmp_path))
        assert paths.neem_database_path() == cramera_paths.neem_database_path()

    def test_the_environment_can_name_another_database(self, tmp_path, monkeypatch):
        monkeypatch.setenv(
            paths.EnvironmentVariable.NEEM_DATABASE, str(tmp_path / "elsewhere.sqlite")
        )
        assert paths.neem_database_path() == tmp_path / "elsewhere.sqlite"

    def test_the_environment_names_another_database_as_it_does_for_cramera(
        self, tmp_path, monkeypatch
    ):
        monkeypatch.setenv(
            paths.EnvironmentVariable.NEEM_DATABASE, str(tmp_path / "elsewhere.sqlite")
        )
        assert paths.neem_database_path() == cramera_paths.neem_database_path()


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

    def test_a_plan_is_how_the_node_it_was_performed_from_ended(self, database):
        first, second = database.plans()
        assert first.status == LifeCycle.FAILED.name
        assert second.status == LifeCycle.SUCCEEDED.name

    def test_a_plan_lasts_until_the_node_it_was_performed_from_ended(self, database):
        first, _ = database.plans()
        assert first.seconds == ended_tick(KeptPlan.POUR) * tick_duration(KeptPlan.POUR)

    def test_a_plan_whose_root_has_not_ended_is_running_and_has_no_duration(
        self, database
    ):
        change_database(database.path, Change.PLAN_NOT_ENDED)
        _, second = database.plans()
        assert second.status == LifeCycle.RUNNING.name
        assert second.seconds is None

    def test_a_plan_counts_every_node_of_its_statechart(self, database):
        assert [plan.nodes for plan in database.plans()] == [
            len(plan.recording["structure"]["nodes"]) for plan in KeptPlan
        ]

    def test_a_plan_counts_the_actions_directly_below_its_root_as_its_steps(
        self, database
    ):
        assert [plan.steps for plan in database.plans()] == [2, 3]

    def test_a_plan_is_written_as_data_for_the_page(self, database):
        first, _ = database.plans()
        assert first.to_json() == {
            "neem": 1,
            "status": first.status,
            "seconds": first.seconds,
            "nodes": first.nodes,
            "steps": first.steps,
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
        assert refused.value.missing == Table.STATECHART
