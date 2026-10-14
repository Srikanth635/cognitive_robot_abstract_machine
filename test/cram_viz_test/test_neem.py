"""
Tests of keeping what the simulation performs as NEEMs: the plan of a pour tried or of a
run performed, saved in one shared database, and reported to the run's log.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field
from pathlib import Path

import pytest
from coraplex.plans.plan import Plan
from krrood.ormatic.data_access_objects.helper import to_dao
from llm_reasoner.observation.sinks import MemorySink
from typing_extensions import Any, List

from cram_viz import paths
from cram_viz.events import NeemKind, NeemNotRecorded, NeemRecorded
from cram_viz.neem import (
    DatabaseNeemRecorder,
    NeemRecorder,
    NeemStore,
    NoNeemRecorder,
    SqliteNeemStore,
)
from cram_viz.resolution.execution import PouringExecution

from .resolutions import objects_given

# %% mimics


@dataclass
class PlanKeepingStore(NeemStore):
    """
    Keeps the plans it is given, numbering them from one.
    """

    plans: List[Any] = field(default_factory=list)
    """
    The plans kept, in order.
    """

    def save(self, plan: Plan) -> int:
        self.plans.append(plan)
        return len(self.plans)


class DiskIsFull(OSError):
    """
    There is no room to keep anything.
    """


@dataclass
class FullStore(NeemStore):
    """
    Keeps nothing: the disk it keeps its plans on is full.
    """

    def save(self, plan: Plan) -> int:
        raise DiskIsFull("no space left on device")


@dataclass
class PlanCollectingRecorder(NeemRecorder):
    """
    Keeps the plans it is asked to record, to see which they were.
    """

    plans: List[Any] = field(default_factory=list)
    """
    The plans recorded, in order.
    """

    def record(self, plan: Plan) -> None:
        self.plans.append(plan)


class MotionFailed(RuntimeError):
    """
    The simulated robot could not finish a motion.
    """


@dataclass
class FailingNode:
    """
    A plan's root that fails when it is performed.
    """

    plan: Any
    """
    The plan the node belongs to.
    """

    def perform(self) -> None:
        raise MotionFailed("the arm cannot reach")


# %% recording a plan


class TestRecordingAPlan:
    def test_a_plan_kept_is_reported_with_what_it_is_of_and_where_it_is_kept(
        self, tmp_path
    ):
        store = PlanKeepingStore(path=tmp_path / "neems.sqlite")
        sink = MemorySink()
        plan = object()
        DatabaseNeemRecorder(
            store=store, kind=NeemKind.TRIAL, episode="a-trial", observer=sink
        ).record(plan)
        (event,) = sink.events
        assert isinstance(event, NeemRecorded)
        assert event.kind == NeemKind.TRIAL
        assert event.episode == "a-trial"
        assert event.database == str(store.path)
        assert event.database_id == len(store.plans)
        assert store.plans == [plan]

    def test_the_time_it_took_to_keep_the_plan_is_reported(self, tmp_path):
        sink = MemorySink()
        DatabaseNeemRecorder(
            store=PlanKeepingStore(path=tmp_path / "neems.sqlite"),
            kind=NeemKind.EXECUTION,
            episode="a-run",
            observer=sink,
        ).record(object())
        (event,) = sink.events
        assert event.seconds >= 0

    def test_a_plan_that_cannot_be_kept_is_reported_and_does_not_stop_the_run(
        self, tmp_path
    ):
        sink = MemorySink()
        DatabaseNeemRecorder(
            store=FullStore(path=tmp_path / "neems.sqlite"),
            kind=NeemKind.EXECUTION,
            episode="a-run",
            observer=sink,
        ).record(object())
        (event,) = sink.events
        assert isinstance(event, NeemNotRecorded)
        assert event.kind == NeemKind.EXECUTION
        assert event.episode == "a-run"
        assert event.error_type == DiskIsFull.__name__
        assert event.reason == str(DiskIsFull("no space left on device"))

    def test_recording_nothing_keeps_nothing(self):
        assert NoNeemRecorder().record(object()) is None


# %% where the plans are kept


class TestWhereNeemsAreKept:
    def test_by_default_they_are_kept_in_one_database_of_the_data_directory(
        self, tmp_path, monkeypatch
    ):
        monkeypatch.delenv(paths.NEEM_DATABASE_ENVIRONMENT, raising=False)
        monkeypatch.setenv("CRAM_VIZ_DATA", str(tmp_path))
        assert paths.neem_database_path() == tmp_path / paths.NEEM_DATABASE_FILE

    def test_the_environment_can_name_another_database(self, tmp_path, monkeypatch):
        monkeypatch.setenv(
            paths.NEEM_DATABASE_ENVIRONMENT, str(tmp_path / "elsewhere.sqlite")
        )
        assert paths.neem_database_path() == tmp_path / "elsewhere.sqlite"


# %% performing records


class TestPerformingRecords:
    def test_performing_records_the_plan_it_performed(self, pouring_kitchen):
        recorder = PlanCollectingRecorder()
        PouringExecution.for_resolution(
            pouring_kitchen, objects_given(), recorder=recorder
        ).perform()
        (plan,) = recorder.plans
        assert isinstance(plan, Plan)

    def test_a_plan_whose_performance_failed_is_recorded_and_the_failure_goes_on(
        self, monkeypatch
    ):
        recorder = PlanCollectingRecorder()
        performed = object()
        execution = PouringExecution(world=None, values={}, recorder=recorder)
        monkeypatch.setattr(
            PouringExecution, "plan", lambda self: FailingNode(plan=performed)
        )
        with pytest.raises(MotionFailed):
            execution.perform()
        assert recorder.plans == [performed]

    def test_an_execution_records_nothing_unless_it_is_given_a_recorder(
        self, monkeypatch
    ):
        performed = object()
        execution = PouringExecution(world=None, values={})
        monkeypatch.setattr(
            PouringExecution, "plan", lambda self: FailingNode(plan=performed)
        )
        with pytest.raises(MotionFailed):
            execution.perform()


# %% the database


@pytest.mark.slow
class TestTheSqliteStore:
    """
    Slow: the first plan saved makes the database's tables, of which there are over a
    thousand.
    """

    def test_plans_performed_are_kept_one_after_another_in_one_database(
        self, pouring_kitchen, tmp_path
    ):
        recorder = PlanCollectingRecorder()
        PouringExecution.for_resolution(
            pouring_kitchen, objects_given(), recorder=recorder
        ).perform()
        (plan,) = recorder.plans
        store = SqliteNeemStore(path=tmp_path / "neems.sqlite")

        first = store.save(plan)
        second = store.save(plan)

        assert second != first
        plans = to_dao(plan).__tablename__
        with sqlite3.connect(store.path) as connection:
            kept = connection.execute(
                f'select database_id from "{plans}" order by database_id'
            ).fetchall()
        assert kept == [(first,), (second,)]
