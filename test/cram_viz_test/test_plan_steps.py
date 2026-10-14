"""
Tests of following a plan's steps as the bridge sees them: the steps of the plan, not
the nodes below them, reported when their status changes, and what Execute reports of a
resolution run being performed.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from llm_reasoner.observation.events import EventKey
from llm_reasoner.observation.sinks import JsonLinesSink, MemorySink
from typing_extensions import Any, Dict, List

from cram_viz.events import (
    NeemKind,
    ExecutionFailed,
    ExecutionFinished,
    ExecutionStarted,
    StepChanged,
)
from cram_viz.live.bridge import BRIDGE, Bridge, PlanNodeKey
from cram_viz.live.runner import ResolvedRun
from cram_viz.live.steps import NOT_STARTED, PlanStepReporter
from cram_viz.neem import DatabaseNeemRecorder, NoNeemRecorder, SqliteNeemStore
from cram_viz.resolution.execution import PouringExecution
from cram_viz.resolution.runs import RunFile

# %% helpers


def node(
    identifier: str, parent: str | None, label: str, status: str
) -> Dict[str, Any]:
    """
    :return: A plan node as the bridge snapshots it.
    """
    return {
        PlanNodeKey.ID: identifier,
        PlanNodeKey.PARENT: parent,
        PlanNodeKey.KIND: "ActionNode",
        PlanNodeKey.LABEL: label,
        PlanNodeKey.STATUS: status,
    }


def plan_at(*statuses: str) -> List[Dict[str, Any]]:
    """
    :return: A plan of a root and two steps, each with a motion beneath it, the steps and
        the motions in the given statuses in turn.
    """
    first, second, motion_of_first, motion_of_second = statuses
    return [
        node("root", None, "SequentialNode", "RUNNING"),
        node("first", "root", "ParkArmsAction", first),
        node("motion-first", "first", "ParkArmsMotion", motion_of_first),
        node("second", "root", "NavigateAction", second),
        node("motion-second", "second", "NavigateMotion", motion_of_second),
    ]


# %% the bridge


class TestBridge:
    def test_an_observer_is_given_the_nodes_of_every_plan_snapshot(self):
        bridge = Bridge()
        seen: List[Any] = []
        bridge.plan_observers.append(seen.append)
        bridge._plan = type("StubPlan", (), {"root": _stub_root()})()
        bridge.snapshot_plan()
        assert seen == [bridge.get_plan()["nodes"]]


def _stub_root():
    """
    :return: A plan node with no children, as far as the bridge reads it.
    """
    status = type("StubStatus", (), {"name": "CREATED"})()
    return type(
        "StubNode",
        (),
        {
            "status": status,
            "designator": None,
            "children": [],
            "parent_action_node": None,
        },
    )()


# %% following the steps


class TestPlanStepReporter:
    def test_a_step_that_starts_is_reported_with_its_status(self):
        sink = MemorySink()
        reporter = PlanStepReporter(sink=sink, run="a-run")
        reporter.observe(plan_at(NOT_STARTED, NOT_STARTED, NOT_STARTED, NOT_STARTED))
        reporter.observe(plan_at("RUNNING", NOT_STARTED, "RUNNING", NOT_STARTED))
        (changed,) = sink.events
        assert type(changed) is StepChanged
        assert (changed.run, changed.step, changed.status) == (
            "a-run",
            "ParkArmsAction",
            "RUNNING",
        )

    def test_a_step_not_started_is_not_reported(self):
        sink = MemorySink()
        reporter = PlanStepReporter(sink=sink, run="a-run")
        reporter.observe(plan_at(NOT_STARTED, NOT_STARTED, NOT_STARTED, NOT_STARTED))
        assert sink.events == []

    def test_a_status_is_reported_once_however_often_it_is_seen(self):
        sink = MemorySink()
        reporter = PlanStepReporter(sink=sink, run="a-run")
        for _ in range(3):
            reporter.observe(plan_at("RUNNING", NOT_STARTED, NOT_STARTED, NOT_STARTED))
        assert len(sink.events) == 1

    def test_each_change_of_a_step_is_reported_in_order(self):
        sink = MemorySink()
        reporter = PlanStepReporter(sink=sink, run="a-run")
        reporter.observe(plan_at("RUNNING", NOT_STARTED, NOT_STARTED, NOT_STARTED))
        reporter.observe(plan_at("SUCCEEDED", "RUNNING", NOT_STARTED, NOT_STARTED))
        reporter.observe(plan_at("SUCCEEDED", "SUCCEEDED", NOT_STARTED, NOT_STARTED))
        assert [(event.step, event.status) for event in sink.events] == [
            ("ParkArmsAction", "RUNNING"),
            ("ParkArmsAction", "SUCCEEDED"),
            ("NavigateAction", "RUNNING"),
            ("NavigateAction", "SUCCEEDED"),
        ]

    def test_only_the_steps_of_the_plan_are_reported_not_the_nodes_below_them(self):
        sink = MemorySink()
        reporter = PlanStepReporter(sink=sink, run="a-run")
        reporter.observe(plan_at(NOT_STARTED, NOT_STARTED, "RUNNING", "FAILED"))
        assert sink.events == []


# %% performing a run


class StubExecution:
    """
    An execution that follows a script instead of the simulation: it reports plan
    snapshots to the bridge's observers, then finishes or raises.
    """

    def __init__(
        self, snapshots: List[List[Dict[str, Any]]], failure: Exception | None = None
    ):
        self.snapshots = snapshots
        self.failure = failure

    def perform(self) -> None:
        for snapshot in self.snapshots:
            for observer in BRIDGE.plan_observers:
                observer(snapshot)
        if self.failure is not None:
            raise self.failure


def performing(monkeypatch: pytest.MonkeyPatch, execution: StubExecution) -> None:
    """
    Make the performing of any run perform the stub execution.
    """
    monkeypatch.setattr(
        PouringExecution,
        "of_run",
        classmethod(lambda cls, run, recorder=None: execution),
    )


def performing_and_keeping_recorders(
    monkeypatch: pytest.MonkeyPatch, execution: StubExecution
) -> List[Any]:
    """
    Make the performing of any run perform the stub execution.

    :return: The recorders the runs asked the execution to keep their NEEMs with, which
        fills as they are performed.
    """
    recorders: List[Any] = []

    def of_run(cls, run, recorder=None):
        recorders.append(recorder)
        return execution

    monkeypatch.setattr(PouringExecution, "of_run", classmethod(of_run))
    return recorders


def logged(directory: Path) -> List[Dict[str, Any]]:
    """
    :return: The events a run's directory holds, as written.
    """
    return [
        json.loads(line)
        for line in (directory / RunFile.EVENTS).read_text().splitlines()
    ]


def kinds(events: List[Dict[str, Any]]) -> List[str]:
    """
    :return: The kinds of the events, in order.
    """
    return [event[EventKey.EVENT] for event in events]


class TestResolvedRun:
    def test_a_run_performed_is_logged_from_start_to_finish(
        self, tmp_path, monkeypatch
    ):
        performing(monkeypatch, StubExecution(snapshots=[]))
        ResolvedRun(directory=tmp_path).run()
        events = logged(tmp_path)
        assert kinds(events) == [ExecutionStarted.__name__, ExecutionFinished.__name__]
        assert events[0]["run"] == tmp_path.name
        assert events[1]["seconds"] >= 0.0

    def test_the_steps_of_a_run_performed_are_logged_between_its_start_and_finish(
        self, tmp_path, monkeypatch
    ):
        performing(
            monkeypatch,
            StubExecution(
                snapshots=[
                    plan_at("RUNNING", NOT_STARTED, NOT_STARTED, NOT_STARTED),
                    plan_at("SUCCEEDED", "RUNNING", NOT_STARTED, NOT_STARTED),
                ]
            ),
        )
        ResolvedRun(directory=tmp_path).run()
        assert kinds(logged(tmp_path)) == [
            ExecutionStarted.__name__,
            StepChanged.__name__,
            StepChanged.__name__,
            StepChanged.__name__,
            ExecutionFinished.__name__,
        ]

    def test_a_run_that_fails_is_logged_as_failed_and_fails_as_before(
        self, tmp_path, monkeypatch
    ):
        performing(
            monkeypatch,
            StubExecution(snapshots=[], failure=RuntimeError("the arm cannot reach")),
        )
        with pytest.raises(RuntimeError, match="cannot reach"):
            ResolvedRun(directory=tmp_path).run()
        events = logged(tmp_path)
        assert kinds(events) == [ExecutionStarted.__name__, ExecutionFailed.__name__]
        assert events[-1]["error_type"] == RuntimeError.__name__
        assert events[-1]["reason"] == "the arm cannot reach"

    def test_a_run_given_a_database_has_its_execution_keep_its_neem_there(
        self, tmp_path, monkeypatch
    ):
        recorders = performing_and_keeping_recorders(
            monkeypatch, StubExecution(snapshots=[])
        )
        database = tmp_path / "neems.sqlite"
        ResolvedRun(directory=tmp_path, neem_database=database).run()
        assert recorders == [
            DatabaseNeemRecorder(
                store=SqliteNeemStore(path=database),
                kind=NeemKind.EXECUTION,
                episode=tmp_path.name,
                observer=JsonLinesSink(path=tmp_path / RunFile.EVENTS),
            )
        ]

    def test_a_run_given_no_database_has_its_execution_keep_nothing(
        self, tmp_path, monkeypatch
    ):
        recorders = performing_and_keeping_recorders(
            monkeypatch, StubExecution(snapshots=[])
        )
        ResolvedRun(directory=tmp_path).run()
        assert recorders == [NoNeemRecorder()]

    def test_the_steps_are_followed_only_while_the_run_is_performed(
        self, tmp_path, monkeypatch
    ):
        before = list(BRIDGE.plan_observers)
        performing(monkeypatch, StubExecution(snapshots=[]))
        ResolvedRun(directory=tmp_path).run()
        assert BRIDGE.plan_observers == before

    def test_the_steps_are_no_longer_followed_after_a_run_fails(
        self, tmp_path, monkeypatch
    ):
        before = list(BRIDGE.plan_observers)
        performing(monkeypatch, StubExecution(snapshots=[], failure=RuntimeError("no")))
        with pytest.raises(RuntimeError):
            ResolvedRun(directory=tmp_path).run()
        assert BRIDGE.plan_observers == before
