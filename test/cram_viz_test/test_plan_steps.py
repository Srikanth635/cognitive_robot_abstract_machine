"""
Tests of following a plan as the bridge sees it: its steps, and every node of it,
reported when their status changes, and what Execute reports of a resolution run being
performed.
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
    PlanNodeChanged,
    StepChanged,
)
from cram_viz.live.bridge import BRIDGE, Bridge, PlanNodeKey
from cram_viz.live.runner import ResolvedRun
from cram_viz.live.steps import NOT_STARTED, PlanNodeReporter, PlanStepReporter
from cram_viz.neem import DatabaseNeemRecorder, NoNeemRecorder, SqliteNeemStore
from cram_viz.resolution.execution import PouringExecution
from cram_viz.resolution.runs import RunFile

# %% helpers


def node(
    identifier: str,
    parent: str | None,
    label: str,
    status: str,
    derived: bool = False,
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
        PlanNodeKey.DERIVED: derived,
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

    def test_a_node_of_a_snapshot_has_every_key_the_reporters_read(self):
        bridge = Bridge()
        bridge._plan = type("StubPlan", (), {"root": _stub_root()})()
        bridge.snapshot_plan()
        (root,) = bridge.get_plan()["nodes"]
        assert set(PlanNodeKey) <= set(root)


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


# %% following every node


class TestPlanNodeReporter:
    def test_every_node_is_reported_when_first_seen_whatever_its_status(self):
        sink = MemorySink()
        reporter = PlanNodeReporter(sink=sink, run="a-run")
        snapshot = plan_at(NOT_STARTED, NOT_STARTED, NOT_STARTED, NOT_STARTED)
        reporter.observe(snapshot)
        assert all(type(event) is PlanNodeChanged for event in sink.events)
        assert [event.node for event in sink.events] == [
            node[PlanNodeKey.ID] for node in snapshot
        ]

    def test_a_node_is_reported_with_what_says_where_it_is_in_the_plan(self):
        sink = MemorySink()
        reporter = PlanNodeReporter(sink=sink, run="a-run")
        reporter.observe(plan_at("RUNNING", NOT_STARTED, NOT_STARTED, NOT_STARTED))
        root, first, *_ = sink.events
        assert (root.run, root.parent, root.kind, root.label, root.status) == (
            "a-run",
            None,
            "ActionNode",
            "SequentialNode",
            "RUNNING",
        )
        assert (first.node, first.parent, first.label, first.status) == (
            "first",
            "root",
            "ParkArmsAction",
            "RUNNING",
        )

    def test_a_status_is_reported_once_however_often_it_is_seen(self):
        sink = MemorySink()
        reporter = PlanNodeReporter(sink=sink, run="a-run")
        for _ in range(3):
            reporter.observe(plan_at("RUNNING", NOT_STARTED, NOT_STARTED, NOT_STARTED))
        assert len(sink.events) == 5

    def test_a_node_below_a_step_is_reported_when_its_status_changes(self):
        sink = MemorySink()
        reporter = PlanNodeReporter(sink=sink, run="a-run")
        reporter.observe(plan_at("RUNNING", NOT_STARTED, NOT_STARTED, NOT_STARTED))
        reporter.observe(plan_at("RUNNING", NOT_STARTED, "RUNNING", NOT_STARTED))
        reporter.observe(plan_at("RUNNING", NOT_STARTED, "SUCCEEDED", NOT_STARTED))
        changes = [
            (event.node, event.status)
            for event in sink.events
            if event.node == "motion-first"
        ]
        assert changes == [
            ("motion-first", NOT_STARTED),
            ("motion-first", "RUNNING"),
            ("motion-first", "SUCCEEDED"),
        ]

    def test_a_node_that_appears_later_is_reported_then_and_the_others_are_not_again(
        self,
    ):
        sink = MemorySink()
        reporter = PlanNodeReporter(sink=sink, run="a-run")
        grown = plan_at("RUNNING", NOT_STARTED, NOT_STARTED, NOT_STARTED)
        reporter.observe(grown[:2])
        reporter.observe(grown)
        assert [event.node for event in sink.events] == [
            "root",
            "first",
            "motion-first",
            "second",
            "motion-second",
        ]

    def test_a_status_the_bridge_derived_says_so(self):
        sink = MemorySink()
        reporter = PlanNodeReporter(sink=sink, run="a-run")
        reporter.observe(
            [
                node("root", None, "SequentialNode", "RUNNING", derived=True),
                node("step", "root", "ParkArmsAction", "CREATED"),
            ]
        )
        assert [event.derived for event in sink.events] == [True, False]

    def test_a_node_is_written_as_plain_json_with_the_root_having_no_parent(self):
        sink = MemorySink()
        reporter = PlanNodeReporter(sink=sink, run="a-run")
        reporter.observe([node("root", None, "SequentialNode", "RUNNING")])
        written = sink.events[0].to_json()
        assert written[EventKey.EVENT] == PlanNodeChanged.__name__
        assert written["parent"] is None
        assert json.loads(json.dumps(written)) == written


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
        assert [
            kind for kind in kinds(logged(tmp_path)) if kind != PlanNodeChanged.__name__
        ] == [
            ExecutionStarted.__name__,
            StepChanged.__name__,
            StepChanged.__name__,
            StepChanged.__name__,
            ExecutionFinished.__name__,
        ]

    def test_every_node_of_a_run_performed_is_logged_as_it_changes_between_its_start_and_finish(
        self, tmp_path, monkeypatch
    ):
        performing(
            monkeypatch,
            StubExecution(
                snapshots=[
                    plan_at("RUNNING", NOT_STARTED, "RUNNING", NOT_STARTED),
                    plan_at("SUCCEEDED", "RUNNING", "SUCCEEDED", NOT_STARTED),
                ]
            ),
        )
        ResolvedRun(directory=tmp_path).run()
        events = logged(tmp_path)
        assert kinds(events)[0] == ExecutionStarted.__name__
        assert kinds(events)[-1] == ExecutionFinished.__name__
        changes = [
            (event["node"], event["status"])
            for event in events
            if event[EventKey.EVENT] == PlanNodeChanged.__name__
        ]
        assert [change for change in changes if change[0] == "motion-first"] == [
            ("motion-first", "RUNNING"),
            ("motion-first", "SUCCEEDED"),
        ]
        assert {node for node, _ in changes} == {
            "root",
            "first",
            "motion-first",
            "second",
            "motion-second",
        }

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
