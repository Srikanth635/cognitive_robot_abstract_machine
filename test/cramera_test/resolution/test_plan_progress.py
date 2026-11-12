"""
Tests of following a performed plan as the live viewer's bridge snapshots it: its steps,
and every node of it, reported to the run's log when their status changes.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest
from cramph.data_types import LifeCycleValues
from llm_reasoner.observation.events import EventKey
from llm_reasoner.observation.sinks import MemorySink
from typing_extensions import Any

from cramera.live.bridge import Bridge, PlanNodeEntry, PlanNodeGroup, PlanSnapshot
from cramera.resolution import live
from cramera.resolution.events import (
    ExecutionFailed,
    ExecutionFinished,
    ExecutionStarted,
    PlanNodeChanged,
    StepChanged,
)
from cramera.resolution.live import ResolvedRun
from cramera.resolution.plan_progress import (
    PlanNodeReporter,
    PlanObservation,
    PlanStepReporter,
)
from cramera.resolution.resolver.execution import PouringExecution

from .test_run_events import logged, of_kind

RUN = "a-run"
"""
The run the plans are performed in.
"""

# %% helpers


def entry(
    identifier: str, parent: str | None, label: str, status: LifeCycleValues
) -> PlanNodeEntry:
    """
    :return: A plan node as the bridge snapshots it.
    """
    return PlanNodeEntry(
        id=identifier,
        parent=parent,
        kind="ActionNode",
        group=PlanNodeGroup.of_plan_node_kind("ActionNode"),
        label=label,
        status=status,
        derived=False,
    )


def plan_at(*statuses: LifeCycleValues) -> PlanSnapshot:
    """
    :return: A plan of a root and two steps, each with a motion beneath it, the steps
        and the motions in the given statuses in turn.
    """
    first, second, motion_of_first, motion_of_second = statuses
    return PlanSnapshot(
        nodes=[
            entry("root", None, "SequentialNode", LifeCycleValues.RUNNING),
            entry("first", "root", "ParkArmsAction", first),
            entry("motion-first", "first", "ParkArmsMotion", motion_of_first),
            entry("second", "root", "NavigateAction", second),
            entry("motion-second", "second", "NavigateMotion", motion_of_second),
        ]
    )


NOT_STARTED = LifeCycleValues.NOT_STARTED
RUNNING = LifeCycleValues.RUNNING
SUCCEEDED = LifeCycleValues.SUCCEEDED


# %% the steps


class TestPlanStepReporter:
    def test_a_step_that_starts_is_reported(self):
        sink = MemorySink()
        PlanStepReporter(sink=sink, run=RUN).observe(
            plan_at(RUNNING, NOT_STARTED, RUNNING, NOT_STARTED)
        )
        assert [(event.run, event.step, event.status) for event in sink.events] == [
            (RUN, "ParkArmsAction", RUNNING.name)
        ]
        assert all(isinstance(event, StepChanged) for event in sink.events)

    def test_steps_not_started_are_not_reported(self):
        sink = MemorySink()
        PlanStepReporter(sink=sink, run=RUN).observe(
            plan_at(NOT_STARTED, NOT_STARTED, NOT_STARTED, NOT_STARTED)
        )
        assert sink.events == []

    def test_a_status_seen_again_is_reported_once(self):
        sink = MemorySink()
        reporter = PlanStepReporter(sink=sink, run=RUN)
        for _ in range(3):
            reporter.observe(plan_at(RUNNING, NOT_STARTED, RUNNING, NOT_STARTED))
        assert len(sink.events) == 1

    def test_the_changes_are_reported_in_the_order_they_happen(self):
        sink = MemorySink()
        reporter = PlanStepReporter(sink=sink, run=RUN)
        reporter.observe(plan_at(RUNNING, NOT_STARTED, RUNNING, NOT_STARTED))
        reporter.observe(plan_at(SUCCEEDED, RUNNING, SUCCEEDED, RUNNING))
        assert [(event.step, event.status) for event in sink.events] == [
            ("ParkArmsAction", RUNNING.name),
            ("ParkArmsAction", SUCCEEDED.name),
            ("NavigateAction", RUNNING.name),
        ]

    def test_the_nodes_below_the_steps_are_not_steps(self):
        sink = MemorySink()
        reporter = PlanStepReporter(sink=sink, run=RUN)
        reporter.observe(plan_at(RUNNING, NOT_STARTED, NOT_STARTED, NOT_STARTED))
        reporter.observe(plan_at(RUNNING, NOT_STARTED, SUCCEEDED, NOT_STARTED))
        assert [event.step for event in sink.events] == ["ParkArmsAction"]


class TestStepsOfNativeLifecycles:
    """
    Coraplex leaves an action's own status at not started while the motions below it
    run, so a step is followed by the nodes below it.
    """

    def test_a_step_runs_once_a_node_below_it_runs(self):
        sink = MemorySink()
        PlanStepReporter(sink=sink, run=RUN).observe(
            plan_at(NOT_STARTED, NOT_STARTED, RUNNING, NOT_STARTED)
        )
        assert [(event.step, event.status) for event in sink.events] == [
            ("ParkArmsAction", RUNNING.name)
        ]

    def test_a_step_whose_motions_ended_still_runs_until_the_plan_moves_on(self):
        sink = MemorySink()
        reporter = PlanStepReporter(sink=sink, run=RUN)
        reporter.observe(plan_at(NOT_STARTED, NOT_STARTED, RUNNING, NOT_STARTED))
        reporter.observe(plan_at(NOT_STARTED, NOT_STARTED, SUCCEEDED, RUNNING))
        assert [(event.step, event.status) for event in sink.events] == [
            ("ParkArmsAction", RUNNING.name),
            ("NavigateAction", RUNNING.name),
        ]

    def test_a_step_fails_when_a_node_below_it_fails(self):
        sink = MemorySink()
        PlanStepReporter(sink=sink, run=RUN).observe(
            plan_at(NOT_STARTED, NOT_STARTED, LifeCycleValues.FAILED, NOT_STARTED)
        )
        assert [event.status for event in sink.events] == [LifeCycleValues.FAILED.name]

    def test_a_node_deeper_below_a_step_moves_it_too(self):
        sink = MemorySink()
        snapshot = PlanSnapshot(
            nodes=[
                entry("root", None, "SequentialNode", RUNNING),
                entry("pick-up", "root", "PickUpAction", NOT_STARTED),
                entry("sequence", "pick-up", "SequentialNode", NOT_STARTED),
                entry("motion", "sequence", "MoveGripperMotion", RUNNING),
            ]
        )
        PlanStepReporter(sink=sink, run=RUN).observe(snapshot)
        assert [(event.step, event.status) for event in sink.events] == [
            ("PickUpAction", RUNNING.name)
        ]


# %% every node


class TestPlanNodeReporter:
    def test_every_node_is_reported_when_first_seen_in_the_order_of_the_snapshot(self):
        sink = MemorySink()
        snapshot = plan_at(NOT_STARTED, NOT_STARTED, NOT_STARTED, NOT_STARTED)
        PlanNodeReporter(sink=sink, run=RUN).observe(snapshot)
        assert [event.node for event in sink.events] == [
            node.id for node in snapshot.nodes
        ]

    def test_a_node_is_reported_with_what_places_it_in_the_plan(self):
        sink = MemorySink()
        snapshot = plan_at(RUNNING, NOT_STARTED, NOT_STARTED, NOT_STARTED)
        PlanNodeReporter(sink=sink, run=RUN).observe(snapshot)
        assert [
            (event.node, event.parent, event.kind, event.label, event.status)
            for event in sink.events
        ] == [
            (node.id, node.parent, node.kind, node.label, node.status.name)
            for node in snapshot.nodes
        ]

    def test_a_node_whose_status_does_not_change_is_reported_once(self):
        sink = MemorySink()
        reporter = PlanNodeReporter(sink=sink, run=RUN)
        snapshot = plan_at(RUNNING, NOT_STARTED, RUNNING, NOT_STARTED)
        reporter.observe(snapshot)
        reporter.observe(snapshot)
        assert len(sink.events) == len(snapshot.nodes)

    def test_a_motion_is_reported_through_every_status_it_takes(self):
        sink = MemorySink()
        reporter = PlanNodeReporter(sink=sink, run=RUN)
        for motion in (NOT_STARTED, RUNNING, SUCCEEDED):
            reporter.observe(plan_at(RUNNING, NOT_STARTED, motion, NOT_STARTED))
        assert [
            event.status for event in sink.events if event.node == "motion-first"
        ] == [NOT_STARTED.name, RUNNING.name, SUCCEEDED.name]

    def test_a_node_that_appears_later_is_reported_then(self):
        sink = MemorySink()
        reporter = PlanNodeReporter(sink=sink, run=RUN)
        reporter.observe(PlanSnapshot(nodes=[entry("root", None, "Root", RUNNING)]))
        reporter.observe(plan_at(RUNNING, NOT_STARTED, NOT_STARTED, NOT_STARTED))
        assert [event.node for event in sink.events][1:] == [
            "first",
            "motion-first",
            "second",
            "motion-second",
        ]

    def test_every_event_is_a_change_of_a_plan_node(self):
        sink = MemorySink()
        PlanNodeReporter(sink=sink, run=RUN).observe(
            plan_at(RUNNING, NOT_STARTED, NOT_STARTED, NOT_STARTED)
        )
        assert all(isinstance(event, PlanNodeChanged) for event in sink.events)


# %% following the bridge


class TestPlanObservation:
    def test_the_observer_follows_the_bridge_while_the_plan_is_followed(self):
        bridge = Bridge()
        observed = []
        with PlanObservation(bridge, observed.append):
            assert bridge.plan_observers == [observed.append]
        assert bridge.plan_observers == []

    def test_the_observer_stops_following_when_the_plan_fails(self):
        bridge = Bridge()
        observed = []
        with pytest.raises(PlanFailedForTheTest):
            with PlanObservation(bridge, observed.append):
                raise PlanFailedForTheTest()
        assert bridge.plan_observers == []


# %% performing a run


class PlanFailedForTheTest(RuntimeError):
    """
    How a scripted performance fails.
    """


@dataclass
class ViewerFollowingTheBridge:
    """
    A live viewer that only holds the bridge the performance's plan is snapshot by.
    """

    bridge: Bridge
    """
    The bridge.
    """

    def plan_callback(self, plan: Any) -> None:
        """
        No callback: the scripted performance snapshots the plan itself.
        """


@dataclass
class ScriptedExecution:
    """
    A performance that follows a script instead of the simulation: it hands plan
    snapshots to the bridge's observers, then finishes or fails.
    """

    bridge: Bridge
    """
    The bridge whose observers are handed the snapshots.
    """

    snapshots: list[PlanSnapshot]
    """
    The snapshots, in turn.
    """

    fails: bool = False
    """
    Whether the performance fails after the snapshots.
    """

    world: Any = None
    """
    The world the viewer is started on.
    """

    visualization: Any = None
    """
    The viewer the performance is shown in.
    """

    def perform(self) -> None:
        """
        Hand every snapshot to the observers, then finish or fail.
        """
        for snapshot in self.snapshots:
            for observer in list(self.bridge.plan_observers):
                observer(snapshot)
        if self.fails:
            raise PlanFailedForTheTest()


def performing(monkeypatch: pytest.MonkeyPatch, execution: ScriptedExecution) -> None:
    """
    Make the performing of any run perform the scripted execution under a viewer holding
    its bridge.
    """
    monkeypatch.setattr(
        PouringExecution,
        "of_run",
        classmethod(lambda cls, run, recorder=None: execution),
    )
    monkeypatch.setattr(
        live, "start", lambda world: ViewerFollowingTheBridge(bridge=execution.bridge)
    )


def kinds(events: list[dict[str, Any]]) -> list[str]:
    """
    :return: The kinds of the events, in order.
    """
    return [event[EventKey.EVENT] for event in events]


STEPS_OF_A_PERFORMANCE = [
    plan_at(RUNNING, NOT_STARTED, RUNNING, NOT_STARTED),
    plan_at(SUCCEEDED, RUNNING, SUCCEEDED, RUNNING),
    plan_at(SUCCEEDED, SUCCEEDED, SUCCEEDED, SUCCEEDED),
]
"""
The snapshots of a plan of two steps performed one after the other.
"""


class TestResolvedRun:
    def test_the_steps_are_logged_between_the_start_and_the_end(
        self, tmp_path, monkeypatch
    ):
        performing(monkeypatch, ScriptedExecution(Bridge(), STEPS_OF_A_PERFORMANCE))
        ResolvedRun(directory=tmp_path).run()
        events = logged(tmp_path)
        assert kinds(events)[0] == ExecutionStarted.__name__
        assert kinds(events)[-1] == ExecutionFinished.__name__
        assert [
            (event["step"], event["status"]) for event in of_kind(events, StepChanged)
        ] == [
            ("ParkArmsAction", RUNNING.name),
            ("ParkArmsAction", SUCCEEDED.name),
            ("NavigateAction", RUNNING.name),
            ("NavigateAction", SUCCEEDED.name),
        ]

    def test_every_node_of_the_plan_is_logged(self, tmp_path, monkeypatch):
        performing(monkeypatch, ScriptedExecution(Bridge(), STEPS_OF_A_PERFORMANCE))
        ResolvedRun(directory=tmp_path).run()
        logged_nodes = {
            event["node"] for event in of_kind(logged(tmp_path), PlanNodeChanged)
        }
        assert logged_nodes == {node.id for node in STEPS_OF_A_PERFORMANCE[0].nodes}

    def test_a_failed_performance_is_logged_as_failed(self, tmp_path, monkeypatch):
        performing(
            monkeypatch,
            ScriptedExecution(Bridge(), STEPS_OF_A_PERFORMANCE[:1], fails=True),
        )
        with pytest.raises(PlanFailedForTheTest):
            ResolvedRun(directory=tmp_path).run()
        assert kinds(logged(tmp_path))[-1] == ExecutionFailed.__name__

    def test_the_bridge_is_no_longer_followed_once_the_performance_ends(
        self, tmp_path, monkeypatch
    ):
        bridge = Bridge()
        performing(monkeypatch, ScriptedExecution(bridge, STEPS_OF_A_PERFORMANCE))
        ResolvedRun(directory=tmp_path).run()
        assert bridge.plan_observers == []
