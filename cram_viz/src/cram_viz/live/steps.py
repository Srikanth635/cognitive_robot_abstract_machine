"""
Following the plan a resolution run performs: its steps, the actions it is made of, and
every node of it, reported as their status changes, from the plan snapshots the bridge
takes anyway.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field

from llm_reasoner.observation.sinks import EventSink
from typing_extensions import Any, Callable, Dict, Iterator, List

from cram_viz.events import PlanNodeChanged, StepChanged
from cram_viz.live.bridge import Bridge, PlanNodeKey

NOT_STARTED = "CREATED"
"""
The status of a step of the plan before it is performed.
"""


@dataclass
class PlanStepReporter:
    """
    Reports each step of a plan, the actions directly below its root, when its status
    changes.
    """

    sink: EventSink
    """
    Where the steps are reported.
    """

    run: str
    """
    The name of the run whose plan is followed.
    """

    statuses: Dict[str, str] = field(default_factory=dict)
    """
    The status last seen of each step, by its identifier.
    """

    def observe(self, nodes: List[Dict[str, Any]]) -> None:
        """
        Report the steps of a snapshot of the plan whose status changed since the last.

        :param nodes: The nodes of the plan as the bridge snapshots them.
        """
        roots = {
            node[PlanNodeKey.ID] for node in nodes if node[PlanNodeKey.PARENT] is None
        }
        for node in nodes:
            if node[PlanNodeKey.PARENT] not in roots:
                continue
            status = node[PlanNodeKey.STATUS]
            if status == self.statuses.get(node[PlanNodeKey.ID], NOT_STARTED):
                continue
            self.statuses[node[PlanNodeKey.ID]] = status
            self.sink.emit(
                StepChanged(run=self.run, step=node[PlanNodeKey.LABEL], status=status)
            )


@dataclass
class PlanNodeReporter:
    """
    Reports every node of a plan when it first appears in a snapshot, and again when its
    status changes.
    """

    sink: EventSink
    """
    Where the nodes are reported.
    """

    run: str
    """
    The name of the run whose plan is followed.
    """

    statuses: Dict[str, str] = field(default_factory=dict)
    """
    The status last seen of each node, by its identifier; a node not in it has not been
    seen.
    """

    def observe(self, nodes: List[Dict[str, Any]]) -> None:
        """
        Report the nodes of a snapshot of the plan that are new or whose status changed
        since the last, each below the one it hangs from.

        :param nodes: The nodes of the plan as the bridge snapshots them, every node
            after the node above it.
        """
        for node in nodes:
            identifier = node[PlanNodeKey.ID]
            status = node[PlanNodeKey.STATUS]
            if self.statuses.get(identifier) == status:
                continue
            self.statuses[identifier] = status
            self.sink.emit(
                PlanNodeChanged(
                    run=self.run,
                    node=identifier,
                    parent=node[PlanNodeKey.PARENT],
                    kind=node[PlanNodeKey.KIND],
                    label=node[PlanNodeKey.LABEL],
                    status=status,
                    derived=node[PlanNodeKey.DERIVED],
                )
            )


@contextmanager
def following_plan(
    bridge: Bridge, observer: Callable[[List[Dict[str, Any]]], None]
) -> Iterator[None]:
    """
    Have the bridge tell an observer the plan's nodes while a plan is performed.

    :param bridge: The bridge snapshotting the plan.
    :param observer: Told the nodes of every snapshot.
    """
    bridge.plan_observers.append(observer)
    try:
        yield
    finally:
        bridge.plan_observers.remove(observer)
