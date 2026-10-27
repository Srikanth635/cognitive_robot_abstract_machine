"""
Reporting how a performed plan progresses to the run's log, as the live viewer's bridge
snapshots it: each step of the plan, and every node of it, when its status changes.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from contextlib import AbstractContextManager
from dataclasses import dataclass, field

from giskardpy.motion_statechart.data_types import LifeCycleValues
from llm_reasoner.observation.events import Event
from llm_reasoner.observation.sinks import EventSink
from typing_extensions import Any, Callable, Optional, Sequence

from cramera.live.bridge import Bridge, PlanNodeEntry, PlanSnapshot
from cramera.resolution.events import PlanNodeChanged, StepChanged

# %% reporting changes


@dataclass
class PlanProgressReporter(ABC):
    """
    Reports the nodes of a plan it follows to a run's log, each time a node's status
    changes from the one reported before.
    """

    sink: EventSink
    """
    Where the changes are reported.
    """

    run: str
    """
    The name of the run whose plan is performed.
    """

    statuses: dict[str, str] = field(default_factory=dict)
    """
    The status reported last of each node, by its identifier.
    """

    def observe(self, snapshot: PlanSnapshot) -> None:
        """
        Report every followed node whose status differs from the one reported before.

        :param snapshot: The plan as the bridge snapshot it.
        """
        for node in self.followed(snapshot.nodes):
            status = self.status_of(node, snapshot.nodes)
            if self.statuses.get(node.id, self.unseen_status) == status:
                continue
            self.statuses[node.id] = status
            self.sink.emit(self.change_of(node, status))

    def status_of(self, node: PlanNodeEntry, nodes: Sequence[PlanNodeEntry]) -> str:
        """
        :param node: A followed node.
        :param nodes: Every node of the plan.
        :return: The status the node is reported in: its own.
        """
        return node.status.name

    @property
    @abstractmethod
    def unseen_status(self) -> Optional[str]:
        """
        The status a node not reported yet is taken to have; a node first seen in it is
        not reported.
        """

    @abstractmethod
    def followed(self, nodes: Sequence[PlanNodeEntry]) -> list[PlanNodeEntry]:
        """
        :param nodes: Every node of the plan, each after the node above it.
        :return: The nodes whose changes are reported.
        """

    @abstractmethod
    def change_of(self, node: PlanNodeEntry, status: str) -> Event:
        """
        :param node: A node whose status changed.
        :param status: The status it changed to.
        :return: The event reporting the change.
        """


@dataclass
class PlanStepReporter(PlanProgressReporter):
    """
    Reports the steps of a plan, the nodes right below its root, once they start.
    """

    @property
    def unseen_status(self) -> Optional[str]:
        return LifeCycleValues.NOT_STARTED.name

    def followed(self, nodes: Sequence[PlanNodeEntry]) -> list[PlanNodeEntry]:
        roots = {node.id for node in nodes if node.parent is None}
        return [node for node in nodes if node.parent in roots]

    def status_of(self, node: PlanNodeEntry, nodes: Sequence[PlanNodeEntry]) -> str:
        """
        :meth:`PlanProgressReporter.status_of`, except that a step whose own status is
        still not started, as coraplex leaves an action's while its motions run, is
        failed or interrupted when a node below it is, and running once one has started;
        it ends when the step after it starts.
        """
        if node.status is not LifeCycleValues.NOT_STARTED:
            return node.status.name
        below = {entry.status for entry in self.descendants_of(node, nodes)}
        for ending in (LifeCycleValues.FAILED, LifeCycleValues.INTERRUPTED):
            if ending in below:
                return ending.name
        if below - {LifeCycleValues.NOT_STARTED}:
            return LifeCycleValues.RUNNING.name
        return LifeCycleValues.NOT_STARTED.name

    @staticmethod
    def descendants_of(
        node: PlanNodeEntry, nodes: Sequence[PlanNodeEntry]
    ) -> list[PlanNodeEntry]:
        """
        :param node: A node of the plan.
        :param nodes: Every node of the plan.
        :return: The nodes below the node, at any depth.
        """
        below: list[PlanNodeEntry] = []
        parents = {node.id}
        for entry in nodes:
            if entry.parent in parents:
                below.append(entry)
                parents.add(entry.id)
        return below

    def change_of(self, node: PlanNodeEntry, status: str) -> StepChanged:
        return StepChanged(run=self.run, step=node.label, status=status)


@dataclass
class PlanNodeReporter(PlanProgressReporter):
    """
    Reports every node of a plan, when first seen and whenever its status changes.
    """

    @property
    def unseen_status(self) -> Optional[str]:
        return None

    def followed(self, nodes: Sequence[PlanNodeEntry]) -> list[PlanNodeEntry]:
        return list(nodes)

    def change_of(self, node: PlanNodeEntry, status: str) -> PlanNodeChanged:
        return PlanNodeChanged(
            run=self.run,
            node=node.id,
            parent=node.parent,
            kind=node.kind,
            label=node.label,
            status=status,
            derived=node.derived,
        )


# %% following the bridge


@dataclass
class PlanObservation(AbstractContextManager):
    """
    An observer handed every plan snapshot the bridge takes while a block runs, however
    the block ends.
    """

    bridge: Bridge
    """
    The bridge snapshotting the plan.
    """

    observer: Callable[[PlanSnapshot], None]
    """
    What is handed each snapshot.
    """

    def __enter__(self) -> PlanObservation:
        self.bridge.plan_observers.append(self.observer)
        return self

    def __exit__(self, *raised: Any) -> None:
        self.bridge.plan_observers.remove(self.observer)
