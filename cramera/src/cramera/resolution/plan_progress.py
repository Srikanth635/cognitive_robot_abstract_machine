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
            status = node.status.name
            if self.statuses.get(node.id, self.unseen_status) == status:
                continue
            self.statuses[node.id] = status
            self.sink.emit(self.change_of(node))

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
    def change_of(self, node: PlanNodeEntry) -> Event:
        """
        :param node: A node whose status changed.
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

    def change_of(self, node: PlanNodeEntry) -> StepChanged:
        return StepChanged(run=self.run, step=node.label, status=node.status.name)


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

    def change_of(self, node: PlanNodeEntry) -> PlanNodeChanged:
        return PlanNodeChanged(
            run=self.run,
            node=node.id,
            parent=node.parent,
            kind=node.kind,
            label=node.label,
            status=node.status.name,
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
