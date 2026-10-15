"""
The NEEMs with where each came from: the plans of the database, each joined to the run
whose log says it was kept, and to how that trial or execution went.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from typing_extensions import Any, Dict, List, Mapping, Optional, Tuple

from llm_reasoner_viz.neems import NeemDatabase, PlanSummary
from llm_reasoner_viz.runs import EventLog, LogCache, RunLibrary
from llm_reasoner_viz.vocabulary import EventField, KnownEvent, NeemKind

EXECUTION_RESULTS = {
    KnownEvent.EXECUTION_FINISHED: "finished",
    KnownEvent.EXECUTION_FAILED: "failed",
}
"""
How an execution ended, by the event that says so.
"""

# %% where a NEEM came from


@dataclass(frozen=True)
class NeemOrigin:
    """
    The trial or the execution a NEEM is the plan of, as the log of its run says.
    """

    kind: NeemKind
    """
    Whether it is the plan of a trial or of an execution.
    """

    episode: str
    """
    The trial, or the run that was performed.
    """

    run: str
    """
    The run whose log says the NEEM was kept.
    """

    recorded_at: float
    """
    When it was kept, in seconds since the epoch.
    """

    outcome: Optional[Mapping[str, Any]]
    """
    What the trial brought about; none for an execution, or while the trial has not
    ended.
    """

    result: Optional[str]
    """
    How the execution ended, ``finished`` or ``failed``; none for a trial, or while the
    execution has not ended.
    """

    def to_json(self) -> Dict[str, Any]:
        """
        :return: The origin as plain JSON.
        """
        return {
            "kind": self.kind,
            "episode": self.episode,
            "run": self.run,
            "recorded_at": self.recorded_at,
            "outcome": self.outcome,
            "result": self.result,
        }


@dataclass(frozen=True)
class KeptNeem:
    """
    A NEEM a log says was kept, and in which database.
    """

    database: Path
    """
    The database it was kept in, as the log names it.
    """

    database_id: int
    """
    The identifier it was given there.
    """

    origin: NeemOrigin
    """
    Where it came from.
    """


@dataclass(frozen=True)
class NeemEntry:
    """
    A plan of the database, and where it came from when a log says.
    """

    plan: PlanSummary
    """
    The plan and how it went.
    """

    origin: Optional[NeemOrigin]
    """
    Where it came from; none when no log names it, as for a plan kept by a script or by
    a run whose log is gone.
    """

    def to_json(self) -> Dict[str, Any]:
        """
        :return: The entry as plain JSON.
        """
        return {
            **self.plan.to_json(),
            "origin": self.origin and self.origin.to_json(),
        }


# %% reading the logs


def neems_kept_by(log: Path) -> List[KeptNeem]:
    """
    :param log: The log of a run.
    :return: The NEEMs the log says were kept, each with how its trial or execution went.
    """
    events = [entry.event for entry in EventLog(path=log).read_from(0).entries]
    outcomes = {
        event[EventField.TRIAL]: event[EventField.OUTCOME]
        for event in events
        if event[EventField.EVENT] == KnownEvent.TRIAL_FINISHED
    }
    results = {
        event[EventField.RUN]: EXECUTION_RESULTS[event[EventField.EVENT]]
        for event in events
        if event[EventField.EVENT] in EXECUTION_RESULTS
    }
    return [
        KeptNeem(
            database=Path(event[EventField.DATABASE]),
            database_id=event[EventField.DATABASE_ID],
            origin=NeemOrigin(
                kind=NeemKind(event[EventField.KIND]),
                episode=event[EventField.EPISODE],
                run=log.parent.name,
                recorded_at=event[EventField.AT],
                outcome=(
                    outcomes.get(event[EventField.EPISODE])
                    if event[EventField.KIND] == NeemKind.TRIAL
                    else None
                ),
                result=(
                    results.get(event[EventField.EPISODE])
                    if event[EventField.KIND] == NeemKind.EXECUTION
                    else None
                ),
            ),
        )
        for event in events
        if event[EventField.EVENT] == KnownEvent.NEEM_RECORDED
    ]


# %% the library


@dataclass
class NeemLibrary:
    """
    The NEEMs of a database, joined to the logs of the runs that kept them.
    """

    database: NeemDatabase
    """
    Where the plans are.
    """

    runs: RunLibrary
    """
    The runs, whose logs say where each NEEM came from.
    """

    kept: LogCache[List[KeptNeem]] = None
    """
    The NEEMs each log says were kept, as the log was when it last changed.
    """

    def __post_init__(self) -> None:
        self.kept = LogCache(read=neems_kept_by)

    def origins(self) -> Dict[int, NeemOrigin]:
        """
        :return: Where each plan of the database came from, by identifier, for those a log
            names: the log that kept it last when several do, and no log that names another
            database.
        """
        here = self.database.path.resolve()
        origins: Dict[int, NeemOrigin] = {}
        for log in self.runs.logs():
            for neem in self.kept.of(log):
                if neem.database.resolve() != here:
                    continue
                known = origins.get(neem.database_id)
                if known is None or neem.origin.recorded_at >= known.recorded_at:
                    origins[neem.database_id] = neem.origin
        return origins

    def entries(self, run: Optional[str] = None) -> List[NeemEntry]:
        """
        :param run: The run whose NEEMs are wanted; all when none is named.
        :return: The plans of the database with where each came from, the newest first.
        """
        origins = self.origins()
        entries = [
            NeemEntry(plan=plan, origin=origins.get(plan.neem))
            for plan in self.database.plans()
        ]
        if run is not None:
            entries = [
                entry for entry in entries if entry.origin and entry.origin.run == run
            ]
        return sorted(entries, key=lambda entry: entry.plan.neem, reverse=True)
