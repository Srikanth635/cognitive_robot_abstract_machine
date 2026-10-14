"""
Keeping what the simulation performs as NEEMs: the plan of a pour tried, or of a run
performed, saved in one database shared by every run and reported to the run's log.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from argparse import ArgumentParser, Namespace
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from time import perf_counter

from filelock import FileLock
from krrood.ormatic.data_access_objects.helper import to_dao
from krrood.ormatic.utils import create_engine
from llm_reasoner.observation.sinks import EventSink, JsonLinesSink, NullSink
from sqlalchemy.engine import Engine
from sqlalchemy.orm import sessionmaker
from typing_extensions import TYPE_CHECKING, List

from cram_viz.events import NeemKind, NeemNotRecorded, NeemRecorded

if TYPE_CHECKING:
    from coraplex.plans.plan import Plan

logger = logging.getLogger(__spec__.name)
"""
This module's logger.
"""

LOCK_SUFFIX = ".lock"
"""
The suffix of the file that keeps processes from preparing a database at once.
"""

# %% where plans are kept


@dataclass
class NeemStore(ABC):
    """
    Where the plans the simulation performed are kept.
    """

    path: Path
    """
    The database file.
    """

    @abstractmethod
    def save(self, plan: Plan) -> int:
        """
        :param plan: A plan that was performed, however it ended.
        :return: The identifier the store gave the plan.
        """


@dataclass
class SqliteNeemStore(NeemStore):
    """
    A SQLite database that holds every plan, saved through the ORM of coraplex.

    Its tables are many, and are created once, which takes about half a minute; a
    database that has them is opened at once.
    """

    def save(self, plan: Plan) -> int:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        engine = create_engine(f"sqlite:///{self.path.resolve()}")
        try:
            self.prepare(engine)
            with sessionmaker(engine)() as session:
                neem = to_dao(plan)
                session.add(neem)
                session.commit()
                return neem.database_id
        finally:
            engine.dispose()

    def prepare(self, engine: Engine) -> None:
        """
        Make sure the database has its tables; processes that find it without wait for
        the one that makes them.
        """
        from coraplex.orm.ormatic_interface import Base

        with FileLock(self.path.with_name(self.path.name + LOCK_SUFFIX)):
            Base.metadata.create_all(bind=engine)


# %% recording


class NeemRecorder(ABC):
    """
    Keeps the plan of a performance as a NEEM, if it is to be kept.
    """

    @abstractmethod
    def record(self, plan: Plan) -> None:
        """
        :param plan: The plan that was performed, however it ended.
        """


@dataclass
class NoNeemRecorder(NeemRecorder):
    """
    Keeps nothing.
    """

    def record(self, plan: Plan) -> None:
        return None


@dataclass
class DatabaseNeemRecorder(NeemRecorder):
    """
    Keeps each plan in a store and reports where, or why it could not.

    .. note:: Keeping a plan is a side effect of the performance, so a plan that cannot be
        kept is reported as an event and logged, and never hides how the performance
        went.
    """

    store: NeemStore
    """
    Where the plans are kept.
    """

    kind: NeemKind
    """
    What the plans are the episodes of.
    """

    episode: str
    """
    The trial or the run the plan is of.
    """

    observer: EventSink = field(default_factory=NullSink)
    """
    Where what was kept, or could not be, is reported to; nowhere by default.
    """

    def record(self, plan: Plan) -> None:
        began = perf_counter()
        try:
            database_id = self.store.save(plan)
        except Exception as error:
            logger.exception(
                "the NEEM of %s %s was not recorded", self.kind, self.episode
            )
            self.observer.emit(
                NeemNotRecorded(
                    kind=self.kind,
                    episode=self.episode,
                    error_type=type(error).__name__,
                    reason=str(error),
                )
            )
            return
        self.observer.emit(
            NeemRecorded(
                kind=self.kind,
                episode=self.episode,
                database=str(self.store.path),
                database_id=database_id,
                seconds=perf_counter() - began,
            )
        )


# %% telling another process where to record


class NeemOption(StrEnum):
    """
    The options of a process that tries a pour, which say where to keep its NEEM.
    """

    DATABASE = "--neem-database"
    """
    The database to keep the NEEM in.
    """

    EVENTS = "--neem-events"
    """
    The log to report the NEEM to.
    """

    EPISODE = "--neem-episode"
    """
    The trial the NEEM is of.
    """


@dataclass
class IncompleteNeemOptions(ValueError):
    """
    Some of the options that say where to keep a NEEM were given, and not all.
    """

    missing: List[str]
    """
    The options that were not given.
    """

    def __post_init__(self):
        super().__init__(
            f"the options {', '.join(self.missing)} are needed with the others that "
            "say where to keep a NEEM"
        )


@dataclass(frozen=True)
class NeemRecording:
    """
    Where the NEEMs of the pours tried in processes of their own are kept and reported.
    """

    database: Path
    """
    The database every NEEM is kept in.
    """

    events: Path
    """
    The log of the run that tries the pours, which the NEEMs are reported to.
    """

    def recorder(self, kind: NeemKind, episode: str) -> DatabaseNeemRecorder:
        """
        :return: A recorder keeping the plan of an episode in the database and reporting
            it to the log.
        """
        return DatabaseNeemRecorder(
            store=SqliteNeemStore(path=self.database),
            kind=kind,
            episode=episode,
            observer=JsonLinesSink(path=self.events),
        )

    def options(self, trial: str) -> List[str]:
        """
        :return: What a process trying a pour is given, to keep the NEEM of the trial.
        """
        return [
            NeemOption.DATABASE,
            str(self.database),
            NeemOption.EVENTS,
            str(self.events),
            NeemOption.EPISODE,
            trial,
        ]

    @staticmethod
    def add_options(parser: ArgumentParser) -> None:
        """
        Let a process read the options that say where to keep its NEEM, all of which it
        may leave out.
        """
        for option in NeemOption:
            parser.add_argument(option, default=None)

    @classmethod
    def recorder_from(cls, arguments: Namespace, kind: NeemKind) -> NeemRecorder:
        """
        :param arguments: What a process read, with the options of :meth:`add_options`.
        :param kind: What the process's NEEM is the episode of.
        :return: The recorder those options name, or one that keeps nothing when the
            process was given none.
        :raises IncompleteNeemOptions: When it was given some of them.
        """
        given = {
            NeemOption.DATABASE: arguments.neem_database,
            NeemOption.EVENTS: arguments.neem_events,
            NeemOption.EPISODE: arguments.neem_episode,
        }
        if all(value is None for value in given.values()):
            return NoNeemRecorder()
        missing = [option for option, value in given.items() if value is None]
        if missing:
            raise IncompleteNeemOptions(missing=missing)
        return cls(
            database=Path(arguments.neem_database), events=Path(arguments.neem_events)
        ).recorder(kind, arguments.neem_episode)
