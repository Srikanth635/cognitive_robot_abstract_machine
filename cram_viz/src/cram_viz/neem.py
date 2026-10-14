"""
Keeping what the simulation performs as NEEMs: the plan of a pour tried, or of a run
performed, saved in one database shared by every run and reported to the run's log.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter

from filelock import FileLock
from krrood.ormatic.data_access_objects.helper import to_dao
from krrood.ormatic.utils import create_engine
from llm_reasoner.observation.sinks import EventSink, NullSink
from sqlalchemy.engine import Engine
from sqlalchemy.orm import sessionmaker
from typing_extensions import TYPE_CHECKING

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
