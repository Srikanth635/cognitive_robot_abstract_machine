"""
The database the NEEMs of the simulations are kept in, read without ever changing it:

which plans it holds and how each went.
"""

from __future__ import annotations

import re
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from typing_extensions import Any, Dict, Iterator, List, Optional

from llm_reasoner_viz.neem_plan import PlanNode, PlanReader, StoredStatechart
from llm_reasoner_viz.neem_schema import Table

# %% what can go wrong


class NeemDatabaseBusy(RuntimeError):
    """
    The database is being written, and cannot be read just now.
    """

    def __init__(self):
        super().__init__("the NEEM database is being written; try again in a moment")


@dataclass
class UnknownNeemSchema(LookupError):
    """
    The database does not have a table or a column the viewer reads, so it is not one
    the NEEMs of cramera are kept in, or not one of this version.
    """

    missing: str
    """
    The table or the column that is not there.
    """

    def __post_init__(self):
        super().__init__(
            f"the database has no {self.missing}, which a NEEM database has"
        )


@dataclass
class UnknownNeem(LookupError):
    """
    The database holds no plan of the identifier asked for.
    """

    neem: int
    """
    The identifier asked for.
    """

    def __post_init__(self):
        super().__init__(f"the NEEM database holds no plan {self.neem}")


# %% the plans


@dataclass(frozen=True)
class PlanSummary:
    """
    A plan the simulation performed, and how it went.
    """

    neem: int
    """
    The identifier the database gave the plan.
    """

    status: str
    """
    How the node the plan was performed from ended, such as ``SUCCEEDED``, or where it
    was when the NEEM was kept.
    """

    seconds: Optional[float]
    """
    How long the plan lasted in the simulation; none while it has not ended.
    """

    nodes: int
    """
    How many nodes the statechart of the plan has, those its executor added included.
    """

    steps: int
    """
    How many actions are directly below the node the plan was performed from.
    """

    def to_json(self) -> Dict[str, Any]:
        """
        :return: The plan as plain JSON.
        """
        return {
            "neem": self.neem,
            "status": self.status,
            "seconds": self.seconds,
            "nodes": self.nodes,
            "steps": self.steps,
        }

    @classmethod
    def from_statechart(cls, *, neem: int, statechart: StoredStatechart) -> PlanSummary:
        """
        :return: The summary of a plan, from how the node it was performed from went.
        """
        root = statechart.plan_root
        started = statechart.started(index=root)
        ended = statechart.ended(index=root)
        return cls(
            neem=neem,
            status=statechart.final_state(index=root).name,
            seconds=ended - started
            if started is not None and ended is not None
            else None,
            nodes=len(statechart.nodes),
            steps=sum(
                statechart.is_action(index=child)
                for child in statechart.children.get(root, [])
            ),
        )


# %% reading


UNMISSED_COLUMN = re.compile(r"no such column: (\S+)")
"""
How the database says that a column is not there.
"""

BUSY_CODES = (sqlite3.SQLITE_BUSY, sqlite3.SQLITE_LOCKED)
"""
The error codes that say another connection is writing.
"""


@dataclass
class NeemDatabase:
    """
    The database cramera keeps the NEEMs of its simulations in, opened to read only.
    """

    path: Path
    """
    The database file.
    """

    timeout_seconds: float = 2.0
    """
    How long a read waits for a writer to finish before it reports the database as busy.
    """

    @property
    def exists(self) -> bool:
        """
        Whether there is a database, which there is not until the first NEEM is kept.
        """
        return self.path.is_file()

    @contextmanager
    def connection(self) -> Iterator[sqlite3.Connection]:
        """
        A connection that cannot change the database, closed afterwards.

        :raises NeemDatabaseBusy: When another process is writing.
        :raises UnknownNeemSchema: When the database lacks a table or a column read.
        """
        connection = sqlite3.connect(
            f"{self.path.resolve().as_uri()}?mode=ro",
            uri=True,
            timeout=self.timeout_seconds,
        )
        try:
            self.require_tables(connection)
            yield connection
        except sqlite3.OperationalError as error:
            if error.sqlite_errorcode in BUSY_CODES:
                raise NeemDatabaseBusy() from error
            missing = UNMISSED_COLUMN.search(str(error))
            if missing:
                raise UnknownNeemSchema(missing.group(1)) from error
            raise
        finally:
            connection.close()

    @staticmethod
    def require_tables(connection: sqlite3.Connection) -> None:
        """
        :raises UnknownNeemSchema: When a table the viewer reads is not there.
        """
        present = {
            name
            for (name,) in connection.execute(
                "select name from sqlite_master where type = 'table'"
            )
        }
        for table in Table:
            if table not in present:
                raise UnknownNeemSchema(table)

    def plans(self) -> List[PlanSummary]:
        """
        :return: Every plan the database holds, in the order of their identifiers; none
            while there is no database.
        """
        if not self.exists:
            return []
        with self.connection() as connection:
            reader = PlanReader(connection=connection)
            return [
                PlanSummary.from_statechart(
                    neem=neem, statechart=reader.statechart(neem=neem)
                )
                for neem in reader.neems()
            ]

    def nodes(self, neem: int) -> List[PlanNode]:
        """
        :return: The steps of a plan, each followed by those below it.
        :raises UnknownNeem: When the database holds no such plan.
        """
        if not self.exists:
            raise UnknownNeem(neem)
        with self.connection() as connection:
            reader = PlanReader(connection=connection)
            statechart = reader.statechart(neem=neem)
            if statechart is None:
                raise UnknownNeem(neem)
            return reader.nodes(statechart=statechart)
