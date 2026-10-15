"""
The database the NEEMs of the simulations are kept in, read without ever changing it:

which plans it holds and how each went.
"""

from __future__ import annotations

import re
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from typing_extensions import Any, Dict, Iterator, List, Optional

from llm_reasoner_viz.neem_schema import Column, Table

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
    the NEEMs of cram_viz are kept in, or not one of this version.
    """

    missing: str
    """
    The table or the column that is not there.
    """

    def __post_init__(self):
        super().__init__(
            f"the database has no {self.missing}, which a NEEM database has"
        )


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
    How the root of the plan ended, such as ``SUCCEEDED``.

    The steps below the root are recorded as not started even in a plan that was
    performed, so only the root says.
    """

    started: Optional[datetime]
    """
    When the plan began.
    """

    ended: Optional[datetime]
    """
    When the plan ended; none while it has not.
    """

    seconds: Optional[float]
    """
    How long the plan lasted; none while it has not ended.
    """

    nodes: int
    """
    How many nodes the plan has, the root included.
    """

    steps: int
    """
    How many actions are directly below the root.
    """

    def to_json(self) -> Dict[str, Any]:
        """
        :return: The plan as plain JSON.
        """
        return {
            "neem": self.neem,
            "status": self.status,
            "started": self.started and self.started.isoformat(),
            "ended": self.ended and self.ended.isoformat(),
            "seconds": self.seconds,
            "nodes": self.nodes,
            "steps": self.steps,
        }


# %% reading


UNMISSED_COLUMN = re.compile(r"no such column: (\S+)")
"""
How the database says that a column is not there.
"""

STATUS_SEPARATOR = "."
"""
What separates the name of a status from the module and the class that name it.
"""

BUSY_CODES = (sqlite3.SQLITE_BUSY, sqlite3.SQLITE_LOCKED)
"""
The error codes that say another connection is writing.
"""


@dataclass
class NeemDatabase:
    """
    The database cram_viz keeps the NEEMs of its simulations in, opened to read only.
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
            rows = connection.execute(f"""
                select plan.{Column.IDENTIFIER}, node.{Column.STATUS},
                       node.{Column.STARTED}, node.{Column.ENDED}, plan.{Column.ROOT}
                from "{Table.PLAN}" plan
                join "{Table.NODE}" node on node.{Column.IDENTIFIER} = plan.{Column.ROOT}
                order by plan.{Column.IDENTIFIER}
                """).fetchall()
            return [self.summary_of(connection, *row) for row in rows]

    def summary_of(
        self,
        connection: sqlite3.Connection,
        neem: int,
        status: str,
        started: Optional[str],
        ended: Optional[str],
        root: int,
    ) -> PlanSummary:
        """
        :return: The summary of a plan, from how its root ended and a count of its nodes.
        """
        begun = datetime.fromisoformat(started) if started else None
        finished = datetime.fromisoformat(ended) if ended else None
        (nodes,) = connection.execute(
            f"""
            with recursive below(node) as (
                select ?
                union all
                select edge.{Column.CHILD} from "{Table.EDGE}" edge
                join below on edge.{Column.PARENT} = below.node
            )
            select count(*) from below
            """,
            (root,),
        ).fetchone()
        (steps,) = connection.execute(
            f"""
            select count(*) from "{Table.EDGE}" edge
            join "{Table.ACTION_NODE}" action
              on action.{Column.IDENTIFIER} = edge.{Column.CHILD}
            where edge.{Column.PARENT} = ?
            """,
            (root,),
        ).fetchone()
        return PlanSummary(
            neem=neem,
            status=status.rsplit(STATUS_SEPARATOR, 1)[-1],
            started=begun,
            ended=finished,
            seconds=(finished - begun).total_seconds() if begun and finished else None,
            nodes=nodes,
            steps=steps,
        )
