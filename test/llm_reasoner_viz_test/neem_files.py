"""
NEEM databases for the tests: SQLite files with the tables the ORM of coraplex generates
for a plan and what it refers to, holding two small plans.
"""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from enum import StrEnum
from pathlib import Path

from typing_extensions import Iterator

NEEM_SQL = Path(__file__).parent / "neems"
"""
Where the SQL that makes a NEEM database for a test is kept.
"""

SCHEMA_FILE = NEEM_SQL / "schema.sql"
"""
The tables, as the ORM generates them.
"""

PLANS_FILE = NEEM_SQL / "plans.sql"
"""
Two plans: the first of a pour that missed, the second of a run performed.
"""

CHANGES = NEEM_SQL / "changes"
"""
Where the SQL that changes a database made for a test is kept.
"""


class Change(StrEnum):
    """
    A change to a database made for a test, by the name of the file that holds it.
    """

    END_TIME_CLEARED = "end_time_cleared"
    """
    The root of the first plan has not ended.
    """

    SOURCE_CONTAINER_UNSET = "source_container_unset"
    """
    A pour was not given the container it pours from.
    """

    DESIGNATOR_DELETED = "designator_deleted"
    """
    The row of what a step was asked to do is gone.
    """

    DESIGNATOR_WITH_BYTES = "designator_with_bytes"
    """
    The pour of the first plan becomes an action whose table has a column of bytes.
    """

    PICK_UP_ACTION = "pick_up_action"
    """
    The pour of the first plan becomes a pick-up, which refers to how the object is
    grasped.
    """

    RE_ATTACH_STEP = "re_attach_step"
    """
    A step is added to the first plan that moves a body to another parent.
    """

    EXECUTION_RECORDED = "execution_recorded"
    """
    What the execution of the pour recorded is kept with its step.
    """

    EXCLUSIVE_LOCK = "exclusive_lock"
    """
    The database is held for writing.
    """

    UNRELATED_TABLE = "unrelated_table"
    """
    A table of something that is no NEEM.
    """

    @property
    def script(self) -> str:
        """
        The SQL of the change.
        """
        return (CHANGES / f"{self.value}.sql").read_text()


def change_database(path: Path, change: Change) -> None:
    """
    Apply a change to a database.
    """
    connection = sqlite3.connect(path)
    try:
        connection.executescript(change.script)
        connection.commit()
    finally:
        connection.close()


@contextmanager
def being_written(path: Path) -> Iterator[None]:
    """
    Hold the database for writing for as long as the block lasts, as a simulation
    recording a NEEM does.
    """
    connection = sqlite3.connect(path, isolation_level=None)
    try:
        connection.executescript(Change.EXCLUSIVE_LOCK.script)
        yield
    finally:
        connection.close()


def write_neem_database(path: Path, plans: bool = True) -> Path:
    """
    :param path: Where to make the database.
    :param plans: Whether to fill it with the two plans, or leave its tables empty.
    :return: The database made.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path)
    try:
        connection.executescript(SCHEMA_FILE.read_text())
        if plans:
            connection.executescript(PLANS_FILE.read_text())
        connection.commit()
    finally:
        connection.close()
    return path
