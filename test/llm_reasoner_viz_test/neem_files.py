"""
NEEM databases for the tests: SQLite files with the tables the ORM of coraplex generates
for a plan and what it refers to, holding two small plans.
"""

from __future__ import annotations

import json
import sqlite3
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from typing_extensions import Any, Dict, Iterator, List

from llm_reasoner_viz.neem_library import NeemLibrary
from llm_reasoner_viz.neems import NeemDatabase
from llm_reasoner_viz.runs import RunLibrary

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


# %% what the page is tested on


@dataclass(frozen=True)
class PageFixture:
    """
    A NEEM as the server sends it, kept as a file for the tests of the page.
    """

    file: str
    """
    The name of the file in the fixtures of the page's tests.
    """

    neem: int
    """
    The plan of the database that is sent.
    """

    changes: List[Change]
    """
    The changes made to the database first.
    """


PAGE_FIXTURES = [
    PageFixture(file="neem_pour.json", neem=1, changes=[]),
    PageFixture(file="neem_navigation.json", neem=2, changes=[]),
    PageFixture(
        file="neem_pick_up.json",
        neem=1,
        changes=[Change.PICK_UP_ACTION, Change.RE_ATTACH_STEP],
    ),
]
"""
The NEEMs the page's tests are run on.
"""

PAGE_FIXTURES_DIRECTORY = Path(__file__).parent / "js" / "fixtures"
"""
Where the fixtures of the page's tests are kept.
"""


def sent_by_the_server(fixture: PageFixture) -> Dict[str, Any]:
    """
    :return: What the server sends for the NEEM of a fixture, from a database made for it.
    """
    with tempfile.TemporaryDirectory() as directory:
        database = write_neem_database(Path(directory) / "neems.sqlite")
        for change in fixture.changes:
            change_database(database, change)
        library = NeemLibrary(
            database=NeemDatabase(path=database),
            runs=RunLibrary(directory=Path(directory) / "runs"),
        )
        return library.detail(fixture.neem).to_json()


def write_page_fixtures() -> None:
    """
    Write the fixtures of the page's tests again, after the server's answer has changed.
    """
    for fixture in PAGE_FIXTURES:
        (PAGE_FIXTURES_DIRECTORY / fixture.file).write_text(
            json.dumps(sent_by_the_server(fixture), indent=2) + "\n"
        )
