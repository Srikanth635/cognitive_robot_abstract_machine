"""
NEEM databases for the tests: SQLite files with the tables the ORM of cramph generates for
a statechart, holding the statecharts of two small plans and the things they refer to.
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

WORLD_THINGS_FILE = NEEM_SQL / "world_things.sql"
"""
The things of the world the plans refer to, with the class each is.
"""

STATECHARTS = NEEM_SQL / "statecharts"
"""
Where the statecharts of the plans are kept, each with how long a tick lasted and the
life cycles of its nodes on the ticks they changed on.
"""


class KeptPlan(StrEnum):
    """
    A plan of the database made for a test, by the name of the file of its statechart, in
    the order of their identifiers.
    """

    POUR = "pour"
    """
    A pour that missed: the cup is picked up, and the pour fails.
    """

    RUN = "run"
    """
    A run performed: the arms are parked, the torso raised, and the robot drives.
    """

    @property
    def recording(self) -> Dict[str, Any]:
        """
        The statechart, how long a tick lasted and the life cycles on each tick.
        """
        return json.loads((STATECHARTS / f"{self.value}.json").read_text())


CHANGES = NEEM_SQL / "changes"
"""
Where the SQL that changes a database made for a test is kept.
"""


class Change(StrEnum):
    """
    A change to a database made for a test, by the name of the file that holds it.
    """

    PLAN_NOT_ENDED = "plan_not_ended"
    """
    The second plan was kept before its steps had all ended.
    """

    SOURCE_CONTAINER_UNSET = "source_container_unset"
    """
    A pour was not given the container it pours from.
    """

    CONTEXT_DELETED = "context_deleted"
    """
    The row of what the first plan was performed in is gone.
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
            connection.executescript(WORLD_THINGS_FILE.read_text())
            for plan in KeptPlan:
                keep_plan(connection, plan.recording)
        connection.commit()
    finally:
        connection.close()
    return path


def keep_plan(connection: sqlite3.Connection, recording: Dict[str, Any]) -> None:
    """
    Keep the statechart of a plan, with what it was performed in and its ticks, as the
    ORM does.
    """
    (link,) = [
        name
        for (name,) in connection.execute(
            "select name from sqlite_master where name like '\\_%' escape '\\'"
        )
    ]
    context = connection.execute(
        "insert into StatechartContextDAO (tick_duration, polymorphic_type) values (?, ?)",
        (recording["tick_duration"], "StatechartContextDAO"),
    ).lastrowid
    statechart = connection.execute(
        """
        insert into StatechartMappingDAO
          (pass_limit, passes_per_nesting_level, structure, polymorphic_type, _context_id)
        values (20, 2, ?, 'StatechartMappingDAO', ?)
        """,
        (json.dumps(recording["structure"]), context),
    ).lastrowid
    for states in recording["ticks"]:
        tick = connection.execute(
            """
            insert into NodeStatesAtTickDAO
              (tick_count, life_cycle_values, observation_values, polymorphic_type)
            values (?, ?, ?, 'NodeStatesAtTickDAO')
            """,
            (
                states["tick"],
                json.dumps([float(state) for state in states["life_cycles"]]),
                json.dumps([0.5 for _ in states["life_cycles"]]),
            ),
        ).lastrowid
        connection.execute(
            f"""
            insert into "{link}"
              (_source_statechartmappingdao_id, _target_nodestatesattickdao_id)
            values (?, ?)
            """,
            (statechart, tick),
        )


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
