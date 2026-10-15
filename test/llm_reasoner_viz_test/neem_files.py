"""
NEEM databases for the tests: SQLite files with the tables the ORM of coraplex generates
for a plan and what it refers to, holding two small plans.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

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
