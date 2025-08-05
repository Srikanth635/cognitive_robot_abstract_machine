from __future__ import annotations

import os
from contextlib import suppress
from copy import deepcopy
from functools import lru_cache, wraps
from typing import Any, Tuple, Iterable
from xml.etree import ElementTree as ET

from sqlalchemy import Engine, inspect, text, MetaData


class IDGenerator:
    """
    A class that generates incrementing, unique IDs and caches them for every object this is called on.
    """

    _counter = 0
    """
    The counter of the unique IDs.
    """

    @lru_cache(maxsize=None)
    def __call__(self, obj: Any) -> int:
        """
        Creates a unique ID and caches it for every object this is called on.

        :param obj: The object to generate a unique ID for, must be hashable.
        :return: The unique ID.
        """
        self._counter += 1
        return self._counter


class suppress_stdout_stderr(object):
    """
    A context manager for doing a "deep suppression" of stdout and stderr in
    Python, i.e. will suppress all prints, even if the print originates in a
    compiled C/Fortran sub-function.

    This will not suppress raised exceptions, since exceptions are printed
    to stderr just before a script exits, and after the context manager has
    exited (at least, I think that is why it lets exceptions through).
    Copied from https://stackoverflow.com/questions/11130156/suppress-stdout-stderr-print-from-python-functions
    """

    def __init__(self):
        # Open a pair of null files
        self.null_fds = [os.open(os.devnull, os.O_RDWR) for _ in range(2)]
        # Save the actual stdout (1) and stderr (2) file descriptors.
        self.save_fds = [os.dup(1), os.dup(2)]

    def __enter__(self):
        # Assign the null pointers to stdout and stderr.
        # This one is not needed for URDF parsing output
        # os.dup2(self.null_fds[0], 1)
        os.dup2(self.null_fds[1], 2)

    def __exit__(self, *_):
        # Re-assign the real stdout/stderr back to (1) and (2)
        # This one is not needed for URDF parsing output
        # os.dup2(self.save_fds[0], 1)
        os.dup2(self.save_fds[1], 2)
        # Close all file descriptors
        for fd in self.null_fds + self.save_fds:
            os.close(fd)


def hacky_urdf_parser_fix(urdf: str, blacklist: Tuple[str] = ('transmission', 'gazebo')) -> str:
    # Parse input string
    root = ET.fromstring(urdf)

    # Iterate through each section in the blacklist
    for section_name in blacklist:
        # Find all sections with the given name and remove them
        for elem in root.findall(f".//{section_name}"):
            parent = root.find(f".//{section_name}/..")
            if parent is not None:
                parent.remove(elem)

    # Turn back to string
    return ET.tostring(root, encoding='unicode')


def robot_name_from_urdf_string(urdf_string: str) -> str:
    """
    Returns the name defined in the robot tag, e.g., 'pr2' from <robot name="pr2"> ... </robot>.
    :param urdf_string: URDF string
    :return: Extracted name
    """
    return urdf_string.split('robot name="')[1].split('"')[0]


def copy_lru_cache(maxsize=None, typed=False):
    def decorator(func):
        cached_func = lru_cache(maxsize=maxsize, typed=typed)(func)

        @wraps(func)
        def wrapper(*args, **kwargs):
            result = cached_func(*args, **kwargs)
            return deepcopy(result)

        # Preserve lru_cache methods
        wrapper.cache_info = cached_func.cache_info
        wrapper.cache_clear = cached_func.cache_clear

        return wrapper

    return decorator


def _drop_fk_constraints(engine: Engine, tables: Iterable[str]) -> None:
    """
    Drop *named* foreign-key constraints for all *tables*.

    MySQL / MariaDB uses ``DROP FOREIGN KEY`` whereas most other back-ends
    accept the ANSI form ``DROP CONSTRAINT``.  Each statement is executed
    inside a ``suppress(Exception)`` so that back-ends that do not support the
    specific syntax simply continue.
    """
    insp = inspect(engine)
    dialect = engine.dialect.name.lower()

    with engine.begin() as conn:
        for table in tables:
            for fk in insp.get_foreign_keys(table):
                name = fk.get("name")
                if not name:                     # unnamed FKs (e.g. SQLite)
                    continue

                if dialect.startswith("mysql"):
                    stmt = text(f"ALTER TABLE `{table}` DROP FOREIGN KEY `{name}`")
                else:  # PostgreSQL, SQLite, MSSQL, …
                    stmt = text(f'ALTER TABLE "{table}" DROP CONSTRAINT "{name}"')

                with suppress(Exception):
                    conn.execute(stmt)


def drop_database(engine: Engine) -> None:
    """
    Remove **all** tables that are currently present in *engine*’s default
    schema, taking care of back-references / foreign keys first.

    Works with SQLite, PostgreSQL, MySQL / MariaDB, and most other SQLAlchemy
    back-ends that support the standard reflection API.
    """
    metadata = MetaData()
    metadata.reflect(bind=engine)

    if not metadata.tables:
        return

    # 1. Drop FK constraints that would otherwise block table deletion.
    _drop_fk_constraints(engine, metadata.tables.keys())

    # 2. On MySQL / MariaDB it is still safest to disable FK checks entirely
    #    while the DROP TABLE statements run; other back-ends don’t need this.
    disable_fk_checks = engine.dialect.name.lower().startswith("mysql")

    with engine.begin() as conn:
        if disable_fk_checks:
            conn.execute(text("SET FOREIGN_KEY_CHECKS = 0"))

        # Drop in reverse dependency order (children first → parents last).
        for table in reversed(metadata.sorted_tables):
            table.drop(bind=conn, checkfirst=True)

        if disable_fk_checks:
            conn.execute(text("SET FOREIGN_KEY_CHECKS = 1"))

