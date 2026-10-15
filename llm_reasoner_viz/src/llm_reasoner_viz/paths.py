"""
Where the runs the viewer shows are kept, and where cram_viz, which it links to, is.
"""

from __future__ import annotations

import os
from pathlib import Path

RUNS_ENVIRONMENT = "LLM_REASONER_VIZ_RUNS"
"""
The environment variable that names the directory of runs to show.
"""

DATA_ENVIRONMENT = "CRAM_VIZ_DATA"
"""
The environment variable that names cram_viz's data directory, in which it keeps runs.
"""

CRAM_VIZ_ENVIRONMENT = "CRAM_VIZ_URL"
"""
The environment variable that names the address cram_viz is served at.
"""

DEFAULT_CRAM_VIZ_URL = "http://localhost:8711"
"""
The address cram_viz is served at unless told otherwise.
"""

NEEM_DATABASE_ENVIRONMENT = "CRAM_VIZ_NEEM_DATABASE"
"""
The environment variable that names the database cram_viz keeps the NEEMs of its
simulations in.
"""

NEEM_DATABASE_FILE = "neems.sqlite"
"""
The name of the database the NEEMs are kept in, in cram_viz's data directory.
"""

RUNS_DIRECTORY = "runs"
"""
The directory of cram_viz's data directory that holds a run in a directory of its own.
"""

USER_DATA_DIRECTORY = ".cram_viz"
"""
The directory of the user's home that cram_viz keeps its data in unless told otherwise.
"""


def default_runs_directory() -> Path:
    """
    :return: The directory of runs to show: the one the environment names, else where
        cram_viz keeps its runs.
    """
    runs = os.environ.get(RUNS_ENVIRONMENT)
    if runs:
        return Path(runs).expanduser()
    data = os.environ.get(DATA_ENVIRONMENT)
    if data:
        return Path(data).expanduser() / RUNS_DIRECTORY
    return Path.home() / USER_DATA_DIRECTORY / RUNS_DIRECTORY


def cram_viz_url() -> str:
    """
    :return: The address cram_viz is served at: the one the environment names, else the
        default, without a closing slash.
    """
    return os.environ.get(CRAM_VIZ_ENVIRONMENT, DEFAULT_CRAM_VIZ_URL).rstrip("/")


def neem_database_path() -> Path:
    """
    :return: The database the NEEMs are kept in: the one the environment names, else
        where cram_viz keeps it in its data directory.
    """
    database = os.environ.get(NEEM_DATABASE_ENVIRONMENT)
    if database:
        return Path(database).expanduser()
    data = os.environ.get(DATA_ENVIRONMENT)
    base = Path(data).expanduser() if data else Path.home() / USER_DATA_DIRECTORY
    return base / NEEM_DATABASE_FILE
