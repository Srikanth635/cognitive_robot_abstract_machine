"""
Where the runs the viewer shows are kept, and where cramera, which it links to, is.
"""

from __future__ import annotations

import os
from enum import StrEnum
from pathlib import Path

# %% what the environment names


class EnvironmentVariable(StrEnum):
    """
    The environment variables that move what the viewer shows or links to.
    """

    RUNS = "LLM_REASONER_VIZ_RUNS"
    """
    The directory of runs to show.
    """

    CRAMERA_DATA = "CRAMERA_DATA"
    """
    cramera's data directory, in which it keeps its runs and its NEEM database.
    """

    CRAMERA_URL = "CRAMERA_URL"
    """
    The address cramera is served at.
    """

    NEEM_DATABASE = "CRAMERA_NEEM_DATABASE"
    """
    The database cramera keeps the NEEMs of its simulations in.
    """


class CrameraDataName(StrEnum):
    """
    The names of what cramera keeps in its data directory, and of that directory in the
    user's home.
    """

    USER_DATA_DIRECTORY = ".cramera"
    """
    The directory of the user's home cramera keeps its data in unless told otherwise.
    """

    RUNS = "runs"
    """
    The directory holding a run in a directory of its own.
    """

    NEEM_DATABASE = "neems.sqlite"
    """
    The database the NEEMs are kept in.
    """


class CrameraAddress(StrEnum):
    """
    Where cramera is served.
    """

    DEFAULT = "http://localhost:8711"
    """
    The address it is served at unless the environment says otherwise.
    """


# %% where things are


def cramera_data_directory() -> Path:
    """
    :return: cramera's data directory: the one the environment names, else the one in
        the user's home.
    """
    data = os.environ.get(EnvironmentVariable.CRAMERA_DATA)
    if data:
        return Path(data).expanduser()
    return Path.home() / CrameraDataName.USER_DATA_DIRECTORY


def default_runs_directory() -> Path:
    """
    :return: The directory of runs to show: the one the environment names, else where
        cramera keeps its runs.
    """
    runs = os.environ.get(EnvironmentVariable.RUNS)
    if runs:
        return Path(runs).expanduser()
    return cramera_data_directory() / CrameraDataName.RUNS


def cramera_url() -> str:
    """
    :return: The address cramera is served at: the one the environment names, else the
        default, without a closing slash.
    """
    return os.environ.get(
        EnvironmentVariable.CRAMERA_URL, CrameraAddress.DEFAULT
    ).rstrip("/")


def neem_database_path() -> Path:
    """
    :return: The database the NEEMs are kept in: the one the environment names, else
        where cramera keeps it in its data directory.
    """
    database = os.environ.get(EnvironmentVariable.NEEM_DATABASE)
    if database:
        return Path(database).expanduser()
    return cramera_data_directory() / CrameraDataName.NEEM_DATABASE
