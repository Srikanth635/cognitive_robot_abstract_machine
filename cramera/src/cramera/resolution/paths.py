"""
Filesystem locations of the resolution feature, overridable through the environment. What it keeps
lives in cramera's own data directory.
"""

from __future__ import annotations

import os
from enum import StrEnum
from pathlib import Path

from cramera import paths as cramera_paths
from krrood.patterns.path_enumeration import PathEnumeration

# %% what the feature ships and what it names


class ResolutionDirectory(PathEnumeration):
    """
    The directories packaged with the resolution feature.
    """

    RESOURCES = Path(__file__).resolve().parent / "resources"
    """
    The resources packaged with the resolution feature.
    """

    SAMPLE_HYPOTHESES = RESOURCES / "sample_hypotheses"
    """
    The store holding the checked-in sample hypothesis.
    """


class EnvironmentVariable(StrEnum):
    """
    The environment variables that move what the add-on keeps or links to.
    """

    HYPOTHESIS_STORE = "CRAMERA_HYPOTHESIS_STORE"
    """
    Where generated hypothesis records are cached.
    """

    NEEM_DATABASE = "CRAMERA_NEEM_DATABASE"
    """
    The database the NEEMs of the simulations are kept in.
    """

    VIEWER_URL = "LLM_REASONER_VIZ_URL"
    """
    Where the llm_reasoner viewer is served.
    """


class DataFileName(StrEnum):
    """
    The names of the files the add-on keeps in its data directory.
    """

    NEEM_DATABASE = "neems.sqlite"
    """
    The database the NEEMs are kept in.
    """


class ViewerAddress(StrEnum):
    """
    Where the llm_reasoner viewer is served.
    """

    DEFAULT = "http://localhost:8721"
    """
    The address it is served at unless the environment says otherwise.
    """


# %% what the add-on keeps


def data_dir() -> Path:
    """
    :return: The writable data directory: cramera's own.
    """
    return cramera_paths.data_directory()


def hypothesis_store_dir() -> Path:
    """
    Where generated hypothesis records are cached, so that running an instruction again
    is answered from the store instead of the model.
    """
    named = os.environ.get(EnvironmentVariable.HYPOTHESIS_STORE)
    if named:
        return Path(named).expanduser()
    return data_dir() / "hypotheses"


def neem_database_path() -> Path:
    """
    :return: The one database every NEEM, of every run, is kept in: the one the
        environment names, else the default file in the data directory.
    """
    named = os.environ.get(EnvironmentVariable.NEEM_DATABASE)
    if named:
        return Path(named).expanduser()
    return data_dir() / DataFileName.NEEM_DATABASE


def resolution_runs_dir() -> Path:
    """
    Where every resolution run is kept: the knowledge it consulted and generated, and
    what it resolved, apart from every other run's.
    """
    return data_dir() / "runs"


def viewer_url(run: str) -> str:
    """
    :param run: The name of a resolution run.
    :return: The address at which the llm_reasoner viewer shows the run.
    """
    base = os.environ.get(EnvironmentVariable.VIEWER_URL, ViewerAddress.DEFAULT)
    return f"{base.rstrip('/')}/?run={run}"
