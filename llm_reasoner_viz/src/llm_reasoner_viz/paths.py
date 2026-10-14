"""
Where the runs the viewer shows are kept.
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
