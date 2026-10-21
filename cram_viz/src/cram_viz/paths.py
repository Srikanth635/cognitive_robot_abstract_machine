"""
Filesystem locations for cram_viz, all overridable via environment.

The frontend (``web/``) ships inside the package, and so do the ready-made demo scenes
in ``cram_viz/scenes`` (tens of MB each, originally onboarded with ``cram-viz-onboard``
and kept in https://github.com/sunava/cram-scenes). Freshly onboarded scenes and live
visualization work without them. :func:`scenes_dir` looks in this order:

    1. CRAM_VIZ_SCENES=/path/to/scenes        explicit override
    2. cram_viz/scenes                        the bundled scenes
    3. ~/.cram_viz/scenes                     default data directory

    CRAM_VIZ_ARCHITECTURE=/path/to/repo       CRAM repo scanned by the KB graph
"""

from __future__ import annotations

import os
from pathlib import Path

#: the packaged frontend: index.html, panels, vendored libraries
WEB_ROOT = Path(__file__).resolve().parent / "web"

#: the packaged static resources
RESOURCES_ROOT = Path(__file__).resolve().parent / "resources"

#: the notebook-exported cereal-pouring knowledge graph, kept as a reference snapshot
#: of the graph the Hypothesis Graph tab now builds live
HYPOTHESIS_KNOWLEDGE_GRAPH_PATH = RESOURCES_ROOT / "hypothesis_knowledge_graph.json"

#: the store holding the checked-in sample hypothesis
SAMPLE_HYPOTHESES_PATH = RESOURCES_ROOT / "sample_hypotheses"


def data_dir() -> Path:
    """
    Writable per-user data directory (architecture scan cache, defaults).
    """
    env = os.environ.get("CRAM_VIZ_DATA")
    if env:
        return Path(env).expanduser()
    return Path.home() / ".cram_viz"


def hypothesis_store_dir() -> Path:
    """
    Where generated hypothesis records are cached.

    Persisting these across restarts is what makes re-running an instruction free:
    the store answers instead of the model.
    """
    env = os.environ.get("CRAM_VIZ_HYPOTHESIS_STORE")
    if env:
        return Path(env).expanduser()
    return data_dir() / "hypotheses"


NEEM_DATABASE_ENVIRONMENT = "CRAM_VIZ_NEEM_DATABASE"
"""
The environment variable that names the database the NEEMs of the simulations are kept
in.
"""

NEEM_DATABASE_FILE = "neems.sqlite"
"""
The name of the database the NEEMs are kept in, in the data directory.
"""


def neem_database_path() -> Path:
    """
    :return: The one database every NEEM, of every run, is kept in: the one the
        environment names, else :data:`NEEM_DATABASE_FILE` in the data directory.
    """
    env = os.environ.get(NEEM_DATABASE_ENVIRONMENT)
    if env:
        return Path(env).expanduser()
    return data_dir() / NEEM_DATABASE_FILE


VIEWER_URL_ENVIRONMENT = "LLM_REASONER_VIZ_URL"
"""
The environment variable that sets where the llm_reasoner viewer is served.
"""

DEFAULT_VIEWER_URL = "http://localhost:8721"
"""
Where the llm_reasoner viewer is served unless told otherwise.
"""


def viewer_url(run: str) -> str:
    """
    :param run: The name of a resolution run.
    :return: The address at which the llm_reasoner viewer shows the run, the viewer
        being served at :data:`DEFAULT_VIEWER_URL` unless
        :data:`VIEWER_URL_ENVIRONMENT` says otherwise.
    """
    base = os.environ.get(VIEWER_URL_ENVIRONMENT, DEFAULT_VIEWER_URL).rstrip("/")
    return f"{base}/?run={run}"


def resolution_runs_dir() -> Path:
    """
    Where every resolution run is kept: the knowledge it consulted and generated, and
    what it resolved, apart from every other run's.
    """
    return data_dir() / "runs"


#: the demo scenes that ship with the package (``<member dir>/scenes``)
BUNDLED_SCENES = WEB_ROOT.parents[2] / "scenes"


def scenes_dir() -> Path:
    """
    Directory holding the onboarded scene bundles (``<name>/scene.json``).

    Search order: ``CRAM_VIZ_SCENES`` env var, then the bundled scenes, then
    ``~/.cram_viz/scenes``. A bundled directory without an ``index.json`` is skipped.
    """
    env = os.environ.get("CRAM_VIZ_SCENES")
    if env:
        return Path(env).expanduser()
    if (BUNDLED_SCENES / "index.json").is_file():
        return BUNDLED_SCENES
    return data_dir() / "scenes"


def architecture_root() -> Path:
    """
    The CRAM repository whose packages/classes the knowledge graph shows.

    Defaults to the repository this package is checked out in, which is the common case
    inside the workspace; falls back to the conventional clone location otherwise.
    """
    env = os.environ.get("CRAM_VIZ_ARCHITECTURE")
    if env:
        return Path(env).expanduser()
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "coraplex").is_dir() and (parent / "krrood").is_dir():
            return parent
    return Path.home() / "cognitive_robot_abstract_machine"


def default_live_demo() -> Path:
    """
    The demo the viewer's "Execute" button launches under the live bridge.

    Hardcoded to one illustrative demo until picking a demo becomes part of the UI, same
    as the hand-authored :mod:`cram_viz.hypothesis_knowledge` content it currently has
    nothing to do with.
    """
    return (
        architecture_root()
        / "coraplex"
        / "demos"
        / "coraplex_gcs_demo"
        / "cram_viz"
        / "kitchen_counter_pouring.py"
    )
