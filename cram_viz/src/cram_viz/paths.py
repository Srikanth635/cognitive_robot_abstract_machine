"""
Filesystem locations for cram_viz, all overridable via environment.

The frontend (``web/``) ships inside the package. Scene bundles are GENERATED
artifacts (tens to hundreds of MB per scene, produced by ``cram-viz-onboard``)
and are deliberately not part of this repository — they are versioned in
https://github.com/sunava/cram-scenes, wired here as the OPTIONAL submodule
``cram_viz/scenes`` (live visualization and freshly onboarded scenes work
without it). :func:`scenes_dir` looks in this order:

    1. CRAM_VIZ_SCENES=/path/to/scenes        explicit override
    2. cram_viz/scenes                        the submodule, if initialized
                                              (git submodule update --init cram_viz/scenes)
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


#: the optional cram-scenes submodule checkout (``<member dir>/scenes``)
SCENES_SUBMODULE = WEB_ROOT.parents[2] / "scenes"


def scenes_dir() -> Path:
    """
    Directory holding the onboarded scene bundles (``<name>/scene.json``).

    Search order: ``CRAM_VIZ_SCENES`` env var, then the initialized
    cram-scenes submodule, then ``~/.cram_viz/scenes``. An un-initialized
    submodule is an empty directory and is skipped (index.json is the marker).
    """
    env = os.environ.get("CRAM_VIZ_SCENES")
    if env:
        return Path(env).expanduser()
    if (SCENES_SUBMODULE / "index.json").is_file():
        return SCENES_SUBMODULE
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
