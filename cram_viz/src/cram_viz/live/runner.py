"""
Starting the live bridge: as a library call, or as the command running a demo script or
a resolution run under it.
"""

from __future__ import annotations

import argparse
import logging
import runpy
import signal
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass
from http.server import ThreadingHTTPServer
from pathlib import Path

from typing_extensions import List, Optional, TYPE_CHECKING

from cram_viz import configure_console_logging, paths
from cram_viz.events import NeemKind, reporting_execution
from cram_viz.live import hooks
from cram_viz.live.bridge import BRIDGE
from cram_viz.live.http import DEFAULT_PORT, serve
from cram_viz.live.launcher import LiveArgument
from cram_viz.live.steps import PlanStepReporter, following_plan
from cram_viz.neem import NeemRecorder, NeemRecording, NoNeemRecorder
from cram_viz.resolution.execution import PouringExecution
from cram_viz.resolution.runs import RunDirectory

if TYPE_CHECKING:
    from semantic_digital_twin.world import World

logger = logging.getLogger(__name__)


def start(
    world: Optional[World] = None, port: int = DEFAULT_PORT
) -> ThreadingHTTPServer:
    """
    Start the live bridge.

    Call once, ideally at the top of a demo.
    :param world: Bind to this world immediately; without it the bridge attaches to the
        executing world on the first executor tick.
    :param port: Port of the bridge's HTTP endpoints.
    :return: The running HTTP server (a daemon thread).
    """
    hooks.install_mesh_hook()  # before the demo parses its objects
    hooks.install_plan_hooks()
    hooks.install_root_node_hook()
    if world is not None:
        BRIDGE.world = world
        BRIDGE._bind()
        BRIDGE.snapshot()  # single-threaded here, before execution starts
    hooks.install_tick_hook()
    server = serve(port)
    logger.info(
        "bridge on http://localhost:%d (the viewer shows a Live button "
        "while this runs)",
        port,
    )
    return server


# %% what the bridge runs


class LiveTarget(ABC):
    """
    What the live bridge runs.
    """

    @abstractmethod
    def run(self) -> None:
        """
        Run under the live bridge, which attaches to the world on its first executor
        tick.
        """

    @staticmethod
    def from_arguments(arguments: List[str]) -> LiveTarget:
        """
        :param arguments: The command line after the program: a demo script, or
            :attr:`LiveArgument.RUN` and a run's directory.
        :return: What the arguments name.
        """
        parser = argparse.ArgumentParser(
            prog="cram-viz-live",
            description="Run a demo script or a resolution run under the live bridge.",
        )
        parser.add_argument("demo", nargs="?", type=Path, help="a demo script")
        parser.add_argument(
            LiveArgument.RUN, type=Path, help="the directory of a resolution run"
        )
        parsed = parser.parse_args(arguments)
        if parsed.run is not None:
            return ResolvedRun(
                directory=parsed.run, neem_database=paths.neem_database_path()
            )
        if parsed.demo is None:
            parser.error("name a demo script or a resolution run")
        return DemoScript(path=parsed.demo)


@dataclass(frozen=True)
class DemoScript(LiveTarget):
    """
    A demo script, run as if started directly.
    """

    path: Path
    """
    The script.
    """

    def run(self) -> None:
        """
        Run the script with its own directory on ``sys.path``, so it can import its
        local helper modules.
        """
        demo = self.path.resolve()
        sys.path.insert(0, str(demo.parent))
        logger.info("running demo: %s", demo)
        runpy.run_path(str(demo), run_name="__main__")


@dataclass(frozen=True)
class ResolvedRun(LiveTarget):
    """
    A resolution run, its resolved action performed from the environment it started
    from.
    """

    directory: Path
    """
    The run's directory.
    """

    neem_database: Optional[Path] = None
    """
    The database the NEEM of the performance is kept in; none is kept without one.
    """

    def run(self) -> None:
        """
        Perform the run's resolved action in the simulation, keep its plan as a NEEM
        when there is a database, and report it to the run's log.
        """
        logger.info("performing run: %s", self.directory)
        run = RunDirectory(directory=self.directory)
        sink = run.events
        reporter = PlanStepReporter(sink=sink, run=run.name)
        with reporting_execution(sink, run.name), following_plan(
            BRIDGE, reporter.observe
        ):
            PouringExecution.of_run(run, self.recorder(sink.path, run.name)).perform()

    def recorder(self, events: Path, run: str) -> NeemRecorder:
        """
        :param events: The run's log.
        :param run: The name of the run.
        :return: What keeps the NEEM of the performance, and reports it to the log.
        """
        if self.neem_database is None:
            return NoNeemRecorder()
        return NeemRecording(database=self.neem_database, events=events).recorder(
            NeemKind.EXECUTION, run
        )


def main() -> None:
    """
    ``cram-viz-live path/to/demo.py`` or ``cram-viz-live --run path/to/run`` — run a
    demo script or perform a resolution run with the live bridge.

    After it finishes the bridge stays up for inspection until Ctrl-C.
    """
    configure_console_logging()
    target = LiveTarget.from_arguments(sys.argv[1:])
    start()
    target.run()
    logger.info("finished — bridge stays up for inspection (Ctrl-C to quit)")
    try:
        signal.pause()  # wait for Ctrl-C (SIGINT) instead of a sleep loop
    except KeyboardInterrupt:
        pass
