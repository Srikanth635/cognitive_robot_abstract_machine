"""
Performing a resolution run in the simulation under cramera's live viewer, started from
the viewer's Execute button as a process of its own.
"""

from __future__ import annotations

import argparse
import logging
import signal
import subprocess
import sys
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path

from cramera.live.runner import start
from typing_extensions import Any, Callable, Sequence

from cramera_resolution import configure_console_logging, paths
from cramera_resolution.events import NeemKind, reporting_execution
from cramera_resolution.neem import NeemRecorder, NeemRecording, NoNeemRecorder
from cramera_resolution.resolution.execution import PouringExecution
from cramera_resolution.resolution.runs import RunDirectory

logger = logging.getLogger(__name__)


class LiveModule(StrEnum):
    """
    The modules run as processes.
    """

    PERFORM_RUN = "cramera_resolution.live"
    """
    Performs a resolution run.
    """


class LiveArgument(StrEnum):
    """
    The command-line arguments of the process performing a run.
    """

    RUN = "--run"
    """
    Precedes the directory of the resolution run to perform.
    """


# %% performing a run


@dataclass(frozen=True)
class ResolvedRun:
    """
    A resolution run, its resolved action performed from the environment it started from
    while cramera's viewer follows the world and the plan.
    """

    directory: Path
    """
    The run's directory.
    """

    neem_database: Path | None = None
    """
    The database the NEEM of the performance is kept in; none is kept without one.
    """

    def run(self) -> None:
        """
        Perform the run's resolved action in the simulation, serving the world and the
        plan to cramera's viewer, keep the plan as a NEEM when there is a database, and
        report the performance to the run's log.
        """
        logger.info("performing run: %s", self.directory)
        run = RunDirectory(directory=self.directory)
        sink = run.events
        with reporting_execution(sink, run.name):
            execution = PouringExecution.of_run(run, self.recorder(sink.path, run.name))
            viewer = start(execution.world)
            execution.perform(
                observe_plan=lambda root: root.plan.node_callbacks.append(
                    viewer.plan_callback(root.plan)
                )
            )

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

    @classmethod
    def from_arguments(cls, arguments: Sequence[str]) -> ResolvedRun:
        """
        :param arguments: The command line after the program.
        :return: The run the arguments name.
        """
        parser = argparse.ArgumentParser(
            prog="cramera-resolution-live",
            description="Perform a resolution run in cramera's live viewer.",
        )
        parser.add_argument(LiveArgument.RUN, type=Path, required=True)
        parsed = parser.parse_args(list(arguments))
        return cls(directory=parsed.run, neem_database=paths.neem_database_path())


def main() -> None:
    """
    ``python -m cramera_resolution.live --run path/to/run``: perform a resolution run
    with cramera's live viewer.

    After it finishes the viewer's bridge stays up for inspection until Ctrl-C.
    """
    configure_console_logging()
    ResolvedRun.from_arguments(sys.argv[1:]).run()
    logger.info("finished — the viewer stays up for inspection (Ctrl-C to quit)")
    try:
        signal.pause()
    except KeyboardInterrupt:
        pass


# %% starting it from the viewer


@dataclass
class LiveRunLauncher:
    """
    Starts the performance of a resolution run as a background process.

    One instance is reused across requests, so a later launch can stop the process it
    started before.
    """

    spawn: Callable[..., Any] = subprocess.Popen
    """
    The process constructor; a stub in tests, so nothing starts a real process.
    """

    termination_timeout_seconds: float = 5.0
    """
    How long the previous process is given to exit after being asked to, before it is
    killed, so restarting never hangs waiting on it.
    """

    _process: Any | None = field(default=None, init=False, repr=False)
    """
    The process started last.
    """

    def launch(self, run: RunDirectory) -> None:
        """
        Perform a resolution run in the background, from the environment it started
        from, restarting whatever this launcher started before; the process's output goes
        to a log file in the data directory.

        :param run: The run to perform.
        """
        self._stop_running_process()
        log_directory = paths.data_dir()
        log_directory.mkdir(parents=True, exist_ok=True)
        arguments: list[str] = [LiveArgument.RUN, str(run.directory)]
        with (log_directory / "live_run.log").open("w") as log_file:
            self._process = self.spawn(
                [sys.executable, "-m", LiveModule.PERFORM_RUN, *arguments],
                stdout=log_file,
                stderr=subprocess.STDOUT,
            )

    def _stop_running_process(self) -> None:
        """
        Terminate the process started before, killing it if it hangs.
        """
        if self._process is None or self._process.poll() is not None:
            return
        self._process.terminate()
        try:
            self._process.wait(timeout=self.termination_timeout_seconds)
        except subprocess.TimeoutExpired:
            self._process.kill()
            self._process.wait(timeout=self.termination_timeout_seconds)


if __name__ == "__main__":
    main()
