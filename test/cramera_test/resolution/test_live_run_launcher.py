"""
Tests of starting the performance of a resolution run from the viewer's Execute button
as a process of its own.

Spawning is replaced throughout, so nothing here starts a real process or a simulation.
"""

from __future__ import annotations

import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import pytest
from typing_extensions import Any

from cramera.resolution.exceptions import NoPerformanceWaiting
from cramera.resolution.live import (
    LiveArgument,
    LiveModule,
    LiveRunLauncher,
    ResolvedRun,
)
from cramera.resolution.resolver.runs import RunDirectory, RunFile

# %% helpers


@dataclass
class ProcessReportingItsState:
    """
    A process that reports running or exited, and records how it was stopped.
    """

    exit_code: int | None = None
    """
    What the process reports from ``poll``; ``None`` while it runs.
    """

    ignores_termination: bool = False
    """
    Whether it keeps running after it is asked to terminate.
    """

    terminated: bool = False
    """
    Whether it was asked to terminate.
    """

    killed: bool = False
    """
    Whether it was killed.
    """

    def poll(self) -> int | None:
        return self.exit_code

    def terminate(self) -> None:
        self.terminated = True
        if not self.ignores_termination:
            self.exit_code = -15

    def kill(self) -> None:
        self.killed = True
        self.exit_code = -9

    def wait(self, timeout: float | None = None) -> None:
        if self.exit_code is None:
            raise subprocess.TimeoutExpired(cmd="run", timeout=timeout)


@dataclass
class SpawnHandingOutProcesses:
    """
    A replacement for the process constructor that records every command and hands out
    the processes it was given, in order.
    """

    processes: list[ProcessReportingItsState]
    """
    What it answers, one per call.
    """

    commands: list[list[str]] = field(default_factory=list)
    """
    The commands it was asked to run.
    """

    def __call__(self, command: list[str], **keywords: Any) -> ProcessReportingItsState:
        self.commands.append(command)
        return self.processes[len(self.commands) - 1]


@pytest.fixture
def a_run(tmp_path) -> RunDirectory:
    """
    A resolution run, by its directory.
    """
    return RunDirectory(directory=tmp_path / "runs" / "a-run")


# %% launching


class TestLaunch:
    def test_the_run_is_performed_by_the_module_that_performs_runs(self, a_run):
        spawn = SpawnHandingOutProcesses([ProcessReportingItsState()])
        LiveRunLauncher(spawn=spawn).launch(a_run)
        assert spawn.commands == [
            [
                sys.executable,
                "-m",
                LiveModule.PERFORM_RUN,
                LiveArgument.RUN,
                str(a_run.directory),
                LiveArgument.WAIT_FOR_START,
            ]
        ]

    def test_the_run_can_be_performed_without_waiting_to_be_started(self, a_run):
        spawn = SpawnHandingOutProcesses([ProcessReportingItsState()])
        LiveRunLauncher(spawn=spawn, waits_for_start=False).launch(a_run)
        assert LiveArgument.WAIT_FOR_START not in spawn.commands[0]

    def test_a_start_requested_for_an_earlier_performance_is_forgotten(self, a_run):
        a_run.directory.mkdir(parents=True)
        a_run.request_start()
        LiveRunLauncher(
            spawn=SpawnHandingOutProcesses([ProcessReportingItsState()])
        ).launch(a_run)
        assert not a_run.start_requested

    def test_its_output_is_kept_in_the_data_directory(self, a_run, tmp_path):
        LiveRunLauncher(
            spawn=SpawnHandingOutProcesses([ProcessReportingItsState()])
        ).launch(a_run)
        assert (tmp_path / "data" / "live_run.log").is_file()

    def test_launching_again_restarts_the_run_that_is_still_going(self, a_run):
        first = ProcessReportingItsState()
        spawn = SpawnHandingOutProcesses([first, ProcessReportingItsState()])
        launcher = LiveRunLauncher(spawn=spawn)
        launcher.launch(a_run)
        launcher.launch(a_run)
        assert (first.terminated, len(spawn.commands)) == (True, 2)

    def test_a_process_that_ignores_termination_is_killed(self, a_run):
        first = ProcessReportingItsState(ignores_termination=True)
        spawn = SpawnHandingOutProcesses([first, ProcessReportingItsState()])
        launcher = LiveRunLauncher(spawn=spawn)
        launcher.launch(a_run)
        launcher.launch(a_run)
        assert (first.terminated, first.killed) == (True, True)

    def test_a_process_that_already_exited_is_left_alone(self, a_run):
        first = ProcessReportingItsState(exit_code=0)
        spawn = SpawnHandingOutProcesses([first, ProcessReportingItsState()])
        launcher = LiveRunLauncher(spawn=spawn)
        launcher.launch(a_run)
        launcher.launch(a_run)
        assert (first.terminated, len(spawn.commands)) == (False, 2)


# %% starting what waits


class TestRequestStart:
    def test_the_performance_that_waits_is_asked_to_start(self, a_run):
        a_run.directory.mkdir(parents=True)
        launcher = LiveRunLauncher(
            spawn=SpawnHandingOutProcesses([ProcessReportingItsState()])
        )
        launcher.launch(a_run)
        launcher.request_start()
        assert a_run.start_requested

    def test_nothing_waits_before_a_run_is_launched(self):
        with pytest.raises(NoPerformanceWaiting):
            LiveRunLauncher(spawn=SpawnHandingOutProcesses([])).request_start()

    def test_nothing_waits_once_the_process_has_exited(self, a_run):
        a_run.directory.mkdir(parents=True)
        launcher = LiveRunLauncher(
            spawn=SpawnHandingOutProcesses([ProcessReportingItsState(exit_code=0)])
        )
        launcher.launch(a_run)
        with pytest.raises(NoPerformanceWaiting):
            launcher.request_start()

    def test_a_performance_not_set_to_wait_has_nothing_to_start(self, a_run):
        a_run.directory.mkdir(parents=True)
        launcher = LiveRunLauncher(
            spawn=SpawnHandingOutProcesses([ProcessReportingItsState()]),
            waits_for_start=False,
        )
        launcher.launch(a_run)
        with pytest.raises(NoPerformanceWaiting):
            launcher.request_start()


class TestWaitingForStart:
    def test_the_marker_is_what_the_run_directory_names_it(self, a_run):
        a_run.directory.mkdir(parents=True)
        a_run.request_start()
        assert (a_run.directory / RunFile.START_REQUEST).is_file()

    def test_a_start_already_requested_does_not_wait(self, a_run):
        a_run.directory.mkdir(parents=True)
        a_run.request_start()
        ResolvedRun(directory=a_run.directory, start_poll_seconds=0.01).wait_for_start(
            a_run
        )

    def test_the_performance_waits_until_a_start_is_requested(self, a_run):
        a_run.directory.mkdir(parents=True)
        waiting = ResolvedRun(directory=a_run.directory, start_poll_seconds=0.01)
        requested_at = []

        def request_later():
            time.sleep(0.15)
            requested_at.append(time.monotonic())
            a_run.request_start()

        thread = threading.Thread(target=request_later)
        thread.start()
        waiting.wait_for_start(a_run)
        released_at = time.monotonic()
        thread.join()
        assert released_at >= requested_at[0]

    def test_the_command_line_says_whether_to_wait(self, a_run):
        arguments = [
            LiveArgument.RUN,
            str(a_run.directory),
            LiveArgument.WAIT_FOR_START,
        ]
        assert ResolvedRun.from_arguments(arguments).waits_for_start is True

    def test_the_command_line_does_not_wait_unless_told_to(self, a_run):
        arguments = [LiveArgument.RUN, str(a_run.directory)]
        assert ResolvedRun.from_arguments(arguments).waits_for_start is False
