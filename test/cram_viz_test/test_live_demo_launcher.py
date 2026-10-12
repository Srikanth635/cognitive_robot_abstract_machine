"""
Tests for launching a demo under the live bridge from the viewer's "Execute" button,
instead of a separate ``cram-viz-live path/to/demo.py`` terminal command.

``spawn`` is stubbed throughout: nothing here starts a real subprocess or binds the live
bridge's port, so these run without coraplex/giskardpy and never touch a real demo.
"""

from __future__ import annotations

import subprocess
import sys

import pytest

from cram_viz.live.launcher import LiveDemoLauncher


class FakeProcess:
    """
    A process handle whose ``poll()`` reports running or exited, on demand.
    """

    def __init__(self, poll_result=None, hangs_on_terminate=False):
        self.poll_result = poll_result
        self.command = None
        self.kwargs = None
        self.terminated = False
        self.killed = False
        self.hangs_on_terminate = hangs_on_terminate
        self.wait_timeouts = []

    def poll(self):
        return self.poll_result

    def terminate(self):
        self.terminated = True
        if not self.hangs_on_terminate:
            self.poll_result = -15

    def kill(self):
        self.killed = True
        self.poll_result = -9

    def wait(self, timeout=None):
        self.wait_timeouts.append(timeout)
        if self.poll_result is None:
            raise subprocess.TimeoutExpired(cmd="demo", timeout=timeout)


class FakeSpawn:
    """
    Records each command it is asked to run and returns a preset process.

    Accepts either a single process (returned for every call, for tests where the exact
    process identity doesn't matter) or a list of processes (returned one per call, in
    order, for tests that need to tell the old and new processes apart).
    """

    def __init__(self, process):
        if isinstance(process, list):
            self._processes = list(process)
            self._fixed_process = None
        else:
            self._processes = None
            self._fixed_process = process
        self.calls = []

    def __call__(self, command, **kwargs):
        process = (
            self._processes.pop(0)
            if self._processes is not None
            else self._fixed_process
        )
        self.calls.append((command, kwargs))
        process.command = command
        process.kwargs = kwargs
        return process


@pytest.fixture()
def demo(tmp_path):
    path = tmp_path / "demo.py"
    path.write_text("# a demo\n")
    return path


class TestLaunch:
    def test_launches_the_demo_under_the_live_bridge_module(
        self, demo, tmp_path, monkeypatch
    ):
        monkeypatch.setenv("CRAM_VIZ_DATA", str(tmp_path / "data"))
        spawn = FakeSpawn(FakeProcess(poll_result=None))
        launcher = LiveDemoLauncher(spawn=spawn)

        launcher.launch(demo)

        assert len(spawn.calls) == 1
        command, kwargs = spawn.calls[0]
        assert command == [sys.executable, "-m", "cram_viz.live", str(demo)]
        assert kwargs["stderr"] is not None

    def test_a_missing_demo_is_rejected_before_spawning(self, tmp_path, monkeypatch):
        monkeypatch.setenv("CRAM_VIZ_DATA", str(tmp_path / "data"))
        spawn = FakeSpawn(FakeProcess())
        launcher = LiveDemoLauncher(spawn=spawn)

        with pytest.raises(FileNotFoundError):
            launcher.launch(tmp_path / "does-not-exist.py")

        assert spawn.calls == []

    def test_launching_again_while_the_first_is_still_running_restarts_it(
        self, demo, tmp_path, monkeypatch
    ):
        # the live bridge (cram_viz.live.runner.main) keeps its process alive
        # indefinitely once a demo finishes, waiting for Ctrl-C -- so "still
        # running" per poll() does not mean the demo logic is still going, and
        # clicking Execute again must restart it rather than refuse
        monkeypatch.setenv("CRAM_VIZ_DATA", str(tmp_path / "data"))
        first_process = FakeProcess(poll_result=None)  # still running
        second_process = FakeProcess(poll_result=None)
        spawn = FakeSpawn([first_process, second_process])
        launcher = LiveDemoLauncher(spawn=spawn)
        launcher.launch(demo)

        launcher.launch(demo)

        assert first_process.terminated is True
        assert len(spawn.calls) == 2

    def test_launching_again_kills_a_process_that_ignores_terminate(
        self, demo, tmp_path, monkeypatch
    ):
        monkeypatch.setenv("CRAM_VIZ_DATA", str(tmp_path / "data"))
        first_process = FakeProcess(poll_result=None, hangs_on_terminate=True)
        second_process = FakeProcess(poll_result=None)
        spawn = FakeSpawn([first_process, second_process])
        launcher = LiveDemoLauncher(spawn=spawn)
        launcher.launch(demo)

        launcher.launch(demo)

        assert first_process.terminated is True
        assert first_process.killed is True
        assert len(spawn.calls) == 2

    def test_launching_again_after_the_first_exited_is_allowed(
        self, demo, tmp_path, monkeypatch
    ):
        monkeypatch.setenv("CRAM_VIZ_DATA", str(tmp_path / "data"))
        spawn = FakeSpawn(FakeProcess(poll_result=0))  # already exited
        launcher = LiveDemoLauncher(spawn=spawn)
        launcher.launch(demo)

        launcher.launch(demo)

        assert len(spawn.calls) == 2

    def test_log_output_is_written_under_the_data_directory(
        self, demo, tmp_path, monkeypatch
    ):
        monkeypatch.setenv("CRAM_VIZ_DATA", str(tmp_path / "data"))
        spawn = FakeSpawn(FakeProcess(poll_result=None))
        launcher = LiveDemoLauncher(spawn=spawn)

        launcher.launch(demo)

        assert (tmp_path / "data" / "live_demo.log").is_file()
