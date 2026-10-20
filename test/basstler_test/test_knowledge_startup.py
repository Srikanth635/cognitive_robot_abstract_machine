"""
Optional knowledge inspection stays outside basstler's interpreter.
"""

from __future__ import annotations

import json
import io
import subprocess
import sys
import time
import tempfile
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import Mock

from pytest import MonkeyPatch

from basstler.knowledge_startup import KnowledgeStartup, StartupState

# %% process mimic


@dataclass
class DelayedInspection:
    """
    A process that remains alive until its deadline expires.
    """

    delay_seconds: float = 0.05
    """
    The simulated time before the child exits.
    """

    killed: bool = False
    """
    Whether the deadline stopped the child.
    """

    def wait(self) -> int:
        """
        Finish after the deadline callback can stop this inspection.
        """
        time.sleep(self.delay_seconds)
        return -1

    def kill(self) -> None:
        """
        Record the deadline's process termination.
        """
        self.killed = True


# %% optional startup contract


def test_knowledge_startup_without_interpreter_is_disabled(tmp_path: Path) -> None:
    """
    Basstler never guesses an interpreter or installs a package.
    """
    result = KnowledgeStartup(tmp_path).inspect()
    assert result.state == StartupState.DISABLED


def test_knowledge_startup_uses_selected_interpreter_and_versioned_report(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    """
    The subprocess is read-only and its small response becomes startup guidance.
    """
    payload = (Path(__file__).parent / "dataset/knowledge-report.json").read_bytes()
    process = Mock()
    process.wait.return_value = 0

    def launch(
        arguments: list[str],
        *,
        stdin: int,
        stdout: io.BufferedRandom | tempfile._TemporaryFileWrapper[bytes],
        stderr: int,
        cwd: Path,
    ) -> Mock:
        """
        Write the independent service's external response.
        """
        stdout.write(payload)
        return process

    launcher = Mock(side_effect=launch)
    monkeypatch.setattr(subprocess, "Popen", launcher)
    startup = KnowledgeStartup(tmp_path, Path(sys.executable))
    result = startup.inspect()
    assert result.state == StartupState.REPORTED
    assert result.summary == json.loads(payload)["summary"]
    assert launcher.call_args.args[0] == [
        sys.executable,
        "-m",
        "repository_knowledge.interfaces.session",
        "--repository",
        str(tmp_path),
        "--revision",
        "HEAD",
    ]
    process.kill.assert_not_called()


def test_knowledge_startup_stops_a_hung_child(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    """
    A slow optional service cannot block the rest of startup.
    """
    process = DelayedInspection()
    monkeypatch.setattr(subprocess, "Popen", Mock(return_value=process))
    result = KnowledgeStartup(
        tmp_path, Path(sys.executable), timeout_seconds=0.01
    ).inspect()
    assert result.state == StartupState.TIMED_OUT
    assert process.killed is True


def test_knowledge_startup_reports_missing_package(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    """
    An unavailable optional module leaves ordinary source navigation usable.
    """
    process = Mock()
    process.wait.return_value = 1
    monkeypatch.setattr(subprocess, "Popen", Mock(return_value=process))
    assert (
        KnowledgeStartup(tmp_path, Path(sys.executable)).inspect().state
        == StartupState.UNAVAILABLE
    )
