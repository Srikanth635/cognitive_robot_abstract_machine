"""
Opt-in commit integration queues work without refreshing or committing.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

from .support import CommittedProject

# %% deterministic refresh request


def test_optional_hook_records_only_the_latest_commit(
    committed_project: CommittedProject,
) -> None:
    """
    An existing hook can invoke the queue helper without creating follow-on commits.
    """
    shell = (
        "C:/Program Files/Git/bin/bash.exe"
        if os.name == "nt"
        else (shutil.which("bash") or "/usr/bin/bash")
    )
    if not Path(shell).is_file():
        pytest.skip("Bash is required for the optional hook helper.")
    helper = (
        Path(__file__).parents[2] / "repository_knowledge/scripts/request-refresh.sh"
    )
    marker = committed_project.root / "repository_knowledge/.cache/refresh-request"
    result = subprocess.run(
        [shell, str(helper)], cwd=committed_project.root, capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
    assert marker.read_text().strip() == committed_project.commit_identifier
    committed_project.git(
        "commit", "--quiet", "--allow-empty", "-m", "New queue fixture"
    )
    latest = committed_project.git("rev-parse", "HEAD")
    result = subprocess.run(
        [shell, str(helper)], cwd=committed_project.root, capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
    assert marker.read_text().strip() == latest
    assert committed_project.git("rev-parse", "HEAD") == latest
    assert list(marker.parent.glob("refresh-request.*")) == []
