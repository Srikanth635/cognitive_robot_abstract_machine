"""
Agent commands expose task evidence behind source freshness checks.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from repository_knowledge.adapters.serialization import RecordCodec
from repository_knowledge.adapters.sqlite_index import SQLiteIndex
from repository_knowledge.assistance import (
    ChangeReport,
    DiscoveryReport,
    TaskContext,
    TaskProfile,
)
from repository_knowledge.model import IndexData

from .support import CommittedProject, FixtureSymbol

# %% bounded commands


def test_cli_prepares_context_and_discovers_tests(
    assistance_project: CommittedProject, assistance_index: IndexData, tmp_path: Path
) -> None:
    """
    All three task queries use the same saved committed snapshot.
    """
    database = tmp_path / "context.sqlite3"
    SQLiteIndex(database).replace(assistance_index)
    command = [
        sys.executable,
        "-m",
        "repository_knowledge",
        "--repository",
        str(assistance_project.root),
        "--database",
        str(database),
    ]
    context_result = subprocess.run(
        command
        + [
            "prepare-context",
            FixtureSymbol.HELPER,
            "--profile",
            TaskProfile.DEBUGGING,
            "--characters",
            "30000",
        ],
        capture_output=True,
        text=True,
    )
    assert context_result.returncode == 0, context_result.stderr
    context = RecordCodec().decode(context_result.stdout, TaskContext)
    assert context.manifest == assistance_index.manifest
    assert context.profile == TaskProfile.DEBUGGING
    for operation in ("find-tests", "find-examples"):
        result = subprocess.run(
            command + [operation, FixtureSymbol.HELPER, "--limit", "2"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stderr
        report = RecordCodec().decode(result.stdout, DiscoveryReport)
        assert report.snapshot_identifier == context.manifest.snapshot_identifier
        assert len(report.tests or report.references) == 2


def test_cli_context_rejects_stale_source_index(
    assistance_project: CommittedProject, assistance_index: IndexData, tmp_path: Path
) -> None:
    """
    A task cannot act on a previous commit without explicitly requesting it.
    """
    database = tmp_path / "context.sqlite3"
    SQLiteIndex(database).replace(assistance_index)
    assistance_project.git(
        "commit", "--quiet", "--allow-empty", "-m", "New task source"
    )
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "repository_knowledge",
            "--repository",
            str(assistance_project.root),
            "--database",
            str(database),
            "prepare-context",
            FixtureSymbol.HELPER,
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 1


def test_cli_change_assessment_pins_both_requested_revisions(
    assistance_project: CommittedProject, assistance_index: IndexData, tmp_path: Path
) -> None:
    """
    Comparison requires an explicit baseline database and revision selector.
    """
    database = tmp_path / "context.sqlite3"
    SQLiteIndex(database).replace(assistance_index)
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "repository_knowledge",
            "--repository",
            str(assistance_project.root),
            "--database",
            str(database),
            "assess-change",
            FixtureSymbol.HELPER,
            "--against-database",
            str(database),
            "--against-revision",
            assistance_index.manifest.commit_identifier,
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    report = RecordCodec().decode(result.stdout, ChangeReport)
    assert report.before_manifest == report.after_manifest == assistance_index.manifest
