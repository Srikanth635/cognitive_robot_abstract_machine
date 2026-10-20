"""
Startup observes source freshness without rebuilding or creating a cache.
"""

from __future__ import annotations

from pathlib import Path

from repository_knowledge.adapters.sqlite_index import SQLiteIndex
from repository_knowledge.interfaces.cli import CommandParser
from repository_knowledge.interfaces.session import SessionApplication
from repository_knowledge.clients import SessionState

from .support import CommittedProject

# %% bounded startup report


def test_session_report_reads_current_manifest(
    committed_project: CommittedProject, indexed_storage: SQLiteIndex
) -> None:
    """
    A current report carries the exact source and explicit wiki entry point.
    """
    arguments = CommandParser().parse(
        [
            "--repository",
            str(committed_project.root),
            "--database",
            str(indexed_storage.database_path),
            "check-freshness",
        ]
    )
    before = indexed_storage.database_path.read_bytes()
    report = SessionApplication().inspect(arguments)
    assert report.state == SessionState.CURRENT
    assert report.source_commit == committed_project.commit_identifier
    assert report.wiki_entrypoint == "repository_knowledge/wiki/index.md"
    assert indexed_storage.database_path.read_bytes() == before


def test_session_report_does_not_create_missing_cache(
    committed_project: CommittedProject, tmp_path: Path
) -> None:
    """
    A session can start with source-only guidance when indexing is absent.
    """
    database = tmp_path / "absent/index.sqlite3"
    arguments = CommandParser().parse(
        [
            "--repository",
            str(committed_project.root),
            "--database",
            str(database),
            "check-freshness",
        ]
    )
    report = SessionApplication().inspect(arguments)
    assert report.state == SessionState.MISSING
    assert database.parent.exists() is False


def test_session_report_marks_changed_source_stale(
    committed_project: CommittedProject, indexed_storage: SQLiteIndex
) -> None:
    """
    Startup reports stale evidence without trying to repair it.
    """
    arguments = CommandParser().parse(
        [
            "--repository",
            str(committed_project.root),
            "--database",
            str(indexed_storage.database_path),
            "check-freshness",
        ]
    )
    committed_project.git(
        "commit", "--quiet", "--allow-empty", "-m", "Advance session fixture"
    )
    report = SessionApplication().inspect(arguments)
    assert report.state == SessionState.STALE
    assert report.source_commit == committed_project.git("rev-parse", "HEAD")
