"""
Agent-facing commands preserve pinned source and explicit freshness.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from repository_knowledge.adapters.serialization import RecordCodec
from repository_knowledge.model import IndexManifest, QueryPage

from .support import CommittedProject, FixtureSymbol

# %% command-line workflow


def test_cli_indexes_and_describes_a_committed_symbol(
    committed_project: CommittedProject, tmp_path: Path
) -> None:
    """
    The CLI connects indexing and source navigation through one saved snapshot.
    """
    database = tmp_path / "index.sqlite3"
    command = [
        sys.executable,
        "-m",
        "repository_knowledge",
        "--repository",
        str(committed_project.root),
        "--database",
        str(database),
    ]
    indexed = subprocess.run(command + ["index"], capture_output=True, text=True)
    assert indexed.returncode == 0, indexed.stderr
    manifest = RecordCodec().decode(indexed.stdout, IndexManifest)
    described = subprocess.run(
        command + ["describe-symbol", FixtureSymbol.CHILD],
        capture_output=True,
        text=True,
    )
    assert described.returncode == 0, described.stderr
    page = RecordCodec().decode(described.stdout, QueryPage)
    assert [symbol.symbol_identifier for symbol in page.symbols] == [
        FixtureSymbol.CHILD
    ]
    assert page.snapshot_identifier == manifest.snapshot_identifier


def test_cli_rejects_queries_after_requested_revision_changes(
    committed_project: CommittedProject, tmp_path: Path
) -> None:
    """
    A query cannot silently reuse an index built for another commit.
    """
    command = [
        sys.executable,
        "-m",
        "repository_knowledge",
        "--repository",
        str(committed_project.root),
        "--database",
        str(tmp_path / "index.sqlite3"),
    ]
    indexed = subprocess.run(command + ["index"], capture_output=True, text=True)
    assert indexed.returncode == 0, indexed.stderr
    committed_project.git(
        "commit", "--quiet", "--allow-empty", "-m", "Changed requested revision fixture"
    )
    result = subprocess.run(
        command + ["describe-symbol", FixtureSymbol.CHILD],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 1


def test_cli_help_does_not_require_a_repository() -> None:
    """
    An agent can discover the command contract before selecting source.
    """
    result = subprocess.run(
        [sys.executable, "-m", "repository_knowledge", "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
