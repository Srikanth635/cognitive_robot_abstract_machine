"""
Refresh and reconciliation expose current-source evidence safely.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

from repository_knowledge.adapters.git_source import GitArgument

from repository_knowledge.adapters.python_source import PythonSource
from repository_knowledge.adapters.serialization import RecordCodec
from repository_knowledge.adapters.sqlite_index import SQLiteIndex
from repository_knowledge.adapters.working_tree import WorkingTreeSource
from repository_knowledge.model import QueryPage, RefreshReport, SourceScope
from repository_knowledge.operations.indexing import IndexBuilder
from repository_knowledge.operations.refresh import IndexRefresh

from .support import CommittedProject, FixturePath, FixtureSymbol

# %% refresh publication


def test_unchanged_refresh_preserves_publication(
    committed_project: CommittedProject, tmp_path: Path
) -> None:
    """
    An unchanged overlay avoids replacing an identical saved graph.
    """
    builder = IndexBuilder(WorkingTreeSource(committed_project.root), PythonSource())
    storage = SQLiteIndex(tmp_path / "overlay.sqlite3")
    index = builder.build(GitArgument.CHECKOUT)
    storage.replace(index)
    report = IndexRefresh(builder, storage, storage).refresh(GitArgument.CHECKOUT)
    assert report.published is False
    assert report.analyzed_paths == []
    assert report.manifest == index.manifest


def test_reconciliation_rebuilds_all_file_analysis(
    committed_project: CommittedProject, tmp_path: Path
) -> None:
    """
    A clean reconciliation republishes the same complete semantics.
    """
    builder = IndexBuilder(WorkingTreeSource(committed_project.root), PythonSource())
    storage = SQLiteIndex(tmp_path / "overlay.sqlite3")
    original = builder.build(GitArgument.CHECKOUT)
    storage.replace(original)
    report = IndexRefresh(builder, storage, storage).refresh(
        GitArgument.CHECKOUT, reconcile=True
    )
    assert report.reused_paths == []
    assert report.analyzed_paths == [
        analysis.identity.relative_path for analysis in original.analyses
    ]
    assert storage.read() == builder.build(GitArgument.CHECKOUT)


def test_cli_refresh_rejects_old_source_and_serves_the_new_overlay(
    committed_project: CommittedProject, tmp_path: Path
) -> None:
    """
    An agent must refresh changed working-file evidence before navigation.
    """
    command = [
        sys.executable,
        "-m",
        "repository_knowledge",
        "--repository",
        str(committed_project.root),
        "--database",
        str(tmp_path / "overlay.sqlite3"),
        "--scope",
        SourceScope.WORKING_TREE,
    ]
    indexed = subprocess.run(command + ["index"], capture_output=True, text=True)
    assert indexed.returncode == 0, indexed.stderr
    shutil.copyfile(
        Path(__file__).parent / FixturePath.REPLACEMENT,
        committed_project.root / FixturePath.API,
    )
    stale = subprocess.run(
        command + ["describe-symbol", FixtureSymbol.CHILD],
        capture_output=True,
        text=True,
    )
    assert stale.returncode == 1
    refreshed = subprocess.run(command + ["refresh"], capture_output=True, text=True)
    assert refreshed.returncode == 0, refreshed.stderr
    report = RecordCodec().decode(refreshed.stdout, RefreshReport)
    assert report.analyzed_paths == [FixturePath.API]
    described = subprocess.run(
        command + ["describe-symbol", FixtureSymbol.REPLACEMENT],
        capture_output=True,
        text=True,
    )
    assert described.returncode == 0, described.stderr
    page = RecordCodec().decode(described.stdout, QueryPage)
    assert [symbol.symbol_identifier for symbol in page.symbols] == [
        FixtureSymbol.REPLACEMENT
    ]
    assert page.source_scope == SourceScope.WORKING_TREE
