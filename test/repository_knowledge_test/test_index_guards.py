"""
Index publication and queries preserve explicit failure boundaries.
"""

from __future__ import annotations

import sqlite3
from dataclasses import replace
from pathlib import Path

import pytest

from repository_knowledge.adapters.git_source import GitSource
from repository_knowledge.adapters.python_source import PythonSource
from repository_knowledge.adapters.sqlite_index import SQLiteIndex
from repository_knowledge.exceptions import InvalidQuery, UnsupportedIndexFormat
from repository_knowledge.model import IndexData
from repository_knowledge.operations.indexing import IndexBuilder
from repository_knowledge.operations.queries import Navigation

from .support import CommittedProject, FixtureSymbol
from .dataset.index_reader import PinnedIndex

# %% publication and query bounds


def test_failed_publication_preserves_previous_snapshot(
    committed_project: CommittedProject, tmp_path: Path
) -> None:
    """
    A failed write cannot expose the cleared or partially inserted snapshot.
    """
    index = IndexBuilder(GitSource(committed_project.root), PythonSource()).build(
        committed_project.commit_identifier
    )
    storage = SQLiteIndex(tmp_path / "index.sqlite3")
    storage.replace(index)
    invalid = IndexData(
        index.manifest,
        index.sources,
        index.symbols + index.symbols[:1],
        index.references,
        index.diagnostics,
    )
    with pytest.raises(sqlite3.IntegrityError):
        storage.replace(invalid)
    assert storage.read() == index


def test_unknown_index_format_requires_rebuild(
    committed_project: CommittedProject, tmp_path: Path
) -> None:
    """
    An unknown persisted schema is never silently interpreted.
    """
    index = IndexBuilder(GitSource(committed_project.root), PythonSource()).build(
        committed_project.commit_identifier
    )
    storage = SQLiteIndex(tmp_path / "index.sqlite3")
    storage.replace(
        replace(
            index,
            manifest=replace(
                index.manifest, format_version=index.manifest.format_version + 1
            ),
        )
    )
    with pytest.raises(UnsupportedIndexFormat):
        storage.read()


def test_query_rejects_zero_result_limit(
    committed_project: CommittedProject, tmp_path: Path
) -> None:
    """
    Invalid result bounds are rejected before querying storage.
    """
    storage = SQLiteIndex(tmp_path / "absent.sqlite3")
    with pytest.raises(InvalidQuery):
        Navigation(storage).find_symbols(FixtureSymbol.CHILD, limit=0)


def test_source_excerpt_reports_omitted_lines(
    committed_project: CommittedProject, tmp_path: Path
) -> None:
    """
    A source excerpt cannot silently exceed its configured line budget.
    """
    index = IndexBuilder(GitSource(committed_project.root), PythonSource()).build(
        committed_project.commit_identifier
    )
    navigation = Navigation(PinnedIndex(index), maximum_excerpt_lines=2)
    page = navigation.describe_symbol(FixtureSymbol.CHILD)
    symbol = page.symbols[0]
    assert (
        page.excerpts[0].location.end_line
        == symbol.location.start_line + navigation.maximum_excerpt_lines - 1
    )
    assert (
        page.excerpts[0].omitted_lines
        == symbol.location.end_line - page.excerpts[0].location.end_line
    )
