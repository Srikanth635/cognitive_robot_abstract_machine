"""
Persistent queries load the evidence required by each operation.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from repository_knowledge.adapters.git_source import GitSource
from repository_knowledge.adapters.python_source import PythonSource
from repository_knowledge.adapters.sqlite_index import SQLiteIndex
from repository_knowledge.model import IndexSelection, SelectionKind, TraversalDirection
from repository_knowledge.operations.indexing import IndexBuilder
from repository_knowledge.operations.queries import Navigation

from .dataset.index_reader import PinnedIndex
from .support import CommittedProject, FixturePath, FixtureSymbol

# %% persistent projections


def test_symbol_selection_loads_only_its_source_evidence(
    committed_project: CommittedProject, tmp_path: Path
) -> None:
    """
    Describing one declaration does not decode unrelated graph records.
    """
    index = IndexBuilder(GitSource(committed_project.root), PythonSource()).build(
        committed_project.commit_identifier
    )
    storage = SQLiteIndex(tmp_path / "index.sqlite3")
    storage.replace(index)
    selected = storage.read(
        IndexSelection(
            SelectionKind.SYMBOL_DETAILS, symbol_identifier=FixtureSymbol.CHILD
        )
    )
    assert [symbol.symbol_identifier for symbol in selected.symbols] == [
        FixtureSymbol.CHILD
    ]
    assert [source.relative_path for source in selected.sources] == [FixturePath.API]
    assert selected.references == []


@pytest.mark.parametrize("direction", list(TraversalDirection))
def test_selected_relationships_equal_complete_graph_navigation(
    committed_project: CommittedProject, tmp_path: Path, direction: TraversalDirection
) -> None:
    """
    Selection retains the same bounded traversal evidence in both directions.
    """
    index = IndexBuilder(GitSource(committed_project.root), PythonSource()).build(
        committed_project.commit_identifier
    )
    storage = SQLiteIndex(tmp_path / "index.sqlite3")
    storage.replace(index)
    expected = Navigation(PinnedIndex(index)).find_relationships(
        FixtureSymbol.CHILD, direction, depth=2
    )
    assert (
        Navigation(storage).find_relationships(FixtureSymbol.CHILD, direction, depth=2)
        == expected
    )
