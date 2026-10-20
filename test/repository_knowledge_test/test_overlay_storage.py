"""
Overlay publication retains reusable evidence and scope boundaries.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from repository_knowledge.adapters.git_source import GitArgument

from repository_knowledge.adapters.python_source import PythonSource
from repository_knowledge.adapters.sqlite_index import SQLiteIndex
from repository_knowledge.adapters.working_tree import WorkingTreeSource
from repository_knowledge.exceptions import SourceScopeMismatch
from repository_knowledge.model import IndexData, IndexSelection, SelectionKind
from repository_knowledge.operations.indexing import IndexBuilder

from .support import CommittedProject

# %% persisted refresh evidence


def test_overlay_storage_preserves_raw_file_analysis(
    committed_project: CommittedProject, tmp_path: Path
) -> None:
    """
    Stored analysis supports refresh without losing original import bindings.
    """
    index = IndexBuilder(
        WorkingTreeSource(committed_project.root), PythonSource()
    ).build(GitArgument.CHECKOUT)
    storage = SQLiteIndex(tmp_path / "overlay.sqlite3")
    storage.replace(index)
    saved = storage.read(IndexSelection(SelectionKind.ANALYSES))
    assert saved.analyses == index.analyses
    assert saved.sources == index.sources


def test_overlay_cannot_replace_committed_cache(
    committed_project: CommittedProject,
    source_index: IndexData,
    indexed_storage: SQLiteIndex,
) -> None:
    """
    An explicit shared database path cannot destroy the committed baseline.
    """
    overlay = IndexBuilder(
        WorkingTreeSource(committed_project.root), PythonSource()
    ).build(GitArgument.CHECKOUT)
    with pytest.raises(SourceScopeMismatch):
        indexed_storage.replace(overlay)
    assert indexed_storage.read() == source_index
