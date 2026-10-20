"""
An unchanged source snapshot does not need persisted analysis decoding.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from repository_knowledge.adapters.git_source import GitArgument

from repository_knowledge.adapters.python_source import PythonSource
from repository_knowledge.adapters.sqlite_index import SQLiteIndex
from repository_knowledge.adapters.working_tree import WorkingTreeSource
from repository_knowledge.model import IndexData, IndexSelection, SelectionKind
from repository_knowledge.operations.indexing import IndexBuilder
from repository_knowledge.operations.refresh import IndexRefresh

from .support import CommittedProject

# %% unchanged source


def test_unchanged_refresh_does_not_decode_file_analysis(
    committed_project: CommittedProject, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    A no-change refresh needs only the saved identity and captured files.
    """
    builder = IndexBuilder(WorkingTreeSource(committed_project.root), PythonSource())
    storage = SQLiteIndex(tmp_path / "overlay.sqlite3")
    storage.replace(builder.build(GitArgument.CHECKOUT))
    original_read = storage.read

    def read_without_analysis(selection: IndexSelection | None = None) -> IndexData:
        """
        Reject decoding evidence that is unnecessary for identical source.
        """
        if selection is None or selection.selection_kind == SelectionKind.ANALYSES:
            pytest.fail("An unchanged refresh must not decode the file-analysis cache.")
        return original_read(selection)

    monkeypatch.setattr(storage, "read", read_without_analysis)
    report = IndexRefresh(builder, storage, storage).refresh(GitArgument.CHECKOUT)
    assert report.published is False
