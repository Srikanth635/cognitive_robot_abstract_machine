"""
A query cannot return working-file evidence changed during navigation.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from repository_knowledge.adapters.git_source import GitArgument

from repository_knowledge.adapters.python_source import PythonSource
from repository_knowledge.adapters.sqlite_index import SQLiteIndex
from repository_knowledge.adapters.working_tree import WorkingTreeSource
from repository_knowledge.exceptions import SourceSnapshotMismatch
from repository_knowledge.interfaces.cli import CommandLine, CommandParser
from repository_knowledge.model import SourceScope
from repository_knowledge.operations.freshness import FreshnessCheck
from repository_knowledge.operations.indexing import IndexBuilder

from .dataset.editing_navigation import EditingNavigation
from .support import CommittedProject, FixturePath, FixtureSymbol

# %% final source recheck


def test_query_rejects_source_changed_during_navigation(
    committed_project: CommittedProject, tmp_path: Path
) -> None:
    """
    An edit after the initial check invalidates the selected evidence.
    """
    source = WorkingTreeSource(committed_project.root)
    builder = IndexBuilder(source, PythonSource())
    storage = SQLiteIndex(tmp_path / "overlay.sqlite3")
    storage.replace(builder.build(GitArgument.CHECKOUT))
    navigation = EditingNavigation(
        storage,
        replacement_path=Path(__file__).parent / FixturePath.REPLACEMENT,
        changed_path=committed_project.root / FixturePath.API,
    )
    application = CommandLine(
        builder, storage, navigation, FreshnessCheck(storage, source)
    )
    arguments = CommandParser().parse(
        [
            "--repository",
            str(committed_project.root),
            "--scope",
            SourceScope.WORKING_TREE,
            "describe-symbol",
            FixtureSymbol.CHILD,
        ]
    )
    with pytest.raises(SourceSnapshotMismatch):
        application.execute(arguments)
