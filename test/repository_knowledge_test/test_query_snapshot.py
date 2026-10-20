"""
Revision validation and evidence capture share a completed snapshot.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from repository_knowledge.adapters.git_source import GitSource
from repository_knowledge.adapters.python_source import PythonSource
from repository_knowledge.adapters.sqlite_index import SQLiteIndex
from repository_knowledge.interfaces.cli import CommandLine, CommandParser
from repository_knowledge.operations.freshness import FreshnessCheck
from repository_knowledge.operations.indexing import IndexBuilder
from repository_knowledge.operations.queries import Navigation

from .dataset.advancing_index import AdvancingIndex
from .support import CommittedProject, FixtureSymbol

# %% query snapshot consistency


def test_query_uses_the_snapshot_checked_for_freshness(
    committed_project: CommittedProject, tmp_path: Path
) -> None:
    """
    A concurrent refresh cannot replace evidence after revision validation.
    """
    source = GitSource(committed_project.root)
    builder = IndexBuilder(source, PythonSource())
    original = builder.build(committed_project.commit_identifier)
    replacement = replace(
        original, manifest=replace(original.manifest, snapshot_identifier="replacement")
    )
    reader = AdvancingIndex([original, replacement])
    application = CommandLine(
        builder,
        SQLiteIndex(tmp_path / "index.sqlite3"),
        Navigation(reader),
        FreshnessCheck(reader, source),
    )
    arguments = CommandParser().parse(
        [
            "--repository",
            str(committed_project.root),
            "describe-symbol",
            FixtureSymbol.CHILD,
        ]
    )
    result = application.execute(arguments)
    assert result.snapshot_identifier == original.manifest.snapshot_identifier
