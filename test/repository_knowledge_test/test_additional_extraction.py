"""
Source evidence retains type guards and safe import boundaries.
"""

from __future__ import annotations

from repository_knowledge.adapters.git_source import GitSource
from repository_knowledge.adapters.python_source import PythonSource
from repository_knowledge.model import RelationshipKind
from repository_knowledge.operations.indexing import IndexBuilder

from .support import CommittedProject, FixtureSymbol

# %% type evidence


def test_type_checking_import_retains_its_guard(
    committed_project: CommittedProject,
) -> None:
    """
    A type-only relationship is not presented as an observed runtime import.
    """
    index = IndexBuilder(GitSource(committed_project.root), PythonSource()).build(
        committed_project.commit_identifier
    )
    imported = [
        reference
        for reference in index.references
        if reference.relationship_kind == RelationshipKind.IMPORTS
        and reference.target_identifier == FixtureSymbol.OTHER_BASE
    ]
    assert [reference.is_type_only for reference in imported] == [True]


def test_snapshot_extraction_is_reproducible(
    committed_project: CommittedProject,
) -> None:
    """
    Identical source and interpreter inputs yield identical semantic records.
    """
    builder = IndexBuilder(GitSource(committed_project.root), PythonSource())
    first = builder.build(committed_project.commit_identifier)
    assert builder.build(committed_project.commit_identifier) == first
