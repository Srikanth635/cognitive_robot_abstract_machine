"""
Task requests retain the source identity checked for freshness.
"""

from __future__ import annotations

import shutil
from dataclasses import replace
from pathlib import Path

import pytest

from repository_knowledge.adapters.git_source import GitSource
from repository_knowledge.adapters.python_source import PythonSource
from repository_knowledge.adapters.pytest_source import PytestSource
from repository_knowledge.adapters.working_tree import WorkingTreeSource
from repository_knowledge.assistance import ChangeKind
from repository_knowledge.concepts import KnowledgeBundle
from repository_knowledge.exceptions import SourceSnapshotMismatch, InvalidQuery
from repository_knowledge.interfaces.cli import CommandParser
from repository_knowledge.interfaces.context import ContextApplication
from repository_knowledge.model import IndexData, SourceScope
from repository_knowledge.operations.changes import ChangeAssessment
from repository_knowledge.operations.indexing import IndexBuilder
from repository_knowledge.operations.queries import IndexContents

from .dataset.advancing_index import AdvancingIndex
from .dataset.editing_test_analysis import EditingTestAnalysis
from .support import CommittedProject, FixturePath, FixtureSymbol

# %% snapshot guards


def test_context_uses_one_completed_index(
    assistance_project: CommittedProject, assistance_index: IndexData
) -> None:
    """
    Concurrent publication cannot replace context after its source check.
    """
    replacement = replace(
        assistance_index,
        manifest=replace(
            assistance_index.manifest, snapshot_identifier="later-publication"
        ),
    )
    reader = AdvancingIndex([assistance_index, replacement])
    arguments = CommandParser().parse(
        [
            "--repository",
            str(assistance_project.root),
            "prepare-context",
            FixtureSymbol.HELPER,
        ]
    )
    context = ContextApplication(
        reader, GitSource(assistance_project.root), PytestSource()
    ).execute(arguments, KnowledgeBundle([]))
    assert context.manifest == assistance_index.manifest
    assert reader.read_count == 1


def test_context_rejects_a_working_edit_during_test_analysis(
    assistance_project: CommittedProject,
) -> None:
    """
    Post-query validation rejects source edited while assembling evidence.
    """
    source = WorkingTreeSource(assistance_project.root)
    index = IndexBuilder(source, PythonSource()).build(
        assistance_project.commit_identifier
    )
    analyzer = EditingTestAnalysis(
        Path(__file__).parent / FixturePath.REPLACEMENT,
        assistance_project.root / FixturePath.API,
    )
    arguments = CommandParser().parse(
        [
            "--repository",
            str(assistance_project.root),
            "--scope",
            SourceScope.WORKING_TREE,
            "prepare-context",
            FixtureSymbol.HELPER,
        ]
    )
    with pytest.raises(SourceSnapshotMismatch):
        ContextApplication(IndexContents(index), source, analyzer).execute(
            arguments, KnowledgeBundle([])
        )


# %% comparison guards


def test_change_assessment_rejects_unrelated_histories(source_index: IndexData) -> None:
    """
    Declarations from another history cannot silently become a baseline.
    """
    unrelated = replace(
        source_index,
        manifest=replace(
            source_index.manifest, repository_identifier="unrelated-history"
        ),
    )
    with pytest.raises(InvalidQuery):
        ChangeAssessment(source_index, unrelated).assess_change(FixtureSymbol.BASE)


def test_change_assessment_detects_a_committed_signature_edit(
    assistance_project: CommittedProject, assistance_index: IndexData
) -> None:
    """
    A real source change becomes a declared contract change between snapshots.
    """
    shutil.copyfile(
        Path(__file__).parent / "dataset/assistance/helper_changed.py",
        assistance_project.root / FixturePath.BASE,
    )
    assistance_project.git("add", ".")
    assistance_project.git("commit", "--quiet", "-m", "Changed helper contract")
    after = IndexBuilder(GitSource(assistance_project.root), PythonSource()).build(
        "HEAD"
    )
    report = ChangeAssessment(assistance_index, after).assess_change(
        FixtureSymbol.HELPER,
        PytestSource().discover(assistance_index),
        PytestSource().discover(after),
    )
    assert report.contracts[0].change_kind == ChangeKind.CHANGED
    assert report.before_manifest == assistance_index.manifest
    assert report.after_manifest == after.manifest
    assert report.suggested_test_directories == ["tests"]
