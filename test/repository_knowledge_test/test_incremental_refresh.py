"""
Incremental extraction stays equivalent to a clean source rebuild.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from repository_knowledge.adapters.git_source import GitArgument

from repository_knowledge.adapters.python_source import PythonSource
from repository_knowledge.adapters.working_tree import WorkingTreeSource
from repository_knowledge.model import DiagnosticKind, RelationshipKind, Resolution
from repository_knowledge.operations.indexing import IndexBuilder

from .dataset.counting_analyzer import CountingAnalyzer
from .dataset.source_mutation import SourceMutation
from .support import CommittedProject, FixturePath, FixtureSymbol

# %% unchanged and changed source


def test_refresh_only_parses_modified_source(
    committed_project: CommittedProject,
) -> None:
    """
    Unchanged source analysis remains reusable after an ordinary edit.
    """
    analyzer = CountingAnalyzer(PythonSource())
    builder = IndexBuilder(WorkingTreeSource(committed_project.root), analyzer)
    previous = builder.build(GitArgument.CHECKOUT)
    analyzer.analyzed_paths.clear()
    shutil.copyfile(
        Path(__file__).parent / FixturePath.REPLACEMENT,
        committed_project.root / FixturePath.API,
    )
    builder.build(GitArgument.CHECKOUT, previous)
    assert analyzer.analyzed_paths == [FixturePath.API]


def test_unchanged_refresh_has_no_semantic_churn(
    committed_project: CommittedProject,
) -> None:
    """
    Identical source inputs produce identical evidence without re-parsing.
    """
    analyzer = CountingAnalyzer(PythonSource())
    builder = IndexBuilder(WorkingTreeSource(committed_project.root), analyzer)
    previous = builder.build(GitArgument.CHECKOUT)
    analyzer.analyzed_paths.clear()
    refreshed = builder.build(GitArgument.CHECKOUT, previous)
    assert refreshed == previous
    assert analyzer.analyzed_paths == []


@pytest.mark.parametrize("mutation", list(SourceMutation))
def test_refresh_equals_clean_rebuild_after_source_mutation(
    committed_project: CommittedProject, mutation: SourceMutation
) -> None:
    """
    Mutation cannot leave declarations or resolved references from old source.
    """
    added_path = FixturePath.ADDED
    reader = WorkingTreeSource(
        committed_project.root, included_untracked_paths=(added_path,)
    )
    builder = IndexBuilder(reader, PythonSource())
    previous = builder.build(GitArgument.CHECKOUT)
    dataset = Path(__file__).parent
    if mutation == SourceMutation.DELETED:
        (committed_project.root / FixturePath.BASE).unlink()
    elif mutation == SourceMutation.RENAMED:
        committed_project.git("mv", FixturePath.BASE, FixturePath.RENAMED)
    elif mutation == SourceMutation.CHANGED_SOURCE_ROOTS:
        shutil.copyfile(
            dataset / FixturePath.SOURCE_ROOTS,
            committed_project.root / FixturePath.FIRST_PROJECT,
        )
    elif mutation in (
        SourceMutation.STAGED_ADDITION,
        SourceMutation.UNTRACKED_ADDITION,
    ):
        shutil.copyfile(
            dataset / FixturePath.REPLACEMENT, committed_project.root / added_path
        )
        if mutation == SourceMutation.STAGED_ADDITION:
            committed_project.git("add", added_path)
    else:
        replacement = (
            FixturePath.INVALID
            if mutation == SourceMutation.INVALID_SYNTAX
            else FixturePath.REPLACEMENT
        )
        shutil.copyfile(dataset / replacement, committed_project.root / FixturePath.API)
        if mutation == SourceMutation.DIFFERENT_BRANCH:
            committed_project.git("checkout", "--quiet", "-b", "changed-source")
            committed_project.git("add", FixturePath.API)
            committed_project.git("commit", "--quiet", "-m", "Changed source fixture")
    assert builder.build(GitArgument.CHECKOUT, previous) == builder.build(
        GitArgument.CHECKOUT
    )


def test_deleted_target_invalidates_unchanged_callers(
    committed_project: CommittedProject,
) -> None:
    """
    A caller cannot retain a resolved target after its module disappears.
    """
    builder = IndexBuilder(WorkingTreeSource(committed_project.root), PythonSource())
    previous = builder.build(GitArgument.CHECKOUT)
    (committed_project.root / FixturePath.BASE).unlink()
    refreshed = builder.build(GitArgument.CHECKOUT, previous)
    calls = [
        reference
        for reference in refreshed.references
        if reference.source_identifier == FixtureSymbol.CALLER
        and reference.relationship_kind == RelationshipKind.CALLS
    ]
    assert [reference.resolution for reference in calls] == [Resolution.UNRESOLVED]
    assert [reference.target_identifier for reference in calls] == [""]


def test_parse_failure_removes_previous_declarations(
    committed_project: CommittedProject,
) -> None:
    """
    A broken file contributes diagnostics instead of historical definitions.
    """
    builder = IndexBuilder(WorkingTreeSource(committed_project.root), PythonSource())
    previous = builder.build(GitArgument.CHECKOUT)
    shutil.copyfile(
        Path(__file__).parent / FixturePath.INVALID,
        committed_project.root / FixturePath.API,
    )
    refreshed = builder.build(GitArgument.CHECKOUT, previous)
    assert [
        symbol.symbol_identifier
        for symbol in refreshed.symbols
        if symbol.location.relative_path == FixturePath.API
    ] == []
    assert [
        diagnostic.diagnostic_kind
        for diagnostic in refreshed.diagnostics
        if diagnostic.relative_path == FixturePath.API
    ] == [DiagnosticKind.SYNTAX]


def test_export_change_relinks_unchanged_consumer(
    committed_project: CommittedProject,
) -> None:
    """
    Changing a re-export changes the target of its existing consumer.
    """
    dataset = Path(__file__).parent / FixturePath.OVERLAY_DATASET
    consumer_path = FixturePath.CONSUMER
    shutil.copyfile(dataset / "consumer.py", committed_project.root / consumer_path)
    reader = WorkingTreeSource(
        committed_project.root, included_untracked_paths=(consumer_path,)
    )
    builder = IndexBuilder(reader, PythonSource())
    previous = builder.build(GitArgument.CHECKOUT)
    shutil.copyfile(
        dataset / "export.py", committed_project.root / FixturePath.PUBLIC_EXPORT
    )
    refreshed = builder.build(GitArgument.CHECKOUT, previous)
    bases = [
        reference
        for reference in refreshed.references
        if reference.source_identifier == FixtureSymbol.CONSUMER
        and reference.relationship_kind == RelationshipKind.INHERITS
    ]
    assert [reference.target_identifier for reference in bases] == [
        FixtureSymbol.OTHER_BASE
    ]
