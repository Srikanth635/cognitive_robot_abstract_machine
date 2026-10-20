"""
Lexical extraction preserves generic bases, aliases and unresolved calls.
"""

from __future__ import annotations

import shutil
from pathlib import Path

from repository_knowledge.adapters.git_source import GitSource
from repository_knowledge.adapters.python_source import PythonSource
from repository_knowledge.model import (
    DiagnosticKind,
    RelationshipKind,
    Resolution,
    SymbolKind,
)
from repository_knowledge.operations.indexing import IndexBuilder

from .support import CommittedProject, FixturePath, FixtureSymbol

# %% source declarations


def test_index_preserves_nested_qualified_names(
    committed_project: CommittedProject,
) -> None:
    """
    Nested classes retain the complete lexical qualification.
    """
    index = IndexBuilder(GitSource(committed_project.root), PythonSource()).build(
        committed_project.commit_identifier
    )
    nested = next(
        symbol
        for symbol in index.symbols
        if symbol.symbol_identifier == FixtureSymbol.NESTED
    )
    assert nested.symbol_name == "Nested"
    assert nested.symbol_kind == SymbolKind.CLASS


def test_generic_base_alias_resolves_to_its_declaration(
    committed_project: CommittedProject,
) -> None:
    """
    An alias and generic argument do not obscure the base declaration.
    """
    index = IndexBuilder(GitSource(committed_project.root), PythonSource()).build(
        committed_project.commit_identifier
    )
    inheritance = [
        reference
        for reference in index.references
        if reference.source_identifier == FixtureSymbol.CHILD
        and reference.relationship_kind == RelationshipKind.INHERITS
    ]
    assert [reference.target_identifier for reference in inheritance] == [
        FixtureSymbol.BASE
    ]
    assert [reference.resolution for reference in inheritance] == [Resolution.RESOLVED]


def test_parameter_shadow_does_not_resolve_to_imported_function(
    committed_project: CommittedProject,
) -> None:
    """
    A local parameter prevents an unrelated imported target from being asserted.
    """
    index = IndexBuilder(GitSource(committed_project.root), PythonSource()).build(
        committed_project.commit_identifier
    )
    calls = [
        reference
        for reference in index.references
        if reference.source_identifier == "first.api.shadow_alias"
        and reference.relationship_kind == RelationshipKind.CALLS
    ]
    assert [reference.resolution for reference in calls] == [Resolution.UNRESOLVED]
    assert [reference.target_identifier for reference in calls] == [""]


def test_imported_function_call_resolves_across_modules(
    committed_project: CommittedProject,
) -> None:
    """
    A lexical import supports a precise cross-module call reference.
    """
    index = IndexBuilder(GitSource(committed_project.root), PythonSource()).build(
        committed_project.commit_identifier
    )
    calls = [
        reference
        for reference in index.references
        if reference.source_identifier == "first.api.use_alias"
        and reference.relationship_kind == RelationshipKind.CALLS
    ]
    assert [reference.target_identifier for reference in calls] == [
        FixtureSymbol.HELPER
    ]


def test_invalid_source_reports_failure_without_losing_valid_files(
    committed_project: CommittedProject,
) -> None:
    """
    A syntax failure remains visible while other declarations stay available.
    """
    shutil.copyfile(
        Path(__file__).parent / FixturePath.INVALID,
        committed_project.root / FixturePath.BROKEN,
    )
    committed_project.git("add", FixturePath.BROKEN)
    committed_project.git("commit", "--quiet", "-m", "Syntax failure fixture")
    index = IndexBuilder(GitSource(committed_project.root), PythonSource()).build(
        "HEAD"
    )
    assert [
        entry.diagnostic_kind
        for entry in index.diagnostics
        if entry.relative_path == FixturePath.BROKEN
    ] == [DiagnosticKind.SYNTAX]
    assert (
        next(
            symbol
            for symbol in index.symbols
            if symbol.symbol_identifier == FixtureSymbol.CHILD
        ).symbol_kind
        == SymbolKind.CLASS
    )


def test_enum_member_remains_a_distinct_declaration(
    committed_project: CommittedProject,
) -> None:
    """
    Enum values are identified as members rather than module variables.
    """
    index = IndexBuilder(GitSource(committed_project.root), PythonSource()).build(
        committed_project.commit_identifier
    )
    member = next(
        symbol
        for symbol in index.symbols
        if symbol.symbol_identifier == FixtureSymbol.ENUM_MEMBER
    )
    assert member.symbol_kind == SymbolKind.ENUM_MEMBER
