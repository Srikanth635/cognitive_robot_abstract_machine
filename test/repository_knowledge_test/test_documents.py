"""
Documentation participates in source navigation with explicit code links.
"""

from __future__ import annotations

from repository_knowledge.adapters.documents import DocumentSource, SourceCollection
from repository_knowledge.adapters.git_source import GitSource
from repository_knowledge.adapters.python_source import PythonSource
from repository_knowledge.model import RelationshipKind, SymbolKind
from repository_knowledge.operations.indexing import IndexBuilder

from .support import CommittedProject, FixtureSymbol

# %% document navigation


def test_rst_headings_are_indexed_as_source_sections(
    committed_project: CommittedProject,
) -> None:
    """
    RST sections remain navigable alongside Python declarations.
    """
    index = IndexBuilder(
        GitSource(committed_project.root),
        SourceCollection([PythonSource(), DocumentSource()]),
    ).build(committed_project.commit_identifier)
    sections = [
        symbol
        for symbol in index.symbols
        if symbol.location.relative_path == "docs/overview.rst"
        and symbol.symbol_kind == SymbolKind.DOCUMENT_SECTION
    ]
    assert [symbol.symbol_name for symbol in sections] == [
        "Fixture architecture",
        "Source navigation",
    ]


def test_markdown_link_resolves_to_committed_python_module(
    committed_project: CommittedProject,
) -> None:
    """
    A documentation link identifies its source module only when it exists.
    """
    index = IndexBuilder(
        GitSource(committed_project.root),
        SourceCollection([PythonSource(), DocumentSource()]),
    ).build(committed_project.commit_identifier)
    links = [
        reference
        for reference in index.references
        if reference.relationship_kind == RelationshipKind.DOCUMENTS
    ]
    assert [reference.target_identifier for reference in links] == [
        FixtureSymbol.BASE.rsplit(".", 1)[0]
    ]
