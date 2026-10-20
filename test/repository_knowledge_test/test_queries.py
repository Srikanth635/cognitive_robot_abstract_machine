"""
Navigation queries return bounded evidence from one pinned snapshot.
"""

from __future__ import annotations

from pathlib import Path

from repository_knowledge.adapters.git_source import GitSource
from repository_knowledge.adapters.python_source import PythonSource
from repository_knowledge.adapters.sqlite_index import SQLiteIndex
from repository_knowledge.model import RelationshipKind, Resolution
from repository_knowledge.operations.indexing import IndexBuilder
from repository_knowledge.operations.queries import Navigation

from .support import CommittedProject, FixtureSymbol

# %% source navigation


def test_describe_symbol_returns_exact_committed_excerpt(
    committed_project: CommittedProject, tmp_path: Path
) -> None:
    """
    Symbol descriptions include the source extent read from the indexed commit.
    """
    index = IndexBuilder(GitSource(committed_project.root), PythonSource()).build(
        committed_project.commit_identifier
    )
    storage = SQLiteIndex(tmp_path / "index.sqlite3")
    storage.replace(index)
    page = Navigation(storage).describe_symbol(FixtureSymbol.CHILD)
    symbol = next(
        symbol
        for symbol in index.symbols
        if symbol.symbol_identifier == FixtureSymbol.CHILD
    )
    source = next(
        source
        for source in index.sources
        if source.relative_path == symbol.location.relative_path
    )
    assert page.symbols == [symbol]
    assert page.excerpts[0].source_text == "\n".join(
        source.source_text.splitlines()[
            symbol.location.start_line - 1 : symbol.location.end_line
        ]
    )
    assert page.snapshot_identifier == index.manifest.snapshot_identifier


def test_find_symbols_reports_pagination(
    committed_project: CommittedProject, tmp_path: Path
) -> None:
    """
    A small page does not imply that only one declaration matched.
    """
    index = IndexBuilder(GitSource(committed_project.root), PythonSource()).build(
        committed_project.commit_identifier
    )
    storage = SQLiteIndex(tmp_path / "index.sqlite3")
    storage.replace(index)
    page = Navigation(storage).find_symbols("Same", limit=1)
    expected = sorted(
        (symbol for symbol in index.symbols if symbol.symbol_name == "Same"),
        key=lambda symbol: symbol.symbol_identifier,
    )
    assert page.symbols == expected[:1]
    assert page.total_matches == len(expected)
    assert page.next_offset == 1


def test_find_references_keeps_cross_module_call_evidence(
    committed_project: CommittedProject, tmp_path: Path
) -> None:
    """
    A caller query exposes resolved import and call relationships.
    """
    index = IndexBuilder(GitSource(committed_project.root), PythonSource()).build(
        committed_project.commit_identifier
    )
    storage = SQLiteIndex(tmp_path / "index.sqlite3")
    storage.replace(index)
    page = Navigation(storage).find_references(FixtureSymbol.HELPER)
    calls = [
        reference
        for reference in page.references
        if reference.relationship_kind == RelationshipKind.CALLS
    ]
    expected = [
        reference
        for reference in index.references
        if reference.target_identifier == FixtureSymbol.HELPER
        and reference.relationship_kind == RelationshipKind.CALLS
    ]
    assert calls == expected


def test_unknown_symbol_remains_unresolved(
    committed_project: CommittedProject, tmp_path: Path
) -> None:
    """
    No-match results do not manufacture a source location.
    """
    index = IndexBuilder(GitSource(committed_project.root), PythonSource()).build(
        committed_project.commit_identifier
    )
    storage = SQLiteIndex(tmp_path / "index.sqlite3")
    storage.replace(index)
    page = Navigation(storage).describe_symbol("missing.Declaration")
    assert page.symbols == []
    assert page.excerpts == []
    assert page.evidence_resolution == Resolution.UNRESOLVED
