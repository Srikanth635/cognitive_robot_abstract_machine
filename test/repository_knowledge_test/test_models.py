"""
Contracts for source locations and bounded query evidence.
"""

from __future__ import annotations

import pytest

from repository_knowledge.model import (
    QueryPage,
    Resolution,
    SourceLocation,
    SymbolKind,
    SymbolRecord,
)
from repository_knowledge.exceptions import InvalidSourceLocation

# %% source location invariants


def test_source_location_rejects_reversed_lines() -> None:
    """
    A source location cannot end before it begins.
    """
    with pytest.raises(InvalidSourceLocation):
        SourceLocation("example.py", 9, 2)


def test_source_location_rejects_parent_traversal() -> None:
    """
    Locations remain relative to their source snapshot.
    """
    with pytest.raises(InvalidSourceLocation):
        SourceLocation("../example.py", 1, 2)


def test_qualified_symbol_identity_distinguishes_same_named_classes() -> None:
    """
    The module and lexical scope participate in a symbol's identity.
    """
    first = SymbolRecord(
        "first.api.Same",
        "Same",
        "first.api",
        SymbolKind.CLASS,
        SourceLocation("first/api.py", 1, 2),
    )
    second = SymbolRecord(
        "second.api.Same",
        "Same",
        "second.api",
        SymbolKind.CLASS,
        SourceLocation("second/api.py", 1, 2),
    )
    assert first.symbol_identifier != second.symbol_identifier


def test_query_page_reports_omitted_results() -> None:
    """
    A bounded result reports how much evidence was omitted.
    """
    page = QueryPage("snapshot", [], [], [], [], 7, 2, Resolution.RESOLVED)
    assert page.total_matches == 7
    assert page.next_offset == 2
