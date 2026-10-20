"""
Earlier symbol-only caches must be rebuilt before source refresh.
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from repository_knowledge.adapters.sqlite_index import SQLiteIndex
from repository_knowledge.exceptions import UnsupportedIndexFormat
from repository_knowledge.model import IndexData, IndexVersion

# %% analysis-cache compatibility


def test_symbol_only_cache_is_refused(
    source_index: IndexData, indexed_storage: SQLiteIndex
) -> None:
    """
    A known earlier format cannot masquerade as reusable file analysis.
    """
    indexed_storage.replace(
        replace(
            source_index,
            manifest=replace(
                source_index.manifest, format_version=IndexVersion.SYMBOL_GRAPH
            ),
        )
    )
    with pytest.raises(UnsupportedIndexFormat):
        indexed_storage.read()
