"""
A pinned source snapshot for operation tests without persistence.
"""

from __future__ import annotations

from dataclasses import dataclass

from repository_knowledge.contracts import IndexReader
from repository_knowledge.model import IndexData, IndexSelection

# %% pinned snapshot reader


@dataclass
class PinnedIndex(IndexReader):
    """
    Return the same completed snapshot throughout a query.
    """

    index: IndexData
    """The source evidence supplied to the operation."""

    def read(self, selection: IndexSelection | None = None) -> IndexData:
        return self.index
