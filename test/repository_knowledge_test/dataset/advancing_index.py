"""
A publication changing between successive reads.
"""

from __future__ import annotations

from dataclasses import dataclass

from repository_knowledge.contracts import IndexReader
from repository_knowledge.model import IndexData, IndexSelection

# %% publication interleaving


@dataclass
class AdvancingIndex(IndexReader):
    """
    Expose a replacement snapshot after the first read.
    """

    snapshots: list[IndexData]
    """The snapshots returned in publication order."""

    read_count: int = 0
    """
    The number of reads observed by the query.
    """

    def read(self, selection: IndexSelection | None = None) -> IndexData:
        selected = self.snapshots[min(self.read_count, len(self.snapshots) - 1)]
        self.read_count += 1
        return selected
