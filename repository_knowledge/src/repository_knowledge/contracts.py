"""
Small source and persistence interfaces owned by indexing operations.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from typing_extensions import TYPE_CHECKING

if TYPE_CHECKING:
    from repository_knowledge.assistance import TestCatalog
    from repository_knowledge.model import (
        FileExtraction,
        IndexData,
        IndexSelection,
        RepositorySnapshot,
        SourceRecord,
        SourceScope,
    )

# %% source boundaries


@dataclass
class SnapshotReader(ABC):
    """
    Capture source without executing indexed modules.
    """

    @property
    @abstractmethod
    def source_scope(self) -> SourceScope:
        """
        Identify whether capture covers committed objects or working files.
        """

    @abstractmethod
    def read_snapshot(self, revision: str) -> RepositorySnapshot:
        """
        Capture source and exclusions using the requested committed baseline.
        """

    @abstractmethod
    def resolve_revision(self, revision: str) -> str:
        """
        Return the exact commit identified by the revision selector.
        """


@dataclass
class SourceAnalyzer(ABC):
    """
    Extract structural evidence without executing the source.
    """

    @abstractmethod
    def extract(self, source: SourceRecord) -> FileExtraction:
        """
        Return declarations, bindings and visible analysis limitations.
        """


@dataclass
class TestAnalyzer(ABC):
    """
    Recognize tests and fixtures without executing repository code.
    """

    @abstractmethod
    def discover(self, index: IndexData) -> TestCatalog:
        """
        Return static test declarations from the captured source graph.
        """


# %% persistence boundaries


@dataclass
class IndexReader(ABC):
    """
    Supply a completed persisted index.
    """

    @abstractmethod
    def read(self, selection: IndexSelection | None = None) -> IndexData:
        """
        Return coherent evidence containing at least the requested selection.

        With no selection, return the complete source graph.
        """


@dataclass
class IndexWriter(ABC):
    """
    Publish complete structural snapshots.
    """

    @abstractmethod
    def replace(self, index: IndexData) -> None:
        """
        Atomically replace the current index with the supplied snapshot.
        """
