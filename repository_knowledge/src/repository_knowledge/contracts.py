"""
Small source and persistence interfaces owned by indexing operations.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from typing_extensions import TYPE_CHECKING

if TYPE_CHECKING:
    from repository_knowledge.publication import (
        PublicationPolicy,
        ControlEvidence,
        ReviewEvidence,
        ArtifactDownload,
        SourceIdentity,
    )
    from repository_knowledge.concepts import KnowledgeBundle
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


# %% independent publication boundaries


@dataclass
class RecordDigester(ABC):
    """
    Bind owned records without coupling operations to their wire encoding.
    """

    @abstractmethod
    def digest(self, record: object) -> str:
        """
        Return the canonical digest of an owned record.
        """

    @abstractmethod
    def digest_records(self, records: list[object]) -> str:
        """
        Return an order-sensitive digest of a framed record sequence.
        """


@dataclass
class KnowledgeReader(ABC):
    """
    Derive a bundle only from the selected committed source capture.
    """

    @abstractmethod
    def read_bundle(self, index: IndexData, bundle_directory: str) -> KnowledgeBundle:
        """
        Decode captured bundle documents without using the working filesystem.
        """


@dataclass
class ReviewReader(ABC):
    """
    Obtain authenticated review and enforcement evidence independently.
    """

    @abstractmethod
    def read_controls(self, policy: PublicationPolicy) -> ControlEvidence:
        """
        Read current repository and environment enforcement.
        """

    @abstractmethod
    def read_review(
        self, policy: PublicationPolicy, request_number: int
    ) -> ReviewEvidence:
        """
        Read the authenticated review history and merged content identity.
        """


@dataclass
class ArtifactReader(ABC):
    """
    Obtain publications through an authenticated immutable artifact identity.
    """

    @abstractmethod
    def read_publication(
        self, policy: PublicationPolicy, artifact_identifier: int, checker_revision: str
    ) -> ArtifactDownload:
        """
        Verify the provider archive and workflow before decoding its publication.
        """


@dataclass
class SourceIdentityReader(ABC):
    """
    Authenticate committed source independently of wiki declarations.
    """

    @abstractmethod
    def read_source(
        self, policy: PublicationPolicy, source_commit: str
    ) -> SourceIdentity:
        """
        Read the exact provider source commit and tree.
        """
