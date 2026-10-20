"""
Compare a committed index with the revision requested by its consumer.
"""

from __future__ import annotations

from dataclasses import dataclass

from repository_knowledge.contracts import IndexReader, SnapshotReader
from repository_knowledge.exceptions import SourceScopeMismatch
from repository_knowledge.model import (
    FreshnessReport,
    IndexSelection,
    SelectionKind,
    SourceScope,
)

# %% committed revision compatibility


@dataclass
class FreshnessCheck:
    """
    Compare saved evidence with the selected committed or working-file source.
    """

    index_reader: IndexReader
    """The saved committed-source index."""

    source_reader: SnapshotReader
    """
    The repository resolving the requested revision.
    """

    def check(self, revision: str) -> FreshnessReport:
        """
        Compare the requested source identity with the saved index.
        """
        index = self.index_reader.read(IndexSelection(SelectionKind.MANIFEST))
        if index.manifest.source_scope != self.source_reader.source_scope:
            raise SourceScopeMismatch(
                index.manifest.source_scope, self.source_reader.source_scope
            )
        requested_commit = self.source_reader.resolve_revision(revision)
        requested_snapshot = ""
        matches_source = True
        if self.source_reader.source_scope == SourceScope.WORKING_TREE:
            requested_snapshot = self.source_reader.read_snapshot(
                revision
            ).manifest.snapshot_identifier
            matches_source = index.manifest.snapshot_identifier == requested_snapshot
        return FreshnessReport(
            index.manifest.snapshot_identifier,
            index.manifest.commit_identifier,
            requested_commit,
            index.manifest.commit_identifier == requested_commit,
            index.manifest.source_scope,
            matches_source,
            requested_snapshot,
        )
