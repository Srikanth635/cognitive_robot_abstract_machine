"""
Refresh source analysis while conservatively resolving current graph targets.
"""

from __future__ import annotations

from dataclasses import dataclass

from repository_knowledge.contracts import IndexReader, IndexWriter
from repository_knowledge.exceptions import SourceScopeMismatch
from repository_knowledge.model import (
    IndexSelection,
    RefreshReport,
    SelectionKind,
    SourceKind,
)
from repository_knowledge.operations.freshness import FreshnessCheck
from repository_knowledge.operations.indexing import IndexBuilder
from repository_knowledge.operations.queries import IndexContents

# %% source refresh and reconciliation


@dataclass
class IndexRefresh:
    """
    Publish coherent current source with reusable per-file extraction.
    """

    index_builder: IndexBuilder
    """The source capture and extraction operation."""

    index_reader: IndexReader
    """
    The completed snapshot supplying previous file analysis.
    """

    index_writer: IndexWriter
    """The atomic snapshot publication boundary."""

    def refresh(self, revision: str, reconcile: bool = False) -> RefreshReport:
        """
        Refresh changed file analysis or reconcile with a clean rebuild.
        """
        saved = self.index_reader.read(IndexSelection(SelectionKind.MANIFEST))
        scope = self.index_builder.source_reader.source_scope
        if saved.manifest.source_scope != scope:
            raise SourceScopeMismatch(saved.manifest.source_scope, scope)
        snapshot = self.index_builder.source_reader.read_snapshot(revision)
        if (
            not reconcile
            and saved.manifest.snapshot_identifier
            == snapshot.manifest.snapshot_identifier
        ):
            FreshnessCheck(
                IndexContents(saved), self.index_builder.source_reader
            ).check(revision).require_current()
            reused = [
                source.relative_path
                for source in snapshot.sources
                if source.source_kind in (SourceKind.PYTHON, SourceKind.DOCUMENT)
            ]
            return RefreshReport(saved.manifest, [], reused, [], False)
        previous = self.index_reader.read(IndexSelection(SelectionKind.ANALYSES))
        if previous.manifest.source_scope != scope:
            raise SourceScopeMismatch(previous.manifest.source_scope, scope)
        index = self.index_builder.build(revision, None if reconcile else previous)
        FreshnessCheck(IndexContents(index), self.index_builder.source_reader).check(
            revision
        ).require_current()
        previous_analyses = {
            analysis.identity: analysis for analysis in previous.analyses
        }
        reused_paths = [
            analysis.identity.relative_path
            for analysis in index.analyses
            if previous_analyses.get(analysis.identity) is analysis
        ]
        reused_path_set = set(reused_paths)
        analyzed_paths = [
            analysis.identity.relative_path
            for analysis in index.analyses
            if analysis.identity.relative_path not in reused_path_set
        ]
        removed_paths = sorted(
            {source.relative_path for source in previous.sources}
            - {source.relative_path for source in index.sources}
        )
        published = (
            reconcile
            or index.manifest.snapshot_identifier
            != previous.manifest.snapshot_identifier
        )
        if published:
            self.index_writer.replace(index)
        FreshnessCheck(IndexContents(index), self.index_builder.source_reader).check(
            revision
        ).require_current()
        return RefreshReport(
            index.manifest, analyzed_paths, reused_paths, removed_paths, published
        )
