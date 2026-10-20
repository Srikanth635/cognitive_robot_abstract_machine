"""
Admit canonical concepts only for the exact committed client source.
"""

from __future__ import annotations

from dataclasses import dataclass

from repository_knowledge.clients import CanonicalStatus, ContextKnowledge
from repository_knowledge.concepts import KnowledgeBundle
from repository_knowledge.contracts import CanonicalReader
from repository_knowledge.model import SourceScope
from repository_knowledge.publication import (
    ConsumptionResult,
    ConsumptionState,
    TrustIssue,
)

# %% source-bound context admission


@dataclass
class CanonicalContext:
    """
    Keep authenticated concepts separate from source-only fallback.
    """

    canonical_reader: CanonicalReader
    """The independently configured artifact consumer."""

    def read(self, revision: str, source_scope: SourceScope) -> ContextKnowledge:
        """
        Refuse working or mismatched source without substituting local prose.
        """
        if source_scope == SourceScope.WORKING_TREE:
            return ContextKnowledge(
                KnowledgeBundle([]),
                CanonicalStatus(
                    ConsumptionState.SOURCE_ONLY, revision, [TrustIssue.WORKING_SOURCE]
                ),
            )
        consumed = self.canonical_reader.read(revision)
        if consumed.state == ConsumptionState.ACCEPTED:
            if (
                consumed.publication is not None
                and consumed.publication.binding.source_commit == revision
            ):
                return ContextKnowledge(
                    consumed.publication.bundle,
                    CanonicalStatus(consumed.state, revision, consumed.issues),
                )
            consumed = ConsumptionResult(
                ConsumptionState.SOURCE_ONLY, None, [TrustIssue.WRONG_SOURCE]
            )
        return ContextKnowledge(
            KnowledgeBundle([]),
            CanonicalStatus(consumed.state, revision, consumed.issues),
        )
