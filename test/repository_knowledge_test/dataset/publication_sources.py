"""
Independent evidence mimics for publication tests.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing_extensions import TYPE_CHECKING

from repository_knowledge.contracts import (
    KnowledgeReader,
    ReviewReader,
    SourceIdentityReader,
)

if TYPE_CHECKING:
    from repository_knowledge.concepts import KnowledgeBundle
    from repository_knowledge.model import IndexData
    from repository_knowledge.publication import (
        PublicationPolicy,
        ControlEvidence,
        ReviewEvidence,
        SourceIdentity,
    )

# %% captured proposal data


@dataclass
class CapturedKnowledge(KnowledgeReader):
    """
    Supply the bundle captured by the trusted document boundary.
    """

    bundle: KnowledgeBundle
    """The captured proposal bundle."""

    def read_bundle(self, index: IndexData, bundle_directory: str) -> KnowledgeBundle:
        return self.bundle


# %% independent authority


@dataclass
class IndependentReviews(ReviewReader):
    """
    Supply provider evidence independently of wiki declarations.
    """

    controls: ControlEvidence
    """The authenticated server controls."""
    review: ReviewEvidence
    """
    The authenticated review state.
    """

    review_reads: int = 0
    """
    The number of human review requests.
    """

    def read_controls(self, policy: PublicationPolicy) -> ControlEvidence:
        return self.controls

    def read_review(
        self, policy: PublicationPolicy, request_number: int
    ) -> ReviewEvidence:
        self.review_reads += 1
        return self.review


@dataclass
class AuthenticatedSource(SourceIdentityReader):
    """
    Supply source identity without requiring human approval.
    """

    identity: SourceIdentity
    """
    The authenticated source commit and tree.
    """

    def read_source(
        self, policy: PublicationPolicy, source_commit: str
    ) -> SourceIdentity:
        return self.identity
