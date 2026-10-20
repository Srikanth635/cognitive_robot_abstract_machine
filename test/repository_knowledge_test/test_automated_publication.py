"""
Owner-operated publication needs machine validation without another reviewer.
"""

from __future__ import annotations

from dataclasses import replace

from repository_knowledge.adapters.digests import CanonicalDigest
from repository_knowledge.adapters.serialization import RecordCodec
from repository_knowledge.concepts import KnowledgeBundle
from repository_knowledge.model import IndexData
from repository_knowledge.operations.publication import (
    KnowledgePublication,
    PublicationConsumer,
    PublicationGate,
)
from repository_knowledge.publication import (
    ArtifactReceipt,
    ConsumptionState,
    ControlEvidence,
    PublicationMode,
    PublicationPolicy,
    Publication,
    ReviewEvidence,
    SourceIdentity,
    TrustIssue,
)

from .dataset.publication_sources import (
    CapturedKnowledge,
    IndependentReviews,
    AuthenticatedSource,
)

# %% automated owner-operated mode


def test_automated_publication_round_trips_authenticated_source(
    source_index: IndexData,
    publication_policy: PublicationPolicy,
    publication_bundle: KnowledgeBundle,
    publication_controls: ControlEvidence,
    review_evidence: ReviewEvidence,
) -> None:
    """
    Serialization preserves the exact source identity and absence of human review.
    """
    policy = replace(
        publication_policy,
        mode=PublicationMode.AUTOMATED,
        reviewers=[],
        minimum_reviews=0,
    )
    identity = SourceIdentity(
        policy.repository,
        policy.branch,
        source_index.manifest.commit_identifier,
        source_index.manifest.tree_identifier,
        "proposal-author",
        True,
    )
    publication = KnowledgePublication(
        policy,
        CapturedKnowledge(publication_bundle),
        IndependentReviews(publication_controls, review_evidence),
        CanonicalDigest(),
        AuthenticatedSource(identity),
    ).publish(source_index, 0, source_index.manifest.commit_identifier)
    codec = RecordCodec()
    assert codec.decode(codec.encode(publication), Publication) == publication


def test_automated_publication_needs_no_pull_request_or_human_approval(
    source_index: IndexData,
    publication_policy: PublicationPolicy,
    publication_bundle: KnowledgeBundle,
    publication_controls: ControlEvidence,
    review_evidence: ReviewEvidence,
) -> None:
    """
    Machine admission remains distinct from optional independent human review.
    """
    policy = replace(
        publication_policy,
        mode=PublicationMode.AUTOMATED,
        reviewers=[],
        minimum_reviews=0,
    )
    controls = replace(
        publication_controls,
        branch_protected=False,
        owners_required=False,
        administrators_enforced=False,
        last_push_approval=False,
        stale_reviews_dismissed=False,
        required_reviews=0,
        required_checks=[],
        environment_protected=False,
    )
    reviews = IndependentReviews(
        controls, replace(review_evidence, decisions=[], merged=False)
    )
    identity = SourceIdentity(
        policy.repository,
        policy.branch,
        source_index.manifest.commit_identifier,
        source_index.manifest.tree_identifier,
        "proposal-author",
        True,
    )
    source = AuthenticatedSource(identity)
    digest = CanonicalDigest()
    publication = KnowledgePublication(
        policy, CapturedKnowledge(publication_bundle), reviews, digest, source
    ).publish(source_index, 0, source_index.manifest.commit_identifier)
    assert publication.mode == PublicationMode.AUTOMATED
    assert publication.review is None
    assert publication.validation.concepts[0].authenticated_review is False
    assert reviews.review_reads == 0
    receipt = ArtifactReceipt(
        policy.repository,
        1,
        1,
        policy.workflow_path,
        publication.checker_revision,
        digest.digest(publication),
        True,
    )
    result = PublicationConsumer(
        policy, reviews, digest, publication.checker_revision, source
    ).consume(publication, receipt, identity.source_commit)
    assert result.state == ConsumptionState.ACCEPTED
    assert reviews.review_reads == 0


def test_automated_publication_still_refuses_the_wrong_source_tree(
    source_index: IndexData,
    publication_policy: PublicationPolicy,
    publication_controls: ControlEvidence,
    review_evidence: ReviewEvidence,
) -> None:
    """
    Removing human approval does not remove exact source identity checks.
    """
    policy = replace(
        publication_policy,
        mode=PublicationMode.AUTOMATED,
        reviewers=[],
        minimum_reviews=0,
    )
    identity = SourceIdentity(
        policy.repository,
        policy.branch,
        source_index.manifest.commit_identifier,
        "different-tree",
        "proposal-author",
        True,
    )
    assert PublicationGate().check_source(
        policy, source_index, publication_controls, identity
    ).issues == [TrustIssue.WRONG_SOURCE]


def test_consumer_refuses_invented_human_review_on_automated_publication(
    source_index: IndexData,
    publication_policy: PublicationPolicy,
    publication_bundle: KnowledgeBundle,
    publication_controls: ControlEvidence,
    review_evidence: ReviewEvidence,
) -> None:
    """
    A valid artifact cannot turn automated checks into human review.
    """
    policy = replace(
        publication_policy,
        mode=PublicationMode.AUTOMATED,
        reviewers=[],
        minimum_reviews=0,
    )
    reviews = IndependentReviews(publication_controls, review_evidence)
    identity = SourceIdentity(
        policy.repository,
        policy.branch,
        source_index.manifest.commit_identifier,
        source_index.manifest.tree_identifier,
        "proposal-author",
        True,
    )
    source = AuthenticatedSource(identity)
    digest = CanonicalDigest()
    publication = KnowledgePublication(
        policy, CapturedKnowledge(publication_bundle), reviews, digest, source
    ).publish(source_index, 0, source_index.manifest.commit_identifier)
    invented = replace(
        publication.validation,
        concepts=[
            replace(item, authenticated_review=True)
            for item in publication.validation.concepts
        ],
    )
    publication = replace(publication, validation=invented)
    receipt = ArtifactReceipt(
        policy.repository,
        1,
        1,
        policy.workflow_path,
        publication.checker_revision,
        digest.digest(publication),
        True,
    )
    result = PublicationConsumer(
        policy, reviews, digest, publication.checker_revision, source
    ).consume(publication, receipt, identity.source_commit)
    assert result.issues == [TrustIssue.INVALID_KNOWLEDGE]
