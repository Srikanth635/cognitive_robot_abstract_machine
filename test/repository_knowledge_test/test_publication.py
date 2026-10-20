"""
Publication authority comes from independent, content-bound evidence.
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from repository_knowledge.adapters.digests import CanonicalDigest
from repository_knowledge.concepts import KnowledgeBundle
from repository_knowledge.model import IndexData, SourceScope
from repository_knowledge.operations.publication import (
    PublicationGate,
    KnowledgePublication,
    PublicationConsumer,
)
from repository_knowledge.operations.exceptions import PublicationDenied
from repository_knowledge.publication import (
    PublicationPolicy,
    ControlEvidence,
    ReviewEvidence,
    ArtifactReceipt,
    TrustIssue,
    ConsumptionState,
)

from .dataset.publication_sources import CapturedKnowledge, IndependentReviews

# %% immutable review binding


def test_publication_binds_bundle_source_policy_and_independent_review(
    source_index: IndexData,
    publication_policy: PublicationPolicy,
    publication_bundle: KnowledgeBundle,
    publication_controls: ControlEvidence,
    review_evidence: ReviewEvidence,
) -> None:
    """
    A canonical payload retains complete identities without rewriting draft prose.
    """
    publisher = KnowledgePublication(
        publication_policy,
        CapturedKnowledge(publication_bundle),
        IndependentReviews(publication_controls, review_evidence),
        CanonicalDigest(),
    )
    publication = publisher.publish(
        source_index,
        review_evidence.request_number,
        source_index.manifest.commit_identifier,
    )
    assert publication.bundle == publication_bundle
    assert publication.source.manifest == source_index.manifest
    assert publication.binding.bundle_digest == CanonicalDigest().digest(
        publication_bundle
    )
    assert publication.binding.policy_digest == CanonicalDigest().digest(
        publication_policy
    )
    assert publication.review == review_evidence
    assert publication.validation.concepts[0].authenticated_review is True
    assert publication.source.analyses == []
    assert publication.source.references == []


def test_frontmatter_review_cannot_replace_authenticated_evidence(
    source_index: IndexData,
    publication_policy: PublicationPolicy,
    publication_bundle: KnowledgeBundle,
    publication_controls: ControlEvidence,
    review_evidence: ReviewEvidence,
) -> None:
    """
    Producer verification strings do not satisfy an empty review record.
    """
    review = replace(review_evidence, decisions=[])
    report = PublicationGate().check(
        publication_policy, source_index, publication_controls, review
    )
    assert report.issues == [TrustIssue.MISSING_REVIEW]


def test_self_review_is_refused(
    source_index: IndexData,
    publication_policy: PublicationPolicy,
    publication_controls: ControlEvidence,
    review_evidence: ReviewEvidence,
) -> None:
    """
    The proposal author cannot supply their own publication approval.
    """
    review = replace(review_evidence, author_login=publication_policy.reviewers[0])
    assert PublicationGate().check(
        publication_policy, source_index, publication_controls, review
    ).issues == [TrustIssue.SELF_REVIEW]


def test_approval_of_an_old_commit_is_refused(
    source_index: IndexData,
    publication_policy: PublicationPolicy,
    publication_controls: ControlEvidence,
    review_evidence: ReviewEvidence,
) -> None:
    """
    An earlier approval cannot follow a changed proposal head.
    """
    review = replace(
        review_evidence,
        decisions=[
            replace(review_evidence.decisions[0], commit_identifier="older-head")
        ],
    )
    assert PublicationGate().check(
        publication_policy, source_index, publication_controls, review
    ).issues == [TrustIssue.STALE_REVIEW]


def test_changed_merge_tree_loses_approval(
    source_index: IndexData,
    publication_policy: PublicationPolicy,
    publication_controls: ControlEvidence,
    review_evidence: ReviewEvidence,
) -> None:
    """
    Conflict resolution cannot silently inherit the pre-merge content approval.
    """
    review = replace(review_evidence, reviewed_tree="different-reviewed-tree")
    assert PublicationGate().check(
        publication_policy, source_index, publication_controls, review
    ).issues == [TrustIssue.CHANGED_CONTENT]


def test_later_changes_requested_supersede_an_approval(
    source_index: IndexData,
    publication_policy: PublicationPolicy,
    publication_controls: ControlEvidence,
    review_evidence: ReviewEvidence,
) -> None:
    """
    Only the latest decisive state from each reviewer can authorize content.
    """
    review = replace(
        review_evidence,
        decisions=[
            *review_evidence.decisions,
            replace(
                review_evidence.decisions[0],
                review_identifier=2,
                state="CHANGES_REQUESTED",
            ),
        ],
    )
    assert PublicationGate().check(
        publication_policy, source_index, publication_controls, review
    ).issues == [TrustIssue.MISSING_REVIEW]


def test_unverified_server_controls_refuse_publication(
    source_index: IndexData,
    publication_policy: PublicationPolicy,
    publication_controls: ControlEvidence,
    review_evidence: ReviewEvidence,
) -> None:
    """
    A CODEOWNERS declaration does not establish enforced branch protection.
    """
    controls = replace(publication_controls, branch_protected=False)
    assert PublicationGate().check(
        publication_policy, source_index, controls, review_evidence
    ).issues == [TrustIssue.UNVERIFIED_CONTROLS]


def test_working_source_cannot_be_canonical(
    source_index: IndexData,
    publication_policy: PublicationPolicy,
    publication_controls: ControlEvidence,
    review_evidence: ReviewEvidence,
) -> None:
    """
    Uncommitted snapshots cannot inherit approvals for committed source.
    """
    index = replace(
        source_index,
        manifest=replace(source_index.manifest, source_scope=SourceScope.WORKING_TREE),
    )
    assert PublicationGate().check(
        publication_policy, index, publication_controls, review_evidence
    ).issues == [TrustIssue.WORKING_SOURCE]


# %% artifact consumption and rollback


def test_consumer_refuses_altered_payload_and_wrong_revision(
    source_index: IndexData,
    publication_policy: PublicationPolicy,
    publication_bundle: KnowledgeBundle,
    publication_controls: ControlEvidence,
    review_evidence: ReviewEvidence,
) -> None:
    """
    Authenticated artifact identity does not excuse mismatched contents or source.
    """
    digest = CanonicalDigest()
    reviews = IndependentReviews(publication_controls, review_evidence)
    publication = KnowledgePublication(
        publication_policy, CapturedKnowledge(publication_bundle), reviews, digest
    ).publish(source_index, 1, source_index.manifest.commit_identifier)
    receipt = ArtifactReceipt(
        publication_policy.repository,
        1,
        1,
        publication_policy.workflow_path,
        publication.checker_revision,
        digest.digest(publication),
        True,
    )
    consumer = PublicationConsumer(
        publication_policy, reviews, digest, publication.checker_revision
    )
    altered = replace(
        publication,
        bundle=KnowledgeBundle(
            [replace(publication_bundle.concepts[0], body="Changed claim")]
        ),
    )
    refused = consumer.consume(
        altered, receipt, source_index.manifest.commit_identifier
    )
    assert refused.state == ConsumptionState.SOURCE_ONLY
    assert refused.publication is None
    assert refused.issues == [TrustIssue.ARTIFACT_DIGEST]
    wrong_revision = consumer.consume(publication, receipt, "another-source")
    assert wrong_revision.issues == [TrustIssue.WRONG_SOURCE]


def test_rollback_restores_a_complete_known_publication(
    source_index: IndexData,
    publication_policy: PublicationPolicy,
    publication_bundle: KnowledgeBundle,
    publication_controls: ControlEvidence,
    review_evidence: ReviewEvidence,
) -> None:
    """
    Rollback selects both historical code evidence and its exact knowledge bundle.
    """
    digest = CanonicalDigest()
    reviews = IndependentReviews(publication_controls, review_evidence)
    publication = KnowledgePublication(
        publication_policy, CapturedKnowledge(publication_bundle), reviews, digest
    ).publish(source_index, 1, source_index.manifest.commit_identifier)
    receipt = ArtifactReceipt(
        publication_policy.repository,
        1,
        1,
        publication_policy.workflow_path,
        publication.checker_revision,
        digest.digest(publication),
        True,
    )
    restored = PublicationConsumer(
        publication_policy, reviews, digest, publication.checker_revision
    ).consume(publication, receipt, publication.source.manifest.commit_identifier)
    assert restored.state == ConsumptionState.ACCEPTED
    assert restored.publication == publication
    assert restored.publication.source.sources == publication.source.sources


def test_missing_reviewer_configuration_keeps_publication_disabled(
    source_index: IndexData,
    publication_policy: PublicationPolicy,
    publication_bundle: KnowledgeBundle,
    publication_controls: ControlEvidence,
    review_evidence: ReviewEvidence,
) -> None:
    """
    An empty authority configuration is a visible refusal, not a permissive default.
    """
    publisher = KnowledgePublication(
        replace(publication_policy, reviewers=[]),
        CapturedKnowledge(publication_bundle),
        IndependentReviews(publication_controls, review_evidence),
        CanonicalDigest(),
    )
    with pytest.raises(PublicationDenied):
        publisher.publish(source_index, 1, source_index.manifest.commit_identifier)
