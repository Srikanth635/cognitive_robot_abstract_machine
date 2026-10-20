"""
Canonical artifacts require authenticated run identity and external digests.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from io import BytesIO
from pathlib import Path
from zipfile import ZipFile

import pytest

from repository_knowledge.adapters.digests import CanonicalDigest
from repository_knowledge.adapters.github_artifacts import GitHubArtifacts
from repository_knowledge.adapters.serialization import RecordCodec
from repository_knowledge.adapters.exceptions import InvalidIdentityEvidence
from repository_knowledge.model import IndexData
from repository_knowledge.operations.publication import KnowledgePublication
from repository_knowledge.publication import (
    PublicationPolicy,
    ControlEvidence,
    ReviewEvidence,
    PublicationMode,
    Publication,
)
from repository_knowledge.concepts import KnowledgeBundle

from .dataset.identity_transport import FixtureIdentity
from .dataset.publication_sources import CapturedKnowledge, IndependentReviews

# %% immutable archive fixtures


@pytest.fixture
def artifact_transport(
    identity_transport: FixtureIdentity,
    source_index: IndexData,
    publication_policy: PublicationPolicy,
    publication_bundle: KnowledgeBundle,
    publication_controls: ControlEvidence,
    review_evidence: ReviewEvidence,
) -> FixtureIdentity:
    """
    Provide a locally built archive and its independent provider digest.
    """
    publication = KnowledgePublication(
        publication_policy,
        CapturedKnowledge(publication_bundle),
        IndependentReviews(publication_controls, review_evidence),
        CanonicalDigest(),
    ).publish(source_index, 1, source_index.manifest.commit_identifier)
    stream = BytesIO()
    with ZipFile(stream, "w") as archive:
        archive.writestr("publication.json", RecordCodec().encode(publication))
    content = stream.getvalue()
    fixture = (Path(__file__).parent / "dataset/knowledge/artifact.json").read_text(
        encoding="utf-8"
    )
    identity_transport.documents[
        "/repos/fixture-owner/project/actions/artifacts/20"
    ] = json.loads(
        fixture.replace("$ARCHIVE_DIGEST", hashlib.sha256(content).hexdigest())
    )
    identity_transport.archives[
        "/repos/fixture-owner/project/actions/artifacts/20/zip"
    ] = content
    return identity_transport


# %% authenticated artifact refusal


def test_owner_selected_publisher_push_authenticates_the_pinned_workflow(
    artifact_transport: FixtureIdentity,
    publication_policy: PublicationPolicy,
    source_index: IndexData,
) -> None:
    """
    A dedicated publisher ref can activate publication without browser credentials.
    """
    policy = replace(
        publication_policy,
        mode=PublicationMode.AUTOMATED,
        reviewers=[],
        minimum_reviews=0,
    )
    run = artifact_transport.documents[
        "/repos/" + policy.repository + "/actions/runs/10"
    ]
    run["event"] = "push"
    run["head_branch"] = policy.publisher_branch
    downloaded = GitHubArtifacts(artifact_transport).read_publication(
        policy, 20, source_index.manifest.commit_identifier
    )
    assert downloaded.receipt.authenticated is True


def test_source_branch_push_cannot_borrow_the_publisher_identity(
    artifact_transport: FixtureIdentity,
    publication_policy: PublicationPolicy,
    source_index: IndexData,
) -> None:
    """
    Only the separately configured publisher ref may authorize push publications.
    """
    policy = replace(
        publication_policy,
        mode=PublicationMode.AUTOMATED,
        reviewers=[],
        minimum_reviews=0,
    )
    run = artifact_transport.documents[
        "/repos/" + policy.repository + "/actions/runs/10"
    ]
    run["event"] = "push"
    run["head_branch"] = policy.branch
    with pytest.raises(InvalidIdentityEvidence):
        GitHubArtifacts(artifact_transport).read_publication(
            policy, 20, source_index.manifest.commit_identifier
        )


def test_artifact_download_verifies_provider_digest_and_workflow(
    artifact_transport: FixtureIdentity,
    publication_policy: PublicationPolicy,
    source_index: IndexData,
) -> None:
    """
    A verified archive yields an external receipt and its complete payload.
    """
    downloaded = GitHubArtifacts(artifact_transport).read_publication(
        publication_policy, 20, source_index.manifest.commit_identifier
    )
    assert downloaded.receipt.authenticated is True
    assert downloaded.receipt.artifact_identifier == 20
    assert downloaded.receipt.run_identifier == 10
    assert downloaded.receipt.payload_digest == CanonicalDigest().digest(
        downloaded.publication
    )
    assert downloaded.publication.source.manifest == source_index.manifest


def test_altered_archive_is_refused_before_decoding(
    artifact_transport: FixtureIdentity,
    publication_policy: PublicationPolicy,
    source_index: IndexData,
) -> None:
    """
    An unchanged payload claim cannot authenticate changed archive bytes.
    """
    artifact_transport.archives[
        "/repos/fixture-owner/project/actions/artifacts/20/zip"
    ] = b"altered archive"
    with pytest.raises(InvalidIdentityEvidence):
        GitHubArtifacts(artifact_transport).read_publication(
            publication_policy, 20, source_index.manifest.commit_identifier
        )


def test_wrong_workflow_revision_is_refused(
    artifact_transport: FixtureIdentity,
    publication_policy: PublicationPolicy,
    source_index: IndexData,
) -> None:
    """
    Another workflow revision cannot borrow the trusted publisher identity.
    """
    artifact_transport.documents["/repos/fixture-owner/project/actions/runs/10"][
        "head_sha"
    ] = "different-workflow-revision"
    with pytest.raises(InvalidIdentityEvidence):
        GitHubArtifacts(artifact_transport).read_publication(
            publication_policy, 20, source_index.manifest.commit_identifier
        )


def test_environment_bypass_without_independent_approval_is_refused(
    artifact_transport: FixtureIdentity,
    publication_policy: PublicationPolicy,
    source_index: IndexData,
) -> None:
    """
    Workflow success alone does not prove a protected-environment review occurred.
    """
    artifact_transport.documents[
        "/repos/fixture-owner/project/actions/runs/10/approvals"
    ] = []
    with pytest.raises(InvalidIdentityEvidence):
        GitHubArtifacts(artifact_transport).read_publication(
            publication_policy, 20, source_index.manifest.commit_identifier
        )


def test_automated_artifact_needs_no_environment_approval(
    artifact_transport: FixtureIdentity,
    publication_policy: PublicationPolicy,
    source_index: IndexData,
) -> None:
    """
    An authenticated pinned workflow can distribute machine-validated evidence.
    """
    policy = replace(
        publication_policy,
        mode=PublicationMode.AUTOMATED,
        reviewers=[],
        minimum_reviews=0,
    )
    resource = "/repos/" + policy.repository + "/actions/artifacts/20"
    content = artifact_transport.archives[resource + "/zip"]
    with ZipFile(BytesIO(content)) as archive:
        publication = RecordCodec().decode(
            archive.read("publication.json").decode(), Publication
        )
    publication = replace(publication, mode=PublicationMode.AUTOMATED, review=None)
    stream = BytesIO()
    with ZipFile(stream, "w") as archive:
        archive.writestr("publication.json", RecordCodec().encode(publication))
    content = stream.getvalue()
    artifact_transport.archives[resource + "/zip"] = content
    artifact_transport.documents[resource]["digest"] = (
        "sha256:" + hashlib.sha256(content).hexdigest()
    )
    del artifact_transport.documents[
        "/repos/" + policy.repository + "/actions/runs/10/approvals"
    ]
    downloaded = GitHubArtifacts(artifact_transport).read_publication(
        policy, 20, source_index.manifest.commit_identifier
    )
    assert downloaded.receipt.authenticated is True
    assert downloaded.publication.mode == PublicationMode.AUTOMATED
    assert artifact_transport.requested_resources == [
        resource,
        "/repos/" + policy.repository + "/actions/runs/10",
        resource + "/zip",
    ]


def test_archive_refuses_additional_payloads() -> None:
    """
    Canonical archives cannot introduce additional files or extraction paths.
    """
    stream = BytesIO()
    with ZipFile(stream, "w") as archive:
        archive.writestr("../outside.json", b"untrusted")
    with pytest.raises(InvalidIdentityEvidence):
        GitHubArtifacts(FixtureIdentity({})).decode_archive(stream.getvalue())
