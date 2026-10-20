"""
Identity APIs remain authenticated boundaries rather than frontmatter input.
"""

from __future__ import annotations

from dataclasses import replace
from urllib.request import Request

import pytest

from repository_knowledge.adapters.github_identity import (
    GitHubReviews,
    CredentialRedirect,
)
from repository_knowledge.adapters.exceptions import InvalidIdentityEvidence
from repository_knowledge.model import IndexData
from repository_knowledge.publication import (
    PublicationPolicy,
    ReviewEvidence,
    ControlEvidence,
    PublicationMode,
    SourceIdentity,
)

from .dataset.identity_transport import FixtureIdentity

# %% mocked provider evidence


def test_github_reviews_read_exact_source_and_current_permissions(
    identity_transport: FixtureIdentity,
    publication_policy: PublicationPolicy,
    review_evidence: ReviewEvidence,
) -> None:
    """
    Provider review identity and merged-tree evidence are read independently.
    """
    assert (
        GitHubReviews(identity_transport).read_review(publication_policy, 1)
        == review_evidence
    )


def test_github_controls_require_enforced_protection_and_environment(
    identity_transport: FixtureIdentity,
    publication_policy: PublicationPolicy,
    publication_controls: ControlEvidence,
) -> None:
    """
    Branch and environment enforcement are read rather than inferred from files.
    """
    assert (
        GitHubReviews(identity_transport).read_controls(publication_policy)
        == publication_controls
    )


def test_github_controls_cannot_treat_unknown_protection_as_enforced(
    identity_transport: FixtureIdentity, publication_policy: PublicationPolicy
) -> None:
    """
    An absent protection response fails closed.
    """
    identity_transport.documents[
        "/repos/fixture-owner/project/branches/main/protection"
    ] = {}
    with pytest.raises(InvalidIdentityEvidence):
        GitHubReviews(identity_transport).read_controls(publication_policy)


def test_artifact_redirect_does_not_forward_credentials() -> None:
    """
    A signed artifact download URL receives no API authorization header.
    """
    request = Request(
        "https://api.github.com/artifact",
        headers={"Authorization": "Bearer fixture-only"},
    )
    redirected = CredentialRedirect().redirect_request(
        request, None, 302, "Found", {}, "https://example.invalid/artifact.zip"
    )
    assert redirected.get_header("Authorization") is None


def test_automated_provider_needs_no_administrative_or_review_access(
    identity_transport: FixtureIdentity,
    publication_policy: PublicationPolicy,
    source_index: IndexData,
) -> None:
    """
    Read-only source identity is sufficient in automated mode.
    """
    policy = replace(
        publication_policy,
        mode=PublicationMode.AUTOMATED,
        reviewers=[],
        minimum_reviews=0,
    )
    provider = GitHubReviews(identity_transport)
    controls = provider.read_controls(policy)
    identity = provider.read_source(policy, source_index.manifest.commit_identifier)
    assert controls.required_reviews == 0
    assert identity == SourceIdentity(
        policy.repository,
        policy.branch,
        source_index.manifest.commit_identifier,
        source_index.manifest.tree_identifier,
        "",
        True,
    )
    assert identity_transport.requested_resources == [
        "/repos/" + policy.repository,
        "/repos/" + policy.repository + "/branches/" + policy.branch,
        "/repos/" + policy.repository + "/commits/" + identity.source_commit,
        "/repos/"
        + policy.repository
        + "/compare/"
        + identity.source_commit
        + "..."
        + policy.branch,
    ]


def test_automated_source_must_belong_to_configured_branch(
    identity_transport: FixtureIdentity,
    publication_policy: PublicationPolicy,
    source_index: IndexData,
) -> None:
    """
    A commit outside the configured source history cannot be published.
    """
    policy = replace(
        publication_policy,
        mode=PublicationMode.AUTOMATED,
        reviewers=[],
        minimum_reviews=0,
    )
    identity_transport.documents[
        "/repos/"
        + policy.repository
        + "/compare/"
        + source_index.manifest.commit_identifier
        + "..."
        + policy.branch
    ]["status"] = "diverged"
    with pytest.raises(InvalidIdentityEvidence):
        GitHubReviews(identity_transport).read_source(
            policy, source_index.manifest.commit_identifier
        )
