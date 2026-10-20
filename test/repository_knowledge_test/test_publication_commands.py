"""
Publication commands keep trusted checker inputs separate from proposal data.
"""

from __future__ import annotations

from pathlib import Path
import sys

import pytest

from repository_knowledge.adapters.publication_policy import PolicyDocument
from repository_knowledge.interfaces.publishing import (
    PublicationParser,
    PublicationCommand,
)
from repository_knowledge.adapters.exceptions import InvalidIdentityEvidence
from repository_knowledge.publication import PublicationPolicy, PublicationMode
from repository_knowledge.composition import publication_main

# %% policy interpretation


def test_policy_document_uses_the_declared_trust_requirements(
    publication_policy: PublicationPolicy,
) -> None:
    """
    Trusted configuration uses the shared policy defaults and explicit authorities.
    """
    path = Path(__file__).parent / "dataset/knowledge/publication.toml"
    assert PolicyDocument(path).read() == publication_policy


def test_policy_rejects_unknown_authority_fields() -> None:
    """
    A misspelled field cannot silently disable a publication restriction.
    """
    with pytest.raises(InvalidIdentityEvidence):
        PolicyDocument.from_json(
            {
                "repository": "fixture-owner/project",
                "branch": "main",
                "reviewers": [],
                "allow_self_review": True,
            }
        )


# %% explicit publication identities


def test_publish_arguments_pin_source_and_trusted_checker(tmp_path: Path) -> None:
    """
    The source revision and checker revision are distinct explicit inputs.
    """
    arguments = PublicationParser().parse(
        [
            "--repository",
            str(tmp_path),
            "--trusted-repository",
            str(tmp_path),
            "--checker-revision",
            "a" * 40,
            "publish",
            "--revision",
            "b" * 40,
            "--request",
            "1",
            "--output",
            str(tmp_path / "publication"),
        ]
    )
    assert arguments.command == PublicationCommand.PUBLISH
    assert arguments.checker_revision == "a" * 40
    assert arguments.revision == "b" * 40
    assert arguments.request_number == 1


def test_consumer_requires_immutable_artifact_identity(tmp_path: Path) -> None:
    """
    Consumers cannot select mutable latest artifacts as canonical evidence.
    """
    arguments = PublicationParser().parse(
        [
            "--repository",
            str(tmp_path),
            "--trusted-repository",
            str(tmp_path),
            "--checker-revision",
            "a" * 40,
            "consume",
            "--revision",
            "b" * 40,
            "--artifact-id",
            "20",
        ]
    )
    assert arguments.command == PublicationCommand.CONSUME
    assert arguments.artifact_identifier == 20


def test_automated_policy_and_command_need_no_reviewer_or_request(
    tmp_path: Path,
) -> None:
    """
    The fork's default policy permits an owner-operated machine publication.
    """
    policy = PolicyDocument(
        Path(__file__).parents[2] / "repository_knowledge/policy/publication.toml"
    ).read()
    arguments = PublicationParser().parse(
        [
            "--repository",
            str(tmp_path),
            "--trusted-repository",
            str(tmp_path),
            "--checker-revision",
            "a" * 40,
            "publish",
            "--revision",
            "b" * 40,
            "--output",
            str(tmp_path / "publication"),
        ]
    )
    assert policy.mode == PublicationMode.AUTOMATED
    assert policy.reviewers == []
    assert policy.minimum_reviews == 0
    assert arguments.request_number == 0


def test_policy_is_not_opened_before_checker_custody_is_verified(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    An unauthorized policy path cannot be read before the trust-root check.
    """
    root = Path(__file__).parents[2]
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "cram-publication",
            "--repository",
            str(root),
            "--trusted-repository",
            str(root),
            "--checker-revision",
            "a" * 40,
            "--policy",
            str(tmp_path / "outside-policy.toml"),
            "fingerprint",
            "--revision",
            "b" * 40,
        ],
    )
    with pytest.raises(InvalidIdentityEvidence):
        publication_main()
