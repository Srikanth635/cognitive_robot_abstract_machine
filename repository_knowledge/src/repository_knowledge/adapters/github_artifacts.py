"""Authenticate immutable workflow artifacts before decoding canonical knowledge."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from enum import StrEnum
from io import BytesIO
from zipfile import ZipFile
from typing_extensions import TYPE_CHECKING

from repository_knowledge.adapters.digests import CanonicalDigest
from repository_knowledge.adapters.exceptions import InvalidIdentityEvidence
from repository_knowledge.adapters.github_identity import (
    ApiField,
    ProviderObject,
    ProviderValue,
)
from repository_knowledge.adapters.serialization import RecordCodec
from repository_knowledge.contracts import ArtifactReader
from repository_knowledge.publication import (
    ArtifactDownload,
    ArtifactReceipt,
    IdentityKind,
    Publication,
    PublicationMode,
)

if TYPE_CHECKING:
    from repository_knowledge.adapters.github_identity import IdentityTransport
    from repository_knowledge.publication import PublicationPolicy

# %% immutable artifact identities


class PublicationFilename(StrEnum):
    """The single payload document allowed in a canonical artifact archive."""

    PAYLOAD = "publication.json"


@dataclass(frozen=True)
class ProviderArtifact:
    """The provider's immutable artifact metadata and external archive digest."""

    artifact_identifier: int
    """The provider artifact identity."""
    artifact_name: str
    """The provider artifact name."""
    run_identifier: int
    """The workflow run that uploaded the artifact."""
    archive_digest: str
    """The provider's SHA-256 archive digest."""
    expired: bool
    """Whether the archive is no longer available."""

    @classmethod
    def from_json(cls, value: object) -> ProviderArtifact:
        """Require explicit identity, lifetime and digest fields from the provider."""
        data = ProviderObject.from_json(value)
        return cls(
            data.integer(ApiField.IDENTIFIER),
            data.text(ApiField.NAME),
            data.nested(ApiField.WORKFLOW_RUN).integer(ApiField.IDENTIFIER),
            data.text(ApiField.DIGEST),
            data.boolean(ApiField.EXPIRED),
        )


@dataclass(frozen=True)
class WorkflowIdentity:
    """Authenticated execution identity of a completed publisher run."""

    run_identifier: int
    """The provider workflow run identity."""
    repository: str
    """The repository containing the run."""
    head_repository: str
    """The repository supplying executed workflow code."""
    workflow_path: str
    """The executed workflow path."""
    head_commit: str
    """The immutable executed workflow revision."""
    event: str
    """The workflow trigger category."""
    conclusion: str
    """The completed run result."""
    status: str
    """The provider execution status."""
    head_branch: str = ""
    """The ref supplying executable workflow code."""

    @classmethod
    def from_json(cls, value: object) -> WorkflowIdentity:
        """Read workflow identity independently of the artifact's claimed contents."""
        data = ProviderObject.from_json(value)
        return cls(
            data.integer(ApiField.IDENTIFIER),
            data.nested(ApiField.REPOSITORY).text(ApiField.FULL_NAME),
            data.nested(ApiField.HEAD_REPOSITORY).text(ApiField.FULL_NAME),
            data.text(ApiField.PATH),
            data.text(ApiField.HEAD_SHA),
            data.text(ApiField.EVENT),
            data.text(ApiField.CONCLUSION),
            data.text(ApiField.STATUS),
            (
                data.text(ApiField.HEAD_BRANCH)
                if ApiField.HEAD_BRANCH in data.contents
                else ""
            ),
        )


# %% authenticated archive boundary


@dataclass
class GitHubArtifacts(ArtifactReader):
    """Verify provider identity, deployment approval and archive bytes before use."""

    transport: IdentityTransport
    """The independent provider transport."""
    codec: RecordCodec = field(default_factory=RecordCodec)
    """The restricted owned-record decoder."""
    maximum_payload_bytes: int = 67108864
    """The maximum decompressed publication size."""

    def read_publication(
        self, policy: PublicationPolicy, artifact_identifier: int, checker_revision: str
    ) -> ArtifactDownload:
        root = "/repos/" + policy.repository
        metadata = ProviderArtifact.from_json(
            self.transport.read_json(root + f"/actions/artifacts/{artifact_identifier}")
        )
        if (
            metadata.artifact_identifier != artifact_identifier
            or metadata.artifact_name != policy.artifact_name
            or metadata.expired
        ):
            raise InvalidIdentityEvidence("artifact identity or lifetime")
        run_resource = root + f"/actions/runs/{metadata.run_identifier}"
        workflow = WorkflowIdentity.from_json(self.transport.read_json(run_resource))
        if (
            workflow.run_identifier != metadata.run_identifier
            or workflow.repository != policy.repository
            or workflow.head_repository != policy.repository
            or workflow.workflow_path != policy.workflow_path
            or workflow.head_commit != checker_revision
            or not (
                workflow.event == ProviderValue.DISPATCH
                or policy.mode == PublicationMode.AUTOMATED
                and workflow.event == ProviderValue.PUSH
                and workflow.head_branch == policy.publisher_branch
            )
            or workflow.conclusion != ProviderValue.SUCCESS
            or workflow.status != ProviderValue.COMPLETED
        ):
            raise InvalidIdentityEvidence("publisher workflow identity")
        content = self.transport.read_bytes(
            root + f"/actions/artifacts/{artifact_identifier}/zip"
        )
        if metadata.archive_digest != "sha256:" + hashlib.sha256(content).hexdigest():
            raise InvalidIdentityEvidence("artifact archive digest")
        publication = self.decode_archive(content)
        if policy.mode == PublicationMode.INDEPENDENT_REVIEW:
            if publication.review is None:
                raise InvalidIdentityEvidence("missing independent review evidence")
            self.require_environment_review(
                self.transport.read_json(run_resource + "/approvals"),
                policy,
                publication.review.author_login,
            )
        receipt = ArtifactReceipt(
            policy.repository,
            artifact_identifier,
            metadata.run_identifier,
            workflow.workflow_path,
            workflow.head_commit,
            CanonicalDigest(self.codec).digest(publication),
            True,
            metadata.archive_digest,
        )
        return ArtifactDownload(publication, receipt)

    def decode_archive(self, content: bytes) -> Publication:
        """Read only a bounded single-file payload without extracting filesystem paths."""
        with ZipFile(BytesIO(content)) as archive:
            if archive.namelist() != [PublicationFilename.PAYLOAD]:
                raise InvalidIdentityEvidence("unexpected archive members")
            information = archive.getinfo(PublicationFilename.PAYLOAD)
            if (
                information.is_dir()
                or information.file_size > self.maximum_payload_bytes
            ):
                raise InvalidIdentityEvidence("oversized archive payload")
            with archive.open(PublicationFilename.PAYLOAD) as stream:
                payload = stream.read(self.maximum_payload_bytes + 1)
            if (
                len(payload) > self.maximum_payload_bytes
                or len(payload) != information.file_size
            ):
                raise InvalidIdentityEvidence("archive payload size")
        return self.codec.decode(payload.decode("utf-8"), Publication)

    @staticmethod
    def require_environment_review(
        value: object, policy: PublicationPolicy, author_login: str
    ) -> None:
        """Require an actual independent environment approval, including after admin bypass."""
        if not isinstance(value, list):
            raise InvalidIdentityEvidence("deployment review history")
        allowed = {login.casefold() for login in policy.reviewers}
        approved = False
        for item in value:
            decision = ProviderObject.from_json(item)
            environments = [
                ProviderObject.from_json(environment).text(ApiField.NAME)
                for environment in decision.sequence(ApiField.ENVIRONMENTS)
            ]
            if policy.environment not in environments:
                continue
            actor = decision.nested(ApiField.USER)
            if decision.text(ApiField.STATE) != ProviderValue.APPROVED:
                raise InvalidIdentityEvidence("deployment review refusal")
            approved = (
                approved
                or actor.text(ApiField.TYPE) == IdentityKind.HUMAN
                and actor.text(ApiField.LOGIN).casefold() in allowed
                and actor.text(ApiField.LOGIN).casefold() != author_login.casefold()
            )
        if not approved:
            raise InvalidIdentityEvidence("missing independent environment approval")
