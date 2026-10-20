"""Content identities and independent evidence for canonical knowledge."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing_extensions import TYPE_CHECKING

if TYPE_CHECKING:
    from repository_knowledge.concepts import BundleValidation, KnowledgeBundle
    from repository_knowledge.model import IndexData, IndexManifest

# %% trust vocabulary


class TrustIssue(StrEnum):
    """The trust condition preventing canonical use."""

    MISSING_REVIEW = "missing-independent-review"
    SELF_REVIEW = "self-review"
    STALE_REVIEW = "stale-review"
    CHANGED_CONTENT = "changed-reviewed-content"
    WRONG_REPOSITORY = "wrong-repository"
    WRONG_SOURCE = "wrong-source-revision"
    POLICY_MISMATCH = "policy-mismatch"
    UNVERIFIED_CONTROLS = "unverified-server-controls"
    INVALID_KNOWLEDGE = "invalid-knowledge"
    WORKING_SOURCE = "working-source"
    ARTIFACT_IDENTITY = "unverified-artifact-identity"
    ARTIFACT_DIGEST = "artifact-digest-mismatch"
    LIFECYCLE = "invalid-lifecycle-transition"
    GOVERNANCE_CHANGE = "trusted-checker-or-policy-change"
    EXTRACTION_CHANGED = "nondeterministic-extraction"


class ConsumptionState(StrEnum):
    """Whether canonical prose is admitted or source-only fallback is required."""

    ACCEPTED = "accepted"
    SOURCE_ONLY = "source-only"


class PublicationMode(StrEnum):
    """The evidence required before publishing validated content."""

    AUTOMATED = "automated"
    INDEPENDENT_REVIEW = "independent-review"


class ReviewState(StrEnum):
    """GitHub review states relevant to approval authority."""

    APPROVED = "APPROVED"
    CHANGES_REQUESTED = "CHANGES_REQUESTED"
    DISMISSED = "DISMISSED"
    COMMENTED = "COMMENTED"


class IdentityKind(StrEnum):
    """Identity categories allowed to independently review prose."""

    HUMAN = "User"


class RepositoryPermission(StrEnum):
    """Repository permissions sufficient for a configured content owner."""

    WRITE = "write"
    MAINTAIN = "maintain"
    ADMIN = "admin"


# %% policy and authenticated evidence


@dataclass(frozen=True)
class PublicationPolicy:
    """Trusted publication requirements selected outside proposal data."""

    repository: str
    """The exact GitHub fork in owner/repository form."""
    branch: str
    """The source branch whose committed history may be published."""
    reviewers: list[str]
    """GitHub users authorized to independently review the complete bundle."""
    minimum_reviews: int = 1
    """The minimum distinct eligible approvals."""
    version: int = 1
    """The policy interpretation version bound into every publication."""
    required_check: str = "knowledge-admission"
    """The required trusted admission status context."""
    environment: str = "knowledge-publication"
    """The deployment environment required only by independent-review mode."""
    workflow_path: str = ".github/workflows/knowledge-publish.yml"
    """The publisher workflow pinned by the consumer's trusted checker revision."""
    bundle_directory: str = "repository_knowledge/wiki"
    """The committed bundle directory, never an external producer-selected path."""
    artifact_name: str = "cram-knowledge"
    """The immutable workflow artifact carrying the canonical payload."""
    protected_paths: list[str] = field(
        default_factory=lambda: [
            "repository_knowledge/src/",
            "repository_knowledge/scripts/",
            "repository_knowledge/policy/",
            "repository_knowledge/runtime/",
            "repository_knowledge/pyproject.toml",
            ".github/CODEOWNERS",
            ".github/workflows/knowledge-",
            "krrood/src/",
            "krrood/pyproject.toml",
        ]
    )
    """Checker, dependency and governance paths requiring separate trust-root review."""
    mode: PublicationMode = PublicationMode.INDEPENDENT_REVIEW
    """Whether publication requires machine validation or additional human review."""
    publisher_branch: str = "codex/knowledge-publisher"
    """The separately selected branch permitted to initiate trusted publications."""


@dataclass(frozen=True)
class SourceIdentity:
    """A source commit and tree authenticated by the repository provider."""

    repository: str
    """The authenticated repository."""
    branch: str
    """The configured source branch."""
    source_commit: str
    """The immutable source revision."""
    source_tree: str
    """The exact tree supplied by that revision."""
    author_login: str
    """The provider-associated author, when available."""
    authenticated: bool
    """Whether the provider boundary verified the source identity."""


@dataclass(frozen=True)
class ControlEvidence:
    """Repository and environment enforcement read from the identity provider."""

    repository: str
    """The authenticated repository."""
    branch: str
    """The audited source branch."""
    branch_protected: bool
    """Whether branch protection is enabled."""
    stale_reviews_dismissed: bool
    """Whether changed content dismisses existing approvals."""
    owners_required: bool
    """Whether code-owner review is enforced."""
    administrators_enforced: bool
    """Whether administrators remain subject to branch protection."""
    last_push_approval: bool
    """Whether the final push needs another actor's approval."""
    required_reviews: int
    """The enforced minimum review count."""
    required_checks: list[str]
    """The enforced status contexts."""
    environment_protected: bool
    """Whether the publication environment enforces independent restricted review."""


@dataclass(frozen=True)
class ReviewDecision:
    """One authenticated GitHub review decision and repository permission."""

    review_identifier: int
    """The immutable provider review identifier."""
    actor_login: str
    """The authenticated GitHub login."""
    actor_kind: str
    """The identity provider's actor category."""
    state: str
    """The provider's current review state."""
    commit_identifier: str
    """The exact proposal head reviewed by this decision."""
    permission: str
    """The reviewer's current repository permission."""


@dataclass(frozen=True)
class ReviewEvidence:
    """Provider evidence binding approvals to reviewed and merged trees."""

    repository: str
    """The authenticated base repository."""
    branch: str
    """The authenticated proposal base branch."""
    request_number: int
    """The merged pull request identifier."""
    author_login: str
    """The proposal author's authenticated login."""
    head_commit: str
    """The exact head named by eligible reviews."""
    reviewed_tree: str
    """The Git tree at the reviewed head."""
    merged_commit: str
    """The source commit actually merged by the provider."""
    merged_tree: str
    """The Git tree at the merged source commit."""
    decisions: list[ReviewDecision]
    """The provider's review history, including dismissals and later changes."""
    merged: bool
    """Whether the provider confirms that the request was merged."""


@dataclass(frozen=True)
class TrustReport:
    """Explicit trust refusals independent of ordinary format validation."""

    issues: list[TrustIssue] = field(default_factory=list)
    """The failed trust conditions, or empty for admission."""


# %% immutable publication payload


@dataclass(frozen=True)
class ConceptDigest:
    """The exact canonical concept record approved in a publication."""

    relative_path: str
    """The concept identity within the bundle."""
    content_digest: str
    """The canonical record digest including body and all metadata."""


@dataclass(frozen=True)
class VerificationBinding:
    """Bind content, source and trusted policy without a self-referential hash."""

    bundle_digest: str
    """The complete bundle digest, including navigation and unknown metadata."""
    policy_digest: str
    """The exact trusted policy digest."""
    snapshot_identifier: str
    """The completed committed source snapshot."""
    source_commit: str
    """The reviewed and merged source revision."""
    source_tree: str
    """The exact reviewed source tree."""
    concepts: list[ConceptDigest]
    """Per-concept content bindings."""


@dataclass(frozen=True)
class Publication:
    """Canonical content with source validation and its explicit review level."""

    binding: VerificationBinding
    """The source, content and policy binding."""
    checker_revision: str
    """The separately trusted immutable checker/publisher revision."""
    bundle: KnowledgeBundle
    """The approved bundle, retaining producer lifecycle declarations."""
    source: IndexData
    """The cited source files and declarations from the completed source snapshot."""
    validation: BundleValidation
    """Machine findings with independent review recorded separately."""
    review: ReviewEvidence | None
    """The authenticated review evidence used at publication."""
    mode: PublicationMode = PublicationMode.INDEPENDENT_REVIEW
    """The evidence level granted by this publication."""
    identity: SourceIdentity | None = None
    """Authenticated source identity for automated publication."""


@dataclass(frozen=True)
class ArtifactReceipt:
    """An artifact identity obtained through an authenticated provider boundary."""

    repository: str
    """The authenticated artifact repository."""
    artifact_identifier: int
    """The immutable provider artifact identifier."""
    run_identifier: int
    """The workflow run supplying the artifact."""
    workflow_path: str
    """The authenticated workflow path."""
    checker_revision: str
    """The workflow run's immutable code revision."""
    payload_digest: str
    """The payload digest after checking the provider's archive digest."""
    authenticated: bool
    """Whether provider identity, digest and mode-specific controls passed."""
    archive_digest: str = ""
    """The provider's digest of the complete downloaded archive."""


@dataclass(frozen=True)
class ArtifactDownload:
    """A publication and its separately authenticated artifact identity."""

    publication: Publication
    """The decoded immutable publication payload."""
    receipt: ArtifactReceipt
    """The authenticated identity outside the hashed payload."""


@dataclass(frozen=True)
class ConsumptionResult:
    """Admitted canonical evidence or an explicit source-only fallback."""

    state: ConsumptionState
    """Whether consumers may use the canonical bundle."""
    publication: Publication | None
    """The admitted complete publication, never a partial replacement."""
    issues: list[TrustIssue]
    """The refusals requiring source-only fallback."""


# %% deterministic admission evidence


@dataclass(frozen=True)
class ExtractionFingerprint:
    """Framed identities of every deterministic extraction record category."""

    manifest: IndexManifest
    """The exact source and interpretation identity."""
    sources: str
    """The ordered source inventory digest."""
    symbols: str
    """The ordered declaration digest."""
    references: str
    """The ordered resolved relationship digest."""
    diagnostics: str
    """The ordered extraction limitation digest."""
    analyses: str
    """The ordered raw analysis digest."""


@dataclass(frozen=True)
class AdmissionResult:
    """Bounded admission findings with reproducible source and policy identities."""

    manifest: IndexManifest
    """The proposed committed source snapshot."""
    policy_digest: str
    """The independently trusted policy identity."""
    extraction_digest: str
    """The verified deterministic extraction fingerprint digest."""
    governance_changes: list[str]
    """Protected proposal paths differing from the trusted checker checkout."""
    report: TrustReport
    """The machine-checkable admission findings."""
