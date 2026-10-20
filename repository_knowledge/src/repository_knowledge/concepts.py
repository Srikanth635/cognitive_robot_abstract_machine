"""Portable curated concepts and independently reported evidence checks."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import PurePosixPath

from typing_extensions import Any

from repository_knowledge.exceptions import InvalidConceptPath
from repository_knowledge.model import SourceLocation, SourceScope

# %% concept vocabulary


class ConceptKind(StrEnum):
    """The concept types curated by the CRAM pilot."""

    PACKAGE = "Package"
    WORKFLOW = "Workflow"
    CONVENTION = "Convention"


class KnowledgeFilename(StrEnum):
    """The portable bundle's reserved document names and extension."""

    INDEX = "index.md"
    LOG = "log.md"
    EXTENSION = ".md"


class Lifecycle(StrEnum):
    """Producer-declared content lifecycle, independent of freshness."""

    DRAFT = "draft"
    STABLE = "stable"
    DEPRECATED = "deprecated"


class EvidenceState(StrEnum):
    """Whether recorded source evidence still matches the selected snapshot."""

    CURRENT = "current"
    STALE = "stale"
    UNKNOWN = "unknown"


class ContentComparison(StrEnum):
    """The explicit content-equality policy applied to curated source evidence."""

    EXACT_BYTES = "exact-bytes"
    UTF8_LINE_ENDINGS = "utf8-lf-or-crlf"


class KnowledgeIssue(StrEnum):
    """Distinct format, local-policy and structural-evidence findings."""

    MISSING_TYPE = "missing-type"
    INVALID_NAVIGATION = "invalid-navigation"
    MISSING_TITLE = "missing-title"
    MISSING_DESCRIPTION = "missing-description"
    UNKNOWN_TYPE = "unknown-repository-type"
    UNKNOWN_LIFECYCLE = "unknown-lifecycle"
    MISSING_PROVENANCE = "missing-provenance"
    MISSING_EVIDENCE = "missing-evidence"
    MISSING_TYPE_EVIDENCE = "missing-type-evidence"
    INVALID_COMMIT = "invalid-source-commit"
    UNAUTHENTICATED_REVIEW = "unauthenticated-review-claim"
    INVALID_ACTOR = "invalid-actor"
    INVALID_TIMESTAMP = "invalid-timestamp"
    BROKEN_LINK = "broken-concept-link"
    MISSING_REPLACEMENT = "missing-replacement"
    REPLACEMENT_CYCLE = "replacement-cycle"
    MISSING_SOURCE = "missing-source"
    CHANGED_SOURCE = "changed-source"
    MISSING_SYMBOL = "missing-symbol"
    CHANGED_SYMBOL_LOCATION = "changed-symbol-location"
    INVALID_SOURCE_RANGE = "invalid-source-range"
    INVALID_ATTRIBUTION = "invalid-attribution"
    EXPIRED = "expired"
    STALE_DEPENDENCY = "stale-dependency"
    UNKNOWN_DEPENDENCY = "unknown-dependency"


# %% curated evidence


@dataclass(frozen=True)
class ConceptEvidence:
    """A source attribution bound to bytes and an optional declaration."""

    source_identifier: str
    """The stable attribution label used by the concept's claims."""
    location: SourceLocation
    """The inclusive source span supporting the explanation."""
    content_digest: str
    """The digest of the complete source file at drafting time."""
    symbol_identifier: str = ""
    """The qualified declaration identity, where applicable."""


@dataclass(frozen=True)
class ConceptProvenance:
    """Repository extensions binding curated prose to source evidence."""

    source_commit: str
    """The immutable source revision used when drafting the concept."""
    evidence: list[ConceptEvidence] = field(default_factory=list)
    """The source attributions requiring structural validation."""
    depends_on: list[str] = field(default_factory=list)
    """Concept paths whose invalidation also invalidates this concept."""
    replacement: str = ""
    """The successor concept preserved by a deprecated tombstone."""
    extensions: dict[str, Any] = field(default_factory=dict)
    """Additional repository metadata preserved during exchange."""


@dataclass(frozen=True)
class Concept:
    """A path-identified explanation with portable metadata and source lineage."""

    relative_path: str
    """The document's stable path within its knowledge bundle."""
    concept_type: str
    """The producer's open-ended concept classification."""
    title: str = ""
    """The optional human-readable concept label."""
    description: str = ""
    """The optional short explanation for discovery."""
    body: str = ""
    """The source-backed explanation, including its claim attributions."""
    lifecycle: str = Lifecycle.DRAFT
    """The declared lifecycle, which grants no publication authority."""
    stale_after: str = ""
    """The optional offset-aware expiry instant."""
    provenance: ConceptProvenance | None = None
    """The optional CRAM source-evidence binding."""
    metadata: dict[str, Any] = field(default_factory=dict)
    """Other portable metadata, including untrusted review declarations."""
    links: list[str] = field(default_factory=list)
    """Normalized local document targets extracted by the format adapter."""
    attributions: list[str] = field(default_factory=list)
    """Source footnote labels attached to claims by the format adapter."""

    def __post_init__(self) -> None:
        """Reject concept identities outside the portable bundle namespace."""
        path = PurePosixPath(self.relative_path)
        if (
            not self.relative_path
            or path.is_absolute()
            or ".." in path.parts
            or "\\" in self.relative_path
            or ":" in self.relative_path
            or path.suffix != KnowledgeFilename.EXTENSION
            or path.name in (KnowledgeFilename.INDEX, KnowledgeFilename.LOG)
            or path.as_posix() != self.relative_path
        ):
            raise InvalidConceptPath(self.relative_path)


@dataclass(frozen=True)
class KnowledgeBundle:
    """Concepts and reserved navigation documents exchanged as one collection."""

    concepts: list[Concept]
    """The curated documents in deterministic bundle order."""
    navigation: dict[str, str] = field(default_factory=dict)
    """Reserved index and log documents, retained independently of concepts."""
    navigation_issues: dict[str, list[KnowledgeIssue]] = field(default_factory=dict)
    """Format findings for reserved navigation documents."""


# %% independent validation results


@dataclass(frozen=True)
class ConceptValidation:
    """Separate interchange, repository-policy and source-evidence findings."""

    relative_path: str
    """The concept whose evidence was examined."""
    lifecycle: str
    """The unchanged producer-declared lifecycle."""
    format_issues: list[KnowledgeIssue]
    """Violations of mandatory interchange requirements."""
    policy_issues: list[KnowledgeIssue]
    """Violations of the stricter CRAM pilot policy."""
    evidence_issues: list[KnowledgeIssue]
    """Failed or incomplete structural evidence checks."""
    evidence_state: EvidenceState
    """Source consistency, which does not establish the truth of prose."""
    declared_verifiers: list[str] = field(default_factory=list)
    """Untrusted actors named by the document's producer."""
    authenticated_review: bool = False
    """Whether independent identity evidence established review authority."""
    text_equivalent_paths: list[str] = field(default_factory=list)
    """Sources whose byte digests differ only by supported UTF-8 line endings."""


@dataclass(frozen=True)
class BundleValidation:
    """Knowledge checks pinned to one completed source snapshot."""

    snapshot_identifier: str
    """The source snapshot used for all evidence checks."""
    concepts: list[ConceptValidation]
    """The findings in bundle concept order."""
    navigation_issues: dict[str, list[KnowledgeIssue]] = field(default_factory=dict)
    """Reserved-document violations reported independently of concept policy."""
    source_scope: SourceScope = SourceScope.COMMITTED
    """The source scope against which the evidence was checked."""
    source_commit: str = ""
    """The selected source commit, independent of producer-declared provenance."""
    checked_at: str = ""
    """The offset-aware instant used for expiry checks."""
    content_comparison: ContentComparison = ContentComparison.UTF8_LINE_ENDINGS
    """The explicit content-equality policy used for these findings."""


@dataclass(frozen=True)
class ConceptPage:
    """Bounded curated explanations with snapshot-specific evidence checks."""

    concepts: list[Concept]
    """The selected explanations in deterministic relevance order."""
    validation: BundleValidation
    """Evidence checks for the selected concepts against one source snapshot."""
    total_matches: int
    """The complete match count before pagination."""
    next_offset: int | None
    """The next unread result position, if any."""
