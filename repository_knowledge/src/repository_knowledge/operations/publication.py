"""Validate committed bundles and enforce the selected publication evidence level."""

from __future__ import annotations

import re
from dataclasses import dataclass, replace
from typing_extensions import TYPE_CHECKING

from repository_knowledge.concepts import ContentComparison, EvidenceState
from repository_knowledge.model import SourceScope
from repository_knowledge.operations.exceptions import PublicationDenied
from repository_knowledge.operations.knowledge import KnowledgeValidation
from repository_knowledge.publication import (
    ConceptDigest,
    ConsumptionResult,
    ConsumptionState,
    IdentityKind,
    Publication,
    RepositoryPermission,
    ReviewState,
    TrustIssue,
    TrustReport,
    VerificationBinding,
    PublicationMode,
)

if TYPE_CHECKING:
    from repository_knowledge.contracts import (
        KnowledgeReader,
        RecordDigester,
        ReviewReader,
        SourceIdentityReader,
    )
    from repository_knowledge.model import IndexData
    from repository_knowledge.publication import (
        ArtifactReceipt,
        ControlEvidence,
        PublicationPolicy,
        ReviewEvidence,
        SourceIdentity,
    )

# %% independent review admission


@dataclass
class PublicationGate:
    """Evaluate authenticated identities independently of document claims."""

    def check_source(
        self,
        policy: PublicationPolicy,
        index: IndexData,
        controls: ControlEvidence,
        identity: SourceIdentity,
    ) -> TrustReport:
        """Require authenticated repository identity and exact committed source."""
        if index.manifest.source_scope != SourceScope.COMMITTED:
            return TrustReport([TrustIssue.WORKING_SOURCE])
        if (
            identity.repository != policy.repository
            or controls.repository != policy.repository
            or identity.branch != policy.branch
            or controls.branch != policy.branch
        ):
            return TrustReport([TrustIssue.WRONG_REPOSITORY])
        if not identity.authenticated:
            return TrustReport([TrustIssue.UNVERIFIED_CONTROLS])
        if (
            identity.source_commit != index.manifest.commit_identifier
            or identity.source_tree != index.manifest.tree_identifier
        ):
            return TrustReport([TrustIssue.WRONG_SOURCE])
        return TrustReport()

    def check(
        self,
        policy: PublicationPolicy,
        index: IndexData,
        controls: ControlEvidence,
        review: ReviewEvidence,
    ) -> TrustReport:
        """Refuse uncommitted, unprotected, mismatched or unreviewed content."""
        if index.manifest.source_scope != SourceScope.COMMITTED:
            return TrustReport([TrustIssue.WORKING_SOURCE])
        if (
            review.repository != policy.repository
            or controls.repository != policy.repository
            or review.branch != policy.branch
            or controls.branch != policy.branch
        ):
            return TrustReport([TrustIssue.WRONG_REPOSITORY])
        if (
            not review.merged
            or review.merged_commit != index.manifest.commit_identifier
            or review.merged_tree != index.manifest.tree_identifier
        ):
            return TrustReport([TrustIssue.WRONG_SOURCE])
        if review.reviewed_tree != review.merged_tree:
            return TrustReport([TrustIssue.CHANGED_CONTENT])
        if (
            not all(
                (
                    controls.branch_protected,
                    controls.stale_reviews_dismissed,
                    controls.owners_required,
                    controls.administrators_enforced,
                    controls.last_push_approval,
                    controls.environment_protected,
                )
            )
            or controls.required_reviews < policy.minimum_reviews
            or policy.required_check not in controls.required_checks
        ):
            return TrustReport([TrustIssue.UNVERIFIED_CONTROLS])
        if policy.minimum_reviews < 1 or not policy.reviewers:
            return TrustReport([TrustIssue.MISSING_REVIEW])
        latest = {}
        for decision in sorted(
            review.decisions, key=lambda item: item.review_identifier
        ):
            if decision.state != ReviewState.COMMENTED:
                latest[decision.actor_login.casefold()] = decision
        allowed = {login.casefold() for login in policy.reviewers}
        approved = [
            decision
            for login, decision in latest.items()
            if login in allowed
            and decision.state == ReviewState.APPROVED
            and decision.actor_kind == IdentityKind.HUMAN
            and decision.permission in RepositoryPermission
        ]
        if any(
            decision.state == ReviewState.CHANGES_REQUESTED
            for login, decision in latest.items()
            if login in allowed
        ):
            return TrustReport([TrustIssue.MISSING_REVIEW])
        independent = [
            decision
            for decision in approved
            if decision.actor_login.casefold() != review.author_login.casefold()
        ]
        if approved and not independent:
            return TrustReport([TrustIssue.SELF_REVIEW])
        current = [
            decision
            for decision in independent
            if decision.commit_identifier == review.head_commit
        ]
        if independent and not current:
            return TrustReport([TrustIssue.STALE_REVIEW])
        if len(current) < policy.minimum_reviews:
            return TrustReport([TrustIssue.MISSING_REVIEW])
        return TrustReport()


# %% canonical bundle construction


@dataclass
class KnowledgePublication:
    """Build a complete publication after authenticated source admission."""

    policy: PublicationPolicy
    """The trusted admission policy."""
    knowledge_reader: KnowledgeReader
    """The captured committed document boundary."""
    review_reader: ReviewReader
    """The independent identity and enforcement boundary."""
    digester: RecordDigester
    """The canonical content identity boundary."""
    source_reader: SourceIdentityReader | None = None
    """The authenticated source boundary for automated publication."""

    def publish(
        self, index: IndexData, request_number: int, checker_revision: str
    ) -> Publication:
        """Bind validated content, policy and authenticated source evidence."""
        controls = self.review_reader.read_controls(self.policy)
        review = None
        identity = None
        if self.policy.mode == PublicationMode.AUTOMATED:
            if self.source_reader is None:
                raise PublicationDenied([TrustIssue.UNVERIFIED_CONTROLS])
            identity = self.source_reader.read_source(
                self.policy, index.manifest.commit_identifier
            )
            report = PublicationGate().check_source(
                self.policy, index, controls, identity
            )
        else:
            review = self.review_reader.read_review(self.policy, request_number)
            report = PublicationGate().check(self.policy, index, controls, review)
        if report.issues:
            raise PublicationDenied(report.issues)
        if re.fullmatch(r"[0-9a-f]{40}", checker_revision) is None:
            raise PublicationDenied([TrustIssue.ARTIFACT_IDENTITY])
        bundle = self.knowledge_reader.read_bundle(index, self.policy.bundle_directory)
        validation = KnowledgeValidation(ContentComparison.EXACT_BYTES).validate(
            bundle, index
        )
        if (
            not bundle.concepts
            or validation.navigation_issues
            or any(
                item.format_issues
                or item.policy_issues
                or item.evidence_state != EvidenceState.CURRENT
                for item in validation.concepts
            )
        ):
            raise PublicationDenied([TrustIssue.INVALID_KNOWLEDGE])
        validation = replace(
            validation,
            concepts=[
                replace(item, authenticated_review=review is not None)
                for item in validation.concepts
            ],
        )
        paths = {
            evidence.location.relative_path
            for concept in bundle.concepts
            if concept.provenance is not None
            for evidence in concept.provenance.evidence
        }
        source = replace(
            index,
            sources=[item for item in index.sources if item.relative_path in paths],
            symbols=[
                item for item in index.symbols if item.location.relative_path in paths
            ],
            references=[],
            analyses=[],
            diagnostics=[
                item for item in index.diagnostics if item.relative_path in paths
            ],
        )
        binding = VerificationBinding(
            self.digester.digest(bundle),
            self.digester.digest(self.policy),
            index.manifest.snapshot_identifier,
            index.manifest.commit_identifier,
            index.manifest.tree_identifier,
            [
                ConceptDigest(concept.relative_path, self.digester.digest(concept))
                for concept in sorted(
                    bundle.concepts, key=lambda item: item.relative_path
                )
            ],
        )
        return Publication(
            binding,
            checker_revision,
            bundle,
            source,
            validation,
            review,
            self.policy.mode,
            identity,
        )


# %% consumer refusal and historical restoration


@dataclass
class PublicationConsumer:
    """Admit complete publications or explicitly fall back to source-only context."""

    policy: PublicationPolicy
    """The consumer's independently trusted policy."""
    review_reader: ReviewReader
    """The independent identity and enforcement boundary."""
    digester: RecordDigester
    """The canonical content identity boundary."""
    checker_revision: str
    """The consumer's externally pinned checker/publisher revision."""
    source_reader: SourceIdentityReader | None = None
    """The authenticated source boundary for automated consumption."""

    def consume(
        self, publication: Publication, receipt: ArtifactReceipt, source_commit: str
    ) -> ConsumptionResult:
        """Return only the exact authenticated bundle/source pair requested by the caller."""
        report = self.inspect(publication, receipt, source_commit)
        if report.issues:
            return ConsumptionResult(ConsumptionState.SOURCE_ONLY, None, report.issues)
        return ConsumptionResult(ConsumptionState.ACCEPTED, publication, [])

    def inspect(
        self, publication: Publication, receipt: ArtifactReceipt, source_commit: str
    ) -> TrustReport:
        """Recheck archive identity, policy, content and selected review requirements."""
        if (
            not receipt.authenticated
            or receipt.artifact_identifier < 1
            or receipt.run_identifier < 1
            or receipt.repository != self.policy.repository
            or receipt.workflow_path != self.policy.workflow_path
            or receipt.checker_revision != self.checker_revision
            or publication.checker_revision != self.checker_revision
        ):
            return TrustReport([TrustIssue.ARTIFACT_IDENTITY])
        if self.digester.digest(publication) != receipt.payload_digest:
            return TrustReport([TrustIssue.ARTIFACT_DIGEST])
        if (
            publication.source.manifest.commit_identifier != source_commit
            or publication.binding.source_commit != source_commit
            or publication.binding.source_tree
            != publication.source.manifest.tree_identifier
            or publication.binding.snapshot_identifier
            != publication.source.manifest.snapshot_identifier
        ):
            return TrustReport([TrustIssue.WRONG_SOURCE])
        if (
            publication.binding.policy_digest != self.digester.digest(self.policy)
            or publication.mode != self.policy.mode
        ):
            return TrustReport([TrustIssue.POLICY_MISMATCH])
        if publication.binding.bundle_digest != self.digester.digest(
            publication.bundle
        ) or publication.binding.concepts != [
            ConceptDigest(concept.relative_path, self.digester.digest(concept))
            for concept in sorted(
                publication.bundle.concepts, key=lambda item: item.relative_path
            )
        ]:
            return TrustReport([TrustIssue.CHANGED_CONTENT])
        controls = self.review_reader.read_controls(self.policy)
        if self.policy.mode == PublicationMode.AUTOMATED:
            if self.source_reader is None or publication.review is not None:
                return TrustReport([TrustIssue.UNVERIFIED_CONTROLS])
            identity = self.source_reader.read_source(self.policy, source_commit)
            report = PublicationGate().check_source(
                self.policy, publication.source, controls, identity
            )
            if identity != publication.identity:
                return TrustReport([TrustIssue.WRONG_SOURCE])
        else:
            if publication.review is None:
                return TrustReport([TrustIssue.MISSING_REVIEW])
            review = self.review_reader.read_review(
                self.policy, publication.review.request_number
            )
            report = PublicationGate().check(
                self.policy, publication.source, controls, review
            )
            if review != publication.review:
                return TrustReport([TrustIssue.STALE_REVIEW])
        if report.issues:
            return report
        validation = KnowledgeValidation(ContentComparison.EXACT_BYTES).validate(
            publication.bundle, publication.source
        )
        validation = replace(
            validation,
            concepts=[
                replace(
                    item,
                    authenticated_review=self.policy.mode
                    == PublicationMode.INDEPENDENT_REVIEW,
                )
                for item in validation.concepts
            ],
        )
        if (
            replace(validation, checked_at=publication.validation.checked_at)
            != publication.validation
            or not publication.bundle.concepts
        ):
            return TrustReport([TrustIssue.INVALID_KNOWLEDGE])
        if validation.navigation_issues or any(
            item.format_issues
            or item.policy_issues
            or item.evidence_state != EvidenceState.CURRENT
            for item in validation.concepts
        ):
            return TrustReport([TrustIssue.INVALID_KNOWLEDGE])
        return TrustReport()
