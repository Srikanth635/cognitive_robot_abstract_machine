"""
Unprivileged source, lifecycle and trusted-policy proposal checks.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing_extensions import TYPE_CHECKING

from repository_knowledge.concepts import ContentComparison, EvidenceState, Lifecycle
from repository_knowledge.operations.knowledge import KnowledgeValidation
from repository_knowledge.publication import TrustIssue, TrustReport

if TYPE_CHECKING:
    from repository_knowledge.concepts import KnowledgeBundle
    from repository_knowledge.contracts import RecordDigester
    from repository_knowledge.model import IndexData

# %% trusted proposal validation


@dataclass
class ProposalAdmission:
    """
    Check proposal data without letting it supply its own checker or policy.
    """

    digester: RecordDigester
    """The trusted canonical record identity boundary."""

    def check(
        self,
        previous: KnowledgeBundle,
        proposed: KnowledgeBundle,
        index: IndexData,
        governance_changes: list[str],
        first_extraction: str,
        second_extraction: str,
    ) -> TrustReport:
        """
        Report machine-checkable refusals without authenticating prose or reviewers.
        """
        if governance_changes:
            return TrustReport([TrustIssue.GOVERNANCE_CHANGE])
        if first_extraction != second_extraction:
            return TrustReport([TrustIssue.EXTRACTION_CHANGED])
        before = {concept.relative_path: concept for concept in previous.concepts}
        after = {concept.relative_path: concept for concept in proposed.concepts}
        if before.keys() - after.keys():
            return TrustReport([TrustIssue.LIFECYCLE])
        for relative, concept in after.items():
            original = before.get(relative)
            if original is None and concept.lifecycle != Lifecycle.DRAFT:
                return TrustReport([TrustIssue.LIFECYCLE])
            if (
                original is not None
                and self.digester.digest(original) != self.digester.digest(concept)
                and concept.lifecycle not in (Lifecycle.DRAFT, Lifecycle.DEPRECATED)
            ):
                return TrustReport([TrustIssue.LIFECYCLE])
        validation = KnowledgeValidation(ContentComparison.EXACT_BYTES).validate(
            proposed, index
        )
        if (
            not proposed.concepts
            or validation.navigation_issues
            or any(
                item.format_issues
                or item.policy_issues
                or item.evidence_state != EvidenceState.CURRENT
                for item in validation.concepts
            )
        ):
            return TrustReport([TrustIssue.INVALID_KNOWLEDGE])
        return TrustReport()
