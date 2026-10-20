"""
Canonical context uses authenticated consumption and explicit source matching.
"""

from __future__ import annotations

import pytest

from repository_knowledge.adapters.digests import CanonicalDigest
from repository_knowledge.adapters.sqlite_index import SQLiteIndex
from repository_knowledge.assistance import ContextRole
from repository_knowledge.composition import LocalApplication
from repository_knowledge.exceptions import SourceRevisionMismatch
from repository_knowledge.concepts import KnowledgeBundle
from repository_knowledge.interfaces.cli import CommandParser
from repository_knowledge.model import IndexData, SourceScope
from repository_knowledge.operations.canonical_context import CanonicalContext
from repository_knowledge.operations.publication import KnowledgePublication
from repository_knowledge.publication import (
    ConsumptionResult,
    ConsumptionState,
    ControlEvidence,
    Publication,
    PublicationPolicy,
    ReviewEvidence,
    TrustIssue,
)

from .dataset.canonical_reader import CapturedConsumption
from .dataset.advancing_consumption import AdvancingConsumption
from .dataset.publication_sources import CapturedKnowledge, IndependentReviews
from .support import CommittedProject, FixtureSymbol

# %% admitted context fixture


@pytest.fixture
def canonical_publication(
    source_index: IndexData,
    publication_policy: PublicationPolicy,
    publication_bundle: KnowledgeBundle,
    publication_controls: ControlEvidence,
    review_evidence: ReviewEvidence,
) -> Publication:
    """
    Produce the same complete publication used by the trust-gate tests.
    """
    return KnowledgePublication(
        publication_policy,
        CapturedKnowledge(publication_bundle),
        IndependentReviews(publication_controls, review_evidence),
        CanonicalDigest(),
    ).publish(
        source_index,
        review_evidence.request_number,
        source_index.manifest.commit_identifier,
    )


@pytest.mark.parametrize("accepted", [True, False])
def test_context_never_substitutes_local_drafts_for_canonical_refusal(
    committed_project: CommittedProject,
    indexed_storage: SQLiteIndex,
    canonical_publication: Publication,
    accepted: bool,
) -> None:
    """
    Canonical refusal leaves a marked source-only context with no wiki prose.
    """
    verdict = ConsumptionResult(
        ConsumptionState.ACCEPTED if accepted else ConsumptionState.SOURCE_ONLY,
        canonical_publication if accepted else None,
        [] if accepted else [TrustIssue.WRONG_SOURCE],
    )
    reader = CapturedConsumption(verdict)
    arguments = CommandParser().parse(
        [
            "--repository",
            str(committed_project.root),
            "--database",
            str(indexed_storage.database_path),
            "prepare-context",
            FixtureSymbol.BASE,
            "--bundle",
            str(committed_project.root / "does-not-exist"),
        ]
    )
    context = LocalApplication(reader).execute(arguments)
    assert context.canonical_status.state == verdict.state
    assert context.canonical_status.issues == verdict.issues
    assert reader.requested_revisions == [committed_project.commit_identifier]
    assert [
        item.label
        for item in context.items
        if item.role == ContextRole.CONCEPT and item.label.endswith(".md")
    ] == ([canonical_publication.bundle.concepts[0].relative_path] if accepted else [])


def test_working_source_never_contacts_the_canonical_reader(
    canonical_publication: Publication,
) -> None:
    """
    Committed canonical prose cannot be mixed with an edited source overlay.
    """
    reader = CapturedConsumption(
        ConsumptionResult(ConsumptionState.ACCEPTED, canonical_publication, [])
    )
    knowledge = CanonicalContext(reader).read(
        canonical_publication.binding.source_commit, SourceScope.WORKING_TREE
    )
    assert knowledge.canonical_status.state == ConsumptionState.SOURCE_ONLY
    assert knowledge.canonical_status.issues == [TrustIssue.WORKING_SOURCE]
    assert knowledge.bundle == KnowledgeBundle([])
    assert reader.requested_revisions == []


def test_canonical_context_rejects_a_different_source_pair(
    canonical_publication: Publication,
) -> None:
    """
    An accepted verdict cannot be reused for a different requested source.
    """
    reader = CapturedConsumption(
        ConsumptionResult(ConsumptionState.ACCEPTED, canonical_publication, [])
    )
    knowledge = CanonicalContext(reader).read("0" * 40, SourceScope.COMMITTED)
    assert knowledge.canonical_status.issues == [TrustIssue.WRONG_SOURCE]
    assert knowledge.canonical_status.state == ConsumptionState.SOURCE_ONLY
    assert knowledge.bundle == KnowledgeBundle([])


def test_canonical_read_cannot_mix_with_a_replaced_source_index(
    committed_project: CommittedProject,
    indexed_storage: SQLiteIndex,
    canonical_publication: Publication,
) -> None:
    """
    Even unchanged cited bytes cannot combine old authority with a new commit.
    """
    reader = AdvancingConsumption(
        ConsumptionResult(ConsumptionState.ACCEPTED, canonical_publication, []),
        committed_project,
        indexed_storage,
    )
    arguments = CommandParser().parse(
        [
            "--repository",
            str(committed_project.root),
            "--database",
            str(indexed_storage.database_path),
            "prepare-context",
            FixtureSymbol.BASE,
        ]
    )
    with pytest.raises(SourceRevisionMismatch):
        LocalApplication(reader).execute(arguments)
