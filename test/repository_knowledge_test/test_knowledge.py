"""
Curated knowledge separates interchange, policy and source evidence.
"""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import pytest

from repository_knowledge.adapters.okf import OkfCodec
from repository_knowledge.adapters.serialization import RecordCodec
from repository_knowledge.concepts import (
    Concept,
    KnowledgeBundle,
    Lifecycle,
    KnowledgeIssue,
    EvidenceState,
)
from repository_knowledge.operations.knowledge import (
    KnowledgeValidation,
    ConceptLifecycle,
)
from repository_knowledge.operations.exceptions import InvalidConceptTransition
from typing_extensions import TYPE_CHECKING

if TYPE_CHECKING:
    from repository_knowledge.model import IndexData

# %% portable interchange


def test_unknown_fields_and_types_round_trip(portable_concept: Concept) -> None:
    """
    Export retains additional metadata and the concept's path identity.
    """
    codec = OkfCodec()
    assert (
        codec.decode(portable_concept.relative_path, codec.encode(portable_concept))
        == portable_concept
    )


def test_repository_policy_is_separate_from_format(
    portable_concept: Concept, source_index: IndexData
) -> None:
    """
    A minimal valid concept can fail local policy without failing interchange.
    """
    result = KnowledgeValidation().validate(
        KnowledgeBundle([portable_concept]), source_index
    )
    assert result.concepts[0].format_issues == []
    assert KnowledgeIssue.MISSING_PROVENANCE in result.concepts[0].policy_issues
    assert result.concepts[0].evidence_state == EvidenceState.UNKNOWN


def test_missing_type_fails_format(source_index: IndexData) -> None:
    """
    The only always-required metadata field is checked independently.
    """
    fixture = Path(__file__).parent / "dataset/knowledge/invalid.md"
    concept = OkfCodec().decode("invalid.md", fixture.read_text(encoding="utf-8"))
    result = KnowledgeValidation().validate(KnowledgeBundle([concept]), source_index)
    assert result.concepts[0].format_issues == [KnowledgeIssue.MISSING_TYPE]


def test_declared_human_review_does_not_grant_authority(
    source_index: IndexData,
) -> None:
    """
    Even stable lifecycle and human-looking metadata establish no authority.
    """
    fixture = Path(__file__).parent / "dataset/knowledge/claimed-review.md"
    concept = OkfCodec().decode(
        "claimed-review.md", fixture.read_text(encoding="utf-8")
    )
    result = KnowledgeValidation().validate(KnowledgeBundle([concept]), source_index)
    assert result.concepts[0].declared_verifiers == ["human:owner"]
    assert result.concepts[0].authenticated_review is False
    assert KnowledgeIssue.UNAUTHENTICATED_REVIEW in result.concepts[0].policy_issues


# %% evidence and invalidation


def test_matching_evidence_is_structurally_current(
    evidenced_concept: Concept, source_index: IndexData
) -> None:
    """
    Digest and symbol checks support freshness, not prose truth or review.
    """
    result = KnowledgeValidation().validate(
        KnowledgeBundle([evidenced_concept]), source_index
    )
    assert result.concepts[0].evidence_state == EvidenceState.CURRENT
    assert result.concepts[0].authenticated_review is False


def test_source_change_invalidates_reverse_dependencies(
    evidenced_concept: Concept, source_index: IndexData
) -> None:
    """
    A changed source also invalidates concepts depending on its explanation.
    """
    dependent = replace(
        evidenced_concept,
        relative_path="workflows/usage.md",
        provenance=replace(
            evidenced_concept.provenance, depends_on=[evidenced_concept.relative_path]
        ),
    )
    evidence = evidenced_concept.provenance.evidence[0]
    changed_sources = [
        (
            replace(item, content_digest="changed")
            if item.relative_path == evidence.location.relative_path
            else item
        )
        for item in source_index.sources
    ]
    changed = replace(source_index, sources=changed_sources)
    result = KnowledgeValidation().validate(
        KnowledgeBundle([evidenced_concept, dependent]), changed
    )
    assert [item.evidence_state for item in result.concepts] == [
        EvidenceState.STALE,
        EvidenceState.STALE,
    ]
    assert KnowledgeIssue.STALE_DEPENDENCY in result.concepts[1].evidence_issues


def test_expiry_does_not_change_lifecycle(
    evidenced_concept: Concept, source_index: IndexData
) -> None:
    """
    Expiry is independent of the producer's lifecycle declaration.
    """
    concept = replace(
        evidenced_concept,
        lifecycle=Lifecycle.STABLE,
        stale_after="2026-10-20T00:00:00Z",
    )
    result = KnowledgeValidation().validate(
        KnowledgeBundle([concept]),
        source_index,
        now=datetime(2026, 10, 20, tzinfo=timezone.utc),
    )
    assert result.concepts[0].evidence_state == EvidenceState.STALE
    assert result.concepts[0].lifecycle == Lifecycle.STABLE


def test_moving_concept_retains_tombstone(evidenced_concept: Concept) -> None:
    """
    The previous identity remains a deprecated link to the replacement.
    """
    bundle = ConceptLifecycle().relocate(
        KnowledgeBundle([evidenced_concept]),
        evidenced_concept.relative_path,
        "packages/renamed.md",
    )
    tombstone, replacement = bundle.concepts
    assert tombstone.relative_path == evidenced_concept.relative_path
    assert tombstone.lifecycle == Lifecycle.DEPRECATED
    assert tombstone.provenance.replacement == replacement.relative_path
    assert replacement.relative_path == "packages/renamed.md"
    assert replacement.lifecycle == Lifecycle.DRAFT


def test_producer_cannot_promote_to_stable(evidenced_concept: Concept) -> None:
    """
    Stable promotion requires the later authenticated publication boundary.
    """
    with pytest.raises(InvalidConceptTransition):
        ConceptLifecycle().transition(evidenced_concept, Lifecycle.STABLE)


def test_knowledge_results_use_owned_serializer(
    evidenced_concept: Concept, source_index: IndexData
) -> None:
    """
    New domain records round-trip through the established safe JSON boundary.
    """
    result = KnowledgeValidation().validate(
        KnowledgeBundle([evidenced_concept]), source_index
    )
    codec = RecordCodec()
    assert codec.decode(codec.encode(result), type(result)) == result
