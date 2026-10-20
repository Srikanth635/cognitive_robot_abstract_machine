"""
Mutation, interchange and metadata boundaries for curated concepts.
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from repository_knowledge.adapters.okf import OkfDirectory
from repository_knowledge.concepts import (
    Concept,
    EvidenceState,
    KnowledgeBundle,
    KnowledgeIssue,
    Lifecycle,
)
from repository_knowledge.exceptions import InvalidConceptPath
from repository_knowledge.adapters.exceptions import InvalidKnowledgeDocument
from repository_knowledge.operations.exceptions import InvalidConceptTransition
from repository_knowledge.model import IndexData, SourceLocation
from repository_knowledge.operations.knowledge import (
    KnowledgeValidation,
    ConceptLifecycle,
    MetadataPolicy,
    KnowledgeQueries,
)
from typing_extensions import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

# %% current source checks


@pytest.mark.parametrize(
    "mutation,issue",
    [
        ("delete", KnowledgeIssue.MISSING_SOURCE),
        ("symbol", KnowledgeIssue.MISSING_SYMBOL),
        ("span", KnowledgeIssue.CHANGED_SYMBOL_LOCATION),
        ("range", KnowledgeIssue.INVALID_SOURCE_RANGE),
    ],
)
def test_source_mutations_reject_original_evidence(
    evidenced_concept: Concept,
    source_index: IndexData,
    mutation: str,
    issue: KnowledgeIssue,
) -> None:
    """
    Deleted, missing and moved evidence cannot remain current.
    """
    provenance = evidenced_concept.provenance
    evidence = provenance.evidence[0]
    if mutation == "delete":
        source_index = replace(source_index, sources=[])
    elif mutation == "symbol":
        source_index = replace(source_index, symbols=[])
    else:
        location = SourceLocation(
            evidence.location.relative_path,
            evidence.location.start_line,
            9999 if mutation == "range" else evidence.location.end_line + 1,
        )
        evidenced_concept = replace(
            evidenced_concept,
            provenance=replace(
                provenance, evidence=[replace(evidence, location=location)]
            ),
        )
    report = KnowledgeValidation().validate(
        KnowledgeBundle([evidenced_concept]), source_index
    )
    assert issue in report.concepts[0].evidence_issues
    assert report.concepts[0].evidence_state == EvidenceState.STALE


def test_unchanged_evidence_is_invalidated_through_dependency(
    evidenced_concept: Concept, source_index: IndexData
) -> None:
    """
    Reverse invalidation reaches a concept with independently unchanged bytes.
    """
    dependency = replace(
        evidenced_concept,
        relative_path="packages/changed.md",
        provenance=replace(
            evidenced_concept.provenance,
            evidence=[
                replace(evidenced_concept.provenance.evidence[0], content_digest="old")
            ],
        ),
    )
    dependent = replace(
        evidenced_concept,
        provenance=replace(
            evidenced_concept.provenance, depends_on=[dependency.relative_path]
        ),
    )
    result = KnowledgeValidation().validate(
        KnowledgeBundle([dependency, dependent]), source_index
    )
    assert result.concepts[1].evidence_issues == [KnowledgeIssue.STALE_DEPENDENCY]


def test_missing_dependency_fails_conservatively(
    evidenced_concept: Concept, source_index: IndexData
) -> None:
    """
    A missing prerequisite does not leave dependent content current.
    """
    concept = replace(
        evidenced_concept,
        provenance=replace(evidenced_concept.provenance, depends_on=["missing.md"]),
    )
    result = KnowledgeValidation().validate(KnowledgeBundle([concept]), source_index)
    assert result.concepts[0].evidence_state == EvidenceState.STALE


@pytest.mark.parametrize(
    "timestamp",
    [
        "2026-02-30T00:00:00Z",
        "2026-10-20",
        "2026-10-20T00:00:00",
        "2026-10-20T00:00:00+24:00",
    ],
)
def test_invalid_timestamps_are_policy_findings(timestamp: str) -> None:
    """
    Invalid calendar dates and missing offsets are not accepted as instants.
    """
    assert MetadataPolicy.parse_instant(timestamp) is None


# %% exchange and proposal safety


def test_bundle_exchange_preserves_concepts_and_indexes(
    portable_concept: Concept, tmp_path: Path
) -> None:
    """
    Exported bundles retain reserved navigation and arbitrary metadata.
    """
    bundle = KnowledgeBundle([portable_concept], {"index.md": "# Entry\n"})
    destination = OkfDirectory(tmp_path / "export")
    destination.write_bundle(bundle)
    assert destination.read_bundle() == bundle


def test_export_refuses_existing_directory(
    portable_concept: Concept, tmp_path: Path
) -> None:
    """
    A proposal export cannot overwrite an existing bundle.
    """
    with pytest.raises(InvalidKnowledgeDocument):
        OkfDirectory(tmp_path).write_bundle(KnowledgeBundle([portable_concept]))


@pytest.mark.parametrize(
    "path",
    ["../outside.md", "/absolute.md", "index.md", "log.md", "a/../b.md", "a\\b.md"],
)
def test_concept_paths_stay_within_bundle(path: str) -> None:
    """
    Portable identities cannot escape the bundle or use reserved filenames.
    """
    with pytest.raises(InvalidConceptPath):
        Concept(path, "Example")


def test_replacement_cycle_is_reported(
    evidenced_concept: Concept, source_index: IndexData
) -> None:
    """
    Tombstones cannot silently resolve through a replacement cycle.
    """
    concept = replace(
        evidenced_concept,
        lifecycle=Lifecycle.DEPRECATED,
        provenance=replace(
            evidenced_concept.provenance, replacement=evidenced_concept.relative_path
        ),
    )
    result = KnowledgeValidation().validate(KnowledgeBundle([concept]), source_index)
    assert KnowledgeIssue.REPLACEMENT_CYCLE in result.concepts[0].policy_issues


def test_draft_proposal_removes_claimed_verification(
    evidenced_concept: Concept,
) -> None:
    """
    A producer cannot carry a declared approval onto changed content.
    """
    concept = replace(evidenced_concept, metadata={"verified": {"by": "human:owner"}})
    assert ConceptLifecycle().transition(concept, Lifecycle.DRAFT).metadata == {}


def test_deprecation_needs_replacement(evidenced_concept: Concept) -> None:
    """
    A deleted concept proposal must retain a meaningful successor link.
    """
    with pytest.raises(InvalidConceptTransition):
        ConceptLifecycle().transition(evidenced_concept, Lifecycle.DEPRECATED)


# %% agent discovery


def test_concept_search_is_bounded_and_snapshot_bound(
    evidenced_concept: Concept, source_index: IndexData
) -> None:
    """
    Agent discovery returns the explanation and its separate validation.
    """
    result = KnowledgeQueries().find(
        KnowledgeBundle([evidenced_concept]), source_index, "base", 1, 0
    )
    assert result.concepts == [evidenced_concept]
    assert (
        result.validation.snapshot_identifier
        == source_index.manifest.snapshot_identifier
    )
    assert result.total_matches == 1
    assert result.next_offset is None
