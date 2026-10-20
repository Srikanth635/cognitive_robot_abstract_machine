"""
Reserved documents and adversarial metadata remain separate from authority.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
import yaml

from repository_knowledge.adapters.okf import OkfCodec, OkfDirectory
from repository_knowledge.adapters.serialization import RecordCodec
from repository_knowledge.concepts import Concept, KnowledgeBundle, KnowledgeIssue
from repository_knowledge.adapters.exceptions import InvalidKnowledgeDocument
from repository_knowledge.operations.knowledge import (
    KnowledgeValidation,
    ConceptLifecycle,
)
from typing_extensions import TYPE_CHECKING

if TYPE_CHECKING:
    from repository_knowledge.model import IndexData

# %% bundle framing


def test_reserved_index_metadata_is_checked(
    tmp_path: Path, source_index: IndexData
) -> None:
    """
    Format results include forbidden metadata on a nested index.
    """
    bundle_root = tmp_path / "bundle"
    nested = bundle_root / "packages"
    nested.mkdir(parents=True)
    fixture = Path(__file__).parent / "dataset/knowledge/invalid-index.md"
    (nested / "index.md").write_text(
        fixture.read_text(encoding="utf-8"), encoding="utf-8"
    )
    result = KnowledgeValidation().validate(
        OkfDirectory(bundle_root).read_bundle(), source_index
    )
    assert result.navigation_issues == {
        "packages/index.md": [KnowledgeIssue.INVALID_NAVIGATION]
    }


def test_recursive_yaml_metadata_is_rejected() -> None:
    """
    An alias cycle cannot exhaust recursive metadata normalization.
    """
    fixture = Path(__file__).parent / "dataset/knowledge/recursive.md"
    with pytest.raises(InvalidKnowledgeDocument):
        OkfCodec().decode("recursive.md", fixture.read_text(encoding="utf-8"))


def test_unattributed_evidence_fails_repository_policy(
    evidenced_concept: Concept, source_index: IndexData
) -> None:
    """
    A source binding must also have a portable per-claim attribution.
    """
    report = KnowledgeValidation().validate(
        KnowledgeBundle([evidenced_concept]), source_index
    )
    assert KnowledgeIssue.INVALID_ATTRIBUTION in report.concepts[0].policy_issues


def test_changed_linked_concept_invalidates_current_neighbor(
    evidenced_concept: Concept, source_index: IndexData
) -> None:
    """
    Untyped concept relationships conservatively propagate invalidation.
    """
    changed = replace(
        evidenced_concept,
        relative_path="changed.md",
        provenance=replace(
            evidenced_concept.provenance,
            evidence=[
                replace(evidenced_concept.provenance.evidence[0], content_digest="old")
            ],
        ),
    )
    neighbor = replace(evidenced_concept, links=[changed.relative_path])
    report = KnowledgeValidation().validate(
        KnowledgeBundle([changed, neighbor]), source_index
    )
    assert report.concepts[1].evidence_issues == [KnowledgeIssue.STALE_DEPENDENCY]


def test_portable_metadata_survives_owned_serialization(
    portable_concept: Concept,
) -> None:
    """
    Opaque portable fields remain data at the typed result boundary.
    """
    codec = RecordCodec()
    assert codec.decode(codec.encode(portable_concept), Concept) == portable_concept


def test_export_checks_all_paths_before_writing(
    portable_concept: Concept, tmp_path: Path
) -> None:
    """
    Invalid navigation paths cannot leave a partially written proposal.
    """
    root = tmp_path / "new-bundle"
    bundle = KnowledgeBundle([portable_concept], {"../outside.md": "invalid"})
    with pytest.raises(InvalidKnowledgeDocument):
        OkfDirectory(root).write_bundle(bundle)
    assert root.exists() is False


def test_package_requires_declaration_evidence(
    evidenced_concept: Concept, source_index: IndexData
) -> None:
    """
    A package explanation needs at least one source declaration binding.
    """
    provenance = evidenced_concept.provenance
    concept = replace(
        evidenced_concept,
        provenance=replace(
            provenance, evidence=[replace(provenance.evidence[0], symbol_identifier="")]
        ),
    )
    report = KnowledgeValidation().validate(KnowledgeBundle([concept]), source_index)
    assert KnowledgeIssue.MISSING_TYPE_EVIDENCE in report.concepts[0].policy_issues


def test_moved_tombstone_survives_export(evidenced_concept: Concept) -> None:
    """
    Computed link and attribution metadata agrees with the exported tombstone.
    """
    from_repository = OkfCodec().decode(
        evidenced_concept.relative_path, OkfCodec().encode(evidenced_concept)
    )
    bundle = ConceptLifecycle().relocate(
        KnowledgeBundle([from_repository]),
        from_repository.relative_path,
        "packages/moved.md",
    )
    tombstone = bundle.concepts[0]
    assert (
        OkfCodec().decode(tombstone.relative_path, OkfCodec().encode(tombstone))
        == tombstone
    )


def test_move_preparation_preserves_relative_link_destinations(
    evidenced_concept: Concept,
) -> None:
    """
    A cross-directory proposal preserves neighboring and page-local links.
    """
    concept = replace(
        evidenced_concept,
        body="See [neighbor](./neighbor.md) and [section](#details).\n",
    )
    prepared = OkfCodec().prepare_move(concept, "workflows/moved.md")
    assert prepared.links == ["packages/neighbor.md", "workflows/moved.md"]


@pytest.mark.parametrize("create_file", [False, True])
def test_invalid_bundle_root_is_not_an_empty_valid_bundle(
    tmp_path: Path, create_file: bool
) -> None:
    """
    A missing directory or regular file cannot pass as an empty bundle.
    """
    root = tmp_path / "bundle"
    if create_file:
        root.touch()
    with pytest.raises(InvalidKnowledgeDocument):
        OkfDirectory(root).read_bundle()


def test_repository_extensions_preserve_keys_used_by_nested_evidence(
    evidenced_concept: Concept,
) -> None:
    """
    A field known inside evidence must not erase an unknown root extension.
    """
    fixture = Path(__file__).parent / "dataset/knowledge/extensions.yaml"
    extensions = yaml.safe_load(fixture.read_text(encoding="utf-8"))
    concept = replace(
        evidenced_concept,
        provenance=replace(evidenced_concept.provenance, extensions=extensions),
    )
    codec = OkfCodec()
    decoded = codec.decode(concept.relative_path, codec.encode(concept))
    assert decoded.provenance.extensions == concept.provenance.extensions
