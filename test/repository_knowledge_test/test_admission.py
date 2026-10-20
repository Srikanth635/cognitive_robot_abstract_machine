"""
Unprivileged checks cannot accept policy weakening or lifecycle promotion.
"""

from __future__ import annotations

from dataclasses import replace

from repository_knowledge.adapters.digests import CanonicalDigest
from repository_knowledge.concepts import KnowledgeBundle, Lifecycle
from repository_knowledge.model import IndexData
from repository_knowledge.operations.admission import ProposalAdmission
from repository_knowledge.publication import TrustIssue

# %% proposal admission


def test_admission_refuses_new_stable_declarations(
    source_index: IndexData, publication_bundle: KnowledgeBundle
) -> None:
    """
    A producer cannot grant canonical lifecycle status during a PR check.
    """
    proposed = KnowledgeBundle(
        [replace(publication_bundle.concepts[0], lifecycle=Lifecycle.STABLE)]
    )
    report = ProposalAdmission(CanonicalDigest()).check(
        KnowledgeBundle([]), proposed, source_index, [], "same", "same"
    )
    assert report.issues == [TrustIssue.LIFECYCLE]


def test_admission_refuses_deleted_concept_identity(
    source_index: IndexData, publication_bundle: KnowledgeBundle
) -> None:
    """
    Existing paths require explicit tombstones rather than disappearing links.
    """
    report = ProposalAdmission(CanonicalDigest()).check(
        publication_bundle, KnowledgeBundle([]), source_index, [], "same", "same"
    )
    assert report.issues == [TrustIssue.LIFECYCLE]


def test_admission_refuses_proposed_checker_changes(
    source_index: IndexData, publication_bundle: KnowledgeBundle
) -> None:
    """
    A changed policy or checker needs a separately reviewed trust-root update.
    """
    report = ProposalAdmission(CanonicalDigest()).check(
        publication_bundle,
        publication_bundle,
        source_index,
        ["repository_knowledge/policy/publication.toml"],
        "same",
        "same",
    )
    assert report.issues == [TrustIssue.GOVERNANCE_CHANGE]


def test_admission_refuses_nondeterministic_extraction(
    source_index: IndexData, publication_bundle: KnowledgeBundle
) -> None:
    """
    Repeated clean extraction must agree before publishing its evidence.
    """
    report = ProposalAdmission(CanonicalDigest()).check(
        publication_bundle, publication_bundle, source_index, [], "first", "second"
    )
    assert report.issues == [TrustIssue.EXTRACTION_CHANGED]


def test_admission_accepts_source_backed_drafts(
    source_index: IndexData, publication_bundle: KnowledgeBundle
) -> None:
    """
    Draft knowledge can pass machine checks without pretending to be reviewed.
    """
    report = ProposalAdmission(CanonicalDigest()).check(
        KnowledgeBundle([]), publication_bundle, source_index, [], "same", "same"
    )
    assert report.issues == []
