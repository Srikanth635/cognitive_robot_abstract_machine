"""
Wiki evidence distinguishes text-equivalent checkout bytes from source edits.
"""

from __future__ import annotations

from dataclasses import replace
from codecs import BOM_UTF8

import pytest

from repository_knowledge.adapters.python_source import PythonSource
from repository_knowledge.adapters.working_tree import WorkingTreeSource
from repository_knowledge.concepts import (
    Concept,
    KnowledgeBundle,
    EvidenceState,
    ContentComparison,
)
from repository_knowledge.model import IndexData
from repository_knowledge.adapters.git_source import GitArgument
from repository_knowledge.operations.indexing import IndexBuilder
from repository_knowledge.operations.knowledge import KnowledgeValidation

from .support import CommittedProject

# %% text-equivalent checkout evidence


@pytest.fixture
def line_ending_index(
    committed_project: CommittedProject, evidenced_concept: Concept
) -> IndexData:
    """
    Capture the same source text with the opposite supported line endings.
    """
    path = evidenced_concept.provenance.evidence[0].location.relative_path
    source_file = committed_project.root / path
    original_bytes = source_file.read_bytes()
    line_feed_bytes = original_bytes.replace(b"\r\n", b"\n")
    changed_bytes = (
        line_feed_bytes
        if b"\r\n" in original_bytes
        else line_feed_bytes.replace(b"\n", b"\r\n")
    )
    source_file.write_bytes(changed_bytes)
    return IndexBuilder(
        WorkingTreeSource(committed_project.root), PythonSource()
    ).build(GitArgument.CHECKOUT)


def test_windows_line_endings_preserve_source_evidence(
    line_ending_index: IndexData, evidenced_concept: Concept
) -> None:
    """
    The original byte digest can prove LF/CRLF equivalence without rebinding.
    """
    path = evidenced_concept.provenance.evidence[0].location.relative_path
    report = KnowledgeValidation().validate(
        KnowledgeBundle([evidenced_concept]), line_ending_index
    )
    assert report.concepts[0].evidence_state == EvidenceState.CURRENT
    assert report.concepts[0].text_equivalent_paths == [path]


def test_unrecognized_byte_changes_do_not_gain_text_equivalence(
    committed_project: CommittedProject, evidenced_concept: Concept
) -> None:
    """
    A changed digest must also match an actual supported byte representation.
    """
    path = evidenced_concept.provenance.evidence[0].location.relative_path
    index = IndexBuilder(
        WorkingTreeSource(committed_project.root), PythonSource()
    ).build(GitArgument.CHECKOUT)
    index = replace(
        index,
        sources=[
            (
                replace(source, content_digest="unrecognized-bytes")
                if source.relative_path == path
                else source
            )
            for source in index.sources
        ],
    )
    report = KnowledgeValidation().validate(KnowledgeBundle([evidenced_concept]), index)
    assert report.concepts[0].evidence_state == EvidenceState.STALE


def test_exact_bytes_policy_refuses_line_ending_changes(
    line_ending_index: IndexData, evidenced_concept: Concept
) -> None:
    """
    The stricter policy remains available when byte equality is required.
    """
    report = KnowledgeValidation(ContentComparison.EXACT_BYTES).validate(
        KnowledgeBundle([evidenced_concept]), line_ending_index
    )
    assert report.concepts[0].evidence_state == EvidenceState.STALE
    assert report.concepts[0].text_equivalent_paths == []


def test_byte_order_mark_is_not_line_ending_equivalence(
    committed_project: CommittedProject, evidenced_concept: Concept
) -> None:
    """
    Encoding-marker changes cannot be admitted through the line-ending policy.
    """
    path = evidenced_concept.provenance.evidence[0].location.relative_path
    source_file = committed_project.root / path
    source_file.write_bytes(BOM_UTF8 + source_file.read_bytes())
    index = IndexBuilder(
        WorkingTreeSource(committed_project.root), PythonSource()
    ).build(GitArgument.CHECKOUT)
    report = KnowledgeValidation().validate(KnowledgeBundle([evidenced_concept]), index)
    assert report.concepts[0].evidence_state == EvidenceState.STALE
