"""
Canonical publishing reads captured Git documents rather than working files.
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from repository_knowledge.adapters.committed_knowledge import CommittedKnowledge
from repository_knowledge.adapters.git_source import GitSource
from repository_knowledge.adapters.okf import OkfCodec
from repository_knowledge.adapters.python_source import PythonSource
from repository_knowledge.adapters.exceptions import InvalidKnowledgeDocument
from repository_knowledge.concepts import KnowledgeBundle
from repository_knowledge.model import IndexData
from repository_knowledge.operations.indexing import IndexBuilder

from .support import CommittedProject

# %% captured bundle identity


def test_captured_bundle_ignores_later_working_file_edits(
    committed_project: CommittedProject, publication_bundle: KnowledgeBundle
) -> None:
    """
    The published proposal comes from one selected committed capture.
    """
    concept = publication_bundle.concepts[0]
    directory = committed_project.root / "repository_knowledge/wiki"
    path = directory / concept.relative_path
    path.parent.mkdir(parents=True)
    path.write_text(OkfCodec().encode(concept), encoding="utf-8")
    committed_project.git("add", ".")
    committed_project.git("commit", "--quiet", "-m", "Captured wiki proposal")
    index = IndexBuilder(GitSource(committed_project.root), PythonSource()).build(
        "HEAD"
    )
    path.write_text(
        OkfCodec().encode(replace(concept, body="Uncommitted replacement")),
        encoding="utf-8",
    )
    captured = CommittedKnowledge().read_bundle(index, "repository_knowledge/wiki")
    assert captured.concepts == publication_bundle.concepts


def test_missing_captured_bundle_is_not_an_empty_publication(
    source_index: IndexData,
) -> None:
    """
    Missing committed content fails visibly rather than publishing nothing.
    """
    with pytest.raises(InvalidKnowledgeDocument):
        CommittedKnowledge().read_bundle(source_index, "repository_knowledge/wiki")
