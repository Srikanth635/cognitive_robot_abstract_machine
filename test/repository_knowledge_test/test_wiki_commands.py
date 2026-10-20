"""
Wiki requests enforce the same source boundaries as structural navigation.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from repository_knowledge.adapters.okf import OkfDirectory
from repository_knowledge.adapters.python_source import PythonSource
from repository_knowledge.adapters.serialization import RecordCodec
from repository_knowledge.adapters.sqlite_index import SQLiteIndex
from repository_knowledge.adapters.working_tree import WorkingTreeSource
from repository_knowledge.composition import wiki_main
from repository_knowledge.concepts import (
    Concept,
    ConceptPage,
    KnowledgeBundle,
    EvidenceState,
)
from repository_knowledge.exceptions import (
    SourceRevisionMismatch,
    SourceSnapshotMismatch,
)
from repository_knowledge.operations.indexing import IndexBuilder

from .support import CommittedProject, FixturePath
from typing_extensions import TYPE_CHECKING

if TYPE_CHECKING:
    from repository_knowledge.model import IndexData

# %% source-bound command results


def test_wiki_command_returns_typed_concept_evidence(
    committed_project: CommittedProject,
    indexed_storage: SQLiteIndex,
    evidenced_concept: Concept,
    source_index: IndexData,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """
    Agent output preserves curated metadata and the selected source snapshot.
    """
    bundle = tmp_path / "bundle"
    OkfDirectory(bundle).write_bundle(KnowledgeBundle([evidenced_concept]))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "cram-wiki",
            "--repository",
            str(committed_project.root),
            "--database",
            str(indexed_storage.database_path),
            "--bundle",
            str(bundle),
            "find",
            "base",
        ],
    )
    wiki_main()
    result = RecordCodec().decode(capsys.readouterr().out, ConceptPage)
    assert result.concepts[0].relative_path == evidenced_concept.relative_path
    assert (
        result.validation.snapshot_identifier
        == source_index.manifest.snapshot_identifier
    )
    assert result.validation.concepts[0].evidence_state == EvidenceState.CURRENT


def test_wiki_command_refuses_a_changed_revision(
    committed_project: CommittedProject,
    indexed_storage: SQLiteIndex,
    evidenced_concept: Concept,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    A changed requested commit cannot reuse the previous source index.
    """
    bundle = tmp_path / "bundle"
    OkfDirectory(bundle).write_bundle(KnowledgeBundle([evidenced_concept]))
    committed_project.git(
        "commit", "--quiet", "--allow-empty", "-m", "Advance source fixture"
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "cram-wiki",
            "--repository",
            str(committed_project.root),
            "--database",
            str(indexed_storage.database_path),
            "--bundle",
            str(bundle),
            "validate",
        ],
    )
    with pytest.raises(SourceRevisionMismatch):
        wiki_main()


def test_wiki_command_refuses_changed_working_source(
    committed_project: CommittedProject,
    evidenced_concept: Concept,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Editing source after capture prevents current-source wiki answers.
    """
    index = IndexBuilder(
        WorkingTreeSource(committed_project.root), PythonSource()
    ).build("HEAD")
    storage = SQLiteIndex(tmp_path / "working.sqlite3")
    storage.replace(index)
    bundle = tmp_path / "bundle"
    OkfDirectory(bundle).write_bundle(KnowledgeBundle([evidenced_concept]))
    replacement = Path(__file__).parent / FixturePath.REPLACEMENT
    (committed_project.root / FixturePath.API).write_bytes(replacement.read_bytes())
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "cram-wiki",
            "--repository",
            str(committed_project.root),
            "--database",
            str(storage.database_path),
            "--scope",
            "working-tree",
            "--bundle",
            str(bundle),
            "validate",
        ],
    )
    with pytest.raises(SourceSnapshotMismatch):
        wiki_main()
