"""
Committed snapshots preserve source identity and exclusions.
"""

from __future__ import annotations

import shutil
from pathlib import Path

from repository_knowledge.adapters.git_source import GitSource
from repository_knowledge.model import DiagnosticKind, SourceKind

from .support import CommittedProject, FixturePath

# %% revision isolation


def test_snapshot_reads_committed_content_after_working_tree_edit(
    committed_project: CommittedProject,
) -> None:
    """
    A committed-source read never substitutes an uncommitted file.
    """
    source_reader = GitSource(committed_project.root)
    before = source_reader.read_snapshot(committed_project.commit_identifier)
    shutil.copyfile(
        Path(__file__).parent / FixturePath.REPLACEMENT,
        committed_project.root / FixturePath.API,
    )
    after = source_reader.read_snapshot(committed_project.commit_identifier)
    assert after == before


def test_source_module_names_follow_project_source_roots(
    committed_project: CommittedProject,
) -> None:
    """
    Import identities omit the project directory and src prefix.
    """
    snapshot = GitSource(committed_project.root).read_snapshot(
        committed_project.commit_identifier
    )
    source = next(
        entry for entry in snapshot.sources if entry.relative_path == FixturePath.API
    )
    assert source.module_name == "first.api"
    assert source.distribution_name == "first"


def test_generated_mapping_is_excluded_before_blob_read(
    committed_project: CommittedProject,
) -> None:
    """
    Generated mapping contents are never loaded into source records.
    """
    generated = committed_project.root / FixturePath.GENERATED
    shutil.copyfile(Path(__file__).parent / FixturePath.REPLACEMENT, generated)
    committed_project.git("add", FixturePath.GENERATED)
    committed_project.git(
        "commit", "--quiet", "-m", "Generated mapping exclusion fixture"
    )
    snapshot = GitSource(committed_project.root).read_snapshot("HEAD")
    source = next(
        entry
        for entry in snapshot.sources
        if entry.relative_path == FixturePath.GENERATED
    )
    assert source.source_kind == SourceKind.EXCLUDED
    assert source.source_text == ""
    assert [
        entry.diagnostic_kind
        for entry in snapshot.diagnostics
        if entry.relative_path == FixturePath.GENERATED
    ] == [DiagnosticKind.GENERATED]


def test_snapshot_accounts_for_rst_source(committed_project: CommittedProject) -> None:
    """
    RST documentation is represented as readable source.
    """
    snapshot = GitSource(committed_project.root).read_snapshot(
        committed_project.commit_identifier
    )
    source = next(
        entry for entry in snapshot.sources if entry.relative_path == FixturePath.RST
    )
    assert source.source_kind == SourceKind.DOCUMENT
    assert source.source_text == (committed_project.root / FixturePath.RST).read_text(
        encoding="utf-8"
    )
