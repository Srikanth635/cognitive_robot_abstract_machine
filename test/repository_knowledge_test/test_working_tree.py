"""
Working-tree capture distinguishes editing state from committed evidence.
"""

from __future__ import annotations

import shutil
from pathlib import Path

from repository_knowledge.adapters.git_source import GitArgument

from repository_knowledge.adapters.working_tree import WorkingTreeSource
from repository_knowledge.model import SourceScope

from .support import CommittedProject, FixturePath

# %% working-file identity


def test_working_tree_snapshot_captures_uncommitted_content(
    committed_project: CommittedProject,
) -> None:
    """
    The overlay contains file contents that committed mode cannot claim.
    """
    replacement = Path(__file__).parent / FixturePath.REPLACEMENT
    shutil.copyfile(replacement, committed_project.root / FixturePath.API)
    snapshot = WorkingTreeSource(committed_project.root).read_snapshot(
        GitArgument.CHECKOUT
    )
    source = next(
        source for source in snapshot.sources if source.relative_path == FixturePath.API
    )
    assert source.source_text == replacement.read_text(encoding="utf-8")
    assert snapshot.manifest.source_scope == SourceScope.WORKING_TREE


def test_working_tree_snapshot_differs_between_checkout_paths(
    committed_project: CommittedProject, tmp_path: Path
) -> None:
    """
    Identical content in another checkout cannot reuse an overlay identity.
    """
    other = tmp_path / "other"
    committed_project.git(
        "worktree",
        "add",
        "--quiet",
        "--detach",
        str(other),
        committed_project.commit_identifier,
    )
    first = WorkingTreeSource(committed_project.root).read_snapshot(
        GitArgument.CHECKOUT
    )
    second = WorkingTreeSource(other).read_snapshot(GitArgument.CHECKOUT)
    assert first.manifest.source_digest == second.manifest.source_digest
    assert first.manifest.snapshot_identifier != second.manifest.snapshot_identifier


def test_untracked_source_requires_explicit_inclusion(
    committed_project: CommittedProject,
) -> None:
    """
    Untracked files enter only the explicitly selected overlay.
    """
    untracked_path = FixturePath.ADDED
    replacement = Path(__file__).parent / FixturePath.REPLACEMENT
    shutil.copyfile(replacement, committed_project.root / untracked_path)
    excluded = WorkingTreeSource(committed_project.root).read_snapshot(
        GitArgument.CHECKOUT
    )
    included = WorkingTreeSource(
        committed_project.root, included_untracked_paths=(untracked_path,)
    ).read_snapshot(GitArgument.CHECKOUT)
    assert [
        source.relative_path
        for source in excluded.sources
        if source.relative_path == untracked_path
    ] == []
    assert [
        source.source_text
        for source in included.sources
        if source.relative_path == untracked_path
    ] == [replacement.read_text(encoding="utf-8")]


def test_staged_addition_is_captured_without_untracked_inclusion(
    committed_project: CommittedProject,
) -> None:
    """
    A staged new source is part of the tracked working-file inventory.
    """
    added_path = FixturePath.ADDED
    shutil.copyfile(
        Path(__file__).parent / FixturePath.REPLACEMENT,
        committed_project.root / added_path,
    )
    committed_project.git("add", added_path)
    snapshot = WorkingTreeSource(committed_project.root).read_snapshot(
        GitArgument.CHECKOUT
    )
    assert [
        source.relative_path
        for source in snapshot.sources
        if source.relative_path == added_path
    ] == [added_path]


def test_working_tree_deletion_removes_committed_source(
    committed_project: CommittedProject,
) -> None:
    """
    Deleted files cannot supply ghost facts in the current overlay.
    """
    (committed_project.root / FixturePath.API).unlink()
    snapshot = WorkingTreeSource(committed_project.root).read_snapshot(
        GitArgument.CHECKOUT
    )
    assert [
        source.relative_path
        for source in snapshot.sources
        if source.relative_path == FixturePath.API
    ] == []


def test_index_changes_invalidate_identical_working_contents(
    committed_project: CommittedProject,
) -> None:
    """
    Staging changes is reflected even when the current source bytes agree.
    """
    reader = WorkingTreeSource(committed_project.root)
    shutil.copyfile(
        Path(__file__).parent / FixturePath.REPLACEMENT,
        committed_project.root / FixturePath.API,
    )
    before = reader.read_snapshot(GitArgument.CHECKOUT)
    committed_project.git("add", FixturePath.API)
    after = reader.read_snapshot(GitArgument.CHECKOUT)
    assert before.sources == after.sources
    assert before.manifest.snapshot_identifier != after.manifest.snapshot_identifier
