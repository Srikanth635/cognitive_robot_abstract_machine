"""
Working-file capture preserves exclusion and checkout safety policies.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import pytest

from repository_knowledge.adapters.git_source import GitArgument, SourceFilename
from repository_knowledge.adapters.working_tree import WorkingTreeSource
from repository_knowledge.exceptions import InvalidUntrackedPath
from repository_knowledge.interfaces.cli import CommandParser, LayoutPath
from repository_knowledge.model import SourceKind, SourceScope

from .support import CommittedProject, FixturePath

# %% inclusion boundaries


@pytest.mark.parametrize("selection", ["../outside", "/outside", "C:/outside"])
def test_untracked_inclusion_cannot_escape_repository(
    committed_project: CommittedProject, selection: str
) -> None:
    """
    Only repository-relative source inclusion is accepted.
    """
    with pytest.raises(InvalidUntrackedPath):
        WorkingTreeSource(committed_project.root, included_untracked_paths=(selection,))


def test_default_scope_caches_are_separate(committed_project: CommittedProject) -> None:
    """
    Ordinary CLI use preserves committed evidence beside its overlay.
    """
    prefix = ["--repository", str(committed_project.root)]
    committed = CommandParser().parse(prefix + ["index"])
    overlay = CommandParser().parse(
        prefix + ["--scope", SourceScope.WORKING_TREE, "index"]
    )
    assert committed.database_path.name == LayoutPath.DATABASE
    assert overlay.database_path.name == LayoutPath.OVERLAY_DATABASE


def test_ignored_untracked_source_is_not_included(
    committed_project: CommittedProject,
) -> None:
    """
    Explicit inclusion continues to respect Git ignore policy.
    """
    dataset = Path(__file__).parent
    shutil.copyfile(
        dataset / FixturePath.IGNORE_POLICY,
        committed_project.root / ".gitignore",
    )
    ignored_path = "first/src/first/ignored.py"
    shutil.copyfile(
        dataset / FixturePath.REPLACEMENT, committed_project.root / ignored_path
    )
    snapshot = WorkingTreeSource(
        committed_project.root, included_untracked_paths=(ignored_path,)
    ).read_snapshot(GitArgument.CHECKOUT)
    assert [
        source.relative_path
        for source in snapshot.sources
        if source.relative_path == ignored_path
    ] == []


def test_generated_working_mapping_is_never_opened(
    committed_project: CommittedProject, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    Staging a generated mapping cannot cause a filesystem content read.
    """
    shutil.copyfile(
        Path(__file__).parent / FixturePath.REPLACEMENT,
        committed_project.root / FixturePath.GENERATED,
    )
    committed_project.git("add", FixturePath.GENERATED)
    original_read = Path.read_bytes

    def read_allowed_source(path: Path) -> bytes:
        """
        Reject content access to the generated mapping sentinel.
        """
        if path.name == SourceFilename.GENERATED_MAPPING:
            pytest.fail("Generated mapping contents must not be opened.")
        return original_read(path)

    monkeypatch.setattr(Path, "read_bytes", read_allowed_source)
    snapshot = WorkingTreeSource(committed_project.root).read_snapshot(
        GitArgument.CHECKOUT
    )
    source = next(
        source
        for source in snapshot.sources
        if source.relative_path == FixturePath.GENERATED
    )
    assert source.source_kind == SourceKind.EXCLUDED
    assert source.source_text == ""


@pytest.mark.skipif(
    sys.platform == "win32",
    reason="Creating symbolic links requires Windows host privileges; Linux CI exercises this policy.",
)
def test_directory_symlink_cannot_supply_source(
    committed_project: CommittedProject, tmp_path: Path
) -> None:
    """
    Tracked paths cannot read source through a replaced directory.
    """
    directory = committed_project.root / "first/src/first"
    directory.rename(committed_project.root / "original-package")
    outside = tmp_path / "outside"
    outside.mkdir()
    shutil.copyfile(Path(__file__).parent / FixturePath.REPLACEMENT, outside / "api.py")
    directory.symlink_to(outside, target_is_directory=True)
    snapshot = WorkingTreeSource(committed_project.root).read_snapshot(
        GitArgument.CHECKOUT
    )
    source = next(
        source for source in snapshot.sources if source.relative_path == FixturePath.API
    )
    assert source.source_kind == SourceKind.EXCLUDED
    assert source.source_text == ""
