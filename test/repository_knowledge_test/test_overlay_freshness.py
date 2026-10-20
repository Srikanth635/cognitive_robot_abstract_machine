"""
Current-source queries reject edited or concurrently captured evidence.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from repository_knowledge.adapters.git_source import GitArgument

from repository_knowledge.adapters.python_source import PythonSource
from repository_knowledge.adapters.working_tree import WorkingTreeSource
from repository_knowledge.exceptions import SourceChangedDuringCapture
from repository_knowledge.operations.freshness import FreshnessCheck
from repository_knowledge.operations.indexing import IndexBuilder
from repository_knowledge.operations.queries import IndexContents

from .dataset.changing_capture import ChangingCapture
from .support import CommittedProject, FixturePath

# %% current source agreement


def test_overlay_freshness_detects_an_uncommitted_edit(
    committed_project: CommittedProject,
) -> None:
    """
    HEAD equality cannot make changed working contents current.
    """
    reader = WorkingTreeSource(committed_project.root)
    previous = IndexBuilder(reader, PythonSource()).build(GitArgument.CHECKOUT)
    shutil.copyfile(
        Path(__file__).parent / FixturePath.REPLACEMENT,
        committed_project.root / FixturePath.API,
    )
    report = FreshnessCheck(IndexContents(previous), reader).check(GitArgument.CHECKOUT)
    assert report.matches_requested_revision is True
    assert report.matches_requested_source is False


def test_capture_rejects_source_changed_between_observations(
    committed_project: CommittedProject,
) -> None:
    """
    A source edit during capture cannot publish mixed file evidence.
    """
    reader = ChangingCapture(
        committed_project.root,
        replacement_path=Path(__file__).parent / FixturePath.REPLACEMENT,
        changed_path=FixturePath.API,
    )
    with pytest.raises(SourceChangedDuringCapture):
        reader.read_snapshot(GitArgument.CHECKOUT)
