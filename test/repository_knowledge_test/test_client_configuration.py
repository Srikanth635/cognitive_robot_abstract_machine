"""
Client authority is selected by the operator, outside tool arguments.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import Mock

import pytest
from pytest import MonkeyPatch

from repository_knowledge.adapters.canonical_process import PublicationProcess
from repository_knowledge.adapters.serialization import RecordCodec
from repository_knowledge.exceptions import InvalidQuery
from repository_knowledge.interfaces.clients import ClientParser
from repository_knowledge.publication import (
    ConsumptionResult,
    ConsumptionState,
    TrustIssue,
)

from .support import CommittedProject

# %% operator configuration


def test_mcp_startup_selects_baseline_and_canonical_authority(
    committed_project: CommittedProject, tmp_path: Path
) -> None:
    """
    Baseline paths and trust pins never come from a query's tool payload.
    """
    checker = committed_project.commit_identifier
    configuration = ClientParser().parse(
        [
            "--repository",
            str(committed_project.root),
            "--trusted-repository",
            str(committed_project.root),
            "--checker-revision",
            checker,
            "--artifact-id",
            "20",
            "--against-database",
            str(tmp_path / "baseline.sqlite3"),
            "--against-revision",
            checker,
        ],
        server=True,
    )
    assert configuration.canonical.artifact_identifier == 20
    assert configuration.canonical.checker_revision == checker
    assert configuration.arguments.baseline_database == tmp_path / "baseline.sqlite3"
    assert configuration.arguments.baseline_revision == checker


def test_partial_canonical_selection_is_refused(
    committed_project: CommittedProject,
) -> None:
    """
    An artifact number alone cannot select its own executable trust root.
    """
    with pytest.raises(InvalidQuery):
        ClientParser().parse(
            [
                "--repository",
                str(committed_project.root),
                "--artifact-id",
                "20",
                "check-freshness",
            ]
        )


def test_canonical_client_uses_only_the_trusted_checker_process(
    committed_project: CommittedProject, monkeypatch: MonkeyPatch
) -> None:
    """
    The client delegates authentication to the pinned consumer, not local JSON.
    """
    verdict = ConsumptionResult(
        ConsumptionState.SOURCE_ONLY, None, [TrustIssue.WRONG_SOURCE]
    )
    run = Mock(
        return_value=subprocess.CompletedProcess(
            [], 2, RecordCodec().encode(verdict), ""
        )
    )
    monkeypatch.setattr(subprocess, "run", run)
    checker = committed_project.commit_identifier
    reader = PublicationProcess(
        committed_project.root,
        committed_project.root,
        checker,
        20,
        Path(sys.executable),
    )
    assert reader.read(checker) == verdict
    called = run.call_args
    assert called.kwargs["cwd"] == committed_project.root
    assert called.kwargs["env"]["PYTHONPATH"] == os.pathsep.join(
        str(committed_project.root / directory)
        for directory in ("repository_knowledge/src", "krrood/src")
    )
    assert called.args[0] == [
        sys.executable,
        "-m",
        "repository_knowledge",
        "publication",
        "--repository",
        str(committed_project.root),
        "--trusted-repository",
        str(committed_project.root),
        "--checker-revision",
        checker,
        "consume",
        "--revision",
        checker,
        "--artifact-id",
        "20",
    ]
