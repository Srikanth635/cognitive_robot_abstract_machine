"""
Trusted execution uses captured Git data and a complete pinned checker.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from repository_knowledge.adapters.digests import CanonicalDigest
from repository_knowledge.adapters.documents import DocumentSource, SourceCollection
from repository_knowledge.adapters.git_source import GitSource
from repository_knowledge.adapters.okf import OkfDirectory
from repository_knowledge.adapters.python_source import PythonSource
from repository_knowledge.adapters.serialization import RecordCodec
from repository_knowledge.concepts import KnowledgeBundle
from repository_knowledge.operations.indexing import IndexBuilder
from repository_knowledge.publication import AdmissionResult, ExtractionFingerprint

from .support import CommittedProject

# %% complete checker fixture


@pytest.fixture
def checker_project(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> CommittedProject:
    """
    Pin checker and dependency source without generated mappings or bytecode.
    """
    root = Path(__file__).parents[2]
    project = CommittedProject.create(tmp_path / "checker")
    for directory in ("repository_knowledge/src", "krrood/src"):
        shutil.copytree(
            root / directory,
            project.root / directory,
            ignore=shutil.ignore_patterns(
                "__pycache__", "*.pyc", "ormatic_interface.py"
            ),
        )
    policy = project.root / "repository_knowledge/policy/publication.toml"
    policy.parent.mkdir()
    shutil.copyfile(
        Path(__file__).parent / "dataset/knowledge/publication.toml", policy
    )
    project.git("add", ".")
    project.git("commit", "--quiet", "-m", "Complete checker fixture")
    project.commit_identifier = project.git("rev-parse", "HEAD")
    monkeypatch.setenv(
        "PYTHONPATH",
        os.pathsep.join(
            str(project.root / path)
            for path in ("repository_knowledge/src", "krrood/src")
        ),
    )
    monkeypatch.setenv("PYTHONDONTWRITEBYTECODE", "1")
    return project


# %% isolated command execution


def test_fresh_checker_process_reproduces_complete_extraction(
    checker_project: CommittedProject, committed_project: CommittedProject
) -> None:
    """
    A separately pinned interpreter agrees with all extraction record categories.
    """
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "repository_knowledge",
            "publication",
            "--repository",
            str(committed_project.root),
            "--trusted-repository",
            str(checker_project.root),
            "--checker-revision",
            checker_project.commit_identifier,
            "fingerprint",
            "--revision",
            committed_project.commit_identifier,
        ],
        cwd=checker_project.root,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    index = IndexBuilder(
        GitSource(committed_project.root),
        SourceCollection([PythonSource(), DocumentSource()]),
    ).build(committed_project.commit_identifier)
    assert RecordCodec().decode(
        result.stdout, ExtractionFingerprint
    ) == CanonicalDigest().fingerprint(index)


def test_admission_command_reextracts_source_in_separate_interpreters(
    checker_project: CommittedProject, publication_bundle: KnowledgeBundle
) -> None:
    """
    The complete checker admits valid wiki data without executing proposed code.
    """
    baseline = checker_project.commit_identifier
    OkfDirectory(checker_project.root / "repository_knowledge/wiki").write_bundle(
        publication_bundle
    )
    checker_project.git("add", ".")
    checker_project.git("commit", "--quiet", "-m", "Knowledge proposal fixture")
    revision = checker_project.git("rev-parse", "HEAD")
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "repository_knowledge",
            "publication",
            "--repository",
            str(checker_project.root),
            "--trusted-repository",
            str(checker_project.root),
            "--checker-revision",
            revision,
            "check",
            "--revision",
            revision,
            "--base-revision",
            baseline,
        ],
        cwd=checker_project.root,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    admission = RecordCodec().decode(result.stdout, AdmissionResult)
    assert admission.report.issues == []
    assert admission.manifest.commit_identifier == revision


def test_changed_checker_checkout_is_refused(
    checker_project: CommittedProject, committed_project: CommittedProject
) -> None:
    """
    An untracked policy file cannot enter an otherwise pinned checker.
    """
    (checker_project.root / "untrusted.txt").touch()
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "repository_knowledge",
            "publication",
            "--repository",
            str(committed_project.root),
            "--trusted-repository",
            str(checker_project.root),
            "--checker-revision",
            checker_project.commit_identifier,
            "fingerprint",
            "--revision",
            committed_project.commit_identifier,
        ],
        cwd=checker_project.root,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 1
