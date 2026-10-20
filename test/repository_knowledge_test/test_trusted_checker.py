"""
The checker trust root includes executable code and workspace dependencies.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path


from .support import CommittedProject

# %% isolated checker checkout


def test_checker_refuses_krrood_from_another_checkout(
    committed_project: CommittedProject, tmp_path: Path
) -> None:
    """
    An immutable checker must not load mutable workspace code from elsewhere.
    """
    repository = Path(__file__).parents[2]
    trusted = CommittedProject.create(tmp_path / "trusted")
    shutil.copytree(
        repository / "repository_knowledge/src",
        trusted.root / "repository_knowledge/src",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    policy = trusted.root / "repository_knowledge/policy/publication.toml"
    policy.parent.mkdir()
    shutil.copyfile(
        Path(__file__).parent / "dataset/knowledge/publication.toml", policy
    )
    trusted.git("add", ".")
    trusted.git("commit", "--quiet", "-m", "Trusted checker fixture")
    revision = trusted.git("rev-parse", "HEAD")
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(trusted.root / "repository_knowledge/src")
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "repository_knowledge",
            "publication",
            "--repository",
            str(committed_project.root),
            "--trusted-repository",
            str(trusted.root),
            "--checker-revision",
            revision,
            "fingerprint",
            "--revision",
            committed_project.commit_identifier,
        ],
        cwd=trusted.root,
        env=environment,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 1
