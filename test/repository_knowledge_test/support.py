"""
Local source datasets and Git operations shared by navigation tests.
"""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

# %% fixture identities


class FixturePath(StrEnum):
    """
    Paths with specific roles in the committed source dataset.
    """

    PROJECT = "dataset/project"
    API = "first/src/first/api.py"
    BASE = "first/src/first/base.py"
    BROKEN = "first/src/first/broken.py"
    GENERATED = "first/src/first/ormatic_interface.py"
    REPLACEMENT = "dataset/replacement_source.py"
    INVALID = "dataset/broken_source.py"
    RST = "docs/overview.rst"
    ADDED = "first/src/first/added.py"
    RENAMED = "first/src/first/renamed.py"
    FIRST_PROJECT = "first/pyproject.toml"
    PUBLIC_EXPORT = "first/src/first/__init__.py"
    CONSUMER = "first/src/first/consumer.py"
    OVERLAY_DATASET = "dataset/overlay"
    SOURCE_ROOTS = "dataset/overlay/source_roots.toml"
    IGNORE_POLICY = "dataset/overlay/ignored.gitignore"


class FixtureSymbol(StrEnum):
    """
    Declaration identities exercised by source-navigation tests.
    """

    BASE = "first.base.Same"
    CHILD = "first.api.Child"
    NESTED = "first.api.Child.Nested"
    HELPER = "first.base.helper"
    OTHER_BASE = "second.base.Same"
    ENUM_MEMBER = "first.api.Choice.READY"
    REPLACEMENT = "first.api.Uncommitted"
    CALLER = "first.api.use_alias"
    CONSUMER = "first.consumer.Consumer"


class FixtureIdentity(StrEnum):
    """
    The approved synthetic identity confined to disposable test histories.
    """

    AUTHOR = "Source Fixture"
    EMAIL = "source-fixture@example.invalid"


@dataclass
class CommittedProject:
    """
    An isolated local repository containing the source-navigation dataset.
    """

    root: Path
    """
    The repository containing the committed fixture files.
    """

    commit_identifier: str
    """
    The dataset's committed source revision.
    """

    def git(self, *arguments: str) -> str:
        """
        Run a Git operation against the isolated fixture repository.
        """
        result = subprocess.run(
            ["git", "-C", str(self.root), *arguments],
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()

    @classmethod
    def create(cls, destination: Path) -> CommittedProject:
        """
        Commit the repository-local dataset using its approved fixture identity.
        """
        dataset = Path(__file__).parent / FixturePath.PROJECT
        shutil.copytree(dataset, destination)
        repository = cls(destination, "")
        repository.git("init", "--quiet")
        repository.git("config", "user.name", FixtureIdentity.AUTHOR)
        repository.git("config", "user.email", FixtureIdentity.EMAIL)
        repository.git("config", "core.autocrlf", "false")
        repository.git("add", ".")
        repository.git("commit", "--quiet", "-m", "Source navigation fixture")
        repository.commit_identifier = repository.git("rev-parse", "HEAD")
        return repository
