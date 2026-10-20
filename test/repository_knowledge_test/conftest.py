"""
Configuration for source-navigation tests without robotics initialization.
"""

from __future__ import annotations

import pytest
from pathlib import Path
import shutil

from repository_knowledge.adapters.git_source import GitSource
from repository_knowledge.adapters.python_source import PythonSource
from repository_knowledge.adapters.sqlite_index import SQLiteIndex
from repository_knowledge.model import IndexData
from repository_knowledge.operations.indexing import IndexBuilder
from repository_knowledge.adapters.okf import OkfCodec
from repository_knowledge.concepts import (
    Concept,
    ConceptEvidence,
    ConceptKind,
    ConceptProvenance,
)

from .support import CommittedProject

# %% ORM build contract


def pytest_addoption(parser: pytest.Parser) -> None:
    """
    Require an explicit disabled ORM build in the isolated test suite.
    """
    parser.addoption("--orm-build", choices=["never"], default="never")


# %% committed source fixture


@pytest.fixture
def committed_project(tmp_path: Path) -> CommittedProject:
    """
    Provide a committed local project without external services.
    """
    return CommittedProject.create(tmp_path / "project")


@pytest.fixture
def source_index(committed_project: CommittedProject) -> IndexData:
    """
    Provide the committed fixture's complete Python graph.
    """
    return IndexBuilder(GitSource(committed_project.root), PythonSource()).build(
        committed_project.commit_identifier
    )


@pytest.fixture
def indexed_storage(source_index: IndexData, tmp_path: Path) -> SQLiteIndex:
    """
    Provide the committed fixture graph after atomic publication.
    """
    storage = SQLiteIndex(tmp_path / "index.sqlite3")
    storage.replace(source_index)
    return storage


# %% curated knowledge fixtures


@pytest.fixture
def portable_concept() -> Concept:
    """
    Read a third-party concept without imposing CRAM's policy.
    """
    fixture = Path(__file__).parent / "dataset/knowledge/minimal.md"
    return OkfCodec().decode("minimal.md", fixture.read_text(encoding="utf-8"))


@pytest.fixture
def evidenced_concept(source_index: IndexData) -> Concept:
    """
    Bind a concept to the exact fixture class and original file bytes.
    """
    symbol = next(
        item
        for item in source_index.symbols
        if item.symbol_identifier == "first.base.Same"
    )
    source = next(
        item
        for item in source_index.sources
        if item.relative_path == symbol.location.relative_path
    )
    return Concept(
        "packages/base.md",
        ConceptKind.PACKAGE,
        "Base",
        "Base declaration.",
        "The base declaration is present.[^base]\n\n[^base]: Base source\n",
        provenance=ConceptProvenance(
            source_index.manifest.commit_identifier,
            [
                ConceptEvidence(
                    "base",
                    symbol.location,
                    source.content_digest,
                    symbol.symbol_identifier,
                )
            ],
        ),
    )


# %% task assistance fixtures


@pytest.fixture
def assistance_project(committed_project: CommittedProject) -> CommittedProject:
    """
    Add static test and instruction evidence to the isolated repository.
    """
    dataset = Path(__file__).parent / "dataset/assistance"
    shutil.copytree(dataset, committed_project.root, dirs_exist_ok=True)
    (committed_project.root / "root_fixtures.py").rename(
        committed_project.root / "conftest.py"
    )
    (committed_project.root / "tests/fixtures.py").rename(
        committed_project.root / "tests/conftest.py"
    )
    (committed_project.root / "tests/values.py").rename(
        committed_project.root / "tests/test_values.py"
    )
    committed_project.git("add", ".")
    committed_project.git("commit", "--quiet", "-m", "Task assistance fixture")
    committed_project.commit_identifier = committed_project.git("rev-parse", "HEAD")
    return committed_project


@pytest.fixture
def assistance_index(assistance_project: CommittedProject) -> IndexData:
    """
    Capture the complete task fixture without executing its modules.
    """
    return IndexBuilder(GitSource(assistance_project.root), PythonSource()).build(
        assistance_project.commit_identifier
    )
