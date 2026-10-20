"""
The optional viewer translates the shared graph without extracting twice.
"""

from __future__ import annotations

from pathlib import Path
import shutil
from unittest.mock import Mock

import pytest
from pytest import MonkeyPatch

from cramera.knowledge.architecture_entities import Package, PythonClass
from cramera.knowledge.architecture_scan import ArchitectureScanner, ArchitectureScan
from cramera.knowledge.architecture_source import (
    ArchitectureSelection,
    ArchitectureSetting,
)
from cramera.knowledge.repository_architecture import IndexedArchitecture
from repository_knowledge.adapters.sqlite_index import SQLiteIndex
from repository_knowledge.adapters.git_source import GitSource
from repository_knowledge.adapters.python_source import PythonSource
from repository_knowledge.operations.indexing import IndexBuilder

from .support import CommittedProject

# %% legacy view compatibility


@pytest.fixture(autouse=True)
def isolate_view_registration(monkeypatch: MonkeyPatch) -> None:
    """
    Keep architecture entity construction independent of runtime registration.
    """
    monkeypatch.setattr(Package, "_cache_instances_", False)
    monkeypatch.setattr(PythonClass, "_cache_instances_", False)


def test_shared_architecture_preserves_existing_view_entities(
    committed_project: CommittedProject, indexed_storage: SQLiteIndex
) -> None:
    """
    Nested classes, enums, module paths and package edges retain their meaning.
    """
    expected = ArchitectureScanner(str(committed_project.root)).scan()
    actual = IndexedArchitecture(
        str(committed_project.root), indexed_storage.database_path
    ).load()
    assert actual.packages == expected.packages
    assert sorted(actual.classes, key=lambda item: item.qualified_name) == sorted(
        expected.classes, key=lambda item: item.qualified_name
    )
    assert actual.dependency_edges == expected.dependency_edges


def test_shared_architecture_does_not_rescan_or_write_another_cache(
    committed_project: CommittedProject,
    indexed_storage: SQLiteIndex,
    monkeypatch: MonkeyPatch,
) -> None:
    """
    A current shared snapshot is the migrated viewer's only graph store.
    """
    scan = Mock(side_effect=RuntimeError("Duplicate extraction"))
    monkeypatch.setattr(ArchitectureScanner, "scan", scan)
    before = indexed_storage.database_path.read_bytes()
    result = IndexedArchitecture(
        str(committed_project.root), indexed_storage.database_path
    ).load()
    assert {item.name for item in result.packages} == {"root", "first", "second"}
    assert indexed_storage.database_path.read_bytes() == before
    scan.assert_not_called()


def test_shared_architecture_falls_back_to_current_source_after_revision_changes(
    committed_project: CommittedProject,
    indexed_storage: SQLiteIndex,
    monkeypatch: MonkeyPatch,
) -> None:
    """
    The optional viewer never presents an old shared graph as current source.
    """
    expected = ArchitectureScan([], [], [])
    scan = Mock(return_value=expected)
    monkeypatch.setattr(ArchitectureScanner, "scan", scan)
    committed_project.git(
        "commit", "--quiet", "--allow-empty", "-m", "Advance architecture fixture"
    )
    assert (
        IndexedArchitecture(
            str(committed_project.root), indexed_storage.database_path
        ).load()
        == expected
    )
    scan.assert_called_once_with()


def test_configured_viewer_selects_shared_architecture(
    committed_project: CommittedProject,
    indexed_storage: SQLiteIndex,
    monkeypatch: MonkeyPatch,
) -> None:
    """
    The existing viewer can opt into the shared cache without changing its entities.
    """
    monkeypatch.setenv("CRAMERA_ARCHITECTURE", str(committed_project.root))
    monkeypatch.setenv(ArchitectureSetting.DATABASE, str(indexed_storage.database_path))
    selected = ArchitectureSelection.of_configured_root()
    assert type(selected) is IndexedArchitecture
    assert selected.root == str(committed_project.root)


def test_existing_scanner_load_selects_the_shared_cache(
    committed_project: CommittedProject,
    indexed_storage: SQLiteIndex,
    monkeypatch: MonkeyPatch,
) -> None:
    """
    The existing architecture consumer reaches the shared graph through load.
    """
    monkeypatch.setenv(ArchitectureSetting.DATABASE, str(indexed_storage.database_path))
    legacy_cache = Mock(side_effect=RuntimeError("Duplicate cache"))
    monkeypatch.setattr(ArchitectureScanner, "_load_raw", legacy_cache)
    actual = ArchitectureScanner(str(committed_project.root)).load()
    expected = IndexedArchitecture(
        str(committed_project.root), indexed_storage.database_path
    ).load()
    assert actual == expected
    legacy_cache.assert_not_called()


def test_missing_shared_cache_retains_the_standalone_view(
    committed_project: CommittedProject, monkeypatch: MonkeyPatch
) -> None:
    """
    An absent optional cache cannot make the standalone viewer unavailable.
    """
    monkeypatch.setenv(
        ArchitectureSetting.DATABASE, str(committed_project.root / "missing.sqlite3")
    )
    assert ArchitectureSelection.of_configured_root(str(committed_project.root)) is None


def test_shared_architecture_preserves_star_import_package_edges(
    committed_project: CommittedProject, indexed_storage: SQLiteIndex
) -> None:
    """
    Unresolved imported names do not erase a known package dependency.
    """
    shutil.copyfile(
        Path(__file__).parent / "dataset/star_consumer.py",
        committed_project.root / "second/src/second/star_consumer.py",
    )
    committed_project.git("add", ".")
    committed_project.git("commit", "--quiet", "-m", "Star import view fixture")
    indexed_storage.replace(
        IndexBuilder(GitSource(committed_project.root), PythonSource()).build("HEAD")
    )
    expected = ArchitectureScanner(str(committed_project.root)).scan()
    actual = IndexedArchitecture(
        str(committed_project.root), indexed_storage.database_path
    ).load()
    assert actual.dependency_edges == expected.dependency_edges


def test_shared_architecture_uses_the_first_nonempty_readme_heading(
    committed_project: CommittedProject, indexed_storage: SQLiteIndex
) -> None:
    """
    An empty heading cannot hide the existing view's package description.
    """
    shutil.copyfile(
        Path(__file__).parent / "dataset/client-readme.md",
        committed_project.root / "first/README.md",
    )
    committed_project.git("add", ".")
    committed_project.git("commit", "--quiet", "-m", "README view fixture")
    indexed_storage.replace(
        IndexBuilder(GitSource(committed_project.root), PythonSource()).build("HEAD")
    )
    expected = ArchitectureScanner(str(committed_project.root)).scan()
    actual = IndexedArchitecture(
        str(committed_project.root), indexed_storage.database_path
    ).load()
    assert actual.packages == expected.packages
