"""
The architecture view contract shared by standalone and indexed readers.
"""

from __future__ import annotations

from dataclasses import dataclass

from cramera.knowledge.architecture_entities import Package, PythonClass

# %% architecture graph


@dataclass
class PackageDependency:
    """
    One import edge between workspace packages.
    """

    source: str
    """The package doing the importing."""
    target: str
    """
    The package being imported.
    """


@dataclass
class ArchitectureScan:
    """
    The architecture entities consumed by the existing viewer.
    """

    packages: list[Package]
    """The workspace package groups."""
    classes: list[PythonClass]
    """
    The handwritten class declarations.
    """

    dependency_edges: list[PackageDependency]
    """The package import relationships."""
