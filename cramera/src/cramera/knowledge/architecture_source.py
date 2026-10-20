"""
Select architecture evidence without requiring the optional source index.
"""

from __future__ import annotations

import importlib
import importlib.util
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from cramera import paths
from cramera.knowledge.architecture_model import ArchitectureScan

# %% view source contract


class ArchitectureSetting(StrEnum):
    """
    Operator choices for optional shared architecture evidence.
    """

    DATABASE = "CRAMERA_KNOWLEDGE_DATABASE"
    REVISION = "CRAMERA_KNOWLEDGE_REVISION"
    CHECKOUT = "HEAD"
    PACKAGE = "repository_knowledge.model"
    ADAPTER = "cramera.knowledge.repository_architecture"


@dataclass
class ArchitectureSource(ABC):
    """
    Supply the architecture view's existing entity contract.
    """

    root: str
    """
    The repository supplying the architecture evidence.
    """

    @abstractmethod
    def load(self) -> ArchitectureScan:
        """
        Return architecture entities for the selected repository.
        """


@dataclass
class ArchitectureSelection:
    """
    Choose the optional integration only when explicitly configured.
    """

    @classmethod
    def of_configured_root(cls, root: str | None = None) -> ArchitectureSource | None:
        """
        Select shared evidence when its optional integration is available.
        """
        root = root if root is not None else str(paths.architecture_root())
        database = os.environ.get(ArchitectureSetting.DATABASE)
        if (
            not database
            or not Path(database).is_file()
            or importlib.util.find_spec(ArchitectureSetting.PACKAGE.split(".")[0])
            is None
            or importlib.util.find_spec(ArchitectureSetting.PACKAGE) is None
        ):
            return None
        adapter = importlib.import_module(ArchitectureSetting.ADAPTER)
        return adapter.IndexedArchitecture(
            root,
            Path(database),
            os.environ.get(ArchitectureSetting.REVISION, ArchitectureSetting.CHECKOUT),
        )
