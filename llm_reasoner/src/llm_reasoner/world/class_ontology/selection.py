"""
Which classes of a code base its ontology describes, as a caller selects them.
"""

from __future__ import annotations

import importlib
import json
import pkgutil
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from typing_extensions import Any, Dict, List, Tuple

PRIVATE_PREFIX = "_"
"""
The prefix of a class that is not meant to be used outside its module.
"""

MODULE_SEPARATOR = "."
"""
What separates the parts of a module's name, and a module from a class in an identifier.
"""


class SelectionKey(StrEnum):
    """
    The keys of a selection file.
    """

    NAMESPACE = "namespace"
    PACKAGES = "packages"
    ROOTS = "roots"
    EXPANDED_PACKAGES = "expanded_packages"
    IGNORED_BASES = "ignored_bases"


@dataclass
class UnknownClass(LookupError):
    """
    A selection names a class its module does not have.
    """

    identifier: str
    """
    The module and name of the class, as the selection gives them.
    """

    def __post_init__(self):
        super().__init__(f"no class {self.identifier}")


@dataclass(frozen=True)
class SkippedModule:
    """
    A module of a selected package that could not be imported, so its classes are not
    described.
    """

    module: str
    """
    The module's name.
    """

    error_type: str
    """
    The kind of error importing it raised.
    """

    reason: str
    """
    What the error said.
    """


@dataclass(frozen=True)
class ClassSelection:
    """
    The classes of a code base its ontology describes.
    """

    namespace: str
    """
    The namespace the ontology is written in.
    """

    packages: Tuple[str, ...]
    """
    The packages imported, each with every module below it; only their classes are
    described as subclasses of the roots.
    """

    roots: Tuple[str, ...]
    """
    The classes, by module and name, whose subclasses in the packages are described,
    besides themselves.
    """

    expanded_packages: Tuple[str, ...]
    """
    The packages whose classes are described with their fields when a described field
    refers to them; a class from elsewhere that a field refers to is described without
    its fields.
    """

    ignored_bases: Tuple[str, ...]
    """
    The classes, by module and name, whose fields are bookkeeping and left out of every
    class that inherits them.
    """

    @classmethod
    def from_json(cls, data: Dict[str, Any]) -> ClassSelection:
        """
        :param data: A selection as its file holds it.
        :return: The selection.
        """
        return cls(
            namespace=data[SelectionKey.NAMESPACE],
            packages=tuple(data[SelectionKey.PACKAGES]),
            roots=tuple(data[SelectionKey.ROOTS]),
            expanded_packages=tuple(data[SelectionKey.EXPANDED_PACKAGES]),
            ignored_bases=tuple(data[SelectionKey.IGNORED_BASES]),
        )

    @classmethod
    def from_file(cls, path: Path) -> ClassSelection:
        """
        :return: The selection a file holds.
        """
        return cls.from_json(json.loads(path.read_text()))

    def import_packages(self) -> List[SkippedModule]:
        """
        Import every module of the packages.

        A module that raises while it is imported is skipped and reported: it runs a
        package's own code, which may need what is not installed.

        :return: The modules skipped.
        """
        skipped: List[SkippedModule] = []
        for package in self.packages:
            imported = importlib.import_module(package)
            for module in pkgutil.walk_packages(
                imported.__path__, package + MODULE_SEPARATOR
            ):
                try:
                    importlib.import_module(module.name)
                except Exception as error:
                    skipped.append(
                        SkippedModule(
                            module=module.name,
                            error_type=type(error).__name__,
                            reason=str(error),
                        )
                    )
        return skipped

    @staticmethod
    def class_named(identifier: str) -> type:
        """
        :param identifier: The module and name of a class defined at the top of its
            module.
        :return: The class.
        :raises UnknownClass: When the module has no class of that name.
        """
        module, name = identifier.rsplit(MODULE_SEPARATOR, 1)
        found = vars(importlib.import_module(module)).get(name)
        if not isinstance(found, type):
            raise UnknownClass(identifier)
        return found

    def selects(self, described: type) -> bool:
        """
        :return: Whether a class belongs to the packages and is meant to be used.
        """
        return self.in_packages(described, self.packages) and not (
            described.__name__.startswith(PRIVATE_PREFIX)
        )

    def expands(self, described: type) -> bool:
        """
        :return: Whether a class a field refers to is described with its fields.
        """
        return self.in_packages(described, self.expanded_packages) and not (
            described.__name__.startswith(PRIVATE_PREFIX)
        )

    @staticmethod
    def in_packages(described: type, packages: Tuple[str, ...]) -> bool:
        """
        :return: Whether a class is defined in one of the packages or below it.
        """
        return any(
            described.__module__ == package
            or described.__module__.startswith(package + MODULE_SEPARATOR)
            for package in packages
        )
