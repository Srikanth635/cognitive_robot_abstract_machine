"""
The registry of the ontology module files shipped with the package, and where the files
and the generated vocabulary of each module lie.

The knowledge is described by families of ontology modules, one Turtle file per module,
each file importing the modules it depends on. A family is registered in
:data:`KNOWLEDGE_ONTOLOGY_MODULES`.
"""

from __future__ import annotations

from importlib import import_module
from pathlib import Path
from types import ModuleType

from typing_extensions import Optional

ONTOLOGIES_DIRECTORY = Path(__file__).parent.parent / "ontologies"
"""
The directory holding the ontology files of every family, and nothing else.
"""

GENERATED_PACKAGE_NAME = "generated"
"""
The name of the package, within this one, holding the generated vocabulary modules.
"""

GENERATED_DIRECTORY = Path(__file__).parent / GENERATED_PACKAGE_NAME
"""
The directory holding the generated vocabulary modules of every family.
"""

GENERATED_PACKAGE = f"{__package__}.{GENERATED_PACKAGE_NAME}"
"""
The Python package the generated vocabulary modules of every family are placed under.
"""

HYPOTHESIS_CONCEPT_NAMESPACE = (
    "http://www.ease-crc.org/ont/llm_reasoner/hypothesis/concept#"
)
"""
The namespace of the concepts hypotheses introduce because no linked ontology declares
them, such as a category of grasp; kept apart so they never pass for linked terms.
"""

SHAPES_DIRECTORY_NAME = "shapes"
"""
The directory, within a family's directory, holding the SHACL shapes of its modules.
"""


class OntologyModuleFile:
    """
    The Turtle file of an ontology module, and what is derived from it: its namespace,
    its shapes and its generated vocabulary module.

    A family of modules is a string enumeration of its files' names that says where the
    family lies.
    """

    @property
    def directory(self) -> Path:
        """
        The directory holding the family's Turtle files.
        """
        raise NotImplementedError

    @property
    def namespace_root(self) -> str:
        """
        The IRI prefix shared by the namespaces of the family's modules.
        """
        raise NotImplementedError

    @property
    def usable_modules(self) -> set[OntologyModuleFile]:
        """
        The modules whose classes a hypothesis rooted in a class of this module may use.
        """
        raise NotImplementedError

    @property
    def holds_generated_knowledge(self) -> bool:
        """
        Whether a language model may generate knowledge in the module's terms.
        """
        return True

    @property
    def path(self) -> Path:
        """
        The location of the module's Turtle file.
        """
        return self.directory / self.value

    @property
    def namespace(self) -> str:
        """
        The namespace the module declares its terms in.
        """
        return self.namespace_root + Path(self.value).stem + "#"

    @property
    def shapes_path(self) -> Path:
        """
        The location of the SHACL shapes stating the module's structural rules; a module
        without such rules has no file there.
        """
        return self.directory / SHAPES_DIRECTORY_NAME / self.value

    @property
    def shapes_namespace(self) -> str:
        """
        The namespace the module's shapes are named in.
        """
        return (
            self.namespace_root
            + SHAPES_DIRECTORY_NAME
            + "/"
            + Path(self.value).stem
            + "#"
        )

    @property
    def vocabulary_module_name(self) -> str:
        """
        The name of the Python module generated from the module, such as
        ``material_transfer``; unique across the families.
        """
        return Path(self.value).stem.replace("-", "_")

    @property
    def vocabulary_package(self) -> str:
        """
        The Python package the generated module is placed in, one per family.
        """
        return GENERATED_PACKAGE + "." + self.directory.name

    @property
    def vocabulary_path(self) -> Path:
        """
        The location of the Python module generated from the module.
        """
        return (
            GENERATED_DIRECTORY
            / self.directory.name
            / (self.vocabulary_module_name + ".py")
        )

    @property
    def vocabulary_module(self) -> ModuleType:
        """
        The Python module generated from the module, imported.
        """
        return import_module(
            self.vocabulary_package + "." + self.vocabulary_module_name
        )


KNOWLEDGE_ONTOLOGY_MODULES: tuple[OntologyModuleFile, ...] = ()
"""
The module of every family, each after the modules it depends on; empty until the
families of the new ontologies are registered.
"""


def module_file_named(name: str) -> Optional[OntologyModuleFile]:
    """
    :param name: The file name of a module.
    :return: The registered module of that file name, ``None`` when none is registered.
    """
    return next(
        (
            module_file
            for module_file in KNOWLEDGE_ONTOLOGY_MODULES
            if module_file.value == name
        ),
        None,
    )
