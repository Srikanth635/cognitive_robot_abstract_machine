"""
Ontology files shipped with the package.

The knowledge is described by two families of ontology modules, one Turtle file per
module: the action perspectives, which describe an action from several points of view
around an action core, and the predictive manipulation microworld, which describes it as
phases of predictive control. Each file imports the modules it depends on, and the first
module of each family imports the CRAM-AICOR and DOLCE+DnS Ultralite ontologies, whose
terms the modules specialise.
"""

from __future__ import annotations

from enum import StrEnum
from importlib import import_module
from pathlib import Path
from types import ModuleType

from typing_extensions import Optional, Tuple

ONTOLOGIES_DIRECTORY = Path(__file__).parent
"""
The directory holding the ontology files of every family.
"""

ONTOLOGIES_PACKAGE = __name__
"""
The Python package the generated vocabulary modules of every family are placed under.
"""

PREDICTIVE_MANIPULATION_ONTOLOGY_DIRECTORY = (
    ONTOLOGIES_DIRECTORY / "predictive_manipulation"
)
"""
The directory holding the Turtle file of every predictive manipulation module.
"""

PREDICTIVE_MANIPULATION_NAMESPACE_ROOT = (
    "http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/"
)
"""
The IRI prefix shared by the namespaces of every predictive manipulation module.
"""

ACTION_PERSPECTIVES_ONTOLOGY_DIRECTORY = ONTOLOGIES_DIRECTORY / "action_perspectives"
"""
The directory holding the Turtle file of every action perspective module.
"""

ACTION_PERSPECTIVES_NAMESPACE_ROOT = (
    "http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/"
)
"""
The IRI prefix shared by the namespaces of every action perspective module.
"""

PREDICTIVE_MANIPULATION_SHAPES_DIRECTORY = (
    PREDICTIVE_MANIPULATION_ONTOLOGY_DIRECTORY / "shapes"
)
"""
The directory holding the SHACL shapes of the modules that state structural rules.
"""

PREDICTIVE_MANIPULATION_SHAPES_NAMESPACE_ROOT = (
    PREDICTIVE_MANIPULATION_NAMESPACE_ROOT + "shapes/"
)
"""
The IRI prefix shared by the namespaces of every module's shapes.
"""

CRAM_AICOR_ONTOLOGY_IRI = "http://aicor.knowledge/l2-ontology.owl"
"""
The IRI of the CRAM-AICOR ontology, which the core module imports.
"""

CRAM_AICOR_NAMESPACE = CRAM_AICOR_ONTOLOGY_IRI + "#"
"""
The namespace of the CRAM-AICOR terms the microworld links its concepts to.
"""

DUL_ONTOLOGY_IRI = "http://www.ontologydesignpatterns.org/ont/dul/DUL.owl"
"""
The IRI of DOLCE+DnS Ultralite, which the core module imports alongside CRAM-AICOR.
"""

DUL_NAMESPACE = DUL_ONTOLOGY_IRI + "#"
"""
The namespace of the DOLCE+DnS Ultralite terms the microworld links its concepts to.
"""

HYPOTHESIS_CONCEPT_NAMESPACE = (
    "http://www.ease-crc.org/ont/llm_reasoner/hypothesis/concept#"
)
"""
The namespace of the concepts hypotheses introduce because no linked ontology declares
them, such as a category of grasp; kept apart so they never pass for linked terms.
"""


class LinkedOntologyVariable(StrEnum):
    """
    Environment variables naming the files of the ontologies the microworld links to,
    which are not part of the package.
    """

    CRAM_AICOR_ONTOLOGY = "CRAM_AICOR_ONTOLOGY"
    """
    Path of the CRAM-AICOR ontology in Turtle.
    """

    DUL_ONTOLOGY = "DUL_ONTOLOGY"
    """
    Path of the DOLCE+DnS Ultralite ontology in RDF/XML.
    """


DESCRIPTIONS_DIRECTORY_NAME = "descriptions"
"""
The directory, within the action perspectives' directory, holding the description of
each module.
"""

DESCRIPTION_SUFFIX = ".md"
"""
The suffix of a module's description file.
"""

SHAPES_DIRECTORY_NAME = "shapes"
"""
The directory, within a family's directory, holding the SHACL shapes of its modules.
"""


class OntologyModuleFile:
    """
    The Turtle file of an ontology module, and what is derived from it: its namespace,
    its shapes and its generated vocabulary module. A family of modules is a string
    enumeration of its files' names that says where the family lies.
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
        ``object_dynamics``; unique across the families.
        """
        return Path(self.value).stem.replace("-", "_")

    @property
    def vocabulary_package(self) -> str:
        """
        The Python package the generated module is placed in.
        """
        return ONTOLOGIES_PACKAGE + "." + self.directory.name

    @property
    def vocabulary_path(self) -> Path:
        """
        The location of the Python module generated from the module.
        """
        return self.directory / (self.vocabulary_module_name + ".py")

    @property
    def vocabulary_module(self) -> ModuleType:
        """
        The Python module generated from the module, imported.
        """
        return import_module(
            self.vocabulary_package + "." + self.vocabulary_module_name
        )


class ActionPerspectiveModule(OntologyModuleFile):
    """
    A module of the action perspectives, registered or proposed: where it lies, and the
    description of what it describes, which its vocabulary is proposed from and its
    knowledge generated from.
    """

    @property
    def directory(self) -> Path:
        """
        The directory holding the action perspective modules.
        """
        return ACTION_PERSPECTIVES_ONTOLOGY_DIRECTORY

    @property
    def namespace_root(self) -> str:
        """
        The IRI prefix of the action perspective modules.
        """
        return ACTION_PERSPECTIVES_NAMESPACE_ROOT

    @property
    def description(self) -> str:
        """
        What the module describes, as a model is told.
        """
        return (
            (
                self.directory
                / DESCRIPTIONS_DIRECTORY_NAME
                / (Path(self.value).stem + DESCRIPTION_SUFFIX)
            )
            .read_text()
            .strip()
        )


class ActionPerspectiveFile(ActionPerspectiveModule, StrEnum):
    """
    The Turtle file of each action perspective module, listed with every module after
    the modules it depends on.
    """

    ACTION_CORE = "action-core.ttl"
    """
    The hub every perspective attaches to: an action's frame, the slots its participants
    fill, what it is for, what it brings about and the kinds of choice it leaves open.
    """

    ACTION_FRAME = "action-frame.ttl"
    """
    The action as FrameNet and Minsky's frames describe it: what each slot's filler is
    expected to be, have or afford, the defaults, what must hold before the action, what
    it changes and why each parameter dimension matters.
    """

    @property
    def usable_modules(self) -> set[OntologyModuleFile]:
        """
        The module and the action core: a perspective describes an action in the
        core's terms and never in another perspective's.
        """
        return {ActionPerspectiveFile.ACTION_CORE, self}


class PredictiveManipulationOntologyFile(OntologyModuleFile, StrEnum):
    """
    The Turtle file of each predictive manipulation module, listed with every module
    after the modules it depends on.
    """

    CORE = "core.ttl"
    """
    Action-neutral phases, controllers, roles, events, conditions, quantities,
    prediction, control and source claims.
    """

    OBJECT_DYNAMICS = "object-dynamics.ttl"
    """
    Object and limb dynamics, efference copies, prediction sources and sensorimotor
    memory.
    """

    PREHENSION = "prehension.ttl"
    """
    Contact, lift, transport, replacement, release, grip-load control and corrections.
    """

    GAZE = "gaze.ttl"
    """
    Gaze fixations on contact and avoidance landmarks, and visual control points.
    """

    POURING = "pouring.ttl"
    """
    Source and target roles, flow, material regimes and transfer variables.
    """

    @property
    def directory(self) -> Path:
        """
        The directory holding the predictive manipulation modules.
        """
        return PREDICTIVE_MANIPULATION_ONTOLOGY_DIRECTORY

    @property
    def namespace_root(self) -> str:
        """
        The IRI prefix of the predictive manipulation modules.
        """
        return PREDICTIVE_MANIPULATION_NAMESPACE_ROOT

    @property
    def usable_modules(self) -> set[OntologyModuleFile]:
        """
        Every predictive manipulation module: they describe one microworld, whose
        general classes the later modules specialise.
        """
        return set(PredictiveManipulationOntologyFile)


KNOWLEDGE_ONTOLOGY_MODULES: Tuple[OntologyModuleFile, ...] = (
    *ActionPerspectiveFile,
    *PredictiveManipulationOntologyFile,
)
"""
The module of every family, each after the modules it depends on.
"""


def module_file_named(name: str) -> Optional[OntologyModuleFile]:
    """
    :param name: The file name of a module, such as ``action-core.ttl``.
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
