"""
Ontology files shipped with the package.

The predictive manipulation microworld ontology is divided into one Turtle file per
module. Each file imports the module it depends on, and the core module imports the
CRAM-AICOR and DOLCE+DnS Ultralite ontologies, whose terms the microworld specialises.
"""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path

PREDICTIVE_MANIPULATION_ONTOLOGY_DIRECTORY = (
    Path(__file__).parent / "predictive_manipulation"
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


class PredictiveManipulationOntologyFile(StrEnum):
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
    def path(self) -> Path:
        """
        The location of the module's Turtle file.
        """
        return PREDICTIVE_MANIPULATION_ONTOLOGY_DIRECTORY / self.value

    @property
    def namespace(self) -> str:
        """
        The namespace the module declares its terms in.
        """
        return PREDICTIVE_MANIPULATION_NAMESPACE_ROOT + Path(self.value).stem + "#"

    @property
    def vocabulary_module_name(self) -> str:
        """
        The name of the Python module generated from the module, such as
        ``object_dynamics``.
        """
        return Path(self.value).stem.replace("-", "_")

    @property
    def vocabulary_path(self) -> Path:
        """
        The location of the Python module generated from the module.
        """
        return PREDICTIVE_MANIPULATION_ONTOLOGY_DIRECTORY / (
            self.vocabulary_module_name + ".py"
        )
