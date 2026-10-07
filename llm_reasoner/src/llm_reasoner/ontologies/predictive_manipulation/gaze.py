"""
Gaze in manipulation: fixations of contact and avoidance landmarks, held until the phase goal, and visual control points.

Generated from ``gaze.ttl`` by ``python -m llm_reasoner.ontology_translation``. Regenerate rather than edit.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

from krrood.ontomatic.property_descriptor.property_descriptor import PropertyDescriptor
from typing_extensions import ClassVar, List, Optional

from llm_reasoner.ontologies.predictive_manipulation import core
from llm_reasoner.ontologies.terms import (
    ClassTerm,
    ComplementOf,
    ConceptReference,
    DescriptionEntity,
    EnumeratedClass,
    IntersectionOf,
    NamedClass,
    PropertyRestriction,
    PropertyTerm,
    Quantifier,
    UnionOf,
)

# %% value partitions


class GazeMode(StrEnum):
    """
    Whether gaze predicts or follows the manipulation.
    """

    PROACTIVE = "http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/gaze#Proactive"
    """
    Gaze arrives at the landmark before the hand or the held object, as in skilled, familiar tasks.
    """

    REACTIVE = (
        "http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/gaze#Reactive"
    )
    """
    Gaze follows the hand or the moving object, as while a novel task is still being learned.
    """


class LandmarkPurpose(StrEnum):
    """
    Why a location is fixated during manipulation.
    """

    CONTACT_LANDMARK = "http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/gaze#ContactLandmark"
    """
    A location where contact is to be made: the goal of the current phase.
    """

    AVOIDANCE_LANDMARK = "http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/gaze#AvoidanceLandmark"
    """
    A location where contact must be avoided, such as an obstacle to circumnavigate.
    """


# %% classes


@dataclass(eq=False)
class ClosestApproach(DescriptionEntity):
    """
    The moment a moving effector or held object passes closest to an obstacle, monitored visually.
    """

    term: ClassVar[ClassTerm] = ClassTerm(
        iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/gaze#ClosestApproach",
        linked_terms=(
            "http://aicor.knowledge/l2-ontology.owl#EventType",
            "http://www.ontologydesignpatterns.org/ont/dul/DUL.owl#EventType",
        ),
        evidence_status="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#EmpiricalFinding",
        claims=(
            "http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#JF09-10",
            "http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#JF09-11",
        ),
        restrictions=(
            PropertyRestriction(
                property_iri="http://aicor.knowledge/l2-ontology.owl#betweenRoleFillers",
                quantifier=Quantifier.SOME,
                filler=NamedClass(
                    iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/gaze#Obstacle"
                ),
            ),
        ),
        definition=None,
    )
    """
    The ontology class this class stands for.
    """

    between_role_fillers: List[core.ManipulationRole] = field(
        default_factory=list,
        metadata=PropertyTerm(
            iri="http://aicor.knowledge/l2-ontology.owl#betweenRoleFillers",
            functional=False,
            range="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#ManipulationRole",
        ).as_dict(),
    )
    """
    The linked ontology property http://aicor.knowledge/l2-ontology.owl#betweenRoleFillers.
    """


@dataclass(eq=False)
class GazeFixation(DescriptionEntity):
    """
    A fixation of a contact or avoidance landmark during a phase, held until a given event.
    """

    term: ClassVar[ClassTerm] = ClassTerm(
        iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/gaze#GazeFixation",
        linked_terms=(
            "http://aicor.knowledge/l2-ontology.owl#dul_Concept",
            "http://www.ontologydesignpatterns.org/ont/dul/DUL.owl#Task",
        ),
        evidence_status="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#EmpiricalFinding",
        claims=(
            "http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#FJ-014",
            "http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#JF09-10",
        ),
        restrictions=(
            PropertyRestriction(
                property_iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/gaze#fixatesRole",
                quantifier=Quantifier.SOME,
                filler=NamedClass(
                    iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#ManipulationRole"
                ),
            ),
            PropertyRestriction(
                property_iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/gaze#hasGazeMode",
                quantifier=Quantifier.SOME,
                filler=NamedClass(
                    iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/gaze#GazeMode"
                ),
            ),
            PropertyRestriction(
                property_iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/gaze#hasLandmarkPurpose",
                quantifier=Quantifier.SOME,
                filler=NamedClass(
                    iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/gaze#LandmarkPurpose"
                ),
            ),
            PropertyRestriction(
                property_iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/gaze#maintainedUntil",
                quantifier=Quantifier.SOME,
                filler=NamedClass(
                    iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#PhaseBoundaryEventType"
                ),
            ),
            PropertyRestriction(
                property_iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/gaze#servesPhase",
                quantifier=Quantifier.SOME,
                filler=NamedClass(
                    iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#ActionPhase"
                ),
            ),
        ),
        definition=None,
    )
    """
    The ontology class this class stands for.
    """

    fixates_role: Optional[core.ManipulationRole] = field(
        default=None,
        metadata=PropertyTerm(
            iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/gaze#fixatesRole",
            functional=True,
            range="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#ManipulationRole",
        ).as_dict(),
    )
    """
    The participant whose location is the landmark.
    """

    has_gaze_mode: Optional[GazeMode] = field(
        default=None,
        metadata=PropertyTerm(
            iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/gaze#hasGazeMode",
            functional=True,
        ).as_dict(),
    )
    """
    Whether gaze leads the hand to the landmark or follows the moving hand or object.
    """

    has_landmark_purpose: Optional[LandmarkPurpose] = field(
        default=None,
        metadata=PropertyTerm(
            iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/gaze#hasLandmarkPurpose",
            functional=True,
        ).as_dict(),
    )
    """
    Whether contact is to be made or avoided at the landmark.
    """

    maintained_until: List[core.PhaseBoundaryEventType] = field(
        default_factory=list,
        metadata=PropertyTerm(
            iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/gaze#maintainedUntil",
            functional=False,
            range="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#PhaseBoundaryEventType",
        ).as_dict(),
    )
    """
    The event at whose predicted time gaze leaves the landmark.
    """

    serves_phase: Optional[core.ActionPhase] = field(
        default=None,
        metadata=PropertyTerm(
            iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/gaze#servesPhase",
            functional=True,
            range="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#ActionPhase",
        ).as_dict(),
    )
    """
    The phase whose goal the fixation supports.
    """


@dataclass(eq=False)
class Obstacle(core.ManipulationRole):
    """
    An object or region that the effector or the held object must pass without contact.
    """

    term: ClassVar[ClassTerm] = ClassTerm(
        iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/gaze#Obstacle",
        linked_terms=(),
        evidence_status="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#EmpiricalFinding",
        claims=(
            "http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#JF09-10",
        ),
        restrictions=(),
        definition=None,
    )
    """
    The ontology class this class stands for.
    """


@dataclass(eq=False)
class VisualControlPoint(core.SensorimotorControlPoint):
    """
    A control point at which a contact event is also predicted and monitored in the visual modality. A pose confirmed by vision alone, such as a lip aligned over a receptacle, is checked by a comparison without a control point.
    """

    term: ClassVar[ClassTerm] = ClassTerm(
        iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/gaze#VisualControlPoint",
        linked_terms=(),
        evidence_status="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#EmpiricalFinding",
        claims=(
            "http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#FJ-015",
            "http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#JF09-10",
        ),
        restrictions=(),
        definition=IntersectionOf(
            operands=(
                NamedClass(
                    iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#SensorimotorControlPoint"
                ),
                PropertyRestriction(
                    property_iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#hasPrediction",
                    quantifier=Quantifier.SOME,
                    filler=PropertyRestriction(
                        property_iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#hasModality",
                        quantifier=Quantifier.VALUE,
                        value="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#Visual",
                    ),
                ),
            )
        ),
    )
    """
    The ontology class this class stands for.
    """


# %% property descriptors


@dataclass
class DefinesGazeFixation(core.DefinesComponent):
    """
    A gaze fixation that supports one of the structure's phases.
    """


@dataclass
class FixatesRole(PropertyDescriptor):
    """
    The participant whose location is the landmark.
    """


@dataclass
class MaintainedUntil(PropertyDescriptor):
    """
    The event at whose predicted time gaze leaves the landmark.
    """


@dataclass
class ServesPhase(PropertyDescriptor):
    """
    The phase whose goal the fixation supports.
    """


# %% descriptor attachments

core.ActionPhaseStructure.defines_gaze_fixation = DefinesGazeFixation(
    core.ActionPhaseStructure, "defines_gaze_fixation"
)
GazeFixation.fixates_role = FixatesRole(GazeFixation, "fixates_role")
GazeFixation.maintained_until = MaintainedUntil(GazeFixation, "maintained_until")
GazeFixation.serves_phase = ServesPhase(GazeFixation, "serves_phase")
ClosestApproach.between_role_fillers = core.BetweenRoleFillers(
    ClosestApproach, "between_role_fillers"
)
