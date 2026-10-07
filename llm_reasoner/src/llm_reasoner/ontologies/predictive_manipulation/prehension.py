"""
Prehensile manipulation: reach, load, lift, hold, transport, replace, unload and release, with their contact events, grip and load forces and corrective programmes.

Generated from ``prehension.ttl`` by ``python -m llm_reasoner.ontology_translation``. Regenerate rather than edit.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

from krrood.ontomatic.property_descriptor.property_descriptor import PropertyDescriptor
from typing_extensions import ClassVar, List, Optional

from llm_reasoner.ontologies.predictive_manipulation import core
from llm_reasoner.ontologies.predictive_manipulation import object_dynamics
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


class LoadOrigin(StrEnum):
    """
    Whether a load on the grasp can be predicted from the actor's own motor commands.
    """

    SELF_GENERATED_LOAD = "http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/prehension#SelfGeneratedLoad"
    """
    A load from the actor's own movement: grip changes in parallel with it.
    """

    EXTERNALLY_GENERATED_LOAD = "http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/prehension#ExternallyGeneratedLoad"
    """
    A load imposed from outside: grip follows it with a delay and a raised baseline.
    """


# %% classes


@dataclass(eq=False)
class AnticipatoryGripAdjustment(DescriptionEntity):
    """
    A feedforward grip change scheduled from a predicted event.
    """

    term: ClassVar[ClassTerm] = ClassTerm(
        iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/prehension#AnticipatoryGripAdjustment",
        linked_terms=(
            "http://aicor.knowledge/l2-ontology.owl#dul_Concept",
            "http://www.ontologydesignpatterns.org/ont/dul/DUL.owl#Task",
        ),
        evidence_status="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#EmpiricalFinding",
        claims=(
            "http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#FJ-009",
            "http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#FJ-025",
        ),
        restrictions=(
            PropertyRestriction(
                property_iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/prehension#adjustmentActiveIn",
                quantifier=Quantifier.SOME,
                filler=NamedClass(
                    iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#ActionPhase"
                ),
            ),
            PropertyRestriction(
                property_iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/prehension#adjustsGripForce",
                quantifier=Quantifier.SOME,
                filler=NamedClass(
                    iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/prehension#GripForceSetpoint"
                ),
            ),
            PropertyRestriction(
                property_iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/prehension#hasAdjustmentDirection",
                quantifier=Quantifier.SOME,
                filler=NamedClass(
                    iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#CorrectionDirection"
                ),
            ),
            PropertyRestriction(
                property_iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/prehension#triggeredByPrediction",
                quantifier=Quantifier.SOME,
                filler=NamedClass(
                    iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#SensoryEventPrediction"
                ),
            ),
        ),
        definition=None,
    )
    """
    The ontology class this class stands for.
    """

    adjustment_active_in: List[core.ActionPhase] = field(
        default_factory=list,
        metadata=PropertyTerm(
            iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/prehension#adjustmentActiveIn",
            functional=False,
            range="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#ActionPhase",
        ).as_dict(),
    )
    """
    A phase during which the anticipatory change is active.
    """

    adjusts_grip_force: Optional[GripForceSetpoint] = field(
        default=None,
        metadata=PropertyTerm(
            iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/prehension#adjustsGripForce",
            functional=True,
            range="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/prehension#GripForceSetpoint",
        ).as_dict(),
    )
    """
    The grip-force target changed before the predicted event.
    """

    has_adjustment_direction: Optional[core.CorrectionDirection] = field(
        default=None,
        metadata=PropertyTerm(
            iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/prehension#hasAdjustmentDirection",
            functional=True,
        ).as_dict(),
    )
    """
    The direction in which grip force changes.
    """

    triggered_by_prediction: Optional[core.SensoryEventPrediction] = field(
        default=None,
        metadata=PropertyTerm(
            iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/prehension#triggeredByPrediction",
            functional=True,
            range="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#SensoryEventPrediction",
        ).as_dict(),
    )
    """
    The predicted event consequence that schedules the adjustment.
    """


@dataclass(eq=False)
class ContactEstablishmentPhase(core.ActionPhase):
    """
    Closing the effector on the manipulandum, for grippers closed separately from the reach.
    """

    term: ClassVar[ClassTerm] = ClassTerm(
        iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/prehension#ContactEstablishmentPhase",
        linked_terms=(),
        evidence_status="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#EngineeringExtension",
        claims=(),
        restrictions=(
            PropertyRestriction(
                property_iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#realizesActionCategory",
                quantifier=Quantifier.VALUE,
                value="http://aicor.knowledge/l2-ontology.owl#Grasping",
            ),
        ),
        definition=None,
    )
    """
    The ontology class this class stands for.
    """


@dataclass(eq=False)
class ContactReleasePhase(core.ActionPhase):
    """
    Opening of the grasp once the forces are unloaded, ending at grasp release.
    """

    term: ClassVar[ClassTerm] = ClassTerm(
        iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/prehension#ContactReleasePhase",
        linked_terms=(),
        evidence_status="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#EmpiricalFinding",
        claims=(),
        restrictions=(
            PropertyRestriction(
                property_iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#terminatedBy",
                quantifier=Quantifier.SOME,
                filler=NamedClass(
                    iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/prehension#GraspRelease"
                ),
            ),
        ),
        definition=None,
    )
    """
    The ontology class this class stands for.
    """


@dataclass(eq=False)
class CorrectiveGripResponse(core.CorrectiveAction):
    """
    A correction that changes the grip force target, such as raising grip after a slip while holding.
    """

    term: ClassVar[ClassTerm] = ClassTerm(
        iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/prehension#CorrectiveGripResponse",
        linked_terms=(),
        evidence_status="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#EmpiricalFinding",
        claims=(
            "http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#FJ-010",
            "http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#FJ-012",
            "http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#JF09-05",
        ),
        restrictions=(
            PropertyRestriction(
                property_iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#targetsSetpoint",
                quantifier=Quantifier.ONLY,
                filler=NamedClass(
                    iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/prehension#GripForceSetpoint"
                ),
            ),
        ),
        definition=None,
    )
    """
    The ontology class this class stands for.
    """


@dataclass(eq=False)
class CrushThreshold(core.ForceParameter, core.MechanicalThreshold):
    """
    The grip force beyond which the manipulandum is damaged, relevant for fragile objects.
    """

    term: ClassVar[ClassTerm] = ClassTerm(
        iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/prehension#CrushThreshold",
        linked_terms=(),
        evidence_status="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#SupportedInterpretation",
        claims=(
            "http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#JF09-07",
        ),
        restrictions=(),
        definition=None,
    )
    """
    The ontology class this class stands for.
    """


@dataclass(eq=False)
class EarlyLiftoffCorrection(core.CorrectiveAction):
    """
    For an object lighter than predicted: the load controller is aborted and the object brought back to the intended position.
    """

    term: ClassVar[ClassTerm] = ClassTerm(
        iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/prehension#EarlyLiftoffCorrection",
        linked_terms=(),
        evidence_status="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#EmpiricalFinding",
        claims=(
            "http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#FJ-010",
            "http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#JF09-04",
        ),
        restrictions=(
            PropertyRestriction(
                property_iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#hasRecoveryMode",
                quantifier=Quantifier.VALUE,
                value="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#AbortAndCorrect",
            ),
            PropertyRestriction(
                property_iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#triggeredByError",
                quantifier=Quantifier.SOME,
                filler=PropertyRestriction(
                    property_iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#hasErrorKind",
                    quantifier=Quantifier.VALUE,
                    value="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#EarlyEvent",
                ),
            ),
        ),
        definition=None,
    )
    """
    The ontology class this class stands for.
    """


@dataclass(eq=False)
class EffectorObjectContact(core.ContactCondition):
    """
    The effector is mechanically in contact with the manipulandum.
    """

    term: ClassVar[ClassTerm] = ClassTerm(
        iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/prehension#EffectorObjectContact",
        linked_terms=(),
        evidence_status="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#EmpiricalFinding",
        claims=(),
        restrictions=(),
        definition=None,
    )
    """
    The ontology class this class stands for.
    """


@dataclass(eq=False)
class EffectorPositioned(core.PhaseConditionType):
    """
    The effector is positioned for establishing contact at the grasp site.
    """

    term: ClassVar[ClassTerm] = ClassTerm(
        iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/prehension#EffectorPositioned",
        linked_terms=(),
        evidence_status="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#EngineeringExtension",
        claims=(),
        restrictions=(),
        definition=None,
    )
    """
    The ontology class this class stands for.
    """


@dataclass(eq=False)
class ExpectedCenterOfMassPosition(core.LengthParameter, core.ObjectStateEstimate):
    """
    The anticipated centre of mass position in the object frame.
    """

    term: ClassVar[ClassTerm] = ClassTerm(
        iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/prehension#ExpectedCenterOfMassPosition",
        linked_terms=(),
        evidence_status="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#EmpiricalFinding",
        claims=(
            "http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#FJ-009",
        ),
        restrictions=(),
        definition=None,
    )
    """
    The ontology class this class stands for.
    """


@dataclass(eq=False)
class ExpectedFrictionCoefficient(
    core.ContactPropertyEstimate, core.DimensionlessParameter
):
    """
    The anticipated friction of the grasp interface, predicted from haptic experience, not from vision.
    """

    term: ClassVar[ClassTerm] = ClassTerm(
        iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/prehension#ExpectedFrictionCoefficient",
        linked_terms=(),
        evidence_status="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#EmpiricalFinding",
        claims=(
            "http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#FJ-012",
            "http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#JF09-08",
        ),
        restrictions=(
            PropertyRestriction(
                property_iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#predictedFrom",
                quantifier=Quantifier.ONLY,
                filler=EnumeratedClass(
                    members=(
                        "http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#RecentHapticExperience",
                        "http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#TactileContactResponse",
                    )
                ),
            ),
        ),
        definition=None,
    )
    """
    The ontology class this class stands for.
    """


@dataclass(eq=False)
class ExpectedMass(core.MassParameter, core.ObjectPropertyEstimate):
    """
    The anticipated mass of the manipulandum, from which its weight force is predicted.
    """

    term: ClassVar[ClassTerm] = ClassTerm(
        iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/prehension#ExpectedMass",
        linked_terms=(),
        evidence_status="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#EmpiricalFinding",
        claims=(
            "http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#FJ-009",
            "http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#FJ-010",
            "http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#JF09-08",
        ),
        restrictions=(),
        definition=None,
    )
    """
    The ontology class this class stands for.
    """


@dataclass(eq=False)
class ExpectedShape(core.CategoricalParameter, core.ObjectPropertyEstimate):
    """
    The anticipated geometry of the grasp surfaces.
    """

    term: ClassVar[ClassTerm] = ClassTerm(
        iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/prehension#ExpectedShape",
        linked_terms=(),
        evidence_status="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#EmpiricalFinding",
        claims=(
            "http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#FJ-009",
            "http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#JF09-08",
        ),
        restrictions=(),
        definition=None,
    )
    """
    The ontology class this class stands for.
    """


@dataclass(eq=False)
class FinalSupport(core.ManipulationRole):
    """
    The surface bearing the manipulandum after touchdown.
    """

    term: ClassVar[ClassTerm] = ClassTerm(
        iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/prehension#FinalSupport",
        linked_terms=(),
        evidence_status="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#EmpiricalFinding",
        claims=(),
        restrictions=(),
        definition=None,
    )
    """
    The ontology class this class stands for.
    """


@dataclass(eq=False)
class ForceLoadingPhase(core.ActionPhase):
    """
    Grip and load forces increase in parallel, at a rate scaled to the predicted weight, until lift-off.
    """

    term: ClassVar[ClassTerm] = ClassTerm(
        iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/prehension#ForceLoadingPhase",
        linked_terms=(),
        evidence_status="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#EmpiricalFinding",
        claims=(
            "http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#FJ-009",
            "http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#FJ-025",
        ),
        restrictions=(
            PropertyRestriction(
                property_iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#hasCouplingRegime",
                quantifier=Quantifier.VALUE,
                value="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#Transitional",
            ),
            PropertyRestriction(
                property_iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#realizesActionCategory",
                quantifier=Quantifier.VALUE,
                value="http://aicor.knowledge/l2-ontology.owl#Grasping",
            ),
        ),
        definition=None,
    )
    """
    The ontology class this class stands for.
    """


@dataclass(eq=False)
class ForcesUnloaded(core.MechanicalEventType, core.SetpointAttainmentEventType):
    """
    Grip and load forces reach their low targets and the weight passes from the grasp to the support.
    """

    term: ClassVar[ClassTerm] = ClassTerm(
        iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/prehension#ForcesUnloaded",
        linked_terms=(),
        evidence_status="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#EmpiricalFinding",
        claims=(),
        restrictions=(
            PropertyRestriction(
                property_iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#endsForce",
                quantifier=Quantifier.SOME,
                filler=PropertyRestriction(
                    property_iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#hasForceKind",
                    quantifier=Quantifier.VALUE,
                    value="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#LoadForce",
                ),
            ),
            PropertyRestriction(
                property_iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/prehension#weightSupportedBy",
                quantifier=Quantifier.SOME,
                filler=NamedClass(
                    iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/prehension#FinalSupport"
                ),
            ),
        ),
        definition=None,
    )
    """
    The ontology class this class stands for.
    """

    weight_supported_by: List[core.ManipulationRole] = field(
        default_factory=list,
        metadata=PropertyTerm(
            iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/prehension#weightSupportedBy",
            functional=False,
            range="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#ManipulationRole",
        ).as_dict(),
    )
    """
    A support role bearing the manipulandum's weight once forces are unloaded.
    """


@dataclass(eq=False)
class GraspContact(core.ContactEventType):
    """
    The effector first contacts the manipulandum; the goal of the reach phase.
    """

    term: ClassVar[ClassTerm] = ClassTerm(
        iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/prehension#GraspContact",
        linked_terms=(),
        evidence_status="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#EmpiricalFinding",
        claims=(
            "http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#FJ-003",
            "http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#FJ-004",
            "http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#JF09-12",
        ),
        restrictions=(
            PropertyRestriction(
                property_iri="http://aicor.knowledge/l2-ontology.owl#classifies",
                quantifier=Quantifier.ONLY,
                filler=NamedClass(
                    iri="http://aicor.knowledge/l2-ontology.owl#GraspEvent"
                ),
            ),
            PropertyRestriction(
                property_iri="http://aicor.knowledge/l2-ontology.owl#initiatesStateType",
                quantifier=Quantifier.SOME,
                filler=NamedClass(
                    iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/prehension#EffectorObjectContact"
                ),
            ),
            PropertyRestriction(
                property_iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#startsForce",
                quantifier=Quantifier.SOME,
                filler=PropertyRestriction(
                    property_iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#hasForceKind",
                    quantifier=Quantifier.VALUE,
                    value="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#GripForce",
                ),
            ),
        ),
        definition=None,
    )
    """
    The ontology class this class stands for.
    """


@dataclass(eq=False)
class GraspRelease(core.ContactEventType):
    """
    The effector's contact surfaces break contact with the manipulandum: the sensory goal of the unload phase.
    """

    term: ClassVar[ClassTerm] = ClassTerm(
        iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/prehension#GraspRelease",
        linked_terms=(),
        evidence_status="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#EmpiricalFinding",
        claims=(
            "http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#FJ-003",
        ),
        restrictions=(
            PropertyRestriction(
                property_iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#endsForce",
                quantifier=Quantifier.SOME,
                filler=PropertyRestriction(
                    property_iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#hasForceKind",
                    quantifier=Quantifier.VALUE,
                    value="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#GripForce",
                ),
            ),
            PropertyRestriction(
                property_iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#terminatesStateType",
                quantifier=Quantifier.SOME,
                filler=NamedClass(
                    iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/prehension#EffectorObjectContact"
                ),
            ),
        ),
        definition=None,
    )
    """
    The ontology class this class stands for.
    """


@dataclass(eq=False)
class GraspSite(core.ManipulationRole):
    """
    The region of the manipulandum where the effector makes contact.
    """

    term: ClassVar[ClassTerm] = ClassTerm(
        iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/prehension#GraspSite",
        linked_terms=(),
        evidence_status="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#EmpiricalFinding",
        claims=(
            "http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#JF09-12",
        ),
        restrictions=(),
        definition=None,
    )
    """
    The ontology class this class stands for.
    """

    grasp_site_suited_for: List[ConceptReference] = field(
        default_factory=list,
        metadata=PropertyTerm(
            iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/prehension#graspSiteSuitedFor",
            functional=False,
        ).as_dict(),
    )
    """
    An action the grasp site is chosen for, as the IRI of a CRAM-AICOR action category.
    """


@dataclass(eq=False)
class GraspStability(core.PhaseConditionType):
    """
    The manipulandum is held with grip force above the slip limit and below damaging it.
    """

    term: ClassVar[ClassTerm] = ClassTerm(
        iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/prehension#GraspStability",
        linked_terms=("http://aicor.knowledge/l2-ontology.owl#GraspedState",),
        evidence_status="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#EmpiricalFinding",
        claims=(
            "http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#FJ-025",
            "http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#JF09-06",
            "http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#JF09-07",
        ),
        restrictions=(),
        definition=None,
    )
    """
    The ontology class this class stands for.
    """


@dataclass(eq=False)
class GraspingEffector(core.ManipulationRole):
    """
    The hand, gripper or other effector establishing the grasp.
    """

    term: ClassVar[ClassTerm] = ClassTerm(
        iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/prehension#GraspingEffector",
        linked_terms=("http://aicor.knowledge/l2-ontology.owl#InstrumentRole",),
        evidence_status="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#EmpiricalFinding",
        claims=(),
        restrictions=(),
        definition=None,
    )
    """
    The ontology class this class stands for.
    """


@dataclass(eq=False)
class GraspingEffectorBinding(DescriptionEntity):
    """
    An execution-time binding of the grasping effector slot, whose filler is a robot end effector.
    """

    term: ClassVar[ClassTerm] = ClassTerm(
        iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/prehension#GraspingEffectorBinding",
        linked_terms=("http://aicor.knowledge/l2-ontology.owl#RoleBinding",),
        evidence_status="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#EngineeringExtension",
        claims=(),
        restrictions=(
            PropertyRestriction(
                property_iri="http://aicor.knowledge/l2-ontology.owl#bindsFiller",
                quantifier=Quantifier.ONLY,
                filler=NamedClass(
                    iri="http://aicor.knowledge/l2-ontology.owl#EndEffector"
                ),
            ),
        ),
        definition=IntersectionOf(
            operands=(
                NamedClass(iri="http://aicor.knowledge/l2-ontology.owl#RoleBinding"),
                PropertyRestriction(
                    property_iri="http://aicor.knowledge/l2-ontology.owl#bindsSlot",
                    quantifier=Quantifier.SOME,
                    filler=NamedClass(
                        iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/prehension#GraspingEffector"
                    ),
                ),
            )
        ),
    )
    """
    The ontology class this class stands for.
    """


@dataclass(eq=False)
class GripForceCorridor(DescriptionEntity):
    """
    The admissible grip-force range, between slipping and crushing the manipulandum.
    """

    term: ClassVar[ClassTerm] = ClassTerm(
        iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/prehension#GripForceCorridor",
        linked_terms=(
            "http://aicor.knowledge/l2-ontology.owl#ForceCorridor",
            "http://www.ontologydesignpatterns.org/ont/dul/DUL.owl#Region",
        ),
        evidence_status="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#SupportedInterpretation",
        claims=(
            "http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#JF09-06",
            "http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#JF09-07",
        ),
        restrictions=(
            PropertyRestriction(
                property_iri="http://aicor.knowledge/l2-ontology.owl#lowerEdge",
                quantifier=Quantifier.SOME,
                filler=NamedClass(
                    iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/prehension#SlipEvent"
                ),
            ),
            PropertyRestriction(
                property_iri="http://aicor.knowledge/l2-ontology.owl#upperEdge",
                quantifier=Quantifier.SOME,
                filler=NamedClass(
                    iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/prehension#ObjectCrush"
                ),
            ),
            PropertyRestriction(
                property_iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/prehension#boundsSetpoint",
                quantifier=Quantifier.SOME,
                filler=NamedClass(
                    iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/prehension#GripForceSetpoint"
                ),
            ),
        ),
        definition=None,
    )
    """
    The ontology class this class stands for.
    """

    bounds_setpoint: List[core.MotorSetpoint] = field(
        default_factory=list,
        metadata=PropertyTerm(
            iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/prehension#boundsSetpoint",
            functional=False,
            range="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#MotorSetpoint",
        ).as_dict(),
    )
    """
    A motor setpoint whose admissible values the corridor bounds.
    """

    lower_edge: List[SlipEvent] = field(
        default_factory=list,
        metadata=PropertyTerm(
            iri="http://aicor.knowledge/l2-ontology.owl#lowerEdge",
            functional=False,
            range="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/prehension#SlipEvent",
        ).as_dict(),
    )
    """
    The linked ontology property http://aicor.knowledge/l2-ontology.owl#lowerEdge.
    """

    upper_edge: List[ObjectCrush] = field(
        default_factory=list,
        metadata=PropertyTerm(
            iri="http://aicor.knowledge/l2-ontology.owl#upperEdge",
            functional=False,
            range="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/prehension#ObjectCrush",
        ).as_dict(),
    )
    """
    The linked ontology property http://aicor.knowledge/l2-ontology.owl#upperEdge.
    """


@dataclass(eq=False)
class GripForceSetpoint(core.ForceParameter, core.MotorSetpoint):
    """
    The target of the force applied normal to the grasp surfaces, which creates the friction that holds the load.
    """

    term: ClassVar[ClassTerm] = ClassTerm(
        iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/prehension#GripForceSetpoint",
        linked_terms=(),
        evidence_status="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#EmpiricalFinding",
        claims=(
            "http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#FJ-025",
        ),
        restrictions=(),
        definition=None,
    )
    """
    The ontology class this class stands for.
    """


@dataclass(eq=False)
class GripLoadCoupling(DescriptionEntity):
    """
    The relation setting grip force from the predicted load, friction and safety margin, coupled in time.
    """

    term: ClassVar[ClassTerm] = ClassTerm(
        iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/prehension#GripLoadCoupling",
        linked_terms=(
            "http://aicor.knowledge/l2-ontology.owl#dul_Relation",
            "http://www.ontologydesignpatterns.org/ont/dul/DUL.owl#Relation",
        ),
        evidence_status="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#EmpiricalFinding",
        claims=(
            "http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#FJ-008",
            "http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#FJ-025",
            "http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#JF09-06",
            "http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#WF01-01",
        ),
        restrictions=(
            PropertyRestriction(
                property_iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/prehension#frictionEstimate",
                quantifier=Quantifier.SOME,
                filler=NamedClass(
                    iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/prehension#ExpectedFrictionCoefficient"
                ),
            ),
            PropertyRestriction(
                property_iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/prehension#gripForceResponse",
                quantifier=Quantifier.SOME,
                filler=NamedClass(
                    iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/prehension#GripForceSetpoint"
                ),
            ),
            PropertyRestriction(
                property_iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/prehension#hasLoadOrigin",
                quantifier=Quantifier.SOME,
                filler=NamedClass(
                    iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/prehension#LoadOrigin"
                ),
            ),
            PropertyRestriction(
                property_iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/prehension#hasSafetyMargin",
                quantifier=Quantifier.SOME,
                filler=NamedClass(
                    iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#SafetyMargin"
                ),
            ),
            PropertyRestriction(
                property_iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/prehension#predictedLoad",
                quantifier=Quantifier.SOME,
                filler=NamedClass(
                    iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/object-dynamics#LoadForceEstimate"
                ),
            ),
            PropertyRestriction(
                property_iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/prehension#temporalCoupling",
                quantifier=Quantifier.SOME,
                filler=NamedClass(
                    iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#TemporalWindow"
                ),
            ),
        ),
        definition=None,
    )
    """
    The ontology class this class stands for.
    """

    coupling_active_in: List[core.ActionPhase] = field(
        default_factory=list,
        metadata=PropertyTerm(
            iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/prehension#couplingActiveIn",
            functional=False,
            range="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#ActionPhase",
        ).as_dict(),
    )
    """
    A phase in which grip and load stay functionally coupled.
    """

    follows_load_prediction: List[core.ContinuousSignalPrediction] = field(
        default_factory=list,
        metadata=PropertyTerm(
            iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/prehension#followsLoadPrediction",
            functional=False,
            range="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#ContinuousSignalPrediction",
        ).as_dict(),
    )
    """
    A predicted load trajectory the grip force follows while the load changes, as when substance flows out of a held container.
    """

    friction_estimate: Optional[ExpectedFrictionCoefficient] = field(
        default=None,
        metadata=PropertyTerm(
            iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/prehension#frictionEstimate",
            functional=True,
            range="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/prehension#ExpectedFrictionCoefficient",
        ).as_dict(),
    )
    """
    The estimated friction of the grasp interface that limits tangential force.
    """

    grip_force_response: Optional[GripForceSetpoint] = field(
        default=None,
        metadata=PropertyTerm(
            iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/prehension#gripForceResponse",
            functional=True,
            range="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/prehension#GripForceSetpoint",
        ).as_dict(),
    )
    """
    The grip-force target the relation produces.
    """

    has_load_origin: Optional[LoadOrigin] = field(
        default=None,
        metadata=PropertyTerm(
            iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/prehension#hasLoadOrigin",
            functional=True,
        ).as_dict(),
    )
    """
    Whether the load is generated by the actor's own movement or imposed from outside.
    """

    has_safety_margin: Optional[core.SafetyMargin] = field(
        default=None,
        metadata=PropertyTerm(
            iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/prehension#hasSafetyMargin",
            functional=True,
            range="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#SafetyMargin",
        ).as_dict(),
    )
    """
    The additional grip force kept above the slip threshold.
    """

    predicted_load: Optional[object_dynamics.LoadForceEstimate] = field(
        default=None,
        metadata=PropertyTerm(
            iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/prehension#predictedLoad",
            functional=True,
            range="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/object-dynamics#LoadForceEstimate",
        ).as_dict(),
    )
    """
    The represented tangential load for which grip force is scaled.
    """

    temporal_coupling: Optional[core.TemporalWindow] = field(
        default=None,
        metadata=PropertyTerm(
            iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/prehension#temporalCoupling",
            functional=True,
            range="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#TemporalWindow",
        ).as_dict(),
    )
    """
    The admissible timing between the load and the grip response.
    """


@dataclass(eq=False)
class GripToLoadRatioSetpoint(core.DimensionlessParameter, core.MotorSetpoint):
    """
    The commanded ratio of grip to load force, adapted to friction and surface shape.
    """

    term: ClassVar[ClassTerm] = ClassTerm(
        iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/prehension#GripToLoadRatioSetpoint",
        linked_terms=(),
        evidence_status="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#EmpiricalFinding",
        claims=(
            "http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#JF09-06",
        ),
        restrictions=(),
        definition=None,
    )
    """
    The ontology class this class stands for.
    """


@dataclass(eq=False)
class HoldPhase(core.ActionPhase):
    """
    Holding the manipulandum aloft with a stable grasp. Optional.
    """

    term: ClassVar[ClassTerm] = ClassTerm(
        iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/prehension#HoldPhase",
        linked_terms=(),
        evidence_status="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#EmpiricalFinding",
        claims=(),
        restrictions=(
            PropertyRestriction(
                property_iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#hasCouplingRegime",
                quantifier=Quantifier.VALUE,
                value="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#Coupled",
            ),
        ),
        definition=None,
    )
    """
    The ontology class this class stands for.
    """


@dataclass(eq=False)
class InitialSupport(core.ManipulationRole):
    """
    The surface bearing the manipulandum before lift-off.
    """

    term: ClassVar[ClassTerm] = ClassTerm(
        iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/prehension#InitialSupport",
        linked_terms=(),
        evidence_status="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#EmpiricalFinding",
        claims=(),
        restrictions=(),
        definition=None,
    )
    """
    The ontology class this class stands for.
    """


@dataclass(eq=False)
class LiftClearanceReached(core.PoseAttainmentEventType):
    """
    The manipulandum reaches its commanded clearance above the support.
    """

    term: ClassVar[ClassTerm] = ClassTerm(
        iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/prehension#LiftClearanceReached",
        linked_terms=(),
        evidence_status="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#EngineeringExtension",
        claims=(),
        restrictions=(
            PropertyRestriction(
                property_iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#attainsSetpoint",
                quantifier=Quantifier.SOME,
                filler=NamedClass(
                    iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/prehension#LiftClearanceSetpoint"
                ),
            ),
        ),
        definition=None,
    )
    """
    The ontology class this class stands for.
    """


@dataclass(eq=False)
class LiftClearanceSetpoint(core.LengthParameter, core.MotorSetpoint):
    """
    The commanded clearance above the initial support.
    """

    term: ClassVar[ClassTerm] = ClassTerm(
        iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/prehension#LiftClearanceSetpoint",
        linked_terms=(),
        evidence_status="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#EngineeringExtension",
        claims=(),
        restrictions=(),
        definition=None,
    )
    """
    The ontology class this class stands for.
    """


@dataclass(eq=False)
class LiftPhase(core.ActionPhase):
    """
    Vertical transport of the manipulandum away from its initial support.
    """

    term: ClassVar[ClassTerm] = ClassTerm(
        iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/prehension#LiftPhase",
        linked_terms=(),
        evidence_status="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#EmpiricalFinding",
        claims=(),
        restrictions=(
            PropertyRestriction(
                property_iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#hasCouplingRegime",
                quantifier=Quantifier.VALUE,
                value="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#Coupled",
            ),
            PropertyRestriction(
                property_iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#realizesActionCategory",
                quantifier=Quantifier.VALUE,
                value="http://aicor.knowledge/l2-ontology.owl#Lifting",
            ),
        ),
        definition=None,
    )
    """
    The ontology class this class stands for.
    """


@dataclass(eq=False)
class Liftoff(core.ContactEventType, core.ThresholdCrossingEventType):
    """
    Load force exceeds the weight and the manipulandum leaves its support; the goal of the load phase.
    """

    term: ClassVar[ClassTerm] = ClassTerm(
        iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/prehension#Liftoff",
        linked_terms=(),
        evidence_status="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#EmpiricalFinding",
        claims=(
            "http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#FJ-003",
            "http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#FJ-010",
        ),
        restrictions=(
            PropertyRestriction(
                property_iri="http://aicor.knowledge/l2-ontology.owl#initiatesStateType",
                quantifier=Quantifier.SOME,
                filler=NamedClass(
                    iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/prehension#ObjectClearOfSupport"
                ),
            ),
            PropertyRestriction(
                property_iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#crossesThreshold",
                quantifier=Quantifier.SOME,
                filler=NamedClass(
                    iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/prehension#LiftoffThreshold"
                ),
            ),
            PropertyRestriction(
                property_iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#endsForce",
                quantifier=Quantifier.SOME,
                filler=PropertyRestriction(
                    property_iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#hasForceKind",
                    quantifier=Quantifier.VALUE,
                    value="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#SupportReaction",
                ),
            ),
            PropertyRestriction(
                property_iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#resolvesOpposition",
                quantifier=Quantifier.SOME,
                filler=IntersectionOf(
                    operands=(
                        NamedClass(
                            iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#ForceOpposition"
                        ),
                        PropertyRestriction(
                            property_iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#hasDrivingForce",
                            quantifier=Quantifier.SOME,
                            filler=PropertyRestriction(
                                property_iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#hasForceKind",
                                quantifier=Quantifier.VALUE,
                                value="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#LoadForce",
                            ),
                        ),
                        PropertyRestriction(
                            property_iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#hasResistingForce",
                            quantifier=Quantifier.SOME,
                            filler=PropertyRestriction(
                                property_iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#hasForceKind",
                                quantifier=Quantifier.VALUE,
                                value="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#Weight",
                            ),
                        ),
                    )
                ),
            ),
            PropertyRestriction(
                property_iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#terminatesStateType",
                quantifier=Quantifier.SOME,
                filler=NamedClass(
                    iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/prehension#ObjectSupportContact"
                ),
            ),
        ),
        definition=None,
    )
    """
    The ontology class this class stands for.
    """


@dataclass(eq=False)
class LiftoffThreshold(core.ForceParameter, core.MechanicalThreshold):
    """
    The load force corresponding to the manipulandum's weight, a force and not a mass.
    """

    term: ClassVar[ClassTerm] = ClassTerm(
        iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/prehension#LiftoffThreshold",
        linked_terms=(),
        evidence_status="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#EmpiricalFinding",
        claims=(
            "http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#FJ-010",
        ),
        restrictions=(),
        definition=None,
    )
    """
    The ontology class this class stands for.
    """


@dataclass(eq=False)
class LoadForceRateSetpoint(core.ForceRateSetpoint):
    """
    The commanded rate of load-force increase during loading, scaled in advance to the predicted weight.
    """

    term: ClassVar[ClassTerm] = ClassTerm(
        iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/prehension#LoadForceRateSetpoint",
        linked_terms=(),
        evidence_status="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#EmpiricalFinding",
        claims=(
            "http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#FJ-009",
        ),
        restrictions=(
            PropertyRestriction(
                property_iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#rateOf",
                quantifier=Quantifier.SOME,
                filler=NamedClass(
                    iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/prehension#LoadForceSetpoint"
                ),
            ),
        ),
        definition=None,
    )
    """
    The ontology class this class stands for.
    """


@dataclass(eq=False)
class LoadForceSetpoint(core.ForceParameter, core.MotorSetpoint):
    """
    The target of the force applied tangentially to the grasp surfaces to move the manipulandum.
    """

    term: ClassVar[ClassTerm] = ClassTerm(
        iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/prehension#LoadForceSetpoint",
        linked_terms=(),
        evidence_status="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#EmpiricalFinding",
        claims=(
            "http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#FJ-025",
        ),
        restrictions=(),
        definition=None,
    )
    """
    The ontology class this class stands for.
    """


@dataclass(eq=False)
class LoadRateReduction(core.CorrectiveAction):
    """
    A correction that slows the load-force increase after a slip during loading.
    """

    term: ClassVar[ClassTerm] = ClassTerm(
        iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/prehension#LoadRateReduction",
        linked_terms=(),
        evidence_status="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#EmpiricalFinding",
        claims=(
            "http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#JF09-05",
        ),
        restrictions=(
            PropertyRestriction(
                property_iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#hasCorrectionDirection",
                quantifier=Quantifier.VALUE,
                value="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#LowerSetpoint",
            ),
            PropertyRestriction(
                property_iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#targetsSetpoint",
                quantifier=Quantifier.ONLY,
                filler=NamedClass(
                    iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/prehension#LoadForceRateSetpoint"
                ),
            ),
        ),
        definition=None,
    )
    """
    The ontology class this class stands for.
    """


@dataclass(eq=False)
class ObjectClearOfSupport(core.PhaseConditionType):
    """
    No support bears the manipulandum's weight.
    """

    term: ClassVar[ClassTerm] = ClassTerm(
        iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/prehension#ObjectClearOfSupport",
        linked_terms=(),
        evidence_status="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#EmpiricalFinding",
        claims=(),
        restrictions=(),
        definition=None,
    )
    """
    The ontology class this class stands for.
    """


@dataclass(eq=False)
class ObjectCrush(core.FailureEventType, core.ThresholdCrossingEventType):
    """
    Grip force exceeds what the manipulandum can bear and damages it.
    """

    term: ClassVar[ClassTerm] = ClassTerm(
        iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/prehension#ObjectCrush",
        linked_terms=("http://aicor.knowledge/l2-ontology.owl#RuptureEventType",),
        evidence_status="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#SupportedInterpretation",
        claims=(
            "http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#JF09-07",
        ),
        restrictions=(
            PropertyRestriction(
                property_iri="http://aicor.knowledge/l2-ontology.owl#classifies",
                quantifier=Quantifier.ONLY,
                filler=NamedClass(
                    iri="http://aicor.knowledge/l2-ontology.owl#FractureEvent"
                ),
            ),
            PropertyRestriction(
                property_iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#crossesThreshold",
                quantifier=Quantifier.SOME,
                filler=NamedClass(
                    iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/prehension#CrushThreshold"
                ),
            ),
            PropertyRestriction(
                property_iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#resolvesOpposition",
                quantifier=Quantifier.SOME,
                filler=IntersectionOf(
                    operands=(
                        NamedClass(
                            iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#ForceOpposition"
                        ),
                        PropertyRestriction(
                            property_iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#hasDrivingForce",
                            quantifier=Quantifier.SOME,
                            filler=PropertyRestriction(
                                property_iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#hasForceKind",
                                quantifier=Quantifier.VALUE,
                                value="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#GripForce",
                            ),
                        ),
                        PropertyRestriction(
                            property_iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#hasResistingForce",
                            quantifier=Quantifier.SOME,
                            filler=PropertyRestriction(
                                property_iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#hasForceKind",
                                quantifier=Quantifier.VALUE,
                                value="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#MaterialStrength",
                            ),
                        ),
                    )
                ),
            ),
            PropertyRestriction(
                property_iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#terminatesStateType",
                quantifier=Quantifier.SOME,
                filler=NamedClass(
                    iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/prehension#ObjectIntact"
                ),
            ),
        ),
        definition=None,
    )
    """
    The ontology class this class stands for.
    """


@dataclass(eq=False)
class ObjectIntact(core.PhaseConditionType):
    """
    The manipulandum has not been damaged by the forces applied to it.
    """

    term: ClassVar[ClassTerm] = ClassTerm(
        iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/prehension#ObjectIntact",
        linked_terms=(),
        evidence_status="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#SupportedInterpretation",
        claims=(
            "http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#JF09-07",
        ),
        restrictions=(),
        definition=None,
    )
    """
    The ontology class this class stands for.
    """


@dataclass(eq=False)
class ObjectSupportContact(core.ContactCondition):
    """
    The manipulandum is mechanically in contact with a support.
    """

    term: ClassVar[ClassTerm] = ClassTerm(
        iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/prehension#ObjectSupportContact",
        linked_terms=(),
        evidence_status="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#EmpiricalFinding",
        claims=(),
        restrictions=(),
        definition=None,
    )
    """
    The ontology class this class stands for.
    """


@dataclass(eq=False)
class PrehensileManipulandum(core.Manipulandum):
    """
    The object coupled to an effector by a prehensile grasp.
    """

    term: ClassVar[ClassTerm] = ClassTerm(
        iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/prehension#PrehensileManipulandum",
        linked_terms=(),
        evidence_status="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#EmpiricalFinding",
        claims=(),
        restrictions=(),
        definition=None,
    )
    """
    The ontology class this class stands for.
    """


@dataclass(eq=False)
class PrehensionPhaseStructure(object_dynamics.ObjectDynamicsPhaseStructure):
    """
    A phase structure with explicit prehensile force-control relations.
    """

    term: ClassVar[ClassTerm] = ClassTerm(
        iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/prehension#PrehensionPhaseStructure",
        linked_terms=(),
        evidence_status="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#EngineeringExtension",
        claims=(),
        restrictions=(),
        definition=None,
    )
    """
    The ontology class this class stands for.
    """

    defines_anticipatory_grip_adjustment: List[AnticipatoryGripAdjustment] = field(
        default_factory=list,
        metadata=PropertyTerm(
            iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/prehension#definesAnticipatoryGripAdjustment",
            functional=False,
            range="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/prehension#AnticipatoryGripAdjustment",
        ).as_dict(),
    )
    """
    A prediction-triggered grip adjustment declared by the structure.
    """

    defines_force_coupling: List[GripLoadCoupling] = field(
        default_factory=list,
        metadata=PropertyTerm(
            iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/prehension#definesForceCoupling",
            functional=False,
            range="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/prehension#GripLoadCoupling",
        ).as_dict(),
    )
    """
    A functional grip-load relation declared by the structure.
    """

    defines_grip_force_corridor: List[GripForceCorridor] = field(
        default_factory=list,
        metadata=PropertyTerm(
            iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/prehension#definesGripForceCorridor",
            functional=False,
            range="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/prehension#GripForceCorridor",
        ).as_dict(),
    )
    """
    The admissible grip-force range declared by the structure.
    """


@dataclass(eq=False)
class ProbingForceIncrease(core.CorrectiveAction):
    """
    For an object heavier than predicted: forces increase slowly until the missing lift-off is sensed.
    """

    term: ClassVar[ClassTerm] = ClassTerm(
        iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/prehension#ProbingForceIncrease",
        linked_terms=(),
        evidence_status="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#EmpiricalFinding",
        claims=(
            "http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#FJ-010",
            "http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#JF09-04",
        ),
        restrictions=(
            PropertyRestriction(
                property_iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#correctionTerminatedBy",
                quantifier=Quantifier.SOME,
                filler=NamedClass(
                    iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/prehension#Liftoff"
                ),
            ),
            PropertyRestriction(
                property_iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#hasCorrectionDirection",
                quantifier=Quantifier.VALUE,
                value="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#RaiseSetpoint",
            ),
            PropertyRestriction(
                property_iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#targetsSetpoint",
                quantifier=Quantifier.SOME,
                filler=NamedClass(
                    iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/prehension#LoadForceSetpoint"
                ),
            ),
            PropertyRestriction(
                property_iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#triggeredByError",
                quantifier=Quantifier.SOME,
                filler=PropertyRestriction(
                    property_iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#hasErrorKind",
                    quantifier=Quantifier.VALUE,
                    value="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#AbsentEvent",
                ),
            ),
        ),
        definition=None,
    )
    """
    The ontology class this class stands for.
    """


@dataclass(eq=False)
class ReachPhase(core.ActionPhase):
    """
    Transport of the uncoupled effector to the grasp site; its goal is marked by grasp contact.
    """

    term: ClassVar[ClassTerm] = ClassTerm(
        iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/prehension#ReachPhase",
        linked_terms=(),
        evidence_status="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#EmpiricalFinding",
        claims=(),
        restrictions=(
            PropertyRestriction(
                property_iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#hasCouplingRegime",
                quantifier=Quantifier.VALUE,
                value="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#Uncoupled",
            ),
            PropertyRestriction(
                property_iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#realizesActionCategory",
                quantifier=Quantifier.VALUE,
                value="http://aicor.knowledge/l2-ontology.owl#Reaching",
            ),
        ),
        definition=None,
    )
    """
    The ontology class this class stands for.
    """


@dataclass(eq=False)
class ReachPoseSetpoint(core.CategoricalParameter, core.MotorSetpoint):
    """
    A categorical target for the pre-contact effector pose.
    """

    term: ClassVar[ClassTerm] = ClassTerm(
        iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/prehension#ReachPoseSetpoint",
        linked_terms=(),
        evidence_status="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#EngineeringExtension",
        claims=(),
        restrictions=(),
        definition=None,
    )
    """
    The ontology class this class stands for.
    """


@dataclass(eq=False)
class ReachTargetReached(core.PoseAttainmentEventType):
    """
    The effector reaches the commanded pre-contact pose.
    """

    term: ClassVar[ClassTerm] = ClassTerm(
        iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/prehension#ReachTargetReached",
        linked_terms=(),
        evidence_status="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#EngineeringExtension",
        claims=(),
        restrictions=(
            PropertyRestriction(
                property_iri="http://aicor.knowledge/l2-ontology.owl#initiatesStateType",
                quantifier=Quantifier.SOME,
                filler=NamedClass(
                    iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/prehension#EffectorPositioned"
                ),
            ),
            PropertyRestriction(
                property_iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#attainsSetpoint",
                quantifier=Quantifier.SOME,
                filler=NamedClass(
                    iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/prehension#ReachPoseSetpoint"
                ),
            ),
        ),
        definition=None,
    )
    """
    The ontology class this class stands for.
    """


@dataclass(eq=False)
class ReplacePhase(core.ActionPhase):
    """
    Lowering of the still-coupled manipulandum until touchdown.
    """

    term: ClassVar[ClassTerm] = ClassTerm(
        iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/prehension#ReplacePhase",
        linked_terms=(),
        evidence_status="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#EmpiricalFinding",
        claims=(),
        restrictions=(
            PropertyRestriction(
                property_iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#hasCouplingRegime",
                quantifier=Quantifier.VALUE,
                value="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#Coupled",
            ),
            PropertyRestriction(
                property_iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#terminatedBy",
                quantifier=Quantifier.SOME,
                filler=NamedClass(
                    iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/prehension#Touchdown"
                ),
            ),
        ),
        definition=None,
    )
    """
    The ontology class this class stands for.
    """


@dataclass(eq=False)
class SafeContactForce(core.PhaseConditionType):
    """
    Contact force stays within the limit beyond which a protective response is needed.
    """

    term: ClassVar[ClassTerm] = ClassTerm(
        iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/prehension#SafeContactForce",
        linked_terms=(),
        evidence_status="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#EngineeringExtension",
        claims=(),
        restrictions=(),
        definition=None,
    )
    """
    The ontology class this class stands for.
    """


@dataclass(eq=False)
class SlipEvent(core.FailureEventType, core.ThresholdCrossingEventType):
    """
    Tangential load exceeds the frictional capacity of the grasp.
    """

    term: ClassVar[ClassTerm] = ClassTerm(
        iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/prehension#SlipEvent",
        linked_terms=("http://aicor.knowledge/l2-ontology.owl#SlipEventType",),
        evidence_status="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#EmpiricalFinding",
        claims=(
            "http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#FJ-012",
            "http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#FJ-025",
            "http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#JF09-03",
        ),
        restrictions=(
            PropertyRestriction(
                property_iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#crossesThreshold",
                quantifier=Quantifier.SOME,
                filler=NamedClass(
                    iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/prehension#SlipThreshold"
                ),
            ),
            PropertyRestriction(
                property_iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#resolvesOpposition",
                quantifier=Quantifier.SOME,
                filler=IntersectionOf(
                    operands=(
                        NamedClass(
                            iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#ForceOpposition"
                        ),
                        PropertyRestriction(
                            property_iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#hasDrivingForce",
                            quantifier=Quantifier.SOME,
                            filler=PropertyRestriction(
                                property_iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#hasForceKind",
                                quantifier=Quantifier.VALUE,
                                value="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#LoadForce",
                            ),
                        ),
                        PropertyRestriction(
                            property_iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#hasResistingForce",
                            quantifier=Quantifier.SOME,
                            filler=PropertyRestriction(
                                property_iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#hasForceKind",
                                quantifier=Quantifier.VALUE,
                                value="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#Friction",
                            ),
                        ),
                    )
                ),
            ),
            PropertyRestriction(
                property_iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#terminatesStateType",
                quantifier=Quantifier.SOME,
                filler=NamedClass(
                    iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/prehension#GraspStability"
                ),
            ),
        ),
        definition=None,
    )
    """
    The ontology class this class stands for.
    """


@dataclass(eq=False)
class SlipThreshold(core.ForceParameter, core.MechanicalThreshold):
    """
    The minimum grip force that holds the predicted tangential load, given the friction of the grasp interface.
    """

    term: ClassVar[ClassTerm] = ClassTerm(
        iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/prehension#SlipThreshold",
        linked_terms=(),
        evidence_status="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#EmpiricalFinding",
        claims=(
            "http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#FJ-025",
        ),
        restrictions=(),
        definition=None,
    )
    """
    The ontology class this class stands for.
    """


@dataclass(eq=False)
class StableHoldReached(core.SetpointAttainmentEventType):
    """
    The requested stable holding state has been reached. An optional, task-specific boundary.
    """

    term: ClassVar[ClassTerm] = ClassTerm(
        iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/prehension#StableHoldReached",
        linked_terms=(),
        evidence_status="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#EngineeringExtension",
        claims=(),
        restrictions=(
            PropertyRestriction(
                property_iri="http://aicor.knowledge/l2-ontology.owl#initiatesStateType",
                quantifier=Quantifier.SOME,
                filler=NamedClass(
                    iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/prehension#GraspStability"
                ),
            ),
        ),
        definition=None,
    )
    """
    The ontology class this class stands for.
    """


@dataclass(eq=False)
class Touchdown(core.ContactEventType, core.ThresholdCrossingEventType):
    """
    The manipulandum contacts its destination support; the goal of the replace phase.
    """

    term: ClassVar[ClassTerm] = ClassTerm(
        iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/prehension#Touchdown",
        linked_terms=(),
        evidence_status="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#EmpiricalFinding",
        claims=(
            "http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#FJ-003",
        ),
        restrictions=(
            PropertyRestriction(
                property_iri="http://aicor.knowledge/l2-ontology.owl#initiatesStateType",
                quantifier=Quantifier.SOME,
                filler=NamedClass(
                    iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/prehension#ObjectSupportContact"
                ),
            ),
            PropertyRestriction(
                property_iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#crossesThreshold",
                quantifier=Quantifier.SOME,
                filler=NamedClass(
                    iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/prehension#TouchdownThreshold"
                ),
            ),
            PropertyRestriction(
                property_iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#startsForce",
                quantifier=Quantifier.SOME,
                filler=PropertyRestriction(
                    property_iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#hasForceKind",
                    quantifier=Quantifier.VALUE,
                    value="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#SupportReaction",
                ),
            ),
            PropertyRestriction(
                property_iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#terminatesStateType",
                quantifier=Quantifier.SOME,
                filler=NamedClass(
                    iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/prehension#ObjectClearOfSupport"
                ),
            ),
        ),
        definition=None,
    )
    """
    The ontology class this class stands for.
    """


@dataclass(eq=False)
class TouchdownThreshold(core.ForceParameter, core.MechanicalThreshold):
    """
    The support-reaction force that registers touchdown.
    """

    term: ClassVar[ClassTerm] = ClassTerm(
        iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/prehension#TouchdownThreshold",
        linked_terms=(),
        evidence_status="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#EngineeringExtension",
        claims=(),
        restrictions=(),
        definition=None,
    )
    """
    The ontology class this class stands for.
    """


@dataclass(eq=False)
class TransportPhase(core.ActionPhase):
    """
    Moving the held manipulandum to a destination.
    """

    term: ClassVar[ClassTerm] = ClassTerm(
        iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/prehension#TransportPhase",
        linked_terms=(),
        evidence_status="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#EmpiricalFinding",
        claims=(
            "http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#FJ-008",
        ),
        restrictions=(
            PropertyRestriction(
                property_iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#hasCouplingRegime",
                quantifier=Quantifier.VALUE,
                value="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#Coupled",
            ),
            PropertyRestriction(
                property_iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#realizesActionCategory",
                quantifier=Quantifier.VALUE,
                value="http://aicor.knowledge/l2-ontology.owl#Transporting",
            ),
        ),
        definition=None,
    )
    """
    The ontology class this class stands for.
    """


@dataclass(eq=False)
class TransportPoseSetpoint(core.CategoricalParameter, core.MotorSetpoint):
    """
    A categorical target for the transported manipulandum's pose.
    """

    term: ClassVar[ClassTerm] = ClassTerm(
        iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/prehension#TransportPoseSetpoint",
        linked_terms=(),
        evidence_status="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#EngineeringExtension",
        claims=(),
        restrictions=(),
        definition=None,
    )
    """
    The ontology class this class stands for.
    """


@dataclass(eq=False)
class TransportTargetReached(core.PoseAttainmentEventType):
    """
    The held manipulandum reaches its commanded transport destination.
    """

    term: ClassVar[ClassTerm] = ClassTerm(
        iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/prehension#TransportTargetReached",
        linked_terms=(),
        evidence_status="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#EngineeringExtension",
        claims=(),
        restrictions=(
            PropertyRestriction(
                property_iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#attainsSetpoint",
                quantifier=Quantifier.SOME,
                filler=NamedClass(
                    iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/prehension#TransportPoseSetpoint"
                ),
            ),
        ),
        definition=None,
    )
    """
    The ontology class this class stands for.
    """


@dataclass(eq=False)
class UnloadPhase(core.ActionPhase):
    """
    Grip and load forces decrease in parallel as the support takes the weight.
    """

    term: ClassVar[ClassTerm] = ClassTerm(
        iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/prehension#UnloadPhase",
        linked_terms=(),
        evidence_status="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#EmpiricalFinding",
        claims=(),
        restrictions=(
            PropertyRestriction(
                property_iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#terminatedBy",
                quantifier=Quantifier.SOME,
                filler=NamedClass(
                    iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/prehension#ForcesUnloaded"
                ),
            ),
        ),
        definition=None,
    )
    """
    The ontology class this class stands for.
    """


@dataclass(eq=False)
class WithdrawPhase(core.ActionPhase):
    """
    Optional withdrawal of the effector after release.
    """

    term: ClassVar[ClassTerm] = ClassTerm(
        iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/prehension#WithdrawPhase",
        linked_terms=(),
        evidence_status="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#EngineeringExtension",
        claims=(),
        restrictions=(
            PropertyRestriction(
                property_iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#hasCouplingRegime",
                quantifier=Quantifier.VALUE,
                value="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#Uncoupled",
            ),
            PropertyRestriction(
                property_iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#initiatedBy",
                quantifier=Quantifier.SOME,
                filler=NamedClass(
                    iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/prehension#GraspRelease"
                ),
            ),
            PropertyRestriction(
                property_iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#terminatedBy",
                quantifier=Quantifier.SOME,
                filler=NamedClass(
                    iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/prehension#WithdrawalComplete"
                ),
            ),
        ),
        definition=None,
    )
    """
    The ontology class this class stands for.
    """


@dataclass(eq=False)
class WithdrawalComplete(core.PoseAttainmentEventType):
    """
    The effector reaches its post-release withdrawal target.
    """

    term: ClassVar[ClassTerm] = ClassTerm(
        iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/prehension#WithdrawalComplete",
        linked_terms=(),
        evidence_status="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#EngineeringExtension",
        claims=(),
        restrictions=(
            PropertyRestriction(
                property_iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#attainsSetpoint",
                quantifier=Quantifier.SOME,
                filler=NamedClass(
                    iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/prehension#WithdrawalDistanceSetpoint"
                ),
            ),
        ),
        definition=None,
    )
    """
    The ontology class this class stands for.
    """


@dataclass(eq=False)
class WithdrawalDistanceSetpoint(core.LengthParameter, core.MotorSetpoint):
    """
    The commanded effector withdrawal distance after release.
    """

    term: ClassVar[ClassTerm] = ClassTerm(
        iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/prehension#WithdrawalDistanceSetpoint",
        linked_terms=(),
        evidence_status="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#EngineeringExtension",
        claims=(),
        restrictions=(),
        definition=None,
    )
    """
    The ontology class this class stands for.
    """


# %% property descriptors


@dataclass
class AdjustmentActiveIn(PropertyDescriptor):
    """
    A phase during which the anticipatory change is active.
    """


@dataclass
class AdjustsGripForce(PropertyDescriptor):
    """
    The grip-force target changed before the predicted event.
    """


@dataclass
class BoundsSetpoint(PropertyDescriptor):
    """
    A motor setpoint whose admissible values the corridor bounds.
    """


@dataclass
class CouplingActiveIn(PropertyDescriptor):
    """
    A phase in which grip and load stay functionally coupled.
    """


@dataclass
class DefinesAnticipatoryGripAdjustment(core.DefinesComponent):
    """
    A prediction-triggered grip adjustment declared by the structure.
    """


@dataclass
class DefinesForceCoupling(core.DefinesComponent):
    """
    A functional grip-load relation declared by the structure.
    """


@dataclass
class DefinesGripForceCorridor(PropertyDescriptor):
    """
    The admissible grip-force range declared by the structure.
    """


@dataclass
class FollowsLoadPrediction(PropertyDescriptor):
    """
    A predicted load trajectory the grip force follows while the load changes, as when substance flows out of a held container.
    """


@dataclass
class FrictionEstimate(PropertyDescriptor):
    """
    The estimated friction of the grasp interface that limits tangential force.
    """


@dataclass
class GripForceResponse(PropertyDescriptor):
    """
    The grip-force target the relation produces.
    """


@dataclass
class HasSafetyMargin(PropertyDescriptor):
    """
    The additional grip force kept above the slip threshold.
    """


@dataclass
class PredictedLoad(PropertyDescriptor):
    """
    The represented tangential load for which grip force is scaled.
    """


@dataclass
class TemporalCoupling(PropertyDescriptor):
    """
    The admissible timing between the load and the grip response.
    """


@dataclass
class TriggeredByPrediction(PropertyDescriptor):
    """
    The predicted event consequence that schedules the adjustment.
    """


@dataclass
class WeightSupportedBy(PropertyDescriptor):
    """
    A support role bearing the manipulandum's weight once forces are unloaded.
    """


@dataclass
class LowerEdge(PropertyDescriptor):
    """
    The linked ontology property http://aicor.knowledge/l2-ontology.owl#lowerEdge.
    """


@dataclass
class UpperEdge(PropertyDescriptor):
    """
    The linked ontology property http://aicor.knowledge/l2-ontology.owl#upperEdge.
    """


# %% descriptor attachments

AnticipatoryGripAdjustment.adjustment_active_in = AdjustmentActiveIn(
    AnticipatoryGripAdjustment, "adjustment_active_in"
)
AnticipatoryGripAdjustment.adjusts_grip_force = AdjustsGripForce(
    AnticipatoryGripAdjustment, "adjusts_grip_force"
)
GripForceCorridor.bounds_setpoint = BoundsSetpoint(GripForceCorridor, "bounds_setpoint")
GripLoadCoupling.coupling_active_in = CouplingActiveIn(
    GripLoadCoupling, "coupling_active_in"
)
PrehensionPhaseStructure.defines_anticipatory_grip_adjustment = (
    DefinesAnticipatoryGripAdjustment(
        PrehensionPhaseStructure, "defines_anticipatory_grip_adjustment"
    )
)
PrehensionPhaseStructure.defines_force_coupling = DefinesForceCoupling(
    PrehensionPhaseStructure, "defines_force_coupling"
)
PrehensionPhaseStructure.defines_grip_force_corridor = DefinesGripForceCorridor(
    PrehensionPhaseStructure, "defines_grip_force_corridor"
)
GripLoadCoupling.follows_load_prediction = FollowsLoadPrediction(
    GripLoadCoupling, "follows_load_prediction"
)
GripLoadCoupling.friction_estimate = FrictionEstimate(
    GripLoadCoupling, "friction_estimate"
)
GripLoadCoupling.grip_force_response = GripForceResponse(
    GripLoadCoupling, "grip_force_response"
)
GripLoadCoupling.has_safety_margin = HasSafetyMargin(
    GripLoadCoupling, "has_safety_margin"
)
GripLoadCoupling.predicted_load = PredictedLoad(GripLoadCoupling, "predicted_load")
GripLoadCoupling.temporal_coupling = TemporalCoupling(
    GripLoadCoupling, "temporal_coupling"
)
AnticipatoryGripAdjustment.triggered_by_prediction = TriggeredByPrediction(
    AnticipatoryGripAdjustment, "triggered_by_prediction"
)
ForcesUnloaded.weight_supported_by = WeightSupportedBy(
    ForcesUnloaded, "weight_supported_by"
)
GripForceCorridor.lower_edge = LowerEdge(GripForceCorridor, "lower_edge")
GripForceCorridor.upper_edge = UpperEdge(GripForceCorridor, "upper_edge")
