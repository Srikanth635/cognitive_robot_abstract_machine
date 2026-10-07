"""
Represented object and limb dynamics: efference copies, dynamics models, prediction sources, sensorimotor memory and transfer between perturbation contexts.

Generated from ``object-dynamics.ttl`` by ``python -m llm_reasoner.ontology_translation``. Regenerate rather than edit.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

from krrood.ontomatic.property_descriptor.mixins import HasInverseProperty
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


class CoordinateFrame(StrEnum):
    """
    The reference frame in which represented dynamics are encoded.
    """

    OBJECT_CENTERED = "http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/object-dynamics#ObjectCentered"
    """
    Coordinates fixed to the manipulated object.
    """

    ARM_CENTERED = "http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/object-dynamics#ArmCentered"
    """
    Intrinsic or extrinsic coordinates associated with the arm.
    """

    HAND_CENTERED = "http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/object-dynamics#HandCentered"
    """
    Coordinates fixed to the hand or end effector.
    """

    GAZE_CENTERED = "http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/object-dynamics#GazeCentered"
    """
    Coordinates organised relative to the current gaze.
    """


class MemoryPersistence(StrEnum):
    """
    How long a sensorimotor memory remains recruited.
    """

    WHILE_OBJECT_COUPLED = "http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/object-dynamics#WhileObjectCoupled"
    """
    The memory is recruited only while its object remains coupled to the effector.
    """

    AFTER_RELEASE = "http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/object-dynamics#AfterRelease"
    """
    The memory remains available after the object is released.
    """


class PerturbationKind(StrEnum):
    """
    The manipulation applied when dynamics transfer is assessed.
    """

    LOAD_PERTURBATION = "http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/object-dynamics#LoadPerturbation"
    """
    An external force or load is applied.
    """

    ROTATION_PERTURBATION = "http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/object-dynamics#RotationPerturbation"
    """
    The object, hand or arm orientation is changed.
    """


class RepresentationPurpose(StrEnum):
    """
    The function for which represented content is recruited.
    """

    SENSORIMOTOR_CONTROL = "http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/object-dynamics#SensorimotorControl"
    """
    Content used to predict and scale motor commands.
    """

    PERCEPTUAL_JUDGMENT = "http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/object-dynamics#PerceptualJudgment"
    """
    Content used for conscious property judgments.
    """


class TransferOutcome(StrEnum):
    """
    Whether adaptation acquired in one context carries over to another.
    """

    GENERALIZES = "http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/object-dynamics#Generalizes"
    """
    Adaptation acquired in the first context also holds in the second.
    """

    DOES_NOT_GENERALIZE = "http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/object-dynamics#DoesNotGeneralize"
    """
    Adaptation acquired in the first context does not hold in the second.
    """


# %% classes


@dataclass(eq=False)
class ContextualRepresentation(core.EstimateRepresentation):
    """
    Represented estimates with an explicit coordinate frame, purpose and set of participants.
    """

    term: ClassVar[ClassTerm] = ClassTerm(
        iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/object-dynamics#ContextualRepresentation",
        linked_terms=(),
        evidence_status="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#EngineeringExtension",
        claims=(),
        restrictions=(
            PropertyRestriction(
                property_iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#holdsEstimate",
                quantifier=Quantifier.SOME,
                filler=NamedClass(
                    iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#Estimate"
                ),
            ),
            PropertyRestriction(
                property_iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/object-dynamics#hasCoordinateFrame",
                quantifier=Quantifier.SOME,
                filler=NamedClass(
                    iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/object-dynamics#CoordinateFrame"
                ),
            ),
            PropertyRestriction(
                property_iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/object-dynamics#hasPurpose",
                quantifier=Quantifier.SOME,
                filler=NamedClass(
                    iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/object-dynamics#RepresentationPurpose"
                ),
            ),
            PropertyRestriction(
                property_iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/object-dynamics#representsRole",
                quantifier=Quantifier.SOME,
                filler=NamedClass(
                    iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#ManipulationRole"
                ),
            ),
        ),
        definition=None,
    )
    """
    The ontology class this class stands for.
    """

    has_coordinate_frame: Optional[CoordinateFrame] = field(
        default=None,
        metadata=PropertyTerm(
            iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/object-dynamics#hasCoordinateFrame",
            functional=True,
        ).as_dict(),
    )
    """
    The frame in which the represented content or the perturbation is expressed.
    """

    has_purpose: Optional[RepresentationPurpose] = field(
        default=None,
        metadata=PropertyTerm(
            iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/object-dynamics#hasPurpose",
            functional=True,
        ).as_dict(),
    )
    """
    The function for which the represented content is recruited.
    """

    represents_role: List[core.ManipulationRole] = field(
        default_factory=list,
        metadata=PropertyTerm(
            iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/object-dynamics#representsRole",
            functional=False,
            range="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#ManipulationRole",
        ).as_dict(),
    )
    """
    A participant role whose properties or dynamics are represented.
    """


@dataclass(eq=False)
class DesiredState(DescriptionEntity):
    """
    A state an inverse model seeks to establish.
    """

    term: ClassVar[ClassTerm] = ClassTerm(
        iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/object-dynamics#DesiredState",
        linked_terms=(
            "http://aicor.knowledge/l2-ontology.owl#Goal",
            "http://www.ontologydesignpatterns.org/ont/dul/DUL.owl#Goal",
        ),
        evidence_status="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#SupportedInterpretation",
        claims=(
            "http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#FJ-002",
        ),
        restrictions=(
            PropertyRestriction(
                property_iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/object-dynamics#desiredCondition",
                quantifier=Quantifier.SOME,
                filler=NamedClass(
                    iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#PhaseConditionType"
                ),
            ),
        ),
        definition=None,
    )
    """
    The ontology class this class stands for.
    """

    desired_condition: List[core.PhaseConditionType] = field(
        default_factory=list,
        metadata=PropertyTerm(
            iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/object-dynamics#desiredCondition",
            functional=False,
            range="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#PhaseConditionType",
        ).as_dict(),
    )
    """
    A condition whose establishment defines the desired state.
    """


@dataclass(eq=False)
class DynamicsModel(ContextualRepresentation):
    """
    A represented mapping from command-related input to mechanical change, recruited for sensorimotor control.
    """

    term: ClassVar[ClassTerm] = ClassTerm(
        iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/object-dynamics#DynamicsModel",
        linked_terms=(),
        evidence_status="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#SupportedInterpretation",
        claims=(
            "http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#FJ-008",
            "http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#FJ-025",
        ),
        restrictions=(
            PropertyRestriction(
                property_iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/object-dynamics#hasPurpose",
                quantifier=Quantifier.VALUE,
                value="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/object-dynamics#SensorimotorControl",
            ),
            PropertyRestriction(
                property_iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/object-dynamics#usesEfferenceCopy",
                quantifier=Quantifier.SOME,
                filler=NamedClass(
                    iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/object-dynamics#EfferenceCopy"
                ),
            ),
        ),
        definition=None,
    )
    """
    The ontology class this class stands for.
    """

    uses_efference_copy: List[EfferenceCopy] = field(
        default_factory=list,
        metadata=PropertyTerm(
            iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/object-dynamics#usesEfferenceCopy",
            functional=False,
            range="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/object-dynamics#EfferenceCopy",
        ).as_dict(),
    )
    """
    An outgoing command copy used to predict the load the command will meet.
    """


@dataclass(eq=False)
class DynamicsPerturbation(DescriptionEntity):
    """
    A type of load or rotation applied to one represented dynamics system, such as the held object or the arm.
    """

    term: ClassVar[ClassTerm] = ClassTerm(
        iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/object-dynamics#DynamicsPerturbation",
        linked_terms=(
            "http://aicor.knowledge/l2-ontology.owl#dul_Concept",
            "http://www.ontologydesignpatterns.org/ont/dul/DUL.owl#EventType",
        ),
        evidence_status="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#EmpiricalFinding",
        claims=(
            "http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#FJ-026",
            "http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#FJ-030",
        ),
        restrictions=(
            PropertyRestriction(
                property_iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/object-dynamics#affectsModel",
                quantifier=Quantifier.SOME,
                filler=NamedClass(
                    iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/object-dynamics#DynamicsModel"
                ),
            ),
            PropertyRestriction(
                property_iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/object-dynamics#hasCoordinateFrame",
                quantifier=Quantifier.SOME,
                filler=NamedClass(
                    iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/object-dynamics#CoordinateFrame"
                ),
            ),
            PropertyRestriction(
                property_iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/object-dynamics#hasPerturbationKind",
                quantifier=Quantifier.SOME,
                filler=NamedClass(
                    iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/object-dynamics#PerturbationKind"
                ),
            ),
            PropertyRestriction(
                property_iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/object-dynamics#matchedWith",
                quantifier=Quantifier.ONLY,
                filler=NamedClass(
                    iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/object-dynamics#DynamicsPerturbation"
                ),
            ),
        ),
        definition=None,
    )
    """
    The ontology class this class stands for.
    """

    affects_model: Optional[DynamicsModel] = field(
        default=None,
        metadata=PropertyTerm(
            iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/object-dynamics#affectsModel",
            functional=True,
            range="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/object-dynamics#DynamicsModel",
        ).as_dict(),
    )
    """
    The object or limb dynamics subjected to the perturbation.
    """

    has_coordinate_frame: Optional[CoordinateFrame] = field(
        default=None,
        metadata=PropertyTerm(
            iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/object-dynamics#hasCoordinateFrame",
            functional=True,
        ).as_dict(),
    )
    """
    The frame in which the represented content or the perturbation is expressed.
    """

    has_perturbation_kind: Optional[PerturbationKind] = field(
        default=None,
        metadata=PropertyTerm(
            iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/object-dynamics#hasPerturbationKind",
            functional=True,
        ).as_dict(),
    )
    """
    The physical transformation used to assess dynamics transfer.
    """

    matched_with: List[DynamicsPerturbation] = field(
        default_factory=list,
        metadata=PropertyTerm(
            iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/object-dynamics#matchedWith",
            functional=False,
            range="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/object-dynamics#DynamicsPerturbation",
        ).as_dict(),
    )
    """
    A perturbation matched physically with this one for a transfer comparison.
    """


@dataclass(eq=False)
class DynamicsTransfer(DescriptionEntity):
    """
    Whether adaptation acquired in one perturbation context holds in another.
    """

    term: ClassVar[ClassTerm] = ClassTerm(
        iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/object-dynamics#DynamicsTransfer",
        linked_terms=(
            "http://aicor.knowledge/l2-ontology.owl#dul_Relation",
            "http://www.ontologydesignpatterns.org/ont/dul/DUL.owl#Relation",
        ),
        evidence_status="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#EmpiricalFinding",
        claims=(
            "http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#FJ-030",
        ),
        restrictions=(
            PropertyRestriction(
                property_iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/object-dynamics#fromPerturbation",
                quantifier=Quantifier.SOME,
                filler=NamedClass(
                    iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/object-dynamics#DynamicsPerturbation"
                ),
            ),
            PropertyRestriction(
                property_iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/object-dynamics#hasTransferOutcome",
                quantifier=Quantifier.SOME,
                filler=NamedClass(
                    iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/object-dynamics#TransferOutcome"
                ),
            ),
            PropertyRestriction(
                property_iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/object-dynamics#toPerturbation",
                quantifier=Quantifier.SOME,
                filler=NamedClass(
                    iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/object-dynamics#DynamicsPerturbation"
                ),
            ),
        ),
        definition=None,
    )
    """
    The ontology class this class stands for.
    """

    from_perturbation: Optional[DynamicsPerturbation] = field(
        default=None,
        metadata=PropertyTerm(
            iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/object-dynamics#fromPerturbation",
            functional=True,
            range="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/object-dynamics#DynamicsPerturbation",
        ).as_dict(),
    )
    """
    The context in which adaptation was acquired.
    """

    has_transfer_outcome: Optional[TransferOutcome] = field(
        default=None,
        metadata=PropertyTerm(
            iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/object-dynamics#hasTransferOutcome",
            functional=True,
        ).as_dict(),
    )
    """
    Whether adaptation generalised across the contexts.
    """

    to_perturbation: Optional[DynamicsPerturbation] = field(
        default=None,
        metadata=PropertyTerm(
            iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/object-dynamics#toPerturbation",
            functional=True,
            range="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/object-dynamics#DynamicsPerturbation",
        ).as_dict(),
    )
    """
    The context in which transfer was assessed.
    """


@dataclass(eq=False)
class EfferenceCopy(DescriptionEntity):
    """
    An internal copy of an outgoing motor command, used to predict its consequences.
    """

    term: ClassVar[ClassTerm] = ClassTerm(
        iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/object-dynamics#EfferenceCopy",
        linked_terms=(
            "http://aicor.knowledge/l2-ontology.owl#dul_InformationObject",
            "http://www.ontologydesignpatterns.org/ont/dul/DUL.owl#InformationObject",
        ),
        evidence_status="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#SupportedInterpretation",
        claims=(
            "http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#FJ-008",
        ),
        restrictions=(
            PropertyRestriction(
                property_iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/object-dynamics#copiesCommand",
                quantifier=Quantifier.SOME,
                filler=NamedClass(
                    iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/object-dynamics#MotorCommand"
                ),
            ),
        ),
        definition=None,
    )
    """
    The ontology class this class stands for.
    """

    copies_command: Optional[MotorCommand] = field(
        default=None,
        metadata=PropertyTerm(
            iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/object-dynamics#copiesCommand",
            functional=True,
            range="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/object-dynamics#MotorCommand",
        ).as_dict(),
    )
    """
    The outgoing command the copy represents.
    """


@dataclass(eq=False)
class InternalModelComposition(DescriptionEntity):
    """
    How a command copy, interpreted by a dynamics model, becomes predicted mechanical and sensory consequences.
    """

    term: ClassVar[ClassTerm] = ClassTerm(
        iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/object-dynamics#InternalModelComposition",
        linked_terms=(
            "http://aicor.knowledge/l2-ontology.owl#dul_Description",
            "http://www.ontologydesignpatterns.org/ont/dul/DUL.owl#Description",
        ),
        evidence_status="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#SupportedInterpretation",
        claims=(
            "http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#FJ-002",
            "http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#FJ-008",
        ),
        restrictions=(
            PropertyRestriction(
                property_iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/object-dynamics#predictsMechanicalConsequence",
                quantifier=Quantifier.SOME,
                filler=NamedClass(
                    iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#MechanicalEventType"
                ),
            ),
            PropertyRestriction(
                property_iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/object-dynamics#predictsSensoryConsequence",
                quantifier=Quantifier.SOME,
                filler=NamedClass(
                    iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#SensoryEventPrediction"
                ),
            ),
            PropertyRestriction(
                property_iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/object-dynamics#usesDynamicsModel",
                quantifier=Quantifier.SOME,
                filler=NamedClass(
                    iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/object-dynamics#DynamicsModel"
                ),
            ),
            PropertyRestriction(
                property_iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/object-dynamics#usesEfferenceCopy",
                quantifier=Quantifier.SOME,
                filler=NamedClass(
                    iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/object-dynamics#EfferenceCopy"
                ),
            ),
            PropertyRestriction(
                property_iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/object-dynamics#usesForwardModel",
                quantifier=Quantifier.SOME,
                filler=NamedClass(
                    iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#ForwardModel"
                ),
            ),
        ),
        definition=None,
    )
    """
    The ontology class this class stands for.
    """

    predicts_mechanical_consequence: List[core.MechanicalEventType] = field(
        default_factory=list,
        metadata=PropertyTerm(
            iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/object-dynamics#predictsMechanicalConsequence",
            functional=False,
            range="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#MechanicalEventType",
        ).as_dict(),
    )
    """
    A mechanical event the composition predicts.
    """

    predicts_sensory_consequence: List[core.SensoryEventPrediction] = field(
        default_factory=list,
        metadata=PropertyTerm(
            iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/object-dynamics#predictsSensoryConsequence",
            functional=False,
            range="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#SensoryEventPrediction",
        ).as_dict(),
    )
    """
    A sensory consequence the forward model generates.
    """

    uses_dynamics_model: Optional[DynamicsModel] = field(
        default=None,
        metadata=PropertyTerm(
            iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/object-dynamics#usesDynamicsModel",
            functional=True,
            range="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/object-dynamics#DynamicsModel",
        ).as_dict(),
    )
    """
    The object or limb dynamics used to interpret the command copy.
    """

    uses_efference_copy: List[EfferenceCopy] = field(
        default_factory=list,
        metadata=PropertyTerm(
            iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/object-dynamics#usesEfferenceCopy",
            functional=False,
            range="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/object-dynamics#EfferenceCopy",
        ).as_dict(),
    )
    """
    An outgoing command copy used to predict the load the command will meet.
    """

    uses_forward_model: Optional[core.ForwardModel] = field(
        default=None,
        metadata=PropertyTerm(
            iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/object-dynamics#usesForwardModel",
            functional=True,
            range="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#ForwardModel",
        ).as_dict(),
    )
    """
    The model that generates the predicted consequences.
    """


@dataclass(eq=False)
class LimbDynamicsModel(DynamicsModel):
    """
    A dynamics model of the actor's own limb, separate from object dynamics.
    """

    term: ClassVar[ClassTerm] = ClassTerm(
        iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/object-dynamics#LimbDynamicsModel",
        linked_terms=("http://aicor.knowledge/l2-ontology.owl#DynamicModel",),
        evidence_status="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#SupportedInterpretation",
        claims=(
            "http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#FJ-026",
            "http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#FJ-027",
        ),
        restrictions=(),
        definition=None,
    )
    """
    The ontology class this class stands for.
    """


@dataclass(eq=False)
class LoadForceEstimate(core.DynamicsEstimate, core.ForceParameter):
    """
    An estimate of the force arising from object or limb dynamics.
    """

    term: ClassVar[ClassTerm] = ClassTerm(
        iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/object-dynamics#LoadForceEstimate",
        linked_terms=(),
        evidence_status="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#EmpiricalFinding",
        claims=(
            "http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#FJ-008",
            "http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#FJ-025",
        ),
        restrictions=(),
        definition=None,
    )
    """
    The ontology class this class stands for.
    """


@dataclass(eq=False)
class MotorCommand(DescriptionEntity):
    """
    A command an inverse model selects for a desired state, specifying motor setpoints.
    """

    term: ClassVar[ClassTerm] = ClassTerm(
        iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/object-dynamics#MotorCommand",
        linked_terms=(
            "http://aicor.knowledge/l2-ontology.owl#ActuatorInput",
            "http://www.ontologydesignpatterns.org/ont/dul/DUL.owl#InformationObject",
        ),
        evidence_status="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#SupportedInterpretation",
        claims=(
            "http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#FJ-002",
        ),
        restrictions=(
            PropertyRestriction(
                property_iri="http://aicor.knowledge/l2-ontology.owl#generatedBy",
                quantifier=Quantifier.SOME,
                filler=NamedClass(
                    iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#InverseModel"
                ),
            ),
            PropertyRestriction(
                property_iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/object-dynamics#commandsSetpoint",
                quantifier=Quantifier.SOME,
                filler=NamedClass(
                    iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#MotorSetpoint"
                ),
            ),
            PropertyRestriction(
                property_iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/object-dynamics#pursuesState",
                quantifier=Quantifier.SOME,
                filler=NamedClass(
                    iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/object-dynamics#DesiredState"
                ),
            ),
        ),
        definition=None,
    )
    """
    The ontology class this class stands for.
    """

    commands_setpoint: List[core.MotorSetpoint] = field(
        default_factory=list,
        metadata=PropertyTerm(
            iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/object-dynamics#commandsSetpoint",
            functional=False,
            range="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#MotorSetpoint",
        ).as_dict(),
    )
    """
    A motor setpoint the command specifies.
    """

    pursues_state: Optional[DesiredState] = field(
        default=None,
        metadata=PropertyTerm(
            iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/object-dynamics#pursuesState",
            functional=True,
            range="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/object-dynamics#DesiredState",
        ).as_dict(),
    )
    """
    The desired state the command is intended to establish.
    """

    generated_by: List[core.InternalModel] = field(
        default_factory=list,
        metadata=PropertyTerm(
            iri="http://aicor.knowledge/l2-ontology.owl#generatedBy",
            functional=False,
            range="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#InternalModel",
        ).as_dict(),
    )
    """
    The linked ontology property http://aicor.knowledge/l2-ontology.owl#generatedBy.
    """


@dataclass(eq=False)
class ObjectDynamicsModel(DynamicsModel, core.ObjectRepresentation):
    """
    A dynamics model of a manipulated object, recruited with grasp and released with release.
    """

    term: ClassVar[ClassTerm] = ClassTerm(
        iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/object-dynamics#ObjectDynamicsModel",
        linked_terms=(),
        evidence_status="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#SupportedInterpretation",
        claims=(
            "http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#FJ-008",
            "http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#FJ-025",
            "http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#FJ-027",
        ),
        restrictions=(),
        definition=None,
    )
    """
    The ontology class this class stands for.
    """


@dataclass(eq=False)
class ObjectDynamicsPhaseStructure(core.ActionPhaseStructure):
    """
    A phase structure with explicit internal models and the command-to-prediction pathways that use them.
    """

    term: ClassVar[ClassTerm] = ClassTerm(
        iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/object-dynamics#ObjectDynamicsPhaseStructure",
        linked_terms=(),
        evidence_status="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#EngineeringExtension",
        claims=(),
        restrictions=(
            PropertyRestriction(
                property_iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#definesInternalModel",
                quantifier=Quantifier.SOME,
                filler=NamedClass(
                    iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#InternalModel"
                ),
            ),
        ),
        definition=None,
    )
    """
    The ontology class this class stands for.
    """

    defines_internal_model: List[core.InternalModel] = field(
        default_factory=list,
        metadata=PropertyTerm(
            iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#definesInternalModel",
            functional=False,
            range="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#InternalModel",
        ).as_dict(),
    )
    """
    An internal model or represented content that governs the controllers.
    """

    defines_desired_state: List[DesiredState] = field(
        default_factory=list,
        metadata=PropertyTerm(
            iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/object-dynamics#definesDesiredState",
            functional=False,
            range="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/object-dynamics#DesiredState",
        ).as_dict(),
    )
    """
    A desired state supplied to inverse models.
    """

    defines_dynamics_perturbation: List[DynamicsPerturbation] = field(
        default_factory=list,
        metadata=PropertyTerm(
            iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/object-dynamics#definesDynamicsPerturbation",
            functional=False,
            range="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/object-dynamics#DynamicsPerturbation",
        ).as_dict(),
    )
    """
    A load or rotation context used to assess transfer.
    """

    defines_dynamics_transfer: List[DynamicsTransfer] = field(
        default_factory=list,
        metadata=PropertyTerm(
            iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/object-dynamics#definesDynamicsTransfer",
            functional=False,
            range="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/object-dynamics#DynamicsTransfer",
        ).as_dict(),
    )
    """
    A transfer result declared by the structure.
    """

    defines_efference_copy: List[EfferenceCopy] = field(
        default_factory=list,
        metadata=PropertyTerm(
            iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/object-dynamics#definesEfferenceCopy",
            functional=False,
            range="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/object-dynamics#EfferenceCopy",
        ).as_dict(),
    )
    """
    A command copy available to dynamics models.
    """

    defines_internal_model_composition: List[InternalModelComposition] = field(
        default_factory=list,
        metadata=PropertyTerm(
            iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/object-dynamics#definesInternalModelComposition",
            functional=False,
            range="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/object-dynamics#InternalModelComposition",
        ).as_dict(),
    )
    """
    A command-to-prediction pathway declared by the structure.
    """

    defines_motor_command: List[MotorCommand] = field(
        default_factory=list,
        metadata=PropertyTerm(
            iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/object-dynamics#definesMotorCommand",
            functional=False,
            range="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/object-dynamics#MotorCommand",
        ).as_dict(),
    )
    """
    An outgoing motor command declared by the structure.
    """

    defines_representation_dissociation: List[RepresentationDissociation] = field(
        default_factory=list,
        metadata=PropertyTerm(
            iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/object-dynamics#definesRepresentationDissociation",
            functional=False,
            range="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/object-dynamics#RepresentationDissociation",
        ).as_dict(),
    )
    """
    A sensorimotor-perceptual dissociation declared by the structure.
    """

    defines_sensorimotor_memory: List[SensorimotorMemory] = field(
        default_factory=list,
        metadata=PropertyTerm(
            iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/object-dynamics#definesSensorimotorMemory",
            functional=False,
            range="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/object-dynamics#SensorimotorMemory",
        ).as_dict(),
    )
    """
    A memory retaining sensorimotor representations.
    """


@dataclass(eq=False)
class PerceptualRepresentation(ContextualRepresentation):
    """
    Represented content used for conscious judgments of object properties.
    """

    term: ClassVar[ClassTerm] = ClassTerm(
        iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/object-dynamics#PerceptualRepresentation",
        linked_terms=(),
        evidence_status="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#EmpiricalFinding",
        claims=(
            "http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#FJ-031",
        ),
        restrictions=(
            PropertyRestriction(
                property_iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/object-dynamics#hasPurpose",
                quantifier=Quantifier.VALUE,
                value="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/object-dynamics#PerceptualJudgment",
            ),
        ),
        definition=None,
    )
    """
    The ontology class this class stands for.
    """


@dataclass(eq=False)
class RepresentationDissociation(DescriptionEntity):
    """
    A sensorimotor estimate that stays accurate while a perceptual judgment of the same property stays biased.
    """

    term: ClassVar[ClassTerm] = ClassTerm(
        iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/object-dynamics#RepresentationDissociation",
        linked_terms=(
            "http://aicor.knowledge/l2-ontology.owl#dul_Relation",
            "http://www.ontologydesignpatterns.org/ont/dul/DUL.owl#Relation",
        ),
        evidence_status="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#EmpiricalFinding",
        claims=(
            "http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#FJ-031",
        ),
        restrictions=(
            PropertyRestriction(
                property_iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/object-dynamics#hasPerceptualRepresentation",
                quantifier=Quantifier.SOME,
                filler=NamedClass(
                    iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/object-dynamics#PerceptualRepresentation"
                ),
            ),
            PropertyRestriction(
                property_iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/object-dynamics#hasSensorimotorRepresentation",
                quantifier=Quantifier.SOME,
                filler=NamedClass(
                    iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/object-dynamics#DynamicsModel"
                ),
            ),
            PropertyRestriction(
                property_iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/object-dynamics#perceptualEstimate",
                quantifier=Quantifier.SOME,
                filler=NamedClass(
                    iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#Estimate"
                ),
            ),
            PropertyRestriction(
                property_iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/object-dynamics#sensorimotorEstimate",
                quantifier=Quantifier.SOME,
                filler=NamedClass(
                    iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#Estimate"
                ),
            ),
        ),
        definition=None,
    )
    """
    The ontology class this class stands for.
    """

    has_perceptual_representation: Optional[PerceptualRepresentation] = field(
        default=None,
        metadata=PropertyTerm(
            iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/object-dynamics#hasPerceptualRepresentation",
            functional=True,
            range="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/object-dynamics#PerceptualRepresentation",
        ).as_dict(),
    )
    """
    The representation used for conscious judgment.
    """

    has_sensorimotor_representation: Optional[DynamicsModel] = field(
        default=None,
        metadata=PropertyTerm(
            iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/object-dynamics#hasSensorimotorRepresentation",
            functional=True,
            range="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/object-dynamics#DynamicsModel",
        ).as_dict(),
    )
    """
    The representation used for motor prediction and scaling.
    """

    perceptual_estimate: Optional[core.Estimate] = field(
        default=None,
        metadata=PropertyTerm(
            iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/object-dynamics#perceptualEstimate",
            functional=True,
            range="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#Estimate",
        ).as_dict(),
    )
    """
    The biased estimate held by the perceptual representation.
    """

    sensorimotor_estimate: Optional[core.Estimate] = field(
        default=None,
        metadata=PropertyTerm(
            iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/object-dynamics#sensorimotorEstimate",
            functional=True,
            range="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#Estimate",
        ).as_dict(),
    )
    """
    The accurate estimate held by the sensorimotor representation.
    """


@dataclass(eq=False)
class RetainedModelUpdate(core.ModelUpdate):
    """
    A revision of represented estimates that a sensorimotor memory retains for later manipulations.
    """

    term: ClassVar[ClassTerm] = ClassTerm(
        iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/object-dynamics#RetainedModelUpdate",
        linked_terms=(),
        evidence_status="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#EmpiricalFinding",
        claims=(
            "http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#FJ-011",
        ),
        restrictions=(
            PropertyRestriction(
                property_iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/object-dynamics#persistsIn",
                quantifier=Quantifier.SOME,
                filler=NamedClass(
                    iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/object-dynamics#SensorimotorMemory"
                ),
            ),
        ),
        definition=None,
    )
    """
    The ontology class this class stands for.
    """

    persists_in: Optional[SensorimotorMemory] = field(
        default=None,
        metadata=PropertyTerm(
            iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/object-dynamics#persistsIn",
            functional=True,
            range="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/object-dynamics#SensorimotorMemory",
        ).as_dict(),
    )
    """
    The memory in which the revised representation is retained.
    """


@dataclass(eq=False)
class SensorimotorMemory(DescriptionEntity):
    """
    A store of sensorimotor representations retained across manipulations.
    """

    term: ClassVar[ClassTerm] = ClassTerm(
        iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/object-dynamics#SensorimotorMemory",
        linked_terms=(
            "http://www.ontologydesignpatterns.org/ont/dul/DUL.owl#Collection",
        ),
        evidence_status="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#EmpiricalFinding",
        claims=(
            "http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#FJ-009",
            "http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#FJ-028",
        ),
        restrictions=(
            PropertyRestriction(
                property_iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/object-dynamics#hasMemoryPersistence",
                quantifier=Quantifier.SOME,
                filler=NamedClass(
                    iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/object-dynamics#MemoryPersistence"
                ),
            ),
            PropertyRestriction(
                property_iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/object-dynamics#storesRepresentation",
                quantifier=Quantifier.SOME,
                filler=NamedClass(
                    iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/object-dynamics#ContextualRepresentation"
                ),
            ),
        ),
        definition=None,
    )
    """
    The ontology class this class stands for.
    """

    has_memory_persistence: Optional[MemoryPersistence] = field(
        default=None,
        metadata=PropertyTerm(
            iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/object-dynamics#hasMemoryPersistence",
            functional=True,
        ).as_dict(),
    )
    """
    Whether recruitment ends when the object is released.
    """

    stores_representation: List[ContextualRepresentation] = field(
        default_factory=list,
        metadata=PropertyTerm(
            iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/object-dynamics#storesRepresentation",
            functional=False,
            range="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/object-dynamics#ContextualRepresentation",
        ).as_dict(),
    )
    """
    A representation the memory retains.
    """


# %% property descriptors


@dataclass
class AffectsModel(PropertyDescriptor):
    """
    The object or limb dynamics subjected to the perturbation.
    """


@dataclass
class CommandsSetpoint(PropertyDescriptor):
    """
    A motor setpoint the command specifies.
    """


@dataclass
class CopiesCommand(PropertyDescriptor):
    """
    The outgoing command the copy represents.
    """


@dataclass
class DefinesDesiredState(core.DefinesComponent):
    """
    A desired state supplied to inverse models.
    """


@dataclass
class DefinesDynamicsPerturbation(core.DefinesComponent):
    """
    A load or rotation context used to assess transfer.
    """


@dataclass
class DefinesDynamicsTransfer(core.DefinesComponent):
    """
    A transfer result declared by the structure.
    """


@dataclass
class DefinesEfferenceCopy(core.DefinesComponent):
    """
    A command copy available to dynamics models.
    """


@dataclass
class DefinesInternalModelComposition(core.DefinesComponent):
    """
    A command-to-prediction pathway declared by the structure.
    """


@dataclass
class DefinesMotorCommand(core.DefinesComponent):
    """
    An outgoing motor command declared by the structure.
    """


@dataclass
class DefinesRepresentationDissociation(core.DefinesComponent):
    """
    A sensorimotor-perceptual dissociation declared by the structure.
    """


@dataclass
class DefinesSensorimotorMemory(core.DefinesComponent):
    """
    A memory retaining sensorimotor representations.
    """


@dataclass
class DesiredCondition(PropertyDescriptor):
    """
    A condition whose establishment defines the desired state.
    """


@dataclass
class FromPerturbation(PropertyDescriptor):
    """
    The context in which adaptation was acquired.
    """


@dataclass
class HasPerceptualRepresentation(PropertyDescriptor):
    """
    The representation used for conscious judgment.
    """


@dataclass
class HasSensorimotorRepresentation(PropertyDescriptor):
    """
    The representation used for motor prediction and scaling.
    """


@dataclass
class MatchedWith(PropertyDescriptor, HasInverseProperty):
    """
    A perturbation matched physically with this one for a transfer comparison.
    """

    @classmethod
    def get_inverse(cls) -> type[PropertyDescriptor]:
        """
        The property is symmetric, so it is its own inverse.
        """
        return cls


@dataclass
class PerceptualEstimate(PropertyDescriptor):
    """
    The biased estimate held by the perceptual representation.
    """


@dataclass
class PersistsIn(PropertyDescriptor):
    """
    The memory in which the revised representation is retained.
    """


@dataclass
class PredictsMechanicalConsequence(PropertyDescriptor):
    """
    A mechanical event the composition predicts.
    """


@dataclass
class PredictsSensoryConsequence(PropertyDescriptor):
    """
    A sensory consequence the forward model generates.
    """


@dataclass
class PursuesState(PropertyDescriptor):
    """
    The desired state the command is intended to establish.
    """


@dataclass
class RepresentsRole(PropertyDescriptor):
    """
    A participant role whose properties or dynamics are represented.
    """


@dataclass
class SensorimotorEstimate(PropertyDescriptor):
    """
    The accurate estimate held by the sensorimotor representation.
    """


@dataclass
class StoresRepresentation(PropertyDescriptor):
    """
    A representation the memory retains.
    """


@dataclass
class ToPerturbation(PropertyDescriptor):
    """
    The context in which transfer was assessed.
    """


@dataclass
class UsesDynamicsModel(PropertyDescriptor):
    """
    The object or limb dynamics used to interpret the command copy.
    """


@dataclass
class UsesEfferenceCopy(PropertyDescriptor):
    """
    An outgoing command copy used to predict the load the command will meet.
    """


@dataclass
class UsesForwardModel(PropertyDescriptor):
    """
    The model that generates the predicted consequences.
    """


# %% descriptor attachments

ObjectDynamicsPhaseStructure.defines_internal_model = core.DefinesInternalModel(
    ObjectDynamicsPhaseStructure, "defines_internal_model"
)
DynamicsPerturbation.affects_model = AffectsModel(DynamicsPerturbation, "affects_model")
MotorCommand.commands_setpoint = CommandsSetpoint(MotorCommand, "commands_setpoint")
EfferenceCopy.copies_command = CopiesCommand(EfferenceCopy, "copies_command")
ObjectDynamicsPhaseStructure.defines_desired_state = DefinesDesiredState(
    ObjectDynamicsPhaseStructure, "defines_desired_state"
)
ObjectDynamicsPhaseStructure.defines_dynamics_perturbation = (
    DefinesDynamicsPerturbation(
        ObjectDynamicsPhaseStructure, "defines_dynamics_perturbation"
    )
)
ObjectDynamicsPhaseStructure.defines_dynamics_transfer = DefinesDynamicsTransfer(
    ObjectDynamicsPhaseStructure, "defines_dynamics_transfer"
)
ObjectDynamicsPhaseStructure.defines_efference_copy = DefinesEfferenceCopy(
    ObjectDynamicsPhaseStructure, "defines_efference_copy"
)
ObjectDynamicsPhaseStructure.defines_internal_model_composition = (
    DefinesInternalModelComposition(
        ObjectDynamicsPhaseStructure, "defines_internal_model_composition"
    )
)
ObjectDynamicsPhaseStructure.defines_motor_command = DefinesMotorCommand(
    ObjectDynamicsPhaseStructure, "defines_motor_command"
)
ObjectDynamicsPhaseStructure.defines_representation_dissociation = (
    DefinesRepresentationDissociation(
        ObjectDynamicsPhaseStructure, "defines_representation_dissociation"
    )
)
ObjectDynamicsPhaseStructure.defines_sensorimotor_memory = DefinesSensorimotorMemory(
    ObjectDynamicsPhaseStructure, "defines_sensorimotor_memory"
)
DesiredState.desired_condition = DesiredCondition(DesiredState, "desired_condition")
DynamicsTransfer.from_perturbation = FromPerturbation(
    DynamicsTransfer, "from_perturbation"
)
RepresentationDissociation.has_perceptual_representation = HasPerceptualRepresentation(
    RepresentationDissociation, "has_perceptual_representation"
)
RepresentationDissociation.has_sensorimotor_representation = (
    HasSensorimotorRepresentation(
        RepresentationDissociation, "has_sensorimotor_representation"
    )
)
DynamicsPerturbation.matched_with = MatchedWith(DynamicsPerturbation, "matched_with")
RepresentationDissociation.perceptual_estimate = PerceptualEstimate(
    RepresentationDissociation, "perceptual_estimate"
)
RetainedModelUpdate.persists_in = PersistsIn(RetainedModelUpdate, "persists_in")
InternalModelComposition.predicts_mechanical_consequence = (
    PredictsMechanicalConsequence(
        InternalModelComposition, "predicts_mechanical_consequence"
    )
)
InternalModelComposition.predicts_sensory_consequence = PredictsSensoryConsequence(
    InternalModelComposition, "predicts_sensory_consequence"
)
MotorCommand.pursues_state = PursuesState(MotorCommand, "pursues_state")
ContextualRepresentation.represents_role = RepresentsRole(
    ContextualRepresentation, "represents_role"
)
RepresentationDissociation.sensorimotor_estimate = SensorimotorEstimate(
    RepresentationDissociation, "sensorimotor_estimate"
)
SensorimotorMemory.stores_representation = StoresRepresentation(
    SensorimotorMemory, "stores_representation"
)
DynamicsTransfer.to_perturbation = ToPerturbation(DynamicsTransfer, "to_perturbation")
InternalModelComposition.uses_dynamics_model = UsesDynamicsModel(
    InternalModelComposition, "uses_dynamics_model"
)
DynamicsModel.uses_efference_copy = UsesEfferenceCopy(
    DynamicsModel, "uses_efference_copy"
)
InternalModelComposition.uses_efference_copy = UsesEfferenceCopy(
    InternalModelComposition, "uses_efference_copy"
)
InternalModelComposition.uses_forward_model = UsesForwardModel(
    InternalModelComposition, "uses_forward_model"
)
MotorCommand.generated_by = core.GeneratedBy(MotorCommand, "generated_by")
