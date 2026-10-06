"""
Internal models and represented dynamics for predictive manipulation.

This module owns action-neutral representations of object and limb dynamics. It depends
on the phase and prediction primitives in :mod:`flanagan.core` and contains no grasping
or substance-transfer vocabulary.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import ClassVar, Optional

from llm_reasoner.knowledge.microworlds.flanagan.core import (
    ActionPhaseStructure,
    DynamicsEstimate,
    Estimate,
    EstimateRepresentation,
    EstimateUpdateDirection,
    ForwardModel,
    InternalModel,
    InverseModel,
    MechanicalEventType,
    MotorSetpoint,
    ObjectRepresentation,
    PhaseConditionType,
    PhysicalDimension,
    RoleSlot,
    SensoryEventPrediction,
)
from llm_reasoner.knowledge.microworlds.flanagan.namespaces import FlanaganNamespace
from llm_reasoner.knowledge.ontological.decorators import (
    KnowledgeEntity,
    ontological,
    prop,
    value_partition,
)

OBJECT_DYNAMICS = FlanaganNamespace.OBJECT_DYNAMICS
"""
Namespace for internal-model and dynamics concepts.
"""

DUL = "http://www.ontologydesignpatterns.org/ont/dul/DUL.owl#"
"""
DOLCE UltraLite namespace used for external alignments.
"""

# %% value partitions


@value_partition(OBJECT_DYNAMICS + "CoordinateFrame")
class CoordinateFrame(StrEnum):
    """
    The reference frame in which represented dynamics are encoded.
    """

    OBJECT_CENTERED = "object_centered"
    """
    Coordinates fixed to the manipulated object.
    """

    ARM_CENTERED = "arm_centered"
    """
    Intrinsic or extrinsic coordinates associated with the arm.
    """

    HAND_CENTERED = "hand_centered"
    """
    Coordinates fixed to the hand or end effector.
    """

    GAZE_CENTERED = "gaze_centered"
    """
    Coordinates organized relative to current gaze.
    """


@value_partition(OBJECT_DYNAMICS + "RepresentationPurpose")
class RepresentationPurpose(StrEnum):
    """
    The function for which represented content is recruited.
    """

    SENSORIMOTOR_CONTROL = "sensorimotor_control"
    """
    Content used to predict and scale motor commands.
    """

    PERCEPTUAL_JUDGMENT = "perceptual_judgment"
    """
    Content used for conscious property judgments.
    """


@value_partition(OBJECT_DYNAMICS + "MemoryPersistence")
class MemoryPersistence(StrEnum):
    """
    How long a sensorimotor memory remains recruited.
    """

    WHILE_OBJECT_COUPLED = "while_object_coupled"
    """
    The memory is recruited only while its object remains coupled.
    """

    AFTER_RELEASE = "after_release"
    """
    The memory remains available after the object is released.
    """


@value_partition(OBJECT_DYNAMICS + "PerturbationKind")
class PerturbationKind(StrEnum):
    """
    The manipulation applied when dynamics transfer is assessed.
    """

    LOAD = "load"
    """
    An external force or load is applied.
    """

    ROTATION = "rotation"
    """
    The object, hand, or arm orientation is changed.
    """


@value_partition(OBJECT_DYNAMICS + "TransferOutcome")
class TransferOutcome(StrEnum):
    """
    How adaptation changes across two dynamics contexts.
    """

    GENERALIZES = "generalizes"
    """
    Adaptation transfers between the compared contexts.
    """

    DIFFERS = "differs"
    """
    Transfer differs between the compared contexts.
    """
    DOES_NOT_GENERALIZE = "does_not_generalize"
    """
    Adaptation does not transfer to the target context.
    """


@value_partition(OBJECT_DYNAMICS + "RepresentationAccuracy")
class RepresentationAccuracy(StrEnum):
    """
    Whether represented content agrees with task-relevant evidence.
    """

    ACCURATE = "accurate"
    """
    The represented estimate agrees with task-relevant evidence.
    """
    BIASED = "biased"
    """
    The represented estimate remains systematically biased.
    """


# %% command pathway


@ontological(iri=OBJECT_DYNAMICS + "DesiredState", aligns_to=DUL + "Situation")
@dataclass(eq=False)
class DesiredState(KnowledgeEntity):
    """
    A state that an inverse model seeks to establish.
    """

    desired_condition: list[PhaseConditionType] = prop(
        min_count=1,
        default_factory=list,
        iri=OBJECT_DYNAMICS + "desiredCondition",
    )
    """
    The conditions whose establishment defines the desired state.
    """


@ontological(iri=OBJECT_DYNAMICS + "MotorCommand", aligns_to=DUL + "Description")
@dataclass(eq=False)
class MotorCommand(KnowledgeEntity):
    """
    A command selected by an inverse model for a desired state.
    """

    pursues_state: Optional[DesiredState] = prop(
        functional=True,
        min_count=1,
        iri=OBJECT_DYNAMICS + "pursuesState",
    )
    """
    The desired state the command is intended to establish.
    """

    generated_by: Optional[InverseModel] = prop(
        functional=True,
        min_count=1,
        iri=OBJECT_DYNAMICS + "generatedBy",
    )
    """
    The inverse model that selected the command.
    """

    commands_setpoint: list[MotorSetpoint] = prop(
        min_count=1,
        default_factory=list,
        iri=OBJECT_DYNAMICS + "commandsSetpoint",
    )
    """
    The motor setpoints specified by the command.
    """


@ontological(iri=OBJECT_DYNAMICS + "EfferenceCopy", aligns_to=DUL + "InformationObject")
@dataclass(eq=False)
class EfferenceCopy(KnowledgeEntity):
    """
    An internal copy of an outgoing motor command.
    """

    copies_command: Optional[MotorCommand] = prop(
        functional=True,
        min_count=1,
        iri=OBJECT_DYNAMICS + "copiesCommand",
    )
    """
    The outgoing command represented by the copy.
    """


# %% represented dynamics


@ontological(
    iri=OBJECT_DYNAMICS + "Representation",
    aligns_to=DUL + "Description",
    abstract=True,
)
@dataclass(eq=False)
class Representation(EstimateRepresentation):
    """
    Represented estimates with an explicit frame and functional purpose.
    """

    holds_estimate: list[Estimate] = prop(
        min_count=1,
        default_factory=list,
        iri=OBJECT_DYNAMICS + "holdsEstimate",
    )
    """
    The estimates stored in the representation.
    """

    coordinate_frame: Optional[CoordinateFrame] = prop(
        functional=True,
        min_count=1,
        iri=OBJECT_DYNAMICS + "hasCoordinateFrame",
    )
    """
    The frame in which the represented content is encoded.
    """

    purpose: Optional[RepresentationPurpose] = prop(
        functional=True,
        min_count=1,
        iri=OBJECT_DYNAMICS + "hasRepresentationPurpose",
    )
    """
    The function for which the represented content is recruited.
    """

    represents_role: list[RoleSlot] = prop(
        min_count=1,
        default_factory=list,
        iri=OBJECT_DYNAMICS + "representsRole",
    )
    """
    The participant roles whose properties or dynamics are represented.
    """


@ontological(
    iri=OBJECT_DYNAMICS + "DynamicsModel",
    abstract=True,
    disjoint_siblings=True,
)
@dataclass(eq=False)
class DynamicsModel(Representation):
    """
    A represented mapping from command-related input to mechanical change.
    """


@ontological(iri=OBJECT_DYNAMICS + "ObjectDynamicsModel")
@dataclass(eq=False)
class ObjectDynamicsModel(DynamicsModel, ObjectRepresentation):
    """
    A dynamics model recruited for a manipulated object.
    """

    uses_efference_copy: list[EfferenceCopy] = prop(
        min_count=1,
        default_factory=list,
        iri=OBJECT_DYNAMICS + "usesEfferenceCopy",
    )
    """
    The outgoing command copies used to predict object-related loads.
    """


@ontological(iri=OBJECT_DYNAMICS + "LimbDynamicsModel")
@dataclass(eq=False)
class LimbDynamicsModel(DynamicsModel):
    """
    A dynamics model representing the actor's limb or effector.
    """


@ontological(iri=OBJECT_DYNAMICS + "PerceptualRepresentation")
@dataclass(eq=False)
class PerceptualRepresentation(Representation):
    """
    Represented content used for perceptual property judgments.
    """


@ontological(iri=OBJECT_DYNAMICS + "LoadForceEstimate")
@dataclass(eq=False)
class LoadForceEstimate(DynamicsEstimate):
    """
    An estimate of force arising from object or limb dynamics.
    """

    required_dimension: ClassVar[PhysicalDimension] = PhysicalDimension.FORCE
    """
    Load-force estimates use the force dimension.
    """


# %% memory and updating


@ontological(iri=OBJECT_DYNAMICS + "SensorimotorMemory", aligns_to=DUL + "Description")
@dataclass(eq=False)
class SensorimotorMemory(KnowledgeEntity):
    """
    Memory that recruits one or more sensorimotor representations.
    """

    stores_representation: list[Representation] = prop(
        min_count=1,
        default_factory=list,
        iri=OBJECT_DYNAMICS + "storesRepresentation",
    )
    """
    The representations retained by the memory.
    """

    persistence: Optional[MemoryPersistence] = prop(
        functional=True,
        min_count=1,
        iri=OBJECT_DYNAMICS + "hasMemoryPersistence",
    )
    """
    Whether recruitment ends with object release.
    """


@ontological(iri=OBJECT_DYNAMICS + "RepresentationUpdate", aligns_to=DUL + "Situation")
@dataclass(eq=False)
class RepresentationUpdate(KnowledgeEntity):
    """
    A revision of estimates held by one representation.
    """

    updates_representation: Optional[Representation] = prop(
        functional=True,
        min_count=1,
        iri=OBJECT_DYNAMICS + "updatesRepresentation",
    )
    """
    The representation changed by the update.
    """

    updates_estimate: list[Estimate] = prop(
        min_count=1,
        default_factory=list,
        iri=OBJECT_DYNAMICS + "updatesEstimate",
    )
    """
    The represented estimates changed by the update.
    """

    direction: Optional[EstimateUpdateDirection] = prop(
        functional=True,
        min_count=1,
        iri=OBJECT_DYNAMICS + "hasUpdateDirection",
    )
    """
    How the represented estimates change.
    """

    persists_in: Optional[SensorimotorMemory] = prop(
        functional=True,
        min_count=1,
        iri=OBJECT_DYNAMICS + "persistsIn",
    )
    """
    The memory in which the revised representation is retained.
    """


# %% predictive composition


@ontological(
    iri=OBJECT_DYNAMICS + "InternalModelComposition",
    aligns_to=DUL + "Description",
)
@dataclass(eq=False)
class InternalModelComposition(KnowledgeEntity):
    """
    The command-to-prediction pathway for one controlled state.
    """

    desired_state: Optional[DesiredState] = prop(
        functional=True,
        min_count=1,
        iri=OBJECT_DYNAMICS + "hasDesiredState",
    )
    """
    The state requested of the inverse model.
    """

    inverse_model: Optional[InverseModel] = prop(
        functional=True,
        min_count=1,
        iri=OBJECT_DYNAMICS + "usesInverseModel",
    )
    """
    The model that selects the motor command.
    """

    motor_command: Optional[MotorCommand] = prop(
        functional=True,
        min_count=1,
        iri=OBJECT_DYNAMICS + "hasMotorCommand",
    )
    """
    The command selected for the desired state.
    """

    efference_copy: Optional[EfferenceCopy] = prop(
        functional=True,
        min_count=1,
        iri=OBJECT_DYNAMICS + "hasEfferenceCopy",
    )
    """
    The internal copy of the outgoing motor command.
    """

    dynamics_model: Optional[DynamicsModel] = prop(
        functional=True,
        min_count=1,
        iri=OBJECT_DYNAMICS + "usesDynamicsModel",
    )
    """
    The object or limb dynamics used to interpret the command copy.
    """

    forward_model: Optional[ForwardModel] = prop(
        functional=True,
        min_count=1,
        iri=OBJECT_DYNAMICS + "usesForwardModel",
    )
    """
    The model that generates predicted consequences.
    """

    predicted_mechanical_consequence: list[MechanicalEventType] = prop(
        min_count=1,
        default_factory=list,
        iri=OBJECT_DYNAMICS + "predictsMechanicalConsequence",
    )
    """
    The mechanical consequences predicted by the composition.
    """

    predicted_sensory_consequence: list[SensoryEventPrediction] = prop(
        min_count=1,
        default_factory=list,
        iri=OBJECT_DYNAMICS + "predictsSensoryConsequence",
    )
    """
    The sensory consequences generated by the forward model.
    """


# %% transfer contexts


@ontological(iri=OBJECT_DYNAMICS + "DynamicsPerturbation", aligns_to=DUL + "Situation")
@dataclass(eq=False)
class DynamicsPerturbation(KnowledgeEntity):
    """
    A load or rotation applied to one represented dynamics system.
    """

    kind: Optional[PerturbationKind] = prop(
        functional=True,
        min_count=1,
        iri=OBJECT_DYNAMICS + "hasPerturbationKind",
    )
    """
    The physical transformation used to assess dynamics transfer.
    """

    affects_model: Optional[DynamicsModel] = prop(
        functional=True,
        min_count=1,
        iri=OBJECT_DYNAMICS + "affectsModel",
    )
    """
    The object or limb dynamics subjected to the perturbation.
    """

    coordinate_frame: Optional[CoordinateFrame] = prop(
        functional=True,
        min_count=1,
        iri=OBJECT_DYNAMICS + "hasCoordinateFrame",
    )
    """
    The frame in which the perturbation is expressed.
    """

    equivalent_to: list[DynamicsPerturbation] = prop(
        default_factory=list,
        iri=OBJECT_DYNAMICS + "equivalentTo",
    )
    """
    Perturbations matched physically for a transfer comparison.
    """


@ontological(iri=OBJECT_DYNAMICS + "DynamicsTransfer", aligns_to=DUL + "Situation")
@dataclass(eq=False)
class DynamicsTransfer(KnowledgeEntity):
    """
    The adaptation result between two perturbation contexts.
    """

    from_perturbation: Optional[DynamicsPerturbation] = prop(
        functional=True,
        min_count=1,
        iri=OBJECT_DYNAMICS + "fromPerturbation",
    )
    """
    The context in which adaptation was acquired.
    """

    to_perturbation: Optional[DynamicsPerturbation] = prop(
        functional=True,
        min_count=1,
        iri=OBJECT_DYNAMICS + "toPerturbation",
    )
    """
    The context in which transfer was assessed.
    """

    outcome: Optional[TransferOutcome] = prop(
        functional=True,
        min_count=1,
        iri=OBJECT_DYNAMICS + "hasTransferOutcome",
    )
    """
    Whether adaptation generalized across the contexts.
    """


# %% representation dissociation


@ontological(
    iri=OBJECT_DYNAMICS + "RepresentationDissociation",
    aligns_to=DUL + "Situation",
)
@dataclass(eq=False)
class RepresentationDissociation(KnowledgeEntity):
    """
    A sensorimotor estimate and perceptual judgment that diverge.
    """

    sensorimotor_representation: Optional[Representation] = prop(
        functional=True,
        min_count=1,
        iri=OBJECT_DYNAMICS + "hasSensorimotorRepresentation",
    )
    """
    The representation used for motor prediction and scaling.
    """

    perceptual_representation: Optional[PerceptualRepresentation] = prop(
        functional=True,
        min_count=1,
        iri=OBJECT_DYNAMICS + "hasPerceptualRepresentation",
    )
    """
    The representation used for conscious judgment.
    """

    sensorimotor_estimate: Optional[Estimate] = prop(
        functional=True,
        min_count=1,
        iri=OBJECT_DYNAMICS + "hasSensorimotorEstimate",
    )
    """
    The estimate used by the sensorimotor representation.
    """

    perceptual_estimate: Optional[Estimate] = prop(
        functional=True,
        min_count=1,
        iri=OBJECT_DYNAMICS + "hasPerceptualEstimate",
    )
    """
    The estimate used by the perceptual representation.
    """

    sensorimotor_accuracy: Optional[RepresentationAccuracy] = prop(
        functional=True,
        min_count=1,
        iri=OBJECT_DYNAMICS + "hasSensorimotorAccuracy",
    )
    """
    Whether the sensorimotor estimate agrees with task evidence.
    """

    perceptual_accuracy: Optional[RepresentationAccuracy] = prop(
        functional=True,
        min_count=1,
        iri=OBJECT_DYNAMICS + "hasPerceptualAccuracy",
    )
    """
    Whether the perceptual estimate agrees with task evidence.
    """


# %% structure


def representation_ownership_contract_description() -> str:
    """Return the ownership rules for represented estimates."""
    return (
        "Every Estimate in structure.defines_parameter must occur in "
        "holds_estimate of at least one Representation in "
        "structure.defines_internal_model. Its represents_role values must include "
        "every role in the estimate's parameter_on_role. ObjectDynamicsModel and "
        "LimbDynamicsModel use the SENSORIMOTOR_CONTROL purpose; "
        "PerceptualRepresentation uses PERCEPTUAL_JUDGMENT. Do not reuse one "
        "estimate across representations with different purpose values."
    )


@ontological(iri=OBJECT_DYNAMICS + "ObjectDynamicsPhaseStructure")
@dataclass(eq=False)
class ObjectDynamicsPhaseStructure(ActionPhaseStructure):
    """
    A phase structure with explicit internal models and dynamics evidence.
    """

    defines_internal_model: list[InternalModel] = prop(
        min_count=1,
        default_factory=list,
        iri=OBJECT_DYNAMICS + "definesInternalModel",
        comment=representation_ownership_contract_description(),
    )
    """
    The internal models and represented content declared by the structure.
    """

    defines_desired_state: list[DesiredState] = prop(
        default_factory=list,
        iri=OBJECT_DYNAMICS + "definesDesiredState",
    )
    """
    The desired states supplied to inverse models.
    """

    defines_motor_command: list[MotorCommand] = prop(
        default_factory=list,
        iri=OBJECT_DYNAMICS + "definesMotorCommand",
    )
    """
    The outgoing motor commands declared by the structure.
    """

    defines_efference_copy: list[EfferenceCopy] = prop(
        default_factory=list,
        iri=OBJECT_DYNAMICS + "definesEfferenceCopy",
    )
    """
    The command copies available to dynamics models.
    """

    has_internal_model_composition: list[InternalModelComposition] = prop(
        default_factory=list,
        iri=OBJECT_DYNAMICS + "hasInternalModelComposition",
    )
    """
    The command-to-prediction pathways declared by the structure.
    """

    has_sensorimotor_memory: list[SensorimotorMemory] = prop(
        default_factory=list,
        iri=OBJECT_DYNAMICS + "hasSensorimotorMemory",
    )
    """
    The memories retaining sensorimotor representations.
    """

    has_representation_update: list[RepresentationUpdate] = prop(
        default_factory=list,
        iri=OBJECT_DYNAMICS + "hasRepresentationUpdate",
    )
    """
    The representation updates declared by the structure.
    """
    defines_dynamics_perturbation: list[DynamicsPerturbation] = prop(
        default_factory=list,
        iri=OBJECT_DYNAMICS + "definesDynamicsPerturbation",
    )
    """
    The load and rotation contexts used to assess transfer.
    """
    has_dynamics_transfer: list[DynamicsTransfer] = prop(
        default_factory=list,
        iri=OBJECT_DYNAMICS + "hasDynamicsTransfer",
    )
    """
    The transfer results declared by the structure.
    """

    has_representation_dissociation: list[RepresentationDissociation] = prop(
        default_factory=list,
        iri=OBJECT_DYNAMICS + "hasRepresentationDissociation",
    )
    """
    The sensorimotor-perceptual dissociations declared by the structure.
    """
