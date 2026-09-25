"""Prehensile manipulation vocabulary for predictive object control.

This module owns grasping, contact, lifting, replacement, release, and grip-force
control. It depends on action-neutral object dynamics and contains no
substance-transfer vocabulary.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar, Optional

from llm_reasoner.knowledge.microworlds.flanagan.core import (
    ActionPhase,
    ContactEventType,
    ContactPropertyEstimate,
    CorrectionDirection,
    CorrectiveAction,
    ForceDynamicEventType,
    ForceRateSetpoint,
    Manipulandum,
    ManipulationRole,
    MechanicalThreshold,
    MotorSetpoint,
    ObjectPropertyEstimate,
    ObjectStateEstimate,
    ParameterValueForm,
    PhaseConditionType,
    PhysicalDimension,
    RoleSlot,
    SafetyMargin,
    SensoryEventPrediction,
    SetpointAttained,
    TemporalWindow,
)
from llm_reasoner.knowledge.microworlds.flanagan.object_dynamics import (
    LoadForceEstimate,
    ObjectDynamicsPhaseStructure,
)
from llm_reasoner.knowledge.ontological.decorators import (
    KnowledgeEntity,
    ontological,
    prop,
)
from llm_reasoner.knowledge.ontological.rigidity import ConceptLevel, Rigidity

PREHENSION = "http://example.org/ontology/flanagan-prehension#"
"""Namespace for prehensile-manipulation concepts."""

SOMA = "http://www.ease-crc.org/ont/SOMA.owl#"
"""SOMA namespace used for external alignments."""

DUL = "http://www.ontologydesignpatterns.org/ont/dul/DUL.owl#"
"""DOLCE UltraLite namespace used for external alignments."""


# %% roles


@ontological(
    iri=PREHENSION + "GraspingEffector",
    rigidity=Rigidity.ANTI_RIGID,
    level=ConceptLevel.OCCURRENCE,
    role_taker="Symbol",
    aligns_to=SOMA + "PrehensileEffector",
    disjoint_with=["PrehensileManipulandum"],
)
@dataclass(eq=False)
class GraspingEffector(ManipulationRole):
    """The hand, gripper, or other effector establishing a grasp."""


@ontological(
    iri=PREHENSION + "GraspSite",
    rigidity=Rigidity.ANTI_RIGID,
    level=ConceptLevel.OCCURRENCE,
    role_taker="Symbol",
)
@dataclass(eq=False)
class GraspSite(ManipulationRole):
    """The manipulandum region at which the effector makes contact."""


@ontological(
    iri=PREHENSION + "PrehensileManipulandum",
    rigidity=Rigidity.ANTI_RIGID,
    level=ConceptLevel.OCCURRENCE,
    role_taker="Symbol",
)
@dataclass(eq=False)
class PrehensileManipulandum(Manipulandum):
    """The object coupled to an effector by a prehensile grasp."""


@ontological(
    iri=PREHENSION + "InitialSupport",
    rigidity=Rigidity.ANTI_RIGID,
    level=ConceptLevel.OCCURRENCE,
    role_taker="Symbol",
    aligns_to=SOMA + "Supporter",
    disjoint_with=["PrehensileManipulandum"],
)
@dataclass(eq=False)
class InitialSupport(ManipulationRole):
    """The surface bearing the manipulandum before liftoff."""


@ontological(
    iri=PREHENSION + "FinalSupport",
    rigidity=Rigidity.ANTI_RIGID,
    level=ConceptLevel.OCCURRENCE,
    role_taker="Symbol",
    aligns_to=SOMA + "Supporter",
    disjoint_with=["PrehensileManipulandum"],
)
@dataclass(eq=False)
class FinalSupport(ManipulationRole):
    """The surface bearing the manipulandum after touchdown."""


# %% events


@ontological(iri=PREHENSION + "ReachTargetReached")
@dataclass(eq=False)
class ReachTargetReached(SetpointAttained):
    """The effector reaches the commanded pre-contact pose."""


@ontological(iri=PREHENSION + "ContactEstablishment")
@dataclass(eq=False)
class ContactEstablishment(ContactEventType):
    """The effector first contacts the manipulandum."""


@ontological(iri=PREHENSION + "ContactBreak")
@dataclass(eq=False)
class ContactBreak(ContactEventType):
    """The effector loses contact with the manipulandum."""


@ontological(iri=PREHENSION + "Liftoff")
@dataclass(eq=False)
class Liftoff(ContactEventType, ForceDynamicEventType):
    """Load exceeds weight and the manipulandum separates from its support."""


@ontological(iri=PREHENSION + "Touchdown")
@dataclass(eq=False)
class Touchdown(ContactEventType, ForceDynamicEventType):
    """The manipulandum contacts its destination support."""


@ontological(iri=PREHENSION + "SlipEvent")
@dataclass(eq=False)
class SlipEvent(ForceDynamicEventType):
    """Tangential load exceeds the frictional capacity of the grasp."""


@ontological(iri=PREHENSION + "LiftClearanceReached")
@dataclass(eq=False)
class LiftClearanceReached(SetpointAttained):
    """The manipulandum reaches its commanded support clearance."""


@ontological(iri=PREHENSION + "StableHoldReached")
@dataclass(eq=False)
class StableHoldReached(SetpointAttained):
    """The requested stable holding state has been reached."""


@ontological(iri=PREHENSION + "TransportTargetReached")
@dataclass(eq=False)
class TransportTargetReached(SetpointAttained):
    """The held manipulandum reaches its commanded transport destination."""


@ontological(iri=PREHENSION + "ForcesUnloaded")
@dataclass(eq=False)
class ForcesUnloaded(SetpointAttained):
    """Support transfer completes and grip and load forces reach their low targets."""

    weight_supported_by: list[RoleSlot] = prop(
        min_count=1,
        default_factory=list,
        iri=PREHENSION + "weightSupportedBy",
    )
    """The support roles bearing manipulandum weight at force completion."""


@ontological(iri=PREHENSION + "WithdrawalComplete")
@dataclass(eq=False)
class WithdrawalComplete(SetpointAttained):
    """The effector reaches its post-release withdrawal target."""


# %% conditions


@ontological(iri=PREHENSION + "EffectorObjectContact", aligns_to=SOMA + "ContactState")
@dataclass(eq=False)
class EffectorObjectContact(PhaseConditionType):
    """The effector is mechanically in contact with the manipulandum."""


@ontological(iri=PREHENSION + "ObjectSupportContact", aligns_to=SOMA + "ContactState")
@dataclass(eq=False)
class ObjectSupportContact(PhaseConditionType):
    """The manipulandum is mechanically in contact with a support."""


@ontological(iri=PREHENSION + "GraspStability")
@dataclass(eq=False)
class GraspStability(PhaseConditionType):
    """Grip force remains above the slip threshold by its safety margin."""


@ontological(iri=PREHENSION + "ObjectClearOfSupport")
@dataclass(eq=False)
class ObjectClearOfSupport(PhaseConditionType):
    """No support bears the manipulandum's weight."""


@ontological(iri=PREHENSION + "EffectorPositioned")
@dataclass(eq=False)
class EffectorPositioned(PhaseConditionType):
    """The effector is positioned for contact establishment."""


@ontological(iri=PREHENSION + "SafeContactForce")
@dataclass(eq=False)
class SafeContactForce(PhaseConditionType):
    """Contact force remains within the protective response limit."""


# %% phases


@ontological(iri=PREHENSION + "ReachPhase", display_name="Reach")
@dataclass(eq=False)
class ReachPhase(ActionPhase):
    """Transport of the uncoupled effector toward a pre-contact pose."""


@ontological(
    iri=PREHENSION + "ContactEstablishmentPhase",
    display_name="ContactEstablishment",
)
@dataclass(eq=False)
class ContactEstablishmentPhase(ActionPhase):
    """Controlled establishment of effector-manipulandum contact."""


@ontological(iri=PREHENSION + "ForceLoadingPhase", display_name="ForceLoading")
@dataclass(eq=False)
class ForceLoadingPhase(ActionPhase):
    """Coordinated grip and load increase from contact until liftoff."""


@ontological(iri=PREHENSION + "LiftPhase", display_name="Lift")
@dataclass(eq=False)
class LiftPhase(ActionPhase):
    """Vertical transport away from the initial support."""


@ontological(iri=PREHENSION + "HoldPhase", display_name="Hold")
@dataclass(eq=False)
class HoldPhase(ActionPhase):
    """Optional maintenance of a stable grasp before transport."""


@ontological(iri=PREHENSION + "TransportPhase", display_name="Transport")
@dataclass(eq=False)
class TransportPhase(ActionPhase):
    """Translation of the coupled effector-manipulandum unit."""


@ontological(iri=PREHENSION + "ReplacePhase", display_name="Replace")
@dataclass(eq=False)
class ReplacePhase(ActionPhase):
    """Lowering of the still-coupled manipulandum until touchdown."""


@ontological(iri=PREHENSION + "UnloadPhase", display_name="Unload")
@dataclass(eq=False)
class UnloadPhase(ActionPhase):
    """Transfer of weight to support while grip and load forces decrease."""


@ontological(iri=PREHENSION + "ContactReleasePhase", display_name="ContactRelease")
@dataclass(eq=False)
class ContactReleasePhase(ActionPhase):
    """Release of effector-manipulandum contact after force unloading."""


@ontological(iri=PREHENSION + "WithdrawPhase", display_name="Withdraw")
@dataclass(eq=False)
class WithdrawPhase(ActionPhase):
    """Optional effector withdrawal following contact release."""


# %% setpoints and estimates


@ontological(iri=PREHENSION + "ReachPoseSetpoint")
@dataclass(eq=False)
class ReachPoseSetpoint(MotorSetpoint):
    """A categorical target for the pre-contact effector pose."""

    required_value_form: ClassVar[ParameterValueForm] = ParameterValueForm.CATEGORICAL
    """A reach pose requires a categorical value."""


@ontological(iri=PREHENSION + "GripForceSetpoint")
@dataclass(eq=False)
class GripForceSetpoint(MotorSetpoint):
    """Normal force target perpendicular to the grasp surface."""

    required_dimension: ClassVar[PhysicalDimension] = PhysicalDimension.FORCE
    """Grip force requires the force dimension."""


@ontological(iri=PREHENSION + "LoadForceSetpoint")
@dataclass(eq=False)
class LoadForceSetpoint(MotorSetpoint):
    """Tangential force target applied to the manipulandum."""

    required_dimension: ClassVar[PhysicalDimension] = PhysicalDimension.FORCE
    """Load force requires the force dimension."""


@ontological(iri=PREHENSION + "LoadForceRateSetpoint")
@dataclass(eq=False)
class LoadForceRateSetpoint(ForceRateSetpoint):
    """The commanded rate of tangential load-force change."""


@ontological(iri=PREHENSION + "LiftClearanceSetpoint")
@dataclass(eq=False)
class LiftClearanceSetpoint(MotorSetpoint):
    """The commanded clearance above the initial support."""

    required_dimension: ClassVar[PhysicalDimension] = PhysicalDimension.LENGTH
    """Lift clearance requires the length dimension."""


@ontological(iri=PREHENSION + "TransportPoseSetpoint")
@dataclass(eq=False)
class TransportPoseSetpoint(MotorSetpoint):
    """A categorical target for transported manipulandum pose."""

    required_value_form: ClassVar[ParameterValueForm] = ParameterValueForm.CATEGORICAL
    """A transport pose requires a categorical value."""


@ontological(iri=PREHENSION + "WithdrawalDistanceSetpoint")
@dataclass(eq=False)
class WithdrawalDistanceSetpoint(MotorSetpoint):
    """The commanded post-release effector withdrawal distance."""

    required_dimension: ClassVar[PhysicalDimension] = PhysicalDimension.LENGTH
    """Withdrawal distance requires the length dimension."""


@ontological(iri=PREHENSION + "SlipThreshold")
@dataclass(eq=False)
class SlipThreshold(MechanicalThreshold):
    """The minimum grip force supporting the predicted tangential load."""

    required_dimension: ClassVar[PhysicalDimension] = PhysicalDimension.FORCE
    """A slip threshold requires the force dimension."""


@ontological(iri=PREHENSION + "LiftoffThreshold")
@dataclass(eq=False)
class LiftoffThreshold(MechanicalThreshold):
    """The load-force threshold corresponding to manipulandum weight."""

    required_dimension: ClassVar[PhysicalDimension] = PhysicalDimension.FORCE
    """A liftoff threshold requires the force dimension."""


@ontological(iri=PREHENSION + "TouchdownThreshold")
@dataclass(eq=False)
class TouchdownThreshold(MechanicalThreshold):
    """The support-reaction threshold registering touchdown."""

    required_dimension: ClassVar[PhysicalDimension] = PhysicalDimension.FORCE
    """A touchdown threshold requires the force dimension."""


@ontological(iri=PREHENSION + "ExpectedMass")
@dataclass(eq=False)
class ExpectedMass(ObjectPropertyEstimate):
    """The anticipated manipulandum mass used to predict weight force."""

    required_dimension: ClassVar[PhysicalDimension] = PhysicalDimension.MASS
    """Expected mass requires the mass dimension."""


@ontological(iri=PREHENSION + "ExpectedFrictionCoefficient")
@dataclass(eq=False)
class ExpectedFrictionCoefficient(ContactPropertyEstimate):
    """The anticipated friction coefficient at the grasp interface."""

    required_dimension: ClassVar[PhysicalDimension] = PhysicalDimension.DIMENSIONLESS
    """A friction coefficient is dimensionless."""


@ontological(iri=PREHENSION + "ExpectedCenterOfMassPosition")
@dataclass(eq=False)
class ExpectedCenterOfMassPosition(ObjectStateEstimate):
    """The anticipated centre-of-mass position in the object frame."""

    required_dimension: ClassVar[PhysicalDimension] = PhysicalDimension.LENGTH
    """Centre-of-mass position requires the length dimension."""


@ontological(iri=PREHENSION + "ExpectedShape")
@dataclass(eq=False)
class ExpectedShape(ObjectPropertyEstimate):
    """The anticipated geometric category used for grasp planning."""

    required_value_form: ClassVar[ParameterValueForm] = ParameterValueForm.CATEGORICAL
    """Shape requires a categorical value."""


# %% grip control


@ontological(iri=PREHENSION + "CorrectiveGripResponse")
@dataclass(eq=False)
class CorrectiveGripResponse(CorrectiveAction):
    """A prediction-error response that changes a grip-force target."""


@ontological(iri=PREHENSION + "AnticipatoryGripAdjustment")
@dataclass(eq=False)
class AnticipatoryGripAdjustment(KnowledgeEntity):
    """A feedforward grip change scheduled from a predicted mechanical event."""

    triggered_by_prediction: Optional[SensoryEventPrediction] = prop(
        functional=True,
        min_count=1,
        iri=PREHENSION + "triggeredByPrediction",
    )
    """The predicted event consequence scheduling the adjustment."""
    targets_grip_force: Optional[GripForceSetpoint] = prop(
        functional=True,
        min_count=1,
        iri=PREHENSION + "targetsGripForce",
    )
    """The grip-force target changed before the predicted event."""
    direction: Optional[CorrectionDirection] = prop(
        functional=True,
        min_count=1,
        iri=PREHENSION + "hasAdjustmentDirection",
    )
    """The direction in which grip force changes."""
    adjustment_active_in: list[ActionPhase] = prop(
        min_count=1,
        default_factory=list,
        iri=PREHENSION + "adjustmentActiveIn",
    )
    """The phases during which the anticipatory change is active."""


@ontological(iri=PREHENSION + "GripLoadCoupling", aligns_to=DUL + "Relation")
@dataclass(eq=False)
class GripLoadCoupling(KnowledgeEntity):
    """A functional grip response derived from load, friction, margin, and timing."""

    predicted_load: Optional[LoadForceEstimate] = prop(
        functional=True,
        min_count=1,
        iri=PREHENSION + "predictedLoad",
    )
    """The represented tangential load for which grip force is scaled."""
    friction_estimate: Optional[ExpectedFrictionCoefficient] = prop(
        functional=True,
        min_count=1,
        iri=PREHENSION + "frictionEstimate",
    )
    """The estimated grasp-interface friction limiting tangential force."""
    safety_margin: Optional[SafetyMargin] = prop(
        functional=True,
        min_count=1,
        iri=PREHENSION + "safetyMargin",
    )
    """The additional grip force maintained above the slip threshold."""
    grip_force_response: Optional[GripForceSetpoint] = prop(
        functional=True,
        min_count=1,
        iri=PREHENSION + "gripForceResponse",
    )
    """The grip-force target produced by the functional relationship."""
    temporal_coupling: Optional[TemporalWindow] = prop(
        functional=True,
        min_count=1,
        iri=PREHENSION + "temporalCoupling",
    )
    """The admissible timing relation between predicted load and grip response."""
    coupling_active_in: list[ActionPhase] = prop(
        default_factory=list,
        iri=PREHENSION + "couplingActiveIn",
    )
    """The phases in which grip and load remain functionally coupled."""


# %% structure


@ontological(iri=PREHENSION + "PrehensionPhaseStructure")
@dataclass(eq=False)
class PrehensionPhaseStructure(ObjectDynamicsPhaseStructure):
    """A phase structure with explicit prehensile force-control relations."""

    has_force_coupling: list[GripLoadCoupling] = prop(
        default_factory=list,
        iri=PREHENSION + "hasForceCoupling",
    )
    """The functional grip-load relations declared by the structure."""
    has_anticipatory_grip_adjustment: list[AnticipatoryGripAdjustment] = prop(
        default_factory=list,
        iri=PREHENSION + "hasAnticipatoryGripAdjustment",
    )
    """The prediction-triggered grip adjustments declared by the structure."""
