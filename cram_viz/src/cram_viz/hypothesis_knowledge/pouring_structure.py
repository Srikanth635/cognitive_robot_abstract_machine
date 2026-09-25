"""
A hand-authored, illustrative ``PouringPhaseStructure`` built entirely from
``llm_reasoner``'s real Flanagan/Johansson classes
(``flanagan.core``/``flanagan.object_dynamics``/``flanagan.prehension``/
``flanagan.pouring``).

Every value here is a plausible illustrative sample -- clearly not a verified fact --
the same honest convention :mod:`cram_viz.action_knowledge` already uses for its own
hand-authored action knowledge. Unlike that module, every object constructed here is a
genuine krrood ``Symbol`` (via ``KnowledgeEntity``), so it can be handed straight to
:mod:`krrood.entity_query_language` as a real EQL domain -- see
:mod:`cram_viz.hypothesis_knowledge`.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import pairwise
from typing import cast

from llm_reasoner.knowledge.microworlds.flanagan.core import (
    ActionPhase,
    AuditorySignal,
    CategoricalValue,
    ComparisonOutcome,
    ConditionMode,
    ConditionPolarity,
    ConditionRequirement,
    ContactTransition,
    ContinuousSignalPrediction,
    CorrectionDirection,
    CorrectiveAction,
    CouplingRegime,
    Estimate,
    EstimateUpdateDirection,
    ForwardModel,
    InitialStateEstimate,
    InternalModel,
    InverseModel,
    MagnitudeRange,
    MeasurementUnit,
    MechanicalEventType,
    ModelUpdate,
    Parameter,
    PhaseConditionType,
    PhaseTransition,
    PredictionComparison,
    PredictionError,
    PredictionErrorKind,
    ProprioceptiveSignal,
    RecoveryMode,
    RegulationMode,
    RoleSlot,
    SafetyMargin,
    SensorimotorControlPoint,
    SensoryEventPrediction,
    SensoryObservationType,
    SensorySignalType,
    SignalTrajectory,
    Subgoal,
    TactileSignal,
    TemporalWindow,
    TrajectoryTrend,
    VisualSignal,
)
from llm_reasoner.knowledge.microworlds.flanagan.object_dynamics import (
    CoordinateFrame,
    DesiredState,
    DynamicsPerturbation,
    DynamicsTransfer,
    EfferenceCopy,
    InternalModelComposition,
    LimbDynamicsModel,
    LoadForceEstimate,
    MemoryPersistence,
    MotorCommand,
    ObjectDynamicsModel,
    PerceptualRepresentation,
    PerturbationKind,
    RepresentationAccuracy,
    RepresentationDissociation,
    RepresentationPurpose,
    RepresentationUpdate,
    SensorimotorMemory,
    TransferOutcome,
)
from llm_reasoner.knowledge.microworlds.flanagan.pouring import (
    POUR,
    AlignmentErrorEstimate,
    AlignmentPhase,
    CenterOfMassShiftEstimate,
    DynamicViscosityEstimate,
    FlowCessation,
    FlowOnset,
    FlowOnsetThreshold,
    FlowRateSetpoint,
    FlowRegulationPhase,
    FlowTerminationPhase,
    GranularFlowDynamicsEstimate,
    LipAligned,
    LipAlignmentSetpoint,
    MaterialRegime,
    PouringPhaseStructure,
    PourPoseSetpoint,
    RemainingSourceQuantityEstimate,
    ReturnUprightPhase,
    SloshDynamicsEstimate,
    SpillEvent,
    SpillThreshold,
    TargetFillStateEstimate,
    TargetQuantityReached,
    TiltAngleSetpoint,
    TiltRateSetpoint,
    TiltToFlowPhase,
    TorqueSetpoint,
    TransferredQuantitySetpoint,
    TransferTerminationReason,
    UprightAngleSetpoint,
    UprightRestored,
)
from llm_reasoner.knowledge.microworlds.flanagan.prehension import (
    PREHENSION,
    AnticipatoryGripAdjustment,
    ContactBreak,
    ContactEstablishment,
    ContactEstablishmentPhase,
    ContactReleasePhase,
    CorrectiveGripResponse,
    EffectorObjectContact,
    EffectorPositioned,
    ExpectedCenterOfMassPosition,
    ExpectedFrictionCoefficient,
    ExpectedMass,
    ExpectedShape,
    ForceLoadingPhase,
    ForcesUnloaded,
    GraspStability,
    GripForceSetpoint,
    GripLoadCoupling,
    HoldPhase,
    LiftClearanceReached,
    LiftClearanceSetpoint,
    Liftoff,
    LiftoffThreshold,
    LiftPhase,
    LoadForceRateSetpoint,
    LoadForceSetpoint,
    ObjectClearOfSupport,
    ReachPhase,
    ReachPoseSetpoint,
    ReachTargetReached,
    ReplacePhase,
    SafeContactForce,
    SlipEvent,
    SlipThreshold,
    StableHoldReached,
    Touchdown,
    TouchdownThreshold,
    TransportPhase,
    TransportTargetReached,
    UnloadPhase,
    WithdrawalComplete,
    WithdrawalDistanceSetpoint,
    WithdrawPhase,
)
from llm_reasoner.knowledge.ontological.decorators import ConceptReference


# %% roles
def _build_roles() -> dict[str, RoleSlot]:
    """
    The unbound participant slots a pouring action presupposes.

    Each ``RoleSlot`` names the *class* it becomes once bound (``binds_as``), never an
    instance of that class -- those classes are occurrence-level and instantiating one
    directly here would violate the knowledge scope (see ``KnowledgeScope.admit``).
    """
    return {
        "effector": RoleSlot(
            label="effector: gripper",
            binds_as=ConceptReference(
                iri=PREHENSION + "GraspingEffector", label="Grasping Effector"
            ),
        ),
        "grasp_site": RoleSlot(
            label="grasp site: cereal box handle",
            binds_as=ConceptReference(iri=PREHENSION + "GraspSite", label="Grasp Site"),
        ),
        "initial_support": RoleSlot(
            label="initial support: kitchen counter",
            binds_as=ConceptReference(
                iri=PREHENSION + "InitialSupport", label="Initial Support"
            ),
        ),
        "final_support": RoleSlot(
            label="final support: kitchen counter",
            binds_as=ConceptReference(
                iri=PREHENSION + "FinalSupport", label="Final Support"
            ),
        ),
        "source_container": RoleSlot(
            label="source container: cereal box",
            binds_as=ConceptReference(
                iri=POUR + "SourceContainer", label="Source Container"
            ),
        ),
        "target_receptacle": RoleSlot(
            label="target receptacle: bowl",
            binds_as=ConceptReference(
                iri=POUR + "TargetReceptacle", label="Target Receptacle"
            ),
        ),
        "poured_substance": RoleSlot(
            label="poured substance: cereal",
            binds_as=ConceptReference(
                iri=POUR + "PouredSubstance", label="Poured Substance"
            ),
        ),
        "pouring_lip": RoleSlot(
            label="pouring lip: box spout",
            binds_as=ConceptReference(iri=POUR + "PouringLip", label="Pouring Lip"),
        ),
    }


# %% event types
def _build_event_types(
    roles: dict[str, RoleSlot], parameters: dict[str, Parameter]
) -> dict[str, object]:
    """
    The discrete mechanical/flow events that delimit the pouring phases.
    """
    return {
        "reach_target_reached": ReachTargetReached(
            label="pre-contact reach target reached",
            event_on_role=[roles["effector"], roles["grasp_site"]],
            attains_setpoint=[parameters["reach_pose"]],
        ),
        "contact_establishment": ContactEstablishment(
            label="contact established",
            event_on_role=[roles["effector"], roles["source_container"]],
            contact_transition=ContactTransition.ESTABLISHES,
        ),
        "contact_break": ContactBreak(
            label="contact broken",
            event_on_role=[roles["effector"], roles["source_container"]],
            contact_transition=ContactTransition.BREAKS,
        ),
        "liftoff": Liftoff(
            label="liftoff",
            event_on_role=[roles["source_container"], roles["initial_support"]],
            contact_transition=ContactTransition.BREAKS,
            crosses_threshold=parameters["liftoff_threshold"],
        ),
        "touchdown": Touchdown(
            label="touchdown",
            event_on_role=[roles["source_container"], roles["final_support"]],
            contact_transition=ContactTransition.ESTABLISHES,
            crosses_threshold=parameters["touchdown_threshold"],
        ),
        "slip_event": SlipEvent(
            label="slip",
            is_failure_event=True,
            event_on_role=[roles["effector"], roles["source_container"]],
            crosses_threshold=parameters["slip_threshold"],
        ),
        "lift_clearance_reached": LiftClearanceReached(
            label="lift clearance reached",
            event_on_role=[roles["source_container"]],
            attains_setpoint=[parameters["lift_clearance"]],
        ),
        "pour_pose_reached": TransportTargetReached(
            label="pour pose reached",
            event_on_role=[roles["source_container"], roles["target_receptacle"]],
            attains_setpoint=[parameters["pour_pose"]],
        ),
        "flow_onset": FlowOnset(
            label="flow onset",
            event_on_role=[roles["poured_substance"], roles["pouring_lip"]],
            crosses_threshold=parameters["flow_onset_threshold"],
        ),
        "flow_cessation": FlowCessation(
            label="flow cessation",
            event_on_role=[roles["poured_substance"], roles["pouring_lip"]],
        ),
        "spill_event": SpillEvent(
            label="spill",
            is_failure_event=True,
            event_on_role=[roles["poured_substance"], roles["target_receptacle"]],
            crosses_threshold=parameters["spill_threshold"],
        ),
        "target_quantity_reached": TargetQuantityReached(
            label="target volume reached",
            event_on_role=[roles["target_receptacle"]],
            attains_setpoint=[parameters["transferred_quantity"]],
        ),
        "lip_aligned": LipAligned(
            label="lip aligned",
            event_on_role=[roles["pouring_lip"], roles["target_receptacle"]],
            attains_setpoint=[parameters["lip_alignment"]],
        ),
        "upright_restored": UprightRestored(
            label="upright restored",
            event_on_role=[roles["source_container"]],
            attains_setpoint=[parameters["upright_angle"]],
        ),
        "ready_to_transport": StableHoldReached(
            label="ready to transport",
            event_on_role=[roles["source_container"]],
            attains_setpoint=[
                parameters["grip_force"],
                parameters["load_force"],
            ],
        ),
        "forces_unloaded": ForcesUnloaded(
            label="support transfer and force unloading complete",
            weight_supported_by=[roles["final_support"]],
            event_on_role=[
                roles["effector"],
                roles["source_container"],
                roles["final_support"],
            ],
            attains_setpoint=[
                parameters["unloaded_grip_force"],
                parameters["unloaded_load_force"],
            ],
        ),
        "withdrawal_complete": WithdrawalComplete(
            label="withdrawal complete",
            event_on_role=[roles["effector"]],
            attains_setpoint=[parameters["withdrawal_distance"]],
        ),
    }


# %% conditions
def _build_condition_requirements(
    roles: dict[str, RoleSlot], events: dict[str, object]
) -> dict[str, ConditionRequirement]:
    """
    The world-state demands individual phases place on their participants.
    """
    grasp_stability = GraspStability(
        label="grasp stability",
        condition_on_role=[roles["effector"], roles["source_container"]],
        violation_signalled_by=events["slip_event"],
    )
    object_clear_of_support = ObjectClearOfSupport(
        label="object clear of support",
        condition_on_role=[roles["source_container"], roles["initial_support"]],
    )
    effector_positioned = EffectorPositioned(
        label="effector positioned",
        condition_on_role=[roles["effector"], roles["grasp_site"]],
    )
    effector_object_contact = EffectorObjectContact(
        label="effector-object contact",
        condition_on_role=[roles["effector"], roles["source_container"]],
    )
    transfer_path_contained = PhaseConditionType(
        label="predicted transfer path enters target receptacle",
        condition_on_role=[
            roles["source_container"],
            roles["pouring_lip"],
            roles["poured_substance"],
            roles["target_receptacle"],
        ],
        violation_signalled_by=events["spill_event"],
    )
    return {
        "grasp_stability_throughout": ConditionRequirement(
            label="grasp stability required throughout",
            on_condition=grasp_stability,
            mode=ConditionMode.THROUGHOUT,
            polarity=ConditionPolarity.MUST_HOLD,
        ),
        "object_clear_at_exit": ConditionRequirement(
            label="object clear of support at exit",
            on_condition=object_clear_of_support,
            mode=ConditionMode.AT_EXIT,
            polarity=ConditionPolarity.MUST_HOLD,
        ),
        "effector_positioned_at_entry": ConditionRequirement(
            label="effector positioned at entry",
            on_condition=effector_positioned,
            mode=ConditionMode.AT_ENTRY,
            polarity=ConditionPolarity.MUST_HOLD,
            is_world_assumption=True,
        ),
        "effector_contact_at_exit": ConditionRequirement(
            label="effector-object contact at exit",
            on_condition=effector_object_contact,
            mode=ConditionMode.AT_EXIT,
            polarity=ConditionPolarity.MUST_HOLD,
        ),
        "grasp_stability_at_exit": ConditionRequirement(
            label="grasp stability established at exit",
            on_condition=grasp_stability,
            mode=ConditionMode.AT_EXIT,
            polarity=ConditionPolarity.MUST_HOLD,
        ),
        "transfer_path_contained_at_exit": ConditionRequirement(
            label="transfer path contained at exit",
            on_condition=transfer_path_contained,
            mode=ConditionMode.AT_EXIT,
            polarity=ConditionPolarity.MUST_HOLD,
        ),
        "transfer_path_contained_throughout": ConditionRequirement(
            label="transfer path contained throughout",
            on_condition=transfer_path_contained,
            mode=ConditionMode.THROUGHOUT,
            polarity=ConditionPolarity.MUST_HOLD,
        ),
    }


# %% internal models and sensory signals
@dataclass
class InternalModelFragments:
    """The internal-model subgraph declared by the reference structure."""

    models: dict[str, InternalModel]
    """The predictive models and representations addressable by phase builders."""
    desired_states: list[DesiredState]
    """The desired states supplied to inverse models."""
    motor_commands: list[MotorCommand]
    """The outgoing commands selected by inverse models."""
    efference_copies: list[EfferenceCopy]
    """The internal copies of outgoing commands."""
    compositions: list[InternalModelComposition]
    """The explicit command-to-prediction pathways."""
    memories: list[SensorimotorMemory]
    """The memories retaining object-specific representations."""
    updates: list[RepresentationUpdate]
    """The representation updates retained by those memories."""
    perturbations: list[DynamicsPerturbation]
    """The equivalent load and rotation contexts used for comparison."""
    transfers: list[DynamicsTransfer]
    """The adaptation-transfer outcomes between perturbation contexts."""
    dissociations: list[RepresentationDissociation]
    """The sensorimotor and perceptual representation dissociations."""

    def connect_predictions(
        self,
        predictions: list[SensoryEventPrediction],
        mechanical_events: list[MechanicalEventType],
    ) -> None:
        """Complete the canonical pathway with generated consequences."""
        composition = self.compositions[0]
        composition.predicted_mechanical_consequence = mechanical_events
        composition.predicted_sensory_consequence = [
            prediction
            for prediction in predictions
            if prediction.generated_by is composition.forward_model
        ]


def _build_internal_models(
    roles: dict[str, RoleSlot],
    parameters: dict[str, Parameter],
    requirements: dict[str, ConditionRequirement],
) -> InternalModelFragments:
    """Build generic internal models and the paper-grounded dynamics contexts."""
    forward_model = ForwardModel(
        label="forward model: predicted contact, load, and flow consequences"
    )
    inverse_model = InverseModel(
        label="inverse model: motor command selected from desired manipulation state"
    )
    desired_state = DesiredState(
        label="stable object manipulation state",
        desired_condition=[
            cast(
                PhaseConditionType,
                requirements["grasp_stability_at_exit"].on_condition,
            )
        ],
    )
    motor_command = MotorCommand(
        label="predicted grip and load command",
        pursues_state=desired_state,
        generated_by=inverse_model,
        commands_setpoint=[
            cast(GripForceSetpoint, parameters["grip_force"]),
            cast(LoadForceSetpoint, parameters["load_force"]),
        ],
    )
    efference_copy = EfferenceCopy(
        label="copy of predicted grip and load command",
        copies_command=motor_command,
    )

    perceptual_estimate = cast(
        ExpectedMass,
        parameters["perceptual_expected_mass"],
    )
    limb_load_estimate = cast(
        LoadForceEstimate,
        parameters["limb_load_estimate"],
    )
    object_estimates = [
        parameter
        for name, parameter in parameters.items()
        if isinstance(parameter, Estimate)
        and name not in {"perceptual_expected_mass", "limb_load_estimate"}
    ]
    represented_object_roles: list[RoleSlot] = []
    for estimate in object_estimates:
        for role in estimate.parameter_on_role:
            if role not in represented_object_roles:
                represented_object_roles.append(role)

    object_model = ObjectDynamicsModel(
        label="object-centered cereal-box dynamics",
        holds_estimate=object_estimates,
        coordinate_frame=CoordinateFrame.OBJECT_CENTERED,
        purpose=RepresentationPurpose.SENSORIMOTOR_CONTROL,
        represents_role=represented_object_roles,
        uses_efference_copy=[efference_copy],
    )
    limb_model = LimbDynamicsModel(
        label="arm-centered limb dynamics",
        holds_estimate=[limb_load_estimate],
        coordinate_frame=CoordinateFrame.ARM_CENTERED,
        purpose=RepresentationPurpose.SENSORIMOTOR_CONTROL,
        represents_role=[roles["effector"]],
    )
    perceptual_model = PerceptualRepresentation(
        label="object-centered perceptual weight judgment",
        holds_estimate=[perceptual_estimate],
        coordinate_frame=CoordinateFrame.OBJECT_CENTERED,
        purpose=RepresentationPurpose.PERCEPTUAL_JUDGMENT,
        represents_role=[roles["source_container"]],
    )

    memory = SensorimotorMemory(
        label="grasp-scoped cereal-box dynamics memory",
        stores_representation=[object_model],
        persistence=MemoryPersistence.WHILE_OBJECT_COUPLED,
    )
    update = RepresentationUpdate(
        label="object-dynamics representation update",
        updates_representation=object_model,
        updates_estimate=[cast(ExpectedMass, parameters["expected_mass"])],
        direction=EstimateUpdateDirection.REVISE,
        persists_in=memory,
    )
    composition = InternalModelComposition(
        label="efference-copy object-load prediction",
        desired_state=desired_state,
        inverse_model=inverse_model,
        motor_command=motor_command,
        efference_copy=efference_copy,
        dynamics_model=object_model,
        forward_model=forward_model,
    )

    object_load = DynamicsPerturbation(
        label="load applied to the held object",
        kind=PerturbationKind.LOAD,
        affects_model=object_model,
        coordinate_frame=CoordinateFrame.OBJECT_CENTERED,
    )
    arm_load = DynamicsPerturbation(
        label="equivalent load applied directly to the arm",
        kind=PerturbationKind.LOAD,
        affects_model=limb_model,
        coordinate_frame=CoordinateFrame.ARM_CENTERED,
    )
    object_load.equivalent_to = [arm_load]
    arm_load.equivalent_to = [object_load]

    object_rotation = DynamicsPerturbation(
        label="rotation of the held object",
        kind=PerturbationKind.ROTATION,
        affects_model=object_model,
        coordinate_frame=CoordinateFrame.OBJECT_CENTERED,
    )
    hand_rotation = DynamicsPerturbation(
        label="rotation of the hand",
        kind=PerturbationKind.ROTATION,
        affects_model=limb_model,
        coordinate_frame=CoordinateFrame.HAND_CENTERED,
    )
    object_rotation.equivalent_to = [hand_rotation]
    hand_rotation.equivalent_to = [object_rotation]

    load_transfer = DynamicsTransfer(
        label="object-load versus arm-load transfer",
        from_perturbation=object_load,
        to_perturbation=arm_load,
        outcome=TransferOutcome.DIFFERS,
    )
    rotation_transfer = DynamicsTransfer(
        label="object-rotation versus hand-rotation transfer",
        from_perturbation=object_rotation,
        to_perturbation=hand_rotation,
        outcome=TransferOutcome.DIFFERS,
    )
    dissociation = RepresentationDissociation(
        label="accurate motor scaling with persistent size-weight illusion",
        sensorimotor_representation=object_model,
        perceptual_representation=perceptual_model,
        sensorimotor_estimate=cast(ExpectedMass, parameters["expected_mass"]),
        perceptual_estimate=perceptual_estimate,
        sensorimotor_accuracy=RepresentationAccuracy.ACCURATE,
        perceptual_accuracy=RepresentationAccuracy.BIASED,
    )

    return InternalModelFragments(
        models={
            "forward_model": forward_model,
            "inverse_model": inverse_model,
            "object_representation": object_model,
            "limb_dynamics": limb_model,
            "perceptual_representation": perceptual_model,
        },
        desired_states=[desired_state],
        motor_commands=[motor_command],
        efference_copies=[efference_copy],
        compositions=[composition],
        memories=[memory],
        updates=[update],
        perturbations=[object_load, arm_load, object_rotation, hand_rotation],
        transfers=[load_transfer, rotation_transfer],
        dissociations=[dissociation],
    )


def _build_sensory_signals() -> dict[str, SensorySignalType]:
    return {
        "tactile": TactileSignal(label="tactile signal"),
        "proprioceptive": ProprioceptiveSignal(label="proprioceptive signal"),
        "visual": VisualSignal(label="visual signal"),
        "auditory": AuditorySignal(label="auditory signal: rising pitch"),
    }


# %% parameters
def _magnitude(lower: float, upper: float, unit: MeasurementUnit) -> MagnitudeRange:
    return MagnitudeRange(
        lower_bound=lower,
        upper_bound=upper,
        physical_dimension=unit.physical_dimension,
        unit=unit,
    )


def _quantity_unit(material_regime: MaterialRegime) -> MeasurementUnit:
    """Return the reference quantity unit appropriate to a material regime."""
    if material_regime is MaterialRegime.GRANULAR:
        return MeasurementUnit.KILOGRAM
    return MeasurementUnit.MILLILITRE


def _flow_rate_unit(material_regime: MaterialRegime) -> MeasurementUnit:
    """Return the reference flow-rate unit appropriate to a material regime."""
    if material_regime is MaterialRegime.GRANULAR:
        return MeasurementUnit.KILOGRAM_PER_SECOND
    return MeasurementUnit.MILLILITRE_PER_SECOND


def _build_parameters(
    roles: dict[str, RoleSlot],
    material_regime: MaterialRegime,
) -> dict[str, Parameter]:
    """
    Commanded setpoints, thresholds and object-property estimates.

    Setpoints/thresholds are built first so the property estimates that scale or adapt
    them (``scales``/``adapts_to``) can reference the real instances.
    """
    reach_pose = ReachPoseSetpoint(
        label="pre-contact reach pose",
        categorical_value=CategoricalValue(
            category=ConceptReference(
                iri=PREHENSION + "NominalReachPose",
                label="nominal pre-contact pose",
            )
        ),
        parameter_on_role=[roles["effector"], roles["grasp_site"]],
    )
    grip_force = GripForceSetpoint(
        label="grip force setpoint",
        parameter_range=_magnitude(5, 15, MeasurementUnit.NEWTON),
        parameter_on_role=[roles["effector"], roles["source_container"]],
    )
    load_force = LoadForceSetpoint(
        label="load force setpoint",
        parameter_range=_magnitude(2, 10, MeasurementUnit.NEWTON),
        parameter_on_role=[roles["effector"], roles["source_container"]],
    )
    unloaded_grip_force = GripForceSetpoint(
        label="unloaded grip-force target",
        parameter_range=_magnitude(0, 0, MeasurementUnit.NEWTON),
        parameter_on_role=[roles["effector"], roles["source_container"]],
    )
    unloaded_load_force = LoadForceSetpoint(
        label="unloaded load-force target",
        parameter_range=_magnitude(0, 0, MeasurementUnit.NEWTON),
        parameter_on_role=[roles["effector"], roles["source_container"]],
    )
    load_force_rate = LoadForceRateSetpoint(
        label="load force rate setpoint",
        parameter_range=_magnitude(20, 80, MeasurementUnit.NEWTON_PER_SECOND),
        parameter_on_role=[roles["effector"], roles["source_container"]],
        rate_of=load_force,
    )
    tilt_angle = TiltAngleSetpoint(
        label="tilt angle setpoint",
        parameter_range=_magnitude(30, 90, MeasurementUnit.DEGREE),
        parameter_on_role=[roles["source_container"]],
    )
    tilt_rate = TiltRateSetpoint(
        label="tilt rate setpoint",
        parameter_range=_magnitude(5, 20, MeasurementUnit.DEGREE_PER_SECOND),
        parameter_on_role=[roles["source_container"]],
    )
    torque = TorqueSetpoint(
        label="torque setpoint",
        parameter_range=_magnitude(0.1, 1.0, MeasurementUnit.NEWTON_METRE),
        parameter_on_role=[roles["effector"], roles["source_container"]],
    )
    quantity_unit = _quantity_unit(material_regime)
    flow_rate_unit = _flow_rate_unit(material_regime)
    is_granular = material_regime is MaterialRegime.GRANULAR
    quantity_lower_bound = 0.2 if is_granular else 200.0
    quantity_upper_bound = 0.4 if is_granular else 400.0
    flow_rate_lower_bound = 0.01 if is_granular else 10.0
    flow_rate_upper_bound = 0.025 if is_granular else 25.0
    transferred_quantity = TransferredQuantitySetpoint(
        label="transferred quantity setpoint",
        parameter_range=_magnitude(
            quantity_lower_bound,
            quantity_upper_bound,
            quantity_unit,
        ),
        parameter_on_role=[roles["poured_substance"], roles["target_receptacle"]],
    )
    flow_rate = FlowRateSetpoint(
        label="flow rate setpoint",
        parameter_range=_magnitude(
            flow_rate_lower_bound,
            flow_rate_upper_bound,
            flow_rate_unit,
        ),
        parameter_on_role=[roles["poured_substance"], roles["target_receptacle"]],
    )
    lift_clearance = LiftClearanceSetpoint(
        label="lift clearance setpoint",
        parameter_range=_magnitude(0.1, 0.2, MeasurementUnit.METRE),
        parameter_on_role=[roles["source_container"], roles["initial_support"]],
    )
    withdrawal_distance = WithdrawalDistanceSetpoint(
        label="withdrawal distance setpoint",
        parameter_range=_magnitude(0.1, 0.3, MeasurementUnit.METRE),
        parameter_on_role=[roles["effector"]],
    )
    pour_pose = PourPoseSetpoint(
        label="pour pose setpoint",
        categorical_value=CategoricalValue(
            category=ConceptReference(
                iri=POUR + "NominalPourPose",
                label="nominal pour pose",
            )
        ),
        parameter_on_role=[
            roles["source_container"],
            roles["target_receptacle"],
        ],
    )
    lip_alignment = LipAlignmentSetpoint(
        label="lip alignment setpoint",
        parameter_range=_magnitude(0, 0, MeasurementUnit.DEGREE),
        parameter_on_role=[roles["pouring_lip"], roles["target_receptacle"]],
    )
    upright_angle = UprightAngleSetpoint(
        label="upright angle setpoint",
        parameter_range=_magnitude(0, 0, MeasurementUnit.DEGREE),
        parameter_on_role=[roles["source_container"]],
    )
    slip_threshold = SlipThreshold(
        label="slip threshold",
        parameter_range=_magnitude(3, 3, MeasurementUnit.NEWTON),
        parameter_on_role=[roles["effector"], roles["source_container"]],
    )
    grip_range = cast(MagnitudeRange, grip_force.parameter_range)
    slip_range = cast(MagnitudeRange, slip_threshold.parameter_range)
    safety_margin = SafetyMargin(
        label="grip-force safety margin over slip",
        parameter_range=_magnitude(
            grip_range.lower_bound - slip_range.upper_bound,
            grip_range.upper_bound - slip_range.lower_bound,
            MeasurementUnit.NEWTON,
        ),
        parameter_on_role=[roles["effector"], roles["source_container"]],
        margin_on=grip_force,
        margin_over=slip_threshold,
    )
    liftoff_threshold = LiftoffThreshold(
        label="liftoff threshold",
        parameter_range=_magnitude(4, 4, MeasurementUnit.NEWTON),
        parameter_on_role=[roles["source_container"], roles["initial_support"]],
    )
    touchdown_threshold = TouchdownThreshold(
        label="touchdown threshold",
        parameter_range=_magnitude(1, 1, MeasurementUnit.NEWTON),
        parameter_on_role=[roles["source_container"], roles["final_support"]],
    )
    flow_onset_threshold = FlowOnsetThreshold(
        label="flow onset threshold",
        parameter_range=_magnitude(45, 45, MeasurementUnit.DEGREE),
        parameter_on_role=[
            roles["source_container"],
            roles["pouring_lip"],
            roles["poured_substance"],
        ],
    )
    spill_threshold = SpillThreshold(
        label="spill threshold",
        parameter_range=_magnitude(
            flow_rate_upper_bound,
            flow_rate_upper_bound,
            flow_rate_unit,
        ),
        parameter_on_role=[
            roles["source_container"],
            roles["poured_substance"],
            roles["target_receptacle"],
        ],
    )

    expected_mass = ExpectedMass(
        label="expected mass",
        parameter_range=_magnitude(0.3, 0.6, MeasurementUnit.KILOGRAM),
        parameter_on_role=[roles["source_container"]],
        scales=[grip_force, load_force],
    )
    perceptual_expected_mass = ExpectedMass(
        label="perceptual mass estimate influenced by object size",
        parameter_range=expected_mass.parameter_range,
        parameter_on_role=[roles["source_container"]],
    )
    object_load_estimate = LoadForceEstimate(
        label="load represented as acting on the held object",
        parameter_range=load_force.parameter_range,
        parameter_on_role=[roles["source_container"]],
    )
    limb_load_estimate = LoadForceEstimate(
        label="equivalent load represented as acting on the arm",
        parameter_range=load_force.parameter_range,
        parameter_on_role=[roles["effector"]],
    )

    expected_friction_coefficient = ExpectedFrictionCoefficient(
        label="expected friction coefficient",
        parameter_range=_magnitude(0.3, 0.6, MeasurementUnit.DIMENSIONLESS),
        parameter_on_role=[roles["effector"], roles["grasp_site"]],
    )
    expected_center_of_mass_position = ExpectedCenterOfMassPosition(
        label="expected centre-of-mass position",
        parameter_range=_magnitude(0.05, 0.15, MeasurementUnit.METRE),
        parameter_on_role=[roles["source_container"]],
        scales=[torque],
    )
    expected_shape = ExpectedShape(
        label="expected shape",
        categorical_value=CategoricalValue(
            category=ConceptReference(
                iri=POUR + "RectangularPrismShape",
                label="rectangular prism",
            )
        ),
        parameter_on_role=[roles["source_container"], roles["grasp_site"]],
    )
    remaining_source_quantity = RemainingSourceQuantityEstimate(
        label="remaining source quantity",
        parameter_range=_magnitude(0.0, quantity_upper_bound, quantity_unit),
        parameter_on_role=[roles["source_container"], roles["poured_substance"]],
        scales=[tilt_angle],
    )
    target_fill_state = TargetFillStateEstimate(
        label="target receptacle fill state",
        parameter_range=_magnitude(0.0, 0.9, MeasurementUnit.FRACTION),
        parameter_on_role=[roles["target_receptacle"]],
    )
    alignment_error = AlignmentErrorEstimate(
        label="pouring-lip alignment error",
        parameter_range=_magnitude(0.0, 5.0, MeasurementUnit.DEGREE),
        parameter_on_role=[roles["pouring_lip"], roles["target_receptacle"]],
    )
    center_of_mass_shift = CenterOfMassShiftEstimate(
        label="expected center of mass shift",
        parameter_range=_magnitude(0.01, 0.05, MeasurementUnit.METRE),
        parameter_on_role=[roles["source_container"], roles["poured_substance"]],
        scales=[torque],
    )

    material_parameters: dict[str, Parameter] = {}
    if material_regime is MaterialRegime.GRANULAR:
        flow_dynamics: Estimate = GranularFlowDynamicsEstimate(
            label="granular flow dynamics",
            categorical_value=CategoricalValue(
                category=ConceptReference(
                    iri=POUR + "FreeFlowingGranularMaterial",
                    label="free-flowing granular material",
                )
            ),
            parameter_on_role=[roles["poured_substance"]],
        )
    else:
        viscosity_magnitude = (
            1.0 if material_regime is MaterialRegime.LIQUID else 1000.0
        )
        flow_dynamics = DynamicViscosityEstimate(
            label="dynamic viscosity estimate",
            parameter_range=_magnitude(
                viscosity_magnitude,
                viscosity_magnitude,
                MeasurementUnit.CENTIPOISE,
            ),
            parameter_on_role=[roles["poured_substance"]],
            scales=[tilt_rate],
        )
        if material_regime is MaterialRegime.LIQUID:
            material_parameters["slosh_dynamics"] = SloshDynamicsEstimate(
                label="slosh dynamics estimate",
                categorical_value=CategoricalValue(
                    category=ConceptReference(
                        iri=POUR + "LowSloshDynamics",
                        label="low slosh dynamics",
                    )
                ),
                parameter_on_role=[
                    roles["source_container"],
                    roles["poured_substance"],
                ],
            )
    material_parameters["flow_dynamics"] = flow_dynamics

    tilt_angle.adapts_to = [remaining_source_quantity]
    torque.adapts_to = [
        expected_center_of_mass_position,
        center_of_mass_shift,
    ]
    tilt_rate.adapts_to = [flow_dynamics]

    parameters = {
        "reach_pose": reach_pose,
        "grip_force": grip_force,
        "load_force": load_force,
        "unloaded_grip_force": unloaded_grip_force,
        "unloaded_load_force": unloaded_load_force,
        "load_force_rate": load_force_rate,
        "tilt_angle": tilt_angle,
        "tilt_rate": tilt_rate,
        "torque": torque,
        "transferred_quantity": transferred_quantity,
        "flow_rate": flow_rate,
        "lift_clearance": lift_clearance,
        "withdrawal_distance": withdrawal_distance,
        "pour_pose": pour_pose,
        "lip_alignment": lip_alignment,
        "upright_angle": upright_angle,
        "slip_threshold": slip_threshold,
        "safety_margin": safety_margin,
        "liftoff_threshold": liftoff_threshold,
        "touchdown_threshold": touchdown_threshold,
        "flow_onset_threshold": flow_onset_threshold,
        "spill_threshold": spill_threshold,
        "perceptual_expected_mass": perceptual_expected_mass,
        "object_load_estimate": object_load_estimate,
        "limb_load_estimate": limb_load_estimate,
        "expected_mass": expected_mass,
        "expected_friction_coefficient": expected_friction_coefficient,
        "expected_center_of_mass_position": expected_center_of_mass_position,
        "expected_shape": expected_shape,
        "remaining_source_quantity": remaining_source_quantity,
        "target_fill_state": target_fill_state,
        "alignment_error": alignment_error,
        "center_of_mass_shift": center_of_mass_shift,
    }
    parameters.update(material_parameters)
    return parameters


#: phase names where the grip-load coupling (and everything it depends on --
#: grip/load force setpoints, the friction estimate setting their ratio, and the
#: slip threshold their ratio must clear) is active: every phase from first
#: contact through release, i.e. every non-UNCOUPLED phase.
_COUPLED_PHASE_NAMES: tuple[str, ...] = (
    "contact",
    "grasp",
    "lift",
    "hold",
    "transport",
    "align",
    "tilt",
    "pour",
    "stop_flow",
    "upright",
    "lower",
    "unload",
)

#: parameter key -> the phase names it is commanded, tracked or estimated in.
#: Grounded in each parameter's own docstring (see ``flanagan.pouring``): a
#: setpoint/threshold names the phases it gates or is commanded during, an
#: estimate names the phases whose feedforward scaling it drives via
#: ``scales``/``adapts_to``.
_PARAMETER_PHASES: dict[str, tuple[str, ...]] = {
    "reach_pose": ("reach",),
    "grip_force": _COUPLED_PHASE_NAMES,
    "load_force": _COUPLED_PHASE_NAMES,
    "unloaded_grip_force": ("unload",),
    "unloaded_load_force": ("unload",),
    "load_force_rate": ("grasp", "lift"),
    "expected_friction_coefficient": _COUPLED_PHASE_NAMES,
    "slip_threshold": _COUPLED_PHASE_NAMES,
    "safety_margin": _COUPLED_PHASE_NAMES,
    "tilt_angle": ("tilt", "pour", "stop_flow", "upright"),
    "tilt_rate": ("tilt", "pour", "stop_flow"),
    "torque": (  # COUPLED-regime phases only; excludes grasp/lower/release
        "lift",
        "hold",
        "transport",
        "align",
        "tilt",
        "pour",
        "stop_flow",
        "upright",
    ),
    "transferred_quantity": ("pour", "stop_flow"),
    "flow_rate": ("pour",),
    "liftoff_threshold": ("grasp", "lift"),
    "lift_clearance": ("lift",),
    "withdrawal_distance": ("withdraw",),
    "pour_pose": ("transport",),
    "lip_alignment": ("align",),
    "upright_angle": ("upright",),
    "flow_onset_threshold": ("tilt",),
    "spill_threshold": ("pour", "stop_flow"),
    "touchdown_threshold": ("lower",),
    "perceptual_expected_mass": ("grasp", "lift"),
    "object_load_estimate": _COUPLED_PHASE_NAMES,
    "limb_load_estimate": ("transport",),
    "expected_mass": ("grasp", "lift"),
    "expected_center_of_mass_position": ("grasp", "lift", "hold", "transport", "align"),
    "center_of_mass_shift": ("tilt", "pour", "stop_flow", "upright", "lower"),
    "expected_shape": ("reach", "grasp"),
    "flow_dynamics": ("tilt", "pour", "stop_flow"),
    "remaining_source_quantity": ("align", "tilt", "pour"),
    "target_fill_state": ("pour",),
    "alignment_error": ("align", "pour"),
    "slosh_dynamics": ("lift", "transport"),
}


def _wire_parameter_phases(
    parameters: dict[str, Parameter], phases: dict[str, object]
) -> None:
    """
    Populate every parameter's ``parameter_in_phase`` from :data:`_PARAMETER_PHASES`.

    ``Parameter.parameter_in_phase`` is a reverse pointer -- like
    ``GripLoadCoupling.coupling_active_in`` -- so nothing else sets it as a side
    effect of building the parameter itself.
    """
    for parameter_key, phase_names in _PARAMETER_PHASES.items():
        parameter = parameters.get(parameter_key)
        if parameter is None:
            continue
        parameter.parameter_in_phase = [
            phases[phase_name] for phase_name in phase_names
        ]


# %% phases
#: (attribute name, phase class, terminating event key, initiating event key or None)
#: in execution order -- the single source of truth for both the phase list and the
#: transitions chaining them.
# "hold" and "withdraw" each get a terminating event distinct from their own
# initiating event ("ready_to_transport"/"withdrawal_complete", both a
# specific SetpointAttained kind): a phase bounded by the same event at both ends
# has zero duration, which contradicts Hold's/Withdraw's own descriptions.
_PHASE_SEQUENCE: tuple[tuple[str, type, str, str | None], ...] = (
    ("reach", ReachPhase, "reach_target_reached", None),
    (
        "contact",
        ContactEstablishmentPhase,
        "contact_establishment",
        "reach_target_reached",
    ),
    ("grasp", ForceLoadingPhase, "liftoff", "contact_establishment"),
    ("lift", LiftPhase, "lift_clearance_reached", "liftoff"),
    ("hold", HoldPhase, "ready_to_transport", "lift_clearance_reached"),
    ("transport", TransportPhase, "pour_pose_reached", "ready_to_transport"),
    ("align", AlignmentPhase, "lip_aligned", "pour_pose_reached"),
    ("tilt", TiltToFlowPhase, "flow_onset", "lip_aligned"),
    ("pour", FlowRegulationPhase, "target_quantity_reached", "flow_onset"),
    ("stop_flow", FlowTerminationPhase, "flow_cessation", "target_quantity_reached"),
    ("upright", ReturnUprightPhase, "upright_restored", "flow_cessation"),
    ("lower", ReplacePhase, "touchdown", "upright_restored"),
    ("unload", UnloadPhase, "forces_unloaded", "touchdown"),
    (
        "release",
        ContactReleasePhase,
        "contact_break",
        "forces_unloaded",
    ),
    ("withdraw", WithdrawPhase, "withdrawal_complete", "contact_break"),
)

#: phase attribute name -> the roles it involves
_PHASE_ROLES: dict[str, tuple[str, ...]] = {
    "reach": ("effector", "grasp_site"),
    "contact": ("effector", "source_container", "grasp_site"),
    "grasp": ("effector", "source_container", "initial_support"),
    "lift": ("effector", "source_container", "initial_support"),
    "hold": ("effector", "source_container"),
    "transport": ("effector", "source_container", "target_receptacle"),
    "align": ("source_container", "pouring_lip", "target_receptacle"),
    "tilt": ("source_container", "pouring_lip"),
    "pour": (
        "source_container",
        "pouring_lip",
        "poured_substance",
        "target_receptacle",
    ),
    "stop_flow": ("source_container", "pouring_lip"),
    "upright": ("source_container",),
    "lower": ("effector", "source_container", "final_support"),
    "unload": ("effector", "source_container", "final_support"),
    "release": ("effector", "source_container"),
    "withdraw": ("effector",),
}

#: phase attribute name -> (coupling regime, regulation mode)
_PHASE_REGIMES: dict[str, tuple[CouplingRegime, RegulationMode]] = {
    "reach": (CouplingRegime.UNCOUPLED, RegulationMode.FEEDFORWARD),
    "contact": (CouplingRegime.TRANSITIONAL, RegulationMode.CONTINUOUS_FEEDBACK),
    "grasp": (CouplingRegime.TRANSITIONAL, RegulationMode.CONTINUOUS_FEEDBACK),
    "lift": (CouplingRegime.COUPLED, RegulationMode.FEEDFORWARD),
    "hold": (CouplingRegime.COUPLED, RegulationMode.FEEDFORWARD),
    "transport": (CouplingRegime.COUPLED, RegulationMode.HYBRID),
    "align": (CouplingRegime.COUPLED, RegulationMode.FEEDFORWARD),
    "tilt": (CouplingRegime.COUPLED, RegulationMode.HYBRID),
    "pour": (CouplingRegime.COUPLED, RegulationMode.HYBRID),
    "stop_flow": (CouplingRegime.COUPLED, RegulationMode.HYBRID),
    "upright": (CouplingRegime.COUPLED, RegulationMode.FEEDFORWARD),
    "lower": (CouplingRegime.COUPLED, RegulationMode.FEEDFORWARD),
    "unload": (CouplingRegime.TRANSITIONAL, RegulationMode.CONTINUOUS_FEEDBACK),
    "release": (CouplingRegime.TRANSITIONAL, RegulationMode.CONTINUOUS_FEEDBACK),
    "withdraw": (CouplingRegime.UNCOUPLED, RegulationMode.FEEDFORWARD),
}

#: phase display labels mirror their explicit control-stage concepts.
_PHASE_LABELS: dict[str, str] = {
    "reach": "Reach",
    "contact": "ContactEstablishment",
    "grasp": "ForceLoading",
    "lift": "Lift",
    "hold": "Hold",
    "transport": "Transport",
    "align": "Align",
    "tilt": "Tilt",
    "pour": "Pour",
    "stop_flow": "StopFlow",
    "upright": "Upright",
    "lower": "Replace",
    "unload": "Unload",
    "release": "ContactRelease",
    "withdraw": "Withdraw",
}


def _build_phases(
    roles: dict[str, RoleSlot],
    events: dict[str, object],
    conditions: dict[str, ConditionRequirement],
    models: dict[str, InternalModel],
    signals: dict[str, SensorySignalType],
) -> dict[str, object]:
    """
    Every pouring phase, its subgoal, and the events/roles/regimes it involves.
    """
    phases: dict[str, object] = {}
    for name, phase_cls, terminating_event_key, _ in _PHASE_SEQUENCE:
        coupling_regime, regulation_mode = _PHASE_REGIMES[name]
        involved_roles = [roles[role_name] for role_name in _PHASE_ROLES[name]]
        terminating_event = events[terminating_event_key]
        subgoal = Subgoal(
            label=f"{_PHASE_LABELS[name]} subgoal",
            condition_on_role=involved_roles,
            attainment_signalled_by=terminating_event,
        )
        phases[name] = phase_cls(
            label=_PHASE_LABELS[name],
            aims_for=subgoal,
            coupling_regime=coupling_regime,
            regulation_mode=regulation_mode,
            terminated_by=[terminating_event],
            phase_involves=involved_roles,
        )

    for name, _, _, initiating_event_key in _PHASE_SEQUENCE:
        if initiating_event_key is not None:
            phases[name].initiated_by = [events[initiating_event_key]]

    phases["contact"].monitors_signal = [signals["tactile"]]
    phases["contact"].has_condition_requirement = [
        conditions["effector_contact_at_exit"],
    ]
    phases["grasp"].monitors_signal = [signals["tactile"]]
    phases["grasp"].governed_by = [models["object_representation"]]
    phases["grasp"].has_condition_requirement = [
        conditions["grasp_stability_at_exit"],
    ]
    phases["lift"].has_condition_requirement = [
        conditions["grasp_stability_throughout"],
        conditions["object_clear_at_exit"],
    ]
    phases["transport"].governed_by = [models["forward_model"]]
    phases["transport"].monitors_signal = [signals["proprioceptive"], signals["visual"]]
    phases["align"].has_condition_requirement = [
        conditions["transfer_path_contained_at_exit"]
    ]
    phases["tilt"].monitors_signal = [signals["proprioceptive"]]
    phases["pour"].monitors_signal = [signals["auditory"], signals["visual"]]
    phases["pour"].governed_by = [models["forward_model"], models["inverse_model"]]
    phases["pour"].has_condition_requirement = [
        conditions["grasp_stability_throughout"],
        conditions["transfer_path_contained_throughout"],
    ]
    phases["stop_flow"].monitors_signal = [signals["auditory"]]
    phases["unload"].monitors_signal = [signals["tactile"], signals["proprioceptive"]]
    phases["release"].monitors_signal = [signals["tactile"]]
    phases["reach"].has_condition_requirement = [
        conditions["effector_positioned_at_entry"]
    ]
    return phases


def _build_transitions(
    phases: dict[str, object], events: dict[str, object]
) -> list[PhaseTransition]:
    """
    One event-gated transition between each consecutive pair of phases.
    """
    ordered_names = [name for name, _, _, _ in _PHASE_SEQUENCE]
    transitions = []
    for from_name, to_name in pairwise(ordered_names):
        to_phase_terminating_key = next(
            terminating_event_key
            for name, _, terminating_event_key, _ in _PHASE_SEQUENCE
            if name == from_name
        )
        transitions.append(
            PhaseTransition(
                label=f"{_PHASE_LABELS[from_name]} -> {_PHASE_LABELS[to_name]}",
                from_phase=phases[from_name],
                to_phase=phases[to_name],
                gated_by=[events[to_phase_terminating_key]],
            )
        )
    return transitions


# %% predictive-control fragments


@dataclass
class PredictiveControlFragments:
    """
    The predictive-control subgraph declared by a reference structure.
    """

    control_points: list[SensorimotorControlPoint]
    """
    The mechanical sites at which comparisons occur.
    """

    predictions: list[SensoryEventPrediction]
    """
    The sensory consequences predicted at control points.
    """

    observations: list[SensoryObservationType]
    """
    The possible observations compared with predictions.
    """

    comparisons: list[PredictionComparison]
    """
    The normal and mismatch comparison outcomes.
    """

    errors: list[PredictionError]
    """
    The prediction errors classified by mismatches.
    """

    corrections: list[CorrectiveAction]
    """
    The motor corrections released by errors.
    """

    model_updates: list[ModelUpdate]
    """
    The estimate revisions supported by prediction errors.
    """

    state_estimates: list[InitialStateEstimate]
    """
    The next-phase states supplied by successful comparisons.
    """

    @classmethod
    def from_control_points(
        cls,
        control_points: list[SensorimotorControlPoint],
    ) -> PredictiveControlFragments:
        """
        Collect the declared predictive subgraph from its control points.
        """
        comparisons = [
            comparison
            for control_point in control_points
            for comparison in control_point.has_comparison
        ]
        return cls(
            control_points=control_points,
            predictions=[
                prediction
                for control_point in control_points
                for prediction in control_point.has_prediction
            ],
            observations=[
                comparison.compares_observation
                for comparison in comparisons
                if comparison.compares_observation is not None
            ],
            comparisons=comparisons,
            errors=[
                error for comparison in comparisons for error in comparison.has_error
            ],
            corrections=[
                correction
                for comparison in comparisons
                for correction in comparison.invokes_correction
            ],
            model_updates=[
                update
                for comparison in comparisons
                for update in comparison.produces_model_update
            ],
            state_estimates=[
                comparison.supplies_state_estimate
                for comparison in comparisons
                if comparison.supplies_state_estimate is not None
            ],
        )


def _timing(
    label: str,
    earliest_offset: float,
    latest_offset: float,
    unit: MeasurementUnit,
) -> TemporalWindow:
    """
    Build a timing window for a reference prediction or observation.
    """
    return TemporalWindow(
        label=label,
        earliest_offset=earliest_offset,
        latest_offset=latest_offset,
        unit=unit,
    )


def _transition_from_phase(
    phases: dict[str, object],
    transitions: list[PhaseTransition],
    phase_name: str,
) -> PhaseTransition:
    """
    Return the transition leaving one named reference phase.
    """
    source_phase = cast(ActionPhase, phases[phase_name])
    return next(
        transition
        for transition in transitions
        if transition.from_phase is source_phase
    )


def _condition(
    requirements: dict[str, ConditionRequirement],
    requirement_name: str,
) -> PhaseConditionType:
    """
    Return the condition carried by one named requirement.
    """
    return cast(PhaseConditionType, requirements[requirement_name].on_condition)


def _matched_comparison(
    label: str,
    prediction: SensoryEventPrediction,
    observation: SensoryObservationType,
    transition: PhaseTransition,
    conditions: list[PhaseConditionType],
) -> PredictionComparison:
    """
    Build and wire a successful comparison-gated transition.
    """
    state_estimate = InitialStateEstimate(
        label=f"{label} next-phase state",
        for_phase=transition.to_phase,
        based_on_observation=[observation],
        estimates_condition=conditions,
    )
    comparison = PredictionComparison(
        label=label,
        compares_prediction=prediction,
        compares_observation=observation,
        outcome=ComparisonOutcome.MATCH,
        confirms_subgoal=cast(Subgoal, transition.from_phase.aims_for),
        supplies_state_estimate=state_estimate,
    )
    comparison.permits_transition = [transition]
    transition.permitted_by = [comparison]
    return comparison


def _contact_control_point(
    roles: dict[str, RoleSlot],
    parameters: dict[str, Parameter],
    events: dict[str, object],
    phases: dict[str, object],
    transitions: list[PhaseTransition],
    models: dict[str, InternalModel],
    signals: dict[str, SensorySignalType],
    requirements: dict[str, ConditionRequirement],
) -> SensorimotorControlPoint:
    """
    Build expected-contact progress and unexpected-contact protection.
    """
    contact_event = cast(ContactEstablishment, events["contact_establishment"])
    forward_model = cast(ForwardModel, models["forward_model"])
    expected_contact = SensoryEventPrediction(
        label="predicted grasp contact",
        modality=signals["tactile"],
        predicts_event=contact_event,
        caused_by_event=contact_event,
        expected_timing=_timing(
            "expected contact timing",
            0.18,
            0.25,
            MeasurementUnit.SECOND,
        ),
        expected_on_role=[roles["effector"], roles["source_container"]],
        generated_by=forward_model,
    )
    clear_approach = SensoryEventPrediction(
        label="predicted unobstructed approach",
        modality=signals["tactile"],
        predicts_feature=ConceptReference(
            iri=POUR + "UnobstructedApproach",
            label="unobstructed approach",
        ),
        caused_by_event=contact_event,
        expected_timing=expected_contact.expected_timing,
        expected_on_role=[roles["effector"], roles["source_container"]],
        generated_by=forward_model,
    )
    contact_observation = SensoryObservationType(
        label="observed expected grasp contact",
        modality=signals["tactile"],
        observes_event=contact_event,
        observed_at=expected_contact.expected_timing,
        observed_on_role=[roles["effector"], roles["source_container"]],
    )
    transition = _transition_from_phase(phases, transitions, "contact")
    matched = _matched_comparison(
        "expected grasp contact matched",
        expected_contact,
        contact_observation,
        transition,
        [_condition(requirements, "effector_contact_at_exit")],
    )

    unexpected_observation = SensoryObservationType(
        label="observed unexpected approach contact",
        modality=signals["tactile"],
        observes_event=contact_event,
        observed_at=_timing(
            "unexpected contact timing",
            0.0,
            0.1,
            MeasurementUnit.SECOND,
        ),
        observed_on_role=[roles["effector"], roles["source_container"]],
    )
    unexpected_error = PredictionError(
        label="unexpected approach contact",
        kind=PredictionErrorKind.UNEXPECTED,
    )
    safe_contact = SafeContactForce(
        label="safe contact force",
        condition_on_role=[roles["effector"], roles["source_container"]],
    )
    protective_correction = CorrectiveAction(
        label="protective contact stop",
        triggered_by_error=[unexpected_error],
        targets_setpoint=[cast(LoadForceRateSetpoint, parameters["load_force_rate"])],
        direction=CorrectionDirection.STOP,
        intended_effect=[safe_contact],
        recovery_mode=RecoveryMode.REPLAN,
    )
    unexpected = PredictionComparison(
        label="unexpected approach contact mismatch",
        compares_prediction=clear_approach,
        compares_observation=unexpected_observation,
        outcome=ComparisonOutcome.MISMATCH,
        has_error=[unexpected_error],
        invokes_correction=[protective_correction],
    )
    return SensorimotorControlPoint(
        label="contact control point",
        at_event_type=contact_event,
        has_prediction=[expected_contact, clear_approach],
        has_comparison=[matched, unexpected],
    )


def _liftoff_control_point(
    roles: dict[str, RoleSlot],
    parameters: dict[str, Parameter],
    events: dict[str, object],
    phases: dict[str, object],
    transitions: list[PhaseTransition],
    models: dict[str, InternalModel],
    signals: dict[str, SensorySignalType],
    requirements: dict[str, ConditionRequirement],
) -> SensorimotorControlPoint:
    """
    Build normal liftoff plus heavy, light, and slip responses.
    """
    liftoff = cast(Liftoff, events["liftoff"])
    slip = cast(SlipEvent, events["slip_event"])
    forward_model = cast(ForwardModel, models["forward_model"])
    representation = cast(ObjectDynamicsModel, models["object_representation"])
    expected_mass = cast(ExpectedMass, parameters["expected_mass"])
    expected_friction = cast(
        ExpectedFrictionCoefficient,
        parameters["expected_friction_coefficient"],
    )
    liftoff_timing = _timing(
        "expected liftoff timing",
        80,
        120,
        MeasurementUnit.MILLISECOND,
    )
    liftoff_prediction = SensoryEventPrediction(
        label="predicted tactile liftoff",
        modality=signals["tactile"],
        predicts_event=liftoff,
        caused_by_event=liftoff,
        expected_timing=liftoff_timing,
        expected_on_role=[roles["source_container"], roles["initial_support"]],
        expected_magnitude=_magnitude(3.8, 4.2, MeasurementUnit.NEWTON),
        tolerance=_magnitude(0.0, 0.2, MeasurementUnit.NEWTON),
        generated_by=forward_model,
    )
    stable_grip_prediction = SensoryEventPrediction(
        label="predicted stable grip at liftoff",
        modality=signals["tactile"],
        predicts_feature=ConceptReference(
            iri=POUR + "StableGripAtLiftoff",
            label="stable grip at liftoff",
        ),
        caused_by_event=liftoff,
        expected_timing=liftoff_timing,
        expected_on_role=[roles["effector"], roles["source_container"]],
        generated_by=forward_model,
    )
    liftoff_observation = SensoryObservationType(
        label="observed tactile liftoff",
        modality=signals["tactile"],
        observes_event=liftoff,
        observed_at=liftoff_timing,
        observed_on_role=[roles["source_container"], roles["initial_support"]],
        observed_magnitude=liftoff_prediction.expected_magnitude,
    )
    transition = _transition_from_phase(phases, transitions, "grasp")
    matched = _matched_comparison(
        "expected liftoff matched",
        liftoff_prediction,
        liftoff_observation,
        transition,
        [_condition(requirements, "object_clear_at_exit")],
    )

    absent_error = PredictionError(
        label="liftoff absent",
        kind=PredictionErrorKind.ABSENT,
    )
    heavy_correction = CorrectiveAction(
        label="increase load force and repeat lift",
        triggered_by_error=[absent_error],
        targets_setpoint=[cast(LoadForceSetpoint, parameters["load_force"])],
        direction=CorrectionDirection.INCREASE,
        intended_effect=[_condition(requirements, "object_clear_at_exit")],
        recovery_mode=RecoveryMode.REPEAT_PHASE,
    )
    heavy_update = ModelUpdate(
        label="increase expected mass",
        triggered_by_error=[absent_error],
        updates_estimate=[expected_mass],
        updates_representation=representation,
        direction=EstimateUpdateDirection.INCREASE,
    )
    heavy_object = PredictionComparison(
        label="heavy object liftoff mismatch",
        compares_prediction=liftoff_prediction,
        outcome=ComparisonOutcome.MISMATCH,
        has_error=[absent_error],
        invokes_correction=[heavy_correction],
        produces_model_update=[heavy_update],
    )

    early_observation = SensoryObservationType(
        label="observed early liftoff",
        modality=signals["tactile"],
        observes_event=liftoff,
        observed_at=_timing(
            "early liftoff timing",
            20,
            60,
            MeasurementUnit.MILLISECOND,
        ),
        observed_on_role=[roles["source_container"], roles["initial_support"]],
        observed_magnitude=_magnitude(2.5, 3.8, MeasurementUnit.NEWTON),
    )
    early_error = PredictionError(
        label="liftoff early",
        kind=PredictionErrorKind.EARLY,
    )
    light_correction = CorrectiveAction(
        label="reduce future load force rate",
        triggered_by_error=[early_error],
        targets_setpoint=[cast(LoadForceRateSetpoint, parameters["load_force_rate"])],
        direction=CorrectionDirection.DECREASE,
        intended_effect=[_condition(requirements, "grasp_stability_at_exit")],
        recovery_mode=RecoveryMode.CONTINUE_PHASE,
    )
    light_update = ModelUpdate(
        label="decrease expected mass",
        triggered_by_error=[early_error],
        updates_estimate=[expected_mass],
        updates_representation=representation,
        direction=EstimateUpdateDirection.DECREASE,
    )
    light_object = PredictionComparison(
        label="light object liftoff mismatch",
        compares_prediction=liftoff_prediction,
        compares_observation=early_observation,
        outcome=ComparisonOutcome.MISMATCH,
        has_error=[early_error],
        invokes_correction=[light_correction],
        produces_model_update=[light_update],
    )

    slip_observation = SensoryObservationType(
        label="observed unexpected slip",
        modality=signals["tactile"],
        observes_event=slip,
        observed_at=liftoff_timing,
        observed_on_role=[roles["effector"], roles["source_container"]],
    )
    slip_error = PredictionError(
        label="unexpected slip",
        kind=PredictionErrorKind.UNEXPECTED,
    )
    slip_correction = CorrectiveGripResponse(
        label="rapid grip force increase",
        triggered_by_error=[slip_error],
        targets_setpoint=[cast(GripForceSetpoint, parameters["grip_force"])],
        direction=CorrectionDirection.INCREASE,
        intended_effect=[_condition(requirements, "grasp_stability_at_exit")],
        recovery_mode=RecoveryMode.CONTINUE_PHASE,
    )
    slip_update = ModelUpdate(
        label="decrease expected friction",
        triggered_by_error=[slip_error],
        updates_estimate=[expected_friction],
        updates_representation=representation,
        direction=EstimateUpdateDirection.DECREASE,
    )
    unexpected_slip = PredictionComparison(
        label="unexpected slip mismatch",
        compares_prediction=stable_grip_prediction,
        compares_observation=slip_observation,
        outcome=ComparisonOutcome.MISMATCH,
        has_error=[slip_error],
        invokes_correction=[slip_correction],
        produces_model_update=[slip_update],
    )
    return SensorimotorControlPoint(
        label="liftoff control point",
        at_event_type=liftoff,
        has_prediction=[liftoff_prediction, stable_grip_prediction],
        has_comparison=[matched, heavy_object, light_object, unexpected_slip],
    )


def _flow_control_point(
    roles: dict[str, RoleSlot],
    parameters: dict[str, Parameter],
    events: dict[str, object],
    phases: dict[str, object],
    transitions: list[PhaseTransition],
    models: dict[str, InternalModel],
    signals: dict[str, SensorySignalType],
) -> SensorimotorControlPoint:
    """
    Build normal flow onset and late-onset correction.
    """
    flow_onset = cast(FlowOnset, events["flow_onset"])
    forward_model = cast(ForwardModel, models["forward_model"])
    representation = cast(ObjectDynamicsModel, models["object_representation"])
    flow_dynamics = cast(Estimate, parameters["flow_dynamics"])
    expected_timing = _timing(
        "expected flow onset timing",
        0.5,
        1.5,
        MeasurementUnit.SECOND,
    )
    prediction = SensoryEventPrediction(
        label="predicted flow onset",
        modality=signals["auditory"],
        predicts_event=flow_onset,
        caused_by_event=flow_onset,
        expected_timing=expected_timing,
        expected_on_role=[roles["poured_substance"], roles["pouring_lip"]],
        generated_by=forward_model,
    )
    observation = SensoryObservationType(
        label="observed flow onset",
        modality=signals["auditory"],
        observes_event=flow_onset,
        observed_at=expected_timing,
        observed_on_role=[roles["poured_substance"], roles["pouring_lip"]],
    )
    transition = _transition_from_phase(phases, transitions, "tilt")
    matched = _matched_comparison(
        "expected flow onset matched",
        prediction,
        observation,
        transition,
        [cast(Subgoal, transition.from_phase.aims_for)],
    )

    late_observation = SensoryObservationType(
        label="observed late flow onset",
        modality=signals["auditory"],
        observes_event=flow_onset,
        observed_at=_timing(
            "late flow onset timing",
            1.6,
            2.5,
            MeasurementUnit.SECOND,
        ),
        observed_on_role=[roles["poured_substance"], roles["pouring_lip"]],
    )
    late_error = PredictionError(
        label="flow onset late",
        kind=PredictionErrorKind.LATE,
    )
    correction = CorrectiveAction(
        label="increase tilt rate",
        triggered_by_error=[late_error],
        targets_setpoint=[cast(TiltRateSetpoint, parameters["tilt_rate"])],
        direction=CorrectionDirection.INCREASE,
        intended_effect=[cast(Subgoal, transition.from_phase.aims_for)],
        recovery_mode=RecoveryMode.CONTINUE_PHASE,
    )
    update = ModelUpdate(
        label="revise represented flow dynamics",
        triggered_by_error=[late_error],
        updates_estimate=[flow_dynamics],
        updates_representation=representation,
        direction=EstimateUpdateDirection.REVISE,
    )
    mismatch = PredictionComparison(
        label="late flow onset mismatch",
        compares_prediction=prediction,
        compares_observation=late_observation,
        outcome=ComparisonOutcome.MISMATCH,
        has_error=[late_error],
        invokes_correction=[correction],
        produces_model_update=[update],
    )
    return SensorimotorControlPoint(
        label="flow onset control point",
        at_event_type=flow_onset,
        has_prediction=[prediction],
        has_comparison=[matched, mismatch],
    )


def _build_predictive_control(
    roles: dict[str, RoleSlot],
    parameters: dict[str, Parameter],
    events: dict[str, object],
    phases: dict[str, object],
    transitions: list[PhaseTransition],
    models: dict[str, InternalModel],
    signals: dict[str, SensorySignalType],
    requirements: dict[str, ConditionRequirement],
) -> PredictiveControlFragments:
    """
    Build all discrete predictive-control scenarios.
    """
    control_points = [
        _contact_control_point(
            roles,
            parameters,
            events,
            phases,
            transitions,
            models,
            signals,
            requirements,
        ),
        _liftoff_control_point(
            roles,
            parameters,
            events,
            phases,
            transitions,
            models,
            signals,
            requirements,
        ),
        _flow_control_point(
            roles,
            parameters,
            events,
            phases,
            transitions,
            models,
            signals,
        ),
    ]
    return PredictiveControlFragments.from_control_points(control_points)


# %% force coupling and continuous predictions


def _build_force_coupling(
    parameters: dict[str, Parameter],
    phases: dict[str, object],
    predictions: list[SensoryEventPrediction],
) -> GripLoadCoupling:
    """Build load-, friction-, margin-, and timing-dependent grip scaling."""
    liftoff_timing = next(
        prediction.expected_timing
        for prediction in predictions
        if isinstance(prediction.predicts_event, Liftoff)
    )
    return GripLoadCoupling(
        label="functional grip-load coupling",
        predicted_load=cast(LoadForceEstimate, parameters["object_load_estimate"]),
        friction_estimate=cast(
            ExpectedFrictionCoefficient,
            parameters["expected_friction_coefficient"],
        ),
        safety_margin=cast(SafetyMargin, parameters["safety_margin"]),
        grip_force_response=cast(GripForceSetpoint, parameters["grip_force"]),
        temporal_coupling=cast(TemporalWindow, liftoff_timing),
        coupling_active_in=[phases[name] for name in _COUPLED_PHASE_NAMES],
    )


def _build_anticipatory_grip_adjustment(
    roles: dict[str, RoleSlot],
    parameters: dict[str, Parameter],
    events: dict[str, object],
    phases: dict[str, object],
    models: dict[str, InternalModel],
    signals: dict[str, SensorySignalType],
    timing: TemporalWindow,
) -> tuple[AnticipatoryGripAdjustment, SensoryEventPrediction]:
    """Build the touchdown prediction and its pre-impact grip adjustment."""
    touchdown = cast(Touchdown, events["touchdown"])
    prediction = SensoryEventPrediction(
        label="predicted tactile touchdown for anticipatory grip",
        modality=signals["tactile"],
        predicts_event=touchdown,
        caused_by_event=touchdown,
        expected_timing=timing,
        expected_on_role=[roles["source_container"], roles["final_support"]],
        generated_by=cast(ForwardModel, models["forward_model"]),
    )
    adjustment = AnticipatoryGripAdjustment(
        label="anticipatory grip increase before touchdown",
        triggered_by_prediction=prediction,
        targets_grip_force=cast(GripForceSetpoint, parameters["grip_force"]),
        direction=CorrectionDirection.INCREASE,
        adjustment_active_in=[cast(ActionPhase, phases["lower"])],
    )
    return adjustment, prediction


@dataclass
class ContinuousPredictionFragments:
    """Continuous predictions and their independently represented observations."""

    predictions: list[ContinuousSignalPrediction]
    """The continuously evaluated model predictions."""
    observations: list[SensoryObservationType]
    """The observation types compared with the predictions."""


def _build_continuous_predictions(
    roles: dict[str, RoleSlot],
    parameters: dict[str, Parameter],
    phases: dict[str, object],
    models: dict[str, InternalModel],
    signals: dict[str, SensorySignalType],
) -> ContinuousPredictionFragments:
    """Build structured flow-rate and tilt-angle comparisons."""
    forward_model = cast(ForwardModel, models["forward_model"])
    flow_rate = cast(FlowRateSetpoint, parameters["flow_rate"])
    flow_range = cast(MagnitudeRange, flow_rate.parameter_range)
    flow_timing = _timing(
        "flow regulation interval",
        0.0,
        10.0,
        MeasurementUnit.SECOND,
    )
    flow_feature = ConceptReference(
        iri=POUR + "ObservedFlowRate",
        label="observed flow rate",
    )
    flow_observation = SensoryObservationType(
        label="visual flow-rate observation",
        modality=signals["visual"],
        observes_feature=flow_feature,
        observed_at=flow_timing,
        observed_on_role=[
            roles["poured_substance"],
            roles["target_receptacle"],
        ],
        observed_magnitude=flow_range,
    )
    flow_tolerance_fraction = 0.2
    flow_prediction = ContinuousSignalPrediction(
        label="predicted visual flow-rate trajectory",
        signal=signals["visual"],
        predicted_trajectory=SignalTrajectory(
            label="stable flow-rate trajectory",
            signal_feature=flow_feature,
            trend=TrajectoryTrend.STABLE,
            expected_range=flow_range,
            expected_timing=flow_timing,
        ),
        tolerance=_magnitude(
            0.0,
            cast(float, flow_range.upper_bound) * flow_tolerance_fraction,
            cast(MeasurementUnit, flow_range.unit),
        ),
        generated_by=forward_model,
        comparison_target=flow_observation,
        controlled_variable=flow_rate,
        predicted_for_phase=[cast(ActionPhase, phases["pour"])],
    )

    tilt_angle = cast(TiltAngleSetpoint, parameters["tilt_angle"])
    tilt_range = cast(MagnitudeRange, tilt_angle.parameter_range)
    tilt_timing = _timing(
        "tilt regulation interval",
        0.0,
        10.0,
        MeasurementUnit.SECOND,
    )
    tilt_feature = ConceptReference(
        iri=POUR + "ObservedTiltAngle",
        label="observed tilt angle",
    )
    tilt_observation = SensoryObservationType(
        label="proprioceptive tilt-angle observation",
        modality=signals["proprioceptive"],
        observes_feature=tilt_feature,
        observed_at=tilt_timing,
        observed_on_role=[roles["source_container"]],
        observed_magnitude=tilt_range,
    )
    tilt_prediction = ContinuousSignalPrediction(
        label="predicted proprioceptive tilt-angle trajectory",
        signal=signals["proprioceptive"],
        predicted_trajectory=SignalTrajectory(
            label="controlled tilt-angle trajectory",
            signal_feature=tilt_feature,
            trend=TrajectoryTrend.STABLE,
            expected_range=tilt_range,
            expected_timing=tilt_timing,
        ),
        tolerance=_magnitude(0.0, 2.0, MeasurementUnit.DEGREE),
        generated_by=forward_model,
        comparison_target=tilt_observation,
        controlled_variable=tilt_angle,
        predicted_for_phase=[
            cast(ActionPhase, phases["tilt"]),
            cast(ActionPhase, phases["pour"]),
        ],
    )
    return ContinuousPredictionFragments(
        predictions=[flow_prediction, tilt_prediction],
        observations=[flow_observation, tilt_observation],
    )


# %% assembly
def _build_reference_pouring_structure(
    material_regime: MaterialRegime,
    task: ConceptReference,
) -> PouringPhaseStructure:
    """Build one material-specific reference pouring structure."""
    roles = _build_roles()
    parameters = _build_parameters(roles, material_regime)
    events = _build_event_types(roles, parameters)
    conditions = _build_condition_requirements(roles, events)
    internal_models = _build_internal_models(roles, parameters, conditions)
    models = internal_models.models
    signals = _build_sensory_signals()
    phases = _build_phases(roles, events, conditions, models, signals)
    _wire_parameter_phases(parameters, phases)
    transitions = _build_transitions(phases, events)
    predictive_control = _build_predictive_control(
        roles,
        parameters,
        events,
        phases,
        transitions,
        models,
        signals,
        conditions,
    )
    force_coupling = _build_force_coupling(
        parameters,
        phases,
        predictive_control.predictions,
    )
    anticipatory_adjustment, touchdown_prediction = _build_anticipatory_grip_adjustment(
        roles,
        parameters,
        events,
        phases,
        models,
        signals,
        cast(TemporalWindow, force_coupling.temporal_coupling),
    )
    predictive_control.predictions.append(touchdown_prediction)
    internal_models.connect_predictions(
        predictive_control.predictions,
        [
            cast(MechanicalEventType, events["liftoff"]),
            cast(MechanicalEventType, events["touchdown"]),
        ],
    )
    continuous_control = _build_continuous_predictions(
        roles,
        parameters,
        phases,
        models,
        signals,
    )

    return PouringPhaseStructure(
        label="ActionPhases",
        decomposes_task=task,
        material_regime=material_regime,
        termination_reason=TransferTerminationReason.TARGET_QUANTITY,
        defines_phase=list(phases.values()),
        defines_role=list(roles.values()),
        defines_event_type=list(events.values()),
        defines_parameter=list(parameters.values()),
        defines_internal_model=list(internal_models.models.values()),
        defines_desired_state=internal_models.desired_states,
        defines_motor_command=internal_models.motor_commands,
        defines_efference_copy=internal_models.efference_copies,
        has_internal_model_composition=internal_models.compositions,
        has_sensorimotor_memory=internal_models.memories,
        has_representation_update=internal_models.updates,
        defines_dynamics_perturbation=internal_models.perturbations,
        has_dynamics_transfer=internal_models.transfers,
        has_representation_dissociation=internal_models.dissociations,
        defines_prediction=predictive_control.predictions,
        defines_observation=[
            *predictive_control.observations,
            *continuous_control.observations,
        ],
        has_comparison=predictive_control.comparisons,
        defines_error=predictive_control.errors,
        defines_correction=predictive_control.corrections,
        has_model_update=predictive_control.model_updates,
        defines_initial_state_estimate=predictive_control.state_estimates,
        has_transition=transitions,
        has_control_point=predictive_control.control_points,
        has_force_coupling=[force_coupling],
        has_anticipatory_grip_adjustment=[anticipatory_adjustment],
        has_continuous_prediction=continuous_control.predictions,
    )


def build_pouring_phase_structure() -> PouringPhaseStructure:
    """Build the granular cereal-pouring reference structure."""
    return _build_reference_pouring_structure(
        material_regime=MaterialRegime.GRANULAR,
        task=ConceptReference(
            iri="urn:cram_viz:CerealPouringIntoBowl",
            label="Pour cereal from the cereal box into the bowl",
        ),
    )


def build_water_pouring_phase_structure() -> PouringPhaseStructure:
    """Build the liquid water-pouring reference structure."""
    return _build_reference_pouring_structure(
        material_regime=MaterialRegime.LIQUID,
        task=ConceptReference(
            iri="urn:cram_viz:WaterPouringIntoGlass",
            label="Pour water from a bottle into a glass",
        ),
    )
