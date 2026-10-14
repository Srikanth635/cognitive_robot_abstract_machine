"""
A lift-and-replace phase structure built only from prehension concepts.

The effector reaches, makes contact, loads its grip, lifts, transports, lowers the
object onto a destination, unloads, releases and withdraws. Every phase is a specific
prehension phase, and the estimates and setpoints of the grip-load control are held by
represented dynamics, so it is the smallest structure that shows every rule of the
module holding at once.
"""

from __future__ import annotations

from dataclasses import dataclass

from llm_reasoner.knowledge_updated.microworlds.flanagan.core.conditions import (
    ConditionRequirement,
    Subgoal,
)
from llm_reasoner.knowledge_updated.microworlds.flanagan.core.control import (
    BoundaryControlExemption,
    PhaseTransition,
)
from llm_reasoner.knowledge_updated.microworlds.flanagan.core.events import (
    PhaseBoundaryEventType,
)
from llm_reasoner.knowledge_updated.microworlds.flanagan.core.models import (
    ForwardModel,
    InverseModel,
)
from llm_reasoner.knowledge_updated.microworlds.flanagan.core.parameters import (
    Parameter,
    SafetyMargin,
)
from llm_reasoner.knowledge_updated.microworlds.flanagan.core.phases import ActionPhase
from llm_reasoner.knowledge_updated.microworlds.flanagan.core.prediction import (
    SensoryEventPrediction,
)
from llm_reasoner.knowledge_updated.microworlds.flanagan.core.quantities import (
    CategoricalValue,
)
from llm_reasoner.knowledge_updated.microworlds.flanagan.core.roles import RoleSlot
from llm_reasoner.knowledge_updated.microworlds.flanagan.core.value_partitions import (
    ConditionMode,
    ContactTransition,
    CorrectionDirection,
    CouplingRegime,
    MeasurementUnit,
    ParameterValueForm,
    PhysicalDimension,
    RegulationMode,
    SensoryModality,
)
from llm_reasoner.knowledge_updated.microworlds.flanagan.object_dynamics.commands import (
    DesiredState,
    EfferenceCopy,
    MotorCommand,
)
from llm_reasoner.knowledge_updated.microworlds.flanagan.object_dynamics.representations import (
    LoadForceEstimate,
    ObjectDynamicsModel,
)
from llm_reasoner.knowledge_updated.microworlds.flanagan.object_dynamics.value_partitions import (
    CoordinateFrame,
    RepresentationPurpose,
)
from llm_reasoner.knowledge_updated.microworlds.flanagan.prehension.conditions import (
    EffectorObjectContact,
    EffectorPositioned,
    GraspStability,
    ObjectClearOfSupport,
    ObjectSupportContact,
)
from llm_reasoner.knowledge_updated.microworlds.flanagan.prehension.control import (
    AnticipatoryGripAdjustment,
    GripLoadCoupling,
)
from llm_reasoner.knowledge_updated.microworlds.flanagan.prehension.events import (
    ContactBreak,
    ContactEstablishment,
    ForcesUnloaded,
    LiftClearanceReached,
    Liftoff,
    ReachTargetReached,
    SlipEvent,
    Touchdown,
    TransportTargetReached,
    WithdrawalComplete,
)
from llm_reasoner.knowledge_updated.microworlds.flanagan.prehension.parameters import (
    ExpectedCenterOfMassPosition,
    ExpectedFrictionCoefficient,
    ExpectedMass,
    ExpectedShape,
    GripForceSetpoint,
    LiftClearanceSetpoint,
    LiftoffThreshold,
    LoadForceSetpoint,
    ReachPoseSetpoint,
    SlipThreshold,
    TouchdownThreshold,
    TransportPoseSetpoint,
    WithdrawalDistanceSetpoint,
)
from llm_reasoner.knowledge_updated.microworlds.flanagan.prehension.phases import (
    ContactEstablishmentPhase,
    ContactReleasePhase,
    ForceLoadingPhase,
    LiftPhase,
    ReachPhase,
    ReplacePhase,
    TransportPhase,
    UnloadPhase,
    WithdrawPhase,
)
from llm_reasoner.knowledge_updated.microworlds.flanagan.prehension.roles import (
    FinalSupport,
    GraspingEffector,
    InitialSupport,
    PrehensileManipulandum,
)
from llm_reasoner.knowledge_updated.microworlds.flanagan.prehension.structure import (
    PrehensionPhaseStructure,
)
from llm_reasoner.knowledge_updated.ontological.entities import KnowledgeEntity
from llm_reasoner.knowledge_updated.ontological.references import ConceptReference

from ..core_structure import SCOPE, TASK, fragments_of, magnitude, role_of, window

# %% small builders


def force(cls: type, label: str, low: float, high: float, roles: list[RoleSlot]):
    """
    Return a parameter that is a range of force.
    """
    return cls(
        label=label,
        value_form=ParameterValueForm.QUANTITATIVE,
        dimension=PhysicalDimension.FORCE,
        parameter_range=magnitude(low, high, MeasurementUnit.NEWTON),
        parameter_on_role=roles,
    )


def length(cls: type, label: str, low: float, high: float, roles: list[RoleSlot]):
    """
    Return a parameter that is a range of length.
    """
    return cls(
        label=label,
        value_form=ParameterValueForm.QUANTITATIVE,
        dimension=PhysicalDimension.LENGTH,
        parameter_range=magnitude(low, high, MeasurementUnit.METRE),
        parameter_on_role=roles,
    )


def category(cls: type, label: str, term: str, roles: list[RoleSlot]):
    """
    Return a parameter that names a category.
    """
    return cls(
        label=label,
        value_form=ParameterValueForm.CATEGORICAL,
        categorical_value=CategoricalValue(
            category=ConceptReference(iri=f"urn:test:category/{term}")
        ),
        parameter_on_role=roles,
    )


def make_phase(
    cls: type,
    label: str,
    condition,
    event: PhaseBoundaryEventType,
    coupling: CouplingRegime,
    parameters: list[Parameter],
    roles: list[RoleSlot],
    initiated: list[PhaseBoundaryEventType],
    regulation: RegulationMode = RegulationMode.FEEDFORWARD,
) -> ActionPhase:
    """
    Return a phase of a kind that aims for a condition and ends at one event.
    """
    return cls(
        label=label,
        aims_for=Subgoal(
            label=f"{label} subgoal",
            targets_condition=condition,
            attainment_signalled_by=event,
        ),
        coupling_regime=coupling,
        regulation_mode=regulation,
        initiated_by=initiated,
        terminated_by=[event],
        phase_involves=roles,
        active_parameters=parameters,
    )


# %% the structure


@dataclass
class PrehensionParts:
    """
    The structure and every named part of it, so a test can break one thing.
    """

    structure: PrehensionPhaseStructure
    effector: RoleSlot
    held: RoleSlot
    destination: RoleSlot
    grip: GripForceSetpoint
    load: LoadForceSetpoint
    margin: SafetyMargin
    friction: ExpectedFrictionCoefficient
    load_force: LoadForceEstimate
    reached: ReachTargetReached
    lifted_off: Liftoff
    touched_down: Touchdown
    unloaded: ForcesUnloaded
    released: ContactBreak
    withdrawn: WithdrawalComplete
    reach: ReachPhase
    lift: LiftPhase
    replace: ReplacePhase
    unload: UnloadPhase
    release: ContactReleasePhase
    withdraw: WithdrawPhase
    prediction: SensoryEventPrediction
    coupling: GripLoadCoupling
    adjustment: AnticipatoryGripAdjustment

    def fragments(self) -> list[KnowledgeEntity]:
        """
        Return the structure and every fragment reachable from it, nested ones included.
        """
        return fragments_of(self.structure)


def build_valid_prehension() -> PrehensionParts:
    """
    Build a fresh structure that satisfies every acceptance rule of the core, of object
    dynamics and of prehension.
    """
    effector = role_of(GraspingEffector, "effector")
    held = role_of(PrehensileManipulandum, "held object")
    source = role_of(InitialSupport, "source support")
    destination = role_of(FinalSupport, "destination support")
    pair = [effector, held]

    reach_pose = category(ReachPoseSetpoint, "pregrasp pose", "pregrasp", [effector])
    grip = force(GripForceSetpoint, "grip force", 2.0, 5.0, pair)
    load = force(LoadForceSetpoint, "load force", 1.0, 4.0, pair)
    clearance = length(LiftClearanceSetpoint, "lift clearance", 0.02, 0.05, [held])
    transport_pose = category(TransportPoseSetpoint, "goal pose", "goal", [held])
    withdrawal = length(WithdrawalDistanceSetpoint, "withdrawal", 0.05, 0.1, [effector])
    slip_threshold = force(SlipThreshold, "slip threshold", 1.0, 3.0, pair)
    liftoff_threshold = force(LiftoffThreshold, "liftoff threshold", 1.0, 4.0, pair)
    touchdown_threshold = force(
        TouchdownThreshold, "touchdown threshold", 0.5, 1.5, [held]
    )
    margin = SafetyMargin(
        label="grip margin",
        value_form=ParameterValueForm.QUANTITATIVE,
        dimension=PhysicalDimension.FORCE,
        parameter_range=magnitude(0.5, 1.0, MeasurementUnit.NEWTON),
        parameter_on_role=pair,
        margin_on=grip,
        margin_over=slip_threshold,
    )

    mass = ExpectedMass(
        label="expected mass",
        value_form=ParameterValueForm.QUANTITATIVE,
        dimension=PhysicalDimension.MASS,
        parameter_range=magnitude(0.1, 0.5, MeasurementUnit.KILOGRAM),
        parameter_on_role=[held],
    )
    friction = ExpectedFrictionCoefficient(
        label="expected friction",
        value_form=ParameterValueForm.QUANTITATIVE,
        dimension=PhysicalDimension.DIMENSIONLESS,
        parameter_range=magnitude(0.4, 0.8, MeasurementUnit.DIMENSIONLESS),
        parameter_on_role=pair,
    )
    centre = length(
        ExpectedCenterOfMassPosition, "expected centre of mass", 0.0, 0.1, [held]
    )
    shape = category(ExpectedShape, "expected shape", "cylinder", [held])
    load_force = force(LoadForceEstimate, "expected load force", 1.0, 4.0, [held])

    reached = ReachTargetReached(
        label="reached", event_on_role=[effector], attains_setpoint=[reach_pose]
    )
    contacted = ContactEstablishment(
        label="contacted",
        event_on_role=pair,
        contact_transition=ContactTransition.ESTABLISHES,
    )
    lifted_off = Liftoff(
        label="lifted off",
        event_on_role=[held, source],
        contact_transition=ContactTransition.BREAKS,
        crosses_threshold=liftoff_threshold,
    )
    cleared = LiftClearanceReached(
        label="cleared", event_on_role=[held], attains_setpoint=[clearance]
    )
    arrived = TransportTargetReached(
        label="arrived", event_on_role=[held], attains_setpoint=[transport_pose]
    )
    touched_down = Touchdown(
        label="touched down",
        event_on_role=[held, destination],
        contact_transition=ContactTransition.ESTABLISHES,
        crosses_threshold=touchdown_threshold,
    )
    unloaded = ForcesUnloaded(
        label="forces unloaded",
        event_on_role=[held],
        attains_setpoint=[grip, load],
        weight_supported_by=[destination],
    )
    released = ContactBreak(
        label="released",
        event_on_role=pair,
        contact_transition=ContactTransition.BREAKS,
    )
    withdrawn = WithdrawalComplete(
        label="withdrawn", event_on_role=[effector], attains_setpoint=[withdrawal]
    )
    slipped = SlipEvent(
        label="slipped",
        is_failure_event=True,
        event_on_role=pair,
        crosses_threshold=slip_threshold,
    )

    positioned = EffectorPositioned(label="positioned", condition_on_role=[effector])
    touching = EffectorObjectContact(label="touching", condition_on_role=pair)
    clear = ObjectClearOfSupport(
        label="clear of support", condition_on_role=[held, source]
    )
    stable = GraspStability(
        label="stable grasp", condition_on_role=pair, violation_signalled_by=slipped
    )
    supported = ObjectSupportContact(
        label="supported", condition_on_role=[held, destination]
    )

    uncoupled, transitional, coupled = (
        CouplingRegime.UNCOUPLED,
        CouplingRegime.TRANSITIONAL,
        CouplingRegime.COUPLED,
    )
    reach = make_phase(
        ReachPhase,
        "reach",
        positioned,
        reached,
        uncoupled,
        [reach_pose],
        [effector],
        [],
    )
    establish = make_phase(
        ContactEstablishmentPhase,
        "establish contact",
        touching,
        contacted,
        transitional,
        [grip],
        pair,
        [reached],
    )
    loading = make_phase(
        ForceLoadingPhase,
        "load",
        clear,
        lifted_off,
        transitional,
        [grip, load, liftoff_threshold, slip_threshold, margin],
        pair,
        [contacted],
    )
    lift = make_phase(
        LiftPhase,
        "lift",
        clear,
        cleared,
        coupled,
        [grip, clearance],
        pair,
        [lifted_off],
    )
    lift.has_condition_requirement = [
        ConditionRequirement(on_condition=stable, mode=ConditionMode.THROUGHOUT)
    ]
    transport = make_phase(
        TransportPhase,
        "transport",
        stable,
        arrived,
        coupled,
        [grip, transport_pose],
        pair,
        [cleared],
    )
    replace = make_phase(
        ReplacePhase,
        "replace",
        supported,
        touched_down,
        coupled,
        [grip, touchdown_threshold],
        pair,
        [arrived],
    )
    unload = make_phase(
        UnloadPhase,
        "unload",
        supported,
        unloaded,
        coupled,
        [grip, load],
        pair,
        [touched_down],
    )
    release = make_phase(
        ContactReleasePhase,
        "release",
        touching,
        released,
        transitional,
        [grip],
        pair,
        [unloaded],
    )
    withdraw = make_phase(
        WithdrawPhase,
        "withdraw",
        positioned,
        withdrawn,
        uncoupled,
        [withdrawal],
        [effector],
        [released],
    )
    phases = [
        reach,
        establish,
        loading,
        lift,
        transport,
        replace,
        unload,
        release,
        withdraw,
    ]
    events: list[PhaseBoundaryEventType] = [
        reached,
        contacted,
        lifted_off,
        cleared,
        arrived,
        touched_down,
        unloaded,
        released,
        withdrawn,
    ]
    transitions = [
        PhaseTransition(
            label=f"{earlier.label} to {later.label}",
            from_phase=earlier,
            to_phase=later,
            gated_by=[later.initiated_by[0]],
        )
        for earlier, later in zip(phases, phases[1:])
    ]

    forward_model = ForwardModel(label="forward model")
    inverse_model = InverseModel(label="grip inverse model")
    desired_state = DesiredState(label="held firmly", desired_condition=[touching])
    command = MotorCommand(
        label="squeeze",
        pursues_state=desired_state,
        generated_by=inverse_model,
        commands_setpoint=[grip],
    )
    copy = EfferenceCopy(label="squeeze copy", copies_command=command)
    dynamics = ObjectDynamicsModel(
        label="object dynamics",
        holds_estimate=[mass, friction, centre, shape, load_force],
        coordinate_frame=CoordinateFrame.OBJECT_CENTERED,
        purpose=RepresentationPurpose.SENSORIMOTOR_CONTROL,
        represents_role=[held, effector],
        uses_efference_copy=[copy],
    )
    prediction = SensoryEventPrediction(
        label="touchdown felt",
        modality=SensoryModality.TACTILE,
        predicts_event=touched_down,
        caused_by_event=touched_down,
        expected_timing=window(0.0, 0.05),
        expected_on_role=[held, destination],
        generated_by=forward_model,
    )
    coupling = GripLoadCoupling(
        label="grip follows load",
        predicted_load=load_force,
        friction_estimate=friction,
        safety_margin=margin,
        grip_force_response=grip,
        temporal_coupling=window(0.0, 0.1),
        coupling_active_in=[lift, transport],
    )
    adjustment = AnticipatoryGripAdjustment(
        label="brace for touchdown",
        triggered_by_prediction=prediction,
        targets_grip_force=grip,
        adjustment_direction=CorrectionDirection.INCREASE,
        adjustment_active_in=[replace],
    )

    parameters: list[Parameter] = [
        reach_pose,
        grip,
        load,
        clearance,
        transport_pose,
        withdrawal,
        slip_threshold,
        liftoff_threshold,
        touchdown_threshold,
        margin,
        mass,
        friction,
        centre,
        shape,
        load_force,
    ]
    structure = PrehensionPhaseStructure(
        label="lift and replace",
        decomposes_task=TASK,
        defines_role=[effector, held, source, destination],
        defines_phase=phases,
        defines_event_type=[*events, slipped],
        defines_transition=transitions,
        defines_parameter=parameters,
        defines_internal_model=[forward_model, inverse_model, dynamics],
        defines_prediction=[prediction],
        defines_control_point_exemption=[
            BoundaryControlExemption(
                label=f"{event.label} exemption",
                exempts_event=event,
                justification=SCOPE,
            )
            for event in events
        ],
        defines_desired_state=[desired_state],
        defines_motor_command=[command],
        defines_efference_copy=[copy],
        defines_force_coupling=[coupling],
        defines_anticipatory_grip_adjustment=[adjustment],
    )
    return PrehensionParts(
        structure=structure,
        effector=effector,
        held=held,
        destination=destination,
        grip=grip,
        load=load,
        margin=margin,
        friction=friction,
        load_force=load_force,
        reached=reached,
        lifted_off=lifted_off,
        touched_down=touched_down,
        unloaded=unloaded,
        released=released,
        withdrawn=withdrawn,
        reach=reach,
        lift=lift,
        replace=replace,
        unload=unload,
        release=release,
        withdraw=withdraw,
        prediction=prediction,
        coupling=coupling,
        adjustment=adjustment,
    )
