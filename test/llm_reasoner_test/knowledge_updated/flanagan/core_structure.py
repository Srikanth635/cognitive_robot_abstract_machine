"""
A small but complete phase structure built only from core concepts.

It reaches for an object, grasps it, and holds it. Every kind of fragment the core
defines appears at least once, so it is the smallest structure that can show the
acceptance rules all hold together, and each rule is then tested by breaking one thing
in it.
"""

from __future__ import annotations

from dataclasses import dataclass

from llm_reasoner.knowledge_updated.microworlds.flanagan.alignments import SOMA
from llm_reasoner.knowledge_updated.microworlds.flanagan.core.conditions import (
    ConditionRequirement,
    PhaseConditionType,
    Subgoal,
)
from llm_reasoner.knowledge_updated.microworlds.flanagan.core.control import (
    BoundaryControlExemption,
    CorrectiveAction,
    InitialStateEstimate,
    PhaseTransition,
    PredictionComparison,
    PredictionError,
    SensorimotorControlPoint,
)
from llm_reasoner.knowledge_updated.microworlds.flanagan.core.events import (
    ContactEventType,
    SetpointAttained,
    ThresholdCrossingEventType,
)
from llm_reasoner.knowledge_updated.microworlds.flanagan.core.models import (
    ForwardModel,
)
from llm_reasoner.knowledge_updated.microworlds.flanagan.core.parameters import (
    MechanicalThreshold,
    MotorSetpoint,
    SafetyMargin,
)
from llm_reasoner.knowledge_updated.microworlds.flanagan.core.phases import ActionPhase
from llm_reasoner.knowledge_updated.microworlds.flanagan.core.prediction import (
    ContinuousSignalPrediction,
    SensoryEventPrediction,
    SensoryObservationType,
    SignalTrajectory,
)
from llm_reasoner.knowledge_updated.microworlds.flanagan.core.quantities import (
    MagnitudeRange,
    TemporalWindow,
)
from llm_reasoner.knowledge_updated.microworlds.flanagan.core.roles import (
    Actor,
    Manipulandum,
    RoleSlot,
)
from llm_reasoner.knowledge_updated.microworlds.flanagan.core.structure import (
    ActionPhaseStructure,
)
from llm_reasoner.knowledge_updated.microworlds.flanagan.core.value_partitions import (
    ComparisonOutcome,
    ConditionMode,
    ContactTransition,
    CorrectionDirection,
    CouplingRegime,
    MeasurementUnit,
    ParameterValueForm,
    PhysicalDimension,
    PredictionErrorKind,
    RecoveryMode,
    RegulationMode,
    SensoryModality,
    TrajectoryTrend,
)
from llm_reasoner.knowledge_updated.microworlds.flanagan.ontology import FLANAGAN
from llm_reasoner.knowledge_updated.ontological.entities import KnowledgeEntity
from llm_reasoner.knowledge_updated.ontological.properties import PropertyMeta
from llm_reasoner.knowledge_updated.ontological.references import ConceptReference
from llm_reasoner.knowledge_updated.ontological.vocabulary import PropertyKind

# %% concrete stand-ins for the extension points the core leaves abstract


@dataclass(eq=False)
class QuantitativeMotorSetpoint(MotorSetpoint):
    """
    A motor setpoint that holds a magnitude, standing in for the concrete ones that
    later modules declare.
    """


@dataclass(eq=False)
class QuantitativeThreshold(MechanicalThreshold):
    """
    A threshold that holds a magnitude, standing in for the concrete ones that later
    modules declare.
    """


# %% small builders


def magnitude(low: float, high: float, unit: MeasurementUnit) -> MagnitudeRange:
    """
    Return a range of a quantity.
    """
    return MagnitudeRange(lower_bound=low, upper_bound=high, unit=unit)


def window(earliest: float, latest: float) -> TemporalWindow:
    """
    Return a window of time, in seconds.
    """
    return TemporalWindow(
        earliest_offset=earliest, latest_offset=latest, unit=MeasurementUnit.SECOND
    )


def role_of(role_class: type, label: str) -> RoleSlot:
    """
    Return an unbound slot for a kind of role.
    """
    return RoleSlot(
        label=label,
        binds_as=ConceptReference(iri=FLANAGAN.find_concept(role_class).iri),
    )


FEATURE = ConceptReference(iri="urn:test:feature/normal-force")
"""
A sensory feature, named by a term of some other vocabulary.
"""

TASK = ConceptReference(iri=SOMA.term("Grasping"))
"""
The task the structure decomposes.
"""

SCOPE = ConceptReference(iri="urn:test:justification/illustrative-scope")
"""
Why a boundary has no control point.
"""


# %% the structure


@dataclass
class CoreStructure:
    """
    The structure and every named part of it, so a test can break one thing.
    """

    structure: ActionPhaseStructure
    actor: RoleSlot
    manipulandum: RoleSlot
    distance: QuantitativeMotorSetpoint
    grip: QuantitativeMotorSetpoint
    threshold: QuantitativeThreshold
    margin: SafetyMargin
    arrived: SetpointAttained
    contact: ContactEventType
    held: SetpointAttained
    slip: ThresholdCrossingEventType
    positioned: PhaseConditionType
    touching: PhaseConditionType
    stable_grip: PhaseConditionType
    reach: ActionPhase
    grasp: ActionPhase
    hold: ActionPhase
    forward_model: ForwardModel
    continuous: ContinuousSignalPrediction
    continuous_observation: SensoryObservationType
    prediction: SensoryEventPrediction
    observation: SensoryObservationType
    absent: PredictionError
    correction: CorrectiveAction
    state_estimate: InitialStateEstimate
    match: PredictionComparison
    mismatch: PredictionComparison
    control_point: SensorimotorControlPoint
    reach_to_grasp: PhaseTransition
    grasp_to_hold: PhaseTransition
    arrived_exemption: BoundaryControlExemption
    held_exemption: BoundaryControlExemption

    def fragments(self) -> list[KnowledgeEntity]:
        """
        Return the structure and every fragment reachable from it, nested ones included.
        """
        return fragments_of(self.structure)


def fragments_of(structure: KnowledgeEntity) -> list[KnowledgeEntity]:
    """
    Return a structure and every fragment reachable from it, nested ones included.
    """
    seen: dict[int, KnowledgeEntity] = {id(structure): structure}
    frontier: list[KnowledgeEntity] = [structure]
    while frontier:
        current = frontier.pop()
        for name, meta in PropertyMeta.declared_on(type(current)).items():
            if meta.kind is not PropertyKind.RELATION:
                continue
            value = current.__dict__[name]
            for member in value if isinstance(value, list) else [value]:
                if isinstance(member, KnowledgeEntity) and id(member) not in seen:
                    seen[id(member)] = member
                    frontier.append(member)
    return list(seen.values())


def build_valid_structure() -> CoreStructure:
    """
    Build a fresh structure that satisfies every acceptance rule.
    """
    actor = role_of(Actor, "actor")
    manipulandum = role_of(Manipulandum, "manipulandum")

    def both() -> list[RoleSlot]:
        return [actor, manipulandum]

    distance = QuantitativeMotorSetpoint(
        label="approach distance",
        value_form=ParameterValueForm.QUANTITATIVE,
        dimension=PhysicalDimension.LENGTH,
        parameter_range=magnitude(0.0, 0.5, MeasurementUnit.METRE),
        parameter_on_role=[actor],
    )
    grip = QuantitativeMotorSetpoint(
        label="grip force",
        value_form=ParameterValueForm.QUANTITATIVE,
        dimension=PhysicalDimension.FORCE,
        parameter_range=magnitude(2.0, 5.0, MeasurementUnit.NEWTON),
        parameter_on_role=both(),
    )
    threshold = QuantitativeThreshold(
        label="slip threshold",
        value_form=ParameterValueForm.QUANTITATIVE,
        dimension=PhysicalDimension.FORCE,
        parameter_range=magnitude(1.0, 3.0, MeasurementUnit.NEWTON),
        parameter_on_role=both(),
    )
    margin = SafetyMargin(
        label="grip margin",
        value_form=ParameterValueForm.QUANTITATIVE,
        dimension=PhysicalDimension.FORCE,
        parameter_range=magnitude(0.5, 1.0, MeasurementUnit.NEWTON),
        parameter_on_role=both(),
        margin_on=grip,
        margin_over=threshold,
    )

    arrived = SetpointAttained(
        label="arrived", event_on_role=[actor], attains_setpoint=[distance]
    )
    contact = ContactEventType(
        label="contact",
        event_on_role=both(),
        contact_transition=ContactTransition.ESTABLISHES,
    )
    held = SetpointAttained(label="held", event_on_role=both(), attains_setpoint=[grip])
    slip = ThresholdCrossingEventType(
        label="slip",
        is_failure_event=True,
        event_on_role=both(),
        crosses_threshold=threshold,
    )

    positioned = PhaseConditionType(label="positioned", condition_on_role=[actor])
    touching = PhaseConditionType(label="touching", condition_on_role=both())
    stable_grip = PhaseConditionType(
        label="stable grip", condition_on_role=both(), violation_signalled_by=slip
    )

    forward_model = ForwardModel(label="forward model")
    continuous_observation = SensoryObservationType(
        label="grip force observed",
        modality=SensoryModality.TACTILE,
        observes_feature=FEATURE,
        observed_at=window(0.0, 1.0),
        observed_on_role=both(),
    )
    continuous = ContinuousSignalPrediction(
        label="grip force stays steady",
        modality=SensoryModality.TACTILE,
        predicted_trajectory=SignalTrajectory(
            signal_feature=FEATURE,
            trend=TrajectoryTrend.STABLE,
            expected_range=magnitude(2.0, 5.0, MeasurementUnit.NEWTON),
        ),
        tolerance=magnitude(0.0, 0.5, MeasurementUnit.NEWTON),
        generated_by=forward_model,
        comparison_target=continuous_observation,
        controlled_variable=grip,
    )

    prediction = SensoryEventPrediction(
        label="contact felt",
        modality=SensoryModality.TACTILE,
        predicts_event=contact,
        caused_by_event=contact,
        expected_timing=window(0.0, 0.05),
        expected_on_role=both(),
        generated_by=forward_model,
    )
    observation = SensoryObservationType(
        label="contact observed",
        modality=SensoryModality.TACTILE,
        observes_event=contact,
        observed_at=window(0.0, 0.05),
        observed_on_role=both(),
    )

    def entry(condition, mode, **options):
        return ConditionRequirement(on_condition=condition, mode=mode, **options)

    reach = ActionPhase(
        label="reach",
        aims_for=Subgoal(
            label="reach subgoal",
            targets_condition=positioned,
            attainment_signalled_by=arrived,
        ),
        coupling_regime=CouplingRegime.UNCOUPLED,
        regulation_mode=RegulationMode.FEEDFORWARD,
        terminated_by=[arrived],
        phase_involves=[actor],
        active_parameters=[distance],
    )
    grasp = ActionPhase(
        label="grasp",
        aims_for=Subgoal(
            label="grasp subgoal",
            targets_condition=touching,
            attainment_signalled_by=contact,
        ),
        coupling_regime=CouplingRegime.COUPLED,
        regulation_mode=RegulationMode.CONTINUOUS_FEEDBACK,
        initiated_by=[arrived],
        terminated_by=[contact],
        phase_involves=both(),
        has_condition_requirement=[
            entry(positioned, ConditionMode.AT_ENTRY),
            entry(stable_grip, ConditionMode.THROUGHOUT),
            entry(touching, ConditionMode.AT_EXIT),
        ],
        monitors_modality=[SensoryModality.TACTILE],
        governed_by=[forward_model],
        active_parameters=[grip, threshold, margin],
        continuous_predictions=[continuous],
    )
    hold = ActionPhase(
        label="hold",
        aims_for=Subgoal(
            label="hold subgoal",
            targets_condition=stable_grip,
            attainment_signalled_by=held,
        ),
        coupling_regime=CouplingRegime.COUPLED,
        regulation_mode=RegulationMode.FEEDFORWARD,
        initiated_by=[contact],
        terminated_by=[held],
        phase_involves=both(),
        has_condition_requirement=[entry(stable_grip, ConditionMode.THROUGHOUT)],
        active_parameters=[grip],
    )

    absent = PredictionError(
        label="contact absent", error_kind=PredictionErrorKind.ABSENT
    )
    correction = CorrectiveAction(
        label="squeeze harder",
        triggered_by_error=[absent],
        targets_setpoint=[grip],
        correction_direction=CorrectionDirection.INCREASE,
        intended_effect=[touching],
        recovery_mode=RecoveryMode.REPEAT_PHASE,
    )
    state_estimate = InitialStateEstimate(
        label="contact made",
        for_phase=hold,
        based_on_observation=[observation],
        estimates_condition=[touching],
    )
    match = PredictionComparison(
        label="contact matches",
        compares_prediction=prediction,
        compares_observation=observation,
        comparison_outcome=ComparisonOutcome.MATCH,
        confirms_subgoal=grasp.aims_for,
        supplies_state_estimate=state_estimate,
    )
    mismatch = PredictionComparison(
        label="contact missing",
        compares_prediction=prediction,
        comparison_outcome=ComparisonOutcome.MISMATCH,
        has_error=[absent],
        invokes_correction=[correction],
    )
    control_point = SensorimotorControlPoint(
        label="contact control point",
        at_event_type=contact,
        has_prediction=[prediction],
        has_comparison=[match, mismatch],
    )

    reach_to_grasp = PhaseTransition(
        label="reach to grasp", from_phase=reach, to_phase=grasp, gated_by=[arrived]
    )
    grasp_to_hold = PhaseTransition(
        label="grasp to hold",
        from_phase=grasp,
        to_phase=hold,
        gated_by=[contact],
        permitted_by=[match],
    )
    arrived_exemption = BoundaryControlExemption(
        label="arrival exemption", exempts_event=arrived, justification=SCOPE
    )
    held_exemption = BoundaryControlExemption(
        label="hold exemption", exempts_event=held, justification=SCOPE
    )

    structure = ActionPhaseStructure(
        label="grasp an object",
        decomposes_task=TASK,
        defines_role=both(),
        defines_phase=[reach, grasp, hold],
        defines_event_type=[arrived, contact, held, slip],
        defines_transition=[reach_to_grasp, grasp_to_hold],
        defines_parameter=[distance, grip, threshold, margin],
        defines_internal_model=[forward_model],
        defines_prediction=[prediction],
        defines_observation=[observation, continuous_observation],
        defines_continuous_prediction=[continuous],
        defines_comparison=[match, mismatch],
        defines_error=[absent],
        defines_correction=[correction],
        defines_initial_state_estimate=[state_estimate],
        defines_control_point=[control_point],
        defines_control_point_exemption=[arrived_exemption, held_exemption],
    )
    return CoreStructure(
        structure=structure,
        actor=actor,
        manipulandum=manipulandum,
        distance=distance,
        grip=grip,
        threshold=threshold,
        margin=margin,
        arrived=arrived,
        contact=contact,
        held=held,
        slip=slip,
        positioned=positioned,
        touching=touching,
        stable_grip=stable_grip,
        reach=reach,
        grasp=grasp,
        hold=hold,
        forward_model=forward_model,
        continuous=continuous,
        continuous_observation=continuous_observation,
        prediction=prediction,
        observation=observation,
        absent=absent,
        correction=correction,
        state_estimate=state_estimate,
        match=match,
        mismatch=mismatch,
        control_point=control_point,
        reach_to_grasp=reach_to_grasp,
        grasp_to_hold=grasp_to_hold,
        arrived_exemption=arrived_exemption,
        held_exemption=held_exemption,
    )
