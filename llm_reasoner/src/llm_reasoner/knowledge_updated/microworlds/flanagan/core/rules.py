"""
What makes a core phase structure admissible.

Whatever the declarations already say is checked without being written here: required
counts, alternatives, fixed and allowed values, value types, reference roots, and that a
structure refers only to fragments it declares. The rules below are the relationships
between several objects that a declaration cannot state. Each names itself with a member
of :class:`CoreRule`, and the documentation of that member is the reason the rule exists.
"""

from __future__ import annotations

from enum import StrEnum

from llm_reasoner.knowledge_updated.microworlds.flanagan.core.conditions import (
    ConditionRequirement,
    Subgoal,
)
from llm_reasoner.knowledge_updated.microworlds.flanagan.core.control import (
    ModelUpdate,
    PhaseTransition,
    PredictionComparison,
    SensorimotorControlPoint,
)
from llm_reasoner.knowledge_updated.microworlds.flanagan.core.parameters import (
    Parameter,
    SafetyMargin,
)
from llm_reasoner.knowledge_updated.microworlds.flanagan.core.phases import ActionPhase
from llm_reasoner.knowledge_updated.microworlds.flanagan.core.prediction import (
    ContinuousSignalPrediction,
    SensoryEventPrediction,
)
from llm_reasoner.knowledge_updated.microworlds.flanagan.core.quantities import (
    MagnitudeRange,
    TemporalWindow,
)
from llm_reasoner.knowledge_updated.microworlds.flanagan.core.structure import (
    ActionPhaseStructure,
)
from llm_reasoner.knowledge_updated.microworlds.flanagan.core.value_partitions import (
    ComparisonOutcome,
    ConditionMode,
    ConditionPolarity,
    ParameterValueForm,
    PredictionErrorKind,
    RegulationMode,
)
from llm_reasoner.knowledge_updated.microworlds.flanagan.ontology import FLANAGAN
from llm_reasoner.knowledge_updated.ontological.acceptance import (
    AcceptanceRule,
    AcceptanceSuite,
    Demand,
    Disjoint,
    Distinct,
    DistinctAcross,
    FlagOnPath,
    Forbidden,
    GraphShape,
    Grounded,
    Inclusion,
    Ordered,
    SharedValue,
    Supply,
    identities,
)
from llm_reasoner.knowledge_updated.ontological.derived import DerivedChecks
from llm_reasoner.knowledge_updated.ontological.vocabulary import Severity

# %% the rules


class CoreRule(StrEnum):
    """
    A rule a core phase structure must satisfy.
    """

    # -- the phase state machine ------------------------------------------

    GATE_DOES_NOT_END_SOURCE_PHASE = "gate_does_not_end_source_phase"
    """
    A transition must fire on an event that actually ends the phase it leaves. Otherwise
    the sequence of phases and the sequence of events are two unrelated stories about the
    same action.
    """

    GATE_DOES_NOT_START_TARGET_PHASE = "gate_does_not_start_target_phase"
    """
    A transition into a phase that declares initiating events must be gated by one of
    them.
    """

    PHASE_STARTS_AND_ENDS_AT_ONE_EVENT = "phase_starts_and_ends_at_one_event"
    """
    A phase begun and ended by the same event has zero extent.
    """

    TRANSITION_IS_A_SELF_LOOP = "transition_is_a_self_loop"
    """
    A transition from a phase to itself is not a change of phase.
    """

    PHASES_INDISTINGUISHABLE = "phases_indistinguishable"
    """
    Two phases whose terminating events coincide cannot be told apart by the controller,
    which usually means they should be merged or one needs its own boundary event.
    """

    FAILURE_EVENT_GATES_TRANSITION = "failure_event_gates_transition"
    """
    A failure event is what the controller guards against and not what it waits for.
    Letting one gate a normal transition would make failing a legitimate way to finish.
    """

    STRUCTURE_NOT_ONE_CHAIN = "structure_not_one_chain"
    """
    A structure needs exactly one entry phase and no orphan. With two entries the
    controller cannot tell where to start, and an orphan is knowledge it can never use.
    """

    CONDITION_NEVER_ESTABLISHED = "condition_never_established"
    """
    A required condition that is not marked as a world assumption and that no phase
    brings about. Either an upstream phase is missing or the requirement should be marked
    as an assumption, so the resolver knows to check it against the world.
    """

    # -- subgoals and conditions ------------------------------------------

    SUBGOAL_NOT_SIGNALLED_BY_OWN_BOUNDARY = "subgoal_not_signalled_by_own_boundary"
    """
    A phase's subgoal must be signalled by one of that phase's own terminating events.
    Otherwise the terminating event is an incident and not a sensorimotor control point.
    """

    SUBGOAL_SIGNALLED_BY_FAILURE = "subgoal_signalled_by_failure"
    """
    Success cannot be evidenced by a failure event.
    """

    INVARIANT_WITHOUT_VIOLATION_EVENT = "invariant_without_violation_event"
    """
    A condition required throughout a phase with no event registering its violation is not
    a guard but a hope.
    """

    VIOLATION_EVENT_NOT_A_FAILURE = "violation_event_not_a_failure"
    """
    An event registering the loss of a condition is a failure event by definition.
    """

    SAFETY_MARGIN_ON_DIFFERENT_PARTICIPANTS = "safety_margin_on_different_participants"
    """
    A margin relating a setpoint and a threshold about different participants is not a
    margin about anything.
    """

    # -- quantities and parameters ----------------------------------------

    RANGE_BOUNDS_INVERTED = "range_bounds_inverted"
    """
    A range whose lower bound exceeds its upper bound is not a range.
    """

    WINDOW_BOUNDS_INVERTED = "window_bounds_inverted"
    """
    A window of time cannot finish before it begins.
    """

    PARAMETER_UNIT_DOES_NOT_MEASURE_ITS_DIMENSION = (
        "parameter_unit_does_not_measure_its_dimension"
    )
    """
    The unit of a parameter's range must measure the dimension the parameter is in. A mass
    cannot serve as a force threshold.
    """

    QUANTITATIVE_PARAMETER_WITHOUT_DIMENSION = (
        "quantitative_parameter_without_dimension"
    )
    """
    A quantitative parameter must say which dimension it is in, or its unit cannot be
    checked.
    """

    PARAMETER_VALUE_DOES_NOT_MATCH_ITS_FORM = "parameter_value_does_not_match_its_form"
    """
    A quantitative parameter holds a range and a categorical one holds a category.
    Categorical properties must not use fabricated numeric ranges.
    """

    # -- predictions ------------------------------------------------------

    PREDICTION_TOLERANCE_IN_OTHER_DIMENSION = "prediction_tolerance_in_other_dimension"
    """
    A tolerance cannot bound a magnitude in another physical dimension.
    """

    TRAJECTORY_TOLERANCE_IN_OTHER_DIMENSION = "trajectory_tolerance_in_other_dimension"
    """
    A tolerance cannot bound a trajectory in another physical dimension.
    """

    CONTINUOUS_CHANNELS_DIFFER = "continuous_channels_differ"
    """
    A continuous prediction and the observation it is compared with must use the same
    sensory channel.
    """

    CONTROLLED_VARIABLE_INACTIVE_IN_PHASE = "controlled_variable_inactive_in_phase"
    """
    A continuous prediction runs in a phase where the variable it adjusts must be active.
    """

    FEEDBACK_PHASE_MONITORS_NOTHING = "feedback_phase_monitors_nothing"
    """
    Feedback and hybrid regulation require at least one monitored sensory channel.
    Without one the phase has no control law.
    """

    CONTINUOUS_PHASE_HAS_NO_CONTROL_LAW = "continuous_phase_has_no_control_law"
    """
    A continuously regulated phase needs a continuous prediction over a channel it
    monitors, with a trajectory and a setpoint to adjust. A continuous controller without
    them regulates nothing.
    """

    # -- comparison and recovery ------------------------------------------

    MATCH_LACKS_SUCCESS_OUTPUT = "match_lacks_success_output"
    """
    A successful comparison must provide every output needed to advance control: the
    observation, the confirmed subgoal and the next phase's state.
    """

    MATCH_DECLARES_MISMATCH_OUTPUT = "match_declares_mismatch_output"
    """
    Match and mismatch consequences are mutually exclusive.
    """

    MISMATCH_LACKS_RESPONSE = "mismatch_lacks_response"
    """
    A mismatch without a classified error and a corrective response has no operational
    consequence.
    """

    MISMATCH_DECLARES_SUCCESS_OUTPUT = "mismatch_declares_success_output"
    """
    A failed comparison cannot simultaneously permit normal progress.
    """

    ABSENT_ERROR_WITH_OBSERVATION = "absent_error_with_observation"
    """
    An event cannot be both absent and observed in one comparison.
    """

    OBSERVED_ERROR_WITHOUT_OBSERVATION = "observed_error_without_observation"
    """
    Timing, magnitude, direction, spatial and unexpected-event errors need an observation
    to classify.
    """

    COMPARISON_EVIDENCE_INCOMPATIBLE = "comparison_evidence_incompatible"
    """
    A comparison needs consistent channel and participants. The event or feature may
    differ only for an explicitly classified unexpected observation.
    """

    TRANSITION_PERMISSION_INCOHERENT = "transition_permission_incoherent"
    """
    A comparison that permits a transition must be a match that confirms the subgoal of
    the phase left and supplies the state of the phase entered.
    """

    MATCH_PERMITS_NO_TRANSITION = "match_permits_no_transition"
    """
    A successful comparison exists to release progress, so some transition must name it.
    """

    MODEL_UPDATE_CHANGES_UNHELD_ESTIMATE = "model_update_changes_unheld_estimate"
    """
    An update must identify the representation that owns the estimate it changes.
    """

    # -- control points ---------------------------------------------------

    PREDICTION_NOT_CAUSED_AT_CONTROL_POINT_EVENT = (
        "prediction_not_caused_at_control_point_event"
    )
    """
    Every prediction held by a control point must be caused by that control point's
    mechanical event.
    """

    COMPARISON_OF_UNDECLARED_PREDICTION = "comparison_of_undeclared_prediction"
    """
    A control point can compare only predictions that it holds.
    """

    CORRECTION_FOR_UNCLASSIFIED_ERROR = "correction_for_unclassified_error"
    """
    A comparison can invoke corrections only for errors it classified.
    """

    UPDATE_FOR_UNCLASSIFIED_ERROR = "update_for_unclassified_error"
    """
    A comparison can update a model only from errors it classified.
    """

    CONTROL_POINT_HANDLES_ONE_OUTCOME = "control_point_handles_one_outcome"
    """
    A control point must define both normal progress and recovery.
    """

    MISMATCH_NAMES_NO_CORRECTION = "mismatch_names_no_correction"
    """
    A control point whose mismatch handling names no corrective response discards the
    diagnostic value of comparison.
    """

    BOUNDARY_WITHOUT_CONTROL_POINT_OR_EXEMPTION = (
        "boundary_without_control_point_or_exemption"
    )
    """
    Every successful boundary must either compare predicted with observed evidence or
    state why that comparison is intentionally omitted.
    """

    EXEMPTION_NOT_FOR_A_NORMAL_BOUNDARY = "exemption_not_for_a_normal_boundary"
    """
    An exemption may justify an omitted comparison only for a boundary that the structure
    actually uses to end a phase, and not for a failure event.
    """


# %% predicates that no declared kind of rule expresses


def feedback_regulated(phase: ActionPhase) -> bool:
    """
    Return whether commands within the phase are corrected from sensory signals.
    """
    return phase.regulation_mode in (
        RegulationMode.CONTINUOUS_FEEDBACK,
        RegulationMode.HYBRID,
    )


def phase_lacks_control_law(phase: ActionPhase) -> bool:
    """
    Return whether a feedback phase has no continuous prediction over a channel it monitors.
    """
    return feedback_regulated(phase) and not any(
        prediction.modality in phase.monitors_modality
        for prediction in phase.continuous_predictions
    )


def phase_adjusts_an_inactive_variable(phase: ActionPhase) -> bool:
    """
    Return whether a continuous prediction adjusts a setpoint the phase does not activate.
    """
    active = identities(phase.active_parameters)
    return any(
        prediction.controlled_variable is not None
        and id(prediction.controlled_variable) not in active
        for prediction in phase.continuous_predictions
    )


def parameter_unit_disagrees_with_dimension(parameter: Parameter) -> bool:
    """
    Return whether a parameter's range is measured in a dimension other than its own.
    """
    return (
        parameter.parameter_range is not None
        and parameter.dimension is not None
        and parameter.parameter_range.physical_dimension is not parameter.dimension
    )


def parameter_value_disagrees_with_form(parameter: Parameter) -> bool:
    """
    Return whether a parameter holds the kind of value its form does not call for.
    """
    if parameter.value_form is ParameterValueForm.QUANTITATIVE:
        return (
            parameter.parameter_range is None or parameter.categorical_value is not None
        )
    if parameter.value_form is ParameterValueForm.CATEGORICAL:
        return (
            parameter.categorical_value is None or parameter.parameter_range is not None
        )
    return False


def prediction_tolerance_disagrees(prediction: SensoryEventPrediction) -> bool:
    """
    Return whether a prediction's tolerance is in another dimension than its magnitude.
    """
    if prediction.expected_magnitude is None or prediction.tolerance is None:
        return False
    return (
        prediction.expected_magnitude.physical_dimension
        is not prediction.tolerance.physical_dimension
    )


def trajectory_tolerance_disagrees(prediction: ContinuousSignalPrediction) -> bool:
    """
    Return whether a continuous prediction's tolerance is in another dimension than its
    trajectory.
    """
    trajectory = prediction.predicted_trajectory
    if trajectory is None or trajectory.expected_range is None:
        return False
    if prediction.tolerance is None:
        return False
    return (
        trajectory.expected_range.physical_dimension
        is not prediction.tolerance.physical_dimension
    )


def continuous_channels_differ(prediction: ContinuousSignalPrediction) -> bool:
    """
    Return whether a continuous prediction uses another channel than its observation.
    """
    target = prediction.comparison_target
    return target is not None and prediction.modality != target.modality


def match_lacks_success_output(comparison: PredictionComparison) -> bool:
    """
    Return whether a match omits an output required to advance control.
    """
    return comparison.comparison_outcome is ComparisonOutcome.MATCH and (
        comparison.compares_observation is None
        or comparison.confirms_subgoal is None
        or comparison.supplies_state_estimate is None
    )


def match_declares_mismatch_output(comparison: PredictionComparison) -> bool:
    """
    Return whether a match also declares consequences that only a mismatch has.
    """
    return comparison.comparison_outcome is ComparisonOutcome.MATCH and bool(
        comparison.has_error
        or comparison.invokes_correction
        or comparison.produces_model_update
    )


def mismatch_lacks_response(comparison: PredictionComparison) -> bool:
    """
    Return whether a mismatch lacks a classified error or a corrective response.
    """
    return comparison.comparison_outcome is ComparisonOutcome.MISMATCH and (
        not comparison.has_error or not comparison.invokes_correction
    )


def mismatch_declares_success_output(comparison: PredictionComparison) -> bool:
    """
    Return whether a mismatch also claims outputs that only a match has.
    """
    return comparison.comparison_outcome is ComparisonOutcome.MISMATCH and bool(
        comparison.confirms_subgoal or comparison.supplies_state_estimate
    )


def absent_error_with_observation(comparison: PredictionComparison) -> bool:
    """
    Return whether a comparison classifies an event as absent yet holds an observation.
    """
    return comparison.compares_observation is not None and any(
        error.error_kind is PredictionErrorKind.ABSENT for error in comparison.has_error
    )


def observed_error_without_observation(comparison: PredictionComparison) -> bool:
    """
    Return whether a comparison classifies an error that needs an observation without one.
    """
    if comparison.compares_observation is not None or not comparison.has_error:
        return False
    return all(
        error.error_kind is not PredictionErrorKind.ABSENT
        for error in comparison.has_error
    )


def comparison_evidence_incompatible(comparison: PredictionComparison) -> bool:
    """
    Return whether the predicted and observed evidence describe different things.
    """
    prediction = comparison.compares_prediction
    observation = comparison.compares_observation
    if prediction is None or observation is None:
        return False
    if prediction.modality != observation.modality:
        return True
    if identities(prediction.expected_on_role) != identities(
        observation.observed_on_role
    ):
        return True
    if comparison.comparison_outcome is ComparisonOutcome.MISMATCH and any(
        error.error_kind is PredictionErrorKind.UNEXPECTED
        for error in comparison.has_error
    ):
        return False
    if prediction.predicts_event is not None:
        return observation.observes_event is not prediction.predicts_event
    if prediction.predicts_feature is not None:
        return observation.observes_feature != prediction.predicts_feature
    return True


def transition_permission_incoherent(transition: PhaseTransition) -> bool:
    """
    Return whether a comparison permitting the transition does not fit it.
    """
    for comparison in transition.permitted_by:
        estimate = comparison.supplies_state_estimate
        if comparison.comparison_outcome is not ComparisonOutcome.MATCH:
            return True
        if (
            transition.from_phase is None
            or comparison.confirms_subgoal is not transition.from_phase.aims_for
        ):
            return True
        if estimate is None or estimate.for_phase is not transition.to_phase:
            return True
    return False


def model_update_changes_unheld_estimate(update: ModelUpdate) -> bool:
    """
    Return whether an update changes an estimate its representation does not hold.
    """
    if update.updates_representation is None:
        return False
    held = identities(update.updates_representation.holds_estimate)
    return any(id(estimate) not in held for estimate in update.updates_estimate)


def control_point_lacks_an_outcome(control_point: SensorimotorControlPoint) -> bool:
    """
    Return whether a control point omits either match or mismatch handling.
    """
    outcomes = {
        comparison.comparison_outcome for comparison in control_point.has_comparison
    }
    return not {ComparisonOutcome.MATCH, ComparisonOutcome.MISMATCH} <= outcomes


def control_point_names_no_correction(control_point: SensorimotorControlPoint) -> bool:
    """
    Return whether no mismatch handled by a control point invokes a correction.
    """
    return not any(
        comparison.invokes_correction
        for comparison in control_point.has_comparison
        if comparison.comparison_outcome is ComparisonOutcome.MISMATCH
    )


def requirement_is_undetectable_invariant(requirement: ConditionRequirement) -> bool:
    """
    Return whether a throughout requirement has no event registering its violation.
    """
    return requirement.mode is ConditionMode.THROUGHOUT and (
        requirement.on_condition is None
        or requirement.on_condition.violation_signalled_by is None
    )


def structure_has_uncontrolled_boundary(structure: ActionPhaseStructure) -> bool:
    """
    Return whether a normal boundary has neither a control point nor an exemption.
    """
    controlled = identities(cp.at_event_type for cp in structure.defines_control_point)
    exempted = identities(
        exemption.exempts_event
        for exemption in structure.defines_control_point_exemption
    )
    return any(
        not event.is_failure_event
        and id(event) not in controlled
        and id(event) not in exempted
        for phase in structure.defines_phase
        for event in phase.terminated_by
    )


def structure_has_invalid_exemption(structure: ActionPhaseStructure) -> bool:
    """
    Return whether an exemption names something other than a normal phase boundary.
    """
    boundaries = identities(
        event for phase in structure.defines_phase for event in phase.terminated_by
    )
    return any(
        exemption.exempts_event is None
        or id(exemption.exempts_event) not in boundaries
        or exemption.exempts_event.is_failure_event
        for exemption in structure.defines_control_point_exemption
    )


def structure_has_unused_match(structure: ActionPhaseStructure) -> bool:
    """
    Return whether a successful comparison is named by no transition.
    """
    permitting = identities(
        comparison
        for transition in structure.defines_transition
        for comparison in transition.permitted_by
    )
    return any(
        comparison.comparison_outcome is ComparisonOutcome.MATCH
        and id(comparison) not in permitting
        for comparison in structure.defines_comparison
    )


def establishes_condition(requirement: ConditionRequirement) -> bool:
    """
    Return whether a requirement states an outcome its phase brings about.
    """
    return requirement.mode is ConditionMode.AT_EXIT


def presupposes_or_brings_about(requirement: ConditionRequirement) -> bool:
    """
    Return whether a requirement needs no other phase to supply its condition.
    """
    return (
        requirement.is_world_assumption
        or requirement.mode is ConditionMode.AT_EXIT
        or requirement.polarity is ConditionPolarity.MUST_NOT_HOLD
    )


# %% the rules


def _forbidden(
    rule: CoreRule, applies_to: type, violated_when, message: str, **options
):
    """
    Declare a rule that fires whenever a predicate holds.
    """
    return Forbidden(
        id=rule,
        applies_to=applies_to,
        violated_when=violated_when,
        message=message,
        **options,
    )


CORE_RULES: tuple[AcceptanceRule, ...] = (
    # the phase state machine
    Inclusion(
        id=CoreRule.GATE_DOES_NOT_END_SOURCE_PHASE,
        applies_to=PhaseTransition,
        source="gated_by",
        target="from_phase.terminated_by",
    ),
    Inclusion(
        id=CoreRule.GATE_DOES_NOT_START_TARGET_PHASE,
        applies_to=PhaseTransition,
        only_if=lambda t: t.to_phase is not None and bool(t.to_phase.initiated_by),
        source="gated_by",
        target="to_phase.initiated_by",
    ),
    Disjoint(
        id=CoreRule.PHASE_STARTS_AND_ENDS_AT_ONE_EVENT,
        applies_to=ActionPhase,
        left="initiated_by",
        right="terminated_by",
    ),
    Distinct(
        id=CoreRule.TRANSITION_IS_A_SELF_LOOP,
        applies_to=PhaseTransition,
        left="from_phase",
        right="to_phase",
    ),
    DistinctAcross(
        id=CoreRule.PHASES_INDISTINGUISHABLE,
        applies_to=ActionPhaseStructure,
        severity=Severity.WARNING,
        collection="defines_phase",
        key_path="terminated_by",
    ),
    FlagOnPath(
        id=CoreRule.FAILURE_EVENT_GATES_TRANSITION,
        applies_to=PhaseTransition,
        path="gated_by",
        flag="is_failure_event",
        expected=False,
    ),
    GraphShape(
        id=CoreRule.STRUCTURE_NOT_ONE_CHAIN,
        applies_to=ActionPhaseStructure,
        container="defines_phase",
        edges="defines_transition",
        tail="from_phase",
        head="to_phase",
    ),
    Grounded(
        id=CoreRule.CONDITION_NEVER_ESTABLISHED,
        applies_to=ActionPhaseStructure,
        severity=Severity.WARNING,
        container="defines_phase",
        demand=Demand(path="has_condition_requirement", target="on_condition"),
        supplies=(
            Supply(path="defines_phase.aims_for", target="targets_condition"),
            Supply(
                path="defines_phase.has_condition_requirement",
                target="on_condition",
                supplied_when=establishes_condition,
            ),
        ),
        excused_when=presupposes_or_brings_about,
    ),
    # subgoals and conditions
    Inclusion(
        id=CoreRule.SUBGOAL_NOT_SIGNALLED_BY_OWN_BOUNDARY,
        applies_to=ActionPhase,
        source="aims_for.attainment_signalled_by",
        target="terminated_by",
    ),
    FlagOnPath(
        id=CoreRule.SUBGOAL_SIGNALLED_BY_FAILURE,
        applies_to=Subgoal,
        path="attainment_signalled_by",
        flag="is_failure_event",
        expected=False,
    ),
    _forbidden(
        CoreRule.INVARIANT_WITHOUT_VIOLATION_EVENT,
        ConditionRequirement,
        requirement_is_undetectable_invariant,
        "required throughout a phase but nothing registers its violation",
    ),
    FlagOnPath(
        id=CoreRule.VIOLATION_EVENT_NOT_A_FAILURE,
        applies_to=ConditionRequirement,
        path="on_condition.violation_signalled_by",
        flag="is_failure_event",
        expected=True,
    ),
    SharedValue(
        id=CoreRule.SAFETY_MARGIN_ON_DIFFERENT_PARTICIPANTS,
        applies_to=SafetyMargin,
        severity=Severity.WARNING,
        left="margin_on.parameter_on_role",
        right="margin_over.parameter_on_role",
    ),
    # quantities and parameters
    Ordered(
        id=CoreRule.RANGE_BOUNDS_INVERTED,
        applies_to=MagnitudeRange,
        lower="lower_bound",
        upper="upper_bound",
    ),
    Ordered(
        id=CoreRule.WINDOW_BOUNDS_INVERTED,
        applies_to=TemporalWindow,
        lower="earliest_offset",
        upper="latest_offset",
    ),
    _forbidden(
        CoreRule.PARAMETER_UNIT_DOES_NOT_MEASURE_ITS_DIMENSION,
        Parameter,
        parameter_unit_disagrees_with_dimension,
        "the unit of the range does not measure the parameter's dimension",
    ),
    _forbidden(
        CoreRule.QUANTITATIVE_PARAMETER_WITHOUT_DIMENSION,
        Parameter,
        lambda p: p.value_form is ParameterValueForm.QUANTITATIVE
        and p.dimension is None,
        "a quantitative parameter does not say which dimension it is in",
    ),
    _forbidden(
        CoreRule.PARAMETER_VALUE_DOES_NOT_MATCH_ITS_FORM,
        Parameter,
        parameter_value_disagrees_with_form,
        "the value held is not the kind the parameter's form calls for",
    ),
    # predictions
    _forbidden(
        CoreRule.PREDICTION_TOLERANCE_IN_OTHER_DIMENSION,
        SensoryEventPrediction,
        prediction_tolerance_disagrees,
        "the tolerance and the magnitude are in different dimensions",
    ),
    _forbidden(
        CoreRule.TRAJECTORY_TOLERANCE_IN_OTHER_DIMENSION,
        ContinuousSignalPrediction,
        trajectory_tolerance_disagrees,
        "the tolerance and the trajectory are in different dimensions",
    ),
    _forbidden(
        CoreRule.CONTINUOUS_CHANNELS_DIFFER,
        ContinuousSignalPrediction,
        continuous_channels_differ,
        "the prediction and its observation use different sensory channels",
    ),
    _forbidden(
        CoreRule.CONTROLLED_VARIABLE_INACTIVE_IN_PHASE,
        ActionPhase,
        phase_adjusts_an_inactive_variable,
        "a continuous prediction adjusts a setpoint that is not active in this phase",
    ),
    _forbidden(
        CoreRule.FEEDBACK_PHASE_MONITORS_NOTHING,
        ActionPhase,
        lambda phase: feedback_regulated(phase) and not phase.monitors_modality,
        "feedback-regulated but monitors no channel, so it has no control law",
    ),
    _forbidden(
        CoreRule.CONTINUOUS_PHASE_HAS_NO_CONTROL_LAW,
        ActionPhase,
        phase_lacks_control_law,
        "no continuous prediction over a monitored channel closes the feedback loop",
    ),
    # comparison and recovery
    _forbidden(
        CoreRule.MATCH_LACKS_SUCCESS_OUTPUT,
        PredictionComparison,
        match_lacks_success_output,
        "a match lacks its observation, confirmed subgoal or next-phase state",
    ),
    _forbidden(
        CoreRule.MATCH_DECLARES_MISMATCH_OUTPUT,
        PredictionComparison,
        match_declares_mismatch_output,
        "a match also declares an error, correction or model update",
    ),
    _forbidden(
        CoreRule.MISMATCH_LACKS_RESPONSE,
        PredictionComparison,
        mismatch_lacks_response,
        "a mismatch lacks a classified error or a corrective response",
    ),
    _forbidden(
        CoreRule.MISMATCH_DECLARES_SUCCESS_OUTPUT,
        PredictionComparison,
        mismatch_declares_success_output,
        "a mismatch also confirms a subgoal or supplies a next-phase state",
    ),
    _forbidden(
        CoreRule.ABSENT_ERROR_WITH_OBSERVATION,
        PredictionComparison,
        absent_error_with_observation,
        "an absent-event error contradicts the observation the comparison holds",
    ),
    _forbidden(
        CoreRule.OBSERVED_ERROR_WITHOUT_OBSERVATION,
        PredictionComparison,
        observed_error_without_observation,
        "an error that needs an observation to classify has none",
    ),
    _forbidden(
        CoreRule.COMPARISON_EVIDENCE_INCOMPATIBLE,
        PredictionComparison,
        comparison_evidence_incompatible,
        "the prediction and the observation describe different evidence",
    ),
    _forbidden(
        CoreRule.TRANSITION_PERMISSION_INCOHERENT,
        PhaseTransition,
        transition_permission_incoherent,
        "a permitting comparison is not a match confirming the phase left and "
        "supplying the state of the phase entered",
    ),
    _forbidden(
        CoreRule.MATCH_PERMITS_NO_TRANSITION,
        ActionPhaseStructure,
        structure_has_unused_match,
        "a successful comparison is named by no transition",
    ),
    _forbidden(
        CoreRule.MODEL_UPDATE_CHANGES_UNHELD_ESTIMATE,
        ModelUpdate,
        model_update_changes_unheld_estimate,
        "the update changes an estimate its representation does not hold",
    ),
    # control points
    Inclusion(
        id=CoreRule.PREDICTION_NOT_CAUSED_AT_CONTROL_POINT_EVENT,
        applies_to=SensorimotorControlPoint,
        source="has_prediction.caused_by_event",
        target="at_event_type",
    ),
    Inclusion(
        id=CoreRule.COMPARISON_OF_UNDECLARED_PREDICTION,
        applies_to=SensorimotorControlPoint,
        source="has_comparison.compares_prediction",
        target="has_prediction",
    ),
    Inclusion(
        id=CoreRule.CORRECTION_FOR_UNCLASSIFIED_ERROR,
        applies_to=PredictionComparison,
        source="invokes_correction.triggered_by_error",
        target="has_error",
    ),
    Inclusion(
        id=CoreRule.UPDATE_FOR_UNCLASSIFIED_ERROR,
        applies_to=PredictionComparison,
        source="produces_model_update.triggered_by_error",
        target="has_error",
    ),
    _forbidden(
        CoreRule.CONTROL_POINT_HANDLES_ONE_OUTCOME,
        SensorimotorControlPoint,
        control_point_lacks_an_outcome,
        "the control point does not define both a match and a mismatch",
    ),
    _forbidden(
        CoreRule.MISMATCH_NAMES_NO_CORRECTION,
        SensorimotorControlPoint,
        control_point_names_no_correction,
        "no mismatch handled here invokes a corrective response",
        severity=Severity.WARNING,
    ),
    _forbidden(
        CoreRule.BOUNDARY_WITHOUT_CONTROL_POINT_OR_EXEMPTION,
        ActionPhaseStructure,
        structure_has_uncontrolled_boundary,
        "a normal phase boundary has neither a control point nor an exemption",
    ),
    _forbidden(
        CoreRule.EXEMPTION_NOT_FOR_A_NORMAL_BOUNDARY,
        ActionPhaseStructure,
        structure_has_invalid_exemption,
        "an exemption does not name a normal boundary that a phase uses",
    ),
)
"""
Every rule of the core, in the order of the enumeration that names them.
"""

CORE_ACCEPTANCE = AcceptanceSuite(
    name="flanagan-core",
    ontology=FLANAGAN,
    rules=CORE_RULES,
    checks=(DerivedChecks(FLANAGAN),),
)
"""
The checks that decide whether a core phase structure is admissible.
"""
