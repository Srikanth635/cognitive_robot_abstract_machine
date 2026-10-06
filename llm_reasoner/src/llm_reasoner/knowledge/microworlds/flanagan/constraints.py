"""
The Flanagan acceptance suite: what makes a generated hypothesis admissible.

Nothing here is a hand-written predicate. Each entry declares a constraint kind
implemented once in ``ontological.constraints``, and the required-content half of the
suite is not declared at all -- it is derived from the ``min_count`` values already
stated on the model's fields, so the two cannot drift.

The check identifiers match the original OWL-era specification, so that a failure here
can be traced back to the axiom or rationale that motivated it. Everything the
specification enforced through cardinality restrictions now arrives as REQ; everything
it could not express -- comparisons of two values, referential closure, graph shape, the
coherence between a subgoal and the event that signals it -- is declared below.
"""

from __future__ import annotations

from llm_reasoner.knowledge.microworlds.flanagan.core import (
    ActionPhase,
    ActionPhaseStructure,
    ComparisonOutcome,
    ConditionMode,
    ConditionPolarity,
    ConditionRequirement,
    ContinuousSignalPrediction,
    MagnitudeRange,
    ManipulationRole,
    MeasurementUnit,
    ModelUpdate,
    Parameter,
    ParameterValueForm,
    PhaseConditionType,
    PhaseTransition,
    PhysicalDimension,
    PredictionComparison,
    PredictionErrorKind,
    RegulationMode,
    RoleSlot,
    SafetyMargin,
    ScientificAssertion,
    ScientificStatus,
    SensorimotorControlPoint,
    SensoryEventPrediction,
    SensoryObservationType,
    Subgoal,
    TemporalWindow,
)
from llm_reasoner.knowledge.microworlds.flanagan.gaze_constraints import (
    GAZE_CONSTRAINTS,
)
from llm_reasoner.knowledge.microworlds.flanagan.learning_constraints import (
    LEARNING_CONSTRAINTS,
)
from llm_reasoner.knowledge.microworlds.flanagan.object_dynamics_constraints import (
    OBJECT_DYNAMICS_CONSTRAINTS,
)
from llm_reasoner.knowledge.microworlds.flanagan.observation_constraints import (
    OBSERVATION_CONSTRAINTS,
)
from llm_reasoner.knowledge.microworlds.flanagan.perception_constraints import (
    PERCEPTION_CONSTRAINTS,
)
from llm_reasoner.knowledge.microworlds.flanagan.pouring_constraints import (
    POURING_CONSTRAINTS,
)
from llm_reasoner.knowledge.microworlds.flanagan.prehension_constraints import (
    PREHENSION_CONSTRAINTS,
)
from llm_reasoner.knowledge.ontological.constraints import (
    Closure,
    ConceptReferenceType,
    ConstraintSuite,
    Disjoint,
    Distinct,
    DistinctAcross,
    ExactlyOne,
    FlagOnPath,
    Forbidden,
    GraphShape,
    Grounded,
    Inclusion,
    NotAmong,
    Ordered,
    SharedValue,
)

# %% Constraint predicates


def invariant_lacks_violation_event(
    requirement: ConditionRequirement,
) -> bool:
    """
    Return whether a throughout requirement lacks a detectable violation.
    """
    if requirement.mode is not ConditionMode.THROUGHOUT:
        return False
    if not isinstance(requirement.on_condition, PhaseConditionType):
        return True
    return requirement.on_condition.violation_signalled_by is None


def establishes_condition(candidate: object) -> bool:
    """
    Return whether a supplied fragment establishes its condition.
    """
    if not isinstance(candidate, ConditionRequirement):
        return True
    return candidate.mode is ConditionMode.AT_EXIT


def magnitude_has_incompatible_unit(magnitude: MagnitudeRange) -> bool:
    """
    Return whether a magnitude's unit and physical dimension disagree.
    """
    if not isinstance(magnitude.unit, MeasurementUnit):
        return True
    if not isinstance(magnitude.physical_dimension, PhysicalDimension):
        return True
    return magnitude.unit.physical_dimension is not magnitude.physical_dimension


def parameter_has_incompatible_dimension(parameter: Parameter) -> bool:
    """
    Return whether a quantitative parameter uses an unsupported dimension.
    """
    if parameter.parameter_range is None:
        return False
    dimension = parameter.parameter_range.physical_dimension
    if parameter.required_dimension is not None:
        return dimension is not parameter.required_dimension
    if parameter.permitted_dimensions is not None:
        return dimension not in parameter.permitted_dimensions
    return False


def parameter_has_incompatible_value_form(parameter: Parameter) -> bool:
    """
    Return whether a parameter uses the wrong value representation.
    """
    if parameter.required_value_form is ParameterValueForm.QUANTITATIVE:
        return parameter.parameter_range is None
    return parameter.categorical_value is None


def temporal_window_has_incompatible_unit(window: TemporalWindow) -> bool:
    """
    Return whether a temporal window uses a non-temporal unit.
    """
    if not isinstance(window.unit, MeasurementUnit):
        return True
    return window.unit.physical_dimension is not PhysicalDimension.TIME


def prediction_has_incompatible_tolerance(
    prediction: SensoryEventPrediction,
) -> bool:
    """
    Return whether prediction magnitude and tolerance dimensions disagree.
    """
    if prediction.expected_magnitude is None or prediction.tolerance is None:
        return False
    return (
        prediction.expected_magnitude.physical_dimension
        is not prediction.tolerance.physical_dimension
    )


def continuous_prediction_has_signal_mismatch(
    prediction: ContinuousSignalPrediction,
) -> bool:
    """
    Return whether prediction and observation use different signals.
    """
    if prediction.signal is None or prediction.comparison_target is None:
        return False
    return prediction.comparison_target.modality is not prediction.signal


def continuous_prediction_has_incompatible_tolerance(
    prediction: ContinuousSignalPrediction,
) -> bool:
    """
    Return whether trajectory and tolerance dimensions differ.
    """
    if prediction.predicted_trajectory is None or prediction.tolerance is None:
        return False
    expected_range = prediction.predicted_trajectory.expected_range
    if expected_range is None:
        return False
    return (
        expected_range.physical_dimension is not prediction.tolerance.physical_dimension
    )


def continuous_prediction_controls_inactive_variable(
    prediction: ContinuousSignalPrediction,
) -> bool:
    """
    Return whether the controlled variable is inactive in a predicted phase.
    """
    if prediction.controlled_variable is None:
        return False
    active_phases = prediction.controlled_variable.parameter_in_phase
    return any(phase not in active_phases for phase in prediction.predicted_for_phase)


def matched_comparison_lacks_success_path(
    comparison: PredictionComparison,
) -> bool:
    """
    Return whether a match omits an output required to advance control.
    """
    if comparison.outcome is not ComparisonOutcome.MATCH:
        return False
    return (
        comparison.compares_observation is None
        or comparison.confirms_subgoal is None
        or comparison.supplies_state_estimate is None
        or not comparison.permits_transition
    )


def matched_comparison_has_mismatch_path(
    comparison: PredictionComparison,
) -> bool:
    """
    Return whether a match also declares mismatch-only consequences.
    """
    if comparison.outcome is not ComparisonOutcome.MATCH:
        return False
    return bool(
        comparison.has_error
        or comparison.invokes_correction
        or comparison.produces_model_update
    )


def mismatched_comparison_lacks_response(
    comparison: PredictionComparison,
) -> bool:
    """
    Return whether a mismatch lacks an error or corrective response.
    """
    if comparison.outcome is not ComparisonOutcome.MISMATCH:
        return False
    return not comparison.has_error or not comparison.invokes_correction


def mismatched_comparison_has_success_path(
    comparison: PredictionComparison,
) -> bool:
    """
    Return whether a mismatch also claims successful control outputs.
    """
    if comparison.outcome is not ComparisonOutcome.MISMATCH:
        return False
    return bool(
        comparison.confirms_subgoal
        or comparison.supplies_state_estimate
        or comparison.permits_transition
    )


def absent_error_has_observation(comparison: PredictionComparison) -> bool:
    """
    Return whether an absent-event error contradicts a received observation.
    """
    return comparison.compares_observation is not None and any(
        error.kind is PredictionErrorKind.ABSENT for error in comparison.has_error
    )


def non_absent_error_lacks_observation(comparison: PredictionComparison) -> bool:
    """
    Return whether a non-absent error has no observation to classify.
    """
    if comparison.compares_observation is not None or not comparison.has_error:
        return False
    return all(
        error.kind is not PredictionErrorKind.ABSENT for error in comparison.has_error
    )


def transition_has_invalid_permission(transition: PhaseTransition) -> bool:
    """
    Return whether a transition's comparison permission is incoherent.
    """
    for comparison in transition.permitted_by:
        state_estimate = comparison.supplies_state_estimate
        if comparison.outcome is not ComparisonOutcome.MATCH:
            return True
        if transition not in comparison.permits_transition:
            return True
        if (
            transition.from_phase is None
            or comparison.confirms_subgoal is not transition.from_phase.aims_for
        ):
            return True
        if (
            state_estimate is None
            or state_estimate.for_phase is not transition.to_phase
        ):
            return True
    return False


def model_update_changes_unheld_estimate(update: ModelUpdate) -> bool:
    """
    Return whether an update targets an estimate outside its representation.
    """
    if update.updates_representation is None:
        return False
    return any(
        estimate not in update.updates_representation.holds_estimate
        for estimate in update.updates_estimate
    )


def control_point_lacks_comparison_outcome(
    control_point: SensorimotorControlPoint,
) -> bool:
    """
    Return whether a control point omits either match or mismatch handling.
    """
    outcomes = {comparison.outcome for comparison in control_point.has_comparison}
    return not {
        ComparisonOutcome.MATCH,
        ComparisonOutcome.MISMATCH,
    }.issubset(outcomes)


def structure_has_uncontrolled_normal_boundary(
    structure: ActionPhaseStructure,
) -> bool:
    """
    Return whether a normal phase boundary lacks control coverage.
    """
    control_events = [
        control_point.at_event_type for control_point in structure.has_control_point
    ]
    exempted_events = [
        exemption.exempts_event for exemption in structure.has_control_point_exemption
    ]
    return any(
        not event.is_failure_event
        and event not in control_events
        and event not in exempted_events
        for phase in structure.defines_phase
        for event in phase.terminated_by
    )


def comparison_has_incompatible_evidence(
    comparison: PredictionComparison,
) -> bool:
    """
    Return whether predicted and observed evidence describe different signals.
    """
    prediction = comparison.compares_prediction
    observation = comparison.compares_observation
    if prediction is None or observation is None:
        return False
    if type(prediction.modality) is not type(observation.modality):
        return True
    if {id(role) for role in prediction.expected_on_role} != {
        id(role) for role in observation.observed_on_role
    }:
        return True
    if comparison.outcome is ComparisonOutcome.MISMATCH and any(
        error.kind is PredictionErrorKind.UNEXPECTED for error in comparison.has_error
    ):
        return False
    if prediction.predicts_event is not None:
        return observation.observes_event is not prediction.predicts_event
    if prediction.predicts_feature is not None:
        return observation.observes_feature != prediction.predicts_feature
    return True


def engineering_extension_lacks_provenance(
    assertion: ScientificAssertion,
) -> bool:
    """
    Return whether an engineering extension omits its source.
    """
    return (
        assertion.scientific_status is ScientificStatus.ENGINEERING_EXTENSION
        and assertion.scientific_provenance is None
    )


def structure_has_invalid_control_exemption(
    structure: ActionPhaseStructure,
) -> bool:
    """
    Return whether an exemption does not name a normal phase boundary.
    """
    boundaries = [
        event for phase in structure.defines_phase for event in phase.terminated_by
    ]
    return any(
        exemption.exempts_event not in boundaries
        or exemption.exempts_event is None
        or exemption.exempts_event.is_failure_event
        for exemption in structure.has_control_point_exemption
    )


# %% Acceptance suite

CONSTRAINTS = [
    *GAZE_CONSTRAINTS,
    *OBSERVATION_CONSTRAINTS,
    *LEARNING_CONSTRAINTS,
    *PERCEPTION_CONSTRAINTS,
    *OBJECT_DYNAMICS_CONSTRAINTS,
    *PREHENSION_CONSTRAINTS,
    # %% Phase 8 semantic hardening
    Forbidden(
        id="C66",
        applies_to=ActionPhaseStructure,
        when=structure_has_uncontrolled_normal_boundary,
        message="normal phase boundary lacks a control point or explicit exemption",
        rationale="Every successful boundary must either compare predicted with "
        "observed evidence or state why that comparison is intentionally omitted.",
    ),
    Forbidden(
        id="C67",
        applies_to=PredictionComparison,
        when=comparison_has_incompatible_evidence,
        message="prediction and observation describe incompatible sensory evidence",
        rationale="A comparison requires consistent modality and participant roles. "
        "Event or feature identity may differ only for an explicitly classified "
        "unexpected observation.",
    ),
    Forbidden(
        id="C68",
        applies_to=ScientificAssertion,
        when=engineering_extension_lacks_provenance,
        message="engineering extension has no explicit provenance",
        rationale="Application-specific additions must remain distinguishable from "
        "claims supported by the reviewed research.",
    ),
    Forbidden(
        id="C69",
        applies_to=ActionPhaseStructure,
        when=structure_has_invalid_control_exemption,
        message="control-point exemption does not name a normal phase boundary",
        rationale="An exemption may justify an omitted comparison only for a boundary "
        "that the represented phase structure actually uses.",
    ),
    *POURING_CONSTRAINTS,
    # %% numeric sanity
    Ordered(
        id="C06",
        applies_to=MagnitudeRange,
        lower="lower_bound",
        upper="upper_bound",
        rationale="An inverted interval is not a range. OWL has no facility for "
        "comparing two data values, so this could only ever live here.",
    ),
    Forbidden(
        id="C33",
        applies_to=MagnitudeRange,
        when=magnitude_has_incompatible_unit,
        message="measurement unit is incompatible with the declared physical dimension",
        rationale="A quantity whose unit and physical dimension disagree cannot be "
        "compared or converted safely.",
    ),
    ExactlyOne(
        id="C34",
        applies_to=Parameter,
        fields=("parameter_range", "categorical_value"),
        rationale="A parameter must have exactly one value representation so numeric "
        "and categorical semantics cannot be conflated.",
    ),
    Forbidden(
        id="C35",
        applies_to=Parameter,
        when=parameter_has_incompatible_dimension,
        message="quantity dimension is incompatible with the parameter's meaning",
        rationale="A mass value cannot serve as a force threshold, and analogous "
        "dimension mismatches must be rejected.",
    ),
    Forbidden(
        id="C36",
        applies_to=Parameter,
        when=parameter_has_incompatible_value_form,
        message="value representation is incompatible with the parameter's meaning",
        rationale="Categorical properties must not use fabricated numeric ranges, "
        "and quantitative parameters must carry magnitudes.",
    ),
    # %% predictive comparison
    ExactlyOne(
        id="C37",
        applies_to=SensoryEventPrediction,
        fields=("predicts_event", "predicts_feature"),
        rationale="A sensory prediction must identify one comparable event or feature.",
    ),
    ExactlyOne(
        id="C37b",
        applies_to=SensoryObservationType,
        fields=("observes_event", "observes_feature"),
        rationale="An observation must identify one event or feature so the "
        "comparison target is unambiguous.",
    ),
    Ordered(
        id="C38",
        applies_to=TemporalWindow,
        lower="earliest_offset",
        upper="latest_offset",
        rationale="A timing window cannot finish before it begins.",
    ),
    Forbidden(
        id="C39",
        applies_to=TemporalWindow,
        when=temporal_window_has_incompatible_unit,
        message="timing window uses a non-temporal measurement unit",
        rationale="Prediction timing must be dimensionally comparable.",
    ),
    Forbidden(
        id="C40",
        applies_to=SensoryEventPrediction,
        when=prediction_has_incompatible_tolerance,
        message="prediction magnitude and tolerance use different physical dimensions",
        rationale="A tolerance cannot bound a magnitude in another physical dimension.",
    ),
    Forbidden(
        id="C56a",
        applies_to=ContinuousSignalPrediction,
        when=continuous_prediction_has_signal_mismatch,
        message="prediction signal differs from its comparison observation modality",
        rationale="A continuous prediction and its observation must compare the same "
        "sensory channel.",
    ),
    Forbidden(
        id="C56b",
        applies_to=ContinuousSignalPrediction,
        when=continuous_prediction_has_incompatible_tolerance,
        message="trajectory and tolerance use different physical dimensions",
        rationale="A tolerance cannot bound a continuous trajectory in another "
        "physical dimension.",
    ),
    Forbidden(
        id="C56c",
        applies_to=ContinuousSignalPrediction,
        when=continuous_prediction_controls_inactive_variable,
        message="controlled variable is not active in every predicted phase",
        rationale="A continuous controller must name a variable active in the phase "
        "where its comparison runs.",
    ),
    Forbidden(
        id="C41",
        applies_to=PredictionComparison,
        when=matched_comparison_lacks_success_path,
        message="match lacks an observation, confirmed subgoal, next-phase state, "
        "or permitted transition",
        rationale="A successful comparison must provide every output needed to "
        "advance control.",
    ),
    Forbidden(
        id="C41b",
        applies_to=PredictionComparison,
        when=matched_comparison_has_mismatch_path,
        message="match also declares an error, correction, or model update",
        rationale="Match and mismatch consequences are mutually exclusive.",
    ),
    Forbidden(
        id="C42",
        applies_to=PredictionComparison,
        when=absent_error_has_observation,
        message="absent-event error contradicts the declared observation",
        rationale="An event cannot be both absent and observed in one comparison.",
    ),
    Forbidden(
        id="C42b",
        applies_to=PredictionComparison,
        when=non_absent_error_lacks_observation,
        message="non-absent prediction error has no observation to classify",
        rationale="Timing, magnitude, direction, spatial, and unexpected-event "
        "errors require an observation.",
    ),
    Forbidden(
        id="C43",
        applies_to=PredictionComparison,
        when=mismatched_comparison_lacks_response,
        message="mismatch lacks a classified error or corrective response",
        rationale="A mismatch without classification and correction has no "
        "operational consequence.",
    ),
    Forbidden(
        id="C43b",
        applies_to=PredictionComparison,
        when=mismatched_comparison_has_success_path,
        message="mismatch also confirms a subgoal, state estimate, or transition",
        rationale="A failed comparison cannot simultaneously permit normal progress.",
    ),
    Forbidden(
        id="C44",
        applies_to=PhaseTransition,
        when=transition_has_invalid_permission,
        message="comparison permission does not confirm the source subgoal, supply "
        "the target state, or link back to this transition",
        rationale="Comparison-gated transitions must form one coherent successful "
        "control path.",
    ),
    Forbidden(
        id="C45",
        applies_to=ModelUpdate,
        when=model_update_changes_unheld_estimate,
        message="model update targets an estimate not held by its representation",
        rationale="An update must identify the representation that owns the changed "
        "estimate.",
    ),
    Inclusion(
        id="C46",
        applies_to=SensorimotorControlPoint,
        source="has_prediction.caused_by_event",
        target="at_event_type",
        rationale="Every control-point prediction must be caused by that control "
        "point's mechanical event.",
    ),
    Inclusion(
        id="C47",
        applies_to=SensorimotorControlPoint,
        source="has_comparison.compares_prediction",
        target="has_prediction",
        rationale="A control point can compare only predictions that it declares.",
    ),
    Inclusion(
        id="C48",
        applies_to=PredictionComparison,
        source="invokes_correction.triggered_by_error",
        target="has_error",
        rationale="A comparison can invoke corrections only for errors it classified.",
    ),
    Inclusion(
        id="C49",
        applies_to=PredictionComparison,
        source="produces_model_update.triggered_by_error",
        target="has_error",
        rationale="A comparison can update a model only from errors it classified.",
    ),
    Forbidden(
        id="C50",
        applies_to=SensorimotorControlPoint,
        when=control_point_lacks_comparison_outcome,
        message="control point does not declare both match and mismatch handling",
        rationale="A control point must define normal progress and recovery.",
    ),
    # %% state-machine coherence
    Inclusion(
        id="C07",
        applies_to=PhaseTransition,
        source="gated_by",
        target="from_phase.terminated_by",
        rationale="The gate mismatch. A transition must fire on an event that "
        "actually ends its source phase; otherwise the phase sequence "
        "and the event sequence are two unrelated stories told about "
        "the same action.",
        note="a transition gate must end the phase it leaves",
    ),
    Inclusion(
        id="C08",
        applies_to=PhaseTransition,
        source="gated_by",
        target="to_phase.initiated_by",
        when=lambda t: bool(t.to_phase and t.to_phase.initiated_by),
        rationale="A transition whose target declares an initiating event must "
        "be gated by that event.",
    ),
    Disjoint(
        id="C09",
        applies_to=ActionPhase,
        left="initiated_by",
        right="terminated_by",
        rationale="A phase begun and ended by the same event has zero extent.",
    ),
    Distinct(
        id="C10",
        applies_to=PhaseTransition,
        left="from_phase",
        right="to_phase",
        rationale="A self-loop is not a phase transition.",
    ),
    DistinctAcross(
        id="C11",
        applies_to=ActionPhaseStructure,
        severity="warning",
        collection="defines_phase",
        key_path="terminated_by",
        rationale="Two phases whose terminating events coincide cannot be told "
        "apart by the controller, which usually means they should be "
        "merged or one needs its own boundary event.",
    ),
    FlagOnPath(
        id="C12",
        applies_to=PhaseTransition,
        path="gated_by",
        flag="is_failure_event",
        expected=False,
        rationale="A failure event must not gate a normal transition. Failures "
        "are what the controller guards against, not what it waits "
        "for; conflating the two makes spilling a legitimate way to "
        "finish pouring.",
    ),
    # %% referential closure
    Closure(
        id="C13a",
        applies_to=ActionPhaseStructure,
        container="defines_phase",
        reached_via=(
            "has_transition.from_phase",
            "has_transition.to_phase",
            "defines_initial_state_estimate.for_phase",
        ),
        rationale="A hypothesis that references a phase it does not declare is "
        "not self-contained, and so cannot be cached, versioned or "
        "shipped as a unit.",
    ),
    Closure(
        id="C13b",
        applies_to=ActionPhaseStructure,
        container="defines_event_type",
        reached_via=(
            "has_transition.gated_by",
            "defines_phase.terminated_by",
            "defines_phase.initiated_by",
            "has_control_point.at_event_type",
            "defines_prediction.predicts_event",
            "defines_prediction.caused_by_event",
            "has_control_point_exemption.exempts_event",
            "defines_observation.observes_event",
        ),
        rationale="Every event a structure gates or bounds on must be one it "
        "declares.",
    ),
    Closure(
        id="C13c",
        applies_to=ActionPhaseStructure,
        container="defines_role",
        reached_via=(
            "defines_phase.phase_involves",
            "defines_prediction.expected_on_role",
            "defines_observation.observed_on_role",
            "defines_correction.intended_effect.condition_on_role",
            "defines_initial_state_estimate.estimates_condition.condition_on_role",
        ),
        rationale="A role a phase involves but the structure never declares "
        "cannot be bound by the resolver.",
    ),
    Closure(
        id="C13d",
        applies_to=ActionPhaseStructure,
        container="defines_parameter",
        reached_via=(
            "defines_event_type.crosses_threshold",
            "defines_event_type.attains_setpoint",
            "defines_parameter.scales",
            "defines_parameter.adapts_to",
            "defines_correction.targets_setpoint",
            "has_model_update.updates_estimate",
        ),
        rationale="Every threshold, setpoint, or estimate linked by the structure "
        "must be one of its declared parameters.",
    ),
    Closure(
        id="C13e",
        applies_to=ActionPhaseStructure,
        container="defines_prediction",
        reached_via=(
            "has_control_point.has_prediction",
            "has_comparison.compares_prediction",
        ),
        rationale="Every compared prediction must be declared by the hypothesis.",
    ),
    Closure(
        id="C13f",
        applies_to=ActionPhaseStructure,
        container="defines_observation",
        reached_via=(
            "has_comparison.compares_observation",
            "defines_initial_state_estimate.based_on_observation",
        ),
        rationale="Every compared or state-supporting observation must be declared.",
    ),
    Closure(
        id="C13g",
        applies_to=ActionPhaseStructure,
        container="has_comparison",
        reached_via=(
            "has_control_point.has_comparison",
            "has_transition.permitted_by",
        ),
        rationale="Every control or transition comparison must be declared.",
    ),
    Closure(
        id="C13h",
        applies_to=ActionPhaseStructure,
        container="defines_error",
        reached_via=(
            "has_comparison.has_error",
            "defines_correction.triggered_by_error",
            "has_model_update.triggered_by_error",
        ),
        rationale="Every recovery trigger must be a declared prediction error.",
    ),
    Closure(
        id="C13i",
        applies_to=ActionPhaseStructure,
        container="defines_correction",
        reached_via=("has_comparison.invokes_correction",),
        rationale="Every invoked correction must be declared by the hypothesis.",
    ),
    Closure(
        id="C13j",
        applies_to=ActionPhaseStructure,
        container="has_model_update",
        reached_via=("has_comparison.produces_model_update",),
        rationale="Every comparison-produced model update must be declared.",
    ),
    Closure(
        id="C13k",
        applies_to=ActionPhaseStructure,
        container="defines_initial_state_estimate",
        reached_via=("has_comparison.supplies_state_estimate",),
        rationale="Every comparison-supplied initial state must be declared.",
    ),
    Closure(
        id="C15",
        applies_to=ActionPhaseStructure,
        severity="warning",
        container="defines_phase",
        reached_via=("defines_parameter.parameter_in_phase",),
        rationale="A parameter scoped to a phase outside its own structure "
        "usually signals a copy-paste artifact in generation.",
    ),
    NotAmong(
        id="C14",
        applies_to=ActionPhaseStructure,
        value_path="decomposes_task",
        collection="defines_phase",
        rationale="The decomposed task must not appear among its own phases. "
        "Not expressible in OWL, because phases genuinely are tasks "
        "and the two cannot be made disjoint classes.",
    ),
    # %% graph shape
    GraphShape(
        id="C16",
        applies_to=ActionPhaseStructure,
        container="defines_phase",
        edges="has_transition",
        tail="from_phase",
        head="to_phase",
        rationale="A structure must have exactly one entry phase, and no phase "
        "may be an orphan. Two entries means the controller cannot "
        "tell where to start; an orphan is knowledge it can never act "
        "on.",
    ),
    Grounded(
        id="C18",
        applies_to=ActionPhaseStructure,
        severity="warning",
        container="defines_phase",
        needs="has_condition_requirement",
        supplies=("defines_phase.aims_for", "defines_phase.has_condition_requirement"),
        supplies_when=establishes_condition,
        excused_when=lambda r: bool(
            r.is_world_assumption
            or r.mode is ConditionMode.AT_EXIT
            or r.polarity is ConditionPolarity.MUST_NOT_HOLD
        ),
        rationale="An ungrounded precondition: a requirement not flagged as a "
        "world assumption whose condition no phase establishes. "
        "Either an upstream phase is missing, or the requirement "
        "should be marked as an assumption so the resolver knows to "
        "check it against the world model rather than expecting the "
        "plan to have produced it.",
    ),
    # %% subgoal attainment
    Inclusion(
        id="C21",
        applies_to=ActionPhase,
        source="aims_for.attainment_signalled_by",
        target="terminated_by",
        rationale="The subgoal/boundary mismatch. A phase's subgoal must be "
        "signalled by one of that phase's own terminating events, or "
        "the terminating event is an incident rather than a "
        "sensorimotor control point.",
        note="this is what makes a boundary event a control point",
    ),
    FlagOnPath(
        id="C23",
        applies_to=Subgoal,
        path="attainment_signalled_by",
        flag="is_failure_event",
        expected=False,
        rationale="Success cannot be marked by a failure event.",
    ),
    # %% force coupling
    SharedValue(
        id="C19",
        applies_to=SafetyMargin,
        severity="warning",
        left="margin_on.parameter_on_role",
        right="margin_over.parameter_on_role",
        rationale="A safety margin relating a setpoint and a threshold about "
        "different participants is not a margin about anything.",
    ),
    # %% regulation
    Forbidden(
        id="C03",
        applies_to=ActionPhase,
        when=lambda phase: phase.regulation_mode
        in (RegulationMode.CONTINUOUS_FEEDBACK, RegulationMode.HYBRID)
        and not phase.monitors_signal,
        message="feedback-regulated but monitors no signal, so it has no control law",
        rationale="Feedback and hybrid regulation require at least one monitored "
        "sensory signal.",
    ),
    # %% control points
    Forbidden(
        id="C20",
        applies_to=SensorimotorControlPoint,
        severity="warning",
        when=lambda control_point: not any(
            comparison.invokes_correction
            for comparison in control_point.has_comparison
            if comparison.outcome is ComparisonOutcome.MISMATCH
        ),
        message="mismatch handling names no corrective response",
        rationale="A control point without a corrective response discards the "
        "diagnostic value of comparison.",
    ),
    # %% invariants
    Forbidden(
        id="C28",
        applies_to=ConditionRequirement,
        when=invariant_lacks_violation_event,
        message="required throughout a phase but nothing registers its "
        "violation, so it is not a guard but a hope",
        rationale="An undetectable invariant. A condition required to hold "
        "throughout a phase, with no event registering its violation, "
        "is not a guard but a hope.",
    ),
    FlagOnPath(
        id="C29",
        applies_to=ConditionRequirement,
        path="on_condition.violation_signalled_by",
        flag="is_failure_event",
        expected=True,
        rationale="An invariant violation signalled by an event not marked as a "
        "failure.",
    ),
    # %% Canonical ontology classification
    ConceptReferenceType(
        id="C30c",
        applies_to=RoleSlot,
        path="binds_as",
        concept_root=ManipulationRole,
        rationale="A role slot must name a concrete role concept so a later "
        "binding step knows what kind of participant to seek.",
    ),
]

SUITE = ConstraintSuite(name="flanagan", constraints=CONSTRAINTS)


def evaluate(instances):
    """
    Run the suite over every instance of a loaded hypothesis.
    """
    return SUITE.evaluate(instances)


def report(instances) -> str:
    return SUITE.report(instances)


__all__ = ["CONSTRAINTS", "SUITE", "evaluate", "report"]
