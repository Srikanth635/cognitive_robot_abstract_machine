"""
Acceptance rules of the core module: the relationships between phases, events,
transitions, conditions and quantities that a restriction on one class cannot state.
"""

from __future__ import annotations

from dataclasses import fields
from enum import StrEnum
from functools import cached_property
from operator import attrgetter

import rustworkx
from krrood.entity_query_language.factories import (
    and_,
    contains,
    flat_variable,
    not_,
    or_,
)
from krrood.entity_query_language.predicate import HasType
from typing_extensions import Any, Sequence

from llm_reasoner.acceptance.rules import (
    FragmentRule,
    RequirementRule,
    Severity,
    StructureRule,
    TypedRule,
    Violation,
)
from llm_reasoner.acceptance.vocabulary import Vocabulary
from llm_reasoner.ontologies.predictive_manipulation import core
from llm_reasoner.ontologies.terms import DescriptionEntity, PropertyTerm

# %% rule names


class CoreRule(StrEnum):
    """
    The acceptance rules of the core module.
    """

    RANGE_BOUNDS_ORDERED = "range_bounds_ordered"
    """
    An inverted interval is not a range. OWL cannot compare two data values, so only a
    rule can catch it.
    """

    WINDOW_BOUNDS_ORDERED = "window_bounds_ordered"
    """
    A timing window cannot finish before it begins.
    """

    TRANSITION_GATE_ENDS_SOURCE = "transition_gate_ends_source"
    """
    A transition must fire on an event that ends the phase it leaves; otherwise the
    controller would switch phases while the current one is still running.
    """

    TRANSITION_GATE_STARTS_TARGET = "transition_gate_starts_target"
    """
    A transition into a phase that names its initiating events must fire on one of
    them.
    """

    PHASE_HAS_EXTENT = "phase_has_extent"
    """
    A phase begun and ended by the same event has no extent.
    """

    TRANSITION_IS_NOT_A_LOOP = "transition_is_not_a_loop"
    """
    A transition from a phase to itself does not move the task on.
    """

    SUBGOAL_SIGNALLED_BY_PHASE_END = "subgoal_signalled_by_phase_end"
    """
    A phase's subgoal must be signalled by one of the phase's own terminating events;
    that is what makes a boundary event a control point.
    """

    SAFETY_MARGIN_ON_ONE_PARTICIPANT = "safety_margin_on_one_participant"
    """
    A safety margin relates a setpoint to a threshold of the same participant; one
    relating different participants is not a margin.
    """

    TASK_IS_NOT_ITS_OWN_PHASE = "task_is_not_its_own_phase"
    """
    No phase may carry out the very task the structure divides into phases.
    """

    ONE_ENTRY_PHASE = "one_entry_phase"
    """
    A structure has exactly one phase no transition leads into; two would leave the
    controller unable to tell where to begin.
    """

    NO_ORPHAN_PHASE = "no_orphan_phase"
    """
    Every phase is connected to the others by transitions; an unconnected phase is
    never reached.
    """

    PHASE_ENDINGS_DISTINCT = "phase_endings_distinct"
    """
    Two phases ending at the same events cannot be told apart by the controller, which
    usually means one of them was duplicated.
    """

    PRECONDITIONS_GROUNDED = "preconditions_grounded"
    """
    A condition a phase requires should be established by some phase of the structure,
    or be declared an assumption about the world.
    """

    MATCH_HAS_NO_RESPONSE = "match_has_no_response"
    """
    Corrections and model updates answer mismatches; a match that invokes them claims
    both outcomes at once.
    """

    ABSENT_EVENT_HAS_NO_OBSERVATION = "absent_event_has_no_observation"
    """
    An error classifying a predicted event as absent contradicts an observation of it.
    """

    ERROR_HAS_OBSERVATION = "error_has_observation"
    """
    An error other than an absent event classifies an observation, so the comparison
    must name the observation.
    """

    PERMISSION_FITS_TRANSITION = "permission_fits_transition"
    """
    A comparison permitting a transition confirms the subgoal of the phase the
    transition leaves, or supplies the initial state of the phase it enters.
    """

    UPDATED_ESTIMATE_IS_HELD = "updated_estimate_is_held"
    """
    A model update changes estimates held by the representation it updates.
    """

    CONTROL_POINT_PREDICTS_ITS_EVENT = "control_point_predicts_its_event"
    """
    The predictions a control point compares are caused by the event it sits at.
    """

    CONTROL_POINT_COMPARES_ITS_PREDICTIONS = "control_point_compares_its_predictions"
    """
    A control point compares only predictions it declares.
    """

    CORRECTION_ANSWERS_CLASSIFIED_ERROR = "correction_answers_classified_error"
    """
    A comparison invokes corrections only for errors it has classified.
    """

    MODEL_UPDATE_ANSWERS_CLASSIFIED_ERROR = "model_update_answers_classified_error"
    """
    A comparison produces model updates only from errors it has classified.
    """

    MISMATCH_INVOKES_CORRECTION = "mismatch_invokes_correction"
    """
    A control point's mismatch handling should name a corrective response, not only a
    model update.
    """

    EVIDENCE_COMPATIBLE = "evidence_compatible"
    """
    A comparison relates a prediction and an observation of the same modality, at the
    same participants, about the same event or feature.
    """

    CONTINUOUS_PREDICTION_COMPARES_ITS_MODALITY = (
        "continuous_prediction_compares_its_modality"
    )
    """
    A continuous prediction is compared with an observation in its own modality.
    """

    TOLERANCE_MATCHES_TRAJECTORY = "tolerance_matches_trajectory"
    """
    A tolerance bounds a trajectory only in the trajectory's own physical dimension.
    """

    CONTROLLED_SETPOINT_IS_ACTIVE = "controlled_setpoint_is_active"
    """
    A continuous prediction of a phase's controller regulates a setpoint active in that
    phase.
    """

    MECHANICAL_BOUNDARY_MONITORED = "mechanical_boundary_monitored"
    """
    Every mechanical event ending a phase, other than a failure, has a control point or
    an explicit exemption; mechanical boundaries are where the controller checks its
    predictions.
    """

    EXEMPTION_NAMES_NORMAL_BOUNDARY = "exemption_names_normal_boundary"
    """
    A control point exemption names an event that ends one of the structure's phases
    and is not a failure.
    """

    CONTINUOUS_PREDICTION_IN_MONITORED_MODALITY = (
        "continuous_prediction_in_monitored_modality"
    )
    """
    A controller regulating from sensory feedback compares a continuous prediction in a
    modality it monitors; otherwise it has no control law. Having a continuous prediction
    at all is required by the ontology.
    """

    COMPONENTS_DECLARED = "components_declared"
    """
    Every fragment a structure reaches that a structure can declare is declared by it,
    so the hypothesis is self-contained and every part can be bound and cited.
    """


# %% quantities


class RangeBoundsOrdered(FragmentRule[core.MagnitudeRange]):
    """
    The lower bound of a magnitude range does not exceed its upper bound.
    """

    @property
    def name(self) -> StrEnum:
        return CoreRule.RANGE_BOUNDS_ORDERED

    @property
    def reason(self) -> str:
        return "lower_bound exceeds upper_bound"

    def violating(self, subject: Any) -> Any:
        return and_(
            subject.lower_bound != None,
            subject.upper_bound != None,
            subject.lower_bound > subject.upper_bound,
        )


class WindowBoundsOrdered(FragmentRule[core.TemporalWindow]):
    """
    The earliest offset of a temporal window does not exceed its latest offset.
    """

    @property
    def name(self) -> StrEnum:
        return CoreRule.WINDOW_BOUNDS_ORDERED

    @property
    def reason(self) -> str:
        return "earliest_offset exceeds latest_offset"

    def violating(self, subject: Any) -> Any:
        return and_(
            subject.earliest_offset != None,
            subject.latest_offset != None,
            subject.earliest_offset > subject.latest_offset,
        )


# %% transitions


class TransitionGateEndsSource(FragmentRule[core.PhaseTransition]):
    """
    Every event gating a transition terminates the phase the transition leaves.
    """

    @property
    def name(self) -> StrEnum:
        return CoreRule.TRANSITION_GATE_ENDS_SOURCE

    @property
    def reason(self) -> str:
        return "gated_by names an event that does not terminate from_phase"

    def violating(self, subject: Any) -> Any:
        gate = flat_variable(subject.gated_by)
        return and_(
            subject.from_phase != None,
            not_(contains(subject.from_phase.terminated_by, gate)),
        )


class TransitionGateStartsTarget(FragmentRule[core.PhaseTransition]):
    """
    Every event gating a transition initiates the phase it enters, when that phase
    names its initiating events.
    """

    @property
    def name(self) -> StrEnum:
        return CoreRule.TRANSITION_GATE_STARTS_TARGET

    @property
    def reason(self) -> str:
        return "gated_by names an event that does not initiate to_phase"

    def violating(self, subject: Any) -> Any:
        gate = flat_variable(subject.gated_by)
        return and_(
            subject.to_phase != None,
            subject.to_phase.initiated_by != [],
            not_(contains(subject.to_phase.initiated_by, gate)),
        )


class TransitionIsNotALoop(FragmentRule[core.PhaseTransition]):
    """
    A transition leaves a phase other than the one it enters.
    """

    @property
    def name(self) -> StrEnum:
        return CoreRule.TRANSITION_IS_NOT_A_LOOP

    @property
    def reason(self) -> str:
        return "from_phase and to_phase are the same phase"

    def violating(self, subject: Any) -> Any:
        return and_(subject.from_phase != None, subject.from_phase == subject.to_phase)


# %% phases, subgoals and conditions


class PhaseHasExtent(FragmentRule[core.ActionPhase]):
    """
    No event both initiates and terminates the same phase.
    """

    @property
    def name(self) -> StrEnum:
        return CoreRule.PHASE_HAS_EXTENT

    @property
    def reason(self) -> str:
        return "an event both initiates and terminates the phase"

    def violating(self, subject: Any) -> Any:
        return contains(subject.terminated_by, flat_variable(subject.initiated_by))


class SubgoalSignalledByPhaseEnd(FragmentRule[core.ActionPhase]):
    """
    The event signalling a phase's subgoal is one of the phase's terminating events.
    """

    @property
    def name(self) -> StrEnum:
        return CoreRule.SUBGOAL_SIGNALLED_BY_PHASE_END

    @property
    def reason(self) -> str:
        return "aims_for is signalled by an event that does not terminate the phase"

    def violating(self, subject: Any) -> Any:
        return and_(
            subject.aims_for != None,
            subject.aims_for.attainment_signalled_by != None,
            not_(
                contains(
                    subject.terminated_by, subject.aims_for.attainment_signalled_by
                )
            ),
        )


class SafetyMarginOnOneParticipant(RequirementRule[core.SafetyMargin]):
    """
    The setpoint and the threshold a safety margin relates concern a common
    participant.
    """

    @property
    def name(self) -> StrEnum:
        return CoreRule.SAFETY_MARGIN_ON_ONE_PARTICIPANT

    @property
    def severity(self) -> Severity:
        return Severity.WARNING

    @property
    def reason(self) -> str:
        return "margin_on and margin_over concern different participants"

    def applies(self, subject: Any) -> Any:
        return and_(subject.margin_on != None, subject.margin_over != None)

    def meets(self, subject: Any) -> Any:
        return contains(
            subject.margin_over.parameter_on_role,
            flat_variable(subject.margin_on.parameter_on_role),
        )


# %% structures


class TaskIsNotItsOwnPhase(FragmentRule[core.ActionPhaseStructure]):
    """
    No phase of a structure carries out the task the structure divides.
    """

    @property
    def name(self) -> StrEnum:
        return CoreRule.TASK_IS_NOT_ITS_OWN_PHASE

    @property
    def reason(self) -> str:
        return "a phase realizes the task the structure decomposes"

    def violating(self, subject: Any) -> Any:
        phase = flat_variable(subject.defines_phase)
        return and_(
            subject.decomposes_task != None,
            contains(phase.realizes_action_category, subject.decomposes_task),
        )


def phase_graph(structure: core.ActionPhaseStructure) -> rustworkx.PyDiGraph:
    """
    :return: The structure's phases as nodes and its transitions between them as edges.
    """
    graph = rustworkx.PyDiGraph()
    nodes = {id(phase): graph.add_node(phase) for phase in structure.defines_phase}
    for transition in structure.defines_transition:
        source, target = id(transition.from_phase), id(transition.to_phase)
        if source in nodes and target in nodes:
            graph.add_edge(nodes[source], nodes[target], transition)
    return graph


class OneEntryPhase(StructureRule[core.ActionPhaseStructure]):
    """
    Exactly one phase of a structure has no transition leading into it.
    """

    @property
    def name(self) -> StrEnum:
        return CoreRule.ONE_ENTRY_PHASE

    @property
    def reason(self) -> str:
        return "the structure does not have exactly one entry phase"

    def is_violated_by(self, fragment: core.ActionPhaseStructure) -> bool:
        graph = phase_graph(fragment)
        entries = [node for node in graph.node_indices() if graph.in_degree(node) == 0]
        return len(entries) != 1


class NoOrphanPhase(StructureRule[core.ActionPhaseStructure]):
    """
    The transitions of a structure connect all its phases.
    """

    @property
    def name(self) -> StrEnum:
        return CoreRule.NO_ORPHAN_PHASE

    @property
    def reason(self) -> str:
        return "a phase is not connected to the others by transitions"

    def is_violated_by(self, fragment: core.ActionPhaseStructure) -> bool:
        graph = phase_graph(fragment)
        return graph.num_nodes() > 1 and not rustworkx.is_weakly_connected(graph)


class PhaseEndingsDistinct(StructureRule[core.ActionPhaseStructure]):
    """
    No two phases of a structure end at the same events.
    """

    @property
    def name(self) -> StrEnum:
        return CoreRule.PHASE_ENDINGS_DISTINCT

    @property
    def severity(self) -> Severity:
        return Severity.WARNING

    @property
    def reason(self) -> str:
        return "two phases end at the same events"

    def is_violated_by(self, fragment: core.ActionPhaseStructure) -> bool:
        endings = [
            frozenset(id(event) for event in phase.terminated_by)
            for phase in fragment.defines_phase
            if phase.terminated_by
        ]
        return len(endings) != len(set(endings))


class PreconditionsGrounded(StructureRule[core.ActionPhaseStructure]):
    """
    Every condition a phase requires at entry or throughout is established within the
    structure, or declared an assumption about the world.
    """

    @property
    def name(self) -> StrEnum:
        return CoreRule.PRECONDITIONS_GROUNDED

    @property
    def severity(self) -> Severity:
        return Severity.WARNING

    @property
    def reason(self) -> str:
        return "a required condition is neither established by a phase nor assumed"

    @staticmethod
    def established(structure: core.ActionPhaseStructure) -> set[int]:
        """
        :return: The identities of the conditions the structure's phases bring about.
        """
        conditions = {
            id(phase.aims_for.targets_condition)
            for phase in structure.defines_phase
            if phase.aims_for is not None
            and phase.aims_for.targets_condition is not None
        }
        conditions |= {
            id(condition)
            for phase in structure.defines_phase
            for event in [*phase.initiated_by, *phase.terminated_by]
            for condition in event.initiates_state_type
        }
        conditions |= {
            id(condition)
            for phase in structure.defines_phase
            for requirement in phase.has_condition_requirement
            if requirement.has_condition_mode == core.ConditionMode.AT_EXIT
            and requirement.has_condition_polarity == core.ConditionPolarity.MUST_HOLD
            for condition in requirement.has_condition_type
        }
        return conditions

    def involved(
        self, fragment: core.ActionPhaseStructure
    ) -> list[core.ConditionRequirement]:
        """
        :return: The requirements on entry or throughout whose conditions no phase
            establishes and that are not assumed of the world.
        """
        established = self.established(fragment)
        ungrounded: list[core.ConditionRequirement] = []
        for phase in fragment.defines_phase:
            for requirement in phase.has_condition_requirement:
                if (
                    requirement.has_condition_mode
                    in (core.ConditionMode.AT_ENTRY, core.ConditionMode.THROUGHOUT)
                    and requirement.has_condition_polarity
                    == core.ConditionPolarity.MUST_HOLD
                    and not requirement.is_world_assumption
                    and not any(
                        id(condition) in established
                        for condition in requirement.has_condition_type
                    )
                    and not any(requirement is seen for seen in ungrounded)
                ):
                    ungrounded.append(requirement)
        return ungrounded

    def is_violated_by(self, fragment: core.ActionPhaseStructure) -> bool:
        return bool(self.involved(fragment))


# %% comparisons


class MatchHasNoResponse(FragmentRule[core.PredictionComparison]):
    """
    A matching comparison invokes no correction and produces no model update.
    """

    @property
    def name(self) -> StrEnum:
        return CoreRule.MATCH_HAS_NO_RESPONSE

    @property
    def reason(self) -> str:
        return "a match that invokes a correction or produces a model update"

    def violating(self, subject: Any) -> Any:
        return and_(
            subject.has_comparison_outcome == core.ComparisonOutcome.MATCH,
            or_(subject.invokes_correction != [], subject.produces_model_update != []),
        )


class AbsentEventHasNoObservation(FragmentRule[core.PredictionComparison]):
    """
    A comparison classifying an absent event names no observation.
    """

    @property
    def name(self) -> StrEnum:
        return CoreRule.ABSENT_EVENT_HAS_NO_OBSERVATION

    @property
    def reason(self) -> str:
        return "an absent-event error alongside compares_observation"

    def violating(self, subject: Any) -> Any:
        error = flat_variable(subject.has_prediction_error)
        return and_(
            subject.compares_observation != None,
            error.has_error_kind == core.PredictionErrorKind.ABSENT_EVENT,
        )


class ErrorHasObservation(RequirementRule[core.PredictionComparison]):
    """
    A comparison classifying only errors other than an absent event names the
    observation they classify.
    """

    @property
    def name(self) -> StrEnum:
        return CoreRule.ERROR_HAS_OBSERVATION

    @property
    def reason(self) -> str:
        return "an error other than an absent event without compares_observation"

    def applies(self, subject: Any) -> Any:
        error = flat_variable(subject.has_prediction_error)
        return and_(
            subject.compares_observation == None,
            error.has_error_kind != core.PredictionErrorKind.ABSENT_EVENT,
        )

    def meets(self, subject: Any) -> Any:
        error = flat_variable(subject.has_prediction_error)
        return error.has_error_kind == core.PredictionErrorKind.ABSENT_EVENT


class PermissionFitsTransition(FragmentRule[core.PhaseTransition]):
    """
    A comparison permitting a transition confirms only the source phase's subgoal and
    supplies only the target phase's initial state.
    """

    @property
    def name(self) -> StrEnum:
        return CoreRule.PERMISSION_FITS_TRANSITION

    @property
    def reason(self) -> str:
        return (
            "permitted_by confirms another phase's subgoal or supplies another phase's "
            "initial state"
        )

    def violating(self, subject: Any) -> Any:
        permission = flat_variable(subject.permitted_by)
        return or_(
            and_(
                subject.from_phase != None,
                permission.confirms_subgoal != None,
                permission.confirms_subgoal != subject.from_phase.aims_for,
            ),
            and_(
                permission.supplies_state_estimate != None,
                permission.supplies_state_estimate.for_phase != subject.to_phase,
            ),
        )


class UpdatedEstimateIsHeld(FragmentRule[core.ModelUpdate]):
    """
    A model update changes only estimates its representation holds.
    """

    @property
    def name(self) -> StrEnum:
        return CoreRule.UPDATED_ESTIMATE_IS_HELD

    @property
    def reason(self) -> str:
        return "updates_estimate names an estimate updates_representation does not hold"

    def violating(self, subject: Any) -> Any:
        estimate = flat_variable(subject.updates_estimate)
        return and_(
            subject.updates_representation != None,
            not_(contains(subject.updates_representation.holds_estimate, estimate)),
        )


class CorrectionAnswersClassifiedError(FragmentRule[core.PredictionComparison]):
    """
    The corrections a comparison invokes are triggered by errors it classifies.
    """

    @property
    def name(self) -> StrEnum:
        return CoreRule.CORRECTION_ANSWERS_CLASSIFIED_ERROR

    @property
    def reason(self) -> str:
        return "invokes a correction triggered by an error the comparison does not classify"

    def violating(self, subject: Any) -> Any:
        correction = flat_variable(subject.invokes_correction)
        error = flat_variable(correction.triggered_by_error)
        return not_(contains(subject.has_prediction_error, error))


class ModelUpdateAnswersClassifiedError(FragmentRule[core.PredictionComparison]):
    """
    The model updates a comparison produces are triggered by errors it classifies.
    """

    @property
    def name(self) -> StrEnum:
        return CoreRule.MODEL_UPDATE_ANSWERS_CLASSIFIED_ERROR

    @property
    def reason(self) -> str:
        return (
            "produces a model update triggered by an error the comparison does not "
            "classify"
        )

    def violating(self, subject: Any) -> Any:
        update = flat_variable(subject.produces_model_update)
        error = flat_variable(update.triggered_by_error)
        return not_(contains(subject.has_prediction_error, error))


class EvidenceCompatible(StructureRule[core.PredictionComparison]):
    """
    A comparison's prediction and observation share modality, participants and the
    event or feature they concern, unless the error is an unexpected event.
    """

    @property
    def name(self) -> StrEnum:
        return CoreRule.EVIDENCE_COMPATIBLE

    @property
    def reason(self) -> str:
        return "compares a prediction and an observation of different evidence"

    def is_violated_by(self, fragment: core.PredictionComparison) -> bool:
        prediction, observation = (
            fragment.compares_prediction,
            fragment.compares_observation,
        )
        if prediction is None or observation is None:
            return False
        if prediction.has_modality != observation.has_modality:
            return True
        if {id(role) for role in prediction.expected_on_role} != {
            id(role) for role in observation.observed_on_role
        }:
            return True
        if any(
            error.has_error_kind == core.PredictionErrorKind.UNEXPECTED_EVENT
            for error in fragment.has_prediction_error
        ):
            return False
        if prediction.predicts_event is not None:
            return observation.observes_event is not prediction.predicts_event
        if prediction.predicts_feature is not None:
            return observation.observes_feature is not prediction.predicts_feature
        return False


# %% control points


class ControlPointPredictsItsEvent(FragmentRule[core.SensorimotorControlPoint]):
    """
    Every prediction of a control point is caused by the event it sits at.
    """

    @property
    def name(self) -> StrEnum:
        return CoreRule.CONTROL_POINT_PREDICTS_ITS_EVENT

    @property
    def reason(self) -> str:
        return "has_prediction names a prediction caused by another event than at_event_type"

    def violating(self, subject: Any) -> Any:
        prediction = flat_variable(subject.has_prediction)
        return and_(
            subject.at_event_type != None,
            prediction.caused_by_event != None,
            prediction.caused_by_event != subject.at_event_type,
        )


class ControlPointComparesItsPredictions(FragmentRule[core.SensorimotorControlPoint]):
    """
    Every comparison of a control point compares one of the control point's
    predictions.
    """

    @property
    def name(self) -> StrEnum:
        return CoreRule.CONTROL_POINT_COMPARES_ITS_PREDICTIONS

    @property
    def reason(self) -> str:
        return "has_comparison compares a prediction the control point does not declare"

    def violating(self, subject: Any) -> Any:
        comparison = flat_variable(subject.has_comparison)
        return and_(
            comparison.compares_prediction != None,
            not_(contains(subject.has_prediction, comparison.compares_prediction)),
        )


class MismatchInvokesCorrection(RequirementRule[core.SensorimotorControlPoint]):
    """
    A control point handling mismatches invokes a correction for at least one of them.
    """

    @property
    def name(self) -> StrEnum:
        return CoreRule.MISMATCH_INVOKES_CORRECTION

    @property
    def severity(self) -> Severity:
        return Severity.WARNING

    @property
    def reason(self) -> str:
        return "mismatch handling names no corrective response"

    def applies(self, subject: Any) -> Any:
        comparison = flat_variable(subject.has_comparison)
        return comparison.has_comparison_outcome == core.ComparisonOutcome.MISMATCH

    def meets(self, subject: Any) -> Any:
        comparison = flat_variable(subject.has_comparison)
        return and_(
            comparison.has_comparison_outcome == core.ComparisonOutcome.MISMATCH,
            comparison.invokes_correction != [],
        )


# %% continuous predictions


class ContinuousPredictionComparesItsModality(
    FragmentRule[core.ContinuousSignalPrediction]
):
    """
    A continuous prediction is compared with an observation in its own modality.
    """

    @property
    def name(self) -> StrEnum:
        return CoreRule.CONTINUOUS_PREDICTION_COMPARES_ITS_MODALITY

    @property
    def reason(self) -> str:
        return "compared_with is an observation in another modality"

    def violating(self, subject: Any) -> Any:
        return and_(
            subject.compared_with != None,
            subject.has_modality != None,
            subject.compared_with.has_modality != subject.has_modality,
        )


class ToleranceMatchesTrajectory(FragmentRule[core.ContinuousSignalPrediction]):
    """
    A continuous prediction's tolerance is measured in the dimension of its
    trajectory's range.
    """

    @property
    def name(self) -> StrEnum:
        return CoreRule.TOLERANCE_MATCHES_TRAJECTORY

    @property
    def reason(self) -> str:
        return "has_tolerance is measured in another dimension than the trajectory"

    def violating(self, subject: Any) -> Any:
        expected_range = subject.predicts_trajectory.has_expected_range
        return and_(
            subject.predicts_trajectory != None,
            expected_range != None,
            expected_range.has_unit != None,
            subject.has_tolerance != None,
            subject.has_tolerance.has_unit != None,
            expected_range.has_unit.measures_dimension
            != subject.has_tolerance.has_unit.measures_dimension,
        )


class ControlledSetpointIsActive(FragmentRule[core.ActionPhase]):
    """
    A setpoint regulated by a continuous prediction of a phase's controller is active in
    that phase.
    """

    @property
    def name(self) -> StrEnum:
        return CoreRule.CONTROLLED_SETPOINT_IS_ACTIVE

    @property
    def reason(self) -> str:
        return "the controller regulates a setpoint that is not an active parameter"

    def violating(self, subject: Any) -> Any:
        prediction = flat_variable(subject.controlled_by.has_continuous_prediction)
        return and_(
            subject.controlled_by != None,
            prediction.controls_setpoint != None,
            not_(contains(subject.has_active_parameter, prediction.controls_setpoint)),
        )


class ContinuousPredictionInMonitoredModality(
    RequirementRule[core.ActionPhaseController]
):
    """
    A feedback-regulated controller with continuous predictions has one in a modality it
    monitors.
    """

    @property
    def name(self) -> StrEnum:
        return CoreRule.CONTINUOUS_PREDICTION_IN_MONITORED_MODALITY

    @property
    def reason(self) -> str:
        return "has no continuous prediction in a modality it monitors"

    def applies(self, subject: Any) -> Any:
        return and_(
            or_(
                subject.has_regulation_mode == core.RegulationMode.CONTINUOUS_FEEDBACK,
                subject.has_regulation_mode == core.RegulationMode.HYBRID,
            ),
            HasType(
                flat_variable(subject.has_continuous_prediction),
                core.ContinuousSignalPrediction,
            ),
        )

    def meets(self, subject: Any) -> Any:
        prediction = flat_variable(subject.has_continuous_prediction)
        return contains(subject.monitors_modality, prediction.has_modality)


# %% structure-wide control


def normal_boundaries(structure: core.ActionPhaseStructure) -> list[Any]:
    """
    :return: The events ending the structure's phases that are not failures.
    """
    return [
        event
        for phase in structure.defines_phase
        for event in phase.terminated_by
        if not isinstance(event, core.FailureEventType)
    ]


class MechanicalBoundaryMonitored(StructureRule[core.ActionPhaseStructure]):
    """
    Every mechanical event ending a phase, other than a failure, has a control point or
    an exemption.
    """

    @property
    def name(self) -> StrEnum:
        return CoreRule.MECHANICAL_BOUNDARY_MONITORED

    @property
    def reason(self) -> str:
        return (
            "a mechanical phase boundary has neither a control point nor an exemption"
        )

    def involved(
        self, fragment: core.ActionPhaseStructure
    ) -> list[core.MechanicalEventType]:
        """
        :return: The mechanical phase boundaries with neither a control point nor an
            exemption.
        """
        covered = {
            id(control_point.at_event_type)
            for control_point in fragment.defines_control_point
        } | {
            id(exemption.exempts_event)
            for exemption in fragment.defines_control_point_exemption
        }
        return [
            event
            for event in normal_boundaries(fragment)
            if isinstance(event, core.MechanicalEventType) and id(event) not in covered
        ]

    def is_violated_by(self, fragment: core.ActionPhaseStructure) -> bool:
        return bool(self.involved(fragment))


class ExemptionNamesNormalBoundary(StructureRule[core.ActionPhaseStructure]):
    """
    Every control point exemption names an event ending one of the structure's phases
    that is not a failure.
    """

    @property
    def name(self) -> StrEnum:
        return CoreRule.EXEMPTION_NAMES_NORMAL_BOUNDARY

    @property
    def reason(self) -> str:
        return "an exemption names an event that is not a normal phase boundary"

    def is_violated_by(self, fragment: core.ActionPhaseStructure) -> bool:
        boundaries = {id(event) for event in normal_boundaries(fragment)}
        return any(
            id(exemption.exempts_event) not in boundaries
            for exemption in fragment.defines_control_point_exemption
        )


# %% closure


class ComponentsDeclared(TypedRule[core.ActionPhaseStructure]):
    """
    Every fragment reachable from a structure, of a kind the structure can declare, is
    among the components the structure declares.
    """

    @property
    def name(self) -> StrEnum:
        return CoreRule.COMPONENTS_DECLARED

    @property
    def reason(self) -> str:
        return "reachable from the structure but not declared by it"

    @staticmethod
    def component_fields(
        structure: core.ActionPhaseStructure,
    ) -> dict[str, type[DescriptionEntity]]:
        """
        :return: The structure's fields declaring components, with the class each
            declares.
        """
        return Vocabulary().component_fields(type(structure))

    def declared(self, structure: core.ActionPhaseStructure) -> set[int]:
        """
        :return: The identities of the components the structure declares.
        """
        return {
            id(component)
            for name in self.component_fields(structure)
            for component in attrgetter(name)(structure)
        }

    @staticmethod
    def reachable(structure: core.ActionPhaseStructure) -> list[DescriptionEntity]:
        """
        :return: Every description fragment reachable from the structure through its
            fields, the structure excluded.
        """
        seen, reached, frontier = {id(structure)}, [], [structure]
        while frontier:
            current = frontier.pop()
            for declared in fields(current):
                if PropertyTerm.of_field(type(current), declared.name) is None:
                    continue
                value = attrgetter(declared.name)(current)
                for linked in value if isinstance(value, list) else [value]:
                    if isinstance(linked, DescriptionEntity) and id(linked) not in seen:
                        seen.add(id(linked))
                        reached.append(linked)
                        frontier.append(linked)
        return reached

    def violations(self, fragments: Sequence[DescriptionEntity]) -> list[Violation]:
        violations: list[Violation] = []
        for structure in self.subjects(fragments):
            declarable, declared = (
                tuple(self.component_fields(structure).values()),
                self.declared(structure),
            )
            violations.extend(
                self.violation(fragment)
                for fragment in self.reachable(structure)
                if isinstance(fragment, declarable) and id(fragment) not in declared
            )
        return violations
