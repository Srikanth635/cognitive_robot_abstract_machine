from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import pytest
from llm_reasoner.knowledge_updated.microworlds.flanagan.core.conditions import (
    ConditionRequirement,
    PhaseConditionType,
)
from llm_reasoner.knowledge_updated.microworlds.flanagan.core.control import (
    ModelUpdate,
    PredictionError,
)
from llm_reasoner.knowledge_updated.microworlds.flanagan.core.models import (
    ObjectRepresentation,
)
from llm_reasoner.knowledge_updated.microworlds.flanagan.core.parameters import (
    ObjectPropertyEstimate,
)
from llm_reasoner.knowledge_updated.microworlds.flanagan.core.prediction import (
    SensoryEventPrediction,
)
from llm_reasoner.knowledge_updated.microworlds.flanagan.core.rules import (
    CORE_ACCEPTANCE,
    CoreRule,
)
from llm_reasoner.knowledge_updated.microworlds.flanagan.core.value_partitions import (
    ConditionMode,
    EstimateUpdateDirection,
    MeasurementUnit,
    ParameterValueForm,
    PhysicalDimension,
    PredictionErrorKind,
    SensoryModality,
)
from llm_reasoner.knowledge_updated.ontological.acceptance import AcceptanceViolation
from llm_reasoner.knowledge_updated.ontological.documentation import (
    DocumentationReader,
)
from llm_reasoner.knowledge_updated.ontological.vocabulary import Severity

from .core_structure import (
    CoreStructure,
    QuantitativeMotorSetpoint,
    build_valid_structure,
    magnitude,
    window,
)

# %% helpers


@dataclass(eq=False)
class QuantitativeEstimate(ObjectPropertyEstimate):
    """
    An object property estimate that holds a magnitude, standing in for the concrete
    ones that later modules declare.
    """


def evaluate(parts: CoreStructure) -> tuple[AcceptanceViolation, ...]:
    """
    Run the whole core acceptance suite over a structure.
    """
    return CORE_ACCEPTANCE.evaluate(parts.fragments())


def add_update(parts: CoreStructure, errors: list[PredictionError]) -> ModelUpdate:
    """
    Declare a model update whose estimate its representation does not hold.
    """
    estimate = QuantitativeEstimate(
        value_form=ParameterValueForm.QUANTITATIVE,
        dimension=PhysicalDimension.MASS,
        parameter_range=magnitude(0.1, 0.5, MeasurementUnit.KILOGRAM),
        parameter_on_role=[parts.manipulandum],
    )
    representation = ObjectRepresentation(holds_estimate=[])
    update = ModelUpdate(
        triggered_by_error=errors,
        updates_estimate=[estimate],
        updates_representation=representation,
        update_direction=EstimateUpdateDirection.INCREASE,
    )
    parts.structure.defines_parameter.append(estimate)
    parts.structure.defines_internal_model.append(representation)
    parts.structure.defines_model_update.append(update)
    return update


def add_error(parts: CoreStructure, kind: PredictionErrorKind) -> PredictionError:
    """
    Declare a prediction error of a kind.
    """
    error = PredictionError(error_kind=kind)
    parts.structure.defines_error.append(error)
    return error


# %% one break for each rule


def break_gate_source(s: CoreStructure) -> None:
    s.grasp_to_hold.gated_by = [s.held]


def break_gate_target(s: CoreStructure) -> None:
    s.grasp.initiated_by = [s.held]


def break_one_event(s: CoreStructure) -> None:
    s.reach.initiated_by = [s.arrived]


def break_self_loop(s: CoreStructure) -> None:
    s.reach_to_grasp.to_phase = s.reach


def break_indistinguishable(s: CoreStructure) -> None:
    s.grasp.terminated_by = [s.arrived]


def break_failure_gate(s: CoreStructure) -> None:
    s.reach_to_grasp.gated_by = [s.slip]


def break_chain(s: CoreStructure) -> None:
    s.structure.defines_transition = [s.grasp_to_hold]


def break_grounding(s: CoreStructure) -> None:
    never = PhaseConditionType(label="never", condition_on_role=[s.actor])
    s.reach.has_condition_requirement = [
        ConditionRequirement(on_condition=never, mode=ConditionMode.AT_ENTRY)
    ]


def break_subgoal_boundary(s: CoreStructure) -> None:
    s.reach.aims_for.attainment_signalled_by = s.contact


def break_subgoal_failure(s: CoreStructure) -> None:
    s.reach.aims_for.attainment_signalled_by = s.slip


def break_margin_participants(s: CoreStructure) -> None:
    s.grip.parameter_on_role = [s.actor]
    s.threshold.parameter_on_role = [s.manipulandum]


def break_feedback_monitoring(s: CoreStructure) -> None:
    s.grasp.monitors_modality = []


def break_control_point_outcomes(s: CoreStructure) -> None:
    s.control_point.has_comparison = [s.match]


def break_mismatch_correction(s: CoreStructure) -> None:
    s.mismatch.invokes_correction = []


def break_invariant(s: CoreStructure) -> None:
    s.stable_grip.violation_signalled_by = None


def break_violation_event(s: CoreStructure) -> None:
    s.slip.is_failure_event = False


def break_range_bounds(s: CoreStructure) -> None:
    s.grip.parameter_range.lower_bound = 9.0


def break_window_bounds(s: CoreStructure) -> None:
    s.prediction.expected_timing.earliest_offset = 1.0


def break_unit_dimension(s: CoreStructure) -> None:
    s.grip.parameter_range.unit = MeasurementUnit.METRE


def break_missing_dimension(s: CoreStructure) -> None:
    s.margin.dimension = None


def break_value_form(s: CoreStructure) -> None:
    s.grip.value_form = ParameterValueForm.CATEGORICAL


def break_prediction_tolerance(s: CoreStructure) -> None:
    s.prediction.expected_magnitude = magnitude(1.0, 2.0, MeasurementUnit.NEWTON)
    s.prediction.tolerance = magnitude(0.0, 1.0, MeasurementUnit.METRE)


def break_trajectory_tolerance(s: CoreStructure) -> None:
    s.continuous.tolerance.unit = MeasurementUnit.METRE


def break_continuous_channels(s: CoreStructure) -> None:
    s.continuous.modality = SensoryModality.VISUAL


def break_controlled_variable(s: CoreStructure) -> None:
    s.grasp.active_parameters = [s.threshold, s.margin]


def break_control_law(s: CoreStructure) -> None:
    s.grasp.continuous_predictions = []


def break_match_success(s: CoreStructure) -> None:
    s.match.confirms_subgoal = None


def break_match_mismatch_output(s: CoreStructure) -> None:
    s.match.has_error = [s.absent]


def break_absent_with_observation(s: CoreStructure) -> None:
    s.mismatch.compares_observation = s.observation


def break_error_without_observation(s: CoreStructure) -> None:
    s.absent.error_kind = PredictionErrorKind.LATE


def break_mismatch_response(s: CoreStructure) -> None:
    s.mismatch.invokes_correction = []


def break_mismatch_success_output(s: CoreStructure) -> None:
    s.mismatch.confirms_subgoal = s.grasp.aims_for


def break_permission(s: CoreStructure) -> None:
    s.grasp_to_hold.permitted_by = [s.mismatch]


def break_unheld_estimate(s: CoreStructure) -> None:
    s.mismatch.produces_model_update = [add_update(s, [s.absent])]


def break_prediction_cause(s: CoreStructure) -> None:
    s.prediction.caused_by_event = s.slip


def break_undeclared_prediction(s: CoreStructure) -> None:
    other = SensoryEventPrediction(label="other prediction")
    s.control_point.has_prediction = [other]


def break_unclassified_correction(s: CoreStructure) -> None:
    s.mismatch.has_error = []


def break_unclassified_update(s: CoreStructure) -> None:
    late = add_error(s, PredictionErrorKind.LATE)
    s.mismatch.produces_model_update = [add_update(s, [late])]


def break_evidence(s: CoreStructure) -> None:
    s.observation.modality = SensoryModality.VISUAL


def break_uncontrolled_boundary(s: CoreStructure) -> None:
    s.structure.defines_control_point_exemption = [s.arrived_exemption]


def break_exemption(s: CoreStructure) -> None:
    s.arrived_exemption.exempts_event = s.slip


def break_unused_match(s: CoreStructure) -> None:
    s.grasp_to_hold.permitted_by = []


BREAKS: dict[CoreRule, Callable[[CoreStructure], None]] = {
    CoreRule.GATE_DOES_NOT_END_SOURCE_PHASE: break_gate_source,
    CoreRule.GATE_DOES_NOT_START_TARGET_PHASE: break_gate_target,
    CoreRule.PHASE_STARTS_AND_ENDS_AT_ONE_EVENT: break_one_event,
    CoreRule.TRANSITION_IS_A_SELF_LOOP: break_self_loop,
    CoreRule.PHASES_INDISTINGUISHABLE: break_indistinguishable,
    CoreRule.FAILURE_EVENT_GATES_TRANSITION: break_failure_gate,
    CoreRule.STRUCTURE_NOT_ONE_CHAIN: break_chain,
    CoreRule.CONDITION_NEVER_ESTABLISHED: break_grounding,
    CoreRule.SUBGOAL_NOT_SIGNALLED_BY_OWN_BOUNDARY: break_subgoal_boundary,
    CoreRule.SUBGOAL_SIGNALLED_BY_FAILURE: break_subgoal_failure,
    CoreRule.SAFETY_MARGIN_ON_DIFFERENT_PARTICIPANTS: break_margin_participants,
    CoreRule.FEEDBACK_PHASE_MONITORS_NOTHING: break_feedback_monitoring,
    CoreRule.CONTROL_POINT_HANDLES_ONE_OUTCOME: break_control_point_outcomes,
    CoreRule.MISMATCH_NAMES_NO_CORRECTION: break_mismatch_correction,
    CoreRule.INVARIANT_WITHOUT_VIOLATION_EVENT: break_invariant,
    CoreRule.VIOLATION_EVENT_NOT_A_FAILURE: break_violation_event,
    CoreRule.RANGE_BOUNDS_INVERTED: break_range_bounds,
    CoreRule.WINDOW_BOUNDS_INVERTED: break_window_bounds,
    CoreRule.PARAMETER_UNIT_DOES_NOT_MEASURE_ITS_DIMENSION: break_unit_dimension,
    CoreRule.QUANTITATIVE_PARAMETER_WITHOUT_DIMENSION: break_missing_dimension,
    CoreRule.PARAMETER_VALUE_DOES_NOT_MATCH_ITS_FORM: break_value_form,
    CoreRule.PREDICTION_TOLERANCE_IN_OTHER_DIMENSION: break_prediction_tolerance,
    CoreRule.TRAJECTORY_TOLERANCE_IN_OTHER_DIMENSION: break_trajectory_tolerance,
    CoreRule.CONTINUOUS_CHANNELS_DIFFER: break_continuous_channels,
    CoreRule.CONTROLLED_VARIABLE_INACTIVE_IN_PHASE: break_controlled_variable,
    CoreRule.CONTINUOUS_PHASE_HAS_NO_CONTROL_LAW: break_control_law,
    CoreRule.MATCH_LACKS_SUCCESS_OUTPUT: break_match_success,
    CoreRule.MATCH_DECLARES_MISMATCH_OUTPUT: break_match_mismatch_output,
    CoreRule.ABSENT_ERROR_WITH_OBSERVATION: break_absent_with_observation,
    CoreRule.OBSERVED_ERROR_WITHOUT_OBSERVATION: break_error_without_observation,
    CoreRule.MISMATCH_LACKS_RESPONSE: break_mismatch_response,
    CoreRule.MISMATCH_DECLARES_SUCCESS_OUTPUT: break_mismatch_success_output,
    CoreRule.TRANSITION_PERMISSION_INCOHERENT: break_permission,
    CoreRule.MODEL_UPDATE_CHANGES_UNHELD_ESTIMATE: break_unheld_estimate,
    CoreRule.PREDICTION_NOT_CAUSED_AT_CONTROL_POINT_EVENT: break_prediction_cause,
    CoreRule.COMPARISON_OF_UNDECLARED_PREDICTION: break_undeclared_prediction,
    CoreRule.CORRECTION_FOR_UNCLASSIFIED_ERROR: break_unclassified_correction,
    CoreRule.UPDATE_FOR_UNCLASSIFIED_ERROR: break_unclassified_update,
    CoreRule.COMPARISON_EVIDENCE_INCOMPATIBLE: break_evidence,
    CoreRule.BOUNDARY_WITHOUT_CONTROL_POINT_OR_EXEMPTION: break_uncontrolled_boundary,
    CoreRule.EXEMPTION_NOT_FOR_A_NORMAL_BOUNDARY: break_exemption,
    CoreRule.MATCH_PERMITS_NO_TRANSITION: break_unused_match,
}
"""
For each core rule, the smallest change to the valid structure that breaks it.
"""

WARNINGS = {
    CoreRule.PHASES_INDISTINGUISHABLE,
    CoreRule.CONDITION_NEVER_ESTABLISHED,
    CoreRule.SAFETY_MARGIN_ON_DIFFERENT_PARTICIPANTS,
    CoreRule.MISMATCH_NAMES_NO_CORRECTION,
}
"""
The rules whose breach is legal but probably unintended, so it is only a warning.
"""


# %% the valid structure


class TestValidStructure:
    """
    A structure built only from core concepts can satisfy every rule at once.
    """

    def test_valid_structure_raises_nothing(self) -> None:
        assert evaluate(build_valid_structure()) == ()

    def test_valid_structure_reaches_every_fragment_it_declares(self) -> None:
        parts = build_valid_structure()

        assert {parts.reach, parts.grasp, parts.hold} <= set(parts.fragments())


# %% each rule fires


class TestEveryRuleCanFire:
    """
    Breaking one thing in a valid structure makes the rule for it fire.
    """

    @pytest.mark.parametrize("rule", list(CoreRule), ids=lambda rule: rule.name)
    def test_rule_fires_when_its_condition_is_broken(self, rule: CoreRule) -> None:
        parts = build_valid_structure()
        BREAKS[rule](parts)

        assert rule in {violation.rule for violation in evaluate(parts)}

    def test_every_rule_has_a_break(self) -> None:
        assert set(BREAKS) == set(CoreRule)

    @pytest.mark.parametrize("rule", list(CoreRule), ids=lambda rule: rule.name)
    def test_rule_has_the_severity_it_is_meant_to_have(self, rule: CoreRule) -> None:
        parts = build_valid_structure()
        BREAKS[rule](parts)
        severities = {
            violation.severity
            for violation in evaluate(parts)
            if violation.rule is rule
        }

        expected = Severity.WARNING if rule in WARNINGS else Severity.ERROR
        assert severities == {expected}


# %% documentation


class TestCoreRulesAreDocumented:
    """
    Every rule says why it exists.
    """

    def test_every_rule_is_documented(self) -> None:
        assert DocumentationReader().find_undocumented_members(CoreRule) == ()
