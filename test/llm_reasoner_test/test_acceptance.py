"""
Tests of the acceptance rules that decide whether a generated hypothesis is admissible,
starting with the rule derived from the ontology's own restrictions.
"""

from __future__ import annotations

import inspect

import pytest
from rdflib import Namespace, URIRef
from krrood.utils import recursive_subclasses

from llm_reasoner.acceptance.core import (
    AbsentEventHasNoObservation,
    ComponentsDeclared,
    ErrorHasObservation,
    MatchHasNoResponse,
)
from llm_reasoner.acceptance.derived import (
    DerivedRule,
    RangesHold,
    RestrictionsHold,
    described,
)
from llm_reasoner.acceptance.pouring import (
    MaterialFitsFlowDynamics,
    PouringEventsSpecific,
    PouringPhasesOrdered,
    TerminationReasonHasEvent,
)
from llm_reasoner.acceptance.rules import AcceptanceRule, Severity
from llm_reasoner.acceptance.shapes import ShapesHold, StructureShapes
from llm_reasoner.acceptance.suite import predictive_manipulation_suite
from llm_reasoner.ontologies import PredictiveManipulationOntologyFile
from llm_reasoner.ontologies.predictive_manipulation import (
    core,
    gaze,
    object_dynamics,
    pouring,
    prehension,
)
from llm_reasoner.ontologies.terms import (
    ClassExpression,
    ComplementOf,
    ConceptReference,
    DescriptionEntity,
    IntersectionOf,
    NamedClass,
    PropertyRestriction,
    PropertyTerm,
    Quantifier,
)

# %% helpers


def property_iri(cls: type[DescriptionEntity], field_name: str) -> str:
    """
    Return the IRI of the ontology property a generated field stands for.
    """
    return PropertyTerm.of_field(cls, field_name).iri


def restriction_on(
    cls: type[DescriptionEntity], field_name: str, quantifier: Quantifier
) -> ClassExpression:
    """
    Return the class's restriction of the given kind on the property of a field.
    """
    (restriction,) = [
        restriction
        for restriction in cls.term.restrictions
        if isinstance(restriction, PropertyRestriction)
        and restriction.property_iri == property_iri(cls, field_name)
        and restriction.quantifier == quantifier
    ]
    return restriction


def violated(fragments: list[DescriptionEntity]) -> set[tuple[int, ClassExpression]]:
    """
    Return each violated restriction with the identity of the fragment violating it.
    """
    return {
        (id(violation.fragment), violation.expression)
        for violation in RestrictionsHold().violations(fragments)
    }


CORE_SHAPES = Namespace(PredictiveManipulationOntologyFile.CORE.shapes_namespace)
"""
The namespace of the core module's structural rules.
"""

PREHENSION_SHAPES = Namespace(
    PredictiveManipulationOntologyFile.PREHENSION.shapes_namespace
)
"""
The namespace of the prehension module's structural rules.
"""

GAZE_SHAPES = Namespace(PredictiveManipulationOntologyFile.GAZE.shapes_namespace)
"""
The namespace of the gaze module's structural rules.
"""

OBJECT_DYNAMICS_SHAPES = Namespace(
    PredictiveManipulationOntologyFile.OBJECT_DYNAMICS.shapes_namespace
)
"""
The namespace of the object dynamics module's structural rules.
"""


def shape_violators(shape: URIRef, fragments: list[DescriptionEntity]) -> list[object]:
    """
    Return the fragments violating one structural rule, in the order given.
    """
    found = {
        id(violation.fragment)
        for violation in ShapesHold().violations(fragments)
        if violation.shape == shape
    }
    return [fragment for fragment in fragments if id(fragment) in found]


def requirement(
    owner: type[DescriptionEntity],
    field_name: str,
    quantifier: Quantifier,
    filler: ClassExpression,
) -> PropertyRestriction:
    """
    Return the restriction an owner states on the property of one of its fields.
    """
    return PropertyRestriction(
        property_iri=property_iri(owner, field_name),
        quantifier=quantifier,
        filler=filler,
    )


def named(cls: type[DescriptionEntity]) -> NamedClass:
    """
    Return the named class a generated class stands for.
    """
    return NamedClass(iri=cls.term.iri)


def complete_phase() -> core.ActionPhase:
    """
    Return a generic phase that states everything the ontology requires of a phase.
    """
    return core.ActionPhase(
        label="phase",
        aims_for=core.Subgoal(),
        has_coupling_regime=core.CouplingRegime.COUPLED,
        terminated_by=[core.PhaseBoundaryEventType()],
        involves_role=[core.ManipulationRole()],
        controlled_by=core.ActionPhaseController(),
    )


# %% generated restrictions


class TestGeneratedRestrictions:
    """
    The generated classes carry the restrictions the ontology states.
    """

    def test_value_restriction_is_carried_by_its_class(self):
        assert (
            PropertyRestriction(
                property_iri=property_iri(
                    prehension.ForceLoadingPhase, "has_coupling_regime"
                ),
                quantifier=Quantifier.VALUE,
                value=core.CouplingRegime.TRANSITIONAL.value,
            )
            in prehension.ForceLoadingPhase.term.restrictions
        )


# %% restrictions read closed-world


class TestRestrictionsHold:
    """
    Every fragment meets the restrictions its classes state, what is absent counting as
    false.
    """

    def test_complete_phase_violates_nothing(self):
        assert violated([complete_phase()]) == set()

    def test_phase_without_terminating_event_violates_its_requirement(self):
        phase = complete_phase()
        phase.terminated_by = []
        assert violated([phase]) == {
            (
                id(phase),
                restriction_on(core.ActionPhase, "terminated_by", Quantifier.SOME),
            )
        }

    def test_load_phase_with_another_coupling_violates_its_fixed_value(self):
        phase = prehension.ForceLoadingPhase(
            aims_for=core.Subgoal(),
            has_coupling_regime=core.CouplingRegime.COUPLED,
            terminated_by=[core.PhaseBoundaryEventType()],
            involves_role=[core.ManipulationRole()],
            controlled_by=core.ActionPhaseController(),
            realizes_action_category=[
                ConceptReference(
                    iri=restriction_on(
                        prehension.ForceLoadingPhase,
                        "realizes_action_category",
                        Quantifier.VALUE,
                    ).value
                )
            ],
        )
        assert (
            id(phase),
            restriction_on(
                prehension.ForceLoadingPhase, "has_coupling_regime", Quantifier.VALUE
            ),
        ) in violated([phase])

    def test_friction_predicted_from_vision_violates_its_allowed_sources(self):
        estimate = prehension.ExpectedFrictionCoefficient(
            predicted_from=[core.PredictionSource.VISUAL_IDENTITY_CUE]
        )
        assert (
            id(estimate),
            restriction_on(
                prehension.ExpectedFrictionCoefficient,
                "predicted_from",
                Quantifier.ONLY,
            ),
        ) in violated([estimate])

    @pytest.mark.parametrize("predicts", ["both", "neither"])
    def test_prediction_must_predict_exactly_one_of_event_and_feature(self, predicts):
        prediction = core.SensoryEventPrediction(
            predicts_event=(
                core.PhaseBoundaryEventType() if predicts == "both" else None
            ),
            predicts_feature=core.SensoryFeature() if predicts == "both" else None,
        )
        violated_expressions = {
            expression for fragment, expression in violated([prediction])
        }
        assert violated_expressions & set(core.SensoryEventPrediction.term.restrictions)

    def test_mismatch_that_confirms_a_subgoal_is_classified_and_rejected(self):
        comparison = core.PredictionComparison(
            has_comparison_outcome=core.ComparisonOutcome.MISMATCH,
            confirms_subgoal=core.Subgoal(),
        )
        violated_expressions = {
            expression for fragment, expression in violated([comparison])
        }
        assert violated_expressions & set(core.MismatchingComparison.term.restrictions)

    def test_force_threshold_with_a_mass_unit_violates_its_dimension(self):
        threshold = prehension.LiftoffThreshold(
            has_value_form=core.ParameterValueForm.QUANTITATIVE,
            has_physical_dimension=core.PhysicalDimension.FORCE,
            has_parameter_range=core.MagnitudeRange(
                lower_bound=0.2, upper_bound=0.4, has_unit=core.MeasurementUnit.KILOGRAM
            ),
        )
        violated_expressions = {
            expression for fragment, expression in violated([threshold])
        }
        assert violated_expressions & set(core.ForceParameter.term.restrictions)

    def test_value_meeting_a_defined_class_counts_as_its_member(self):
        transition = core.PhaseTransition(
            permitted_by=[
                core.PredictionComparison(
                    has_comparison_outcome=core.ComparisonOutcome.MATCH
                )
            ]
        )
        assert (
            id(transition),
            restriction_on(core.PhaseTransition, "permitted_by", Quantifier.ONLY),
        ) not in violated([transition])

    def test_rule_names_itself(self):
        assert RestrictionsHold().name == DerivedRule.RESTRICTIONS_HOLD

    def test_defined_class_in_a_requirement_is_explained_by_its_definition(self):
        setpoint = pouring.TransferredQuantitySetpoint(
            has_value_form=core.ParameterValueForm.QUANTITATIVE
        )
        reasons = [
            violation.reason for violation in RestrictionsHold().violations([setpoint])
        ]
        definition = described(core.VolumeParameter.term.definition, False)
        assert any(definition in reason for reason in reasons)


# %% ranges


class TestRangesHold:
    """
    Every value a fragment holds is of the type its field's ontology property allows.
    """

    def test_invariant_signalled_by_a_non_failure_event_violates_the_range(self):
        requirement = core.ConditionRequirement(
            violation_signalled_by=core.PhaseBoundaryEventType()
        )
        assert [
            (violation.fragment, violation.field_name)
            for violation in RangesHold().violations([requirement])
        ] == [(requirement, "violation_signalled_by")]

    def test_value_outside_a_range_generated_in_a_later_module_is_reported(self):
        structure = core.ActionPhaseStructure(defines_gaze_fixation=[core.Subgoal()])
        assert [
            violation.field_name for violation in RangesHold().violations([structure])
        ] == ["defines_gaze_fixation"]

    def test_values_of_the_allowed_type_violate_nothing(self):
        requirement = core.ConditionRequirement(
            violation_signalled_by=prehension.ObjectCrush()
        )
        assert RangesHold().violations([requirement]) == []


# %% core rules


def violators(rule: AcceptanceRule, fragments: list[DescriptionEntity]) -> list[object]:
    """
    Return the fragments a rule reports, in the order it reports them.
    """
    return [violation.fragment for violation in rule.violations(fragments)]


class TestCoreRules:
    """
    The relationships between phases, events, transitions and conditions that a
    restriction on one class cannot state.
    """

    def test_rule_is_about_the_type_it_binds(self):
        assert ComponentsDeclared().subject_type is core.ActionPhaseStructure

    def test_inverted_magnitude_range_is_reported(self):
        inverted = core.MagnitudeRange(lower_bound=5.0, upper_bound=1.0)
        ordered = core.MagnitudeRange(lower_bound=1.0, upper_bound=5.0)
        assert shape_violators(CORE_SHAPES.RangeBoundsOrdered, [inverted, ordered]) == [
            inverted
        ]

    def test_inverted_temporal_window_is_reported(self):
        inverted = core.TemporalWindow(earliest_offset=300.0, latest_offset=100.0)
        ordered = core.TemporalWindow(earliest_offset=100.0, latest_offset=300.0)
        assert shape_violators(
            CORE_SHAPES.WindowBoundsOrdered, [inverted, ordered]
        ) == [inverted]

    def test_transition_gated_by_an_event_not_ending_its_source_is_reported(self):
        liftoff, contact = prehension.Liftoff(), prehension.GraspContact()
        load = prehension.ForceLoadingPhase(terminated_by=[liftoff])
        lift = prehension.LiftPhase(initiated_by=[liftoff])
        gated_correctly = core.PhaseTransition(
            from_phase=load, to_phase=lift, gated_by=[liftoff]
        )
        gated_wrongly = core.PhaseTransition(
            from_phase=load, to_phase=lift, gated_by=[contact]
        )
        assert shape_violators(
            CORE_SHAPES.TransitionGateEndsSource, [gated_correctly, gated_wrongly]
        ) == [gated_wrongly]

    def test_transition_gated_by_an_event_not_starting_its_target_is_reported(self):
        liftoff, clearance = prehension.Liftoff(), prehension.LiftClearanceReached()
        lift = prehension.LiftPhase(initiated_by=[liftoff])
        gated_correctly = core.PhaseTransition(to_phase=lift, gated_by=[liftoff])
        gated_wrongly = core.PhaseTransition(to_phase=lift, gated_by=[clearance])
        assert shape_violators(
            CORE_SHAPES.TransitionGateStartsTarget, [gated_correctly, gated_wrongly]
        ) == [gated_wrongly]

    def test_phase_started_and_ended_by_the_same_event_is_reported(self):
        event = core.PhaseBoundaryEventType()
        empty = core.ActionPhase(initiated_by=[event], terminated_by=[event])
        extended = core.ActionPhase(
            initiated_by=[event], terminated_by=[core.PhaseBoundaryEventType()]
        )
        assert shape_violators(CORE_SHAPES.PhaseHasExtent, [empty, extended]) == [empty]

    def test_transition_from_a_phase_to_itself_is_reported(self):
        phase, other = core.ActionPhase(), core.ActionPhase()
        loop = core.PhaseTransition(from_phase=phase, to_phase=phase)
        step = core.PhaseTransition(from_phase=phase, to_phase=other)
        assert shape_violators(CORE_SHAPES.TransitionIsNotALoop, [loop, step]) == [loop]

    def test_subgoal_signalled_by_an_event_not_ending_its_phase_is_reported(self):
        liftoff = prehension.Liftoff()
        signalled_elsewhere = core.ActionPhase(
            aims_for=core.Subgoal(attainment_signalled_by=prehension.GraspContact()),
            terminated_by=[liftoff],
        )
        signalled_at_end = core.ActionPhase(
            aims_for=core.Subgoal(attainment_signalled_by=liftoff),
            terminated_by=[liftoff],
        )
        assert shape_violators(
            CORE_SHAPES.SubgoalSignalledByPhaseEnd,
            [signalled_elsewhere, signalled_at_end],
        ) == [signalled_elsewhere]

    def test_invariant_without_a_violation_event_is_reported(self):
        unguarded = core.ConditionRequirement(
            has_condition_mode=core.ConditionMode.THROUGHOUT,
            has_condition_polarity=core.ConditionPolarity.MUST_HOLD,
        )
        guarded = core.ConditionRequirement(
            has_condition_mode=core.ConditionMode.THROUGHOUT,
            has_condition_polarity=core.ConditionPolarity.MUST_HOLD,
            violation_signalled_by=prehension.ObjectCrush(),
        )
        precondition = core.ConditionRequirement(
            has_condition_mode=core.ConditionMode.AT_ENTRY,
            has_condition_polarity=core.ConditionPolarity.MUST_HOLD,
        )
        required = requirement(
            core.InvariantRequirement,
            "violation_signalled_by",
            Quantifier.SOME,
            named(core.FailureEventType),
        )
        found = violated([unguarded, guarded, precondition])
        assert [
            requirement_
            for requirement_ in (unguarded, guarded, precondition)
            if (id(requirement_), required) in found
        ] == [unguarded]

    def test_safety_margin_relating_different_participants_is_reported(self):
        gripper, carton = core.ManipulationRole(), core.ManipulationRole()
        mismatched = core.SafetyMargin(
            margin_on=prehension.GripForceSetpoint(parameter_on_role=[gripper]),
            margin_over=prehension.SlipThreshold(parameter_on_role=[carton]),
        )
        matched = core.SafetyMargin(
            margin_on=prehension.GripForceSetpoint(parameter_on_role=[gripper]),
            margin_over=prehension.SlipThreshold(parameter_on_role=[gripper]),
        )
        assert shape_violators(
            CORE_SHAPES.SafetyMarginOnOneParticipant, [mismatched, matched]
        ) == [mismatched]

    def test_phase_realizing_the_task_it_belongs_to_is_reported(self):
        pouring_task = ConceptReference(
            iri="http://aicor.knowledge/l2-ontology.owl#Pouring"
        )
        circular_phase = core.ActionPhase(realizes_action_category=[pouring_task])
        circular = core.ActionPhaseStructure(
            decomposes_task=pouring_task, defines_phase=[circular_phase]
        )
        proper = core.ActionPhaseStructure(
            decomposes_task=pouring_task, defines_phase=[core.ActionPhase()]
        )
        assert shape_violators(
            CORE_SHAPES.TaskIsNotItsOwnPhase, [circular, proper]
        ) == [circular]

    def test_structure_with_two_entry_phases_is_reported(self):
        first, second, third = (
            core.ActionPhase(),
            core.ActionPhase(),
            core.ActionPhase(),
        )
        branching_in = core.ActionPhaseStructure(
            defines_phase=[first, second, third],
            defines_transition=[
                core.PhaseTransition(from_phase=first, to_phase=third),
                core.PhaseTransition(from_phase=second, to_phase=third),
            ],
        )
        chained = core.ActionPhaseStructure(
            defines_phase=[first, second, third],
            defines_transition=[
                core.PhaseTransition(from_phase=first, to_phase=second),
                core.PhaseTransition(from_phase=second, to_phase=third),
            ],
        )
        assert shape_violators(CORE_SHAPES.OneEntryPhase, [branching_in, chained]) == [
            branching_in
        ]

    def test_structure_with_an_unconnected_phase_is_reported(self):
        first, second, orphan = (
            core.ActionPhase(),
            core.ActionPhase(),
            core.ActionPhase(),
        )
        with_orphan = core.ActionPhaseStructure(
            defines_phase=[first, second, orphan],
            defines_transition=[
                core.PhaseTransition(from_phase=first, to_phase=second)
            ],
        )
        assert shape_violators(CORE_SHAPES.NoOrphanPhase, [with_orphan]) == [
            with_orphan
        ]

    def test_two_phases_ending_at_the_same_events_are_reported(self):
        liftoff = prehension.Liftoff()
        indistinct = core.ActionPhaseStructure(
            defines_phase=[
                core.ActionPhase(terminated_by=[liftoff]),
                core.ActionPhase(terminated_by=[liftoff]),
            ]
        )
        distinct = core.ActionPhaseStructure(
            defines_phase=[
                core.ActionPhase(terminated_by=[liftoff]),
                core.ActionPhase(terminated_by=[prehension.Touchdown()]),
            ]
        )
        assert shape_violators(
            CORE_SHAPES.PhaseEndingsDistinct, [indistinct, distinct]
        ) == [indistinct]

    def test_precondition_no_phase_establishes_is_reported(self):
        carton_held = core.PhaseConditionType()
        ungrounded = core.ActionPhaseStructure(
            defines_phase=[
                core.ActionPhase(
                    has_condition_requirement=[
                        core.ConditionRequirement(
                            has_condition_type=[carton_held],
                            has_condition_mode=core.ConditionMode.AT_ENTRY,
                            has_condition_polarity=core.ConditionPolarity.MUST_HOLD,
                        )
                    ]
                )
            ]
        )
        grounded = core.ActionPhaseStructure(
            defines_phase=[
                core.ActionPhase(aims_for=core.Subgoal(targets_condition=carton_held)),
                core.ActionPhase(
                    has_condition_requirement=[
                        core.ConditionRequirement(
                            has_condition_type=[carton_held],
                            has_condition_mode=core.ConditionMode.AT_ENTRY,
                            has_condition_polarity=core.ConditionPolarity.MUST_HOLD,
                        )
                    ]
                ),
            ]
        )
        assert shape_violators(
            CORE_SHAPES.PreconditionsGrounded, [ungrounded, grounded]
        ) == [ungrounded]

    def test_ungrounded_requirement_is_named_by_the_violation(self):
        requirement = core.ConditionRequirement(
            has_condition_type=[core.PhaseConditionType()],
            has_condition_mode=core.ConditionMode.THROUGHOUT,
            has_condition_polarity=core.ConditionPolarity.MUST_HOLD,
        )
        structure = core.ActionPhaseStructure(
            defines_phase=[core.ActionPhase(has_condition_requirement=[requirement])]
        )
        (violation,) = ShapesHold(
            checked_shapes=frozenset({CORE_SHAPES.PreconditionsGrounded})
        ).violations([structure])
        assert violation.involved == (requirement,)


class TestFailureEventsInTheOntology:
    """
    The ontology itself rules out failures gating a transition or signalling success.
    """

    def test_transition_gated_by_a_failure_violates_its_requirement(self):
        transition = core.PhaseTransition(gated_by=[core.FailureEventType()])
        assert (
            id(transition),
            requirement(
                core.PhaseTransition,
                "gated_by",
                Quantifier.ONLY,
                ComplementOf(operand=named(core.FailureEventType)),
            ),
        ) in violated([transition])

    def test_subgoal_signalled_by_a_failure_violates_its_requirement(self):
        subgoal = core.Subgoal(attainment_signalled_by=core.FailureEventType())
        assert (
            id(subgoal),
            requirement(
                core.Subgoal,
                "attainment_signalled_by",
                Quantifier.ONLY,
                ComplementOf(operand=named(core.FailureEventType)),
            ),
        ) in violated([subgoal])

    def test_transition_gated_by_a_boundary_event_meets_its_requirement(self):
        transition = core.PhaseTransition(gated_by=[prehension.Liftoff()])
        assert (
            id(transition),
            requirement(
                core.PhaseTransition,
                "gated_by",
                Quantifier.ONLY,
                ComplementOf(operand=named(core.FailureEventType)),
            ),
        ) not in violated([transition])

    def test_subgoal_signalled_by_a_boundary_event_meets_its_requirement(self):
        subgoal = core.Subgoal(attainment_signalled_by=prehension.Liftoff())
        assert (
            id(subgoal),
            requirement(
                core.Subgoal,
                "attainment_signalled_by",
                Quantifier.ONLY,
                ComplementOf(operand=named(core.FailureEventType)),
            ),
        ) not in violated([subgoal])


# %% comparisons and control points


def matching_comparison(**fields: object) -> core.PredictionComparison:
    """
    Return a comparison whose observation agrees with its prediction.
    """
    return core.PredictionComparison(
        has_comparison_outcome=core.ComparisonOutcome.MATCH, **fields
    )


def mismatching_comparison(**fields: object) -> core.PredictionComparison:
    """
    Return a comparison whose observation disagrees with its prediction.
    """
    return core.PredictionComparison(
        has_comparison_outcome=core.ComparisonOutcome.MISMATCH, **fields
    )


def prediction_error(kind: core.PredictionErrorKind) -> core.PredictionError:
    """
    Return a prediction error of the given kind.
    """
    return core.PredictionError(has_error_kind=kind)


class TestComparisonRules:
    """
    Comparisons, their errors and responses, and the control points holding them.
    """

    def test_match_without_an_observation_is_reported(self):
        unobserved = matching_comparison()
        observed = matching_comparison(
            compares_observation=core.SensoryObservationType()
        )
        required = requirement(
            core.MatchingComparison,
            "compares_observation",
            Quantifier.SOME,
            named(core.SensoryObservationType),
        )
        found = violated([unobserved, observed])
        assert (id(unobserved), required) in found
        assert (id(observed), required) not in found

    def test_match_with_a_correction_is_reported(self):
        correcting = matching_comparison(invokes_correction=[core.CorrectiveAction()])
        plain = matching_comparison()
        assert violators(MatchHasNoResponse(), [correcting, plain]) == [correcting]

    def test_absent_event_with_an_observation_is_reported(self):
        contradictory = mismatching_comparison(
            compares_observation=core.SensoryObservationType(),
            has_prediction_error=[
                prediction_error(core.PredictionErrorKind.ABSENT_EVENT)
            ],
        )
        coherent = mismatching_comparison(
            has_prediction_error=[
                prediction_error(core.PredictionErrorKind.ABSENT_EVENT)
            ]
        )
        assert violators(AbsentEventHasNoObservation(), [contradictory, coherent]) == [
            contradictory
        ]

    def test_error_other_than_absence_without_an_observation_is_reported(self):
        unobserved = mismatching_comparison(
            has_prediction_error=[
                prediction_error(core.PredictionErrorKind.EARLY_EVENT)
            ]
        )
        observed = mismatching_comparison(
            compares_observation=core.SensoryObservationType(),
            has_prediction_error=[
                prediction_error(core.PredictionErrorKind.EARLY_EVENT)
            ],
        )
        assert violators(ErrorHasObservation(), [unobserved, observed]) == [unobserved]

    def test_permission_confirming_another_subgoal_is_reported(self):
        subgoal = core.Subgoal()
        source = core.ActionPhase(aims_for=subgoal)
        wrong = core.PhaseTransition(
            from_phase=source,
            permitted_by=[matching_comparison(confirms_subgoal=core.Subgoal())],
        )
        right = core.PhaseTransition(
            from_phase=source,
            permitted_by=[matching_comparison(confirms_subgoal=subgoal)],
        )
        assert shape_violators(
            CORE_SHAPES.PermissionFitsTransition, [wrong, right]
        ) == [wrong]

    def test_permission_supplying_another_phases_state_is_reported(self):
        target = core.ActionPhase()
        wrong = core.PhaseTransition(
            to_phase=target,
            permitted_by=[
                matching_comparison(
                    supplies_state_estimate=core.InitialStateEstimate(
                        for_phase=core.ActionPhase()
                    )
                )
            ],
        )
        right = core.PhaseTransition(
            to_phase=target,
            permitted_by=[
                matching_comparison(
                    supplies_state_estimate=core.InitialStateEstimate(for_phase=target)
                )
            ],
        )
        assert shape_violators(
            CORE_SHAPES.PermissionFitsTransition, [wrong, right]
        ) == [wrong]

    def test_update_of_an_estimate_its_representation_does_not_hold_is_reported(self):
        mass, friction = (
            prehension.ExpectedMass(),
            prehension.ExpectedFrictionCoefficient(),
        )
        model = object_dynamics.ObjectDynamicsModel(holds_estimate=[mass])
        foreign = core.ModelUpdate(
            updates_representation=model, updates_estimate=[friction]
        )
        held = core.ModelUpdate(updates_representation=model, updates_estimate=[mass])
        assert shape_violators(CORE_SHAPES.UpdatedEstimateIsHeld, [foreign, held]) == [
            foreign
        ]

    def test_control_point_prediction_caused_elsewhere_is_reported(self):
        liftoff = prehension.Liftoff()
        elsewhere = core.SensorimotorControlPoint(
            at_event_type=liftoff,
            has_prediction=[
                core.SensoryEventPrediction(caused_by_event=prehension.GraspContact())
            ],
        )
        here = core.SensorimotorControlPoint(
            at_event_type=liftoff,
            has_prediction=[core.SensoryEventPrediction(caused_by_event=liftoff)],
        )
        assert shape_violators(
            CORE_SHAPES.ControlPointPredictsItsEvent, [elsewhere, here]
        ) == [elsewhere]

    def test_control_point_comparing_an_undeclared_prediction_is_reported(self):
        declared = core.SensoryEventPrediction()
        foreign = core.SensorimotorControlPoint(
            has_prediction=[declared],
            has_comparison=[
                matching_comparison(compares_prediction=core.SensoryEventPrediction())
            ],
        )
        own = core.SensorimotorControlPoint(
            has_prediction=[declared],
            has_comparison=[matching_comparison(compares_prediction=declared)],
        )
        assert shape_violators(
            CORE_SHAPES.ControlPointComparesItsPredictions, [foreign, own]
        ) == [foreign]

    def test_correction_for_an_unclassified_error_is_reported(self):
        error = prediction_error(core.PredictionErrorKind.EARLY_EVENT)
        foreign = mismatching_comparison(
            has_prediction_error=[error],
            invokes_correction=[
                core.CorrectiveAction(
                    triggered_by_error=[
                        prediction_error(core.PredictionErrorKind.LATE_EVENT)
                    ]
                )
            ],
        )
        own = mismatching_comparison(
            has_prediction_error=[error],
            invokes_correction=[core.CorrectiveAction(triggered_by_error=[error])],
        )
        assert shape_violators(
            CORE_SHAPES.CorrectionAnswersClassifiedError, [foreign, own]
        ) == [foreign]

    def test_model_update_for_an_unclassified_error_is_reported(self):
        error = prediction_error(core.PredictionErrorKind.EARLY_EVENT)
        foreign = mismatching_comparison(
            has_prediction_error=[error],
            produces_model_update=[
                core.ModelUpdate(
                    triggered_by_error=[
                        prediction_error(core.PredictionErrorKind.LATE_EVENT)
                    ]
                )
            ],
        )
        own = mismatching_comparison(
            has_prediction_error=[error],
            produces_model_update=[core.ModelUpdate(triggered_by_error=[error])],
        )
        assert shape_violators(
            CORE_SHAPES.ModelUpdateAnswersClassifiedError, [foreign, own]
        ) == [foreign]

    def test_control_point_without_mismatch_handling_is_reported(self):
        match_only = core.SensorimotorControlPoint(
            has_comparison=[matching_comparison()]
        )
        both = core.SensorimotorControlPoint(
            has_comparison=[matching_comparison(), mismatching_comparison()]
        )
        required = requirement(
            core.SensorimotorControlPoint,
            "has_comparison",
            Quantifier.SOME,
            named(core.MismatchingComparison),
        )
        found = violated([match_only, both])
        assert (id(match_only), required) in found
        assert (id(both), required) not in found

    def test_control_point_without_match_handling_is_reported(self):
        mismatch_only = core.SensorimotorControlPoint(
            has_comparison=[mismatching_comparison()]
        )
        assert (
            id(mismatch_only),
            requirement(
                core.SensorimotorControlPoint,
                "has_comparison",
                Quantifier.SOME,
                named(core.MatchingComparison),
            ),
        ) in violated([mismatch_only])

    def test_mismatch_handling_without_a_correction_is_reported(self):
        uncorrected = core.SensorimotorControlPoint(
            has_comparison=[
                mismatching_comparison(produces_model_update=[core.ModelUpdate()])
            ]
        )
        corrected = core.SensorimotorControlPoint(
            has_comparison=[
                mismatching_comparison(invokes_correction=[core.CorrectiveAction()])
            ]
        )
        assert shape_violators(
            CORE_SHAPES.MismatchInvokesCorrection, [uncorrected, corrected]
        ) == [uncorrected]

    def test_comparison_of_different_modalities_is_reported(self):
        lip = core.ManipulationRole()
        incompatible = matching_comparison(
            compares_prediction=core.SensoryEventPrediction(
                has_modality=core.SensoryModality.AUDITORY, expected_on_role=[lip]
            ),
            compares_observation=core.SensoryObservationType(
                has_modality=core.SensoryModality.VISUAL, observed_on_role=[lip]
            ),
        )
        feature = core.SensoryFeature()
        compatible = matching_comparison(
            compares_prediction=core.SensoryEventPrediction(
                has_modality=core.SensoryModality.AUDITORY,
                expected_on_role=[lip],
                predicts_feature=feature,
            ),
            compares_observation=core.SensoryObservationType(
                has_modality=core.SensoryModality.AUDITORY,
                observed_on_role=[lip],
                observes_feature=feature,
            ),
        )
        assert shape_violators(
            CORE_SHAPES.EvidenceCompatible, [incompatible, compatible]
        ) == [incompatible]


class TestContinuousPredictionRules:
    """
    Continuous predictions compare like with like and control an active setpoint.
    """

    def test_continuous_prediction_compared_in_another_modality_is_reported(self):
        mismatched = core.ContinuousSignalPrediction(
            has_modality=core.SensoryModality.VISUAL,
            compared_with=core.SensoryObservationType(
                has_modality=core.SensoryModality.TACTILE
            ),
        )
        matched = core.ContinuousSignalPrediction(
            has_modality=core.SensoryModality.VISUAL,
            compared_with=core.SensoryObservationType(
                has_modality=core.SensoryModality.VISUAL
            ),
        )
        assert shape_violators(
            CORE_SHAPES.ContinuousPredictionComparesItsModality, [mismatched, matched]
        ) == [mismatched]

    def test_tolerance_in_another_dimension_than_the_trajectory_is_reported(self):
        trajectory = core.SignalTrajectory(
            has_expected_range=core.MagnitudeRange(has_unit=core.MeasurementUnit.NEWTON)
        )
        mismatched = core.ContinuousSignalPrediction(
            predicts_trajectory=trajectory,
            has_tolerance=core.MagnitudeRange(has_unit=core.MeasurementUnit.MILLILITRE),
        )
        matched = core.ContinuousSignalPrediction(
            predicts_trajectory=trajectory,
            has_tolerance=core.MagnitudeRange(has_unit=core.MeasurementUnit.NEWTON),
        )
        assert shape_violators(
            CORE_SHAPES.ToleranceMatchesTrajectory, [mismatched, matched]
        ) == [mismatched]

    def test_phase_whose_controller_regulates_an_inactive_setpoint_is_reported(self):
        flow_rate = pouring.FlowRateSetpoint()
        controller = core.ActionPhaseController(
            has_continuous_prediction=[
                core.ContinuousSignalPrediction(controls_setpoint=flow_rate)
            ]
        )
        inactive = core.ActionPhase(controlled_by=controller)
        active = core.ActionPhase(
            controlled_by=controller, has_active_parameter=[flow_rate]
        )
        assert shape_violators(
            CORE_SHAPES.ControlledSetpointIsActive, [inactive, active]
        ) == [inactive]


class TestStructureControlRules:
    """
    A structure monitors its mechanical phase boundaries and declares what it uses.
    """

    def test_unmonitored_mechanical_boundary_is_reported(self):
        liftoff = prehension.Liftoff()
        unmonitored = core.ActionPhaseStructure(
            defines_phase=[core.ActionPhase(terminated_by=[liftoff])]
        )
        monitored = core.ActionPhaseStructure(
            defines_phase=[core.ActionPhase(terminated_by=[liftoff])],
            defines_control_point=[
                core.SensorimotorControlPoint(at_event_type=liftoff)
            ],
        )
        exempted = core.ActionPhaseStructure(
            defines_phase=[core.ActionPhase(terminated_by=[liftoff])],
            defines_control_point_exemption=[
                core.BoundaryControlExemption(exempts_event=liftoff)
            ],
        )
        assert shape_violators(
            CORE_SHAPES.MechanicalBoundaryMonitored, [unmonitored, monitored, exempted]
        ) == [unmonitored]

    def test_unmonitored_boundary_event_is_named_by_the_violation(self):
        liftoff, touchdown = prehension.Liftoff(), prehension.Touchdown()
        structure = core.ActionPhaseStructure(
            defines_phase=[core.ActionPhase(terminated_by=[liftoff, touchdown])],
            defines_control_point=[
                core.SensorimotorControlPoint(at_event_type=touchdown)
            ],
        )
        (violation,) = [
            violation
            for violation in ShapesHold().violations([structure])
            if violation.shape == CORE_SHAPES.MechanicalBoundaryMonitored
        ]
        assert violation.involved == (liftoff,)

    def test_exemption_of_an_event_that_is_no_boundary_is_reported(self):
        liftoff = prehension.Liftoff()
        misplaced = core.ActionPhaseStructure(
            defines_phase=[core.ActionPhase(terminated_by=[liftoff])],
            defines_control_point_exemption=[
                core.BoundaryControlExemption(exempts_event=prehension.Touchdown())
            ],
        )
        placed = core.ActionPhaseStructure(
            defines_phase=[core.ActionPhase(terminated_by=[liftoff])],
            defines_control_point_exemption=[
                core.BoundaryControlExemption(exempts_event=liftoff)
            ],
        )
        assert shape_violators(
            CORE_SHAPES.ExemptionNamesNormalBoundary, [misplaced, placed]
        ) == [misplaced]

    def test_fragment_reachable_but_not_declared_is_reported(self):
        subgoal = core.Subgoal()
        undeclared_phase = core.ActionPhase(aims_for=subgoal)
        structure = core.ActionPhaseStructure(
            defines_subgoal=[subgoal],
            defines_transition=[core.PhaseTransition(to_phase=undeclared_phase)],
        )
        reported = violators(ComponentsDeclared(), [structure])
        assert undeclared_phase in reported and subgoal not in reported

    def test_fragment_no_field_can_declare_is_not_reported(self):
        tolerance = core.MagnitudeRange()
        structure = core.ActionPhaseStructure(
            defines_continuous_prediction=[
                core.ContinuousSignalPrediction(has_tolerance=tolerance)
            ],
        )
        assert tolerance not in violators(ComponentsDeclared(), [structure])


# %% object dynamics


class TestObjectDynamicsRules:
    """
    The command-to-prediction pathway, represented estimates, memories and transfers.
    """

    def test_composition_prediction_from_another_forward_model_is_reported(self):
        forward_model = core.ForwardModel()
        foreign = object_dynamics.InternalModelComposition(
            uses_forward_model=forward_model,
            predicts_sensory_consequence=[
                core.SensoryEventPrediction(generated_by=[core.ForwardModel()])
            ],
        )
        own = object_dynamics.InternalModelComposition(
            uses_forward_model=forward_model,
            predicts_sensory_consequence=[
                core.SensoryEventPrediction(generated_by=[forward_model])
            ],
        )
        assert shape_violators(
            OBJECT_DYNAMICS_SHAPES.CompositionPredictsWithItsForwardModel,
            [foreign, own],
        ) == [foreign]

    def test_composition_whose_dynamics_ignores_its_copy_is_reported(self):
        copy = object_dynamics.EfferenceCopy()
        ignoring = object_dynamics.InternalModelComposition(
            uses_efference_copy=[copy],
            uses_dynamics_model=object_dynamics.ObjectDynamicsModel(),
        )
        using = object_dynamics.InternalModelComposition(
            uses_efference_copy=[copy],
            uses_dynamics_model=object_dynamics.ObjectDynamicsModel(
                uses_efference_copy=[copy]
            ),
        )
        assert shape_violators(
            OBJECT_DYNAMICS_SHAPES.CompositionDynamicsUsesItsCopy, [ignoring, using]
        ) == [ignoring]

    def test_estimate_no_representation_holds_is_reported(self):
        mass = prehension.ExpectedMass()
        unowned = object_dynamics.ObjectDynamicsPhaseStructure(
            defines_parameter=[mass],
            defines_internal_model=[object_dynamics.ObjectDynamicsModel()],
        )
        owned = object_dynamics.ObjectDynamicsPhaseStructure(
            defines_parameter=[mass],
            defines_internal_model=[
                object_dynamics.ObjectDynamicsModel(holds_estimate=[mass])
            ],
        )
        assert shape_violators(
            OBJECT_DYNAMICS_SHAPES.EstimateHeldByARepresentation, [unowned, owned]
        ) == [unowned]

    def test_estimate_serving_control_and_judgment_is_reported(self):
        mass = prehension.ExpectedMass()
        shared = object_dynamics.ObjectDynamicsPhaseStructure(
            defines_internal_model=[
                object_dynamics.ObjectDynamicsModel(
                    has_purpose=object_dynamics.RepresentationPurpose.SENSORIMOTOR_CONTROL,
                    holds_estimate=[mass],
                ),
                object_dynamics.PerceptualRepresentation(
                    has_purpose=object_dynamics.RepresentationPurpose.PERCEPTUAL_JUDGMENT,
                    holds_estimate=[mass],
                ),
            ]
        )
        separate = object_dynamics.ObjectDynamicsPhaseStructure(
            defines_internal_model=[
                object_dynamics.ObjectDynamicsModel(
                    has_purpose=object_dynamics.RepresentationPurpose.SENSORIMOTOR_CONTROL,
                    holds_estimate=[mass],
                ),
                object_dynamics.PerceptualRepresentation(
                    has_purpose=object_dynamics.RepresentationPurpose.PERCEPTUAL_JUDGMENT,
                    holds_estimate=[prehension.ExpectedMass()],
                ),
            ]
        )
        assert shape_violators(
            OBJECT_DYNAMICS_SHAPES.EstimateServesOnePurpose, [shared, separate]
        ) == [shared]

    def test_dissociation_of_one_estimate_with_itself_is_reported(self):
        mass = prehension.ExpectedMass()
        perceived = prehension.ExpectedMass()
        sensorimotor = object_dynamics.ObjectDynamicsModel(holds_estimate=[mass])
        perceptual = object_dynamics.PerceptualRepresentation(
            holds_estimate=[mass, perceived]
        )
        same = object_dynamics.RepresentationDissociation(
            has_sensorimotor_representation=sensorimotor,
            has_perceptual_representation=perceptual,
            sensorimotor_estimate=mass,
            perceptual_estimate=mass,
        )
        distinct = object_dynamics.RepresentationDissociation(
            has_sensorimotor_representation=sensorimotor,
            has_perceptual_representation=perceptual,
            sensorimotor_estimate=mass,
            perceptual_estimate=perceived,
        )
        assert shape_violators(
            OBJECT_DYNAMICS_SHAPES.DissociationCoherent, [same, distinct]
        ) == [same]

    def test_retained_update_outside_its_memory_is_reported(self):
        model = object_dynamics.ObjectDynamicsModel()
        outside = object_dynamics.RetainedModelUpdate(
            updates_representation=model,
            persists_in=object_dynamics.SensorimotorMemory(),
        )
        inside = object_dynamics.RetainedModelUpdate(
            updates_representation=model,
            persists_in=object_dynamics.SensorimotorMemory(
                stores_representation=[model]
            ),
        )
        assert shape_violators(
            OBJECT_DYNAMICS_SHAPES.RetainedUpdateInItsMemory, [outside, inside]
        ) == [outside]

    def test_transfer_between_unmatched_perturbations_is_reported(self):
        load = object_dynamics.PerturbationKind.LOAD_PERTURBATION
        first = object_dynamics.DynamicsPerturbation(has_perturbation_kind=load)
        second = object_dynamics.DynamicsPerturbation(has_perturbation_kind=load)
        third = object_dynamics.DynamicsPerturbation(has_perturbation_kind=load)
        first.matched_with = [second]
        unmatched = object_dynamics.DynamicsTransfer(
            from_perturbation=first, to_perturbation=third
        )
        matched = object_dynamics.DynamicsTransfer(
            from_perturbation=second, to_perturbation=first
        )
        assert shape_violators(
            OBJECT_DYNAMICS_SHAPES.TransferComparesMatchedPerturbations,
            [unmatched, matched],
        ) == [unmatched]

    def test_representation_missing_a_role_its_estimates_concern_is_reported(self):
        carton = core.ManipulationRole()
        estimate = prehension.ExpectedMass(parameter_on_role=[carton])
        uncovered = object_dynamics.ObjectDynamicsModel(holds_estimate=[estimate])
        covered = object_dynamics.ObjectDynamicsModel(
            holds_estimate=[estimate], represents_role=[carton]
        )
        assert shape_violators(
            OBJECT_DYNAMICS_SHAPES.RepresentationCoversEstimatedRoles,
            [uncovered, covered],
        ) == [uncovered]


# %% prehension


class TestPrehensionRules:
    """
    Grip and load control, and the boundaries of the putting-down phases.
    """

    def test_safety_margin_on_another_grip_force_is_reported(self):
        grip = prehension.GripForceSetpoint()
        unrelated = prehension.GripLoadCoupling(
            grip_force_response=grip,
            has_safety_margin=core.SafetyMargin(
                margin_on=prehension.GripForceSetpoint()
            ),
        )
        related = prehension.GripLoadCoupling(
            grip_force_response=grip,
            has_safety_margin=core.SafetyMargin(margin_on=grip),
        )
        assert shape_violators(
            PREHENSION_SHAPES.SafetyMarginOnCouplingResponse, [unrelated, related]
        ) == [unrelated]

    def test_adjustment_active_after_the_predicted_event_is_reported(self):
        flow_onset = pouring.FlowOnset()
        tilt = pouring.TiltToFlowPhase(terminated_by=[flow_onset])
        regulate = pouring.FlowRegulationPhase(initiated_by=[flow_onset])
        prediction = core.SensoryEventPrediction(caused_by_event=flow_onset)
        late = prehension.AnticipatoryGripAdjustment(
            triggered_by_prediction=prediction, adjustment_active_in=[regulate]
        )
        anticipatory = prehension.AnticipatoryGripAdjustment(
            triggered_by_prediction=prediction, adjustment_active_in=[tilt]
        )
        assert shape_violators(
            PREHENSION_SHAPES.AdjustmentPrecedesPredictedEvent, [late, anticipatory]
        ) == [late]

    def test_prehension_structure_with_a_generic_phase_is_reported(self):
        generic = prehension.PrehensionPhaseStructure(
            defines_phase=[core.ActionPhase()]
        )
        specific = prehension.PrehensionPhaseStructure(
            defines_phase=[prehension.LiftPhase()]
        )
        assert shape_violators(
            PREHENSION_SHAPES.PrehensionPhasesSpecific, [generic, specific]
        ) == [generic]


class TestPuttingDownBoundariesInTheOntology:
    """
    The ontology itself states the event each putting-down phase starts or ends at.
    """

    @pytest.mark.parametrize(
        "phase_type, field_name, event_type",
        [
            (prehension.ReplacePhase, "terminated_by", prehension.Touchdown),
            (prehension.UnloadPhase, "terminated_by", prehension.ForcesUnloaded),
            (prehension.ContactReleasePhase, "terminated_by", prehension.GraspRelease),
            (prehension.WithdrawPhase, "initiated_by", prehension.GraspRelease),
            (prehension.WithdrawPhase, "terminated_by", prehension.WithdrawalComplete),
        ],
    )
    def test_phase_without_its_boundary_event_violates_its_requirement(
        self, phase_type, field_name, event_type
    ):
        phase = phase_type()
        assert (
            id(phase),
            requirement(phase_type, field_name, Quantifier.SOME, named(event_type)),
        ) in violated([phase])

    def test_phase_with_its_boundary_event_meets_its_requirement(self):
        phase = prehension.ReplacePhase(terminated_by=[prehension.Touchdown()])
        assert (
            id(phase),
            requirement(
                prehension.ReplacePhase,
                "terminated_by",
                Quantifier.SOME,
                named(prehension.Touchdown),
            ),
        ) not in violated([phase])


# %% pouring


def pouring_structure(**fields: object) -> pouring.PouringPhaseStructure:
    """
    Return a pouring structure of liquid ending at the target quantity.
    """
    defaults: dict[str, object] = {
        "has_material_regime": pouring.MaterialRegime.LIQUID,
        "has_termination_reason": pouring.TransferTerminationReason.TARGET_QUANTITY_ATTAINED,
    }
    defaults.update(fields)
    return pouring.PouringPhaseStructure(**defaults)


class TestPouringPhaseRequirements:
    """
    A pouring structure covers the whole sequence of acquiring, pouring from and putting
    back its source.
    """

    @pytest.mark.parametrize(
        "phase_type",
        [
            prehension.ReachPhase,
            prehension.ContactEstablishmentPhase,
            prehension.ForceLoadingPhase,
            prehension.LiftPhase,
            prehension.TransportPhase,
            pouring.AlignmentPhase,
            pouring.TiltToFlowPhase,
            pouring.FlowRegulationPhase,
            pouring.FlowTerminationPhase,
            pouring.ReturnUprightPhase,
            prehension.ReplacePhase,
            prehension.ContactReleasePhase,
        ],
    )
    def test_pouring_structure_without_the_phase_violates_its_requirement(
        self, phase_type
    ):
        structure = pouring_structure()
        required = PropertyRestriction(
            property_iri=property_iri(pouring.PouringPhaseStructure, "defines_phase"),
            quantifier=Quantifier.SOME,
            filler=NamedClass(iri=phase_type.term.iri),
        )
        assert (id(structure), required) in violated([structure])


class TestPouringCouplingAndGazeRequirements:
    """
    A pouring structure couples grip to load, and fixes its gaze while aligning and
    while regulating the flow.
    """

    @staticmethod
    def gaze_serving(phase_type: type[DescriptionEntity]) -> PropertyRestriction:
        return requirement(
            pouring.PouringPhaseStructure,
            "defines_gaze_fixation",
            Quantifier.SOME,
            IntersectionOf(
                operands=(
                    named(gaze.GazeFixation),
                    requirement(
                        gaze.GazeFixation,
                        "serves_phase",
                        Quantifier.SOME,
                        named(phase_type),
                    ),
                )
            ),
        )

    def test_structure_without_a_coupling_violates_its_requirement(self):
        structure = pouring_structure()
        assert (
            id(structure),
            requirement(
                pouring.PouringPhaseStructure,
                "defines_force_coupling",
                Quantifier.SOME,
                named(prehension.GripLoadCoupling),
            ),
        ) in violated([structure])

    @pytest.mark.parametrize(
        "phase_type", [pouring.AlignmentPhase, pouring.FlowRegulationPhase]
    )
    def test_structure_without_gaze_on_the_phase_violates_its_requirement(
        self, phase_type
    ):
        structure = pouring_structure(
            defines_gaze_fixation=[
                gaze.GazeFixation(serves_phase=prehension.LiftPhase())
            ]
        )
        assert (id(structure), self.gaze_serving(phase_type)) in violated([structure])

    def test_structure_with_coupling_and_gaze_meets_those_requirements(self):
        structure = pouring_structure(
            defines_force_coupling=[prehension.GripLoadCoupling()],
            defines_gaze_fixation=[
                gaze.GazeFixation(serves_phase=pouring.AlignmentPhase()),
                gaze.GazeFixation(serves_phase=pouring.FlowRegulationPhase()),
            ],
        )
        expressions = {expression for _, expression in violated([structure])}
        assert not expressions & {
            self.gaze_serving(pouring.AlignmentPhase),
            self.gaze_serving(pouring.FlowRegulationPhase),
            requirement(
                pouring.PouringPhaseStructure,
                "defines_force_coupling",
                Quantifier.SOME,
                named(prehension.GripLoadCoupling),
            ),
        }


class TestPouringRules:
    """
    What a pouring structure must contain, in what order, and for which material.
    """

    def test_pouring_with_a_generic_event_is_reported(self):
        generic = pouring_structure(defines_event_type=[core.MechanicalEventType()])
        specific = pouring_structure(defines_event_type=[pouring.FlowOnset()])
        assert violators(PouringEventsSpecific(), [generic, specific]) == [generic]

    def test_generic_event_is_named_by_the_violation(self):
        generic_event = core.MechanicalEventType()
        structure = pouring_structure(
            defines_event_type=[generic_event, pouring.FlowOnset()]
        )
        (violation,) = PouringEventsSpecific().violations([structure])
        assert violation.involved == (generic_event,)

    def test_transition_against_the_pouring_order_is_reported(self):
        tilt, regulate = pouring.TiltToFlowPhase(), pouring.FlowRegulationPhase()
        backwards = pouring_structure(
            defines_phase=[tilt, regulate],
            defines_transition=[
                core.PhaseTransition(from_phase=regulate, to_phase=tilt)
            ],
        )
        forwards = pouring_structure(
            defines_phase=[tilt, regulate],
            defines_transition=[
                core.PhaseTransition(from_phase=tilt, to_phase=regulate)
            ],
        )
        assert violators(PouringPhasesOrdered(), [backwards, forwards]) == [backwards]

    def test_liquid_with_granular_flow_dynamics_is_reported(self):
        granular_liquid = pouring_structure(
            defines_parameter=[pouring.GranularFlowDynamicsEstimate()]
        )
        sloshing_liquid = pouring_structure(
            defines_parameter=[pouring.SloshDynamicsEstimate()]
        )
        assert violators(
            MaterialFitsFlowDynamics(), [granular_liquid, sloshing_liquid]
        ) == [granular_liquid]

    def test_termination_reason_without_its_event_is_reported(self):
        unsupported = pouring_structure(defines_event_type=[pouring.FlowOnset()])
        supported = pouring_structure(
            defines_event_type=[pouring.TargetQuantityReached()]
        )
        assert violators(TerminationReasonHasEvent(), [unsupported, supported]) == [
            unsupported
        ]


class TestFeedbackControlRule:
    """
    A feedback-regulated controller compares a continuous prediction in a modality it
    monitors.
    """

    def test_feedback_controller_without_a_continuous_prediction_is_reported(self):
        feedback = core.ActionPhaseController(
            has_regulation_mode=core.RegulationMode.CONTINUOUS_FEEDBACK,
            monitors_modality=[core.SensoryModality.VISUAL],
        )
        feedforward = core.ActionPhaseController(
            has_regulation_mode=core.RegulationMode.FEEDFORWARD
        )
        required = requirement(
            core.FeedbackRegulatedController,
            "has_continuous_prediction",
            Quantifier.SOME,
            named(core.ContinuousSignalPrediction),
        )
        found = violated([feedback, feedforward])
        assert (id(feedback), required) in found
        assert (id(feedforward), required) not in found

    def test_feedback_controller_without_predictions_is_left_to_the_ontology(self):
        unpredicting = core.ActionPhaseController(
            has_regulation_mode=core.RegulationMode.CONTINUOUS_FEEDBACK,
            monitors_modality=[core.SensoryModality.VISUAL],
        )
        assert (
            shape_violators(
                CORE_SHAPES.ContinuousPredictionInMonitoredModality, [unpredicting]
            )
            == []
        )

    def test_feedback_controller_without_a_monitored_prediction_is_reported(self):
        prediction = core.ContinuousSignalPrediction(
            has_modality=core.SensoryModality.VISUAL,
            predicts_trajectory=core.SignalTrajectory(),
            controls_setpoint=pouring.FlowRateSetpoint(),
        )
        unmonitored = core.ActionPhaseController(
            has_regulation_mode=core.RegulationMode.CONTINUOUS_FEEDBACK,
            monitors_modality=[core.SensoryModality.AUDITORY],
            has_continuous_prediction=[prediction],
        )
        monitored = core.ActionPhaseController(
            has_regulation_mode=core.RegulationMode.CONTINUOUS_FEEDBACK,
            monitors_modality=[core.SensoryModality.VISUAL],
            has_continuous_prediction=[prediction],
        )
        feedforward = core.ActionPhaseController(
            has_regulation_mode=core.RegulationMode.FEEDFORWARD
        )
        assert shape_violators(
            CORE_SHAPES.ContinuousPredictionInMonitoredModality,
            [unmonitored, monitored, feedforward],
        ) == [unmonitored]


# %% gaze


class TestGazeRules:
    """
    Gaze rests on a participant of the phase it serves until that phase's goal.
    """

    def test_fixation_released_at_another_event_is_reported(self):
        volume_reached = pouring.TargetQuantityReached()
        regulate = pouring.FlowRegulationPhase(terminated_by=[volume_reached])
        early = gaze.GazeFixation(
            serves_phase=regulate, maintained_until=[pouring.FlowOnset()]
        )
        held = gaze.GazeFixation(
            serves_phase=regulate, maintained_until=[volume_reached]
        )
        assert shape_violators(
            GAZE_SHAPES.FixationHeldUntilPhaseGoal, [early, held]
        ) == [early]

    def test_fixation_on_a_participant_outside_its_phase_is_reported(self):
        bowl = core.ManipulationRole()
        regulate = pouring.FlowRegulationPhase(involves_role=[bowl])
        elsewhere = gaze.GazeFixation(
            serves_phase=regulate, fixates_role=core.ManipulationRole()
        )
        on_target = gaze.GazeFixation(serves_phase=regulate, fixates_role=bowl)
        assert shape_violators(
            GAZE_SHAPES.FixationOnPhaseParticipant, [elsewhere, on_target]
        ) == [elsewhere]


# %% structural rules


class TestShapesHold:
    """
    The structural rules of the shapes files are checked closed-world, and each
    violation names the rule, the fragment at fault and the fragment it concerns.
    """

    @staticmethod
    def two_entry_structure() -> (
        tuple[core.ActionPhaseStructure, list[core.ActionPhase]]
    ):
        phases = [core.ActionPhase(), core.ActionPhase(), core.ActionPhase()]
        structure = core.ActionPhaseStructure(
            defines_phase=phases,
            defines_transition=[
                core.PhaseTransition(from_phase=phases[0], to_phase=phases[2]),
                core.PhaseTransition(from_phase=phases[1], to_phase=phases[2]),
            ],
        )
        return structure, phases

    def test_violation_names_its_shape_as_its_rule(self):
        structure, _ = self.two_entry_structure()
        violations = [
            violation
            for violation in ShapesHold().violations([structure])
            if violation.shape == CORE_SHAPES.OneEntryPhase
        ]
        assert violations
        assert {violation.rule for violation in violations} == {
            StructureShapes.shipped().named(CORE_SHAPES.OneEntryPhase).name
        }

    def test_violation_concerns_each_offending_fragment(self):
        structure, phases = self.two_entry_structure()
        involved = {
            violation.involved
            for violation in ShapesHold().violations([structure])
            if violation.shape == CORE_SHAPES.OneEntryPhase
        }
        assert involved == {(phases[0],), (phases[1],)}

    def test_only_the_given_fragments_are_reported(self):
        structure, phases = self.two_entry_structure()
        assert ShapesHold().violations(phases) == []

    def test_violation_is_an_error(self):
        structure, _ = self.two_entry_structure()
        assert {
            violation.severity for violation in ShapesHold().violations([structure])
        } == {Severity.ERROR}

    def test_shape_applies_to_its_target_and_more_specific_classes_only(self):
        shapes = StructureShapes.shipped()
        estimate_rule = shapes.named(
            OBJECT_DYNAMICS_SHAPES.EstimateHeldByARepresentation
        )
        assert estimate_rule in shapes.applicable_to([pouring.PouringPhaseStructure])
        assert estimate_rule not in shapes.applicable_to([core.ActionPhaseStructure])

    def test_every_shape_explains_itself(self):
        assert all(shape.description for shape in StructureShapes.shipped().shapes)


# %% suite


class TestPredictiveManipulationSuite:
    """
    The suite of the microworld checks every rule there is.
    """

    def test_suite_contains_every_concrete_rule(self):
        concrete = {
            rule
            for rule in recursive_subclasses(AcceptanceRule)
            if not inspect.isabstract(rule)
            and rule.__module__.startswith("llm_reasoner.acceptance")
        }
        assert {
            type(rule) for rule in predictive_manipulation_suite().rules
        } == concrete
