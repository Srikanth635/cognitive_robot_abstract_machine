"""
Tests of the acceptance rules that decide whether a generated hypothesis is admissible,
starting with the rule derived from the ontology's own restrictions.
"""

from __future__ import annotations

import inspect

import pytest
from krrood.utils import recursive_subclasses

from llm_reasoner.acceptance.core import (
    AbsentEventHasNoObservation,
    ComponentsDeclared,
    ContinuousPredictionComparesItsModality,
    ControlledSetpointIsActive,
    ControlPointComparesItsPredictions,
    ControlPointHandlesBothOutcomes,
    ControlPointPredictsItsEvent,
    CorrectionAnswersClassifiedError,
    ErrorHasObservation,
    EvidenceCompatible,
    ExemptionNamesNormalBoundary,
    FeedbackControlHasContinuousPrediction,
    MatchHasNoResponse,
    MatchHasObservation,
    MechanicalBoundaryMonitored,
    MismatchInvokesCorrection,
    ModelUpdateAnswersClassifiedError,
    PermissionFitsTransition,
    ToleranceMatchesTrajectory,
    UpdatedEstimateIsHeld,
    InvariantHasViolationEvent,
    NormalTransitionIsNotGatedByFailure,
    NoOrphanPhase,
    OneEntryPhase,
    PhaseEndingsDistinct,
    PhaseHasExtent,
    PreconditionsGrounded,
    RangeBoundsOrdered,
    SafetyMarginOnOneParticipant,
    SubgoalSignalledByPhaseEnd,
    SuccessIsNotSignalledByFailure,
    TaskIsNotItsOwnPhase,
    TransitionGateEndsSource,
    TransitionGateStartsTarget,
    TransitionIsNotALoop,
    WindowBoundsOrdered,
)
from llm_reasoner.acceptance.derived import (
    DerivedRule,
    RangesHold,
    RestrictionsHold,
    described,
)
from llm_reasoner.acceptance.gaze import (
    FixationHeldUntilPhaseGoal,
    FixationOnPhaseParticipant,
)
from llm_reasoner.acceptance.pouring import (
    MaterialFitsFlowDynamics,
    PouringEventsSpecific,
    PouringPhasesCovered,
    PouringPhasesOrdered,
    TerminationReasonHasEvent,
)
from llm_reasoner.acceptance.prehension import (
    AdjustmentPrecedesPredictedEvent,
    PrehensionPhasesSpecific,
    ReleaseEndsAtGraspRelease,
    ReplaceEndsAtTouchdown,
    SafetyMarginOnCouplingResponse,
    UnloadEndsWhenForcesUnloaded,
    WithdrawalFollowsRelease,
)
from llm_reasoner.acceptance.object_dynamics import (
    CompositionDynamicsUsesItsCopy,
    CompositionPredictsWithItsForwardModel,
    DissociationCoherent,
    EstimateHeldByARepresentation,
    EstimateServesOnePurpose,
    RepresentationCoversEstimatedRoles,
    RetainedUpdateInItsMemory,
    TransferComparesMatchedPerturbations,
)
from llm_reasoner.acceptance.rules import AcceptanceRule
from llm_reasoner.acceptance.suite import predictive_manipulation_suite
from llm_reasoner.ontologies.predictive_manipulation import (
    core,
    gaze,
    object_dynamics,
    pouring,
    prehension,
)
from llm_reasoner.ontologies.terms import (
    ClassExpression,
    ConceptReference,
    DescriptionEntity,
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
        assert TransitionGateEndsSource().subject_type is core.PhaseTransition

    def test_inverted_magnitude_range_is_reported(self):
        inverted = core.MagnitudeRange(lower_bound=5.0, upper_bound=1.0)
        ordered = core.MagnitudeRange(lower_bound=1.0, upper_bound=5.0)
        assert violators(RangeBoundsOrdered(), [inverted, ordered]) == [inverted]

    def test_inverted_temporal_window_is_reported(self):
        inverted = core.TemporalWindow(earliest_offset=300.0, latest_offset=100.0)
        ordered = core.TemporalWindow(earliest_offset=100.0, latest_offset=300.0)
        assert violators(WindowBoundsOrdered(), [inverted, ordered]) == [inverted]

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
        assert violators(
            TransitionGateEndsSource(), [gated_correctly, gated_wrongly]
        ) == [gated_wrongly]

    def test_transition_gated_by_an_event_not_starting_its_target_is_reported(self):
        liftoff, clearance = prehension.Liftoff(), prehension.LiftClearanceReached()
        lift = prehension.LiftPhase(initiated_by=[liftoff])
        gated_correctly = core.PhaseTransition(to_phase=lift, gated_by=[liftoff])
        gated_wrongly = core.PhaseTransition(to_phase=lift, gated_by=[clearance])
        assert violators(
            TransitionGateStartsTarget(), [gated_correctly, gated_wrongly]
        ) == [gated_wrongly]

    def test_phase_started_and_ended_by_the_same_event_is_reported(self):
        event = core.PhaseBoundaryEventType()
        empty = core.ActionPhase(initiated_by=[event], terminated_by=[event])
        extended = core.ActionPhase(
            initiated_by=[event], terminated_by=[core.PhaseBoundaryEventType()]
        )
        assert violators(PhaseHasExtent(), [empty, extended]) == [empty]

    def test_transition_from_a_phase_to_itself_is_reported(self):
        phase, other = core.ActionPhase(), core.ActionPhase()
        loop = core.PhaseTransition(from_phase=phase, to_phase=phase)
        step = core.PhaseTransition(from_phase=phase, to_phase=other)
        assert violators(TransitionIsNotALoop(), [loop, step]) == [loop]

    def test_transition_gated_by_a_failure_is_reported(self):
        failing = core.PhaseTransition(gated_by=[prehension.SlipEvent()])
        normal = core.PhaseTransition(gated_by=[prehension.Liftoff()])
        assert violators(NormalTransitionIsNotGatedByFailure(), [failing, normal]) == [
            failing
        ]

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
        assert violators(
            SubgoalSignalledByPhaseEnd(), [signalled_elsewhere, signalled_at_end]
        ) == [signalled_elsewhere]

    def test_subgoal_signalled_by_a_failure_is_reported(self):
        failing = core.Subgoal(attainment_signalled_by=prehension.ObjectCrush())
        normal = core.Subgoal(attainment_signalled_by=prehension.Liftoff())
        assert violators(SuccessIsNotSignalledByFailure(), [failing, normal]) == [
            failing
        ]

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
        assert violators(
            InvariantHasViolationEvent(), [unguarded, guarded, precondition]
        ) == [unguarded]

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
        assert violators(SafetyMarginOnOneParticipant(), [mismatched, matched]) == [
            mismatched
        ]

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
        assert violators(TaskIsNotItsOwnPhase(), [circular, proper]) == [circular]

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
        assert violators(OneEntryPhase(), [branching_in, chained]) == [branching_in]

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
        assert violators(NoOrphanPhase(), [with_orphan]) == [with_orphan]

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
        assert violators(PhaseEndingsDistinct(), [indistinct, distinct]) == [indistinct]

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
        assert violators(PreconditionsGrounded(), [ungrounded, grounded]) == [
            ungrounded
        ]


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
        assert violators(MatchHasObservation(), [unobserved, observed]) == [unobserved]

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
        assert violators(PermissionFitsTransition(), [wrong, right]) == [wrong]

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
        assert violators(PermissionFitsTransition(), [wrong, right]) == [wrong]

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
        assert violators(UpdatedEstimateIsHeld(), [foreign, held]) == [foreign]

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
        assert violators(ControlPointPredictsItsEvent(), [elsewhere, here]) == [
            elsewhere
        ]

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
        assert violators(ControlPointComparesItsPredictions(), [foreign, own]) == [
            foreign
        ]

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
        assert violators(CorrectionAnswersClassifiedError(), [foreign, own]) == [
            foreign
        ]

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
        assert violators(ModelUpdateAnswersClassifiedError(), [foreign, own]) == [
            foreign
        ]

    def test_control_point_without_mismatch_handling_is_reported(self):
        match_only = core.SensorimotorControlPoint(
            has_comparison=[matching_comparison()]
        )
        both = core.SensorimotorControlPoint(
            has_comparison=[matching_comparison(), mismatching_comparison()]
        )
        assert violators(ControlPointHandlesBothOutcomes(), [match_only, both]) == [
            match_only
        ]

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
        assert violators(MismatchInvokesCorrection(), [uncorrected, corrected]) == [
            uncorrected
        ]

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
        assert violators(EvidenceCompatible(), [incompatible, compatible]) == [
            incompatible
        ]


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
        assert violators(
            ContinuousPredictionComparesItsModality(), [mismatched, matched]
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
        assert violators(ToleranceMatchesTrajectory(), [mismatched, matched]) == [
            mismatched
        ]

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
        assert violators(ControlledSetpointIsActive(), [inactive, active]) == [inactive]


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
        assert violators(
            MechanicalBoundaryMonitored(), [unmonitored, monitored, exempted]
        ) == [unmonitored]

    def test_unmonitored_boundary_event_is_named_by_the_violation(self):
        liftoff, touchdown = prehension.Liftoff(), prehension.Touchdown()
        structure = core.ActionPhaseStructure(
            defines_phase=[core.ActionPhase(terminated_by=[liftoff, touchdown])],
            defines_control_point=[
                core.SensorimotorControlPoint(at_event_type=touchdown)
            ],
        )
        (violation,) = MechanicalBoundaryMonitored().violations([structure])
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
        assert violators(ExemptionNamesNormalBoundary(), [misplaced, placed]) == [
            misplaced
        ]

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
        assert violators(CompositionPredictsWithItsForwardModel(), [foreign, own]) == [
            foreign
        ]

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
        assert violators(CompositionDynamicsUsesItsCopy(), [ignoring, using]) == [
            ignoring
        ]

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
        assert violators(EstimateHeldByARepresentation(), [unowned, owned]) == [unowned]

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
        assert violators(EstimateServesOnePurpose(), [shared, separate]) == [shared]

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
        assert violators(DissociationCoherent(), [same, distinct]) == [same]

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
        assert violators(RetainedUpdateInItsMemory(), [outside, inside]) == [outside]

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
        assert violators(
            TransferComparesMatchedPerturbations(), [unmatched, matched]
        ) == [unmatched]

    def test_representation_missing_a_role_its_estimates_concern_is_reported(self):
        carton = core.ManipulationRole()
        estimate = prehension.ExpectedMass(parameter_on_role=[carton])
        uncovered = object_dynamics.ObjectDynamicsModel(holds_estimate=[estimate])
        covered = object_dynamics.ObjectDynamicsModel(
            holds_estimate=[estimate], represents_role=[carton]
        )
        assert violators(
            RepresentationCoversEstimatedRoles(), [uncovered, covered]
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
        assert violators(SafetyMarginOnCouplingResponse(), [unrelated, related]) == [
            unrelated
        ]

    def test_replace_phase_not_ending_at_touchdown_is_reported(self):
        not_touching = prehension.ReplacePhase(
            terminated_by=[prehension.TransportTargetReached()]
        )
        touching = prehension.ReplacePhase(terminated_by=[prehension.Touchdown()])
        assert violators(ReplaceEndsAtTouchdown(), [not_touching, touching]) == [
            not_touching
        ]

    def test_unload_phase_not_ending_when_forces_are_unloaded_is_reported(self):
        early = prehension.UnloadPhase(terminated_by=[prehension.GraspRelease()])
        complete = prehension.UnloadPhase(terminated_by=[prehension.ForcesUnloaded()])
        assert violators(UnloadEndsWhenForcesUnloaded(), [early, complete]) == [early]

    def test_release_phase_not_ending_at_release_is_reported(self):
        unreleased = prehension.ContactReleasePhase(
            terminated_by=[prehension.ForcesUnloaded()]
        )
        released = prehension.ContactReleasePhase(
            terminated_by=[prehension.GraspRelease()]
        )
        assert violators(ReleaseEndsAtGraspRelease(), [unreleased, released]) == [
            unreleased
        ]

    def test_withdrawal_not_following_release_is_reported(self):
        premature = prehension.WithdrawPhase(
            initiated_by=[prehension.ForcesUnloaded()],
            terminated_by=[prehension.WithdrawalComplete()],
        )
        following = prehension.WithdrawPhase(
            initiated_by=[prehension.GraspRelease()],
            terminated_by=[prehension.WithdrawalComplete()],
        )
        assert violators(WithdrawalFollowsRelease(), [premature, following]) == [
            premature
        ]

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
        assert violators(AdjustmentPrecedesPredictedEvent(), [late, anticipatory]) == [
            late
        ]

    def test_prehension_structure_with_a_generic_phase_is_reported(self):
        generic = prehension.PrehensionPhaseStructure(
            defines_phase=[core.ActionPhase()]
        )
        specific = prehension.PrehensionPhaseStructure(
            defines_phase=[prehension.LiftPhase()]
        )
        assert violators(PrehensionPhasesSpecific(), [generic, specific]) == [generic]


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

    def test_pouring_without_flow_regulation_is_reported(self):
        partial = pouring_structure(
            defines_phase=[pouring.TiltToFlowPhase(), pouring.FlowTerminationPhase()]
        )
        complete = pouring_structure(
            defines_phase=[
                pouring.TiltToFlowPhase(),
                pouring.FlowRegulationPhase(),
                pouring.FlowTerminationPhase(),
            ]
        )
        assert violators(PouringPhasesCovered(), [partial, complete]) == [partial]

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
    A feedback-regulated controller compares a continuous prediction it monitors.
    """

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
        assert violators(
            FeedbackControlHasContinuousPrediction(),
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
        assert violators(FixationHeldUntilPhaseGoal(), [early, held]) == [early]

    def test_fixation_on_a_participant_outside_its_phase_is_reported(self):
        bowl = core.ManipulationRole()
        regulate = pouring.FlowRegulationPhase(involves_role=[bowl])
        elsewhere = gaze.GazeFixation(
            serves_phase=regulate, fixates_role=core.ManipulationRole()
        )
        on_target = gaze.GazeFixation(serves_phase=regulate, fixates_role=bowl)
        assert violators(FixationOnPhaseParticipant(), [elsewhere, on_target]) == [
            elsewhere
        ]


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
