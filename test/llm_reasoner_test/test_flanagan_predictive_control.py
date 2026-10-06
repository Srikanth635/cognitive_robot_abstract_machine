from __future__ import annotations

from llm_reasoner.knowledge.microworlds.flanagan.constraints import SUITE
from llm_reasoner.knowledge.microworlds.flanagan.core import (
    ActionPhase,
    ActionPhaseStructure,
    ComparisonOutcome,
    CorrectionDirection,
    CorrectiveAction,
    CouplingRegime,
    EstimateUpdateDirection,
    ForwardModel,
    InitialStateEstimate,
    MagnitudeRange,
    MeasurementUnit,
    ModelUpdate,
    ObjectRepresentation,
    PhaseTransition,
    PredictionComparison,
    PredictionError,
    PredictionErrorKind,
    RecoveryMode,
    RegulationMode,
    RoleSlot,
    SensorimotorControlPoint,
    SensoryEventPrediction,
    SensoryObservationType,
    Subgoal,
    TactileSignal,
    TemporalWindow,
)
from llm_reasoner.knowledge.microworlds.flanagan.pouring import POUR, SourceContainer
from llm_reasoner.knowledge.microworlds.flanagan.prehension import (
    ExpectedMass,
    Liftoff,
    LiftoffThreshold,
    LoadForceSetpoint,
    ObjectClearOfSupport,
)
from llm_reasoner.knowledge.ontological.decorators import ConceptReference
from llm_reasoner.knowledge.ontological.validation import validate

# %% predictive-control test data


def make_source_role() -> RoleSlot:
    """
    Make a source-container role for predictive-control tests.
    """
    return RoleSlot(
        label="source container",
        binds_as=ConceptReference(
            iri=POUR + "SourceContainer",
            label=SourceContainer.__name__,
        ),
    )


def make_magnitude(
    lower_bound: float,
    upper_bound: float,
    unit: MeasurementUnit,
) -> MagnitudeRange:
    """
    Make a dimensionally consistent magnitude.
    """
    return MagnitudeRange(
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        unit=unit,
    )


def make_liftoff(role: RoleSlot) -> Liftoff:
    """
    Make a liftoff event with its force threshold.
    """
    threshold = LiftoffThreshold(
        label="liftoff threshold",
        parameter_range=make_magnitude(4.0, 4.0, MeasurementUnit.NEWTON),
        parameter_on_role=[role],
    )
    return Liftoff(
        label="liftoff",
        crosses_threshold=threshold,
        event_on_role=[role],
    )


def make_prediction(
    role: RoleSlot,
    liftoff: Liftoff,
) -> SensoryEventPrediction:
    """
    Make a fully specified tactile liftoff prediction.
    """
    return SensoryEventPrediction(
        label="predicted tactile liftoff",
        modality=TactileSignal(label="tactile signal"),
        predicts_event=liftoff,
        caused_by_event=liftoff,
        expected_timing=TemporalWindow(
            label="liftoff timing",
            earliest_offset=80.0,
            latest_offset=120.0,
            unit=MeasurementUnit.MILLISECOND,
        ),
        expected_on_role=[role],
        expected_magnitude=make_magnitude(3.8, 4.2, MeasurementUnit.NEWTON),
        tolerance=make_magnitude(0.0, 0.2, MeasurementUnit.NEWTON),
        generated_by=ForwardModel(label="object-load forward model"),
    )


# %% prediction vocabulary


class TestPredictionErrorVocabulary:
    """
    Prediction mismatches use the complete paper-grounded error partition.
    """

    def test_all_required_error_kinds_are_distinct_values(self) -> None:
        assert set(PredictionErrorKind) == {
            PredictionErrorKind.ABSENT,
            PredictionErrorKind.UNEXPECTED,
            PredictionErrorKind.EARLY,
            PredictionErrorKind.LATE,
            PredictionErrorKind.MAGNITUDE_MISMATCH,
            PredictionErrorKind.DIRECTION_MISMATCH,
            PredictionErrorKind.SPATIAL_MISMATCH,
        }


class TestPredictionSpecification:
    """
    A sensory prediction states what, when, where, and from which model.
    """

    def test_prediction_carries_every_operational_comparison_dimension(self) -> None:
        role = make_source_role()
        liftoff = make_liftoff(role)

        prediction = make_prediction(role, liftoff)

        assert prediction.modality.label == "tactile signal"
        assert prediction.predicts_event is liftoff
        assert prediction.caused_by_event is liftoff
        assert prediction.expected_timing.unit is MeasurementUnit.MILLISECOND
        assert prediction.expected_on_role == [role]
        assert prediction.expected_magnitude is not None
        assert prediction.tolerance is not None
        assert isinstance(prediction.generated_by, ForwardModel)

    def test_prediction_cannot_name_both_an_event_and_a_feature(self) -> None:
        role = make_source_role()
        liftoff = make_liftoff(role)
        prediction = make_prediction(role, liftoff)
        prediction.predicts_feature = ConceptReference(
            iri="urn:feature:load-force-release",
            label="load force release",
        )

        violations = SUITE.evaluate((prediction,))

        assert any(violation.check == "C37" for violation in violations)


class TestTemporalWindowSemantics:
    """
    Prediction timing is ordered and carries a time unit.
    """

    def test_inverted_prediction_window_is_rejected(self) -> None:
        timing = TemporalWindow(
            label="inverted timing",
            earliest_offset=120.0,
            latest_offset=80.0,
            unit=MeasurementUnit.MILLISECOND,
        )

        violations = SUITE.evaluate((timing,))

        assert any(violation.check == "C38" for violation in violations)


# %% comparison outcomes


class TestMatchedPredictionComparison:
    """
    A match confirms the subgoal and supplies the transition state.
    """

    def test_match_permits_transition_through_an_initial_state_estimate(self) -> None:
        role = make_source_role()
        liftoff = make_liftoff(role)
        prediction = make_prediction(role, liftoff)
        observation = SensoryObservationType(
            label="observed tactile liftoff",
            modality=prediction.modality,
            observes_event=liftoff,
            observed_at=prediction.expected_timing,
            observed_on_role=[role],
            observed_magnitude=prediction.expected_magnitude,
        )
        lifted = ObjectClearOfSupport(
            label="object clear of support",
            condition_on_role=[role],
        )
        load_goal = Subgoal(
            label="liftoff attained",
            condition_on_role=[role],
            attainment_signalled_by=liftoff,
        )
        lift_goal = Subgoal(
            label="lift continues",
            condition_on_role=[role],
            attainment_signalled_by=liftoff,
        )
        load_phase = ActionPhase(
            label="load",
            aims_for=load_goal,
            coupling_regime=CouplingRegime.TRANSITIONAL,
            regulation_mode=RegulationMode.FEEDFORWARD,
            terminated_by=[liftoff],
            phase_involves=[role],
        )
        lift_phase = ActionPhase(
            label="lift",
            aims_for=lift_goal,
            coupling_regime=CouplingRegime.COUPLED,
            regulation_mode=RegulationMode.FEEDFORWARD,
            initiated_by=[liftoff],
            terminated_by=[liftoff],
            phase_involves=[role],
        )
        state = InitialStateEstimate(
            label="initial lift state",
            for_phase=lift_phase,
            based_on_observation=[observation],
            estimates_condition=[lifted],
        )
        comparison = PredictionComparison(
            label="liftoff match",
            compares_prediction=prediction,
            compares_observation=observation,
            outcome=ComparisonOutcome.MATCH,
            confirms_subgoal=load_goal,
            supplies_state_estimate=state,
        )
        transition = PhaseTransition(
            label="load to lift",
            from_phase=load_phase,
            to_phase=lift_phase,
            gated_by=[liftoff],
            permitted_by=[comparison],
        )
        comparison.permits_transition = [transition]

        assert comparison.confirms_subgoal is load_goal
        assert comparison.supplies_state_estimate is state
        assert transition.permitted_by == [comparison]
        assert comparison.permits_transition == [transition]


class TestMismatchedPredictionComparison:
    """
    A mismatch classifies recovery and updates the implicated model.
    """

    def test_absent_liftoff_expresses_the_heavy_object_response_without_labels(
        self,
    ) -> None:
        role = make_source_role()
        liftoff = make_liftoff(role)
        prediction = make_prediction(role, liftoff)
        load_force = LoadForceSetpoint(
            label="load force",
            parameter_range=make_magnitude(2.0, 10.0, MeasurementUnit.NEWTON),
            parameter_on_role=[role],
        )
        expected_mass = ExpectedMass(
            label="expected mass",
            parameter_range=make_magnitude(0.3, 0.6, MeasurementUnit.KILOGRAM),
            parameter_on_role=[role],
            scales=[load_force],
        )
        representation = ObjectRepresentation(
            label="object representation",
            holds_estimate=[expected_mass],
        )
        error = PredictionError(kind=PredictionErrorKind.ABSENT)
        lift_goal = ObjectClearOfSupport(
            label="object clear of support",
            condition_on_role=[role],
        )
        correction = CorrectiveAction(
            label="increase load force and retry lift",
            triggered_by_error=[error],
            targets_setpoint=[load_force],
            direction=CorrectionDirection.INCREASE,
            intended_effect=[lift_goal],
            recovery_mode=RecoveryMode.REPEAT_PHASE,
        )
        update = ModelUpdate(
            label="increase expected weight",
            triggered_by_error=[error],
            updates_estimate=[expected_mass],
            updates_representation=representation,
            direction=EstimateUpdateDirection.INCREASE,
        )
        comparison = PredictionComparison(
            label="liftoff absent",
            compares_prediction=prediction,
            outcome=ComparisonOutcome.MISMATCH,
            has_error=[error],
            invokes_correction=[correction],
            produces_model_update=[update],
        )

        assert comparison.compares_observation is None
        assert comparison.has_error[0].kind is PredictionErrorKind.ABSENT
        assert correction.targets_setpoint == [load_force]
        assert correction.direction is CorrectionDirection.INCREASE
        assert correction.recovery_mode is RecoveryMode.REPEAT_PHASE
        assert update.updates_estimate == [expected_mass]
        assert update.updates_representation is representation

    def test_absent_error_with_an_observation_is_rejected(self) -> None:
        role = make_source_role()
        liftoff = make_liftoff(role)
        prediction = make_prediction(role, liftoff)
        error = PredictionError(kind=PredictionErrorKind.ABSENT)
        comparison = PredictionComparison(
            label="contradictory absent observation",
            compares_prediction=prediction,
            compares_observation=SensoryObservationType(
                modality=prediction.modality,
                observes_event=liftoff,
                observed_at=prediction.expected_timing,
                observed_on_role=[role],
            ),
            outcome=ComparisonOutcome.MISMATCH,
            has_error=[error],
        )

        violations = SUITE.evaluate((comparison,))

        assert any(violation.check == "C42" for violation in violations)


# %% structure closure


class TestPredictiveStructureClosure:
    """
    A hypothesis explicitly declares every predictive fragment it references.
    """

    def test_control_point_prediction_missing_from_root_is_rejected(self) -> None:
        role = make_source_role()
        liftoff = make_liftoff(role)
        prediction = make_prediction(role, liftoff)
        structure = ActionPhaseStructure(
            label="incomplete predictive structure",
            has_control_point=[
                SensorimotorControlPoint(
                    label="liftoff control point",
                    at_event_type=liftoff,
                    has_prediction=[prediction],
                )
            ],
        )

        violations = SUITE.evaluate((structure,))

        assert any(violation.check == "C13e" for violation in violations)


# %% structural ontology validation


class TestPredictiveOntologyValidation:
    """
    The expanded ontology preserves every structural invariant.
    """

    def test_flanagan_modules_have_no_structural_errors(self) -> None:
        violations = validate(
            module="llm_reasoner.knowledge.microworlds.flanagan",
        )

        assert [
            violation for violation in violations if violation.severity == "error"
        ] == []
