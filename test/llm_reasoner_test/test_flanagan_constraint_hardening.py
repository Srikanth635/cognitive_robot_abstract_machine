from __future__ import annotations

from dataclasses import dataclass

from llm_reasoner.knowledge.microworlds.flanagan.constraints import CONSTRAINTS, SUITE
from llm_reasoner.knowledge.microworlds.flanagan.core import (
    ActionPhase,
    ActionPhaseStructure,
    BoundaryControlExemption,
    ComparisonOutcome,
    ContactTransition,
    CorrectiveAction,
    CouplingRegime,
    EstimateUpdateDirection,
    ExternalTerminationEventType,
    ForwardModel,
    MagnitudeRange,
    MeasurementUnit,
    ModelUpdate,
    PhaseTransition,
    PredictionComparison,
    RegulationMode,
    RoleSlot,
    ScientificAssertion,
    ScientificProvenance,
    ScientificStatus,
    SensoryEventPrediction,
    SensoryObservationType,
    Subgoal,
    TactileSignal,
    TemporalWindow,
)
from llm_reasoner.knowledge.microworlds.flanagan.pouring import (
    POUR,
    FlowRegulationPhase,
    MaterialRegime,
    PouringPhaseStructure,
    TransferTerminationReason,
)
from llm_reasoner.knowledge.microworlds.flanagan.prehension import (
    ContactBreak,
    ContactEstablishment,
    GripForceSetpoint,
)
from llm_reasoner.knowledge.ontological.decorators import ConceptReference

# %% constraint identity


class TestConstraintIdentity:
    """
    Aggregate constraint reports identify one unambiguous rule.
    """

    def test_constraint_identifiers_are_unique_across_modules(self) -> None:
        identifiers = [constraint.id for constraint in CONSTRAINTS]

        assert len(identifiers) == len(set(identifiers))


# %% reference fixture


@dataclass(frozen=True)
class CanonicalConstraintFixture:
    """
    A complete reference hypothesis and every fragment evaluated with it.
    """

    root: ActionPhaseStructure
    """
    The reference hypothesis root.
    """

    instances: tuple[object, ...]
    """
    Every fragment reachable from the root.
    """


@dataclass(eq=False)
class EngineeringAssertion(ScientificAssertion):
    """
    A test assertion representing an engineering extension.
    """


def make_role() -> RoleSlot:
    """
    Make a valid manipulation-role slot.
    """
    return RoleSlot(
        label="manipulandum",
        binds_as=ConceptReference(
            iri=POUR + "SourceContainer",
            label="SourceContainer",
        ),
    )


def make_external_boundary(label: str) -> ExternalTerminationEventType:
    """
    Make a typed external phase boundary.
    """
    return ExternalTerminationEventType(
        label=label,
        triggered_by=ConceptReference(
            iri=f"urn:test:{label.replace(' ', '-')}",
            label=label,
        ),
    )


def make_phase(
    label: str,
    role: RoleSlot,
    boundary: ExternalTerminationEventType,
    initiated_by: list[ExternalTerminationEventType] | None = None,
) -> tuple[ActionPhase, Subgoal]:
    """
    Make a feedforward phase with a distinct boundary and subgoal.
    """
    subgoal = Subgoal(
        label=f"{label} complete",
        condition_on_role=[role],
        attainment_signalled_by=boundary,
    )
    phase = ActionPhase(
        label=label,
        aims_for=subgoal,
        coupling_regime=CouplingRegime.UNCOUPLED,
        regulation_mode=RegulationMode.FEEDFORWARD,
        initiated_by=initiated_by or [],
        terminated_by=[boundary],
        phase_involves=[role],
    )
    return phase, subgoal


def make_canonical_constraint_fixture() -> CanonicalConstraintFixture:
    """
    Make a self-contained hypothesis accepted without warnings.
    """
    role = make_role()
    first_boundary = make_external_boundary("first instruction")
    final_boundary = make_external_boundary("final instruction")
    first_phase, first_subgoal = make_phase("first phase", role, first_boundary)
    final_phase, final_subgoal = make_phase(
        "final phase",
        role,
        final_boundary,
        initiated_by=[first_boundary],
    )
    transition = PhaseTransition(
        label="first to final",
        from_phase=first_phase,
        to_phase=final_phase,
        gated_by=[first_boundary],
    )
    first_exemption = BoundaryControlExemption(
        label="external first boundary",
        exempts_event=first_boundary,
        justification=first_boundary.triggered_by,
    )
    final_exemption = BoundaryControlExemption(
        label="external final boundary",
        exempts_event=final_boundary,
        justification=final_boundary.triggered_by,
    )
    root = ActionPhaseStructure(
        label="canonical constraint fixture",
        decomposes_task=ConceptReference(
            iri="urn:test:canonical-manipulation",
            label="canonical manipulation",
        ),
        defines_phase=[first_phase, final_phase],
        defines_role=[role],
        defines_event_type=[first_boundary, final_boundary],
        has_transition=[transition],
        has_control_point_exemption=[first_exemption, final_exemption],
    )
    return CanonicalConstraintFixture(
        root=root,
        instances=(
            root,
            role,
            first_boundary,
            final_boundary,
            first_subgoal,
            final_subgoal,
            first_phase,
            final_phase,
            transition,
            first_exemption,
            final_exemption,
        ),
    )


# %% phase boundaries


class TestBoundaryControlCoverage:
    """
    Every normal boundary is a control point or an explicit exemption.
    """

    def test_uncontrolled_normal_boundary_is_rejected(self) -> None:
        fixture = make_canonical_constraint_fixture()
        fixture.root.has_control_point_exemption.pop()

        violations = SUITE.evaluate(fixture.instances)

        assert any(violation.check == "C66" for violation in violations)

    def test_exemption_for_an_unused_boundary_is_rejected(self) -> None:
        fixture = make_canonical_constraint_fixture()
        unused_boundary = make_external_boundary("unused instruction")
        fixture.root.has_control_point_exemption.append(
            BoundaryControlExemption(
                label="unused exemption",
                exempts_event=unused_boundary,
                justification=unused_boundary.triggered_by,
            )
        )

        violations = SUITE.evaluate((*fixture.instances, unused_boundary))

        assert any(violation.check == "C69" for violation in violations)


# %% predictive evidence


class TestComparedEvidenceCompatibility:
    """
    Prediction comparisons require matching predicted and observed evidence.
    """

    def test_comparison_of_different_event_types_is_rejected(self) -> None:
        role = make_role()
        predicted_event = ContactEstablishment(
            label="predicted contact",
            event_on_role=[role],
            contact_transition=ContactTransition.ESTABLISHES,
        )
        observed_event = ContactBreak(
            label="observed contact break",
            event_on_role=[role],
            contact_transition=ContactTransition.BREAKS,
        )
        signal = TactileSignal(label="tactile")
        timing = TemporalWindow(
            label="contact timing",
            earliest_offset=0.0,
            latest_offset=100.0,
            unit=MeasurementUnit.MILLISECOND,
        )
        prediction = SensoryEventPrediction(
            label="contact prediction",
            modality=signal,
            predicts_event=predicted_event,
            caused_by_event=predicted_event,
            expected_timing=timing,
            expected_on_role=[role],
            generated_by=ForwardModel(label="contact model"),
        )
        observation = SensoryObservationType(
            label="contact observation",
            modality=signal,
            observes_event=observed_event,
            observed_at=timing,
            observed_on_role=[role],
        )
        comparison = PredictionComparison(
            label="contradictory comparison",
            compares_prediction=prediction,
            compares_observation=observation,
            outcome=ComparisonOutcome.MISMATCH,
        )

        violations = SUITE.evaluate((comparison,))

        assert any(violation.check == "C67" for violation in violations)


# %% recovery contracts


class TestRecoveryContractCompleteness:
    """
    Recovery fragments expose their trigger, target, and intended outcome.
    """

    def test_correction_requires_triggering_error_and_intended_effect(self) -> None:
        correction = CorrectiveAction(
            label="incomplete correction",
            targets_setpoint=[GripForceSetpoint(label="grip force")],
        )

        violations = SUITE.evaluate((correction,))

        messages = [violation.message for violation in violations]
        assert any("'triggered_by_error'" in message for message in messages)
        assert any("'intended_effect'" in message for message in messages)

    def test_model_update_requires_owned_content(self) -> None:
        update = ModelUpdate(
            label="incomplete update",
            direction=EstimateUpdateDirection.INCREASE,
        )

        violations = SUITE.evaluate((update,))

        messages = [violation.message for violation in violations]
        assert any("'updates_estimate'" in message for message in messages)
        assert any("'updates_representation'" in message for message in messages)


# %% continuous regulation


class TestContinuousRegulationContract:
    """
    Feedback phases require a complete continuous-prediction contract.
    """

    def test_feedback_phase_without_continuous_prediction_is_rejected(self) -> None:
        role = make_role()
        boundary = make_external_boundary("stop flow")
        subgoal = Subgoal(
            label="flow stopped",
            condition_on_role=[role],
            attainment_signalled_by=boundary,
        )
        phase = FlowRegulationPhase(
            label="regulate flow",
            aims_for=subgoal,
            coupling_regime=CouplingRegime.COUPLED,
            regulation_mode=RegulationMode.CONTINUOUS_FEEDBACK,
            terminated_by=[boundary],
            phase_involves=[role],
            monitors_signal=[TactileSignal(label="flow signal")],
        )
        structure = PouringPhaseStructure(
            label="incomplete continuous control",
            material_regime=MaterialRegime.LIQUID,
            termination_reason=TransferTerminationReason.EXPLICIT_INSTRUCTION,
            defines_phase=[phase],
        )

        violations = SUITE.evaluate((structure, phase))

        assert any(violation.check == "POUR-C66" for violation in violations)

    def test_continuous_control_rejection_names_every_required_link(self) -> None:
        role = make_role()
        boundary = make_external_boundary("stop flow")
        subgoal = Subgoal(
            label="flow stopped",
            condition_on_role=[role],
            attainment_signalled_by=boundary,
        )
        phase = FlowRegulationPhase(
            label="regulate flow",
            aims_for=subgoal,
            coupling_regime=CouplingRegime.COUPLED,
            regulation_mode=RegulationMode.CONTINUOUS_FEEDBACK,
            terminated_by=[boundary],
            phase_involves=[role],
            monitors_signal=[TactileSignal(label="flow signal")],
        )
        structure = PouringPhaseStructure(
            label="incomplete continuous control",
            material_regime=MaterialRegime.LIQUID,
            termination_reason=TransferTerminationReason.EXPLICIT_INSTRUCTION,
            defines_phase=[phase],
        )

        violation = next(
            violation
            for violation in SUITE.evaluate((structure, phase))
            if violation.check == "POUR-C66"
        )

        assert all(
            field_name in violation.message
            for field_name in (
                "predicted_for_phase",
                "monitors_signal",
                "predicted_trajectory",
                "controlled_variable",
                "defines_parameter",
            )
        )


# %% parameter context


class TestParameterContext:
    """
    Quantitative parameters identify both their participant and active phase.
    """

    def test_parameter_without_active_phase_is_rejected(self) -> None:
        role = make_role()
        setpoint = GripForceSetpoint(
            label="unscoped grip force",
            parameter_range=MagnitudeRange(
                lower_bound=2.0,
                upper_bound=4.0,
                unit=MeasurementUnit.NEWTON,
            ),
            parameter_on_role=[role],
        )

        violations = SUITE.evaluate((setpoint,))

        assert any(
            violation.check == "REQ" and "'parameter_in_phase'" in violation.message
            for violation in violations
        )


# %% contact semantics


class TestContactTransitionSemantics:
    """
    Typed contact events agree with their declared topology change.
    """

    def test_schema_valid_contradictory_contact_event_is_rejected(self) -> None:
        role = make_role()
        event = ContactEstablishment(
            label="contradictory contact",
            event_on_role=[role],
            contact_transition=ContactTransition.BREAKS,
        )

        violations = SUITE.evaluate((role, event))

        assert [(violation.check, violation.severity) for violation in violations] == [
            ("PRE-C66", "error")
        ]


# %% provenance


class TestEngineeringExtensionProvenance:
    """
    Engineering additions identify the source that introduced them.
    """

    def test_engineering_extension_without_provenance_is_rejected(self) -> None:
        assertion = EngineeringAssertion(
            label="engineering addition",
            scientific_status=ScientificStatus.ENGINEERING_EXTENSION,
        )

        violations = SUITE.evaluate((assertion,))

        assert any(violation.check == "C68" for violation in violations)

    def test_engineering_extension_with_provenance_is_accepted(self) -> None:
        assertion = EngineeringAssertion(
            label="engineering addition",
            scientific_status=ScientificStatus.ENGINEERING_EXTENSION,
            scientific_provenance=ScientificProvenance(
                label="design source",
                source=ConceptReference(
                    iri="urn:test:design-source",
                    label="design source",
                ),
            ),
        )

        violations = SUITE.evaluate((assertion, assertion.scientific_provenance))

        assert not any(violation.check == "C68" for violation in violations)


# %% existing safeguards


class TestExistingPhaseSafeguards:
    """
    Phase 8 retains the existing failure-gate invariant.
    """

    def test_failure_event_cannot_gate_normal_progress(self) -> None:
        fixture = make_canonical_constraint_fixture()
        fixture.root.has_transition[0].gated_by[0].is_failure_event = True

        violations = SUITE.evaluate(fixture.instances)

        assert any(violation.check == "C12" for violation in violations)


# %% accepted reference


class TestCanonicalConstraintFixture:
    """
    The complete reference hypothesis has no unexplained diagnostics.
    """

    def test_reference_hypothesis_has_zero_errors_and_warnings(self) -> None:
        fixture = make_canonical_constraint_fixture()

        assert SUITE.evaluate(fixture.instances) == []
