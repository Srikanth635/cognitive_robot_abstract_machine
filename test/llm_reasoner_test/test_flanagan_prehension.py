from __future__ import annotations

from llm_reasoner.knowledge.microworlds.flanagan.constraints import SUITE
from llm_reasoner.knowledge.microworlds.flanagan.core import (
    CorrectionDirection,
    CouplingRegime,
    MagnitudeRange,
    MeasurementUnit,
    PredictionError,
    PredictionErrorKind,
    RecoveryMode,
    RoleSlot,
    SafetyMargin,
    TemporalWindow,
)
from llm_reasoner.knowledge.microworlds.flanagan.object_dynamics import (
    LoadForceEstimate,
)
from llm_reasoner.knowledge.microworlds.flanagan.prehension import (
    AnticipatoryGripAdjustment,
    ContactBreak,
    ContactReleasePhase,
    CorrectiveGripResponse,
    ExpectedFrictionCoefficient,
    ForceLoadingPhase,
    ForcesUnloaded,
    GripForceSetpoint,
    GripLoadCoupling,
    Liftoff,
    PrehensionPhaseStructure,
    ReplacePhase,
    SlipThreshold,
    Touchdown,
    WithdrawPhase,
)
from llm_reasoner.knowledge.ontological.decorators import ConceptReference

# %% fixtures


def role(label: str) -> RoleSlot:
    """
    Build one manipulation role slot.
    """
    return RoleSlot(
        label=label,
        binds_as=ConceptReference(iri=f"urn:test:{label}"),
    )


def magnitude(
    lower_bound: float,
    upper_bound: float,
    unit: MeasurementUnit,
) -> MagnitudeRange:
    """
    Build a dimensionally coherent magnitude.
    """
    return MagnitudeRange(
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        physical_dimension=unit.physical_dimension,
        unit=unit,
    )


def force_coupling() -> GripLoadCoupling:
    """
    Build a complete functional grip-load relationship.
    """
    effector = role("effector")
    manipulandum = role("manipulandum")
    grip_force = GripForceSetpoint(
        label="grip response",
        parameter_range=magnitude(5.0, 15.0, MeasurementUnit.NEWTON),
        parameter_on_role=[effector, manipulandum],
    )
    slip_threshold = SlipThreshold(
        label="slip threshold",
        parameter_range=magnitude(3.0, 3.0, MeasurementUnit.NEWTON),
        parameter_on_role=[effector, manipulandum],
    )
    safety_margin = SafetyMargin(
        label="grip safety margin",
        parameter_range=magnitude(2.0, 12.0, MeasurementUnit.NEWTON),
        parameter_on_role=[effector, manipulandum],
        margin_on=grip_force,
        margin_over=slip_threshold,
    )
    predicted_load = LoadForceEstimate(
        label="predicted object load",
        parameter_range=magnitude(2.0, 10.0, MeasurementUnit.NEWTON),
        parameter_on_role=[manipulandum],
    )
    friction = ExpectedFrictionCoefficient(
        label="expected friction",
        parameter_range=magnitude(0.3, 0.6, MeasurementUnit.DIMENSIONLESS),
        parameter_on_role=[effector, manipulandum],
    )
    timing = TemporalWindow(
        label="grip-load timing",
        earliest_offset=0.0,
        latest_offset=100.0,
        unit=MeasurementUnit.MILLISECOND,
    )
    return GripLoadCoupling(
        label="functional grip-load coupling",
        predicted_load=predicted_load,
        friction_estimate=friction,
        safety_margin=safety_margin,
        grip_force_response=grip_force,
        temporal_coupling=timing,
    )


# %% ownership and vocabulary


class TestPrehensionOwnership:
    """
    General grasping concepts belong to the prehension module.
    """

    def test_general_types_are_defined_by_prehension(self) -> None:
        types = (
            ForceLoadingPhase,
            ReplacePhase,
            ContactReleasePhase,
            WithdrawPhase,
            Liftoff,
            Touchdown,
            GripLoadCoupling,
        )

        assert {class_type.__module__ for class_type in types} == {
            "llm_reasoner.knowledge.microworlds.flanagan.prehension"
        }

    def test_prehension_structure_extends_object_dynamics(self) -> None:
        assert "defines_internal_model" in PrehensionPhaseStructure.__dataclass_fields__
        assert "has_force_coupling" in PrehensionPhaseStructure.__dataclass_fields__


class TestReleaseVocabulary:
    """
    Force unloading, contact release, and withdrawal remain distinct.
    """

    def test_each_release_stage_has_its_own_phase_and_boundary(self) -> None:
        assert ForcesUnloaded is not ContactBreak
        assert ContactReleasePhase is not WithdrawPhase


# %% functional grip-load relationship


class TestGripLoadCoupling:
    """
    Grip-load coupling exposes every scientifically meaningful input.
    """

    def test_coupling_names_load_friction_margin_response_and_timing(self) -> None:
        coupling = force_coupling()

        assert isinstance(coupling.predicted_load, LoadForceEstimate)
        assert isinstance(coupling.friction_estimate, ExpectedFrictionCoefficient)
        assert coupling.safety_margin.margin_on is coupling.grip_force_response
        assert isinstance(coupling.safety_margin.margin_over, SlipThreshold)
        assert coupling.temporal_coupling.latest_offset == 100.0

    def test_coupling_with_unrelated_margin_is_rejected(self) -> None:
        coupling = force_coupling()
        unrelated_force = GripForceSetpoint(
            label="unrelated force",
            parameter_range=coupling.grip_force_response.parameter_range,
            parameter_on_role=coupling.grip_force_response.parameter_on_role,
        )
        coupling.safety_margin.margin_on = unrelated_force

        violations = SUITE.evaluate((coupling, coupling.safety_margin, unrelated_force))

        assert any(violation.check == "C59" for violation in violations)


# %% grip responses


class TestGripResponses:
    """
    Reactive and anticipatory grip changes are represented separately.
    """

    def test_slip_response_is_a_typed_correction(self) -> None:
        coupling = force_coupling()
        error = PredictionError(
            label="unexpected slip",
            kind=PredictionErrorKind.UNEXPECTED,
        )
        response = CorrectiveGripResponse(
            label="increase grip force",
            triggered_by_error=[error],
            targets_setpoint=[coupling.grip_force_response],
            direction=CorrectionDirection.INCREASE,
            intended_effect=[],
            recovery_mode=RecoveryMode.CONTINUE_PHASE,
        )

        assert response.targets_setpoint == [coupling.grip_force_response]

    def test_anticipatory_adjustment_is_not_a_prediction_error_correction(self) -> None:
        assert not issubclass(AnticipatoryGripAdjustment, CorrectiveGripResponse)


class TestReplacePhaseCoupling:
    """
    Lowering remains mechanically coupled until touchdown.
    """

    def test_transitional_replace_phase_is_rejected(self) -> None:
        phase = ReplacePhase(
            label="prematurely decoupled lowering",
            coupling_regime=CouplingRegime.TRANSITIONAL,
            terminated_by=[Touchdown(label="touchdown")],
        )

        violations = SUITE.evaluate((phase, *phase.terminated_by))

        assert any(violation.check == "C60" for violation in violations)
