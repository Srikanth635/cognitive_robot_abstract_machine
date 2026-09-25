from __future__ import annotations

from llm_reasoner.knowledge.generation.schema import GenerationSchema, derive_schema
from llm_reasoner.knowledge.microworlds.flanagan.constraints import SUITE
from llm_reasoner.knowledge.microworlds.flanagan.core import (
    ActionPhase,
    AuditorySignal,
    ContinuousSignalPrediction,
    ForwardModel,
    MagnitudeRange,
    MeasurementUnit,
    PhysicalDimension,
    SensoryObservationType,
    SignalTrajectory,
    TrajectoryTrend,
)
from llm_reasoner.knowledge.microworlds.flanagan.pouring import (
    AlignmentErrorEstimate,
    DynamicViscosityEstimate,
    FlowRateSetpoint,
    FlowRegulationPhase,
    FlowTerminationPhase,
    GranularFlowDynamicsEstimate,
    MaterialRegime,
    PouringPhaseStructure,
    RemainingSourceQuantityEstimate,
    SloshDynamicsEstimate,
    SourceEmpty,
    TargetFillStateEstimate,
    TargetQuantityReached,
    TiltToFlowPhase,
    TransferredQuantitySetpoint,
    TransferTerminationReason,
)
from llm_reasoner.knowledge.microworlds.registry import (
    FLANAGAN_CORE,
    FLANAGAN_OBJECT_DYNAMICS,
    FLANAGAN_POURING,
    FLANAGAN_PREHENSION,
)
from llm_reasoner.knowledge.ontological.decorators import ConceptReference

# %% generation schema


def pouring_schema() -> GenerationSchema:
    """
    Return the generation schema for the complete pouring composition.
    """
    return derive_schema(
        modules=(
            FLANAGAN_CORE.name,
            FLANAGAN_OBJECT_DYNAMICS.name,
            FLANAGAN_PREHENSION.name,
            FLANAGAN_POURING.name,
        ),
        root=PouringPhaseStructure.__name__,
    )


# %% shared continuous prediction


class TestContinuousPredictionContract:
    """
    Continuous predictions expose their comparison contract as fields.
    """

    def test_prediction_vocabulary_is_action_neutral(self) -> None:
        assert AuditorySignal.__module__.endswith("flanagan.core")
        assert ContinuousSignalPrediction.__module__.endswith("flanagan.core")
        assert SignalTrajectory.__module__.endswith("flanagan.core")

    def test_generation_requires_every_continuous_comparison_component(self) -> None:
        prediction = pouring_schema().get(ContinuousSignalPrediction.__name__)

        assert prediction is not None
        assert {
            "signal",
            "predicted_trajectory",
            "tolerance",
            "generated_by",
            "comparison_target",
            "controlled_variable",
            "predicted_for_phase",
        } == {field.name for field in prediction.required_fields}

    def test_mismatched_signal_and_observation_modalities_are_rejected(self) -> None:
        phase = FlowRegulationPhase(label="Regulate flow")
        auditory = AuditorySignal(label="Auditory flow signal")
        observation = SensoryObservationType(
            label="Visual flow observation",
            modality=None,
        )
        prediction = ContinuousSignalPrediction(
            label="Auditory flow trajectory",
            signal=auditory,
            predicted_trajectory=SignalTrajectory(
                label="Rising pitch",
                signal_feature=ConceptReference(iri="urn:test:pitch"),
                trend=TrajectoryTrend.INCREASING,
                expected_range=MagnitudeRange(
                    lower_bound=200.0,
                    upper_bound=800.0,
                    unit=MeasurementUnit.HERTZ,
                ),
            ),
            tolerance=MagnitudeRange(
                lower_bound=0.0,
                upper_bound=20.0,
                unit=MeasurementUnit.HERTZ,
            ),
            generated_by=ForwardModel(label="Flow forward model"),
            comparison_target=observation,
            controlled_variable=FlowRateSetpoint(label="Flow rate"),
            predicted_for_phase=[phase],
        )

        violations = SUITE.evaluate((phase, auditory, observation, prediction))

        assert any(violation.check == "C56a" for violation in violations)


# %% material regimes


class TestMaterialRegime:
    """
    Pouring distinguishes liquid, granular, and cohesive transfer dynamics.
    """

    def test_generation_exposes_each_supported_regime(self) -> None:
        structure = pouring_schema().get(PouringPhaseStructure.__name__)
        assert structure is not None
        regime = next(
            field for field in structure.fields if field.name == "material_regime"
        )

        assert regime.options == tuple(member.name for member in MaterialRegime)
        assert regime.required == 1

    def test_granular_transfer_rejects_liquid_only_dynamics(self) -> None:
        viscosity = DynamicViscosityEstimate(label="Viscosity")
        slosh = SloshDynamicsEstimate(label="Slosh")
        structure = PouringPhaseStructure(
            label="Granular transfer",
            material_regime=MaterialRegime.GRANULAR,
            defines_parameter=[viscosity, slosh],
        )

        violations = SUITE.evaluate((structure, viscosity, slosh))

        assert any(violation.check == "C57" for violation in violations)

    def test_liquid_transfer_rejects_granular_dynamics(self) -> None:
        granular_dynamics = GranularFlowDynamicsEstimate(label="Granular dynamics")
        structure = PouringPhaseStructure(
            label="Liquid transfer",
            material_regime=MaterialRegime.LIQUID,
            defines_parameter=[granular_dynamics],
        )

        violations = SUITE.evaluate((structure, granular_dynamics))

        assert any(violation.check == "C57" for violation in violations)


# %% controlled variables


class TestPouringControlledVariables:
    """
    Every planned transfer-control variable has a typed representation.
    """

    def test_quantity_and_rate_support_mass_or_volume(self) -> None:
        assert TransferredQuantitySetpoint.permitted_dimensions == frozenset(
            {PhysicalDimension.MASS, PhysicalDimension.VOLUME}
        )
        assert FlowRateSetpoint.permitted_dimensions == frozenset(
            {
                PhysicalDimension.MASS_FLOW_RATE,
                PhysicalDimension.VOLUMETRIC_FLOW_RATE,
            }
        )

    def test_remaining_quantity_and_target_fill_are_distinct_states(self) -> None:
        assert RemainingSourceQuantityEstimate.permitted_dimensions == frozenset(
            {PhysicalDimension.MASS, PhysicalDimension.VOLUME}
        )
        assert (
            TargetFillStateEstimate.required_dimension
            is PhysicalDimension.DIMENSIONLESS
        )
        assert AlignmentErrorEstimate.required_dimension is PhysicalDimension.ANGLE


# %% task variants


class TestPouringVariants:
    """
    Valid pouring variants omit irrelevant prehensile and return phases.
    """

    def test_already_held_source_needs_only_transfer_phases(self) -> None:
        phases: list[ActionPhase] = [
            TiltToFlowPhase(label="Tilt"),
            FlowRegulationPhase(label="Transfer"),
            FlowTerminationPhase(label="Stop"),
        ]
        structure = PouringPhaseStructure(
            label="Already-held transfer",
            material_regime=MaterialRegime.GRANULAR,
            termination_reason=TransferTerminationReason.TARGET_QUANTITY,
            defines_phase=phases,
        )

        violations = SUITE.evaluate((structure, *phases))

        assert not any(violation.check == "C31" for violation in violations)

    def test_empty_source_and_requested_quantity_have_distinct_events(self) -> None:
        assert SourceEmpty is not TargetQuantityReached
        assert {
            TransferTerminationReason.TARGET_QUANTITY,
            TransferTerminationReason.SOURCE_EMPTY,
        } < set(TransferTerminationReason)

    def test_stop_reason_must_match_a_declared_boundary_event(self) -> None:
        target_reached = TargetQuantityReached(label="Target quantity reached")
        structure = PouringPhaseStructure(
            label="Source-empty transfer",
            material_regime=MaterialRegime.GRANULAR,
            termination_reason=TransferTerminationReason.SOURCE_EMPTY,
            defines_event_type=[target_reached],
        )

        violations = SUITE.evaluate((structure, target_reached))

        assert any(violation.check == "C58" for violation in violations)

    def test_stop_reason_accepts_its_declared_boundary_event(self) -> None:
        source_empty = SourceEmpty(label="Source empty")
        structure = PouringPhaseStructure(
            label="Source-empty transfer",
            material_regime=MaterialRegime.GRANULAR,
            termination_reason=TransferTerminationReason.SOURCE_EMPTY,
            defines_event_type=[source_empty],
        )

        violations = SUITE.evaluate((structure, source_empty))

        assert not any(violation.check == "C58" for violation in violations)
