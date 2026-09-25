from __future__ import annotations

from cram_viz.hypothesis_knowledge.pouring_structure import (
    build_pouring_phase_structure,
    build_water_pouring_phase_structure,
)
from llm_reasoner.knowledge.microworlds.flanagan.core import PhysicalDimension
from llm_reasoner.knowledge.microworlds.flanagan.pouring import (
    DynamicViscosityEstimate,
    FlowRateSetpoint,
    GranularFlowDynamicsEstimate,
    MaterialRegime,
    SloshDynamicsEstimate,
    TransferredQuantitySetpoint,
)

# %% material-specific reference structures


class TestMaterialSpecificReferenceStructures:
    """
    Reference structures distinguish cereal and water dynamics.
    """

    def test_cereal_uses_granular_dynamics_and_mass_quantities(self) -> None:
        structure = build_pouring_phase_structure()

        assert structure.material_regime is MaterialRegime.GRANULAR
        assert any(
            isinstance(parameter, GranularFlowDynamicsEstimate)
            for parameter in structure.defines_parameter
        )
        assert not any(
            isinstance(parameter, (DynamicViscosityEstimate, SloshDynamicsEstimate))
            for parameter in structure.defines_parameter
        )
        quantity = next(
            parameter
            for parameter in structure.defines_parameter
            if isinstance(parameter, TransferredQuantitySetpoint)
        )
        flow_rate = next(
            parameter
            for parameter in structure.defines_parameter
            if isinstance(parameter, FlowRateSetpoint)
        )

        assert quantity.parameter_range is not None
        assert quantity.parameter_range.physical_dimension is PhysicalDimension.MASS
        assert flow_rate.parameter_range is not None
        assert (
            flow_rate.parameter_range.physical_dimension
            is PhysicalDimension.MASS_FLOW_RATE
        )

    def test_water_uses_viscosity_slosh_and_volume_quantities(self) -> None:
        structure = build_water_pouring_phase_structure()

        assert structure.material_regime is MaterialRegime.LIQUID
        assert any(
            isinstance(parameter, DynamicViscosityEstimate)
            for parameter in structure.defines_parameter
        )
        assert any(
            isinstance(parameter, SloshDynamicsEstimate)
            for parameter in structure.defines_parameter
        )
        assert not any(
            isinstance(parameter, GranularFlowDynamicsEstimate)
            for parameter in structure.defines_parameter
        )
        quantity = next(
            parameter
            for parameter in structure.defines_parameter
            if isinstance(parameter, TransferredQuantitySetpoint)
        )

        assert quantity.parameter_range is not None
        assert quantity.parameter_range.physical_dimension is PhysicalDimension.VOLUME


# %% continuous comparison fixtures


class TestContinuousComparisonFixture:
    """
    Canonical continuous predictions expose executable comparison inputs.
    """

    def test_each_cereal_prediction_names_its_complete_contract(self) -> None:
        structure = build_pouring_phase_structure()

        assert structure.has_continuous_prediction
        for prediction in structure.has_continuous_prediction:
            assert prediction.signal is prediction.comparison_target.modality
            assert prediction.predicted_trajectory is not None
            assert prediction.tolerance is not None
            assert prediction.generated_by is not None
            assert prediction.controlled_variable is not None
            assert prediction.predicted_for_phase
