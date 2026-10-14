from __future__ import annotations

from enum import Enum

import pytest
from llm_reasoner.knowledge_updated.microworlds.flanagan.core.value_partitions import (
    UNIT_DIMENSIONS,
    ComparisonOutcome,
    ConditionMode,
    ConditionPolarity,
    ContactTransition,
    CorrectionDirection,
    CouplingRegime,
    EstimateUpdateDirection,
    MeasurementUnit,
    ParameterValueForm,
    PhysicalDimension,
    PredictionErrorKind,
    RecoveryMode,
    RegulationMode,
    SensoryModality,
    TrajectoryTrend,
)
from llm_reasoner.knowledge_updated.microworlds.flanagan.ontology import CORE, FLANAGAN
from llm_reasoner.knowledge_updated.ontological.documentation import (
    DocumentationReader,
)

# %% fixtures

CORE_ENUMERATIONS = (
    CouplingRegime,
    RegulationMode,
    ComparisonOutcome,
    PredictionErrorKind,
    CorrectionDirection,
    RecoveryMode,
    EstimateUpdateDirection,
    ContactTransition,
    PhysicalDimension,
    MeasurementUnit,
    TrajectoryTrend,
    ConditionMode,
    ConditionPolarity,
    SensoryModality,
    ParameterValueForm,
)
"""
Every value partition of the core module.
"""

# %% declaration


class TestCorePartitionsAreDeclared:
    """
    Each enumeration is recorded in the ontology as a closed set of alternatives.
    """

    @pytest.mark.parametrize("enumeration", CORE_ENUMERATIONS)
    def test_enumeration_is_declared_in_the_core_module(
        self, enumeration: type[Enum]
    ) -> None:
        partition = FLANAGAN.find_value_partition(enumeration)

        assert partition is not None
        assert partition.module == CORE.name
        assert partition.iri == CORE.namespace.term(enumeration.__name__)

    def test_core_declares_no_other_partitions(self) -> None:
        declared = {p.enumeration for p in FLANAGAN.value_partitions(CORE.name)}

        assert declared == set(CORE_ENUMERATIONS)


# %% documentation


class TestCorePartitionsAreDocumented:
    """
    Every member of every enumeration says what it means.
    """

    @pytest.mark.parametrize("enumeration", CORE_ENUMERATIONS)
    def test_every_member_is_documented(self, enumeration: type[Enum]) -> None:
        assert DocumentationReader().find_undocumented_members(enumeration) == ()

    @pytest.mark.parametrize("enumeration", CORE_ENUMERATIONS)
    def test_enumeration_is_documented(self, enumeration: type[Enum]) -> None:
        assert DocumentationReader().read_class(enumeration) is not None


# %% modality


class TestSensoryModality:
    """
    A sensory modality is one of a closed set of channels.
    """

    def test_modalities_are_the_four_channels_the_review_distinguishes(self) -> None:
        assert {member.name for member in SensoryModality} == {
            "TACTILE",
            "PROPRIOCEPTIVE",
            "VISUAL",
            "AUDITORY",
        }

    def test_modality_equality_is_plain_equality(self) -> None:
        assert SensoryModality.VISUAL == SensoryModality.VISUAL
        assert SensoryModality.VISUAL != SensoryModality.TACTILE


# %% units and dimensions


class TestUnitsAndDimensions:
    """
    Every unit measures exactly one physical dimension.
    """

    def test_every_unit_has_a_dimension(self) -> None:
        assert set(UNIT_DIMENSIONS) == set(MeasurementUnit)

    def test_unit_reports_its_dimension(self) -> None:
        for unit in MeasurementUnit:
            assert unit.physical_dimension is UNIT_DIMENSIONS[unit]

    def test_every_dimension_is_measured_by_some_unit(self) -> None:
        assert set(UNIT_DIMENSIONS.values()) == set(PhysicalDimension)

    def test_both_time_units_measure_time(self) -> None:
        assert MeasurementUnit.SECOND.physical_dimension is PhysicalDimension.TIME
        assert MeasurementUnit.MILLISECOND.physical_dimension is PhysicalDimension.TIME

    def test_both_dimensionless_markers_measure_no_dimension(self) -> None:
        assert (
            MeasurementUnit.DIMENSIONLESS.physical_dimension
            is PhysicalDimension.DIMENSIONLESS
        )
        assert (
            MeasurementUnit.FRACTION.physical_dimension
            is PhysicalDimension.DIMENSIONLESS
        )

    def test_mass_and_weight_are_kept_apart(self) -> None:
        assert MeasurementUnit.KILOGRAM.physical_dimension is PhysicalDimension.MASS
        assert MeasurementUnit.NEWTON.physical_dimension is PhysicalDimension.FORCE
