from __future__ import annotations

import pytest
from llm_reasoner.knowledge_updated.microworlds.flanagan.alignments import (
    DUL,
    subclass_of,
)
from llm_reasoner.knowledge_updated.microworlds.flanagan.core.quantities import (
    CategoricalValue,
    MagnitudeRange,
    TemporalWindow,
)
from llm_reasoner.knowledge_updated.microworlds.flanagan.core.value_partitions import (
    TIME_UNITS,
    MeasurementUnit,
    PhysicalDimension,
)
from llm_reasoner.knowledge_updated.microworlds.flanagan.ontology import CORE, FLANAGAN
from llm_reasoner.knowledge_updated.ontological.documentation import (
    DocumentationReader,
)
from llm_reasoner.knowledge_updated.ontological.entities import KnowledgeEntity
from llm_reasoner.knowledge_updated.ontological.properties import PropertyMeta
from llm_reasoner.knowledge_updated.ontological.vocabulary import PropertyKind

# %% declaration


class TestQuantitiesAreDeclared:
    """
    Quantities are description-level knowledge declared in the core module.
    """

    @pytest.mark.parametrize("cls", [MagnitudeRange, TemporalWindow, CategoricalValue])
    def test_concept_is_a_core_entity(self, cls: type) -> None:
        described = FLANAGAN.find_concept(cls)

        assert issubclass(cls, KnowledgeEntity)
        assert described.module == CORE.name

    @pytest.mark.parametrize("cls", [MagnitudeRange, TemporalWindow])
    def test_ranges_are_aligned_as_regions_of_a_dimensional_space(
        self, cls: type
    ) -> None:
        assert FLANAGAN.find_concept(cls).alignments == (subclass_of(DUL, "Region"),)


# %% magnitude ranges


class TestMagnitudeRange:
    """
    A magnitude range is an interval whose unit fixes its physical dimension.
    """

    @pytest.mark.parametrize("name", ["lower_bound", "upper_bound", "unit"])
    def test_every_property_is_a_required_single_valued_attribute(
        self, name: str
    ) -> None:
        described = PropertyMeta.declared_on(MagnitudeRange)[name]

        assert described.kind is PropertyKind.ATTRIBUTE
        assert described.functional is True
        assert described.minimum_count == 1

    def test_dimension_is_derived_from_the_unit_and_never_stored(self) -> None:
        assert "physical_dimension" not in PropertyMeta.declared_on(MagnitudeRange)

    def test_dimension_follows_the_unit(self) -> None:
        magnitude = MagnitudeRange(
            lower_bound=0.2, upper_bound=0.5, unit=MeasurementUnit.KILOGRAM
        )

        assert magnitude.physical_dimension is PhysicalDimension.MASS

        magnitude.unit = MeasurementUnit.NEWTON

        assert magnitude.physical_dimension is PhysicalDimension.FORCE

    def test_range_without_a_unit_has_no_dimension(self) -> None:
        assert MagnitudeRange().physical_dimension is None


# %% temporal windows


class TestTemporalWindow:
    """
    A temporal window is an interval of time relative to a control event.
    """

    @pytest.mark.parametrize("name", ["earliest_offset", "latest_offset", "unit"])
    def test_every_property_is_a_required_single_valued_attribute(
        self, name: str
    ) -> None:
        described = PropertyMeta.declared_on(TemporalWindow)[name]

        assert described.kind is PropertyKind.ATTRIBUTE
        assert described.functional is True
        assert described.minimum_count == 1

    def test_unit_is_restricted_to_units_of_time(self) -> None:
        axioms = FLANAGAN.find_concept(TemporalWindow).axioms

        assert axioms.allowed_values_of("unit") == TIME_UNITS

    def test_time_units_are_exactly_the_units_that_measure_time(self) -> None:
        assert TIME_UNITS == {
            unit
            for unit in MeasurementUnit
            if unit.physical_dimension is PhysicalDimension.TIME
        }
        assert TIME_UNITS == {MeasurementUnit.SECOND, MeasurementUnit.MILLISECOND}


# %% categorical values


class TestCategoricalValue:
    """
    A categorical value names a category instead of giving a number.
    """

    def test_category_is_a_required_single_term_reference(self) -> None:
        described = PropertyMeta.declared_on(CategoricalValue)["category"]

        assert described.kind is PropertyKind.REFERENCE
        assert described.functional is True
        assert described.minimum_count == 1


# %% documentation


class TestQuantitiesAreDocumented:
    """
    Every quantity concept and property says what it means.
    """

    @pytest.mark.parametrize("cls", [MagnitudeRange, TemporalWindow, CategoricalValue])
    def test_concept_and_properties_are_documented(self, cls: type) -> None:
        reader = DocumentationReader()

        assert reader.read_class(cls) is not None
        assert reader.find_undocumented_fields(cls) == ()
