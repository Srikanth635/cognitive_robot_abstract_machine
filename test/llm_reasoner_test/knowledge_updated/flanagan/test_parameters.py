from __future__ import annotations

import pytest
from llm_reasoner.knowledge_updated.microworlds.flanagan.alignments import (
    DUL,
    SOMA,
    subclass_of,
)
from llm_reasoner.knowledge_updated.microworlds.flanagan.core.parameters import (
    ContactPropertyEstimate,
    DynamicsEstimate,
    Estimate,
    ForceRateSetpoint,
    MechanicalThreshold,
    MotorSetpoint,
    ObjectPropertyEstimate,
    ObjectStateEstimate,
    Parameter,
    SafetyMargin,
    Setpoint,
    TaskSetpoint,
)
from llm_reasoner.knowledge_updated.microworlds.flanagan.core.quantities import (
    CategoricalValue,
    MagnitudeRange,
)
from llm_reasoner.knowledge_updated.microworlds.flanagan.core.roles import RoleSlot
from llm_reasoner.knowledge_updated.microworlds.flanagan.core.value_partitions import (
    ParameterValueForm,
    PhysicalDimension,
)
from llm_reasoner.knowledge_updated.microworlds.flanagan.ontology import CORE, FLANAGAN
from llm_reasoner.knowledge_updated.ontological.documentation import (
    DocumentationReader,
)
from llm_reasoner.knowledge_updated.ontological.entities import KnowledgeEntity
from llm_reasoner.knowledge_updated.ontological.properties import PropertyMeta
from llm_reasoner.knowledge_updated.ontological.vocabulary import PropertyKind

# %% fixtures

ABSTRACT_PARAMETERS = (
    Parameter,
    Estimate,
    ObjectPropertyEstimate,
    ContactPropertyEstimate,
    ObjectStateEstimate,
    DynamicsEstimate,
    Setpoint,
    MotorSetpoint,
    TaskSetpoint,
    MechanicalThreshold,
)
"""
The parameter concepts that only group more specific ones.
"""

CONCRETE_PARAMETERS = (ForceRateSetpoint, SafetyMargin)
"""
The parameter concepts the core itself instantiates.
"""

# %% taxonomy


class TestParameterTaxonomy:
    """
    Parameters are of four disjoint kinds: estimates, setpoints, thresholds and margins.
    """

    @pytest.mark.parametrize("cls", ABSTRACT_PARAMETERS + CONCRETE_PARAMETERS)
    def test_concept_is_a_core_entity(self, cls: type) -> None:
        described = FLANAGAN.find_concept(cls)

        assert issubclass(cls, KnowledgeEntity)
        assert described.module == CORE.name

    @pytest.mark.parametrize("cls", ABSTRACT_PARAMETERS)
    def test_grouping_concepts_are_abstract(self, cls: type) -> None:
        assert FLANAGAN.find_concept(cls).abstract is True

    @pytest.mark.parametrize("cls", CONCRETE_PARAMETERS)
    def test_instantiated_concepts_are_concrete(self, cls: type) -> None:
        assert FLANAGAN.find_concept(cls).abstract is False

    @pytest.mark.parametrize(
        ("child", "parent"),
        [
            (Estimate, Parameter),
            (Setpoint, Parameter),
            (MechanicalThreshold, Parameter),
            (SafetyMargin, Parameter),
            (ObjectPropertyEstimate, Estimate),
            (ContactPropertyEstimate, Estimate),
            (ObjectStateEstimate, Estimate),
            (DynamicsEstimate, Estimate),
            (MotorSetpoint, Setpoint),
            (TaskSetpoint, Setpoint),
            (ForceRateSetpoint, MotorSetpoint),
        ],
    )
    def test_specialisations(self, child: type, parent: type) -> None:
        assert issubclass(child, parent)

    @pytest.mark.parametrize("parent", [Parameter, Estimate, Setpoint])
    def test_kinds_of_a_parameter_share_no_instance(self, parent: type) -> None:
        assert FLANAGAN.find_concept(parent).axioms.disjoint_siblings is True

    def test_parameters_are_aligned_where_the_foundational_terms_fit(self) -> None:
        assert FLANAGAN.find_concept(Parameter).alignments == (
            subclass_of(DUL, "Parameter"),
        )
        assert FLANAGAN.find_concept(Setpoint).alignments == (
            subclass_of(SOMA, "Setpoint"),
        )
        assert FLANAGAN.find_concept(MechanicalThreshold).alignments == (
            subclass_of(SOMA, "Threshold"),
        )


# %% what every parameter carries


class TestParameterProperties:
    """
    Every parameter says what form of value it holds, in which dimension, and for whom.
    """

    def declared(self) -> dict[str, PropertyMeta]:
        return PropertyMeta.declared_on(Parameter)

    def test_value_form_is_required_and_single_valued(self) -> None:
        described = self.declared()["value_form"]

        assert described.kind is PropertyKind.ATTRIBUTE
        assert described.functional is True
        assert described.minimum_count == 1

    def test_dimension_is_single_valued_and_optional(self) -> None:
        described = self.declared()["dimension"]

        assert described.kind is PropertyKind.ATTRIBUTE
        assert described.functional is True
        assert described.minimum_count == 0

    @pytest.mark.parametrize("name", ["parameter_range", "categorical_value"])
    def test_value_is_one_of_two_single_valued_relations(self, name: str) -> None:
        described = self.declared()[name]

        assert described.kind is PropertyKind.RELATION
        assert described.functional is True

    def test_a_parameter_gives_exactly_one_of_its_two_kinds_of_value(self) -> None:
        declared = self.declared()

        assert declared["parameter_range"].exclusive_group == "value"
        assert declared["categorical_value"].exclusive_group == "value"

    def test_parameter_names_the_roles_it_applies_to(self) -> None:
        described = self.declared()["parameter_on_role"]

        assert described.kind is PropertyKind.RELATION
        assert described.functional is False
        assert described.minimum_count == 1

    def test_parameter_does_not_name_the_phases_it_is_active_in(self) -> None:
        assert "parameter_in_phase" not in self.declared()

    def test_value_relations_range_over_the_quantity_concepts(self) -> None:
        from llm_reasoner.knowledge_updated.ontological.ranges import RangeResolver

        resolver = RangeResolver()

        assert resolver.resolve(Parameter, "parameter_range").element_types == (
            MagnitudeRange,
        )
        assert resolver.resolve(Parameter, "categorical_value").element_types == (
            CategoricalValue,
        )
        assert resolver.resolve(Parameter, "parameter_on_role").element_types == (
            RoleSlot,
        )


# %% links between parameters


class TestParameterLinks:
    """
    Estimates scale setpoints, and margins relate a setpoint to a threshold.
    """

    def test_estimate_scales_motor_setpoints(self) -> None:
        described = PropertyMeta.declared_on(Estimate)["scales"]

        assert described.kind is PropertyKind.RELATION
        assert described.functional is False

    def test_the_scaling_link_is_stored_on_one_side_only(self) -> None:
        assert "adapts_to" not in PropertyMeta.declared_on(MotorSetpoint)

    @pytest.mark.parametrize("name", ["margin_on", "margin_over"])
    def test_margin_relates_exactly_one_setpoint_and_one_threshold(
        self, name: str
    ) -> None:
        described = PropertyMeta.declared_on(SafetyMargin)[name]

        assert described.kind is PropertyKind.RELATION
        assert described.functional is True
        assert described.minimum_count == 1

    def test_force_rate_setpoint_names_the_setpoint_whose_rate_it_commands(
        self,
    ) -> None:
        described = PropertyMeta.declared_on(ForceRateSetpoint)["rate_of"]

        assert described.kind is PropertyKind.RELATION
        assert described.functional is True
        assert described.minimum_count == 1


# %% what the classes state about their values


class TestParameterValueRestrictions:
    """
    Concrete parameters say the form and dimension of their value in their declaration.
    """

    def test_force_rate_setpoint_is_a_quantity_in_force_rate(self) -> None:
        axioms = FLANAGAN.find_concept(ForceRateSetpoint).axioms

        assert axioms.fixed_value_of("value_form") is ParameterValueForm.QUANTITATIVE
        assert axioms.fixed_value_of("dimension") is PhysicalDimension.FORCE_RATE

    def test_safety_margin_is_a_quantity_in_any_dimension(self) -> None:
        axioms = FLANAGAN.find_concept(SafetyMargin).axioms

        assert axioms.fixed_value_of("value_form") is ParameterValueForm.QUANTITATIVE
        assert axioms.fixed_value_of("dimension") is None
        assert axioms.allowed_values_of("dimension") is None


# %% documentation


class TestParametersAreDocumented:
    """
    Every parameter concept and property says what it means.
    """

    @pytest.mark.parametrize("cls", ABSTRACT_PARAMETERS + CONCRETE_PARAMETERS)
    def test_concept_and_properties_are_documented(self, cls: type) -> None:
        reader = DocumentationReader()

        assert reader.read_class(cls) is not None
        assert reader.find_undocumented_fields(cls) == ()
