from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Optional

import pytest
from llm_reasoner.knowledge_updated.ontological.axioms import (
    AllowedValues,
    AxiomFault,
    ClassAxioms,
    FixedValue,
    InvalidAxiom,
)
from llm_reasoner.knowledge_updated.ontological.documentation import (
    DocumentationReader,
)
from llm_reasoner.knowledge_updated.ontological.properties import (
    attribute,
    reference,
    relation,
)
from llm_reasoner.knowledge_updated.ontological.references import ConceptReference

# %% fixtures


class Direction(StrEnum):
    """
    A way to move.
    """

    UP = "up"
    """
    Towards the ceiling.
    """

    DOWN = "down"
    """
    Towards the floor.
    """


@dataclass
class Mover:
    """
    Something that moves in a direction.
    """

    direction: Optional[Direction] = attribute(functional=True)
    """
    The way the mover travels.
    """

    waypoints: list[int] = attribute()
    """
    The stops on the way.
    """

    partner: Optional[Mover] = relation(functional=True)
    """
    Another mover travelling alongside.
    """

    destination: Optional[ConceptReference] = reference(functional=True)
    """
    The term naming where the mover is heading.
    """

    note: str = ""
    """
    A remark that is not a described property.
    """


@dataclass
class Lifter(Mover):
    """
    A mover that only goes up.
    """


# %% construction


class TestClassAxiomConstruction:
    """
    Class axioms are built from the arguments a concept declaration passes.
    """

    def test_no_axioms_is_the_neutral_case(self) -> None:
        axioms = ClassAxioms.of()

        assert axioms.disjoint_with == ()
        assert axioms.disjoint_siblings is False
        assert axioms.fixed_values == ()

    def test_fixed_values_keep_the_order_they_were_given_in(self) -> None:
        axioms = ClassAxioms.of(
            fixed_values={"direction": Direction.UP, "note": "fixed"}
        )

        assert axioms.fixed_values == (
            FixedValue(property_name="direction", value=Direction.UP),
            FixedValue(property_name="note", value="fixed"),
        )

    def test_disjoint_names_are_kept_as_given(self) -> None:
        axioms = ClassAxioms.of(disjoint_with=["Mover", "Lifter"])

        assert axioms.disjoint_with == ("Mover", "Lifter")

    def test_fixed_value_is_found_by_property_name(self) -> None:
        axioms = ClassAxioms.of(fixed_values={"direction": Direction.UP})

        assert axioms.fixed_value_of("direction") == Direction.UP
        assert axioms.fixed_value_of("waypoints") is None


# %% checking against a class


class TestAxiomsAreCheckedAgainstTheirClass:
    """
    Fixed values must name a property the class can fix.
    """

    def test_fixed_functional_attribute_is_accepted(self) -> None:
        ClassAxioms.of(fixed_values={"direction": Direction.UP}).check_against(Mover)

    def test_property_inherited_from_a_base_class_is_accepted(self) -> None:
        ClassAxioms.of(fixed_values={"direction": Direction.UP}).check_against(Lifter)

    def test_axioms_without_fixed_values_are_always_accepted(self) -> None:
        ClassAxioms.of(disjoint_with=["Anything"]).check_against(Mover)

    def fault_of(self, axioms: ClassAxioms) -> InvalidAxiom:
        with pytest.raises(InvalidAxiom) as rejection:
            axioms.check_against(Mover)
        return rejection.value

    def test_property_the_class_does_not_declare(self) -> None:
        rejection = self.fault_of(ClassAxioms.of(fixed_values={"missing": 1}))

        assert rejection.reason is AxiomFault.UNKNOWN_PROPERTY
        assert rejection.subject == "missing"

    def test_field_that_is_not_a_described_property(self) -> None:
        rejection = self.fault_of(ClassAxioms.of(fixed_values={"note": "x"}))

        assert rejection.reason is AxiomFault.UNKNOWN_PROPERTY
        assert rejection.subject == "note"

    @pytest.mark.parametrize("name", ["waypoints", "partner", "destination"])
    def test_only_a_functional_attribute_can_be_fixed(self, name: str) -> None:
        rejection = self.fault_of(ClassAxioms.of(fixed_values={name: 1}))

        assert rejection.reason is AxiomFault.NOT_A_FUNCTIONAL_ATTRIBUTE
        assert rejection.subject == name

    def test_property_fixed_twice(self) -> None:
        axioms = ClassAxioms(
            fixed_values=(
                FixedValue(property_name="direction", value=Direction.UP),
                FixedValue(property_name="direction", value=Direction.DOWN),
            )
        )

        rejection = self.fault_of(axioms)

        assert rejection.reason is AxiomFault.PROPERTY_FIXED_TWICE
        assert rejection.subject == "direction"


# %% intended overlap


class TestIntendedOverlap:
    """
    A concept can say that it may share instances with a sibling.
    """

    def test_overlap_is_neutral_by_default(self) -> None:
        assert ClassAxioms.of().may_overlap_with == ()

    def test_overlap_names_are_kept_as_given(self) -> None:
        axioms = ClassAxioms.of(may_overlap_with=["Mover", "Lifter"])

        assert axioms.may_overlap_with == ("Mover", "Lifter")

    def test_overlap_names_need_no_check_against_the_class(self) -> None:
        ClassAxioms.of(may_overlap_with=["Anything"]).check_against(Mover)


# %% allowed values


class TestAllowedValues:
    """
    A concept can restrict a functional attribute to some of the values it could take.
    """

    def test_allowed_values_are_built_from_a_mapping(self) -> None:
        axioms = ClassAxioms.of(allowed_values={"direction": [Direction.UP]})

        assert axioms.allowed_values == (
            AllowedValues(property_name="direction", values=frozenset({Direction.UP})),
        )

    def test_allowed_values_are_found_by_property_name(self) -> None:
        axioms = ClassAxioms.of(
            allowed_values={"direction": [Direction.UP, Direction.DOWN]}
        )

        assert axioms.allowed_values_of("direction") == frozenset(Direction)
        assert axioms.allowed_values_of("waypoints") is None

    def test_no_restriction_is_the_neutral_case(self) -> None:
        assert ClassAxioms.of().allowed_values == ()

    def test_restriction_of_a_functional_attribute_is_accepted(self) -> None:
        ClassAxioms.of(allowed_values={"direction": [Direction.UP]}).check_against(
            Lifter
        )

    def test_fixed_value_inside_the_allowed_values_is_accepted(self) -> None:
        ClassAxioms.of(
            fixed_values={"direction": Direction.UP},
            allowed_values={"direction": [Direction.UP, Direction.DOWN]},
        ).check_against(Mover)

    def rejection_of(self, axioms: ClassAxioms) -> InvalidAxiom:
        with pytest.raises(InvalidAxiom) as rejection:
            axioms.check_against(Mover)
        return rejection.value

    def test_restricting_a_property_the_class_does_not_declare(self) -> None:
        rejection = self.rejection_of(ClassAxioms.of(allowed_values={"missing": [1]}))

        assert rejection.reason is AxiomFault.UNKNOWN_PROPERTY
        assert rejection.subject == "missing"

    @pytest.mark.parametrize("name", ["waypoints", "partner", "destination"])
    def test_only_a_functional_attribute_can_be_restricted(self, name: str) -> None:
        rejection = self.rejection_of(ClassAxioms.of(allowed_values={name: [1]}))

        assert rejection.reason is AxiomFault.NOT_A_FUNCTIONAL_ATTRIBUTE

    def test_a_restriction_must_allow_something(self) -> None:
        rejection = self.rejection_of(ClassAxioms.of(allowed_values={"direction": []}))

        assert rejection.reason is AxiomFault.NO_ALLOWED_VALUES
        assert rejection.subject == "direction"

    def test_property_restricted_twice(self) -> None:
        axioms = ClassAxioms(
            allowed_values=(
                AllowedValues("direction", frozenset({Direction.UP})),
                AllowedValues("direction", frozenset({Direction.DOWN})),
            )
        )

        assert self.rejection_of(axioms).reason is AxiomFault.PROPERTY_RESTRICTED_TWICE

    def test_fixed_value_outside_the_allowed_values(self) -> None:
        axioms = ClassAxioms.of(
            fixed_values={"direction": Direction.DOWN},
            allowed_values={"direction": [Direction.UP]},
        )

        rejection = self.rejection_of(axioms)

        assert rejection.reason is AxiomFault.FIXED_VALUE_NOT_ALLOWED
        assert rejection.subject == "direction"


# %% documentation


class TestAxiomVocabularyIsDocumented:
    """
    Every field and member of the axiom vocabulary carries documentation.
    """

    @pytest.mark.parametrize("dataclass_type", [FixedValue, AllowedValues, ClassAxioms])
    def test_every_field_is_documented(self, dataclass_type: type) -> None:
        assert DocumentationReader().find_undocumented_fields(dataclass_type) == ()

    def test_every_fault_is_documented(self) -> None:
        assert DocumentationReader().find_undocumented_members(AxiomFault) == ()
