from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pytest
from llm_reasoner.knowledge_updated.ontological.documentation import (
    DocumentationReader,
)
from llm_reasoner.knowledge_updated.ontological.properties import (
    PROPERTY_METADATA_KEY,
    InvalidPropertyDeclaration,
    PropertyFault,
    PropertyMeta,
    attribute,
    reference,
    relation,
)
from llm_reasoner.knowledge_updated.ontological.references import ConceptReference
from llm_reasoner.knowledge_updated.ontological.vocabulary import PropertyKind

# %% fixtures


@dataclass
class Wheel:
    """
    A part that a vehicle rolls on.
    """

    radius: float = attribute(functional=True, default=0.3)
    """
    Distance from the axle to the tyre.
    """


@dataclass
class Vehicle:
    """
    A vehicle with described and undescribed fields.
    """

    driver_name: Optional[str] = attribute(functional=True, minimum_count=1)
    front_wheel: Optional[Wheel] = relation(functional=True)
    wheels: list[Wheel] = relation(minimum_count=2, inverse_of="vehicle")
    towed: list[Vehicle] = relation(transitive=True)
    nicknames: list[str] = attribute()
    kind: Optional[ConceptReference] = reference(functional=True, concept_root=Wheel)
    note: str = ""


@dataclass
class ElectricVehicle(Vehicle):
    """
    A vehicle that adds one described field to those it inherits.
    """

    battery: Optional[Wheel] = relation(functional=True, iri="urn:test:battery")


@dataclass
class Alternatives:
    """
    A class whose two fields are alternatives.
    """

    by_name: Optional[str] = attribute(
        functional=True, exclusive_group="identification"
    )
    by_number: Optional[int] = attribute(
        functional=True, exclusive_group="identification"
    )


@dataclass
class Container:
    """
    A class with one relation that declares its parts and one that only refers.
    """

    parts: list[Wheel] = relation(owns=True)
    borrowed: list[Wheel] = relation()


@dataclass
class Marked:
    """
    A class with a property that marks its fragment as an assumption.
    """

    presupposed: bool = attribute(functional=True, default=False, marks_assumption=True)


class NotADataclass:
    """
    A class that declares no fields.
    """


# %% declared metadata


class TestDeclaredMetadata:
    """
    A helper attaches the properties' characteristics to the field it returns.
    """

    def test_kinds_follow_the_helper_used(self) -> None:
        declared = PropertyMeta.declared_on(Vehicle)

        assert declared["driver_name"].kind is PropertyKind.ATTRIBUTE
        assert declared["front_wheel"].kind is PropertyKind.RELATION
        assert declared["kind"].kind is PropertyKind.REFERENCE

    def test_characteristics_are_recorded(self) -> None:
        declared = PropertyMeta.declared_on(Vehicle)

        assert declared["front_wheel"].functional is True
        assert declared["wheels"].functional is False
        assert declared["wheels"].minimum_count == 2
        assert declared["wheels"].inverse_of == "vehicle"
        assert declared["towed"].transitive is True
        assert declared["nicknames"].transitive is False

    def test_unspecified_characteristics_take_their_neutral_value(self) -> None:
        described = PropertyMeta.declared_on(Vehicle)["nicknames"]

        assert described.functional is False
        assert described.minimum_count == 0
        assert described.transitive is False
        assert described.inverse_of is None
        assert described.iri is None
        assert described.concept_root is None

    def test_reference_records_the_concept_root_that_closes_it(self) -> None:
        assert PropertyMeta.declared_on(Vehicle)["kind"].concept_root is Wheel

    def test_explicit_iri_is_recorded(self) -> None:
        assert (
            PropertyMeta.declared_on(ElectricVehicle)["battery"].iri
            == "urn:test:battery"
        )

    def test_only_described_fields_are_listed_in_declaration_order(self) -> None:
        assert list(PropertyMeta.declared_on(Vehicle)) == [
            "driver_name",
            "front_wheel",
            "wheels",
            "towed",
            "nicknames",
            "kind",
        ]

    def test_inherited_properties_are_listed_before_added_ones(self) -> None:
        declared = PropertyMeta.declared_on(ElectricVehicle)

        assert list(declared)[-1] == "battery"
        assert declared["wheels"] == PropertyMeta.declared_on(Vehicle)["wheels"]

    def test_class_that_is_not_a_dataclass_declares_nothing(self) -> None:
        assert PropertyMeta.declared_on(NotADataclass) == {}


# %% field defaults


class TestFieldDefaults:
    """
    Cardinality decides what a new instance holds.
    """

    def test_functional_field_without_a_default_starts_empty(self) -> None:
        assert Vehicle(driver_name=None).front_wheel is None

    def test_functional_attribute_starts_at_its_default(self) -> None:
        assert Wheel().radius == 0.3

    def test_collection_field_starts_as_an_empty_list(self) -> None:
        assert Vehicle(driver_name=None).wheels == []

    def test_collections_are_not_shared_between_instances(self) -> None:
        first = Vehicle(driver_name=None)
        second = Vehicle(driver_name=None)

        first.wheels.append(Wheel())

        assert second.wheels == []


# %% rejected declarations


class TestRejectedDeclarations:
    """
    A contradictory declaration is refused where it is written.
    """

    def fault_of(self, declaration) -> PropertyFault:
        with pytest.raises(InvalidPropertyDeclaration) as rejection:
            declaration()
        return rejection.value.reason

    def test_negative_minimum_count(self) -> None:
        assert (
            self.fault_of(lambda: attribute(minimum_count=-1))
            is PropertyFault.NEGATIVE_MINIMUM_COUNT
        )

    def test_functional_property_cannot_require_more_than_one_value(self) -> None:
        assert (
            self.fault_of(lambda: relation(functional=True, minimum_count=2))
            is PropertyFault.REQUIRED_EXCEEDS_SINGLE_VALUE
        )

    def test_collection_attribute_cannot_carry_a_default(self) -> None:
        assert (
            self.fault_of(lambda: attribute(default=3))
            is PropertyFault.DEFAULT_FOR_COLLECTION
        )

    def test_only_a_relation_can_be_transitive(self) -> None:
        assert (
            self.fault_of(
                lambda: PropertyMeta(kind=PropertyKind.ATTRIBUTE, transitive=True)
            )
            is PropertyFault.TRANSITIVE_NON_RELATION
        )

    def test_only_a_relation_can_have_an_inverse(self) -> None:
        assert (
            self.fault_of(
                lambda: PropertyMeta(kind=PropertyKind.REFERENCE, inverse_of="x")
            )
            is PropertyFault.INVERSE_NON_RELATION
        )

    def test_only_a_reference_can_have_a_concept_root(self) -> None:
        assert (
            self.fault_of(
                lambda: PropertyMeta(kind=PropertyKind.RELATION, concept_root=Wheel)
            )
            is PropertyFault.CONCEPT_ROOT_NON_REFERENCE
        )


# %% ownership


class TestOwnership:
    """
    A relation can declare that its concept is where the related instances are declared.
    """

    def test_owning_relation_is_recorded(self) -> None:
        assert PropertyMeta.declared_on(Container)["parts"].owns is True

    def test_relations_own_nothing_unless_declared(self) -> None:
        assert PropertyMeta.declared_on(Container)["borrowed"].owns is False
        assert PropertyMeta.declared_on(Vehicle)["towed"].owns is False

    def test_only_a_relation_can_own(self) -> None:
        with pytest.raises(InvalidPropertyDeclaration) as rejection:
            PropertyMeta(kind=PropertyKind.ATTRIBUTE, owns=True)

        assert rejection.value.reason is PropertyFault.OWNERSHIP_NON_RELATION


# %% assumption markers


class TestAssumptionMarkers:
    """
    A boolean attribute can declare that, when true, its fragment is presupposed.
    """

    def test_marker_is_recorded_on_an_attribute(self) -> None:
        described = PropertyMeta.declared_on(Marked)["presupposed"]

        assert described.marks_assumption is True

    def test_properties_mark_no_assumption_unless_declared(self) -> None:
        assert PropertyMeta.declared_on(Vehicle)["nicknames"].marks_assumption is False

    def test_only_an_attribute_can_mark_an_assumption(self) -> None:
        with pytest.raises(InvalidPropertyDeclaration) as rejection:
            PropertyMeta(
                kind=PropertyKind.RELATION, functional=True, marks_assumption=True
            )

        assert rejection.value.reason is PropertyFault.ASSUMPTION_NON_ATTRIBUTE

    def test_a_marker_holds_one_value(self) -> None:
        with pytest.raises(InvalidPropertyDeclaration) as rejection:
            attribute(marks_assumption=True)

        assert rejection.value.reason is PropertyFault.ASSUMPTION_NOT_FUNCTIONAL


# %% exclusive groups


class TestExclusiveGroups:
    """
    Fields sharing a group label are alternatives of which exactly one is given.
    """

    def test_group_label_is_recorded(self) -> None:
        declared = PropertyMeta.declared_on(Alternatives)

        assert declared["by_name"].exclusive_group == "identification"
        assert declared["by_number"].exclusive_group == "identification"

    def test_properties_belong_to_no_group_unless_declared(self) -> None:
        assert PropertyMeta.declared_on(Vehicle)["nicknames"].exclusive_group is None

    @pytest.mark.parametrize("declare", [relation, attribute, reference])
    def test_every_kind_of_property_can_join_a_group(self, declare) -> None:
        described = declare(exclusive_group="choice").metadata[PROPERTY_METADATA_KEY]

        assert described.exclusive_group == "choice"

    def test_a_member_cannot_be_required_by_itself(self) -> None:
        with pytest.raises(InvalidPropertyDeclaration) as rejection:
            relation(exclusive_group="choice", minimum_count=1)

        assert rejection.value.reason is PropertyFault.EXCLUSIVE_GROUP_MEMBER_REQUIRED


# %% documentation


class TestPropertyVocabularyIsDocumented:
    """
    Every field and member of the property vocabulary carries documentation.
    """

    def test_metadata_fields_are_documented(self) -> None:
        assert DocumentationReader().find_undocumented_fields(PropertyMeta) == ()

    def test_fault_members_are_documented(self) -> None:
        assert DocumentationReader().find_undocumented_members(PropertyFault) == ()
