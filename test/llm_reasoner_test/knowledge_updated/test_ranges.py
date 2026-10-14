from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Optional, Union

import pytest
from llm_reasoner.knowledge_updated.ontological.entities import KnowledgeEntity
from llm_reasoner.knowledge_updated.ontological.properties import (
    attribute,
    reference,
    relation,
)
from llm_reasoner.knowledge_updated.ontological.ranges import (
    PropertyRange,
    RangeResolver,
    UndeclaredField,
)
from llm_reasoner.knowledge_updated.ontological.references import ConceptReference

# %% fixtures


class Gait(StrEnum):
    """
    How a mover walks.
    """

    WALK = "walk"
    """
    One foot always on the ground.
    """


@dataclass(eq=False)
class Wheel(KnowledgeEntity):
    """
    A part that a vehicle rolls on.
    """


@dataclass(eq=False)
class Axle(KnowledgeEntity):
    """
    A part that joins two wheels.
    """


@dataclass(eq=False)
class Vehicle(KnowledgeEntity):
    """
    A vehicle whose fields exercise the ways a range can be written.
    """

    front: Optional[Wheel] = relation(functional=True)
    spare: Wheel | None = relation(functional=True)
    wheels: list[Wheel] = relation()
    quoted_wheels: list["Wheel"] = relation()
    later: Optional[Trailer] = relation(functional=True)
    either: Optional[Union[Wheel, Axle]] = relation(functional=True)
    towed: list[Vehicle] = relation()
    optional_list: Optional[list[Wheel]] = relation()
    count: int = attribute(functional=True, default=0)
    speed: Optional[float] = attribute(functional=True)
    gait: Optional[Gait] = attribute(functional=True)
    names: list[str] = attribute()
    kind: Optional[ConceptReference] = reference(functional=True)
    bare: list = attribute()
    anything: Any = attribute(functional=True)
    plain: str = ""


@dataclass(eq=False)
class Trailer(Vehicle):
    """
    A vehicle towed by another, declared after the class that refers to it.
    """


@dataclass
class Unresolvable:
    """
    A plain dataclass, not an entity, because a symbol with an undefined name in an
    annotation would break the symbol graph every test shares.
    """

    missing: Optional[NotDefinedAnywhere] = relation(functional=True)
    partly_missing: list[Wheel | AlsoNotDefined] = relation()
    fine: Optional[Wheel] = relation(functional=True)


def range_of(field_name: str) -> PropertyRange:
    """
    Resolve the range of one field of the fixture vehicle.
    """
    return RangeResolver().resolve(Vehicle, field_name)


# %% resolved ranges


class TestResolvedRanges:
    """
    A field's annotation is read as element types, cardinality and optionality.
    """

    @pytest.mark.parametrize(
        ("field_name", "element_types", "collection", "optional"),
        [
            ("front", (Wheel,), False, True),
            ("spare", (Wheel,), False, True),
            ("wheels", (Wheel,), True, False),
            ("quoted_wheels", (Wheel,), True, False),
            ("later", (Trailer,), False, True),
            ("either", (Wheel, Axle), False, True),
            ("towed", (Vehicle,), True, False),
            ("optional_list", (Wheel,), True, True),
            ("count", (int,), False, False),
            ("speed", (float,), False, True),
            ("gait", (Gait,), False, True),
            ("names", (str,), True, False),
            ("kind", (ConceptReference,), False, True),
        ],
    )
    def test_annotation_is_decomposed(
        self,
        field_name: str,
        element_types: tuple[type, ...],
        collection: bool,
        optional: bool,
    ) -> None:
        resolved = range_of(field_name)

        assert resolved.element_types == element_types
        assert resolved.collection is collection
        assert resolved.optional is optional
        assert resolved.unresolved_names == ()

    def test_class_declared_later_in_the_module_is_found(self) -> None:
        assert range_of("later").element_types == (Trailer,)

    def test_collection_without_an_element_type_has_no_element_types(self) -> None:
        resolved = range_of("bare")

        assert resolved.collection is True
        assert resolved.element_types == ()

    def test_unconstrained_annotation_has_no_element_types(self) -> None:
        resolved = range_of("anything")

        assert resolved.element_types == ()
        assert resolved.collection is False


# %% unresolved ranges


class TestUnresolvedRanges:
    """
    Names that cannot be found are reported instead of raising.
    """

    def test_undefined_name_is_reported(self) -> None:
        resolved = RangeResolver().resolve(Unresolvable, "missing")

        assert resolved.unresolved_names == ("NotDefinedAnywhere",)
        assert resolved.element_types == ()

    def test_only_the_undefined_names_of_a_hint_are_reported(self) -> None:
        resolved = RangeResolver().resolve(Unresolvable, "partly_missing")

        assert resolved.unresolved_names == ("AlsoNotDefined",)

    def test_unresolved_range_is_not_resolved(self) -> None:
        resolver = RangeResolver()

        assert resolver.resolve(Unresolvable, "missing").resolved is False
        assert resolver.resolve(Unresolvable, "fine").resolved is True

    def test_field_the_class_does_not_declare_is_refused(self) -> None:
        with pytest.raises(UndeclaredField) as refusal:
            RangeResolver().resolve(Unresolvable, "nonexistent")

        assert refusal.value.owner is Unresolvable
        assert refusal.value.field_name == "nonexistent"

    def test_inherited_field_must_be_asked_of_the_class_that_declares_it(self) -> None:
        with pytest.raises(UndeclaredField):
            RangeResolver().resolve(Trailer, "front")


# %% documentation


class TestRangeVocabularyIsDocumented:
    """
    Every field of the range vocabulary carries documentation.
    """

    def test_every_field_is_documented(self) -> None:
        from llm_reasoner.knowledge_updated.ontological.documentation import (
            DocumentationReader,
        )

        assert DocumentationReader().find_undocumented_fields(PropertyRange) == ()
