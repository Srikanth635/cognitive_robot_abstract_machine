"""
Declaring the properties of a concept: what a field holds and how many values it takes.

A property is an ordinary dataclass field that carries its characteristics in the
field's metadata, so nothing downstream needs to know a field is described. There is one
helper per kind of property, each accepting only the characteristics that make sense for
it.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass
from enum import StrEnum
from typing import Any, Optional

from llm_reasoner.exceptions import LLMReasonerError
from llm_reasoner.knowledge_updated.ontological.vocabulary import PropertyKind

PROPERTY_METADATA_KEY = "ontological_property"
"""
The key under which a field's :class:`PropertyMeta` is stored in its metadata.
"""

# %% invalid declarations


class PropertyFault(StrEnum):
    """
    Why a property declaration contradicts itself.
    """

    NEGATIVE_MINIMUM_COUNT = "negative_minimum_count"
    """
    The minimum number of values is below zero.
    """

    REQUIRED_EXCEEDS_SINGLE_VALUE = "required_exceeds_single_value"
    """
    A property with at most one value is required to have more than one.
    """

    DEFAULT_FOR_COLLECTION = "default_for_collection"
    """
    A property with many values is given a single default value.
    """

    TRANSITIVE_NON_RELATION = "transitive_non_relation"
    """
    A property that does not link instances is declared transitive.
    """

    INVERSE_NON_RELATION = "inverse_non_relation"
    """
    A property that does not link instances is given an inverse.
    """

    CONCEPT_ROOT_NON_REFERENCE = "concept_root_non_reference"
    """
    A property that does not hold a term reference is restricted to a concept root.
    """

    ASSUMPTION_NON_ATTRIBUTE = "assumption_non_attribute"
    """
    A property that is not a plain attribute is declared to mark an assumption.
    """

    ASSUMPTION_NOT_FUNCTIONAL = "assumption_not_functional"
    """
    A property with many values is declared to mark an assumption.
    """

    OWNERSHIP_NON_RELATION = "ownership_non_relation"
    """
    A property that does not link instances is declared to own what it holds.
    """

    EXCLUSIVE_GROUP_MEMBER_REQUIRED = "exclusive_group_member_required"
    """
    A property that belongs to a group of alternatives is also required by itself.
    """


@dataclass
class InvalidPropertyDeclaration(LLMReasonerError):
    """
    A property is declared with characteristics that contradict each other.
    """

    reason: PropertyFault
    """
    The contradiction found.
    """

    def __str__(self) -> str:
        return f"invalid property declaration: {self.reason.value}"


# %% property metadata


@dataclass(frozen=True)
class PropertyMeta:
    """
    The characteristics of one described field.
    """

    kind: PropertyKind
    """
    What the property holds.
    """

    functional: bool = False
    """
    Whether the property has at most one value, in which case its field is optional
    instead of a list.
    """

    minimum_count: int = 0
    """
    How many values a well-formed instance needs. Not enforced at construction, since a
    hypothesis is assembled incrementally.
    """

    transitive: bool = False
    """
    Whether a value of a value of the property is also a value of it.
    """

    inverse_of: Optional[str] = None
    """
    Name of the field on the value's class that runs the other way.
    """

    iri: Optional[str] = None
    """
    The property's own identifier, when it is not derived from the field name.
    """

    concept_root: Optional[type] = None
    """
    The concept whose descendants a reference may name, when the reference is closed
    over a local vocabulary.
    """

    owns: bool = False
    """
    Whether the concept is where the related instances are declared, so that every
    other reference to such an instance must point at one declared here.
    """

    exclusive_group: Optional[str] = None
    """
    Label shared by properties of one concept that are alternatives: exactly one of them
    is given, and no member is required by itself.
    """

    marks_assumption: bool = False
    """
    Whether a true value flags its fragment as something the plan presupposes and does not
    bring about. Declared on the property so assumptions are found by metadata and not by
    field name.
    """

    def __post_init__(self) -> None:
        """
        Reject characteristics that contradict each other or the property's kind.

        :raises InvalidPropertyDeclaration: on the first contradiction found.
        """
        if self.minimum_count < 0:
            raise InvalidPropertyDeclaration(PropertyFault.NEGATIVE_MINIMUM_COUNT)
        if self.functional and self.minimum_count > 1:
            raise InvalidPropertyDeclaration(
                PropertyFault.REQUIRED_EXCEEDS_SINGLE_VALUE
            )
        if self.transitive and self.kind is not PropertyKind.RELATION:
            raise InvalidPropertyDeclaration(PropertyFault.TRANSITIVE_NON_RELATION)
        if self.inverse_of is not None and self.kind is not PropertyKind.RELATION:
            raise InvalidPropertyDeclaration(PropertyFault.INVERSE_NON_RELATION)
        if self.concept_root is not None and self.kind is not PropertyKind.REFERENCE:
            raise InvalidPropertyDeclaration(PropertyFault.CONCEPT_ROOT_NON_REFERENCE)
        if self.owns and self.kind is not PropertyKind.RELATION:
            raise InvalidPropertyDeclaration(PropertyFault.OWNERSHIP_NON_RELATION)
        if self.exclusive_group is not None and self.minimum_count > 0:
            raise InvalidPropertyDeclaration(
                PropertyFault.EXCLUSIVE_GROUP_MEMBER_REQUIRED
            )
        if self.marks_assumption and self.kind is not PropertyKind.ATTRIBUTE:
            raise InvalidPropertyDeclaration(PropertyFault.ASSUMPTION_NON_ATTRIBUTE)
        if self.marks_assumption and not self.functional:
            raise InvalidPropertyDeclaration(PropertyFault.ASSUMPTION_NOT_FUNCTIONAL)

    @classmethod
    def declared_on(cls, owner: type) -> dict[str, PropertyMeta]:
        """
        Return the described fields of a dataclass, base classes first.

        :param owner: The class to read. A class that is not a dataclass declares none.
        """
        if not is_dataclass(owner):
            return {}
        return {
            declared.name: declared.metadata[PROPERTY_METADATA_KEY]
            for declared in fields(owner)
            if PROPERTY_METADATA_KEY in declared.metadata
        }


# %% declaring properties


def relation(
    *,
    functional: bool = False,
    minimum_count: int = 0,
    transitive: bool = False,
    inverse_of: Optional[str] = None,
    owns: bool = False,
    exclusive_group: Optional[str] = None,
    iri: Optional[str] = None,
) -> Any:
    """
    Declare a field that links an instance to instances of a declared concept.

    :param owns: Whether the concept is where the related instances are declared.
    """
    return _described_field(
        PropertyMeta(
            kind=PropertyKind.RELATION,
            functional=functional,
            minimum_count=minimum_count,
            transitive=transitive,
            inverse_of=inverse_of,
            owns=owns,
            exclusive_group=exclusive_group,
            iri=iri,
        ),
        default=None,
    )


def attribute(
    *,
    functional: bool = False,
    minimum_count: int = 0,
    default: Any = None,
    marks_assumption: bool = False,
    exclusive_group: Optional[str] = None,
    iri: Optional[str] = None,
) -> Any:
    """
    Declare a field that holds plain values: numbers, strings, booleans or enum members.

    :param default: The value of a property with at most one value. A property with many
        values always starts empty.
    :param marks_assumption: Whether a true value flags the fragment as presupposed.
    :param exclusive_group: Label shared with the alternatives of which exactly one is given.
    :raises InvalidPropertyDeclaration: when a default is given to a property with many
        values.
    """
    if not functional and default is not None:
        raise InvalidPropertyDeclaration(PropertyFault.DEFAULT_FOR_COLLECTION)
    return _described_field(
        PropertyMeta(
            kind=PropertyKind.ATTRIBUTE,
            functional=functional,
            minimum_count=minimum_count,
            marks_assumption=marks_assumption,
            exclusive_group=exclusive_group,
            iri=iri,
        ),
        default=default,
    )


def reference(
    *,
    functional: bool = False,
    minimum_count: int = 0,
    concept_root: Optional[type] = None,
    exclusive_group: Optional[str] = None,
    iri: Optional[str] = None,
) -> Any:
    """
    Declare a field that names terms of another ontology by their IRIs.

    :param concept_root: When given, the field may only name descendants of this local
        concept.
    """
    return _described_field(
        PropertyMeta(
            kind=PropertyKind.REFERENCE,
            functional=functional,
            minimum_count=minimum_count,
            concept_root=concept_root,
            exclusive_group=exclusive_group,
            iri=iri,
        ),
        default=None,
    )


def _described_field(meta: PropertyMeta, default: Any) -> Any:
    """
    Build the dataclass field for a property.

    A property with at most one value starts at its default and one with many values
    starts as its own empty list.
    """
    metadata = {PROPERTY_METADATA_KEY: meta}
    if meta.functional:
        return field(default=default, metadata=metadata)
    return field(default_factory=list, metadata=metadata)
