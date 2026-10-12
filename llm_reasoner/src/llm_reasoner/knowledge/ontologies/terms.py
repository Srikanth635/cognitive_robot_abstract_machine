"""
What a generated vocabulary class records about the ontology term it stands for.

The vocabulary modules are generated from the ontology files. A class carries its
ontology class as a :class:`ClassTerm`, and a field carries its ontology property as a
:class:`PropertyTerm`, so every value a language model fills in can be written back as
the ontology term it instantiates.
"""

from __future__ import annotations

import inspect
import re
from abc import ABC
from dataclasses import dataclass
from enum import StrEnum

from krrood.ontomatic.property_descriptor.property_descriptor import PropertyDescriptor
from krrood.patterns.field_metadata import FieldMetadata
from krrood.symbol_graph.symbol_graph import Symbol
from typing_extensions import ClassVar, Optional, Union

# %% naming

WORD_BOUNDARY = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")
"""
The position between a lower-case letter or digit and the upper-case letter after it.
"""


def local_name(iri: str) -> str:
    """
    :return: The part of an IRI after its namespace.
    """
    return re.split(r"[#/]", iri)[-1]


def python_attribute_name(property_iri: str) -> str:
    """
    :return: The name of the field or enumeration property standing for an ontology
        property, its local name in lower case with underscores between words.
    """
    return WORD_BOUNDARY.sub("_", local_name(property_iri)).lower()


# %% class expressions


@dataclass(frozen=True)
class ClassExpression(ABC):
    """
    An OWL class expression, kept as data so it can be checked against generated
    instances.
    """


@dataclass(frozen=True)
class NamedClass(ClassExpression):
    """
    A class named by its IRI, in the microworld or in a linked ontology.
    """

    iri: str
    """
    The IRI of the class.
    """


@dataclass(frozen=True)
class EnumeratedClass(ClassExpression):
    """
    The class whose members are exactly the listed individuals.
    """

    members: tuple[str, ...]
    """
    The IRIs of the members.
    """


@dataclass(frozen=True)
class IntersectionOf(ClassExpression):
    """
    The class of what belongs to every operand.
    """

    operands: tuple[ClassExpression, ...]
    """
    The intersected class expressions.
    """


@dataclass(frozen=True)
class UnionOf(ClassExpression):
    """
    The class of what belongs to at least one operand.
    """

    operands: tuple[ClassExpression, ...]
    """
    The united class expressions.
    """


@dataclass(frozen=True)
class ComplementOf(ClassExpression):
    """
    The class of what does not belong to the operand.
    """

    operand: ClassExpression
    """
    The complemented class expression.
    """


class Quantifier(StrEnum):
    """
    How a property restriction constrains the values of its property.
    """

    SOME = "some"
    """
    At least one value belongs to the filler.
    """

    ONLY = "only"
    """
    Every value belongs to the filler.
    """

    VALUE = "value"
    """
    The given value is among the values.
    """

    MINIMUM = "minimum"
    """
    At least the given number of values belong to the filler.
    """

    MAXIMUM = "maximum"
    """
    At most the given number of values belong to the filler.
    """

    EXACTLY = "exactly"
    """
    Exactly the given number of values belong to the filler.
    """


@dataclass(frozen=True)
class PropertyRestriction(ClassExpression):
    """
    The class of what has values of one property that meet a condition.
    """

    property_iri: str
    """
    The IRI of the restricted property.
    """

    quantifier: Quantifier
    """
    How the values are constrained.
    """

    filler: Optional[ClassExpression] = None
    """
    The class the values must belong to, when the restriction names one.
    """

    value: Optional[Union[str, bool, float, int]] = None
    """
    The required value of a value restriction: an IRI or a literal.
    """

    cardinality: Optional[int] = None
    """
    The number of values of a cardinality restriction.
    """


# %% ontology terms


@dataclass(frozen=True)
class ClassTerm:
    """
    The ontology class a generated class stands for.
    """

    iri: str
    """
    The IRI of the ontology class.
    """

    linked_terms: tuple[str, ...] = ()
    """
    The IRIs of the CRAM-AICOR and DUL classes the ontology class specialises.
    """

    evidence_status: str = ""
    """
    The IRI of the evidence status the ontology states for the class.
    """

    claims: tuple[str, ...] = ()
    """
    The IRIs of the source claims the class represents.
    """

    restrictions: tuple[ClassExpression, ...] = ()
    """
    The class expressions every instance must belong to, besides its named
    superclasses.
    """

    definition: Optional[ClassExpression] = None
    """
    The class expression the class is equivalent to, when the ontology defines it.
    """


@dataclass
class PropertyTerm(FieldMetadata):
    """
    The ontology property a generated field stands for.
    """

    iri: str = ""
    """
    The IRI of the ontology property.
    """

    functional: bool = False
    """
    Whether the property takes at most one value, so the field holds one value.
    """

    range: str = ""
    """
    The IRI of the vocabulary class the property's values belong to, when they are
    fragments. The field's annotation names a superclass of it when the class is
    generated in a later module than the field.
    """


@dataclass(frozen=True)
class ConceptReference:
    """
    A term of an ontology the vocabulary does not compile, such as ``aicor:Pouring``.
    """

    iri: str
    """
    The IRI of the referenced term.
    """


# %% description entities


@dataclass(eq=False)
class DescriptionEntity(Symbol):
    """
    Base of every generated vocabulary class: a description-level entity, such as a
    phase concept or an event type, never an occurrence.
    """

    term: ClassVar[ClassTerm]
    """
    The ontology class the generated class stands for.
    """

    label: str = ""
    """
    A human-readable name for the entity.
    """


def is_inferred(python_type: type[DescriptionEntity], name: str) -> bool:
    """
    :return: Whether krrood fills the field from its sub-properties, so a model never
        states it: its descriptor stands for a property other properties specialise.
    """
    descriptor = inspect.getattr_static(python_type, name, None)
    return isinstance(descriptor, PropertyDescriptor) and bool(
        type(descriptor).__subclasses__()
    )
