"""
Deciding which entities belong to an OWL class expression, reading the hypothesis as a
closed world: a value that is absent counts as false.

Entities reached through a property are found with EQL queries. EQL answers positive
questions, and drops an entity whose collection is empty instead of reporting that it
has no matching value, so a negative condition is computed as the entities a positive
query did not return: negation as failure over the fragments of one hypothesis.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from enum import StrEnum
from operator import attrgetter

from krrood.entity_query_language.factories import (
    an,
    entity,
    flat_variable,
    in_,
    variable,
)
from rdflib import XSD
from typing_extensions import Any, Iterable, Sequence

from llm_reasoner.knowledge.ontologies.vocabulary import Vocabulary
from llm_reasoner.exceptions import LLMReasonerError
from llm_reasoner.knowledge.ontologies.terms import (
    ClassExpression,
    ComplementOf,
    DescriptionEntity,
    EnumeratedClass,
    IntersectionOf,
    NamedClass,
    PropertyRestriction,
    Quantifier,
    UnionOf,
    python_attribute_name,
)

# %% exceptions


@dataclass
class UnsupportedClassExpression(LLMReasonerError):
    """
    A class expression of a kind the closed-world reading does not cover.
    """

    expression: ClassExpression
    """
    The class expression.
    """

    def __post_init__(self):
        super().__init__(f"cannot evaluate class expression {self.expression}")


# %% evaluation


@dataclass
class ClosedWorldMembership:
    """
    Decides which of some entities belong to a class expression.
    """

    vocabulary: Vocabulary = field(default_factory=Vocabulary)
    """
    The generated vocabulary the expressions' IRIs are looked up in.
    """

    def members(
        self, expression: ClassExpression, entities: Sequence[Any]
    ) -> list[Any]:
        """
        :return: The given entities that belong to the class expression, in their order.
        """
        if isinstance(expression, NamedClass):
            return self.instances(expression, entities)
        if isinstance(expression, EnumeratedClass):
            return [
                entity_
                for entity_ in entities
                if isinstance(entity_, StrEnum) and entity_.value in expression.members
            ]
        if isinstance(expression, IntersectionOf):
            remaining = list(entities)
            for operand in expression.operands:
                remaining = self.members(operand, remaining)
            return remaining
        if isinstance(expression, UnionOf):
            belonging = {
                id(member)
                for operand in expression.operands
                for member in self.members(operand, entities)
            }
            return [entity_ for entity_ in entities if id(entity_) in belonging]
        if isinstance(expression, ComplementOf):
            excluded = {
                id(member) for member in self.members(expression.operand, entities)
            }
            return [entity_ for entity_ in entities if id(entity_) not in excluded]
        if isinstance(expression, PropertyRestriction):
            return self.restricted(expression, entities)
        raise UnsupportedClassExpression(expression)

    def instances(self, expression: NamedClass, entities: Sequence[Any]) -> list[Any]:
        """
        :return: The given entities that are instances of a named class, or that meet
            its definition when the class is defined. Every entity belongs to a class of
            a linked ontology, which only the open-world check can judge, and every
            present value belongs to a datatype.
        """
        if expression.iri.startswith(str(XSD)):
            return [entity_ for entity_ in entities if entity_ is not None]
        definition = self.vocabulary.definition_named(expression.iri)
        if definition is None:
            return list(entities)
        typed = [entity_ for entity_ in entities if isinstance(entity_, definition)]
        if (
            not issubclass(definition, DescriptionEntity)
            or definition.term.definition is None
        ):
            return typed
        typed_ids = {id(entity_) for entity_ in typed}
        classified = {
            id(member)
            for member in self.members(
                definition.term.definition,
                [entity_ for entity_ in entities if id(entity_) not in typed_ids],
            )
        }
        return [
            entity_
            for entity_ in entities
            if id(entity_) in typed_ids or id(entity_) in classified
        ]

    # %% property restrictions

    def restricted(
        self, restriction: PropertyRestriction, entities: Sequence[Any]
    ) -> list[Any]:
        """
        :return: The given entities meeting a property restriction.
        """
        values = [
            value
            for entity_ in entities
            for value in self.vocabulary.values_of(entity_, restriction.property_iri)
        ]
        if restriction.quantifier == Quantifier.VALUE:
            return self.having_any(
                restriction.property_iri,
                entities,
                [self.vocabulary.value_of(restriction.value)],
            )
        qualifying = (
            values
            if restriction.filler is None
            else self.members(restriction.filler, values)
        )
        if restriction.quantifier == Quantifier.SOME:
            return self.having_any(restriction.property_iri, entities, qualifying)
        if restriction.quantifier == Quantifier.ONLY:
            qualifying_ids = {id(value) for value in qualifying}
            excluded = self.having_any(
                restriction.property_iri,
                entities,
                [value for value in values if id(value) not in qualifying_ids],
            )
            excluded_ids = {id(entity_) for entity_ in excluded}
            return [entity_ for entity_ in entities if id(entity_) not in excluded_ids]
        return self.counted(restriction, entities, qualifying)

    def counted(
        self,
        restriction: PropertyRestriction,
        entities: Sequence[Any],
        qualifying: Sequence[Any],
    ) -> list[Any]:
        """
        :return: The given entities whose number of qualifying values meets a
            cardinality restriction.
        """
        qualifying_ids = {id(value) for value in qualifying}
        accepted = []
        for entity_ in entities:
            count = sum(
                id(value) in qualifying_ids
                for value in self.vocabulary.values_of(
                    entity_, restriction.property_iri
                )
            )
            if (
                (
                    restriction.quantifier == Quantifier.MINIMUM
                    and count >= restriction.cardinality
                )
                or (
                    restriction.quantifier == Quantifier.MAXIMUM
                    and count <= restriction.cardinality
                )
                or (
                    restriction.quantifier == Quantifier.EXACTLY
                    and count == restriction.cardinality
                )
            ):
                accepted.append(entity_)
        return accepted

    def having_any(
        self, property_iri: str, entities: Sequence[Any], accepted_values: Sequence[Any]
    ) -> list[Any]:
        """
        :return: The given entities having at least one of the accepted values for the
            property, found with one EQL query per entity type.
        """
        if not accepted_values:
            return []
        name = python_attribute_name(property_iri)
        found_ids: set[int] = set()
        for entity_type, functional, group in self.groups(property_iri, entities):
            subject = variable(entity_type, domain=group)
            value = attrgetter(name)(subject)
            candidate = value if functional else flat_variable(value)
            query = an(entity(subject).where(in_(candidate, list(accepted_values))))
            found_ids |= {id(found) for found in query.evaluate()}
        return [entity_ for entity_ in entities if id(entity_) in found_ids]

    def groups(
        self, property_iri: str, entities: Iterable[Any]
    ) -> list[tuple[type, bool, list[Any]]]:
        """
        :return: The entities that have an attribute for the property, grouped by type
            and by whether the attribute holds one value.
        """
        grouped: dict[tuple[type, bool], list[Any]] = defaultdict(list)
        for entity_ in entities:
            functional = self.vocabulary.is_functional(entity_, property_iri)
            if functional is not None:
                grouped[(type(entity_), functional)].append(entity_)
        return [
            (entity_type, functional, group)
            for (entity_type, functional), group in grouped.items()
        ]
