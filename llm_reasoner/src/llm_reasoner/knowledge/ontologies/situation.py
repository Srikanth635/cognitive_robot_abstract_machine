"""
The situation features a knowledge structure accepts: its functional fields whose values
the ontology uses to define variants of it, such as a pouring structure's material
regime defining the granular pouring structure.

The features are read from the ontology's definitions, not listed, so a structure of
another action brings its own.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from enum import StrEnum

from krrood.entity_query_language.verbalization.rendering.source_documentation import (
    _attribute_docstrings,
)

from llm_reasoner.knowledge.ontologies.terms import (
    DescriptionEntity,
    IntersectionOf,
    PropertyRestriction,
    PropertyTerm,
    Quantifier,
)
from llm_reasoner.knowledge.ontologies.vocabulary import Vocabulary


@dataclass(frozen=True)
class SituationFeature:
    """
    A field of a knowledge structure whose values define variants of the structure.
    """

    field_name: str
    """
    The name of the field.
    """

    property_iri: str
    """
    The IRI of the ontology property the field stands for.
    """

    meaning: str
    """
    The field's docstring.
    """

    values: tuple[StrEnum, ...]
    """
    Every value the field may take.
    """

    variants: dict[StrEnum, type[DescriptionEntity]]
    """
    The variant of the structure each value defines, for the values that define one.
    """


@dataclass
class SituationFeatures:
    """
    Reads the situation features of knowledge structures from the ontology's
    definitions.
    """

    vocabulary: Vocabulary = field(default_factory=Vocabulary)
    """
    The generated vocabulary, which knows the defined classes and the values their
    definitions name.
    """

    def of(self, root_type: type[DescriptionEntity]) -> list[SituationFeature]:
        """
        :param root_type: A knowledge structure, such as the pouring structure.
        :return: Its situation features, in the order of its fields: each functional
            field that the definition of one of its specialisations restricts to a
            value.
        """
        properties = self.functional_properties_of(root_type)
        variants: dict[str, dict[StrEnum, type[DescriptionEntity]]] = {}
        for defined in self.vocabulary.defined_classes:
            if defined is root_type or not issubclass(defined, root_type):
                continue
            for restriction in self.value_restrictions_of(defined):
                name = properties.get(restriction.property_iri)
                if name is None:
                    continue
                value = self.vocabulary.value_of(restriction.value)
                variants.setdefault(name, {})[value] = defined
        docstrings = self.docstrings_of(root_type)
        features = []
        for property_iri, name in properties.items():
            if name not in variants:
                continue
            values = tuple(type(next(iter(variants[name]))))
            features.append(
                SituationFeature(
                    field_name=name,
                    property_iri=property_iri,
                    meaning=docstrings.get(name, ""),
                    values=values,
                    variants={
                        value: variants[name][value]
                        for value in values
                        if value in variants[name]
                    },
                )
            )
        return features

    @staticmethod
    def functional_properties_of(root_type: type[DescriptionEntity]) -> dict[str, str]:
        """
        :return: The field names of a structure's functional fields, by the IRIs of the
            properties they stand for, in the order of the fields.
        """
        functional = {}
        for declared in fields(root_type):
            term = PropertyTerm.of_field(root_type, declared.name)
            if term is not None and term.functional:
                functional[term.iri] = declared.name
        return functional

    @staticmethod
    def value_restrictions_of(
        defined: type[DescriptionEntity],
    ) -> list[PropertyRestriction]:
        """
        :return: The value restrictions a defined class's definition is the
            intersection of.
        """
        definition = defined.term.definition
        if not isinstance(definition, IntersectionOf):
            return []
        return [
            operand
            for operand in definition.operands
            if isinstance(operand, PropertyRestriction)
            and operand.quantifier == Quantifier.VALUE
        ]

    @staticmethod
    def docstrings_of(root_type: type[DescriptionEntity]) -> dict[str, str]:
        """
        :return: The docstrings of a structure's fields, a more specific class's
            docstring replacing a more general one's.
        """
        docstrings: dict[str, str] = {}
        for owner in reversed(root_type.__mro__):
            if isinstance(owner, type) and issubclass(owner, DescriptionEntity):
                docstrings.update(_attribute_docstrings(owner))
        return docstrings
