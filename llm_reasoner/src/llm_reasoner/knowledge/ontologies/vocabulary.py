"""
Finding the generated Python counterparts of the IRIs a class expression names.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass, field, fields
from enum import StrEnum
from functools import cache, cached_property
from operator import attrgetter
from types import ModuleType

from krrood.class_diagrams.class_diagram import WrappedClass
from typing_extensions import Any, Optional

from llm_reasoner.attribute_docstrings import attribute_docstrings
from llm_reasoner.knowledge.ontologies import (
    KNOWLEDGE_ONTOLOGY_MODULES,
    OntologyModuleFile,
)
from llm_reasoner.knowledge.ontologies.predictive_manipulation import core
from llm_reasoner.knowledge.ontologies.terms import (
    ClassExpression,
    ConceptReference,
    DescriptionEntity,
    IntersectionOf,
    PropertyRestriction,
    PropertyTerm,
    Quantifier,
    local_name,
    python_attribute_name,
)

VOCABULARY_MODULES: dict[OntologyModuleFile, ModuleType] = {
    module_file: module_file.vocabulary_module
    for module_file in KNOWLEDGE_ONTOLOGY_MODULES
}
"""
The generated vocabulary module of every ontology module.
"""


@cache
def declared_value_types(owner: type[DescriptionEntity]) -> dict[str, type]:
    """
    :return: The type each field of a generated class is declared to hold, by field
        name.
    """
    return {
        wrapped.field.name: wrapped.type_endpoint
        for wrapped in WrappedClass(owner).fields
    }


@dataclass
class Vocabulary:
    """
    The generated classes, enumerations and fields, looked up by the IRIs they stand
    for.
    """

    modules: dict[OntologyModuleFile, ModuleType] = field(
        default_factory=lambda: dict(VOCABULARY_MODULES)
    )
    """
    The generated module of every ontology module.
    """

    def module_of(self, iri: str) -> Optional[ModuleType]:
        """
        :return: The generated module of the ontology module declaring the IRI, or
            ``None`` for an IRI of a linked ontology.
        """
        for module_file, module in self.modules.items():
            if iri.startswith(module_file.namespace):
                return module
        return None

    def module_file_of(self, python_type: type) -> Optional[OntologyModuleFile]:
        """
        :return: The ontology module a generated class or enumeration stands for a term
            of, ``None`` for a type generated from no module.
        """
        return next(
            (
                module_file
                for module_file, module in self.modules.items()
                if python_type.__module__ == module.__name__
            ),
            None,
        )

    def definition_named(self, iri: str) -> Optional[type]:
        """
        :return: The generated class or enumeration standing for a microworld class, or
            ``None`` for a class of a linked ontology.
        """
        module = self.module_of(iri)
        if module is None:
            return None
        return vars(module)[local_name(iri)]

    def value_type_of(
        self, owner: type[DescriptionEntity], name: str, annotated: type
    ) -> type:
        """
        :return: The class a field's values belong to: the range its ontology property
            records, which is more specific than the field's annotation when the class
            is generated in a later module, or else the annotated type.
        """
        term = PropertyTerm.of_field(owner, name)
        if term is None or not term.range:
            return annotated
        return self.definition_named(term.range)

    def held_by_field(
        self, owner: type[DescriptionEntity], name: str, value: Any
    ) -> Any:
        """
        :return: The value as the field holds it: an enumeration member becomes the
            member of the field's own enumeration standing for the same individual, since
            several enumerations may list one individual; any other value is unchanged.
        """
        enumeration = declared_value_types(owner).get(name)
        if not (
            isinstance(value, StrEnum)
            and isinstance(enumeration, type)
            and issubclass(enumeration, StrEnum)
        ):
            return value
        return next(
            (member for member in enumeration if member.value == value.value), value
        )

    def component_fields(
        self, structure_type: type[DescriptionEntity]
    ) -> dict[str, type[DescriptionEntity]]:
        """
        :return: The fields through which a structure declares its components, each with
            the class of component it declares; the field holding every component,
            which the others specialise, is left out.
        """
        return {
            declared.name: self.definition_named(
                PropertyTerm.of_field(structure_type, declared.name).range
            )
            for declared in fields(structure_type)
            for descriptor in [
                inspect.getattr_static(structure_type, declared.name, None)
            ]
            if isinstance(descriptor, core.DefinesComponent)
            and type(descriptor) is not core.DefinesComponent
        }

    def entailed_values(self, python_type: type[DescriptionEntity]) -> dict[str, Any]:
        """
        :return: The values every instance of a class has because its classes say so,
            through value restrictions they state or are defined by, by field name.
            A more specific class decides a field before a more general one.
        """
        names = {declared.name for declared in fields(python_type)}
        entailed: dict[str, Any] = {}
        for owner in python_type.__mro__:
            if "term" not in vars(owner):
                continue
            definition = owner.term.definition
            expressions: list[ClassExpression] = list(owner.term.restrictions)
            if isinstance(definition, IntersectionOf):
                expressions.extend(definition.operands)
            elif definition is not None:
                expressions.append(definition)
            for expression in expressions:
                if not (
                    isinstance(expression, PropertyRestriction)
                    and expression.quantifier == Quantifier.VALUE
                ):
                    continue
                name = python_attribute_name(expression.property_iri)
                if name in names:
                    entailed.setdefault(name, self.value_of(expression.value))
        return entailed

    def fill_entailed_values(self, instance: DescriptionEntity) -> None:
        """
        Give a fragment the enumeration members, concepts and literals its classes
        entail, where nothing else is stated; values that would be fragments are left
        to be stated.
        """
        wrapped_fields = {
            wrapped.field.name: wrapped
            for wrapped in WrappedClass(type(instance)).fields
        }
        for name, entailed in self.entailed_values(type(instance)).items():
            value = self.held_by_field(type(instance), name, entailed)
            value_type = self.value_type_of(
                type(instance), name, wrapped_fields[name].type_endpoint
            )
            if isinstance(value_type, type) and issubclass(
                value_type, DescriptionEntity
            ):
                continue
            stated = attrgetter(name)(instance)
            if PropertyTerm.of_field(type(instance), name).functional:
                if stated is None:
                    setattr(instance, name, value)
            elif value not in stated:
                setattr(instance, name, [*stated, value])

    @cached_property
    def members(self) -> dict[str, StrEnum]:
        """
        Every generated enumeration member, by the IRI of the individual it stands for.
        """
        return {
            member.value: member
            for module in self.modules.values()
            for definition in vars(module).values()
            if isinstance(definition, type) and issubclass(definition, StrEnum)
            for member in definition
        }

    @cached_property
    def classes(self) -> list[type[DescriptionEntity]]:
        """
        Every generated class.
        """
        return [
            definition
            for module in self.modules.values()
            for definition in vars(module).values()
            if isinstance(definition, type)
            and issubclass(definition, DescriptionEntity)
            and definition.__module__ == module.__name__
        ]

    def class_named(self, name: str) -> Optional[type[DescriptionEntity]]:
        """
        :param name: The name of a generated class, such as ``LiftPhase``.
        :return: The generated class of that name, ``None`` when there is none.
        """
        return next(
            (generated for generated in self.classes if generated.__name__ == name),
            None,
        )

    @staticmethod
    def field_meanings(python_type: type[DescriptionEntity]) -> dict[str, str]:
        """
        :return: The docstrings of a generated class's fields, by field name, a more
            specific class's docstring replacing a more general one's.
        """
        meanings: dict[str, str] = {}
        for owner in reversed(python_type.__mro__):
            if isinstance(owner, type) and issubclass(owner, DescriptionEntity):
                meanings.update(attribute_docstrings(owner))
        return meanings

    @cached_property
    def defined_classes(self) -> list[type[DescriptionEntity]]:
        """
        Every generated class whose ontology class is defined by a class expression.
        """
        return [
            definition
            for definition in self.classes
            if definition.term.definition is not None
        ]

    def value_of(self, value: Any) -> Any:
        """
        :return: The Python value a value restriction names: an enumeration member, a
            concept reference for a linked term, or the literal itself.
        """
        if not isinstance(value, str):
            return value
        if value in self.members:
            return self.members[value]
        return ConceptReference(iri=value)

    # %% property values

    def is_functional(self, entity: Any, property_iri: str) -> Optional[bool]:
        """
        :return: Whether the entity's attribute for the property holds one value, or
            ``None`` when the entity has no attribute for the property.
        """
        name = python_attribute_name(property_iri)
        if isinstance(entity, StrEnum):
            return True if isinstance(vars(type(entity)).get(name), property) else None
        if not isinstance(entity, DescriptionEntity):
            return None
        for declared in fields(entity):
            term = PropertyTerm.of_field(type(entity), declared.name)
            if term is not None and term.iri == property_iri:
                return term.functional
        return None

    def values_of(self, entity: Any, property_iri: str) -> list[Any]:
        """
        :return: The values the entity has for the property, none when it has no
            attribute for the property.
        """
        functional = self.is_functional(entity, property_iri)
        if functional is None:
            return []
        value = attrgetter(python_attribute_name(property_iri))(entity)
        if functional:
            return [] if value is None else [value]
        return list(value)
