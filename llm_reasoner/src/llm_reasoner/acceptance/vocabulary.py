"""
Finding the generated Python counterparts of the IRIs a class expression names.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from enum import StrEnum
from functools import cached_property
from operator import attrgetter
from types import ModuleType

from typing_extensions import Any, Optional

from llm_reasoner.ontologies import PredictiveManipulationOntologyFile
from llm_reasoner.ontologies.predictive_manipulation import (
    core,
    gaze,
    object_dynamics,
    pouring,
    prehension,
)
from llm_reasoner.ontologies.terms import (
    ConceptReference,
    DescriptionEntity,
    PropertyTerm,
    local_name,
    python_attribute_name,
)

VOCABULARY_MODULES: dict[PredictiveManipulationOntologyFile, ModuleType] = {
    PredictiveManipulationOntologyFile.CORE: core,
    PredictiveManipulationOntologyFile.OBJECT_DYNAMICS: object_dynamics,
    PredictiveManipulationOntologyFile.PREHENSION: prehension,
    PredictiveManipulationOntologyFile.POURING: pouring,
    PredictiveManipulationOntologyFile.GAZE: gaze,
}
"""
The generated vocabulary module of every ontology module.
"""


@dataclass
class Vocabulary:
    """
    The generated classes, enumerations and fields, looked up by the IRIs they stand
    for.
    """

    modules: dict[PredictiveManipulationOntologyFile, ModuleType] = field(
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

    def definition_named(self, iri: str) -> Optional[type]:
        """
        :return: The generated class or enumeration standing for a microworld class, or
            ``None`` for a class of a linked ontology.
        """
        module = self.module_of(iri)
        if module is None:
            return None
        return vars(module)[local_name(iri)]

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
    def defined_classes(self) -> list[type[DescriptionEntity]]:
        """
        Every generated class whose ontology class is defined by a class expression.
        """
        return [
            definition
            for module in self.modules.values()
            for definition in vars(module).values()
            if isinstance(definition, type)
            and issubclass(definition, DescriptionEntity)
            and definition.__module__ == module.__name__
            and definition.term.definition is not None
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
