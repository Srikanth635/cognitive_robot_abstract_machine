"""
What the reader finds in the ontology, in the form the vocabulary modules are written
from.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

from typing_extensions import Optional

from llm_reasoner.knowledge.ontologies.terms import ClassExpression

# %% value kinds


class ValueKind(StrEnum):
    """
    What a field holds, which decides how it is typed and whether krrood manages it.
    """

    DESCRIPTION = "description"
    """
    An instance of a generated vocabulary class, managed by a property descriptor.
    """

    ENUMERATION = "enumeration"
    """
    A member of a generated value-partition enumeration.
    """

    CONCEPT_REFERENCE = "concept_reference"
    """
    A term of an ontology the vocabulary does not compile, such as a CRAM-AICOR action.
    """

    LITERAL = "literal"
    """
    A number or a truth value.
    """


# %% names


@dataclass(frozen=True)
class PythonName:
    """
    Where a generated class or enumeration is defined.
    """

    module: str
    """
    The name of the generated module that defines it, such as ``core``, or an empty
    string for a name every generated module imports directly.
    """

    name: str
    """
    The name of the class or enumeration.
    """

    def reference_from(self, module: str) -> str:
        """
        :return: How code in the given generated module refers to it.
        """
        if self.module in ("", module):
            return self.name
        return f"{self.module}.{self.name}"


# %% value partitions


@dataclass(frozen=True)
class MemberDeclaration:
    """
    One member of a value partition.
    """

    iri: str
    """
    The IRI of the member individual.
    """

    name: str
    """
    The enumeration member name.
    """

    comment: str
    """
    The member's definition.
    """


@dataclass(frozen=True)
class MemberValue:
    """
    The value a member property takes for one member.
    """

    member: str
    """
    The enumeration member name.
    """

    value: PythonName
    """
    The member of another enumeration the property takes, as its enumeration and
    member name.
    """


@dataclass(frozen=True)
class MemberPropertyDeclaration:
    """
    A functional property stated of every member of a value partition, written as a
    property of the enumeration.
    """

    name: str
    """
    The property name.
    """

    property_iri: str
    """
    The IRI of the ontology property.
    """

    comment: str
    """
    The property's definition.
    """

    value_type: PythonName
    """
    The enumeration of the values.
    """

    values: tuple[MemberValue, ...]
    """
    The value of every member, in member order.
    """


@dataclass(frozen=True)
class ValuePartitionDeclaration:
    """
    A class defined by enumerating its members, written as an enumeration.
    """

    iri: str
    """
    The IRI of the value-partition class.
    """

    python_name: PythonName
    """
    Where the enumeration is defined.
    """

    comment: str
    """
    The partition's definition.
    """

    members: tuple[MemberDeclaration, ...]
    """
    The members, in the order the ontology lists them.
    """

    member_properties: tuple[MemberPropertyDeclaration, ...] = ()
    """
    The properties stated of every member.
    """


# %% properties


@dataclass(frozen=True)
class DescriptorDeclaration:
    """
    The krrood property descriptor class written for one ontology property.
    """

    property_iri: str
    """
    The IRI of the ontology property.
    """

    python_name: PythonName
    """
    Where the descriptor class is defined.
    """

    comment: str
    """
    The property's definition.
    """

    super_descriptors: tuple[PythonName, ...] = ()
    """
    The descriptor classes of the microworld super-properties it specialises.
    """

    symmetric: bool = False
    """
    Whether the property is symmetric, so krrood infers each relation in the other
    direction as well.
    """


@dataclass(frozen=True)
class FieldDeclaration:
    """
    One field of a generated class, standing for an ontology property.
    """

    name: str
    """
    The field name.
    """

    property_iri: str
    """
    The IRI of the ontology property.
    """

    comment: str
    """
    The property's definition.
    """

    functional: bool
    """
    Whether the field holds one value rather than a list.
    """

    value_kind: ValueKind
    """
    What the field holds.
    """

    value_type: Optional[PythonName] = None
    """
    The class or enumeration of the values, when they are not concept references or
    literals.
    """

    literal_type: str = ""
    """
    The Python type of a literal value, such as ``float``.
    """

    range_iri: str = ""
    """
    The IRI of the class the values belong to, when they are fragments of a class of
    the microworld; ``value_type`` names a superclass of it when the class is generated
    in a later module.
    """


@dataclass(frozen=True)
class AttachmentDeclaration:
    """
    The attachment of a property descriptor to the field it manages.
    """

    host: PythonName
    """
    The class declaring the field.
    """

    field_name: str
    """
    The name of the field.
    """

    descriptor: PythonName
    """
    The descriptor class managing the field.
    """


# %% classes


@dataclass(frozen=True)
class ClassDeclaration:
    """
    One ontology class, written as a dataclass.
    """

    iri: str
    """
    The IRI of the ontology class.
    """

    python_name: PythonName
    """
    Where the class is defined.
    """

    comment: str
    """
    The class's definition.
    """

    bases: tuple[PythonName, ...]
    """
    The microworld superclasses, or none when the class has no microworld superclass.
    """

    linked_terms: tuple[str, ...]
    """
    The IRIs of the CRAM-AICOR and DUL classes the class specialises.
    """

    evidence_status: str
    """
    The IRI of the class's evidence status.
    """

    claims: tuple[str, ...]
    """
    The IRIs of the source claims the class represents.
    """

    restrictions: tuple[ClassExpression, ...] = ()
    """
    The anonymous class expressions the class is declared a subclass of.
    """

    definition: Optional[ClassExpression] = None
    """
    The class expression the class is declared equivalent to, if any.
    """

    fields: tuple[FieldDeclaration, ...] = ()
    """
    The fields the class introduces; inherited fields are declared on its bases.
    """


# %% modules


@dataclass
class ModuleDeclaration:
    """
    Everything one generated module defines.
    """

    name: str
    """
    The module name, such as ``object_dynamics``.
    """

    source_file: str
    """
    The name of the ontology file the module is generated from.
    """

    comment: str
    """
    The ontology module's description.
    """

    value_partitions: list[ValuePartitionDeclaration] = field(default_factory=list)
    """
    The enumerations, in ontology order.
    """

    classes: list[ClassDeclaration] = field(default_factory=list)
    """
    The dataclasses, every class after its bases.
    """

    descriptors: list[DescriptorDeclaration] = field(default_factory=list)
    """
    The property descriptor classes, every class after its super-properties.
    """

    @property
    def has_symmetric_descriptor(self) -> bool:
        """
        :return: Whether a descriptor of the module stands for a symmetric property.
        """
        return any(descriptor.symmetric for descriptor in self.descriptors)

    attachments: list[AttachmentDeclaration] = field(default_factory=list)
    """
    The descriptors attached to fields, once the module sees both the field's class and
    the descriptor.
    """

    @property
    def referenced_modules(self) -> list[str]:
        """
        :return: The other generated modules whose classes, enumerations or descriptors
            this module refers to, in name order.
        """
        names = [
            *(base for cls in self.classes for base in cls.bases),
            *(
                name
                for cls in self.classes
                for declared in cls.fields
                for name in (declared.value_type,)
                if name is not None
            ),
            *(
                superclass
                for descriptor in self.descriptors
                for superclass in descriptor.super_descriptors
            ),
            *(
                member_property.value_type
                for partition in self.value_partitions
                for member_property in partition.member_properties
            ),
            *(
                name
                for attachment in self.attachments
                for name in (attachment.host, attachment.descriptor)
            ),
        ]
        return sorted({name.module for name in names} - {self.name, ""})
