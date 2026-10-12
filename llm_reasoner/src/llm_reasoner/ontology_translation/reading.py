"""
Reading the ontology modules into declarations of the Python vocabulary.

An OWL property belongs to no class, while a dataclass field must belong to one. A
property's field is placed on its ``rdfs:domain``. A property without a domain is placed
on the most specific microworld class that subsumes every class restricting it, or, when
no single class does, on each most general restricting class. CRAM-AICOR properties that
microworld classes restrict with microworld fillers are placed the same way, typed by
those fillers.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from functools import cached_property

from rdflib import OWL, RDF, RDFS, XSD, BNode, Graph, URIRef
from rdflib.term import Node
from typing_extensions import Iterable, Optional

from llm_reasoner.exceptions import LLMReasonerError
from llm_reasoner.knowledge.ontologies import (
    PREDICTIVE_MANIPULATION_NAMESPACE_ROOT,
    PredictiveManipulationOntologyFile,
)
from llm_reasoner.knowledge.ontologies.terms import (
    ClassExpression,
    ComplementOf,
    EnumeratedClass,
    IntersectionOf,
    NamedClass,
    PropertyRestriction,
    Quantifier,
    UnionOf,
    python_attribute_name,
)
from llm_reasoner.ontology_translation.declarations import (
    AttachmentDeclaration,
    ClassDeclaration,
    DescriptorDeclaration,
    FieldDeclaration,
    MemberDeclaration,
    MemberPropertyDeclaration,
    MemberValue,
    ModuleDeclaration,
    PythonName,
    ValueKind,
    ValuePartitionDeclaration,
)

# %% constants

DESCRIPTION_ENTITY = PythonName(module="", name="DescriptionEntity")
"""
The base every generated class shares, imported directly by every generated module.
"""

LITERAL_TYPES: dict[URIRef, str] = {
    XSD.double: "float",
    XSD.boolean: "bool",
    XSD.positiveInteger: "int",
    XSD.nonNegativeInteger: "int",
    XSD.string: "str",
}
"""
The Python type of each XML Schema datatype the ontology uses.
"""

EXTERNAL_PROPERTY_COMMENT = "The linked ontology property {iri}."
"""
The definition given to a field standing for a property of a linked ontology.
"""

EVIDENCE_STATUS = URIRef(PREDICTIVE_MANIPULATION_NAMESPACE_ROOT + "core#evidenceStatus")
"""
The annotation property stating a class's evidence status.
"""

REPRESENTS_CLAIM = URIRef(
    PREDICTIVE_MANIPULATION_NAMESPACE_ROOT + "core#representsClaim"
)
"""
The annotation property naming the source claims a class represents.
"""

QUANTIFIED_RESTRICTIONS: dict[URIRef, Quantifier] = {
    OWL.someValuesFrom: Quantifier.SOME,
    OWL.allValuesFrom: Quantifier.ONLY,
}
"""
The restriction predicates that name a filler class, with their quantifier.
"""

CARDINALITY_RESTRICTIONS: dict[URIRef, Quantifier] = {
    OWL.minCardinality: Quantifier.MINIMUM,
    OWL.minQualifiedCardinality: Quantifier.MINIMUM,
    OWL.maxCardinality: Quantifier.MAXIMUM,
    OWL.maxQualifiedCardinality: Quantifier.MAXIMUM,
    OWL.cardinality: Quantifier.EXACTLY,
    OWL.qualifiedCardinality: Quantifier.EXACTLY,
}
"""
The restriction predicates that count values, with their quantifier.
"""

# %% naming


def upper_snake_case(name: str) -> str:
    """
    :return: The name written in upper case with underscores between words.
    """
    return python_attribute_name(name).upper()


def capitalised(name: str) -> str:
    """
    :return: The name with its first letter in upper case.
    """
    return name[0].upper() + name[1:]


def local_name(term: URIRef) -> str:
    """
    :return: The part of an IRI after its namespace.
    """
    return re.split(r"[#/]", str(term))[-1]


# %% exceptions


@dataclass
class UnsupportedRestriction(LLMReasonerError):
    """
    The ontology states a property restriction the translation cannot represent.
    """

    node: str
    """
    The RDF node of the restriction.
    """

    def __post_init__(self):
        super().__init__(f"unsupported property restriction: {self.node}")


@dataclass
class ConflictingPythonNames(LLMReasonerError):
    """
    Several ontology terms would be written as Python definitions of the same name.
    """

    conflicts: dict[str, list[str]]
    """
    Every shared name, with the IRIs of the terms sharing it.
    """

    def __post_init__(self):
        super().__init__(
            "ontology terms would share Python names; rename them in the ontology: "
            + "; ".join(
                f"{name}: {', '.join(iris)}" for name, iris in self.conflicts.items()
            )
        )


@dataclass
class FieldTypeOutOfReach(LLMReasonerError):
    """
    A field would hold values of an enumeration declared in a module after the one
    declaring the field's class, which that module cannot import.
    """

    property_iri: str
    """
    The IRI of the property the field stands for.
    """

    def __post_init__(self):
        super().__init__(
            f"{self.property_iri} takes values from a later module than its domain; "
            "move the property or its range in the ontology"
        )


@dataclass
class MissingPropertyRange(LLMReasonerError):
    """
    A property has no range, so its values would be read as terms of a linked ontology,
    while a restriction shows they are microworld classes.
    """

    property_iri: str
    """
    The IRI of the property.
    """

    def __post_init__(self):
        super().__init__(
            f"{self.property_iri} has no rdfs:range although a restriction names a "
            "microworld class as its filler; declare the range in the ontology"
        )


# %% reader


@dataclass
class OntologyReader:
    """
    Reads the predictive manipulation modules into one declaration per generated module.
    """

    module_graphs: dict[PredictiveManipulationOntologyFile, Graph]
    """
    Every module file parsed on its own, in dependency order.
    """

    graph: Graph = field(init=False)
    """
    Every module in one graph.
    """

    def __post_init__(self):
        self.graph = Graph()
        for module_graph in self.module_graphs.values():
            self.graph += module_graph

    @classmethod
    def for_predictive_manipulation(cls) -> OntologyReader:
        """
        :return: A reader over every predictive manipulation module file.
        """
        return cls(
            {
                module_file: Graph().parse(module_file.path, format="turtle")
                for module_file in PredictiveManipulationOntologyFile
            }
        )

    # %% terms

    def is_microworld_term(self, node: Node) -> bool:
        """
        :return: Whether the node is a term the microworld declares.
        """
        return isinstance(node, URIRef) and str(node).startswith(
            PREDICTIVE_MANIPULATION_NAMESPACE_ROOT
        )

    def module_of(self, term: URIRef) -> PredictiveManipulationOntologyFile:
        """
        :return: The module whose namespace the microworld term is declared in.
        """
        (module_file,) = [
            module_file
            for module_file in PredictiveManipulationOntologyFile
            if str(term).startswith(module_file.namespace)
        ]
        return module_file

    def python_name(self, term: URIRef) -> PythonName:
        """
        :return: Where the class or enumeration for a microworld class is defined.
        """
        return PythonName(
            module=self.module_of(term).vocabulary_module_name, name=local_name(term)
        )

    def comment(self, term: URIRef) -> str:
        """
        :return: The term's definition, or an empty string when it has none.
        """
        return str(self.graph.value(term, RDFS.comment) or "")

    @cached_property
    def value_partitions(self) -> dict[URIRef, list[URIRef]]:
        """
        Every microworld class defined by enumerating its members, with its members.
        """
        return {
            cls: list(self.graph.items(self.graph.value(equivalent, OWL.oneOf)))
            for cls, equivalent in self.graph.subject_objects(OWL.equivalentClass)
            if self.is_microworld_term(cls)
            and self.graph.value(equivalent, OWL.oneOf) is not None
        }

    @cached_property
    def description_classes(self) -> list[URIRef]:
        """
        Every microworld class that is not a value partition, in IRI order.
        """
        return sorted(
            cls
            for cls in self.graph.subjects(RDF.type, OWL.Class)
            if self.is_microworld_term(cls) and cls not in self.value_partitions
        )

    # %% class hierarchy

    def named_superclasses(self, cls: URIRef) -> set[URIRef]:
        """
        :return: The named classes the class is declared a subclass of, directly or as
            an operand of the intersection it is defined by.
        """
        superclasses = {
            superclass
            for superclass in self.graph.objects(cls, RDFS.subClassOf)
            if isinstance(superclass, URIRef)
        }
        for equivalent in self.graph.objects(cls, OWL.equivalentClass):
            operands = self.graph.value(equivalent, OWL.intersectionOf)
            if operands is not None:
                superclasses |= {
                    operand
                    for operand in self.graph.items(operands)
                    if isinstance(operand, URIRef)
                }
        return superclasses

    @cached_property
    def direct_bases(self) -> dict[URIRef, set[URIRef]]:
        """
        Every description class's microworld superclasses, with redundant ones removed.
        """
        declared = {
            cls: {
                superclass
                for superclass in self.named_superclasses(cls)
                if superclass in self.description_classes
            }
            for cls in self.description_classes
        }
        ancestors: dict[URIRef, set[URIRef]] = {}

        def ancestors_of(cls: URIRef) -> set[URIRef]:
            if cls not in ancestors:
                ancestors[cls] = set()
                for base in declared[cls]:
                    ancestors[cls] |= {base} | ancestors_of(base)
            return ancestors[cls]

        return {
            cls: {
                base
                for base in bases
                if not any(base in ancestors_of(other) for other in bases)
            }
            for cls, bases in declared.items()
        }

    @cached_property
    def ancestors(self) -> dict[URIRef, set[URIRef]]:
        """
        Every description class's microworld superclasses, including itself.
        """
        result: dict[URIRef, set[URIRef]] = {}

        def ancestors_of(cls: URIRef) -> set[URIRef]:
            if cls not in result:
                result[cls] = {cls}
                for base in self.direct_bases[cls]:
                    result[cls] |= ancestors_of(base)
            return result[cls]

        for cls in self.description_classes:
            ancestors_of(cls)
        return result

    def most_general(self, classes: Iterable[URIRef]) -> list[URIRef]:
        """
        :return: The classes that no other given class subsumes, in IRI order.
        """
        classes = set(classes)
        return sorted(
            cls
            for cls in classes
            if not any(
                other != cls and other in self.ancestors[cls] for other in classes
            )
        )

    def lowest_common_ancestor(self, classes: Iterable[URIRef]) -> Optional[URIRef]:
        """
        :return: The single most specific class subsuming every given class, or
            ``None`` when there is no such single class.
        """
        classes = list(classes)
        if not classes:
            return None
        common = set.intersection(*(self.ancestors[cls] for cls in classes))
        most_specific = [
            cls
            for cls in common
            if not any(
                other != cls and cls in self.ancestors[other] for other in common
            )
        ]
        return most_specific[0] if len(most_specific) == 1 else None

    # %% properties

    @cached_property
    def restrictions(self) -> list[tuple[URIRef, URIRef, Optional[URIRef]]]:
        """
        Every restriction a description class is directly declared a subclass of, as the
        class, the restricted property and the named filler, if any.
        """
        result = []
        for cls in [*self.description_classes, *sorted(self.value_partitions)]:
            for restriction in self.graph.objects(cls, RDFS.subClassOf):
                prop = self.graph.value(restriction, OWL.onProperty)
                if prop is None:
                    continue
                filler = next(
                    (
                        value
                        for predicate in (OWL.someValuesFrom, OWL.allValuesFrom)
                        for value in self.graph.objects(restriction, predicate)
                        if isinstance(value, URIRef)
                    ),
                    None,
                )
                result.append((cls, prop, filler))
        return result

    @cached_property
    def microworld_properties(self) -> list[URIRef]:
        """
        Every object and datatype property the microworld declares, in IRI order.
        """
        return sorted(
            prop
            for property_type in (OWL.ObjectProperty, OWL.DatatypeProperty)
            for prop in self.graph.subjects(RDF.type, property_type)
            if self.is_microworld_term(prop)
        )

    @cached_property
    def linked_properties(self) -> list[URIRef]:
        """
        Every property of a linked ontology that a description class restricts with a
        microworld filler, in IRI order.
        """
        return sorted(
            {
                prop
                for _, prop, filler in self.restrictions
                if not self.is_microworld_term(prop)
                and filler in self.description_classes
            }
        )

    @cached_property
    def field_hosts(self) -> dict[URIRef, list[URIRef]]:
        """
        The classes each described property's field is declared on.
        """
        return {prop: self.hosts(prop) for prop in self.described_properties}

    def hosts(self, prop: URIRef) -> list[URIRef]:
        """
        :return: The classes a property's field is declared on.
        """
        domain = self.graph.value(prop, RDFS.domain)
        if domain is not None:
            return [domain]
        restricting = {
            cls
            for cls, restricted, filler in self.restrictions
            if restricted == prop
            and (self.is_microworld_term(prop) or filler in self.description_classes)
        }
        if restricting & set(self.value_partitions):
            return sorted(restricting)
        common = self.lowest_common_ancestor(restricting)
        return [common] if common is not None else self.most_general(restricting)

    def sub_properties(self, prop: URIRef) -> list[URIRef]:
        """
        :return: The microworld properties declared sub-properties of the property.
        """
        return sorted(self.graph.subjects(RDFS.subPropertyOf, prop))

    def value_of(self, prop: URIRef) -> tuple[ValueKind, Optional[PythonName], str]:
        """
        :return: What the property's field holds: its kind, its class or enumeration,
            and its literal type.
        """
        if not self.is_microworld_term(prop):
            fillers = {
                filler
                for _, restricted, filler in self.restrictions
                if restricted == prop and filler in self.description_classes
            }
            common = self.lowest_common_ancestor(fillers)
            return (
                ValueKind.DESCRIPTION,
                (
                    self.python_name(common)
                    if common is not None
                    else DESCRIPTION_ENTITY
                ),
                "",
            )
        range_ = self.graph.value(prop, RDFS.range)
        if range_ in LITERAL_TYPES:
            return ValueKind.LITERAL, None, LITERAL_TYPES[range_]
        if range_ in self.value_partitions:
            return ValueKind.ENUMERATION, self.python_name(range_), ""
        if range_ in self.description_classes:
            return ValueKind.DESCRIPTION, self.python_name(range_), ""
        if self.sub_properties(prop) and all(
            self.value_of(sub)[0] == ValueKind.DESCRIPTION
            for sub in self.sub_properties(prop)
        ):
            return ValueKind.DESCRIPTION, DESCRIPTION_ENTITY, ""
        return ValueKind.CONCEPT_REFERENCE, None, ""

    def is_functional(self, prop: URIRef) -> bool:
        """
        :return: Whether the microworld declares the property functional.
        """
        return (prop, RDF.type, OWL.FunctionalProperty) in self.graph

    def descriptor_module(self, prop: URIRef) -> PredictiveManipulationOntologyFile:
        """
        :return: The module defining the property's descriptor: its own module, or for
            a linked property the earliest module declaring one of its fields.
        """
        if self.is_microworld_term(prop):
            return self.module_of(prop)
        modules = list(PredictiveManipulationOntologyFile)
        return min(
            (self.module_of(host) for host in self.field_hosts[prop]),
            key=modules.index,
        )

    def descriptor_name(self, prop: URIRef) -> PythonName:
        """
        :return: Where the property's descriptor class is defined.
        """
        return PythonName(
            module=self.descriptor_module(prop).vocabulary_module_name,
            name=capitalised(local_name(prop)),
        )

    @cached_property
    def described_properties(self) -> list[URIRef]:
        """
        Every property written as a field, microworld properties first.
        """
        return [
            prop
            for prop in self.microworld_properties + self.linked_properties
            if prop not in self.member_properties
        ]

    @cached_property
    def member_properties(self) -> list[URIRef]:
        """
        Every microworld property stated of the members of value partitions.
        """
        return [
            prop
            for prop in self.microworld_properties
            if set(self.hosts(prop)) <= set(self.value_partitions) and self.hosts(prop)
        ]

    def visible_type(self, value_type: PythonName, host: URIRef) -> PythonName:
        """
        :return: The given class, or when it is defined in a module after the host's,
            its most specific superclass that the host's module sees.
        """
        modules = [
            module_file.vocabulary_module_name
            for module_file in PredictiveManipulationOntologyFile
        ]
        host_position = modules.index(self.module_of(host).vocabulary_module_name)
        if value_type.module in ("", *modules[: host_position + 1]):
            return value_type
        value_class = self.class_named(value_type)
        visible = [
            ancestor
            for ancestor in self.ancestors[value_class]
            if modules.index(self.module_of(ancestor).vocabulary_module_name)
            <= host_position
        ]
        most_specific = self.most_specific(visible)
        return (
            self.python_name(most_specific)
            if most_specific is not None
            else DESCRIPTION_ENTITY
        )

    def class_named(self, value_type: PythonName) -> Optional[URIRef]:
        """
        :return: The microworld class a generated class name stands for, or ``None``
            for the vocabulary's base class.
        """
        named = [
            cls
            for cls in self.description_classes
            if self.python_name(cls) == value_type
        ]
        return named[0] if named else None

    def is_visible(self, value_type: PythonName, host: URIRef) -> bool:
        """
        :return: Whether the module declaring the host class can refer to the type.
        """
        modules = [
            module_file.vocabulary_module_name
            for module_file in PredictiveManipulationOntologyFile
        ]
        host_position = modules.index(self.module_of(host).vocabulary_module_name)
        return value_type.module in ("", *modules[: host_position + 1])

    def most_specific(self, classes: list[URIRef]) -> Optional[URIRef]:
        """
        :return: The single given class every other given class subsumes, or ``None``.
        """
        candidates = [
            cls
            for cls in classes
            if all(other in self.ancestors[cls] for other in classes)
        ]
        return candidates[0] if len(candidates) == 1 else None

    def field_declaration(self, prop: URIRef, host: URIRef) -> FieldDeclaration:
        """
        :return: The field standing for a property on one of its host classes.
        """
        kind, value_type, literal_type = self.value_of(prop)
        if kind == ValueKind.CONCEPT_REFERENCE and any(
            restricted == prop and filler in self.description_classes
            for _, restricted, filler in self.restrictions
        ):
            raise MissingPropertyRange(str(prop))
        range_class = (
            self.class_named(value_type) if kind == ValueKind.DESCRIPTION else None
        )
        if kind == ValueKind.DESCRIPTION:
            value_type = self.visible_type(value_type, host)
        if kind == ValueKind.ENUMERATION and not self.is_visible(value_type, host):
            raise FieldTypeOutOfReach(str(prop))
        return FieldDeclaration(
            name=python_attribute_name(str(prop)),
            property_iri=str(prop),
            comment=self.comment(prop)
            or EXTERNAL_PROPERTY_COMMENT.format(iri=str(prop)),
            functional=self.is_functional(prop),
            value_kind=kind,
            value_type=value_type,
            literal_type=literal_type,
            range_iri=str(range_class) if range_class is not None else "",
        )

    def descriptor_declaration(self, prop: URIRef) -> DescriptorDeclaration:
        """
        :return: The descriptor class managing a property's fields.
        """
        return DescriptorDeclaration(
            property_iri=str(prop),
            python_name=self.descriptor_name(prop),
            comment=self.comment(prop)
            or EXTERNAL_PROPERTY_COMMENT.format(iri=str(prop)),
            super_descriptors=tuple(
                self.descriptor_name(superproperty)
                for superproperty in sorted(
                    self.graph.objects(prop, RDFS.subPropertyOf)
                )
                if superproperty in self.microworld_properties
                and self.value_of(superproperty)[0] == ValueKind.DESCRIPTION
            ),
            symmetric=(prop, RDF.type, OWL.SymmetricProperty) in self.graph,
        )

    # %% declarations

    def expression(self, node: Node) -> ClassExpression:
        """
        :return: The class expression an RDF node states.
        """
        if isinstance(node, URIRef):
            return NamedClass(iri=str(node))
        members = self.graph.value(node, OWL.oneOf)
        if members is not None:
            return EnumeratedClass(
                members=tuple(str(member) for member in self.graph.items(members))
            )
        operands = self.graph.value(node, OWL.intersectionOf)
        if operands is not None:
            return IntersectionOf(
                operands=tuple(
                    self.expression(operand) for operand in self.graph.items(operands)
                )
            )
        operands = self.graph.value(node, OWL.unionOf)
        if operands is not None:
            return UnionOf(
                operands=tuple(
                    self.expression(operand) for operand in self.graph.items(operands)
                )
            )
        complemented = self.graph.value(node, OWL.complementOf)
        if complemented is not None:
            return ComplementOf(operand=self.expression(complemented))
        return self.restriction(node)

    def restriction(self, node: BNode) -> PropertyRestriction:
        """
        :return: The property restriction an RDF node states.
        """
        prop = str(self.graph.value(node, OWL.onProperty))
        for predicate, quantifier in QUANTIFIED_RESTRICTIONS.items():
            filler = self.graph.value(node, predicate)
            if filler is not None:
                return PropertyRestriction(
                    property_iri=prop,
                    quantifier=quantifier,
                    filler=self.expression(filler),
                )
        value = self.graph.value(node, OWL.hasValue)
        if value is not None:
            return PropertyRestriction(
                property_iri=prop,
                quantifier=Quantifier.VALUE,
                value=str(value) if isinstance(value, URIRef) else value.toPython(),
            )
        on_class = self.graph.value(node, OWL.onClass)
        for predicate, quantifier in CARDINALITY_RESTRICTIONS.items():
            count = self.graph.value(node, predicate)
            if count is not None:
                return PropertyRestriction(
                    property_iri=prop,
                    quantifier=quantifier,
                    filler=self.expression(on_class) if on_class is not None else None,
                    cardinality=int(count),
                )
        raise UnsupportedRestriction(str(node))

    def class_restrictions(self, cls: URIRef) -> tuple[ClassExpression, ...]:
        """
        :return: The anonymous class expressions a class is declared a subclass of, in a
            stable order.
        """
        return tuple(
            sorted(
                (
                    self.expression(superclass)
                    for superclass in self.graph.objects(cls, RDFS.subClassOf)
                    if isinstance(superclass, BNode)
                ),
                key=repr,
            )
        )

    def definition(self, cls: URIRef) -> Optional[ClassExpression]:
        """
        :return: The class expression a class is declared equivalent to, if any.
        """
        equivalent = self.graph.value(cls, OWL.equivalentClass)
        return self.expression(equivalent) if equivalent is not None else None

    def class_declaration(self, cls: URIRef) -> ClassDeclaration:
        """
        :return: The dataclass standing for a description class.
        """
        return ClassDeclaration(
            iri=str(cls),
            python_name=self.python_name(cls),
            comment=self.comment(cls),
            bases=tuple(
                self.python_name(base)
                for base in sorted(self.direct_bases[cls], key=local_name)
            ),
            linked_terms=tuple(
                sorted(
                    str(superclass)
                    for superclass in self.named_superclasses(cls)
                    if not self.is_microworld_term(superclass)
                )
            ),
            evidence_status=str(self.graph.value(cls, EVIDENCE_STATUS) or ""),
            claims=tuple(
                sorted(
                    str(claim) for claim in self.graph.objects(cls, REPRESENTS_CLAIM)
                )
            ),
            restrictions=self.class_restrictions(cls),
            definition=self.definition(cls),
            fields=tuple(
                self.field_declaration(prop, cls)
                for prop in self.described_properties
                if cls in self.field_hosts[prop]
            ),
        )

    def partition_declaration(self, cls: URIRef) -> ValuePartitionDeclaration:
        """
        :return: The enumeration standing for a value partition.
        """
        return ValuePartitionDeclaration(
            iri=str(cls),
            python_name=self.python_name(cls),
            comment=self.comment(cls),
            members=tuple(
                MemberDeclaration(
                    iri=str(member),
                    name=upper_snake_case(local_name(member)),
                    comment=self.comment(member),
                )
                for member in self.value_partitions[cls]
            ),
            member_properties=tuple(
                self.member_property_declaration(prop, cls)
                for prop in self.member_properties
                if cls in self.hosts(prop)
            ),
        )

    def member_property_declaration(
        self, prop: URIRef, partition: URIRef
    ) -> MemberPropertyDeclaration:
        """
        :return: The enumeration property standing for a property of a partition's
            members.
        """
        value_partition = self.graph.value(prop, RDFS.range)
        enumeration = self.python_name(value_partition)
        return MemberPropertyDeclaration(
            name=python_attribute_name(str(prop)),
            property_iri=str(prop),
            comment=self.comment(prop),
            value_type=enumeration,
            values=tuple(
                MemberValue(
                    member=upper_snake_case(local_name(member)),
                    value=PythonName(
                        module=enumeration.module,
                        name=f"{enumeration.name}.{upper_snake_case(local_name(self.graph.value(member, prop)))}",
                    ),
                )
                for member in self.value_partitions[partition]
            ),
        )

    def ordered_classes(
        self, module_file: PredictiveManipulationOntologyFile
    ) -> list[URIRef]:
        """
        :return: The module's description classes, every class after its bases.
        """
        in_module = [
            cls
            for cls in self.description_classes
            if self.module_of(cls) == module_file
        ]
        ordered: list[URIRef] = []

        def place(cls: URIRef) -> None:
            if cls in ordered:
                return
            for base in sorted(self.direct_bases[cls]):
                if base in in_module:
                    place(base)
            ordered.append(cls)

        for cls in in_module:
            place(cls)
        return ordered

    def ordered_descriptors(
        self, module_file: PredictiveManipulationOntologyFile
    ) -> list[URIRef]:
        """
        :return: The properties whose descriptors the module defines, every property
            after its super-properties.
        """
        in_module = [
            prop
            for prop in self.described_properties
            if self.value_of(prop)[0] == ValueKind.DESCRIPTION
            and self.descriptor_module(prop) == module_file
        ]
        ordered: list[URIRef] = []

        def place(prop: URIRef) -> None:
            if prop in ordered:
                return
            for superproperty in sorted(self.graph.objects(prop, RDFS.subPropertyOf)):
                if superproperty in in_module:
                    place(superproperty)
            ordered.append(prop)

        for prop in in_module:
            place(prop)
        return ordered

    def module_declaration(
        self, module_file: PredictiveManipulationOntologyFile
    ) -> ModuleDeclaration:
        """
        :return: Everything the module's generated vocabulary module defines.
        """
        self.verify_unique_names()
        graph = self.module_graphs[module_file]
        (ontology,) = graph.subjects(RDF.type, OWL.Ontology)
        return ModuleDeclaration(
            name=module_file.vocabulary_module_name,
            source_file=module_file.value,
            comment=str(graph.value(ontology, RDFS.comment) or ""),
            value_partitions=[
                self.partition_declaration(cls)
                for cls in sorted(self.value_partitions)
                if self.module_of(cls) == module_file
            ],
            classes=[
                self.class_declaration(cls) for cls in self.ordered_classes(module_file)
            ],
            descriptors=[
                self.descriptor_declaration(prop)
                for prop in self.ordered_descriptors(module_file)
            ],
            attachments=self.attachments(module_file),
        )

    def attachments(
        self, module_file: PredictiveManipulationOntologyFile
    ) -> list[AttachmentDeclaration]:
        """
        :return: The descriptor attachments the module makes: those whose field's
            class and descriptor are both visible first from this module.
        """
        modules = list(PredictiveManipulationOntologyFile)
        return [
            AttachmentDeclaration(
                host=self.python_name(host),
                field_name=python_attribute_name(str(prop)),
                descriptor=self.descriptor_name(prop),
            )
            for prop in self.described_properties
            if self.value_of(prop)[0] == ValueKind.DESCRIPTION
            for host in self.field_hosts[prop]
            if max(
                modules.index(self.module_of(host)),
                modules.index(self.descriptor_module(prop)),
            )
            == modules.index(module_file)
        ]

    def verify_unique_names(self) -> None:
        """
        Refuse ontology terms that would be written as Python definitions of the same
        name, since the later one would silently replace the earlier.
        """
        terms_by_name: dict[str, list[str]] = {}
        for cls in [*self.description_classes, *sorted(self.value_partitions)]:
            terms_by_name.setdefault(local_name(cls), []).append(str(cls))
        for prop in self.described_properties:
            if self.value_of(prop)[0] == ValueKind.DESCRIPTION:
                terms_by_name.setdefault(capitalised(local_name(prop)), []).append(
                    str(prop)
                )
        conflicts = {
            name: iris for name, iris in terms_by_name.items() if len(iris) > 1
        }
        if conflicts:
            raise ConflictingPythonNames(conflicts)
