"""
Writing the description of a code base as OWL: a class for each class and enumeration,
an individual for each member, and a property for each field, scoped to its class.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from rdflib import OWL, RDF, RDFS, XSD, Graph, Literal, URIRef

from llm_reasoner.world.class_ontology.field_types import PlainType
from llm_reasoner.world.class_ontology.reading import (
    ClassDescription,
    ClassOntology,
    EnumerationDescription,
    FieldDescription,
)

ONTOLOGY_LABEL = "The classes of a code base"
"""
The label of the ontology.
"""

ONTOLOGY_COMMENT = (
    "Written from the live classes of a code base by llm_reasoner.world.class_ontology, "
    "which a selection file tells which classes to describe; write it again rather than "
    "editing it. Inheritance is Python inheritance through the classes described, "
    "mixins left out; a field is a property scoped to the class that adds it."
)
"""
What the ontology says of itself.
"""

MEMBER_SEPARATOR = "."
"""
What separates a class from one of its fields, and an enumeration from one of its
members, in an IRI.
"""

PLAIN_RANGES = {
    PlainType.BOOLEAN: XSD.boolean,
    PlainType.INTEGER: XSD.integer,
    PlainType.FLOAT: XSD.double,
    PlainType.TEXT: XSD.string,
}
"""
The datatype each plain value is written as.
"""

PREFIX = "code"
"""
The prefix the ontology's namespace is written with.
"""


class AnnotationName(StrEnum):
    """
    The annotation properties the ontology writes, in its own namespace.
    """

    PYTHON_NAME = "pythonName"
    """
    The module and name of the class or enumeration a term describes.
    """

    PYTHON_TYPE = "pythonType"
    """
    The type of a field as written.
    """

    DEFAULT = "defaultValue"
    """
    The default of a field written as Python, when it is a plain value.
    """

    OPTIONAL = "isOptional"
    """
    That a field may be left empty.
    """

    SKIPPED_MODULE = "skippedModule"
    """
    A module of a selected package that could not be imported, with the kind of error;
    what the error said is left out, since it may name where the code base lies.
    """


@dataclass(frozen=True)
class ClassOntologyWriter:
    """
    Writes the description of a code base as OWL.
    """

    ontology: ClassOntology
    """
    What is written.
    """

    namespace: str
    """
    The namespace of every term written.
    """

    def iri(self, identifier: str) -> URIRef:
        """
        :return: The IRI of a class or enumeration, by its module and name.
        """
        return URIRef(self.namespace + identifier)

    def annotation(self, name: AnnotationName) -> URIRef:
        """
        :return: The IRI of an annotation property.
        """
        return URIRef(self.namespace + name)

    def graph(self) -> Graph:
        """
        :return: The ontology as a graph.
        """
        graph = Graph()
        graph.bind(PREFIX, self.namespace)
        graph.bind("owl", OWL)
        self.write_header(graph)
        for description in self.ontology.classes:
            self.write_class(graph, description)
        for description in self.ontology.enumerations:
            self.write_enumeration(graph, description)
        return graph

    def turtle(self) -> str:
        """
        :return: The ontology as Turtle.
        """
        return self.graph().serialize(format="turtle")

    def write_header(self, graph: Graph) -> None:
        """
        Write what the ontology is, the annotation properties it uses, and the modules
        it skipped.
        """
        header = URIRef(self.namespace.rstrip("#/"))
        graph.add((header, RDF.type, OWL.Ontology))
        graph.add((header, RDFS.label, Literal(ONTOLOGY_LABEL)))
        graph.add((header, RDFS.comment, Literal(ONTOLOGY_COMMENT)))
        for name in AnnotationName:
            graph.add((self.annotation(name), RDF.type, OWL.AnnotationProperty))
            graph.add((self.annotation(name), RDFS.label, Literal(name.value)))
        for skipped in self.ontology.skipped_modules:
            graph.add(
                (
                    header,
                    self.annotation(AnnotationName.SKIPPED_MODULE),
                    Literal(f"{skipped.module}: {skipped.error_type}"),
                )
            )

    def write_named(
        self, graph: Graph, term: URIRef, name: str, documentation: str, identifier: str
    ) -> None:
        """
        Write a class's or an enumeration's label, documentation and Python name.
        """
        graph.add((term, RDF.type, OWL.Class))
        graph.add((term, RDFS.label, Literal(name)))
        if documentation:
            graph.add((term, RDFS.comment, Literal(documentation)))
        graph.add(
            (term, self.annotation(AnnotationName.PYTHON_NAME), Literal(identifier))
        )

    def write_class(self, graph: Graph, description: ClassDescription) -> None:
        """
        Write a class, the classes it specialises and its fields.
        """
        term = self.iri(description.identifier)
        self.write_named(
            graph,
            term,
            description.name,
            description.documentation,
            description.identifier,
        )
        for parent in description.parents:
            graph.add((term, RDFS.subClassOf, self.iri(parent)))
        for declared in description.fields:
            self.write_field(graph, term, declared)

    def write_field(
        self, graph: Graph, owner: URIRef, description: FieldDescription
    ) -> None:
        """
        Write a field as a property of the class that adds it.
        """
        term = URIRef(owner + MEMBER_SEPARATOR + description.name)
        field_type = description.field_type
        if field_type.plain is not None:
            graph.add((term, RDF.type, OWL.DatatypeProperty))
            graph.add((term, RDFS.range, PLAIN_RANGES[field_type.plain]))
        else:
            graph.add((term, RDF.type, OWL.ObjectProperty))
            if field_type.target is not None:
                graph.add((term, RDFS.range, self.iri(field_type.target)))
        if not field_type.collection:
            graph.add((term, RDF.type, OWL.FunctionalProperty))
        graph.add((term, RDFS.label, Literal(description.name)))
        graph.add((term, RDFS.domain, owner))
        if description.documentation:
            graph.add((term, RDFS.comment, Literal(description.documentation)))
        graph.add(
            (
                term,
                self.annotation(AnnotationName.PYTHON_TYPE),
                Literal(field_type.written),
            )
        )
        if description.default is not None:
            graph.add(
                (
                    term,
                    self.annotation(AnnotationName.DEFAULT),
                    Literal(description.default),
                )
            )
        if field_type.optional:
            graph.add((term, self.annotation(AnnotationName.OPTIONAL), Literal(True)))

    def write_enumeration(
        self, graph: Graph, description: EnumerationDescription
    ) -> None:
        """
        Write an enumeration as a class, and each of its members as an individual of it.
        """
        term = self.iri(description.identifier)
        self.write_named(
            graph,
            term,
            description.name,
            description.documentation,
            description.identifier,
        )
        for member in description.members:
            individual = URIRef(term + MEMBER_SEPARATOR + member)
            graph.add((individual, RDF.type, OWL.NamedIndividual))
            graph.add((individual, RDF.type, term))
            graph.add((individual, RDFS.label, Literal(member)))
