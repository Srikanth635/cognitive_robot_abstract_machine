"""Validate the positive anonymous expressions admitted by Phase 6."""

from dataclasses import dataclass
from graphlib import CycleError, TopologicalSorter
from typing import NoReturn

from rdflib import OWL, RDF, XSD, BNode, Literal, URIRef
from rdflib.term import Identifier

from krrood_ontology.model import (
    ClassExpression,
    CompilationDiagnostic,
    ExpressionDefinition,
    ExpressionKind,
    LiteralDefinition,
    ModelDefinition,
    OntologyTranslationError,
    PropertyKind,
)
from krrood_ontology.ontology import OntologyDocument


def expression_members(graph, head, fail):
    """Read an RDF list using the existing expression profile checks."""
    cells = set()
    cell = head
    while cell != RDF.nil:
        if not isinstance(cell, BNode) or cell in cells:
            fail(
                "expression_list",
                "Expression lists require acyclic blank-node cells",
            )
        cells.add(cell)
        first = tuple(graph.objects(cell, RDF.first))
        rest = tuple(graph.objects(cell, RDF.rest))
        if len(first) != 1 or len(rest) != 1:
            fail(
                "expression_list",
                "Each list cell requires exactly one rdf:first and rdf:rest",
            )
        cell = rest[0]
    return tuple(graph.items(head))


@dataclass
class ExpressionTranslator:
    """Map anonymous equivalent-class expressions to finite positive queries."""

    def translate(
        self, document: OntologyDocument, model: ModelDefinition
    ) -> tuple[ExpressionDefinition, ...]:
        graph = document.graph
        classes = {item.iri for item in model.classes}
        properties = {item.iri: item for item in model.properties}
        visited = set()
        definitions = {}

        def parse(
            term: Identifier, ancestors: frozenset[BNode] = frozenset()
        ) -> ClassExpression:
            if isinstance(term, URIRef):
                if str(term) not in classes:
                    self.fail(
                        "expression_class",
                        "Expression operands require declared named classes",
                        str(term),
                    )
                return ClassExpression(ExpressionKind.CLASS, iri=str(term))
            if not isinstance(term, BNode) or term in ancestors:
                self.fail(
                    "expression_cycle",
                    "Expressions require acyclic anonymous nodes",
                    str(term),
                )
            visited.add(term)
            path = ancestors | {term}
            operations = [
                (predicate, tuple(graph.objects(term, predicate)))
                for predicate in (
                    OWL.intersectionOf,
                    OWL.unionOf,
                    OWL.oneOf,
                    OWL.hasValue,
                    OWL.someValuesFrom,
                )
            ]
            present = [
                (predicate, values) for predicate, values in operations if values
            ]
            if len(present) != 1 or len(present[0][1]) != 1:
                self.fail(
                    "expression_shape",
                    "Each expression requires one supported operator and one value",
                    str(term),
                )
            predicate, (value,) = present[0]
            allowed = {RDF.type, predicate, OWL.equivalentClass}
            if predicate in (OWL.hasValue, OWL.someValuesFrom):
                allowed.add(OWL.onProperty)
                if set(graph.objects(term, RDF.type)) != {OWL.Restriction}:
                    self.fail(
                        "expression_restriction",
                        "Restrictions must be explicitly typed owl:Restriction",
                    )
                terms = tuple(graph.objects(term, OWL.onProperty))
                if len(terms) != 1 or str(terms[0]) not in properties:
                    self.fail(
                        "expression_property",
                        "Restrictions require one mapped named property",
                    )
                prop = properties[str(terms[0])]
                if predicate == OWL.someValuesFrom:
                    if prop.kind is not PropertyKind.OBJECT:
                        self.fail(
                            "expression_datatype_existential",
                            "Datatype existential restrictions are deferred",
                        )
                    result = ClassExpression(
                        ExpressionKind.SOME,
                        property_iri=prop.iri,
                        operands=(parse(value, path),),
                    )
                elif prop.kind is PropertyKind.OBJECT:
                    if not isinstance(value, URIRef):
                        self.fail(
                            "expression_value",
                            "Object hasValue requires a named individual",
                        )
                    result = ClassExpression(
                        ExpressionKind.HAS_VALUE, iri=str(value), property_iri=prop.iri
                    )
                else:
                    if (
                        not isinstance(value, Literal)
                        or value.language
                        or str(value.datatype or XSD.string) != prop.range_iri
                        or value.ill_typed
                        or isinstance(value.toPython(), Literal)
                    ):
                        self.fail(
                            "expression_literal",
                            "hasValue requires a valid literal of the declared datatype",
                        )
                    result = ClassExpression(
                        ExpressionKind.HAS_VALUE,
                        property_iri=prop.iri,
                        literal=LiteralDefinition(str(value), prop.range_iri),
                    )
            else:
                values = expression_members(graph, value, self.fail)
                if predicate == OWL.oneOf:
                    if not values or any(
                        not isinstance(item, URIRef) for item in values
                    ):
                        self.fail(
                            "expression_enumeration",
                            "oneOf requires a nonempty list of named individuals",
                        )
                    result = ClassExpression(
                        ExpressionKind.ENUMERATION, individuals=tuple(map(str, values))
                    )
                else:
                    if len(values) < 2:
                        self.fail(
                            "expression_operands",
                            "Intersection and union require at least two operands",
                        )
                    kind = (
                        ExpressionKind.INTERSECTION
                        if predicate == OWL.intersectionOf
                        else ExpressionKind.UNION
                    )
                    result = ClassExpression(
                        kind, operands=tuple(parse(item, path) for item in values)
                    )
            if set(graph.predicates(term)) - allowed:
                self.fail(
                    "expression_extra_predicate",
                    "Anonymous expression has unsupported additional predicates",
                    str(term),
                )
            types = set(graph.objects(term, RDF.type))
            if predicate not in (OWL.hasValue, OWL.someValuesFrom) and types - {
                OWL.Class
            }:
                self.fail(
                    "expression_type",
                    "Anonymous class expressions may only have owl:Class type",
                )
            return result

        for subject, target in sorted(
            graph.subject_objects(OWL.equivalentClass),
            key=lambda pair: tuple(map(str, pair)),
        ):
            if isinstance(subject, BNode) and isinstance(target, URIRef):
                subject, target = target, subject
            if (
                not isinstance(subject, URIRef)
                or str(subject) not in classes
                or not isinstance(target, BNode)
            ):
                self.fail(
                    "expression_root",
                    "Only a declared named class equivalent to an anonymous expression is supported",
                )
            parsed = parse(target)
            if str(subject) in definitions and definitions[str(subject)] != parsed:
                self.fail(
                    "expression_multiple_definitions",
                    "Only one expression definition per named class is supported",
                    str(subject),
                )
            definitions[str(subject)] = parsed
        expression_nodes = {
            subject
            for predicate in (
                OWL.intersectionOf,
                OWL.unionOf,
                OWL.oneOf,
                OWL.onProperty,
                OWL.hasValue,
                OWL.someValuesFrom,
            )
            for subject in graph.subjects(predicate)
        } | set(graph.subjects(RDF.type, OWL.Restriction))
        if expression_nodes - visited:
            self.fail(
                "expression_unattached",
                "Supported expression nodes must belong to a compiled equivalent-class definition",
            )

        def references(node: ClassExpression) -> set[str]:
            result = {node.iri} if node.kind is ExpressionKind.CLASS else set()
            for child in node.operands:
                result.update(references(child))
            return result

        try:
            tuple(
                TopologicalSorter(
                    {
                        iri: references(node) & definitions.keys()
                        for iri, node in definitions.items()
                    }
                ).static_order()
            )
        except CycleError as error:
            raise OntologyTranslationError(
                (
                    CompilationDiagnostic(
                        "expression_definition_cycle",
                        "Recursive named expression definitions are deferred",
                    ),
                )
            ) from error
        return tuple(
            ExpressionDefinition(iri, node) for iri, node in sorted(definitions.items())
        )

    @staticmethod
    def fail(code: str, message: str, iri: str | None = None) -> NoReturn:
        raise OntologyTranslationError((CompilationDiagnostic(code, message, iri),))
