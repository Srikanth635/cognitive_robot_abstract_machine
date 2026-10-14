"""Translate preserved OWL constraints into the existing expression record types."""

from dataclasses import dataclass

from rdflib import OWL, RDF, RDFS, XSD, BNode, Graph, Literal, URIRef

from krrood_ontology.expressions import expression_members
from krrood_ontology.model import ClassExpression, ExpressionKind, LiteralDefinition
from krrood_ontology.runtime import GeneratedModel
from krrood_ontology.translation import OntologyTranslator


class ConstraintTranslationError(ValueError):
    """An axiom that the scoped validator cannot faithfully evaluate."""


@dataclass
class ConstraintParser:
    graph: Graph
    model: GeneratedModel

    def __post_init__(self):
        if not isinstance(self.model.classes, dict):
            from types import SimpleNamespace

            self.model = SimpleNamespace(
                classes={c.iri: c for c in self.model.classes},
                properties={v.iri: v for v in self.model.properties},
            )

    def fail(self, code, message, iri=None):
        raise ConstraintTranslationError(
            f"{code}: {message}" + (f" [{iri}]" if iri else "")
        )

    def parse(self, term, ancestors=frozenset()):
        if isinstance(term, URIRef):
            if str(term) in self.model.classes or term in (OWL.Thing, OWL.Nothing):
                return ClassExpression(ExpressionKind.CLASS, iri=str(term))
            if str(term).startswith(str(XSD)):
                try:
                    OntologyTranslator()._datatype_type(term)
                except ValueError as error:
                    raise ConstraintTranslationError(str(error)) from error
                return ClassExpression(ExpressionKind.DATATYPE, iri=str(term))
            self.fail(
                "constraint_class",
                "Unknown named class or unsupported datatype",
                str(term),
            )
        if not isinstance(term, BNode) or term in ancestors:
            self.fail(
                "constraint_cycle",
                "Anonymous expressions must be acyclic blank nodes",
                str(term),
            )
        path = ancestors | {term}
        operators = {
            OWL.unionOf: ExpressionKind.UNION,
            OWL.intersectionOf: ExpressionKind.INTERSECTION,
            OWL.oneOf: ExpressionKind.ENUMERATION,
            OWL.someValuesFrom: ExpressionKind.SOME,
            OWL.allValuesFrom: ExpressionKind.ALL,
            OWL.hasValue: ExpressionKind.HAS_VALUE,
            OWL.minQualifiedCardinality: ExpressionKind.MIN,
            OWL.maxQualifiedCardinality: ExpressionKind.MAX,
            OWL.qualifiedCardinality: ExpressionKind.EXACT,
            OWL.minCardinality: ExpressionKind.MIN,
            OWL.maxCardinality: ExpressionKind.MAX,
            OWL.cardinality: ExpressionKind.EXACT,
        }
        present = [
            (p, tuple(self.graph.objects(term, p)))
            for p in operators
            if tuple(self.graph.objects(term, p))
        ]
        if len(present) != 1 or len(present[0][1]) != 1:
            self.fail(
                "constraint_operator",
                "Requires exactly one supported expression operator",
                term.n3(),
            )
        predicate, (value,) = present[0]
        kind = operators[predicate]
        allowed = {RDF.type, predicate, RDFS.label, RDFS.comment}
        reverse_owners = tuple(self.graph.objects(term, OWL.equivalentClass))
        if reverse_owners:
            if any(
                not isinstance(v, URIRef) or str(v) not in self.model.classes
                for v in reverse_owners
            ):
                self.fail(
                    "constraint_owner",
                    "Reverse definitions require registered named owners",
                )
            allowed.add(OWL.equivalentClass)
        if kind in (
            ExpressionKind.UNION,
            ExpressionKind.INTERSECTION,
            ExpressionKind.ENUMERATION,
        ):
            members = expression_members(self.graph, value, self.fail)
            if kind == ExpressionKind.ENUMERATION:
                if not members or any(not isinstance(v, URIRef) for v in members):
                    self.fail(
                        "constraint_enumeration",
                        "Only nonempty named-individual enumerations are supported",
                    )
                result = ClassExpression(kind, individuals=tuple(map(str, members)))
            else:
                if not members:
                    self.fail(
                        "constraint_list",
                        "Union/intersection requires at least one operand",
                    )
                result = ClassExpression(
                    kind, operands=tuple(self.parse(v, path) for v in members)
                )
        else:
            allowed.add(OWL.onProperty)
            if set(self.graph.objects(term, RDF.type)) != {OWL.Restriction}:
                self.fail(
                    "constraint_restriction",
                    "Restrictions require owl:Restriction type",
                )
            props = tuple(self.graph.objects(term, OWL.onProperty))
            if len(props) != 1 or str(props[0]) not in self.model.properties:
                self.fail(
                    "constraint_property",
                    "Restriction requires one registered named property",
                )
            prop = self.model.properties[str(props[0])]
            if kind == ExpressionKind.HAS_VALUE:
                if prop.kind == "object":
                    if not isinstance(value, URIRef):
                        self.fail(
                            "constraint_value",
                            "Object hasValue requires a named individual",
                        )
                    result = ClassExpression(
                        kind, iri=str(value), property_iri=prop.iri
                    )
                else:
                    if (
                        not isinstance(value, Literal)
                        or value.language
                        or value.ill_typed
                        or isinstance(value.toPython(), Literal)
                    ):
                        self.fail(
                            "constraint_literal",
                            "Requires a supported well-typed literal without language tags",
                        )
                    datatype = str(value.datatype or XSD.string)
                    try:
                        OntologyTranslator()._datatype_type(URIRef(datatype))
                    except ValueError as error:
                        raise ConstraintTranslationError(str(error)) from error
                    result = ClassExpression(
                        kind,
                        property_iri=prop.iri,
                        literal=LiteralDefinition(str(value), datatype),
                    )
            elif kind in (ExpressionKind.SOME, ExpressionKind.ALL):
                filler = self.parse(value, path)
                self.check_filler(prop.kind, filler)
                result = ClassExpression(
                    kind, property_iri=prop.iri, operands=(filler,)
                )
            else:
                if (
                    not isinstance(value, Literal)
                    or value.datatype != XSD.nonNegativeInteger
                    or value.ill_typed
                    or type(value.toPython()) is not int
                    or value.toPython() < 0
                ):
                    self.fail(
                        "constraint_cardinality",
                        "Cardinality requires a valid xsd:nonNegativeInteger",
                    )
                qualifier = OWL.onClass if prop.kind == "object" else OWL.onDataRange
                qualified = predicate in {
                    OWL.minQualifiedCardinality,
                    OWL.maxQualifiedCardinality,
                    OWL.qualifiedCardinality,
                }
                fillers = tuple(self.graph.objects(term, qualifier))
                if qualified:
                    allowed.add(qualifier)
                    if len(fillers) != 1:
                        self.fail(
                            "constraint_qualifier",
                            "Qualified cardinality requires one kind-compatible filler",
                        )
                    filler = self.parse(fillers[0], path)
                    self.check_filler(prop.kind, filler)
                    operands = (filler,)
                else:
                    operands = ()
                result = ClassExpression(
                    kind,
                    property_iri=prop.iri,
                    operands=operands,
                    cardinality=value.toPython(),
                )
        if set(self.graph.predicates(term)) - allowed:
            self.fail(
                "constraint_extra_terms",
                "Expression contains additional unsupported terms",
                term.n3(),
            )
        return result

    def check_filler(self, property_kind, node):
        if node.kind in (ExpressionKind.UNION, ExpressionKind.INTERSECTION):
            for operand in node.operands:
                self.check_filler(property_kind, operand)
        elif property_kind == "datatype" and node.kind != ExpressionKind.DATATYPE:
            self.fail(
                "constraint_datatype_filler",
                "Only supported named datatypes and their unions/intersections are supported",
            )
        elif property_kind == "object" and node.kind == ExpressionKind.DATATYPE:
            self.fail(
                "constraint_object_filler",
                "Object restrictions cannot use a datatype filler",
            )
