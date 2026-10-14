"""Positive class-expression metadata and scoped native KRROOD EQL queries."""

from krrood_ontology.model import ClassExpression, ExpressionKind, LiteralDefinition
from krrood_ontology.queries import ModelQueries

from .runtime import MODEL

CLASS_EXPRESSIONS = {
    "https://example.org/scoped#Choice": ClassExpression(
        kind=ExpressionKind.ENUMERATION,
        iri=None,
        property_iri=None,
        literal=None,
        operands=(),
        individuals=(
            "https://example.org/scoped#one",
            "https://example.org/scoped#two",
        ),
    ),
    "https://example.org/scoped#Either": ClassExpression(
        kind=ExpressionKind.UNION,
        iri=None,
        property_iri=None,
        literal=None,
        operands=(
            ClassExpression(
                kind=ExpressionKind.CLASS,
                iri="https://example.org/scoped#Alpha",
                property_iri=None,
                literal=None,
                operands=(),
                individuals=(),
            ),
            ClassExpression(
                kind=ExpressionKind.CLASS,
                iri="https://example.org/scoped#Beta",
                property_iri=None,
                literal=None,
                operands=(),
                individuals=(),
            ),
        ),
        individuals=(),
    ),
    "https://example.org/scoped#Linked": ClassExpression(
        kind=ExpressionKind.SOME,
        iri=None,
        property_iri="https://example.org/scoped#link",
        literal=None,
        operands=(
            ClassExpression(
                kind=ExpressionKind.UNION,
                iri=None,
                property_iri=None,
                literal=None,
                operands=(
                    ClassExpression(
                        kind=ExpressionKind.CLASS,
                        iri="https://example.org/scoped#Alpha",
                        property_iri=None,
                        literal=None,
                        operands=(),
                        individuals=(),
                    ),
                    ClassExpression(
                        kind=ExpressionKind.CLASS,
                        iri="https://example.org/scoped#Beta",
                        property_iri=None,
                        literal=None,
                        operands=(),
                        individuals=(),
                    ),
                ),
                individuals=(),
            ),
        ),
        individuals=(),
    ),
    "https://example.org/scoped#Marked": ClassExpression(
        kind=ExpressionKind.HAS_VALUE,
        iri=None,
        property_iri="https://example.org/scoped#flag",
        literal=LiteralDefinition(
            lexical="false",
            datatype="http://www.w3.org/2001/XMLSchema#boolean",
        ),
        operands=(),
        individuals=(),
    ),
    "https://example.org/scoped#MarkedAlias": ClassExpression(
        kind=ExpressionKind.HAS_VALUE,
        iri=None,
        property_iri="https://example.org/scoped#flag",
        literal=LiteralDefinition(
            lexical="false",
            datatype="http://www.w3.org/2001/XMLSchema#boolean",
        ),
        operands=(),
        individuals=(),
    ),
    "https://example.org/scoped#Nested": ClassExpression(
        kind=ExpressionKind.SOME,
        iri=None,
        property_iri="https://example.org/scoped#link",
        literal=None,
        operands=(
            ClassExpression(
                kind=ExpressionKind.SOME,
                iri=None,
                property_iri="https://example.org/scoped#link",
                literal=None,
                operands=(
                    ClassExpression(
                        kind=ExpressionKind.CLASS,
                        iri="https://example.org/scoped#Alpha",
                        property_iri=None,
                        literal=None,
                        operands=(),
                        individuals=(),
                    ),
                ),
                individuals=(),
            ),
        ),
        individuals=(),
    ),
    "https://example.org/scoped#ReverseMarked": ClassExpression(
        kind=ExpressionKind.HAS_VALUE,
        iri=None,
        property_iri="https://example.org/scoped#flag",
        literal=LiteralDefinition(
            lexical="false",
            datatype="http://www.w3.org/2001/XMLSchema#boolean",
        ),
        operands=(),
        individuals=(),
    ),
}

CLASS_ALIASES = dict(
    (
        ("https://example.org/scoped#Alias", ("https://example.org/scoped#Alpha",)),
        ("https://example.org/scoped#Alpha", ("https://example.org/scoped#Alias",)),
        (
            "https://example.org/scoped#FlagChild",
            ("https://example.org/scoped#InheritedFlagAlias",),
        ),
        (
            "https://example.org/scoped#InheritedFlagAlias",
            ("https://example.org/scoped#FlagChild",),
        ),
        (
            "https://example.org/scoped#Marked",
            ("https://example.org/scoped#MarkedAlias",),
        ),
        (
            "https://example.org/scoped#MarkedAlias",
            ("https://example.org/scoped#Marked",),
        ),
    )
)
model_queries = ModelQueries(
    MODEL, CLASS_EXPRESSIONS, CLASS_ALIASES, partial_individuals=True
)

__all__ = [
    "CLASS_EXPRESSIONS",
    "CLASS_ALIASES",
    "model_queries",
    "ClassExpression",
    "ExpressionKind",
    "LiteralDefinition",
]
