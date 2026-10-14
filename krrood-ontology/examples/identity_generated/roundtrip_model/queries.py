"""Positive class-expression metadata and scoped native KRROOD EQL queries."""

from krrood_ontology.model import ClassExpression, ExpressionKind, LiteralDefinition
from krrood_ontology.queries import ModelQueries

from .runtime import MODEL

CLASS_EXPRESSIONS = {
    "https://example.org/roundtrip#Pick": ClassExpression(
        kind=ExpressionKind.ENUMERATION,
        iri=None,
        property_iri=None,
        literal=None,
        operands=(),
        individuals=(
            "https://example.org/roundtrip#one",
            "https://example.org/roundtrip#two",
            "https://example.org/roundtrip#three",
        ),
    ),
}

CLASS_ALIASES = dict(())
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
