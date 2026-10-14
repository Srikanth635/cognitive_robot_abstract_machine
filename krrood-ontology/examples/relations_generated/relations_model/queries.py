"""Positive class-expression metadata and scoped native KRROOD EQL queries."""

from krrood_ontology.model import ClassExpression, ExpressionKind, LiteralDefinition
from krrood_ontology.queries import ModelQueries

from .runtime import MODEL

CLASS_EXPRESSIONS = {}

model_queries = ModelQueries(MODEL, CLASS_EXPRESSIONS)

__all__ = [
    "CLASS_EXPRESSIONS",
    "model_queries",
    "ClassExpression",
    "ExpressionKind",
    "LiteralDefinition",
]
