"""Explicit scope helpers around native KRROOD relations."""

from krrood_ontology.relation_runtime import ModelRelations
from .runtime import MODEL
from .relations import PROPERTY_DESCRIPTORS

model_relations = ModelRelations(
    MODEL, PROPERTY_DESCRIPTORS, ("https://example.org/scoped#connected",)
)
