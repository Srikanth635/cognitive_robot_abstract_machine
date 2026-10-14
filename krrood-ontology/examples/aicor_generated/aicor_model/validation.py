"""Explicitly scoped validation; no mutation, inference or automatic enforcement."""

from pathlib import Path as _Path
from krrood_ontology.validation import ModelValidator as _ModelValidator
from .runtime import MODEL
from .queries import model_queries

model_validator = _ModelValidator(
    MODEL,
    _Path(__file__).resolve().parents[1] / "source" / "closure.ttl",
    definition_queries=model_queries,
)
