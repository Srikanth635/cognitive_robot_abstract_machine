"""minimal_llmr — lightweight LLM action resolution pipeline for KRROOD.

Three isolated tasks:
  1. Inference   — classify a NL instruction into a known action class
  2. Generation  — describe free action parameters using an LLM
  3. Grounding   — resolve descriptions to concrete world objects (SymbolGraph)

Public API
----------
  LLMBackend                  KRROOD GenerativeBackend (main integration point)
  instance_from_instruction   NL → resolved action
  instance_from_match         Match → resolved action
  underspecified_match_for    build a fully-free Match for an action class


  ReferentDescription         Generation → Grounding boundary object
  ParameterInterpretations    full output of the Generation task
  ActionClassification        output of the Inference task
"""

from minimal_llmr.backend import LLMBackend
from minimal_llmr.core.entrypoints import (
    instance_from_instruction,
    instance_from_match,
    underspecified_match_for,
)
from minimal_llmr.inference.classifier import build_action_registry
from minimal_llmr.core.errors import (
    ActionClassificationFailed,
    ParameterDescriptionFailed,
    UnresolvedRequiredParameters,
)
from minimal_llmr.core.schemas import (
    ActionClassification,
    ParameterInterpretation,
    ParameterInterpretations,
    ReferentDescription,
)

__all__ = [
    "LLMBackend",
    "instance_from_instruction",
    "instance_from_match",
    "underspecified_match_for",
    "build_action_registry",

    "ReferentDescription",
    "ParameterInterpretation",
    "ParameterInterpretations",
    "ActionClassification",
    "ActionClassificationFailed",
    "ParameterDescriptionFailed",
    "UnresolvedRequiredParameters",
]
