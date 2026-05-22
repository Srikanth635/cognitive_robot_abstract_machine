"""Cross-boundary data contracts for minimal_llmr.

These Pydantic models are the only objects that cross task boundaries:

  ReferentDescription    crosses Generation → Grounding
  ParameterInterpretation  one parameter's generated interpretation
  ParameterInterpretations  full output of the Generation task
  ActionClassification     output of the Inference task
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class ReferentDescription(BaseModel):
    """LLM-generated description of a world entity — input to Grounding."""

    name: Optional[str] = None
    """Entity name as it appears in the instruction or world context."""

    semantic_type: Optional[str] = None
    """Exact type name from Available Semantic Types in the world context."""

    spatial_context: Optional[str] = None
    """Spatial qualifier from the instruction, e.g. 'on the table'."""

    attributes: Optional[dict[str, str]] = None
    """Discriminating key/value attributes such as color or size."""


class ReasoningTrace(BaseModel):
    """Structured reasoning for one parameter choice."""

    causal: str = ""
    """Why this choice leads to the correct physical outcome (cause → effect chain)."""

    counterfactual: str = ""
    """Which alternative(s) were considered and why they were rejected."""

    spatial: str = ""
    """Spatial evidence: positions, distances, relative locations, and reachability
    from the world context that support this choice."""

    geometric: str = ""
    """Geometric evidence: object shapes, sizes, and orientations from the world
    context that influence this choice (e.g. approach direction, grasp offset)."""


class ParameterInterpretation(BaseModel):
    """LLM's interpretation of one free action parameter."""

    param_name: str
    """Exact parameter name as listed in the action schema."""

    value: Optional[str] = None
    """String value for DISCRETE parameters (enum member name or primitive)."""

    referent_description: Optional[ReferentDescription] = None
    """Entity description for REFERENT parameters; None for DISCRETE."""

    reasoning: ReasoningTrace = Field(default_factory=ReasoningTrace)
    """Causal and counterfactual justification for this interpretation."""


class ParameterInterpretations(BaseModel):
    """Full output of the Generation task — one interpretation per free parameter."""

    interpretations: list[ParameterInterpretation] = Field(default_factory=list)
    overall_reasoning: str = ""
    coherence_assessment: str = ""
    """How the full set of parameter choices forms a consistent, non-contradictory plan."""


class ActionClassification(BaseModel):
    """Output of the Inference task — action class identified from a NL instruction."""

    action_type: str
    """Exact Python class name, e.g. 'PickUpAction'."""

    confidence: float = 1.0
    """Model confidence in [0, 1]."""

    reasoning: str = ""
    """Brief justification."""
