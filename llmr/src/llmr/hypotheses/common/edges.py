"""Reusable hypothesis graph relations shared across reasoners."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from llmr.hypotheses.elements import HypothesisEdge


@dataclass
class ProducedClaimEdge(HypothesisEdge):
    """Links a reasoner run node to a claim node it produced."""

    RELATION_NAME: ClassVar[str] = "produced_claim"


@dataclass
class AboutActionEdge(HypothesisEdge):
    """Links a hypothesis to the concrete action it describes."""

    RELATION_NAME: ClassVar[str] = "about_action"


@dataclass
class SupportedByEdge(HypothesisEdge):
    """Links a claim node to supporting evidence."""

    RELATION_NAME: ClassVar[str] = "supported_by"


@dataclass
class GroundedByEdge(HypothesisEdge):
    """Links a claim node to grounding evidence."""

    RELATION_NAME: ClassVar[str] = "grounded_by"
