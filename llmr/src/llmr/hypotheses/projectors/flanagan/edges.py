"""Flanagan-specific hypothesis graph relations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from llmr.hypotheses.elements import HypothesisEdge


@dataclass
class EvokesMotionPlanEdge(HypothesisEdge):
    """Links an instruction node to a Flanagan motion-plan hypothesis."""

    RELATION_NAME: ClassVar[str] = "evokes_motion_plan"


@dataclass
class HasMotionPhaseEdge(HypothesisEdge):
    """Links a motion-plan hypothesis node to one of its phase nodes."""

    RELATION_NAME: ClassVar[str] = "has_motion_phase"
