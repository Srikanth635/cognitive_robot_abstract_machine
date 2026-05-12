"""Flanagan-specific hypothesis graph relations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from llmr.hypotheses.elements import HypothesisEdge


@dataclass
class EvokesPlanEdge(HypothesisEdge):
    """Links an instruction node to a Flanagan motion-plan claim."""

    RELATION_NAME: ClassVar[str] = "evokes_motion_plan"


@dataclass
class HasPhaseEdge(HypothesisEdge):
    """Links a motion-plan claim node to one of its phase nodes."""

    RELATION_NAME: ClassVar[str] = "has_motion_phase"
