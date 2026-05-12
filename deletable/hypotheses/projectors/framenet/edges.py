"""FrameNet-specific hypothesis graph relations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from llmr.hypotheses.elements import HypothesisEdge


@dataclass
class EvokesFrameEdge(HypothesisEdge):
    """Links an instruction node to a FrameNet frame hypothesis it evokes."""

    RELATION_NAME: ClassVar[str] = "evokes_frame"


@dataclass
class HasRoleEdge(HypothesisEdge):
    """Links a FrameNet frame hypothesis node to one of its role nodes."""

    RELATION_NAME: ClassVar[str] = "has_role"
