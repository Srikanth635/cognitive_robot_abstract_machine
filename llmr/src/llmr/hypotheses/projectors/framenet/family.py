"""FrameNet hypothesis family descriptor."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from llmr.hypotheses.families import HypothesisFamily
from llmr.hypotheses.projectors.framenet.constants import FRAMENET_REASONER_NAME
from llmr.hypotheses.projectors.framenet.projector import FrameNetProjector
from llmr.hypotheses.projectors.framenet.view import FrameNetGraphView


@dataclass(frozen=True)
class FrameNetFamily(HypothesisFamily[FrameNetProjector, FrameNetGraphView]):
    """Family descriptor for FrameNet projections and queries."""

    REASONER_NAME: ClassVar[str] = FRAMENET_REASONER_NAME
    PROJECTOR_TYPE: ClassVar[type[FrameNetProjector]] = FrameNetProjector
    VIEW_TYPE: ClassVar[type[FrameNetGraphView]] = FrameNetGraphView
