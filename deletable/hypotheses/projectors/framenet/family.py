"""FrameNet hypothesis family descriptor."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from llmr.hypotheses.families import HypothesisFamily, hypothesis_family
from llmr.hypotheses.projectors.framenet.constants import FRAMENET_REASONER_NAME
from llmr.hypotheses.projectors.framenet.projector import FrameNetProjector
from llmr.hypotheses.projectors.framenet.view import FrameNetGraphView


@hypothesis_family(reasoner=FRAMENET_REASONER_NAME)
@dataclass(frozen=True)
class FrameNetFamily(HypothesisFamily[FrameNetProjector, FrameNetGraphView]):
    """Family descriptor for FrameNet projections and queries."""

    PROJECTOR_TYPE: ClassVar[type[FrameNetProjector]] = FrameNetProjector
    VIEW_TYPE: ClassVar[type[FrameNetGraphView]] = FrameNetGraphView
