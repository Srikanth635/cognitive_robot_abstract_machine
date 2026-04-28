"""Flanagan hypothesis family descriptor."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from llmr.hypotheses.families import HypothesisFamily
from llmr.hypotheses.projectors.flanagan.constants import FLANAGAN_REASONER_NAME
from llmr.hypotheses.projectors.flanagan.projector import FlanaganProjector
from llmr.hypotheses.projectors.flanagan.view import FlanaganGraphView


@dataclass(frozen=True)
class FlanaganFamily(HypothesisFamily[FlanaganProjector, FlanaganGraphView]):
    """Family descriptor for Flanagan projections and queries."""

    REASONER_NAME: ClassVar[str] = FLANAGAN_REASONER_NAME
    PROJECTOR_TYPE: ClassVar[type[FlanaganProjector]] = FlanaganProjector
    VIEW_TYPE: ClassVar[type[FlanaganGraphView]] = FlanaganGraphView
