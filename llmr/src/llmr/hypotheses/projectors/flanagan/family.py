"""Flanagan hypothesis family descriptor."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from llmr.hypotheses.families import HypothesisFamily, hypothesis_family
from llmr.hypotheses.projectors.flanagan.constants import FLANAGAN_REASONER_NAME
from llmr.hypotheses.projectors.flanagan.projector import FlanaganProjector
from llmr.hypotheses.projectors.flanagan.view import FlanaganGraphView


@hypothesis_family(reasoner=FLANAGAN_REASONER_NAME)
@dataclass(frozen=True)
class FlanaganFamily(HypothesisFamily[FlanaganProjector, FlanaganGraphView]):
    """Family descriptor for Flanagan projections and queries."""

    PROJECTOR_TYPE: ClassVar[type[FlanaganProjector]] = FlanaganProjector
    VIEW_TYPE: ClassVar[type[FlanaganGraphView]] = FlanaganGraphView
