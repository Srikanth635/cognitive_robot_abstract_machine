"""Typed view facades over sg_model repositories."""

from llmr_updated_arch.hypotheses.views.base import HypothesisGraphView, ReasonerGraphView
from llmr_updated_arch.hypotheses.views.flanagan import FlanaganGraphView
from llmr_updated_arch.hypotheses.views.framenet import FrameNetGraphView

__all__ = [
    "FlanaganGraphView",
    "FrameNetGraphView",
    "HypothesisGraphView",
    "ReasonerGraphView",
]
