"""
The acceptance suite of the predictive manipulation microworld.
"""

from __future__ import annotations

from llm_reasoner.acceptance import core, pouring
from llm_reasoner.acceptance.derived import RangesHold, RestrictionsHold
from llm_reasoner.acceptance.rules import AcceptanceSuite
from llm_reasoner.acceptance.shapes import ShapesHold


def predictive_manipulation_suite() -> AcceptanceSuite:
    """
    :return: Every acceptance rule of the microworld: the restrictions and ranges the
        ontology states, the structural rules its shapes state, then the rules still
        written in Python.
    """
    return AcceptanceSuite(
        rules=[
            RestrictionsHold(),
            RangesHold(),
            ShapesHold(),
            core.MatchHasNoResponse(),
            core.AbsentEventHasNoObservation(),
            core.ErrorHasObservation(),
            core.ComponentsDeclared(),
            pouring.PouringEventsSpecific(),
            pouring.PouringPhasesOrdered(),
            pouring.MaterialFitsFlowDynamics(),
            pouring.TerminationReasonHasEvent(),
        ]
    )
