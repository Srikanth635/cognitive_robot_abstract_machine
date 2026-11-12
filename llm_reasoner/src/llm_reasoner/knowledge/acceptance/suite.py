"""
The acceptance suite of the rules the ontology itself states.
"""

from __future__ import annotations

from llm_reasoner.knowledge.acceptance.derived import RangesHold, RestrictionsHold
from llm_reasoner.knowledge.acceptance.rules import AcceptanceSuite
from llm_reasoner.knowledge.acceptance.shapes import ShapesHold


def ontology_rules_suite() -> AcceptanceSuite:
    """
    :return: The acceptance rules the ontology states: its restrictions and ranges, and
        the structural rules of its shapes.
    """
    return AcceptanceSuite(rules=[RestrictionsHold(), RangesHold(), ShapesHold()])
