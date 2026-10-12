"""
Tests of which knowledge describes which action designator: a mapping kept as data, by
identifiers, so that llm_reasoner imports no designator package.
"""

from __future__ import annotations

from llm_reasoner.knowledge.ontologies import CRAM_AICOR_NAMESPACE
from llm_reasoner.knowledge.ontologies.described_actions import DescribedActions
from llm_reasoner.knowledge.ontologies.predictive_manipulation import pouring
from llm_reasoner.knowledge.ontologies.terms import ConceptReference, DescriptionEntity

POURING_DESIGNATOR = "coraplex.robot_plans.actions.composite.tool_based.PouringAction"
"""
The identifier of the action designator the pouring knowledge describes.
"""


class TestDescribedActions:
    def test_the_pouring_designator_is_described_by_the_pouring_structure(self):
        described = DescribedActions().describing(POURING_DESIGNATOR)
        assert described.structure is pouring.PouringPhaseStructure
        assert described.concept == ConceptReference(
            iri=CRAM_AICOR_NAMESPACE + "Pouring"
        )

    def test_every_entry_names_a_structure_of_the_vocabulary(self):
        for described in DescribedActions().entries:
            assert issubclass(described.structure, DescriptionEntity)

    def test_a_designator_is_described_by_its_own_entry(self):
        actions = DescribedActions()
        for described in actions.entries:
            assert actions.describing(described.designator) is described

    def test_a_designator_no_knowledge_describes_is_described_by_nothing(self):
        assert DescribedActions().describing("some.module.UndescribedAction") is None
