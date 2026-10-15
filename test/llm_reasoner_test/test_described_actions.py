"""
Tests of which knowledge describes which action designator: a mapping kept as data, by
identifiers, so that llm_reasoner imports no designator package.
"""

from __future__ import annotations

from llm_reasoner.knowledge.ontologies import CRAM_AICOR_NAMESPACE
from llm_reasoner.knowledge.ontologies.described_actions import (
    DescribedAction,
    DescribedActions,
)
from llm_reasoner.knowledge.ontologies.action_perspectives import action_core
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

    def test_a_mapping_given_explicitly_replaces_the_one_kept_as_data(self):
        described = DescribedAction(
            designator="some.module.DescribedAction",
            concept=ConceptReference(iri=CRAM_AICOR_NAMESPACE + "Pouring"),
            structure=pouring.PouringPhaseStructure,
        )
        actions = DescribedActions(entries=[described])
        assert actions.describing(described.designator) is described
        assert actions.describing(POURING_DESIGNATOR) is None


class TestStructureQuery:
    def test_the_query_is_for_the_structure_decomposing_the_described_concept(self):
        described = DescribedActions().describing(POURING_DESIGNATOR)
        query = described.query({})
        assert query.type is described.structure
        assert query.construct_instance().decomposes_task == described.concept

    def test_the_query_fixes_the_values_given(self):
        described = DescribedActions().describing(POURING_DESIGNATOR)
        structure = described.query(
            {"has_material_regime": pouring.MaterialRegime.GRANULAR}
        ).construct_instance()
        assert structure.has_material_regime is pouring.MaterialRegime.GRANULAR


class TestFrameQuery:
    def test_the_pouring_designator_is_framed_by_the_cores_pouring_frame(self):
        described = DescribedActions().describing(POURING_DESIGNATOR)
        assert described.frame is action_core.PouringFrame

    def test_the_frame_query_is_for_the_frame_framing_the_described_concept(self):
        described = DescribedActions().describing(POURING_DESIGNATOR)
        query = described.frame_query()
        assert query.type is described.frame
        assert query.construct_instance().frames_action == described.concept

    def test_an_entry_without_a_frame_has_none(self):
        described = DescribedAction(
            designator="some.module.DescribedAction",
            concept=ConceptReference(iri=CRAM_AICOR_NAMESPACE + "Pouring"),
            structure=pouring.PouringPhaseStructure,
        )
        assert described.frame is None
