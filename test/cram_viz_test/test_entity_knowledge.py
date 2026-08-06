"""
Tests for the static entity knowledge module.
"""

import pytest

from cram_viz.action_knowledge import KnowledgeRelation, OntologyClass
from cram_viz.entity_knowledge import (
    ENTITY_KNOWLEDGE,
    Affordance,
    EntityKnowledge,
    EntityKnowledgeGraphFragment,
    EntityRelation,
    Property,
    Quality,
)


# %% the shipped registry
class TestEntityKnowledgeRegistry:
    def test_milk_is_registered(self):
        knowledge = ENTITY_KNOWLEDGE["milk"]
        assert [c.name for c in knowledge.classes] == [
            "Entity",
            "PhysicalObject",
            "Container",
        ]
        assert knowledge.properties and knowledge.qualities
        assert knowledge.affordances and knowledge.relations
        assert all(p.entity_name == "milk" for p in knowledge.properties)
        assert all(q.entity_name == "milk" for q in knowledge.qualities)
        assert all(a.entity_name == "milk" for a in knowledge.affordances)
        assert all(r.entity_name == "milk" for r in knowledge.relations)

    def test_milk_affordances_and_qualities(self):
        knowledge = ENTITY_KNOWLEDGE["milk"]
        assert [a.name for a in knowledge.affordances] == [
            "GraspAffordance",
            "PourableAffordance",
            "ContainmentAffordance",
            "PlaceableAffordance",
            "OpenableAffordance",
        ]
        quality_names = [q.name for q in knowledge.qualities]
        assert len(quality_names) == 4
        assert any(name.startswith("color:") for name in quality_names)
        assert any(name.startswith("temperature:") for name in quality_names)

    def test_milk_properties_and_relations(self):
        knowledge = ENTITY_KNOWLEDGE["milk"]
        property_names = [p.name for p in knowledge.properties]
        assert len(property_names) == 4
        assert any(name.startswith("mass:") for name in property_names)
        assert any(name.startswith("size:") for name in property_names)

        relation_names = [r.name for r in knowledge.relations]
        assert "patient_of: PickUpAction" in relation_names
        assert "patient_of: TransportAction" in relation_names
        assert any(name.startswith("supported_by:") for name in relation_names)
        assert any(name.startswith("contains:") for name in relation_names)

    def test_milk_matches_its_own_object_name_only(self):
        knowledge = ENTITY_KNOWLEDGE["milk"]
        assert knowledge.matches("milk")
        assert not knowledge.matches("bowl")
        assert not knowledge.matches("milk_carton")  # exact match only, unlike verbs


# %% rendering entity knowledge into graph nodes/edges
class TestEntityKnowledgeGraphFragment:
    @staticmethod
    def _knowledge():
        return EntityKnowledge(
            name="milk",
            classes=(OntologyClass("Entity"), OntologyClass("Container")),
            properties=(Property("shape: rectangular carton"),),
            qualities=(Quality("material: cardboard"),),
            affordances=(Affordance("GraspAffordance"),),
            relations=(EntityRelation("patient_of: TransportAction"),),
        )

    def test_anchor_edges_link_directly_to_classification_root_and_every_member(self):
        fragment = EntityKnowledgeGraphFragment(self._knowledge())
        edges = fragment.anchor_edges("milk")
        targets = {edge.target for edge in edges}
        assert targets == {
            "classes:milk",
            "properties:shape: rectangular carton",
            "qualities:material: cardboard",
            "affordances:GraspAffordance",
            "relations:patient_of: TransportAction",
        }
        assert all(edge.source == "milk" for edge in edges)
        relation_by_target = {edge.target: edge.relation for edge in edges}
        assert relation_by_target["classes:milk"] is KnowledgeRelation.HAS_KNOWLEDGE
        assert (
            relation_by_target["properties:shape: rectangular carton"]
            is KnowledgeRelation.OBSERVED
        )
        assert (
            relation_by_target["qualities:material: cardboard"]
            is KnowledgeRelation.HAS_QUALITY
        )
        assert (
            relation_by_target["affordances:GraspAffordance"]
            is KnowledgeRelation.AFFORDS
        )
        assert (
            relation_by_target["relations:patient_of: TransportAction"]
            is KnowledgeRelation.RELATED_BY
        )

    def test_anchor_edges_skip_empty_categories(self):
        knowledge = EntityKnowledge(
            name="bowl", affordances=(Affordance("ContainmentAffordance"),)
        )
        fragment = EntityKnowledgeGraphFragment(knowledge)
        edges = fragment.anchor_edges("bowl")
        assert {edge.target for edge in edges} == {"affordances:ContainmentAffordance"}

    def test_affordance_node_is_shared_between_two_different_entities(self):
        # two entities that happen to register the same affordance point at the
        # very same graph node, not a private copy each
        milk = EntityKnowledgeGraphFragment(
            EntityKnowledge(name="milk", affordances=(Affordance("GraspAffordance"),))
        )
        bowl = EntityKnowledgeGraphFragment(
            EntityKnowledge(name="bowl", affordances=(Affordance("GraspAffordance"),))
        )
        milk_id = next(
            n.identifier
            for n in milk.nodes_and_edges()[0]
            if n.label == "GraspAffordance"
        )
        bowl_id = next(
            n.identifier
            for n in bowl.nodes_and_edges()[0]
            if n.label == "GraspAffordance"
        )
        assert milk_id == bowl_id == "affordances:GraspAffordance"

    def test_class_chain_is_classified_then_specialized(self):
        fragment = EntityKnowledgeGraphFragment(self._knowledge())
        nodes, edges = fragment.nodes_and_edges()
        node_ids = {node.identifier for node in nodes}
        assert {"class:Entity", "class:Container"} <= node_ids

        classified = next(
            edge for edge in edges if edge.relation is KnowledgeRelation.CLASSIFIED_AS
        )
        assert classified.source == "classes:milk"
        assert classified.target == "class:Entity"

        specialized = next(
            edge for edge in edges if edge.relation is KnowledgeRelation.SPECIALIZED_AS
        )
        assert specialized.source == "class:Entity"
        assert specialized.target == "class:Container"

    def test_properties_qualities_affordances_relations_use_their_own_relations(self):
        fragment = EntityKnowledgeGraphFragment(self._knowledge())
        edges = fragment.anchor_edges("milk")
        relations = {edge.relation for edge in edges}
        assert KnowledgeRelation.OBSERVED in relations
        assert KnowledgeRelation.HAS_QUALITY in relations
        assert KnowledgeRelation.AFFORDS in relations
        assert KnowledgeRelation.RELATED_BY in relations

    def test_class_nodes_are_shared_with_action_knowledge(self):
        # Container/PhysicalObject etc. are global concepts (see class_node_id) —
        # rendering two entities that share a class must produce the SAME node id
        first = EntityKnowledgeGraphFragment(
            EntityKnowledge(name="milk", classes=(OntologyClass("Container"),))
        )
        second = EntityKnowledgeGraphFragment(
            EntityKnowledge(name="bowl", classes=(OntologyClass("Container"),))
        )
        first_nodes, _ = first.nodes_and_edges()
        second_nodes, _ = second.nodes_and_edges()
        first_class_id = next(
            n.identifier for n in first_nodes if n.label == "Container"
        )
        second_class_id = next(
            n.identifier for n in second_nodes if n.label == "Container"
        )
        assert first_class_id == second_class_id == "class:Container"


# %% owner back-reference auto-stamping
class TestEntityKnowledgePostInit:
    def test_bare_affordance_has_no_owner_by_default(self):
        assert Affordance("GraspAffordance").entity_name == ""

    def test_construction_stamps_the_owning_entity_onto_every_member(self):
        knowledge = EntityKnowledge(
            name="bowl",
            properties=(Property("shape: round bowl"),),
            qualities=(Quality("material: ceramic"),),
            affordances=(Affordance("ContainmentAffordance"),),
            relations=(EntityRelation("patient_of: TransportAction"),),
        )
        assert knowledge.properties[0].entity_name == "bowl"
        assert knowledge.qualities[0].entity_name == "bowl"
        assert knowledge.affordances[0].entity_name == "bowl"
        assert knowledge.relations[0].entity_name == "bowl"

    def test_entity_name_is_not_a_constructor_argument(self):
        with pytest.raises(TypeError):
            Affordance("GraspAffordance", "milk")
