"""
Tests for the static action knowledge module.
"""

import pytest

from cram_viz.action_knowledge import (
    ACTION_KNOWLEDGE,
    ACTION_PHASE_KNOWLEDGE,
    ActionKnowledge,
    ActionKnowledgeGraphFragment,
    ActionPhase,
    ActionPhaseGraphFragment,
    ActionPhaseSequence,
    Concept,
    Goal,
    KnowledgeRelation,
    OntologyClass,
    Parameter,
    Precondition,
    Role,
    category_node_id,
    class_node_id,
    concept_node_id,
    member_node_id,
    phase_node_id,
)


# %% graph node id builders — the single source of truth kb.py's highlighting relies on
class TestGraphNodeIdBuilders:
    def test_phase_node_id_is_shared_by_name_only(self):
        assert phase_node_id(ActionPhase(name="Reach", order=1)) == "phase:Reach"

    def test_concept_node_id_is_shared_across_verbs(self):
        assert concept_node_id(Concept("CollisionFreeMotion")) == (
            "concept:CollisionFreeMotion"
        )

    def test_class_node_id_is_shared_across_owners(self):
        assert class_node_id(OntologyClass("PhysicalObject")) == (
            "class:PhysicalObject"
        )

    def test_category_node_id_is_namespaced_by_owner(self):
        assert category_node_id("roles", "transport") == "roles:transport"

    def test_member_node_id_is_shared_by_category_and_name_only(self):
        assert member_node_id("roles", "agent: pr2") == "roles:agent: pr2"


# %% the shipped phase registry
class TestActionPhaseKnowledgeRegistry:
    def test_transport_verb_is_registered(self):
        sequence = ACTION_PHASE_KNOWLEDGE["transport"]
        assert [phase.name for phase in sequence.phases] == [
            "Reach",
            "Grasp",
            "Lift",
            "Carry",
            "Place",
        ]

    def test_transport_phases_maintain_and_target_expected_concepts(self):
        # StableGrasp is established in Grasp and must still be *maintained* through
        # Lift and Carry (not re-targeted) -- a phase's own outcome becomes the next
        # phases' constraint, which is exactly what a shared Concept node models
        sequence = ACTION_PHASE_KNOWLEDGE["transport"]
        fragment = ActionPhaseGraphFragment(sequence)
        _, edges = fragment.nodes_and_edges()

        maintained_by = {
            edge.source
            for edge in edges
            if edge.relation is KnowledgeRelation.MAINTAINS
            and edge.target == "concept:StableGrasp"
        }
        assert maintained_by == {"phase:Lift", "phase:Carry"}

    def test_pickup_phases_end_at_lift(self):
        # pickup is the shorter Reach -> Grasp -> Lift prefix of transport's chain --
        # it stops once the object is grasped and clear of its support, matching
        # ACTION_KNOWLEDGE["pickup"]'s own goals
        sequence = ACTION_PHASE_KNOWLEDGE["pickup"]
        assert [phase.name for phase in sequence.phases] == ["Reach", "Grasp", "Lift"]

        fragment = ActionPhaseGraphFragment(sequence)
        nodes, _ = fragment.nodes_and_edges()
        node_ids = {node.identifier for node in nodes}
        assert {"phase:Reach", "phase:Grasp", "phase:Lift"} <= node_ids

    def test_pickup_and_transport_share_the_same_concept_nodes(self):
        # both verbs' Reach/Grasp/Lift phases reference the SAME globally shared
        # concepts (see concept_node_id) -- pickup doesn't get its own copies
        pickup_fragment = ActionPhaseGraphFragment(ACTION_PHASE_KNOWLEDGE["pickup"])
        transport_fragment = ActionPhaseGraphFragment(
            ACTION_PHASE_KNOWLEDGE["transport"]
        )
        pickup_nodes, _ = pickup_fragment.nodes_and_edges()
        transport_nodes, _ = transport_fragment.nodes_and_edges()

        pickup_concept_ids = {
            n.identifier for n in pickup_nodes if n.identifier.startswith("concept:")
        }
        transport_concept_ids = {
            n.identifier for n in transport_nodes if n.identifier.startswith("concept:")
        }
        assert pickup_concept_ids == {
            "concept:CollisionFreeMotion",
            "concept:PreGraspPose",
            "concept:StableGrasp",
            "concept:ClearOfSupport",
        }
        assert pickup_concept_ids <= transport_concept_ids

    def test_carry_phase_graph_node_is_shared_between_transport_and_place(self):
        # transport's Carry (order 4) and place's Carry (order 1) are two DISTINCT
        # ActionPhase instances (see test_phase_names_repeat_across_verbs below), but
        # they must still render to the very same graph node so that selecting either
        # verb highlights it
        transport_fragment = ActionPhaseGraphFragment(
            ACTION_PHASE_KNOWLEDGE["transport"]
        )
        place_fragment = ActionPhaseGraphFragment(ACTION_PHASE_KNOWLEDGE["place"])
        transport_nodes, _ = transport_fragment.nodes_and_edges()
        place_nodes, _ = place_fragment.nodes_and_edges()

        transport_carry_id = next(
            n.identifier for n in transport_nodes if n.label == "Carry"
        )
        place_carry_id = next(n.identifier for n in place_nodes if n.label == "Carry")
        assert transport_carry_id == place_carry_id == "phase:Carry"

    def test_transport_matches_real_episode_names(self):
        # coraplex's TransportAction produces episode names like these (see
        # cram_viz.onboard.demo.derive_segments)
        sequence = ACTION_PHASE_KNOWLEDGE["transport"]
        assert sequence.matches("transport_milk")
        assert sequence.matches("transport_bowl")
        assert not sequence.matches("prepare")

    def test_place_is_the_carry_and_place_suffix_of_transports_chain(self):
        # place picks up where pickup leaves off -- pickup + place together cover
        # exactly the same ground as transport's full Reach..Place chain
        sequence = ACTION_PHASE_KNOWLEDGE["place"]
        assert [phase.name for phase in sequence.phases] == ["Carry", "Place"]

    def test_phase_names_repeat_across_verbs_action_verb_disambiguates(self):
        # Carry exists in both transport and place -- two DISTINCT ActionPhase
        # instances (different order), only distinguishable by their auto-stamped
        # action_verb, even though they share one graph node (see above)
        transport_carry = next(
            p for p in ACTION_PHASE_KNOWLEDGE["transport"].phases if p.name == "Carry"
        )
        place_carry = next(
            p for p in ACTION_PHASE_KNOWLEDGE["place"].phases if p.name == "Carry"
        )
        assert transport_carry.action_verb == "transport"
        assert place_carry.action_verb == "place"
        assert transport_carry != place_carry


# %% owner back-reference auto-stamping (phases)
class TestActionPhaseSequencePostInit:
    def test_bare_phase_has_no_owner_by_default(self):
        assert ActionPhase(name="Reach", order=1).action_verb == ""

    def test_construction_stamps_the_owning_verb_onto_every_phase(self):
        sequence = ActionPhaseSequence(
            name="wave",
            phases=(
                ActionPhase(name="Raise", order=1),
                ActionPhase(name="Shake", order=2),
            ),
        )
        assert [phase.action_verb for phase in sequence.phases] == ["wave", "wave"]


# %% matching episodes to a sequence's verb
class TestActionPhaseSequenceMatching:
    def test_matches_exact_episode_name(self):
        sequence = ActionPhaseSequence(name="pickup", phases=())
        assert sequence.matches("pickup")

    def test_matches_prefixed_episode_name(self):
        sequence = ActionPhaseSequence(name="pickup", phases=())
        assert sequence.matches("pickup_milk")

    def test_does_not_match_unrelated_episode_name(self):
        sequence = ActionPhaseSequence(name="pickup", phases=())
        assert not sequence.matches("transport_milk")

    def test_does_not_match_verb_as_bare_substring(self):
        sequence = ActionPhaseSequence(name="pickup", phases=())
        assert not sequence.matches("prepickup")
        assert not sequence.matches("pickupstuff")


# %% rendering a sequence into graph nodes/edges
class TestActionPhaseGraphFragment:
    @staticmethod
    def _one_phase_sequence():
        return ActionPhaseSequence(
            name="pickup",
            phases=(
                ActionPhase(
                    name="Reach",
                    order=1,
                    maintains=(Concept("CollisionFreeMotion"),),
                    targets=(Concept("PreGraspPose"),),
                ),
            ),
        )

    def test_anchor_edges_link_directly_to_every_phase(self):
        fragment = ActionPhaseGraphFragment(self._one_phase_sequence())
        edges = fragment.anchor_edges("pickup")
        assert len(edges) == 1
        assert edges[0].source == "pickup"
        assert edges[0].target == "phase:Reach"
        assert edges[0].relation is KnowledgeRelation.HAS_PHASE

    def test_anchor_edges_cover_every_phase_in_the_sequence(self):
        sequence = ActionPhaseSequence(
            name="pickup",
            phases=(
                ActionPhase(name="Reach", order=1),
                ActionPhase(name="Grasp", order=2),
            ),
        )
        fragment = ActionPhaseGraphFragment(sequence)
        edges = fragment.anchor_edges("pickup")
        assert {edge.target for edge in edges} == {"phase:Reach", "phase:Grasp"}
        assert all(edge.source == "pickup" for edge in edges)
        assert all(edge.relation is KnowledgeRelation.HAS_PHASE for edge in edges)

    def test_phase_node_is_included(self):
        fragment = ActionPhaseGraphFragment(self._one_phase_sequence())
        nodes, _ = fragment.nodes_and_edges()
        assert any(
            node.identifier == "phase:Reach" and node.label == "Reach" for node in nodes
        )

    def test_phase_maintains_and_targets_concepts(self):
        fragment = ActionPhaseGraphFragment(self._one_phase_sequence())
        nodes, edges = fragment.nodes_and_edges()
        node_ids = {node.identifier for node in nodes}
        assert {"concept:CollisionFreeMotion", "concept:PreGraspPose"} <= node_ids

        maintains = next(
            edge for edge in edges if edge.relation is KnowledgeRelation.MAINTAINS
        )
        assert maintains.source == "phase:Reach"
        assert maintains.target == "concept:CollisionFreeMotion"

        targets = next(
            edge for edge in edges if edge.relation is KnowledgeRelation.TARGETS
        )
        assert targets.source == "phase:Reach"
        assert targets.target == "concept:PreGraspPose"

    def test_phase_nodes_are_emitted_in_order_field_order_regardless_of_tuple_order(
        self,
    ):
        sequence = ActionPhaseSequence(
            name="pickup",
            phases=(
                ActionPhase(name="Grasp", order=2),
                ActionPhase(name="Reach", order=1),
            ),
        )
        fragment = ActionPhaseGraphFragment(sequence)
        nodes, _ = fragment.nodes_and_edges()
        phase_labels = [n.label for n in nodes if n.identifier.startswith("phase:")]
        assert phase_labels == ["Reach", "Grasp"]


# %% the shipped action-knowledge registry
class TestActionKnowledgeRegistry:
    def test_transport_verb_is_registered(self):
        knowledge = ACTION_KNOWLEDGE["transport"]
        assert [c.name for c in knowledge.classes] == [
            "Action",
            "ManipulationAction",
            "TransportAction",
        ]
        assert knowledge.roles and knowledge.parameters
        assert knowledge.preconditions and knowledge.goals
        assert all(role.action_verb == "transport" for role in knowledge.roles)
        assert all(p.action_verb == "transport" for p in knowledge.parameters)
        assert all(p.action_verb == "transport" for p in knowledge.preconditions)
        assert all(g.action_verb == "transport" for g in knowledge.goals)

    def test_transport_matches_real_episode_names(self):
        knowledge = ACTION_KNOWLEDGE["transport"]
        assert knowledge.matches("transport_milk")
        assert not knowledge.matches("prepare")

    def test_place_verb_is_registered(self):
        knowledge = ACTION_KNOWLEDGE["place"]
        assert [c.name for c in knowledge.classes] == [
            "Action",
            "ManipulationAction",
            "PlaceAction",
        ]
        assert knowledge.roles and knowledge.parameters
        assert knowledge.preconditions and knowledge.goals

    def test_place_preconditions_match_pickups_goals(self):
        # the pickup -> place pipeline: what pickup promises to achieve is exactly
        # what place requires to already hold true
        pickup_goals = {goal.name for goal in ACTION_KNOWLEDGE["pickup"].goals}
        place_preconditions = {
            precondition.name
            for precondition in ACTION_KNOWLEDGE["place"].preconditions
        }
        assert "object is grasped" in pickup_goals
        assert "object is grasped" in place_preconditions

    def test_place_and_transport_share_the_action_and_manipulation_classes(self):
        place_class_ids = {class_node_id(c) for c in ACTION_KNOWLEDGE["place"].classes}
        transport_class_ids = {
            class_node_id(c) for c in ACTION_KNOWLEDGE["transport"].classes
        }
        assert {"class:Action", "class:ManipulationAction"} <= place_class_ids
        assert place_class_ids & transport_class_ids == {
            "class:Action",
            "class:ManipulationAction",
        }

    def test_agent_role_graph_node_is_shared_across_every_verb_that_has_one(self):
        # 'agent: pr2' is registered on transport, pickup and place independently --
        # all three must point at the very same graph node, not a private copy each
        role_ids = {
            next(
                n.identifier
                for n in ActionKnowledgeGraphFragment(
                    ACTION_KNOWLEDGE[verb]
                ).nodes_and_edges()[0]
                if n.label == "agent: pr2"
            )
            for verb in ("transport", "pickup", "place")
        }
        assert role_ids == {"roles:agent: pr2"}


# %% rendering action knowledge into graph nodes/edges
class TestActionKnowledgeGraphFragment:
    @staticmethod
    def _knowledge():
        return ActionKnowledge(
            name="transport",
            classes=(OntologyClass("Action"), OntologyClass("TransportAction")),
            roles=(Role("agent: pr2"),),
            parameters=(Parameter("arm: left"),),
            preconditions=(Precondition("gripper is empty"),),
            goals=(Goal("object is at the destination"),),
        )

    def test_anchor_edges_link_directly_to_classification_root_and_every_member(self):
        fragment = ActionKnowledgeGraphFragment(self._knowledge())
        edges = fragment.anchor_edges("transport")
        targets = {edge.target for edge in edges}
        assert targets == {
            "classes:transport",
            "roles:agent: pr2",
            "parameters:arm: left",
            "preconditions:gripper is empty",
            "goals:object is at the destination",
        }
        assert all(edge.source == "transport" for edge in edges)
        relation_by_target = {edge.target: edge.relation for edge in edges}
        assert (
            relation_by_target["classes:transport"] is KnowledgeRelation.HAS_KNOWLEDGE
        )
        assert relation_by_target["roles:agent: pr2"] is KnowledgeRelation.HAS_ROLE
        assert (
            relation_by_target["parameters:arm: left"] is KnowledgeRelation.RESOLVED_TO
        )
        assert (
            relation_by_target["preconditions:gripper is empty"]
            is KnowledgeRelation.REQUIRES
        )
        assert (
            relation_by_target["goals:object is at the destination"]
            is KnowledgeRelation.EXPECTS
        )

    def test_anchor_edges_skip_empty_categories(self):
        knowledge = ActionKnowledge(name="pickup", roles=(Role("agent: pr2"),))
        fragment = ActionKnowledgeGraphFragment(knowledge)
        edges = fragment.anchor_edges("pickup")
        assert {edge.target for edge in edges} == {"roles:agent: pr2"}

    def test_class_chain_is_classified_then_specialized(self):
        fragment = ActionKnowledgeGraphFragment(self._knowledge())
        nodes, edges = fragment.nodes_and_edges()
        node_ids = {node.identifier for node in nodes}
        assert {"class:Action", "class:TransportAction"} <= node_ids

        classified = next(
            edge for edge in edges if edge.relation is KnowledgeRelation.CLASSIFIED_AS
        )
        assert classified.source == "classes:transport"
        assert classified.target == "class:Action"

        specialized = next(
            edge for edge in edges if edge.relation is KnowledgeRelation.SPECIALIZED_AS
        )
        assert specialized.source == "class:Action"
        assert specialized.target == "class:TransportAction"

    def test_role_node_is_included_and_directly_linked(self):
        fragment = ActionKnowledgeGraphFragment(self._knowledge())
        nodes, _ = fragment.nodes_and_edges()
        node_ids = {node.identifier for node in nodes}
        assert "roles:agent: pr2" in node_ids

        role_edge = next(
            edge
            for edge in fragment.anchor_edges("transport")
            if edge.relation is KnowledgeRelation.HAS_ROLE
        )
        assert role_edge.source == "transport"
        assert role_edge.target == "roles:agent: pr2"

    def test_parameters_preconditions_and_goals_use_their_own_relations(self):
        fragment = ActionKnowledgeGraphFragment(self._knowledge())
        edges = fragment.anchor_edges("transport")
        relations = {edge.relation for edge in edges}
        assert KnowledgeRelation.RESOLVED_TO in relations
        assert KnowledgeRelation.REQUIRES in relations
        assert KnowledgeRelation.EXPECTS in relations


# %% owner back-reference auto-stamping
class TestActionKnowledgePostInit:
    def test_bare_role_has_no_owner_by_default(self):
        assert Role("agent: pr2").action_verb == ""

    def test_construction_stamps_the_owning_verb_onto_every_member(self):
        knowledge = ActionKnowledge(
            name="pickup",
            roles=(Role("agent: pr2"),),
            parameters=(Parameter("arm: left"),),
            preconditions=(Precondition("gripper is empty"),),
            goals=(Goal("object is at destination"),),
        )
        assert knowledge.roles[0].action_verb == "pickup"
        assert knowledge.parameters[0].action_verb == "pickup"
        assert knowledge.preconditions[0].action_verb == "pickup"
        assert knowledge.goals[0].action_verb == "pickup"

    def test_action_verb_is_not_a_constructor_argument(self):
        with pytest.raises(TypeError):
            Role("agent: pr2", "transport")
