"""
Tests for the scene-driven knowledge base and its graph-panel payloads.
"""

import copy
from dataclasses import replace
from types import SimpleNamespace

import pytest

krrood = pytest.importorskip("krrood", reason="EQL requires krrood")

from llm_reasoner.ontologies.predictive_manipulation import (  # noqa: E402
    core,
    pouring,
    prehension,
)

from cram_viz import kb  # noqa: E402  (importable once krrood is present)
from cram_viz import action_knowledge  # noqa: E402
from cram_viz import entity_knowledge  # noqa: E402
from cram_viz import hypothesis_knowledge  # noqa: E402


@pytest.fixture()
def fresh_kb(fixture_scene):
    kb.reset_kb()
    return kb.get_kb()


class TestKB:
    def test_scene_entities(self, fresh_kb):
        assert [o.name for o in fresh_kb.objects] == ["milk", "place_area"]
        assert fresh_kb.robot.name == "pr2"
        assert [a.side for a in fresh_kb.arms] == ["left"]
        assert fresh_kb.arms[0].gripper.name == "left_gripper"

    def test_episodes_link_objects(self, fresh_kb):
        transport = next(e for e in fresh_kb.episodes if e.name == "transport_milk")
        assert transport.picks is fresh_kb.objects[0]
        assert transport.places_at.name == "place_area"
        assert transport.performed_by.side == "left"

    def test_joint_motion_ranges(self, fresh_kb):
        torso = next(j for j in fresh_kb.joints if j.name == "torso_lift_joint")
        assert torso.min_rad == 0.0 and torso.max_rad == 0.3

    def test_architecture_scan(self, fresh_kb):
        names = {p.name for p in fresh_kb.packages}
        assert {"coraplex", "krrood"} <= names
        assert any(c.name == "Plan" for c in fresh_kb.classes)


class TestQueries:
    def test_entity_query(self, fixture_scene):
        result = kb.run_query("the(entity(obj).where(obj.name == 'milk'))")
        assert result["ok"] and result["count"] == 1
        assert result["rows"][0]["__entity__"] == "milk"
        assert "milk" in result["highlight"]

    def test_error_is_reported_not_raised(self, fixture_scene):
        with pytest.raises(Exception):
            kb.run_query("this is not python")

    def test_domain_knowledge_presets_return_real_content(self, fixture_scene):
        # EQL's .where() silently treats a nonexistent attribute access as "no
        # match" rather than raising, so a broken preset can look identical to a
        # correct one that legitimately found nothing (ok=True, count=0). These
        # presets query the real, always-populated ACTION_KNOWLEDGE/ENTITY_KNOWLEDGE
        # registries (unlike the architecture presets, which scan whatever repo
        # CRAM_VIZ_ARCHITECTURE points at and can legitimately be empty against the
        # fixture's tiny sample architecture, so they're excluded here).
        kb.reset_kb()
        for preset in kb._domain_knowledge_presets(kb.get_kb()):
            result = kb.run_query(preset["code"])
            assert result["ok"], "%s -> %s" % (preset["text"], result.get("error"))
            assert result["count"], "%s -> 0 rows (likely a broken field access)" % (
                preset["text"]
            )

    def test_domain_knowledge_presets_include_a_parameters_query(self, fixture_scene):
        kb.reset_kb()
        preset = next(
            p
            for p in kb._domain_knowledge_presets(kb.get_kb())
            if "parameterized" in p["text"]
        )
        result = kb.run_query(preset["code"])
        assert result["ok"] and result["count"]
        assert all(row["__type__"] == "Parameter" for row in result["rows"])

    def test_domain_knowledge_presets_include_an_agent_role_query(self, fixture_scene):
        kb.reset_kb()
        preset = next(
            p for p in kb._domain_knowledge_presets(kb.get_kb()) if "agent" in p["text"]
        )
        result = kb.run_query(preset["code"])
        assert result["ok"] and result["count"] == 1
        assert result["rows"][0]["__entity__"].startswith("agent:")

    def test_domain_knowledge_presets_include_the_pickup_place_pipeline_query(
        self, fixture_scene
    ):
        # this is a question only the action knowledge -- across two verbs at
        # once -- can answer: what pickup leaves true is exactly what place
        # already requires
        kb.reset_kb()
        preset = next(
            p
            for p in kb._domain_knowledge_presets(kb.get_kb())
            if "pickup" in p["text"] and "place" in p["text"]
        )
        result = kb.run_query(preset["code"])
        assert result["ok"] and result["count"]
        assert all(
            row["Goal.name"] == row["Precondition.name"] for row in result["rows"]
        )

    def test_scene_presets_no_longer_ask_which_arm_performed_each_episode(
        self, fixture_scene
    ):
        # every episode in this demo is performed by the same single arm, so the
        # question is never actually interesting -- dropped in favour of presets
        # the action knowledge itself can answer
        kb.reset_kb()
        texts = [p["text"] for p in kb.get_presets()]
        assert not any("which arm carries out" in text for text in texts)


# %% presets connecting recorded episodes to their action verb
class TestScenePresets:
    def test_no_presets_with_fewer_than_two_matching_episodes(self, fixture_scene):
        # fixture_scene has exactly one 'transport_milk' episode -- "every X in this
        # demo" / "each X" framing isn't useful for a single episode, so these
        # presets stay hidden rather than duplicate the plain scene presets above
        kb.reset_kb()
        assert kb._scene_verb_presets(kb.get_kb()) == []

    def test_presets_appear_and_query_correctly_with_multiple_matching_episodes(self):
        # a lightweight stand-in is enough: _scene_verb_presets only reads
        # episode.name and the verb, it never has to run a query itself
        fake_kb = SimpleNamespace(
            episodes=[
                SimpleNamespace(name="transport_milk"),
                SimpleNamespace(name="transport_bowl"),
                SimpleNamespace(name="prepare"),
            ],
            action_knowledge_entries=[SimpleNamespace(name="transport")],
            action_phase_sequences=[],
        )
        presets = kb._scene_verb_presets(fake_kb)
        assert [preset["text"] for preset in presets] == [
            "every 'transport' action in this demo",
            "what does each 'transport' move, and for how long?",
        ]
        assert all("transport_" in preset["code"] for preset in presets)

    def test_presets_query_correctly_against_the_real_demo(self, fixture_scene):
        # end-to-end against a scene shaped like this (fixture_scene itself only has
        # one transport episode, so this exercises the generator with a fake KB that
        # matches the real pr2_kitchen demo's episode names, then runs the generated
        # code against the real, single-episode fixture to confirm it's valid EQL)
        fake_kb = SimpleNamespace(
            episodes=[
                SimpleNamespace(name="transport_milk"),
                SimpleNamespace(name="transport_bowl"),
            ],
            action_knowledge_entries=[SimpleNamespace(name="transport")],
            action_phase_sequences=[],
        )
        kb.reset_kb()
        for preset in kb._scene_verb_presets(fake_kb):
            result = kb.run_query(preset["code"])
            assert result["ok"], "%s -> %s" % (preset["text"], result.get("error"))


class TestViewPayloads:
    def test_knowledge_view(self, fixture_scene):
        payload = kb.view_payload("knowledge")
        assert payload["ok"]
        ids = {n["id"] for n in payload["nodes"]}
        assert {"pr2", "milk", "transport_milk", "plan"} <= ids
        assert payload["presets"]

    def test_kinematics_view(self, fixture_scene):
        payload = kb.view_payload("kinematics")
        assert payload["ok"]
        ids = {n["id"] for n in payload["nodes"]}
        assert "urdf:base_link" in ids and "urdf:l_gripper_link" in ids
        # fixed joints render dashed ('type'), movable solid ('prop')
        kinds = {e["label"].split(" ")[0]: e["kind"] for e in payload["edges"]}
        assert kinds["torso_lift_joint"] == "prop"
        assert kinds["l_gripper_joint"] == "type"

    def test_plan_view_carries_status(self, fixture_scene):
        payload = kb.view_payload("plan")
        assert payload["ok"] and payload["layout"] == "hier"
        assert payload["live"] == "plan" and payload["statusLegend"]
        by_label = {n["label"]: n for n in payload["nodes"]}
        assert by_label["SequentialNode"]["status"] == "SUCCEEDED"
        # recorded inner nodes stay CREATED (only the root is performed)
        assert by_label["Transport"]["status"] == "CREATED"
        assert len(payload["edges"]) == len(payload["nodes"]) - 1

    def test_chart_view_is_live_only(self, fixture_scene):
        payload = kb.view_payload("chart")
        assert payload["ok"] and payload["nodes"] == []
        assert payload["live"] == "chart" and payload["empty"]

    def test_unknown_view(self, fixture_scene):
        payload = kb.view_payload("bogus")
        assert not payload["ok"]


# %% static action-phase knowledge merged onto matching episodes
class TestActionPhaseKnowledgeMerge:
    @staticmethod
    def _register_transport_sequence(monkeypatch):
        """
        Register phase knowledge for verb 'transport' -- matches the fixture scene's
        real 'transport_milk' episode, so the merge is exercised without touching the
        shared fixture data.
        """
        sequence = action_knowledge.ActionPhaseSequence(
            name="transport",
            phases=(
                action_knowledge.ActionPhase(
                    name="Carry",
                    order=1,
                    maintains=(action_knowledge.Concept("CollisionFreeMotion"),),
                    targets=(action_knowledge.Concept("PreGraspPose"),),
                ),
            ),
        )
        monkeypatch.setitem(
            action_knowledge.ACTION_PHASE_KNOWLEDGE, "transport", sequence
        )
        return sequence

    def test_shared_phase_node_is_not_duplicated_across_episodes(
        self, fixture_scene, monkeypatch
    ):
        self._register_transport_sequence(monkeypatch)
        kb.reset_kb()
        fresh = kb.get_kb()

        added_node_ids = []

        def add(node_id, label, group, lines):
            added_node_ids.append(node_id)
            details[node_id] = {"label": label, "group": group, "lines": lines}

        edges = []
        details = {}
        # two episodes sharing the 'transport' verb, as would happen with two
        # transport actions in one recording
        kb._merge_action_phase_knowledge(
            add, edges, details, "transport_milk", fresh.action_phase_sequences
        )
        kb._merge_action_phase_knowledge(
            add, edges, details, "transport_cup", fresh.action_phase_sequences
        )

        assert added_node_ids.count("phase:Carry") == 1
        has_phase_edges = [e for e in edges if e["label"] == "has_phase"]
        assert {e["from"] for e in has_phase_edges} == {
            "transport_milk",
            "transport_cup",
        }

    def test_concept_shared_by_two_different_verbs_is_not_duplicated(
        self, fixture_scene
    ):
        # the real registry: pickup's Reach and transport's Carry both maintain
        # CollisionFreeMotion — two DIFFERENT verbs sharing one global concept node
        kb.reset_kb()
        fresh = kb.get_kb()

        added_node_ids = []

        def add(node_id, label, group, lines):
            added_node_ids.append(node_id)
            details[node_id] = {"label": label, "group": group, "lines": lines}

        edges = []
        details = {}
        kb._merge_action_phase_knowledge(
            add, edges, details, "pickup_milk", fresh.action_phase_sequences
        )
        kb._merge_action_phase_knowledge(
            add, edges, details, "transport_milk", fresh.action_phase_sequences
        )

        assert added_node_ids.count("concept:CollisionFreeMotion") == 1


class TestActionPhaseKnowledgeQueries:
    def test_phase_entities_are_queryable(self, fixture_scene, monkeypatch):
        TestActionPhaseKnowledgeMerge._register_transport_sequence(monkeypatch)
        kb.reset_kb()

        result = kb.run_query("an(entity(phase))")
        assert result["ok"]
        assert any(row.get("__entity__") == "Carry" for row in result["rows"])

    def test_concept_entities_are_queryable(self, fixture_scene, monkeypatch):
        TestActionPhaseKnowledgeMerge._register_transport_sequence(monkeypatch)
        kb.reset_kb()

        result = kb.run_query("an(entity(concept))")
        assert result["ok"]
        assert any(
            row.get("__entity__") == "CollisionFreeMotion" for row in result["rows"]
        )

    # A query's "highlight" ids are what the graph panel zooms/glows onto
    # (Graph.highlight, gated on matching an actual graph node id) — a bare entity
    # name like "Carry" never matches, since phase/concept nodes are id-prefixed to
    # stay unique across verbs. These pin the fix: highlight must carry the real,
    # prefixed graph node id, not the bare entity name.
    def test_phase_query_highlights_the_real_graph_node(
        self, fixture_scene, monkeypatch
    ):
        TestActionPhaseKnowledgeMerge._register_transport_sequence(monkeypatch)
        kb.reset_kb()

        # phase names repeat across verbs (transport/place both have a "Carry"), so
        # action_verb is what disambiguates -- exactly what it exists for
        result = kb.run_query(
            "the(entity(phase).where(phase.name == 'Carry')"
            ".where(phase.action_verb == 'transport'))"
        )
        assert "phase:Carry" in result["highlight"]

    def test_concept_query_highlights_the_real_graph_node(
        self, fixture_scene, monkeypatch
    ):
        TestActionPhaseKnowledgeMerge._register_transport_sequence(monkeypatch)
        kb.reset_kb()

        result = kb.run_query(
            "the(entity(concept).where(concept.name == 'CollisionFreeMotion'))"
        )
        assert "concept:CollisionFreeMotion" in result["highlight"]

    def test_phase_sequence_query_highlights_the_real_graph_node(
        self, fixture_scene, monkeypatch
    ):
        # the sequence has no dedicated hub node any more -- its anchor in the
        # domain view is the verb's own bare node
        TestActionPhaseKnowledgeMerge._register_transport_sequence(monkeypatch)
        kb.reset_kb()

        result = kb.run_query(
            "the(entity(phase_sequence).where(phase_sequence.name == 'transport'))"
        )
        assert "transport" in result["highlight"]

    def test_phase_highlight_matches_an_actual_knowledge_graph_node(
        self, fixture_scene, monkeypatch
    ):
        # end-to-end: the id the EQL panel emits on entity:highlight must be one the
        # graph panel's inGraphSet (built from the Domain Knowledge tab's own nodes,
        # where phases live now) actually contains
        TestActionPhaseKnowledgeMerge._register_transport_sequence(monkeypatch)
        kb.reset_kb()

        result = kb.run_query("an(entity(phase))")
        graph_node_ids = {node["id"] for node in kb.view_payload("domain")["nodes"]}
        assert set(result["highlight"]) & graph_node_ids


# %% static action knowledge (classes/roles/parameters/preconditions/goals)
class TestActionKnowledgeMerge:
    def test_shared_role_node_is_not_duplicated_across_episodes(
        self, fixture_scene, monkeypatch
    ):
        knowledge = action_knowledge.ActionKnowledge(
            name="transport", roles=(action_knowledge.Role("agent: pr2"),)
        )
        monkeypatch.setitem(action_knowledge.ACTION_KNOWLEDGE, "transport", knowledge)
        kb.reset_kb()
        fresh = kb.get_kb()

        added_node_ids = []

        def add(node_id, label, group, lines):
            added_node_ids.append(node_id)
            details[node_id] = {"label": label, "group": group, "lines": lines}

        edges = []
        details = {}
        kb._merge_action_knowledge(
            add, edges, details, "transport_milk", fresh.action_knowledge_entries
        )
        kb._merge_action_knowledge(
            add, edges, details, "transport_cup", fresh.action_knowledge_entries
        )

        assert added_node_ids.count("roles:agent: pr2") == 1
        has_role_edges = [e for e in edges if e["label"] == "has_role"]
        assert {e["from"] for e in has_role_edges} == {
            "transport_milk",
            "transport_cup",
        }


class TestActionKnowledgeQueries:
    def test_role_query_highlights_the_real_graph_node(self, fixture_scene):
        kb.reset_kb()

        result = kb.run_query("an(entity(role))")
        assert result["ok"] and result["count"]
        graph_node_ids = {node["id"] for node in kb.view_payload("domain")["nodes"]}
        assert set(result["highlight"]) & graph_node_ids

    def test_ontology_class_query_highlights_the_real_graph_node(self, fixture_scene):
        kb.reset_kb()

        result = kb.run_query(
            "the(entity(oclass).where(oclass.name == 'TransportAction'))"
        )
        assert "class:TransportAction" in result["highlight"]

    def test_role_query_can_be_filtered_by_owning_action_verb(self, fixture_scene):
        kb.reset_kb()

        result = kb.run_query("an(entity(role).where(role.action_verb == 'transport'))")
        assert result["ok"] and result["count"]
        assert all(row["action_verb"] == "transport" for row in result["rows"])


# %% static entity knowledge (classes/properties/qualities/affordances/relations)
class TestEntityKnowledgeMerge:
    def test_shared_category_roots_are_not_duplicated_across_objects(
        self, fixture_scene, monkeypatch
    ):
        milk_knowledge = entity_knowledge.EntityKnowledge(
            name="milk",
            classes=(action_knowledge.OntologyClass("Container"),),
        )
        cup_knowledge = entity_knowledge.EntityKnowledge(
            name="cup",
            classes=(action_knowledge.OntologyClass("Container"),),
        )
        monkeypatch.setitem(entity_knowledge.ENTITY_KNOWLEDGE, "milk", milk_knowledge)
        monkeypatch.setitem(entity_knowledge.ENTITY_KNOWLEDGE, "cup", cup_knowledge)
        kb.reset_kb()
        fresh = kb.get_kb()

        added_node_ids = []

        def add(node_id, label, group, lines):
            added_node_ids.append(node_id)
            details[node_id] = {"label": label, "group": group, "lines": lines}

        edges = []
        details = {}
        kb._merge_entity_knowledge(
            add, edges, details, "milk", fresh.entity_knowledge_entries
        )
        kb._merge_entity_knowledge(
            add, edges, details, "cup", fresh.entity_knowledge_entries
        )

        # the Container CLASS node is shared globally (class_node_id has no owner
        # prefix), so it must be added once even though two different entities use it
        assert added_node_ids.count("class:Container") == 1
        classified_edges = [e for e in edges if e["label"] == "classified_as"]
        assert {e["from"] for e in classified_edges} == {"classes:milk", "classes:cup"}


class TestEntityKnowledgeQueries:
    def test_property_query_highlights_the_real_graph_node(self, fixture_scene):
        kb.reset_kb()

        result = kb.run_query("an(entity(property))")
        assert result["ok"] and result["count"]
        graph_node_ids = {node["id"] for node in kb.view_payload("domain")["nodes"]}
        assert set(result["highlight"]) & graph_node_ids

    def test_affordance_query_highlights_the_real_graph_node(self, fixture_scene):
        kb.reset_kb()

        result = kb.run_query(
            "the(entity(affordance).where(affordance.name == 'GraspAffordance'))"
        )
        assert "affordances:GraspAffordance" in result["highlight"]

    def test_affordance_query_can_be_filtered_by_owning_entity_name(
        self, fixture_scene
    ):
        kb.reset_kb()

        result = kb.run_query(
            "an(entity(affordance).where(affordance.entity_name == 'milk'))"
        )
        assert result["ok"] and result["count"]
        assert all(row["entity_name"] == "milk" for row in result["rows"])


# %% the Domain Knowledge tab (independent of any recorded scene)
class TestDomainKnowledgeView:
    #: node-id prefixes that only ever appear as part of the static action/entity
    #: knowledge ontology, never in the scene-driven Knowledge tab
    _DOMAIN_PREFIXES = (
        "phases:",
        "phase:",
        "concept:",
        "class:",
        "classes:",
        "roles:",
        "parameters:",
        "preconditions:",
        "goals:",
        "properties:",
        "qualities:",
        "affordances:",
        "relations:",
    )

    def test_contains_every_registered_verb_and_entity(self, fixture_scene):
        kb.reset_kb()
        payload = kb.view_payload("domain")
        assert payload["ok"]
        node_ids = {n["id"] for n in payload["nodes"]}
        # 'transport' has a real matching episode in this scene, so it anchors
        # there instead of a bare placeholder; 'pickup'/'place' have none, so they
        # still fall back to their own bare verb name
        assert {"transport_milk", "pickup", "place", "milk"} <= node_ids

    def test_transport_anchors_on_its_real_recorded_episode(self, fixture_scene):
        # 'transport' has a matching 'transport_milk' episode in this scene --
        # domain knowledge links directly to the REAL episode node instead of an
        # unconnected bare 'transport' placeholder
        kb.reset_kb()
        payload = kb.view_payload("domain")

        node_ids = [n["id"] for n in payload["nodes"]]
        assert {"phase:Reach", "roles:agent: pr2"} <= set(node_ids)
        assert node_ids.count("transport_milk") == 1
        assert "transport" not in node_ids

        labels_by_edge = {(e["from"], e["to"]): e["label"] for e in payload["edges"]}
        assert labels_by_edge[("transport_milk", "phase:Reach")] == "has_phase"
        assert labels_by_edge[("transport_milk", "roles:agent: pr2")] == "has_role"
        assert (
            labels_by_edge[("transport_milk", "classes:transport")] == "has_knowledge"
        )

    def test_transport_episode_anchor_carries_real_episode_detail_lines(
        self, fixture_scene
    ):
        # the anchor node accumulates BOTH kinds of information: the real recorded
        # facts (frames/duration/picks) a Knowledge-tab episode node would show,
        # and the domain-knowledge edges fanning out from it
        kb.reset_kb()
        fresh = kb.get_kb()
        payload = kb.view_payload("domain")

        episode = next(e for e in fresh.episodes if e.name == "transport_milk")
        assert payload["details"]["transport_milk"][
            "lines"
        ] == kb._episode_detail_lines(episode)

    def test_two_episodes_of_the_same_verb_both_link_to_the_same_shared_phase(
        self, fixture_scene
    ):
        # two recorded transports (e.g. milk and bowl) each get their own real
        # episode anchor, but both point directly at the very same shared phase
        # node -- exactly the cross-owner sharing this design exists for, now
        # exercised between two episodes of the same verb rather than two verbs
        kb.reset_kb()
        fresh = kb.get_kb()
        second_episode = replace(fresh.episodes[0], name="transport_bowl")
        scene_with_two_transports = copy.copy(fresh)
        scene_with_two_transports.episodes = fresh.episodes + [second_episode]

        payload = kb._domain_knowledge_view(scene_with_two_transports)

        node_ids = [n["id"] for n in payload["nodes"]]
        assert node_ids.count("phase:Reach") == 1
        assert node_ids.count("transport_milk") == 1
        assert node_ids.count("transport_bowl") == 1

        labels_by_edge = {(e["from"], e["to"]): e["label"] for e in payload["edges"]}
        assert labels_by_edge[("transport_milk", "phase:Reach")] == "has_phase"
        assert labels_by_edge[("transport_bowl", "phase:Reach")] == "has_phase"

    def test_pickup_merges_phases_and_action_knowledge_under_one_root(
        self, fixture_scene
    ):
        # 'pickup' is ALSO registered in both -- its own phases/roles link directly
        # to it, and it shares "Action"/"ManipulationAction" ontology-class nodes
        # with 'transport' but has its own "PickUpAction" leaf
        kb.reset_kb()
        payload = kb.view_payload("domain")

        node_ids = [n["id"] for n in payload["nodes"]]
        assert {"phase:Reach", "roles:agent: pr2", "class:PickUpAction"} <= set(
            node_ids
        )
        assert node_ids.count("pickup") == 1
        assert node_ids.count("class:Action") == 1  # shared with transport, not doubled

        labels_by_edge = {(e["from"], e["to"]): e["label"] for e in payload["edges"]}
        assert labels_by_edge[("pickup", "phase:Reach")] == "has_phase"
        assert labels_by_edge[("pickup", "roles:agent: pr2")] == "has_role"
        assert labels_by_edge[("classes:pickup", "class:Action")] == "classified_as"

    def test_partially_registered_verb_is_handled_without_error(
        self, fixture_scene, monkeypatch
    ):
        # a verb with an ACTION_PHASE_KNOWLEDGE entry but no ACTION_KNOWLEDGE one (or
        # vice versa) must not break the domain view -- pickup/transport both happen
        # to be fully registered today, so this is exercised with a dedicated mock
        # verb instead of depending on which real verb is currently partial
        sequence = action_knowledge.ActionPhaseSequence(
            name="wave",
            phases=(action_knowledge.ActionPhase(name="Wave", order=1),),
        )
        monkeypatch.setitem(action_knowledge.ACTION_PHASE_KNOWLEDGE, "wave", sequence)
        kb.reset_kb()

        payload = kb.view_payload("domain")

        node_ids = {n["id"] for n in payload["nodes"]}
        assert "phase:Wave" in node_ids
        edges_from_wave = [e for e in payload["edges"] if e["from"] == "wave"]
        assert {e["label"] for e in edges_from_wave} == {"has_phase"}

    def test_milk_entity_knowledge_is_anchored_on_the_real_bench_object(
        self, fixture_scene
    ):
        kb.reset_kb()
        fresh = kb.get_kb()
        payload = kb.view_payload("domain")
        milk = entity_knowledge.ENTITY_KNOWLEDGE["milk"]

        # the node is the REAL scene object, not an unconnected placeholder --
        # its detail lines carry the real BenchObject facts (kind/position/height),
        # exactly like the Knowledge tab's own 'milk' node
        milk_object = next(o for o in fresh.objects if o.name == "milk")
        assert payload["details"]["milk"]["lines"] == kb._bench_object_detail_lines(
            milk_object
        )

        labels_by_edge = {(e["from"], e["to"]): e["label"] for e in payload["edges"]}
        assert labels_by_edge[("milk", "classes:milk")] == "has_knowledge"
        assert (
            labels_by_edge[("milk", "properties:%s" % milk.properties[0].name)]
            == "observed"
        )
        assert (
            labels_by_edge[("milk", "qualities:%s" % milk.qualities[0].name)]
            == "has_quality"
        )
        assert (
            labels_by_edge[("milk", "affordances:%s" % milk.affordances[0].name)]
            == "affords"
        )
        assert (
            labels_by_edge[("milk", "relations:%s" % milk.relations[0].name)]
            == "related_by"
        )

    def test_knowledge_view_has_no_domain_knowledge_nodes(self, fixture_scene):
        # the whole point of the Domain Knowledge tab: the scene-driven Knowledge tab
        # goes back to how it looked before this feature existed
        kb.reset_kb()
        node_ids = {n["id"] for n in kb.view_payload("knowledge")["nodes"]}
        leaked = {
            node_id for node_id in node_ids if node_id.startswith(self._DOMAIN_PREFIXES)
        }
        assert not leaked

    def test_shared_members_are_not_duplicated_as_separate_nodes_per_owner(
        self, fixture_scene
    ):
        # pickup/transport/place share phase names (e.g. 'Grasp', 'Carry') and role
        # text (e.g. 'agent: pr2') -- each is rendered as a SINGLE graph node linked
        # directly to every owner that has one, not a private copy per owner
        kb.reset_kb()
        payload = kb.view_payload("domain")

        node_ids = [n["id"] for n in payload["nodes"]]
        assert node_ids.count("phase:Grasp") == 1
        assert node_ids.count("phase:Carry") == 1
        assert node_ids.count("roles:agent: pr2") == 1
        assert node_ids.count("goals:object is at the destination") == 1

    def test_clicking_either_owner_would_highlight_the_same_shared_phase(
        self, fixture_scene
    ):
        # the concrete scenario the shared-node design exists for: pickup and
        # transport both have a direct edge to the very same 'Grasp' phase node,
        # so the frontend's 1-hop click-spotlight lights it up for either one --
        # 'transport' anchors on its real 'transport_milk' episode in this scene,
        # 'pickup' has no matching episode so it keeps its bare verb anchor
        kb.reset_kb()
        payload = kb.view_payload("domain")

        neighbours_of = {
            owner: {edge["to"] for edge in payload["edges"] if edge["from"] == owner}
            | {edge["from"] for edge in payload["edges"] if edge["to"] == owner}
            for owner in ("pickup", "transport_milk")
        }
        assert "phase:Grasp" in neighbours_of["pickup"]
        assert "phase:Grasp" in neighbours_of["transport_milk"]


# %% highlighting owned items via their own field (no more knowledge_owner dict)
class TestOwnedHighlight:
    def test_entity_row_highlights_affordance_by_its_entity_name_field(self):
        affordance = entity_knowledge.Affordance("GraspAffordance")
        affordance.entity_name = "milk"
        highlight = []
        kb._entity_row(affordance, highlight)
        assert "affordances:GraspAffordance" in highlight

    def test_entity_row_highlights_role_by_its_action_verb_field(self):
        role = action_knowledge.Role("agent: pr2")
        role.action_verb = "transport"
        highlight = []
        kb._entity_row(role, highlight)
        assert "roles:agent: pr2" in highlight

    def test_entity_row_skips_highlight_when_owner_field_is_unstamped(self):
        role = action_knowledge.Role("agent: pr2")  # never owned by an ActionKnowledge
        highlight = []
        kb._entity_row(role, highlight)
        assert not any(h.startswith("roles:") for h in highlight)


# %% a knowledge fragment's row names only the fragment -- its relational fields belong
# to the Hypothesis Graph tab, not a flat EQL answer
@pytest.mark.usefixtures("sample_hypothesis_knowledge")
class TestKnowledgeFragmentRowIsNameOnly:
    def test_the_row_carries_only_the_entity_and_type_keys(self):
        phase = structure().defines_phase[0]
        assert set(kb._entity_row(phase, [])) == {"__entity__", "__type__"}

    def test_the_row_names_the_fragments_own_label(self):
        role = next(role for role in structure().defines_role if role.label)
        row = kb._entity_row(role, [])
        assert row["__entity__"] == role.label
        assert row["__type__"] == type(role).__name__

    def test_an_unlabelled_fragment_is_named_by_its_type(self):
        phase = next(phase for phase in structure().defines_phase if not phase.label)
        assert kb._entity_row(phase, [])["__entity__"] == type(phase).__name__


# %% a tuple field is content a query answer names, not metadata -- it renders as a
# real list, not a Python repr string
class TestJsonableTupleRendering:
    def test_a_tuple_of_strings_renders_as_a_list(self):
        assert kb._jsonable(("Graspable", "Liftable")) == ["Graspable", "Liftable"]

    def test_a_list_of_strings_renders_as_a_list(self):
        assert kb._jsonable(["Graspable", "Liftable"]) == ["Graspable", "Liftable"]

    def test_an_empty_tuple_renders_as_an_empty_list(self):
        assert kb._jsonable(()) == []


def structure():
    """
    Return the active hypothesis's phase structure.
    """
    return hypothesis_knowledge.active().view.root


def entity_names(fragments):
    """
    Return the row names of the fragments, sorted.
    """
    return sorted(kb._entity_name(fragment) for fragment in fragments)


def row_names(result):
    """
    Return the row names of a query result, sorted.
    """
    return sorted(row["__entity__"] for row in result["rows"])


def phase_of_type(phase_type):
    (phase,) = [
        phase for phase in structure().defines_phase if type(phase) is phase_type
    ]
    return phase


# %% the Hypothesis Graph tab
@pytest.mark.usefixtures("sample_hypothesis_knowledge")
class TestHypothesisGraphView:
    def test_payload_is_ok_with_the_hypothesis_graph_crumb(self, fixture_scene):
        kb.reset_kb()
        payload = kb.view_payload("hypothesis")
        assert payload["ok"]
        assert payload["crumb"] == "hypothesis graph"

    def test_contains_every_node_and_relation_of_the_projection(self, fixture_scene):
        kb.reset_kb()
        payload = kb.view_payload("hypothesis")
        projection = hypothesis_knowledge.active().projection
        assert len(payload["nodes"]) == len(projection.nodes)
        assert len(payload["edges"]) == len(projection.relations)
        assert {n["id"] for n in payload["nodes"]} == {
            node.identifier for node in projection.nodes
        }

    def test_root_field_matches_the_instruction_nodes_identifier(self, fixture_scene):
        kb.reset_kb()
        payload = kb.view_payload("hypothesis")
        assert (
            payload["root"]
            == hypothesis_knowledge.active().projection.instruction.identifier
        )

    def test_edges_carry_their_relations_endpoints_and_label(self, fixture_scene):
        kb.reset_kb()
        payload = kb.view_payload("hypothesis")
        relation = hypothesis_knowledge.active().projection.relations[0]
        assert {
            "from": relation.source.identifier,
            "to": relation.target.identifier,
            "label": relation.predicate,
        } in payload["edges"]

    def test_details_carry_each_nodes_label_and_lines(self, fixture_scene):
        kb.reset_kb()
        payload = kb.view_payload("hypothesis")
        for node in hypothesis_knowledge.active().projection.nodes:
            assert payload["details"][node.identifier] == {
                "label": node.label,
                "group": "hypothesis",
                "lines": kb._hypothesis_node_lines(node),
            }


@pytest.mark.usefixtures("sample_hypothesis_knowledge")
class TestHypothesisKnowledgeQueries:
    def test_pouring_phase_entities_are_queryable(self, fixture_scene):
        kb.reset_kb()
        result = kb.run_query("an(entity(pouring_phase))")
        assert result["ok"]
        assert row_names(result) == entity_names(structure().defines_phase)

    def test_pouring_phase_query_can_be_filtered_by_coupling_regime(
        self, fixture_scene
    ):
        kb.reset_kb()
        expected = [
            phase
            for phase in structure().defines_phase
            if phase.has_coupling_regime is core.CouplingRegime.COUPLED
        ]
        result = kb.run_query(
            "an(entity(pouring_phase)"
            ".where(pouring_phase.has_coupling_regime == CouplingRegime.COUPLED))"
        )
        assert result["ok"] and expected
        assert row_names(result) == entity_names(expected)

    def test_pouring_phase_query_can_be_filtered_by_its_controllers_mode(
        self, fixture_scene
    ):
        kb.reset_kb()
        expected = [
            phase
            for phase in structure().defines_phase
            if phase.controlled_by is not None
            and phase.controlled_by.has_regulation_mode
            is core.RegulationMode.CONTINUOUS_FEEDBACK
        ]
        result = kb.run_query(
            "an(entity(pouring_phase)"
            ".where(pouring_phase.controlled_by.has_regulation_mode"
            " == RegulationMode.CONTINUOUS_FEEDBACK))"
        )
        assert result["ok"] and expected, result.get("error")
        assert row_names(result) == entity_names(expected)

    def test_pouring_role_entities_are_queryable(self, fixture_scene):
        kb.reset_kb()
        result = kb.run_query("an(entity(pouring_role))")
        assert result["ok"]
        assert row_names(result) == entity_names(structure().defines_role)

    def test_pouring_event_type_entities_are_queryable(self, fixture_scene):
        kb.reset_kb()
        result = kb.run_query("an(entity(pouring_event_type))")
        assert result["ok"]
        assert row_names(result) == entity_names(structure().defines_event_type)

    def test_pouring_event_type_query_can_be_filtered_by_class(self, fixture_scene):
        kb.reset_kb()
        result = kb.run_query(
            "an(entity(pouring_event_type)"
            ".where(HasType(pouring_event_type, core.FailureEventType)))"
        )
        assert result["ok"], result.get("error")
        assert row_names(result) == entity_names(
            hypothesis_knowledge.active().view.failure_events()
        )

    def test_pouring_condition_requirement_entities_are_queryable(self, fixture_scene):
        kb.reset_kb()
        seen = {}
        for phase in structure().defines_phase:
            for requirement in phase.has_condition_requirement:
                seen[id(requirement)] = requirement
        result = kb.run_query("an(entity(pouring_condition_requirement))")
        assert result["ok"]
        assert result["count"] == len(seen)

    def test_pouring_condition_requirement_query_can_be_filtered_by_mode(
        self, fixture_scene
    ):
        kb.reset_kb()
        expected = {
            id(requirement): requirement
            for phase in structure().defines_phase
            for requirement in phase.has_condition_requirement
            if requirement.has_condition_mode is core.ConditionMode.THROUGHOUT
        }
        result = kb.run_query(
            "an(entity(pouring_condition_requirement)"
            ".where(pouring_condition_requirement.has_condition_mode"
            " == ConditionMode.THROUGHOUT))"
        )
        assert result["ok"] and expected
        assert row_names(result) == entity_names(expected.values())

    def test_force_dynamic_events_are_the_events_changing_forces(self, fixture_scene):
        kb.reset_kb()
        result = kb.run_query("an(entity(pouring_force_dynamic_event))")
        assert result["ok"]
        assert row_names(result) == entity_names(
            [
                dynamics.event
                for dynamics in hypothesis_knowledge.active().view.force_dynamics()
            ]
        )

    def test_pouring_transition_query_returns_every_transition(self, fixture_scene):
        kb.reset_kb()
        result = kb.run_query("an(entity(pouring_transition))")
        assert result["ok"]
        assert result["count"] == len(structure().defines_transition)

    def test_a_phase_of_a_class_has_its_initiating_events_queryable(
        self, fixture_scene
    ):
        kb.reset_kb()
        result = kb.run_query(
            "an(entity(pouring_event_type)"
            ".where(HasType(pouring_phase, prehension.LiftPhase))"
            ".where(in_(pouring_event_type, pouring_phase.initiated_by)))"
        )
        assert result["ok"], result.get("error")
        assert row_names(result) == entity_names(
            phase_of_type(prehension.LiftPhase).initiated_by
        )

    def test_a_phase_with_no_initiating_event_returns_nothing(self, fixture_scene):
        kb.reset_kb()
        assert not phase_of_type(prehension.ReachPhase).initiated_by
        result = kb.run_query(
            "an(entity(pouring_event_type)"
            ".where(HasType(pouring_phase, prehension.ReachPhase))"
            ".where(in_(pouring_event_type, pouring_phase.initiated_by)))"
        )
        assert result["ok"]
        assert result["count"] == 0

    def test_a_phase_result_highlights_its_projected_node(self, fixture_scene):
        kb.reset_kb()
        result = kb.run_query(
            "the(entity(pouring_phase).where(HasType(pouring_phase, prehension.LiftPhase)))"
        )
        lift = phase_of_type(prehension.LiftPhase)
        assert hypothesis_knowledge.node_identifier_for(lift) in result["highlight"]

    @pytest.mark.parametrize(
        "query",
        [
            "an(entity(pouring_phase))",
            "an(entity(pouring_transition))",
            "an(entity(pouring_force_dynamic_event))",
            "an(entity(pouring_control_point))",
        ],
    )
    def test_results_highlight_real_hypothesis_graph_nodes(self, fixture_scene, query):
        kb.reset_kb()
        result = kb.run_query(query)
        graph_node_ids = {node["id"] for node in kb.view_payload("hypothesis")["nodes"]}
        assert set(result["highlight"]) & graph_node_ids

    def test_hypothesis_graph_presets_return_real_content(self, fixture_scene):
        kb.reset_kb()
        for preset in kb._hypothesis_graph_presets(kb.get_kb()):
            result = kb.run_query(preset["code"])
            assert result["ok"], "%s -> %s" % (preset["text"], result.get("error"))
            assert result["count"], "%s -> 0 rows (likely a broken field access)" % (
                preset["text"]
            )

    def test_hypothesis_graph_presets_are_included_in_get_presets(self, fixture_scene):
        kb.reset_kb()
        preset_codes = {preset["code"] for preset in kb.get_presets()}
        hypothesis_preset_codes = {
            preset["code"] for preset in kb._hypothesis_graph_presets(kb.get_kb())
        }
        assert hypothesis_preset_codes <= preset_codes


# %% the Designator tab
@pytest.mark.usefixtures("sample_hypothesis_knowledge")
class TestDesignatorView:
    def test_lisp_slot_names_the_role_it_fills(self, fixture_scene):
        kb.reset_kb()
        slots = {
            slot["keyword"]: slot
            for slot in kb.view_payload("designator")["lisp"]["slots"]
        }
        assert slots[":source-container"]["role"] == pouring.SourceContainer.__name__
        assert slots[":tilt-angle"]["role"] is None


class TestHypothesisTabBeforeGenerating:
    def test_the_view_is_empty_but_valid(self, fixture_scene):
        kb.reset_kb()
        payload = kb.view_payload("hypothesis")
        assert payload["ok"]
        assert payload["nodes"] == []
        assert payload["edges"] == []
        assert "Generate" in payload["empty"]

    def test_pouring_queries_return_nothing_rather_than_failing(self, fixture_scene):
        kb.reset_kb()
        result = kb.run_query("an(entity(pouring_phase))")
        assert result["ok"]
        assert result["count"] == 0

    def test_no_pouring_presets_are_offered(self, fixture_scene):
        kb.reset_kb()
        assert kb._hypothesis_graph_presets(kb.get_kb()) == []

    def test_pouring_presets_appear_once_knowledge_is_active(
        self, fixture_scene, sample_hypothesis_knowledge
    ):
        assert kb._hypothesis_graph_presets(kb.get_kb())
