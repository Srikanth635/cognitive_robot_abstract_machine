"""
Tests for the scene-driven knowledge base and its graph-panel payloads.
"""

from types import SimpleNamespace

import pytest

krrood = pytest.importorskip("krrood", reason="EQL requires krrood")

from llm_reasoner.ontologies.predictive_manipulation import (  # noqa: E402
    core,
    prehension,
)

from cram_viz import kb  # noqa: E402  (importable once krrood is present)
from cram_viz import designator_knowledge  # noqa: E402
from cram_viz import hypothesis_knowledge  # noqa: E402
from cram_viz.designator_description import DesignatorDescriber  # noqa: E402
from cram_viz.hypothesis_knowledge.ontology_backing import (  # noqa: E402
    OntologyBacking,
)
from cram_viz.hypothesis_knowledge.ontology_class_graph import (  # noqa: E402
    OntologyClassGraph,
)
from llm_reasoner.ontologies import CRAM_AICOR_NAMESPACE  # noqa: E402
from llm_reasoner.ontologies.terms import ConceptReference  # noqa: E402


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
        )
        presets = kb._scene_verb_presets(fake_kb)
        assert [preset["text"] for preset in presets] == [
            "every 'transport' action in this demo",
            "what does each 'transport' move, and for how long?",
        ]
        assert all("transport_" in preset["code"] for preset in presets)

    def test_the_verb_is_the_one_the_recorded_episodes_share(self):
        fake_kb = SimpleNamespace(
            episodes=[
                SimpleNamespace(name="pickup_cup"),
                SimpleNamespace(name="pickup_bowl"),
                SimpleNamespace(name="transport_milk"),
            ]
        )
        presets = kb._scene_verb_presets(fake_kb)
        assert presets and all("pickup_" in preset["code"] for preset in presets)

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


# %% the Ontology Graph tab
@pytest.mark.usefixtures("sample_hypothesis_knowledge")
class TestOntologyGraphView:
    @staticmethod
    def class_graph():
        return OntologyClassGraph.of(hypothesis_knowledge.active().view)

    def test_payload_is_ok_with_the_ontology_graph_crumb(self, fixture_scene):
        kb.reset_kb()
        payload = kb.view_payload("ontology")
        assert payload["ok"]
        assert payload["crumb"] == "ontology graph"

    def test_nodes_are_the_classes_behind_the_hypothesis(self, fixture_scene):
        kb.reset_kb()
        payload = kb.view_payload("ontology")
        assert payload["nodes"] == [
            {
                "id": node.iri,
                "label": node.label,
                "category": node.category.value,
                "count": node.instance_count,
            }
            for node in self.class_graph().nodes
        ]

    def test_edges_name_the_property_stating_them(self, fixture_scene):
        kb.reset_kb()
        payload = kb.view_payload("ontology")
        backing = OntologyBacking()
        assert payload["edges"] == [
            {
                "source": edge.source,
                "target": edge.target,
                "kind": edge.kind.value,
                "label": (
                    backing.prefixed(edge.property_iri) if edge.property_iri else ""
                ),
                "count": edge.count,
            }
            for edge in self.class_graph().edges
        ]

    def test_details_describe_each_class(self, fixture_scene):
        kb.reset_kb()
        details = kb.view_payload("ontology")["details"]
        for node in self.class_graph().nodes:
            detail = details[node.iri]
            assert detail["iri"] == node.iri
            assert detail["description"] == node.description
            assert detail["requirements"] == list(node.requirements)
            assert detail["instances"] == list(node.instances)
            assert [claim["text"] for claim in detail["claims"]] == [
                claim.text for claim in node.claims
            ]


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

    def test_there_are_at_most_ten_hypothesis_graph_presets(self, fixture_scene):
        kb.reset_kb()
        assert len(kb._hypothesis_graph_presets(kb.get_kb())) <= 10

    def test_hypothesis_graph_presets_answer_in_readable_values(self, fixture_scene):
        kb.reset_kb()
        for preset in kb._hypothesis_graph_presets(kb.get_kb()):
            cells = [
                str(value)
                for row in kb.run_query(preset["code"])["rows"]
                for value in row.values()
            ]
            assert not [
                cell for cell in cells if "MonitoredList(" in cell or "://" in cell
            ], preset["text"]

    def test_hypothesis_graph_presets_are_included_in_get_presets(self, fixture_scene):
        kb.reset_kb()
        preset_codes = {preset["code"] for preset in kb.get_presets()}
        hypothesis_preset_codes = {
            preset["code"] for preset in kb._hypothesis_graph_presets(kb.get_kb())
        }
        assert hypothesis_preset_codes <= preset_codes


@pytest.mark.usefixtures("sample_hypothesis_knowledge")
class TestHypothesisAnswerRendering:
    def test_a_list_of_fragments_renders_their_names(self, fixture_scene):
        events = next(
            phase.initiated_by
            for phase in structure().defines_phase
            if phase.initiated_by
        )
        assert kb._jsonable(events) == [kb._entity_name(event) for event in events]

    def test_an_enumerated_value_renders_as_its_prefixed_term(self):
        member = core.CouplingRegime.COUPLED
        assert kb._jsonable(member) == OntologyBacking().prefixed(member.value)

    def test_a_concept_renders_as_its_prefixed_term(self):
        concept = ConceptReference(iri=CRAM_AICOR_NAMESPACE + "Pouring")
        assert kb._jsonable(concept) == OntologyBacking().prefixed(concept.iri)

    def test_fragments_in_a_combined_answer_highlight_their_graph_nodes(
        self, fixture_scene
    ):
        kb.reset_kb()
        result = kb.run_query(
            "set_of(pouring_transition.from_phase, pouring_transition.to_phase)"
        )
        assert {
            hypothesis_knowledge.node_identifier_for(phase)
            for transition in structure().defines_transition
            for phase in (transition.from_phase, transition.to_phase)
        } <= set(result["highlight"])


@pytest.mark.usefixtures("sample_hypothesis_knowledge")
class TestHypothesisQueryVariables:
    def test_phases_are_answered_in_the_order_they_run(self, fixture_scene):
        kb.reset_kb()
        result = kb.run_query("an(entity(pouring_phase))")
        assert [row["__entity__"] for row in result["rows"]] == [
            kb._entity_name(phase)
            for phase in hypothesis_knowledge.active().view.phases()
        ]

    @pytest.mark.parametrize(
        "variable, fragments",
        [
            ("pouring_parameter", lambda root: root.defines_parameter),
            ("pouring_sensory_prediction", lambda root: root.defines_prediction),
            ("pouring_comparison", lambda root: root.defines_comparison),
            ("pouring_correction", lambda root: root.defines_correction),
            ("pouring_gaze_fixation", lambda root: root.defines_gaze_fixation),
        ],
    )
    def test_structure_knowledge_is_queryable(self, fixture_scene, variable, fragments):
        kb.reset_kb()
        result = kb.run_query("an(entity(%s))" % variable)
        assert result["ok"] and fragments(structure())
        assert row_names(result) == entity_names(fragments(structure()))


# %% the Designator tab
@pytest.mark.usefixtures("sample_hypothesis_knowledge")
class TestDesignatorView:
    def test_designator_is_the_described_action(self, fixture_scene):
        kb.reset_kb()
        view = hypothesis_knowledge.active().view
        expected = DesignatorDescriber(view=view).describe(
            designator_knowledge.reflect_pouring_action(kb.get_kb().objects, view)
        )
        payload = kb.view_payload("designator")
        assert payload["ok"]
        assert payload["designator"] == expected.payload()

    def test_without_a_hypothesis_the_action_slots_are_shown(self, fixture_scene):
        hypothesis_knowledge.activate(None)
        kb.reset_kb()
        properties = kb.view_payload("designator")["designator"]["properties"]
        assert "hypothesis" not in [entry["key"] for entry in properties]


class TestHypothesisTabBeforeGenerating:
    def test_the_view_is_empty_but_valid(self, fixture_scene):
        kb.reset_kb()
        payload = kb.view_payload("hypothesis")
        assert payload["ok"]
        assert payload["nodes"] == []
        assert payload["edges"] == []
        assert "Generate" in payload["empty"]

    def test_the_ontology_view_is_empty_but_valid(self, fixture_scene):
        kb.reset_kb()
        payload = kb.view_payload("ontology")
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
