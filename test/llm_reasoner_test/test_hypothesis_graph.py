from __future__ import annotations

import pytest
import rustworkx as rx
from llm_reasoner.knowledge.hypothesis.graph import (
    HypothesisGraphPresentation,
    HypothesisGraphProjector,
    HypothesisGraphVisualizer,
)
from llm_reasoner.knowledge.hypothesis.provenance import Provenance
from llm_reasoner.knowledge.hypothesis.record import HypothesisRecord
from llm_reasoner.knowledge.microworlds.flanagan.core import ActionPhaseStructure
from llm_reasoner.knowledge.microworlds.flanagan.prehension import Liftoff, ReachPhase
from llm_reasoner.knowledge.ontological.decorators import ConceptReference

# %% Hypothesis fixture


@pytest.fixture
def branching_hypothesis() -> HypothesisRecord:
    """
    Build a hypothesis in which one event has two incoming relations.
    """
    action = ConceptReference(iri="soma:PouringInto", label="PouringInto")
    liftoff = Liftoff(label="Liftoff")
    reach = ReachPhase(label="Reach", terminated_by=[liftoff])
    structure = ActionPhaseStructure(
        label="Pouring structure",
        decomposes_task=action,
        defines_phase=[reach],
        defines_event_type=[liftoff],
    )
    return HypothesisRecord.from_root(
        root=structure,
        action_type=action,
        provenance=Provenance(generator="test", prompt_version="test"),
    )


# %% Rooted projection


class TestRootedProjection:
    def test_instruction_is_the_center_and_structure_is_its_child(
        self, branching_hypothesis: HypothesisRecord
    ) -> None:
        projection = HypothesisGraphProjector().project(
            branching_hypothesis,
            instruction="Pour liquid into the receptacle",
        )

        assert projection.instruction.label == "Pour liquid into the receptacle"
        assert projection.root.label == "Pouring structure"
        assert (
            projection.instruction,
            "describes",
            projection.root,
        ) in {
            (relation.source, relation.predicate, relation.target)
            for relation in projection.relations
        }

    def test_declared_properties_become_labelled_parent_child_relations(
        self, branching_hypothesis: HypothesisRecord
    ) -> None:
        projection = HypothesisGraphProjector().project(
            branching_hypothesis,
            presentation=HypothesisGraphPresentation.COMPLETE,
        )

        relations = {
            (relation.source.label, relation.predicate, relation.target.label)
            for relation in projection.relations
        }
        assert relations == {
            ("PouringInto", "describes", "Pouring structure"),
            ("Pouring structure", "defines_phase", "Reach"),
            ("Pouring structure", "defines_event_type", "Liftoff"),
            ("Reach", "terminated_by", "Liftoff"),
        }

    def test_every_projected_entity_is_reachable_from_the_instruction(
        self, branching_hypothesis: HypothesisRecord
    ) -> None:
        projection = HypothesisGraphProjector().project(branching_hypothesis)

        descendants = rx.descendants(projection.graph, projection.instruction_index)

        assert descendants == {
            index
            for index in projection.graph.node_indices()
            if index != projection.instruction_index
        }


# %% Visualization data


class TestCenteredVisualization:
    def test_only_instruction_is_visible_initially(
        self, branching_hypothesis: HypothesisRecord
    ) -> None:
        projection = HypothesisGraphProjector().project(branching_hypothesis)
        visualizer = HypothesisGraphVisualizer.for_graph(projection)

        visible_labels = {node["label"] for node in visualizer.graph_nodes()}

        assert visible_labels == {"PouringInto"}
        assert visualizer.graph_edges() == []

    def test_instruction_has_fixed_center_position(
        self, branching_hypothesis: HypothesisRecord
    ) -> None:
        projection = HypothesisGraphProjector().project(branching_hypothesis)
        visualizer = HypothesisGraphVisualizer.for_graph(projection)

        nodes = {int(node["id"]): node for node in visualizer.graph_nodes()}

        assert nodes[projection.instruction_index]["x"] == 0.0
        assert nodes[projection.instruction_index]["y"] == 0.0
        assert nodes[projection.instruction_index]["fixed"] is True

    def test_click_reveals_only_immediate_children(
        self, branching_hypothesis: HypothesisRecord
    ) -> None:
        projection = HypothesisGraphProjector().project(branching_hypothesis)
        visualizer = HypothesisGraphVisualizer.for_graph(projection)

        visualizer.node_details(projection.instruction_index)

        assert {node["label"] for node in visualizer.graph_nodes()} == {
            "PouringInto",
            "Pouring structure",
        }
        assert {edge["label"] for edge in visualizer.graph_edges()} == {"describes"}

        visualizer.node_details(projection.root_index)

        assert {node["label"] for node in visualizer.graph_nodes()} == {
            "PouringInto",
            "Pouring structure",
            "Phases (1)",
            "Event Types (1)",
        }
        assert {edge["label"] for edge in visualizer.graph_edges()} == {
            "describes",
            "groups",
        }

        phase_group_index = next(
            index
            for index in projection.graph.node_indices()
            if projection.graph[index].label == "Phases (1)"
        )
        visualizer.node_details(phase_group_index)

        assert {node["label"] for node in visualizer.graph_nodes()} == {
            "PouringInto",
            "Pouring structure",
            "Phases (1)",
            "Event Types (1)",
            "Reach",
        }
        assert {edge["label"] for edge in visualizer.graph_edges()} == {
            "describes",
            "groups",
            "defines_phase",
        }

    def test_expanding_child_reveals_its_outgoing_relation(
        self, branching_hypothesis: HypothesisRecord
    ) -> None:
        projection = HypothesisGraphProjector().project(branching_hypothesis)
        visualizer = HypothesisGraphVisualizer.for_graph(projection)
        phase_group_index = next(
            index
            for index in projection.graph.node_indices()
            if projection.graph[index].label == "Phases (1)"
        )
        reach_index = next(
            index
            for index in projection.graph.node_indices()
            if projection.graph[index].label == "Reach"
        )
        visualizer.node_details(projection.instruction_index)
        visualizer.node_details(projection.root_index)
        visualizer.node_details(phase_group_index)
        visualizer.node_details(reach_index)

        labels = {edge["label"] for edge in visualizer.graph_edges()}

        assert labels == {
            "describes",
            "groups",
            "defines_phase",
            "terminated_by",
        }

    def test_collapse_all_hides_expanded_branches(
        self, branching_hypothesis: HypothesisRecord
    ) -> None:
        projection = HypothesisGraphProjector().project(branching_hypothesis)
        visualizer = HypothesisGraphVisualizer.for_graph(projection)
        visualizer.node_details(projection.instruction_index)
        visualizer.node_details(projection.root_index)

        visualizer.collapse_all()

        assert {node["label"] for node in visualizer.graph_nodes()} == {"PouringInto"}

    def test_collapsing_node_hides_its_expanded_branch(
        self, branching_hypothesis: HypothesisRecord
    ) -> None:
        projection = HypothesisGraphProjector().project(branching_hypothesis)
        visualizer = HypothesisGraphVisualizer.for_graph(projection)
        phase_group_index = next(
            index
            for index in projection.graph.node_indices()
            if projection.graph[index].label == "Phases (1)"
        )
        visualizer.node_details(projection.instruction_index)
        visualizer.node_details(projection.root_index)
        visualizer.node_details(phase_group_index)

        visualizer.collapse_branch(projection.root_index)

        assert {node["label"] for node in visualizer.graph_nodes()} == {
            "PouringInto",
            "Pouring structure",
        }
        assert {edge["label"] for edge in visualizer.graph_edges()} == {"describes"}

    def test_non_center_nodes_participate_in_repulsion_physics(
        self, branching_hypothesis: HypothesisRecord
    ) -> None:
        projection = HypothesisGraphProjector().project(branching_hypothesis)
        visualizer = HypothesisGraphVisualizer.for_graph(projection)
        visualizer.node_details(projection.instruction_index)

        nodes = {int(node["id"]): node for node in visualizer.graph_nodes()}
        physics_options = visualizer.network_options()["physics"]

        assert nodes[projection.instruction_index]["fixed"] is True
        assert nodes[projection.instruction_index]["physics"] is False
        assert nodes[projection.root_index]["fixed"] is False
        assert nodes[projection.root_index]["physics"] is True
        assert physics_options["solver"] == "repulsion"
        assert physics_options["repulsion"]["nodeDistance"] == 220

    def test_visual_style_uses_lightweight_text_on_white_canvas(
        self, branching_hypothesis: HypothesisRecord
    ) -> None:
        projection = HypothesisGraphProjector().project(branching_hypothesis)
        visualizer = HypothesisGraphVisualizer.for_graph(projection)

        options = visualizer.network_options()

        assert visualizer.style.canvas_background_color == "#ffffff"
        assert options["nodes"]["font"] == {
            "color": "#202124",
            "face": "Arial",
            "size": 12,
            "strokeWidth": 0,
        }
        assert options["edges"]["font"] == {
            "background": "#ffffff",
            "color": "#5f6368",
            "face": "Arial",
            "size": 10,
            "strokeWidth": 0,
        }

    def test_application_exposes_double_click_collapse(
        self, branching_hypothesis: HypothesisRecord
    ) -> None:
        projection = HypothesisGraphProjector().project(branching_hypothesis)
        visualizer = HypothesisGraphVisualizer.for_graph(projection)
        application = visualizer.build_application()
        client = application.test_client()
        client.get(f"/node/{projection.instruction_index}")
        client.get(f"/node/{projection.root_index}")

        response = client.post(f"/collapse/{projection.root_index}")
        page = client.get("/").get_data(as_text=True)

        assert response.status_code == 200
        assert response.get_json() == {"collapsed": projection.root_index}
        assert "doubleClick" in page
        assert visualizer.style.canvas_background_color in page
        assert {node["label"] for node in visualizer.graph_nodes()} == {
            "PouringInto",
            "Pouring structure",
        }
