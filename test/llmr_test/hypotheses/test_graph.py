"""Tests for llmr.hypotheses.graph."""

from __future__ import annotations

from llmr.hypotheses import (
    AboutActionEdge,
    ActionNode,
    ClaimStatus,
    EvokesFrameEdge,
    FrameHypothesisNode,
    FrameNetGraphView,
    FrameRoleHypothesisNode,
    GroundedByEdge,
    GroundingState,
    HasRoleEdge,
    HypothesisGraph,
    HypothesisMeta,
    InstructionNode,
    ProducedClaimEdge,
    ReasonerRunNode,
    SlotBindingEvidenceNode,
    SupportedByEdge,
    SymbolGroundingEvidenceNode,
)


class _FakeAction:
    pass


class _FakeSymbol:
    def __init__(self, name: str) -> None:
        self.name = name


def _meta(
    *,
    status: ClaimStatus = ClaimStatus.HYPOTHESIS,
    grounding: GroundingState = GroundingState.TEXT_ONLY,
    run_id: str | None = None,
) -> HypothesisMeta:
    return HypothesisMeta(
        source_reasoner="framenet_reasoner",
        status=status,
        grounding=grounding,
        run_id=run_id,
    )


def _sample_graph() -> tuple[HypothesisGraph, dict[str, object]]:
    graph = HypothesisGraph()
    action = _FakeAction()
    milk = _FakeSymbol("milk")

    nodes = {
        "instruction": InstructionNode(
            id="i1",
            meta=_meta(),
            text="pick up the milk from the table",
            normalized_text="pick up the milk from the table",
        ),
        "action": ActionNode(
            id="a1",
            meta=_meta(),
            action_ref=action,
            action_type="PickUpAction",
        ),
        "run": ReasonerRunNode(
            id="run1",
            meta=_meta(run_id="run1"),
            reasoner_name="framenet_reasoner",
            run_id="run1",
            model_name="gpt-test",
            prompt_version="framenet_v1",
            action_type="PickUpAction",
            instruction_text="pick up the milk from the table",
        ),
        "frame": FrameHypothesisNode(
            id="f1",
            meta=_meta(run_id="run1"),
            frame="Getting",
            lexical_unit="pick_up.v",
            framenet_label="picking_up_object",
            action_type="PickUpAction",
            instruction_text="pick up the milk from the table",
        ),
        "theme": FrameRoleHypothesisNode(
            id="r1",
            meta=_meta(
                status=ClaimStatus.SUPPORTED,
                grounding=GroundingState.SYMBOL_GROUNDED,
                run_id="run1",
            ),
            role_family="core",
            role_name="theme",
            filler_text="milk",
            filler_kind="entity",
        ),
        "goal": FrameRoleHypothesisNode(
            id="r2",
            meta=_meta(run_id="run1"),
            role_family="core",
            role_name="goal",
            filler_text="robot grasp",
            filler_kind="abstract",
        ),
        "slot_ev": SlotBindingEvidenceNode(
            id="ev1",
            meta=_meta(
                status=ClaimStatus.SUPPORTED,
                grounding=GroundingState.SLOT_ALIGNED,
                run_id="run1",
            ),
            slot_name="object_designator",
            value_ref=milk,
            value_repr="milk",
        ),
        "ground_ev": SymbolGroundingEvidenceNode(
            id="ev2",
            meta=_meta(
                status=ClaimStatus.SUPPORTED,
                grounding=GroundingState.SYMBOL_GROUNDED,
                run_id="run1",
            ),
            query_text="milk",
            symbol_ref=milk,
            symbol_type="WorldBody",
            grounding_method="symbol_grounder",
        ),
    }

    for node in nodes.values():
        graph.add_node(node)

    edges = [
        EvokesFrameEdge(id="e1", meta=_meta(run_id="run1"), src_id="i1", dst_id="f1"),
        AboutActionEdge(id="e2", meta=_meta(run_id="run1"), src_id="f1", dst_id="a1"),
        ProducedClaimEdge(id="e3", meta=_meta(run_id="run1"), src_id="run1", dst_id="f1"),
        ProducedClaimEdge(id="e4", meta=_meta(run_id="run1"), src_id="run1", dst_id="r1"),
        ProducedClaimEdge(id="e5", meta=_meta(run_id="run1"), src_id="run1", dst_id="r2"),
        HasRoleEdge(id="e6", meta=_meta(run_id="run1"), src_id="f1", dst_id="r1"),
        HasRoleEdge(id="e7", meta=_meta(run_id="run1"), src_id="f1", dst_id="r2"),
        SupportedByEdge(id="e8", meta=_meta(run_id="run1"), src_id="r1", dst_id="ev1"),
        GroundedByEdge(id="e9", meta=_meta(run_id="run1"), src_id="r1", dst_id="ev2"),
    ]
    for edge in edges:
        graph.add_edge(edge)

    return graph, {"action": action, "milk": milk}


def _add_second_action_with_shared_instruction(
    graph: HypothesisGraph,
) -> dict[str, object]:
    action = _FakeAction()
    shared_instruction = graph.add_node(
        InstructionNode(
            id="i2",
            meta=_meta(),
            text="pick up the milk from the table",
            normalized_text="pick up the milk from the table",
        )
    )
    assert shared_instruction.id == "i1"

    nodes = {
        "action": ActionNode(
            id="a2",
            meta=_meta(),
            action_ref=action,
            action_type="PlaceAction",
        ),
        "run": ReasonerRunNode(
            id="run2",
            meta=_meta(run_id="run2"),
            reasoner_name="framenet_reasoner",
            run_id="run2",
            model_name="gpt-test",
            prompt_version="framenet_v1",
            action_type="PlaceAction",
            instruction_text="pick up the milk from the table",
        ),
        "frame": FrameHypothesisNode(
            id="f2",
            meta=_meta(run_id="run2"),
            frame="Placing",
            lexical_unit="place.v",
            framenet_label="placing_object",
            action_type="PlaceAction",
            instruction_text="pick up the milk from the table",
        ),
        "role": FrameRoleHypothesisNode(
            id="r3",
            meta=_meta(run_id="run2"),
            role_family="core",
            role_name="goal",
            filler_text="counter",
            filler_kind="place",
        ),
    }
    for node in nodes.values():
        graph.add_node(node)

    edges = [
        EvokesFrameEdge(id="e10", meta=_meta(run_id="run2"), src_id="i1", dst_id="f2"),
        AboutActionEdge(id="e11", meta=_meta(run_id="run2"), src_id="f2", dst_id="a2"),
        ProducedClaimEdge(id="e12", meta=_meta(run_id="run2"), src_id="run2", dst_id="f2"),
        ProducedClaimEdge(id="e13", meta=_meta(run_id="run2"), src_id="run2", dst_id="r3"),
        HasRoleEdge(id="e14", meta=_meta(run_id="run2"), src_id="f2", dst_id="r3"),
    ]
    for edge in edges:
        graph.add_edge(edge)

    return {"action": action}


class TestGraphDomains:
    def test_typed_domains(self) -> None:
        graph, _ = _sample_graph()
        view = FrameNetGraphView(graph)
        assert len(graph.instructions) == 1
        assert len(graph.actions) == 1
        assert len(graph.reasoner_runs) == 1
        assert len(view.frames()) == 1
        assert len(view.roles()) == 2
        assert len(graph.evidences) == 2

    def test_domain_by_type(self) -> None:
        graph, _ = _sample_graph()
        assert len(graph.domain(FrameHypothesisNode)) == 1
        assert len(graph.domain(FrameRoleHypothesisNode)) == 2


class TestGraphIndexes:
    def test_instruction_lookup(self) -> None:
        graph, _ = _sample_graph()
        node = graph.get_instruction("Pick up   the milk from the table")
        assert node is not None
        assert node.id == "i1"

    def test_action_lookup(self) -> None:
        graph, refs = _sample_graph()
        node = graph.get_action(refs["action"])
        assert node is not None
        assert node.id == "a1"

    def test_frame_lookup(self) -> None:
        graph, _ = _sample_graph()
        frames = FrameNetGraphView(graph).frames_by_frame("Getting")
        assert [frame.id for frame in frames] == ["f1"]

    def test_role_lookup(self) -> None:
        graph, _ = _sample_graph()
        roles = FrameNetGraphView(graph).roles_by_role_name("theme")
        assert [role.id for role in roles] == ["r1"]

    def test_status_lookup(self) -> None:
        graph, _ = _sample_graph()
        supported_ids = {node.id for node in graph.nodes_by_status(ClaimStatus.SUPPORTED)}
        assert {"r1", "ev1", "ev2"}.issubset(supported_ids)

    def test_grounding_lookup(self) -> None:
        graph, _ = _sample_graph()
        grounded_ids = {
            node.id for node in graph.nodes_by_grounding(GroundingState.SYMBOL_GROUNDED)
        }
        assert {"r1", "ev2"}.issubset(grounded_ids)

    def test_symbol_grounding_lookup(self) -> None:
        graph, refs = _sample_graph()
        evidence = graph.groundings_for_symbol(refs["milk"])
        assert [node.id for node in evidence] == ["ev2"]


class TestGraphAdjacency:
    def test_out_edges(self) -> None:
        graph, _ = _sample_graph()
        assert len(graph.out_edges("f1")) == 3
        assert len(graph.out_edges("f1", HasRoleEdge)) == 2

    def test_in_edges(self) -> None:
        graph, _ = _sample_graph()
        assert len(graph.in_edges("f1")) == 2
        assert len(graph.in_edges("r1", HasRoleEdge)) == 1

    def test_get_targets(self) -> None:
        graph, _ = _sample_graph()
        targets = graph.get_targets("f1", HasRoleEdge)
        assert [node.id for node in targets] == ["r1", "r2"]

    def test_get_sources(self) -> None:
        graph, _ = _sample_graph()
        sources = graph.get_sources("r1", HasRoleEdge)
        assert [node.id for node in sources] == ["f1"]

    def test_neighbors(self) -> None:
        graph, _ = _sample_graph()
        neighbors = graph.neighbors("f1")
        assert {node.id for node in neighbors} == {"i1", "a1", "run1", "r1", "r2"}

    def test_filtered_edge_helpers(self) -> None:
        graph, _ = _sample_graph()
        outgoing = graph.out_edges("f1", HasRoleEdge)
        incoming = graph.get_incoming_edges_with_condition(
            "r1",
            lambda edge: edge.meta.run_id == "run1",
        )
        assert [edge.id for edge in outgoing] == ["e6", "e7"]
        assert {edge.id for edge in incoming} == {"e4", "e6"}


class TestGraphNativeBackend:
    def test_counts_reflect_graph_contents(self) -> None:
        graph, _ = _sample_graph()
        assert graph.node_count == 8
        assert graph.edge_count == 9

    def test_get_edge(self) -> None:
        graph, _ = _sample_graph()
        edge = graph.get_edge("e6")
        assert edge is not None
        assert edge.src_id == "f1"
        assert edge.dst_id == "r1"

    def test_run_lookup(self) -> None:
        graph, _ = _sample_graph()
        assert {node.id for node in graph.nodes_for_run("run1")} == {
            "run1",
            "f1",
            "r1",
            "r2",
            "ev1",
            "ev2",
        }
        assert {edge.id for edge in graph.edges_for_run("run1")} == {
            "e1",
            "e2",
            "e3",
            "e4",
            "e5",
            "e6",
            "e7",
            "e8",
            "e9",
        }

    def test_reasoner_lookup(self) -> None:
        graph, _ = _sample_graph()
        assert len(graph.nodes_from_reasoner("framenet_reasoner")) == graph.node_count
        assert len(graph.edges_from_reasoner("framenet_reasoner")) == graph.edge_count

    def test_relation_existence_lookup(self) -> None:
        graph, _ = _sample_graph()
        assert graph.edge_exists("f1", "r1", HasRoleEdge)
        assert graph.edge_exists("f1", "r1", HasRoleEdge, run_id="run1")
        assert not graph.edge_exists("f1", "ev1", HasRoleEdge)

    def test_to_dot_contains_node_and_relation_labels(self) -> None:
        graph, _ = _sample_graph()
        dot = graph.to_dot()
        assert "FrameHypothesisNode" in dot
        assert "has_role" in dot


class TestGraphDedup:
    def test_instruction_nodes_dedup_by_normalized_text(self) -> None:
        graph = HypothesisGraph()
        first = graph.add_node(
            InstructionNode(
                id="i1",
                meta=_meta(),
                text="Pick up the milk",
                normalized_text="pick up the milk",
            )
        )
        second = graph.add_node(
            InstructionNode(
                id="i2",
                meta=_meta(),
                text="pick up   the milk",
                normalized_text="pick up the milk",
            )
        )
        assert first is second
        assert len(graph.instructions) == 1

    def test_action_nodes_dedup_by_action_identity(self) -> None:
        graph = HypothesisGraph()
        action = _FakeAction()
        first = graph.add_node(
            ActionNode(
                id="a1",
                meta=_meta(),
                action_ref=action,
                action_type="PickUpAction",
            )
        )
        second = graph.add_node(
            ActionNode(
                id="a2",
                meta=_meta(),
                action_ref=action,
                action_type="PickUpAction",
            )
        )
        assert first is second
        assert len(graph.actions) == 1

    def test_claim_nodes_do_not_dedup_across_runs(self) -> None:
        graph = HypothesisGraph()
        graph.add_node(
            FrameHypothesisNode(
                id="f1",
                meta=_meta(),
                frame="Getting",
                lexical_unit="pick_up.v",
                framenet_label="picking_up_object",
                action_type="PickUpAction",
                instruction_text="pick up the milk",
            )
        )
        graph.add_node(
            FrameHypothesisNode(
                id="f2",
                meta=_meta(),
                frame="Getting",
                lexical_unit="pick_up.v",
                framenet_label="picking_up_object",
                action_type="PickUpAction",
                instruction_text="pick up the milk",
            )
        )
        assert len(FrameNetGraphView(graph).frames()) == 2


class TestGraphSubgraphs:
    def test_subgraph_for_run_includes_context_endpoints(self) -> None:
        graph, _ = _sample_graph()
        subgraph = graph.subgraph_for_run("run1")
        assert {node.id for node in subgraph.iter_nodes()} == {
            "i1",
            "a1",
            "run1",
            "f1",
            "r1",
            "r2",
            "ev1",
            "ev2",
        }
        assert {edge.id for edge in subgraph.iter_edges()} == {
            "e1",
            "e2",
            "e3",
            "e4",
            "e5",
            "e6",
            "e7",
            "e8",
            "e9",
        }

    def test_subgraph_for_reasoner_includes_reasoner_slice(self) -> None:
        graph, _ = _sample_graph()
        subgraph = graph.subgraph_for_reasoner("framenet_reasoner")
        assert subgraph.node_count == graph.node_count
        assert subgraph.edge_count == graph.edge_count

    def test_subgraph_for_action_does_not_walk_through_shared_instruction(self) -> None:
        graph, refs = _sample_graph()
        second_refs = _add_second_action_with_shared_instruction(graph)
        subgraph = graph.subgraph_for_action(refs["action"])
        subgraph_ids = {node.id for node in subgraph.iter_nodes()}
        assert "a1" in subgraph_ids
        assert "i1" in subgraph_ids
        assert {"run2", "f2", "r3", "a2"}.isdisjoint(subgraph_ids)
        assert graph.subgraph_for_action(second_refs["action"]).has_node("a2")


class TestGraphLifecycle:
    def test_remove_edge_updates_indexes(self) -> None:
        graph, _ = _sample_graph()
        removed = graph.remove_edge("e9")
        assert removed is not None
        assert removed.id == "e9"
        assert not graph.has_edge("e9")
        assert graph.edge_count == 8
        assert not graph.edge_exists("r1", "ev2", GroundedByEdge)

    def test_remove_node_removes_incident_edges(self) -> None:
        graph, _ = _sample_graph()
        removed = graph.remove_node("r1")
        assert removed is not None
        assert removed.id == "r1"
        assert not graph.has_node("r1")
        assert graph.node_count == 7
        assert graph.edge_count == 5
        assert FrameNetGraphView(graph).roles_by_role_name("theme") == []

    def test_clear_empties_graph(self) -> None:
        graph, _ = _sample_graph()
        graph.clear()
        assert graph.node_count == 0
        assert graph.edge_count == 0
        assert graph.instructions == []
        assert graph.edges == []

    def test_prune_run_keeps_shared_context_nodes(self) -> None:
        graph, _ = _sample_graph()
        removed_nodes, removed_edges = graph.prune_run("run1")
        assert (removed_nodes, removed_edges) == (6, 9)
        assert {node.id for node in graph.iter_nodes()} == {"i1", "a1"}
        assert graph.edge_count == 0

    def test_prune_action_keeps_shared_instruction(self) -> None:
        graph, refs = _sample_graph()
        _add_second_action_with_shared_instruction(graph)
        removed_nodes, removed_edges = graph.prune_action(refs["action"])
        assert removed_nodes == 7
        assert removed_edges == 9
        assert {node.id for node in graph.iter_nodes()} == {"i1", "a2", "run2", "f2", "r3"}
        assert {edge.id for edge in graph.iter_edges()} == {"e10", "e11", "e12", "e13", "e14"}
