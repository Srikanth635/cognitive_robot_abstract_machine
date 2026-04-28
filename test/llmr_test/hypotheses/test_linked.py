"""Tests for llmr.hypotheses.linked — GraphLinked mixin and graph_context."""

from __future__ import annotations

import pytest

from llmr.hypotheses import (
    AboutActionEdge,
    ActionNode,
    ClaimStatus,
    EvokesFrameEdge,
    FrameHypothesisNode,
    FrameRoleHypothesisNode,
    GraphLinked,
    GroundingState,
    HasMotionPhaseEdge,
    HasRoleEdge,
    HypothesisGraph,
    HypothesisMeta,
    InstructionNode,
    MotionPhaseHypothesisNode,
    MotionPlanHypothesisNode,
    ProducedClaimEdge,
    ReasonerRunNode,
    graph_context,
)
from llmr.hypotheses.projectors.flanagan.edges import EvokesMotionPlanEdge


class _FakeAction:
    pass


def _meta(*, run_id: str | None = None, reasoner: str = "test_reasoner") -> HypothesisMeta:
    return HypothesisMeta(
        source_reasoner=reasoner,
        status=ClaimStatus.HYPOTHESIS,
        grounding=GroundingState.TEXT_ONLY,
        run_id=run_id,
    )


# ---------------------------------------------------------------------------
# Small FrameNet graph fixture
# ---------------------------------------------------------------------------

def _framenet_graph() -> tuple[HypothesisGraph, FrameHypothesisNode]:
    graph = HypothesisGraph()
    action = _FakeAction()

    i1 = graph.add_node(
        InstructionNode(
            id="i1", meta=_meta(), text="pick up", normalized_text="pick up"
        )
    )
    a1 = graph.add_node(
        ActionNode(id="a1", meta=_meta(), action_ref=action, action_type="PickUpAction")
    )
    run1 = graph.add_node(
        ReasonerRunNode(
            id="run1",
            meta=_meta(run_id="run1"),
            reasoner_name="test_reasoner",
            run_id="run1",
            model_name=None,
            prompt_version=None,
            action_type="PickUpAction",
            instruction_text="pick up",
        )
    )
    f1 = graph.add_node(
        FrameHypothesisNode(
            id="f1",
            meta=_meta(run_id="run1"),
            frame="Getting",
            lexical_unit="pick_up.v",
            framenet_label="picking_up_object",
            action_type="PickUpAction",
            instruction_text="pick up",
        )
    )
    r1 = graph.add_node(
        FrameRoleHypothesisNode(
            id="r1",
            meta=_meta(run_id="run1"),
            role_family="core",
            role_name="theme",
            filler_text="milk",
            filler_kind="entity",
        )
    )
    r2 = graph.add_node(
        FrameRoleHypothesisNode(
            id="r2",
            meta=_meta(run_id="run1"),
            role_family="core",
            role_name="goal",
            filler_text="robot grasp",
            filler_kind="abstract",
        )
    )

    for edge in [
        EvokesFrameEdge(id="e1", meta=_meta(run_id="run1"), src_id="i1", dst_id="f1"),
        AboutActionEdge(id="e2", meta=_meta(run_id="run1"), src_id="f1", dst_id="a1"),
        ProducedClaimEdge(id="e3", meta=_meta(run_id="run1"), src_id="run1", dst_id="f1"),
        HasRoleEdge(id="e4", meta=_meta(run_id="run1"), src_id="f1", dst_id="r1"),
        HasRoleEdge(id="e5", meta=_meta(run_id="run1"), src_id="f1", dst_id="r2"),
    ]:
        graph.add_edge(edge)

    return graph, f1


# ---------------------------------------------------------------------------
# Small Flanagan graph fixture
# ---------------------------------------------------------------------------

def _flanagan_graph() -> tuple[HypothesisGraph, MotionPlanHypothesisNode]:
    graph = HypothesisGraph()
    action = _FakeAction()

    i1 = graph.add_node(
        InstructionNode(
            id="i1", meta=_meta(), text="pick up cup", normalized_text="pick up cup"
        )
    )
    plan = graph.add_node(
        MotionPlanHypothesisNode(
            id="plan1",
            meta=_meta(run_id="r1"),
            action_type="PickUpAction",
            instruction_text="pick up cup",
            phase_count=2,
        )
    )
    ph1 = graph.add_node(
        MotionPhaseHypothesisNode(
            id="ph1",
            meta=_meta(run_id="r1"),
            phase_index=0,
            phase_name="reach",
            target_object="cup",
            description=None,
            symbol="reach_cup",
        )
    )
    ph2 = graph.add_node(
        MotionPhaseHypothesisNode(
            id="ph2",
            meta=_meta(run_id="r1"),
            phase_index=1,
            phase_name="grasp",
            target_object="cup",
            description=None,
            symbol="grasp_cup",
        )
    )

    for edge in [
        EvokesMotionPlanEdge(id="ep1", meta=_meta(run_id="r1"), src_id="i1", dst_id="plan1"),
        HasMotionPhaseEdge(id="ep2", meta=_meta(run_id="r1"), src_id="plan1", dst_id="ph1"),
        HasMotionPhaseEdge(id="ep3", meta=_meta(run_id="r1"), src_id="plan1", dst_id="ph2"),
    ]:
        graph.add_edge(edge)

    return graph, plan


# ---------------------------------------------------------------------------
# graph_context and GraphLinked
# ---------------------------------------------------------------------------


class TestGraphContext:
    def test_context_manager_activates_graph(self) -> None:
        graph, f1 = _framenet_graph()
        with graph_context(graph):
            roles = f1.roles
        assert len(roles) == 2

    def test_query_context_on_graph_activates_graph(self) -> None:
        graph, f1 = _framenet_graph()
        with graph.query_context():
            roles = f1.roles
        assert len(roles) == 2

    def test_no_context_raises_runtime_error(self) -> None:
        _, f1 = _framenet_graph()
        with pytest.raises(RuntimeError, match="No HypothesisGraph in context"):
            _ = f1.roles

    def test_nested_contexts_restore_outer_on_exit(self) -> None:
        graph1, f1 = _framenet_graph()
        graph2 = HypothesisGraph()  # empty — f1 not in it
        with graph_context(graph1):
            with graph_context(graph2):
                # inner context: graph2 is empty, f1 unknown → []
                assert f1.roles == []
            # outer context restored: f1 is in graph1
            assert len(f1.roles) == 2

    def test_context_exits_cleanly_on_exception(self) -> None:
        graph, f1 = _framenet_graph()
        with pytest.raises(ValueError):
            with graph_context(graph):
                raise ValueError("test error")
        # after exception, context is cleared — accessing roles must raise
        with pytest.raises(RuntimeError):
            _ = f1.roles


class TestFrameHypothesisNodeRoles:
    def test_roles_returns_all_attached_role_nodes(self) -> None:
        graph, f1 = _framenet_graph()
        with graph.query_context():
            roles = f1.roles
        assert len(roles) == 2
        assert all(isinstance(r, FrameRoleHypothesisNode) for r in roles)

    def test_roles_returns_correct_node_ids(self) -> None:
        graph, f1 = _framenet_graph()
        with graph.query_context():
            role_ids = {r.id for r in f1.roles}
        assert role_ids == {"r1", "r2"}

    def test_roles_preserves_edge_insertion_order(self) -> None:
        graph, f1 = _framenet_graph()
        with graph.query_context():
            role_ids = [r.id for r in f1.roles]
        assert role_ids == ["r1", "r2"]

    def test_frame_node_is_graphlinked_instance(self) -> None:
        _, f1 = _framenet_graph()
        assert isinstance(f1, GraphLinked)

    def test_frame_node_fields_unchanged_by_mixin(self) -> None:
        _, f1 = _framenet_graph()
        assert f1.frame == "Getting"
        assert f1.lexical_unit == "pick_up.v"
        assert f1.action_type == "PickUpAction"


class TestMotionPlanHypothesisNodePhases:
    def test_phases_returns_all_attached_phase_nodes(self) -> None:
        graph, plan = _flanagan_graph()
        with graph.query_context():
            phases = plan.phases
        assert len(phases) == 2
        assert all(isinstance(p, MotionPhaseHypothesisNode) for p in phases)

    def test_phases_returns_correct_node_ids(self) -> None:
        graph, plan = _flanagan_graph()
        with graph.query_context():
            phase_ids = {p.id for p in plan.phases}
        assert phase_ids == {"ph1", "ph2"}

    def test_phases_preserves_edge_insertion_order(self) -> None:
        graph, plan = _flanagan_graph()
        with graph.query_context():
            phase_ids = [p.id for p in plan.phases]
        assert phase_ids == ["ph1", "ph2"]

    def test_plan_node_is_graphlinked_instance(self) -> None:
        _, plan = _flanagan_graph()
        assert isinstance(plan, GraphLinked)

    def test_plan_node_fields_unchanged_by_mixin(self) -> None:
        _, plan = _flanagan_graph()
        assert plan.action_type == "PickUpAction"
        assert plan.phase_count == 2


class TestGraphLinkedMixin:
    def test_linked_sources_returns_incoming_edge_nodes(self) -> None:
        graph, f1 = _framenet_graph()
        with graph.query_context():
            # f1's linked_sources via EvokesFrameEdge should return i1
            sources = f1.linked_sources(EvokesFrameEdge, InstructionNode)
        assert len(sources) == 1
        assert sources[0].id == "i1"

    def test_linked_with_no_matching_edges_returns_empty(self) -> None:
        graph, f1 = _framenet_graph()
        with graph.query_context():
            # HasMotionPhaseEdge doesn't exist in this framenet graph
            result = f1.linked(HasMotionPhaseEdge)
        assert result == []
