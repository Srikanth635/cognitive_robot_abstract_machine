"""Tests for HypothesisGraph as an EQL-compatible instance provider.

HypothesisGraph.get_instances_of_type has the same signature as
SymbolGraph.get_instances_of_type, enabling hypothesis nodes to be
queried through EQL variable() with an explicit domain parameter.

These tests verify:
- The instance provider contract itself
- Attribute-based filtering over the returned instances
- GraphLinked navigability for multi-hop query patterns
"""

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
from llmr.hypotheses.common.nodes import ClaimNode, EvidenceNode
from llmr.hypotheses.projectors.flanagan.edges import EvokesMotionPlanEdge, HasMotionPhaseEdge


def _meta(
    reasoner: str = "framenet_reasoner",
    run_id: str | None = None,
    status: ClaimStatus = ClaimStatus.HYPOTHESIS,
    grounding: GroundingState = GroundingState.TEXT_ONLY,
) -> HypothesisMeta:
    return HypothesisMeta(
        source_reasoner=reasoner, status=status, grounding=grounding, run_id=run_id
    )


def _framenet_graph() -> tuple[HypothesisGraph, list[FrameRoleHypothesisNode]]:
    graph = HypothesisGraph()
    action_ref = object()
    symbol_ref = object()

    graph.add_node(InstructionNode(
        id="i1", meta=_meta(), text="pick up cup", normalized_text="pick up cup"
    ))
    graph.add_node(ActionNode(id="a1", meta=_meta(), action_ref=action_ref, action_type="PickUp"))
    graph.add_node(ReasonerRunNode(
        id="run1", meta=_meta(run_id="run1"), reasoner_name="framenet_reasoner",
        run_id="run1", model_name=None, prompt_version=None,
        action_type="PickUp", instruction_text="pick up cup",
    ))
    f1 = graph.add_node(FrameHypothesisNode(
        id="f1", meta=_meta(run_id="run1"), frame="Getting",
        lexical_unit="pick_up.v", framenet_label="pick", action_type="PickUp",
        instruction_text="pick up cup",
    ))
    r_theme = graph.add_node(FrameRoleHypothesisNode(
        id="r1", meta=_meta(run_id="run1"), role_family="core",
        role_name="theme", filler_text="cup", filler_kind="entity",
    ))
    r_goal = graph.add_node(FrameRoleHypothesisNode(
        id="r2", meta=_meta(run_id="run1"), role_family="core",
        role_name="goal", filler_text="robot hand", filler_kind="abstract",
    ))
    r_source = graph.add_node(FrameRoleHypothesisNode(
        id="r3", meta=_meta(run_id="run1"), role_family="peripheral",
        role_name="source", filler_text="table", filler_kind="entity",
    ))

    for edge in [
        EvokesFrameEdge(id="e1", meta=_meta(run_id="run1"), src_id="i1", dst_id="f1"),
        AboutActionEdge(id="e2", meta=_meta(run_id="run1"), src_id="f1", dst_id="a1"),
        ProducedClaimEdge(id="e3", meta=_meta(run_id="run1"), src_id="run1", dst_id="f1"),
        HasRoleEdge(id="e4", meta=_meta(run_id="run1"), src_id="f1", dst_id="r1"),
        HasRoleEdge(id="e5", meta=_meta(run_id="run1"), src_id="f1", dst_id="r2"),
        HasRoleEdge(id="e6", meta=_meta(run_id="run1"), src_id="f1", dst_id="r3"),
    ]:
        graph.add_edge(edge)

    return graph, [r_theme, r_goal, r_source]


class TestInstanceProviderContract:
    def test_returns_list(self) -> None:
        graph, _ = _framenet_graph()
        result = graph.get_instances_of_type(FrameRoleHypothesisNode)
        assert isinstance(result, list)

    def test_concrete_type_returns_all_instances(self) -> None:
        graph, roles = _framenet_graph()
        result = graph.get_instances_of_type(FrameRoleHypothesisNode)
        assert len(result) == 3
        assert {r.id for r in result} == {"r1", "r2", "r3"}

    def test_matches_domain_for_concrete_type(self) -> None:
        graph, _ = _framenet_graph()
        assert graph.get_instances_of_type(FrameRoleHypothesisNode) == graph.domain(
            FrameRoleHypothesisNode
        )

    def test_matches_domain_for_abstract_type(self) -> None:
        graph, _ = _framenet_graph()
        assert graph.get_instances_of_type(ClaimNode) == graph.domain(ClaimNode)

    def test_empty_for_unregistered_type(self) -> None:
        graph, _ = _framenet_graph()

        class _Unknown:
            pass

        assert graph.get_instances_of_type(_Unknown) == []

    def test_empty_graph_returns_empty(self) -> None:
        assert HypothesisGraph().get_instances_of_type(FrameRoleHypothesisNode) == []

    def test_result_contains_typed_instances(self) -> None:
        graph, _ = _framenet_graph()
        result = graph.get_instances_of_type(FrameRoleHypothesisNode)
        assert all(isinstance(r, FrameRoleHypothesisNode) for r in result)

    def test_preserves_insertion_order(self) -> None:
        graph, _ = _framenet_graph()
        ids = [r.id for r in graph.get_instances_of_type(FrameRoleHypothesisNode)]
        assert ids == ["r1", "r2", "r3"]


class TestEQLStyleAttributeFiltering:
    def test_filter_by_role_name(self) -> None:
        graph, _ = _framenet_graph()
        instances = graph.get_instances_of_type(FrameRoleHypothesisNode)
        theme_roles = [r for r in instances if r.role_name == "theme"]
        assert len(theme_roles) == 1
        assert theme_roles[0].id == "r1"

    def test_filter_by_filler_kind(self) -> None:
        graph, _ = _framenet_graph()
        instances = graph.get_instances_of_type(FrameRoleHypothesisNode)
        entity_roles = [r for r in instances if r.filler_kind == "entity"]
        assert {r.id for r in entity_roles} == {"r1", "r3"}

    def test_filter_by_role_family(self) -> None:
        graph, _ = _framenet_graph()
        instances = graph.get_instances_of_type(FrameRoleHypothesisNode)
        peripheral = [r for r in instances if r.role_family == "peripheral"]
        assert len(peripheral) == 1 and peripheral[0].id == "r3"

    def test_filter_by_status(self) -> None:
        graph, _ = _framenet_graph()
        all_claims = graph.get_instances_of_type(ClaimNode)
        hypothesis_claims = [c for c in all_claims if c.meta.status == ClaimStatus.HYPOTHESIS]
        assert len(hypothesis_claims) == len(all_claims)

    def test_multi_condition_filter(self) -> None:
        graph, _ = _framenet_graph()
        instances = graph.get_instances_of_type(FrameRoleHypothesisNode)
        matches = [
            r for r in instances
            if r.role_family == "core" and r.filler_kind == "entity"
        ]
        assert len(matches) == 1 and matches[0].id == "r1"

    def test_instance_provider_used_as_domain_yields_correct_count(self) -> None:
        graph, _ = _framenet_graph()
        domain = graph.get_instances_of_type(FrameHypothesisNode)
        getting_frames = [f for f in domain if f.frame == "Getting"]
        assert len(getting_frames) == 1

    def test_abstract_domain_query_via_instance_provider(self) -> None:
        graph, _ = _framenet_graph()
        all_claims = graph.get_instances_of_type(ClaimNode)
        frame_claims = [c for c in all_claims if isinstance(c, FrameHypothesisNode)]
        role_claims = [c for c in all_claims if isinstance(c, FrameRoleHypothesisNode)]
        assert len(frame_claims) == 1
        assert len(role_claims) == 3


class TestGraphLinkedNavigability:
    def test_frame_roles_accessible_inside_context(self) -> None:
        graph, _ = _framenet_graph()
        frames = graph.get_instances_of_type(FrameHypothesisNode)
        assert len(frames) == 1
        with graph_context(graph):
            roles = frames[0].roles
        assert len(roles) == 3
        assert all(isinstance(r, FrameRoleHypothesisNode) for r in roles)

    def test_filter_linked_roles_by_attribute(self) -> None:
        graph, _ = _framenet_graph()
        frames = graph.get_instances_of_type(FrameHypothesisNode)
        with graph_context(graph):
            entity_roles = [r for r in frames[0].roles if r.filler_kind == "entity"]
        assert {r.id for r in entity_roles} == {"r1", "r3"}

    def test_multi_hop_via_instance_provider_and_linked(self) -> None:
        graph, _ = _framenet_graph()
        with graph_context(graph):
            theme_fillers = [
                role.filler_text
                for frame in graph.get_instances_of_type(FrameHypothesisNode)
                for role in frame.roles
                if role.role_name == "theme"
            ]
        assert theme_fillers == ["cup"]

    def test_graphlinked_is_mixin_on_frame_node(self) -> None:
        graph, _ = _framenet_graph()
        frames = graph.get_instances_of_type(FrameHypothesisNode)
        assert isinstance(frames[0], GraphLinked)

    def test_roles_outside_context_raises(self) -> None:
        graph, _ = _framenet_graph()
        frames = graph.get_instances_of_type(FrameHypothesisNode)
        with pytest.raises(RuntimeError, match="No HypothesisGraph in context"):
            _ = frames[0].roles

    def test_flanagan_phases_navigable_via_instance_provider(self) -> None:
        graph = HypothesisGraph()
        meta = _meta(reasoner="flanagan_reasoner")
        graph.add_node(InstructionNode(id="i1", meta=meta, text="reach", normalized_text="reach"))
        plan = graph.add_node(MotionPlanHypothesisNode(
            id="p1", meta=meta, action_type="PickUp", instruction_text="reach", phase_count=2,
        ))
        ph1 = graph.add_node(MotionPhaseHypothesisNode(
            id="ph1", meta=meta, phase_index=0, phase_name="approach",
            target_object="cup", description=None, symbol="approach_cup",
        ))
        ph2 = graph.add_node(MotionPhaseHypothesisNode(
            id="ph2", meta=meta, phase_index=1, phase_name="grasp",
            target_object="cup", description=None, symbol="grasp_cup",
        ))
        for edge in [
            EvokesMotionPlanEdge(id="e1", meta=meta, src_id="i1", dst_id="p1"),
            HasMotionPhaseEdge(id="e2", meta=meta, src_id="p1", dst_id="ph1"),
            HasMotionPhaseEdge(id="e3", meta=meta, src_id="p1", dst_id="ph2"),
        ]:
            graph.add_edge(edge)

        plans = graph.get_instances_of_type(MotionPlanHypothesisNode)
        assert len(plans) == 1
        with graph_context(graph):
            phase_names = [p.phase_name for p in plans[0].phases]
        assert phase_names == ["approach", "grasp"]
