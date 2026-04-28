"""Tests for HypothesisGraph MRO-aware type index.

Verifies that domain(cls) returns instances of cls and all its concrete
subclasses, that the index is updated on removal/clear, and that sibling
branches of the hierarchy don't contaminate each other.

Type hierarchy under test:
  HypothesisNode (abstract, excluded from index)
    ContextNode  (abstract)  → InstructionNode, ActionNode, ReasonerRunNode
    ClaimNode    (abstract)  → ReasonerClaimNode (abstract)
                               → FrameHypothesisNode, FrameRoleHypothesisNode
                               → MotionPlanHypothesisNode, MotionPhaseHypothesisNode
    EvidenceNode (abstract)  → SlotBindingEvidenceNode, SymbolGroundingEvidenceNode
"""

from __future__ import annotations

import pytest

from llmr.hypotheses import (
    ActionNode,
    ClaimStatus,
    FrameHypothesisNode,
    FrameRoleHypothesisNode,
    GroundingState,
    HypothesisGraph,
    HypothesisMeta,
    InstructionNode,
    MotionPhaseHypothesisNode,
    MotionPlanHypothesisNode,
    ReasonerRunNode,
    SlotBindingEvidenceNode,
    SymbolGroundingEvidenceNode,
)
from llmr.hypotheses.common.nodes import (
    ClaimNode,
    ContextNode,
    EvidenceNode,
    ReasonerClaimNode,
)


def _meta(reasoner: str = "test") -> HypothesisMeta:
    return HypothesisMeta(
        source_reasoner=reasoner,
        status=ClaimStatus.HYPOTHESIS,
        grounding=GroundingState.TEXT_ONLY,
    )


def _full_graph() -> HypothesisGraph:
    """Return a graph with one node of every concrete type."""
    graph = HypothesisGraph()
    action_ref = object()
    symbol_ref = object()

    graph.add_node(InstructionNode(id="i1", meta=_meta(), text="pick up", normalized_text="pick up"))
    graph.add_node(ActionNode(id="a1", meta=_meta(), action_ref=action_ref, action_type="PickUp"))
    graph.add_node(ReasonerRunNode(
        id="run1", meta=_meta(reasoner="framenet_reasoner"), reasoner_name="framenet_reasoner",
        run_id="run1", model_name=None, prompt_version=None,
        action_type="PickUp", instruction_text="pick up",
    ))
    graph.add_node(FrameHypothesisNode(
        id="f1", meta=_meta(), frame="Getting", lexical_unit="pick_up.v",
        framenet_label="pick", action_type="PickUp", instruction_text="pick up",
    ))
    graph.add_node(FrameRoleHypothesisNode(
        id="r1", meta=_meta(), role_family="core", role_name="theme",
        filler_text="cup", filler_kind="entity",
    ))
    graph.add_node(MotionPlanHypothesisNode(
        id="p1", meta=_meta(), action_type="PickUp", instruction_text="pick up", phase_count=1,
    ))
    graph.add_node(MotionPhaseHypothesisNode(
        id="mp1", meta=_meta(), phase_index=0, phase_name="reach",
        target_object="cup", description=None, symbol="reach_cup",
    ))
    graph.add_node(SlotBindingEvidenceNode(
        id="ev1", meta=_meta(), slot_name="arm", value_ref=None, value_repr="left",
    ))
    graph.add_node(SymbolGroundingEvidenceNode(
        id="ev2", meta=_meta(), query_text="cup", symbol_ref=symbol_ref,
        symbol_type="_FakeSym", grounding_method="name_match",
    ))
    return graph


class TestDomainConcreteTypes:
    def test_instruction_node_domain_is_exact(self) -> None:
        graph = _full_graph()
        result = graph.domain(InstructionNode)
        assert len(result) == 1
        assert result[0].id == "i1"

    def test_frame_hypothesis_node_domain_is_exact(self) -> None:
        graph = _full_graph()
        result = graph.domain(FrameHypothesisNode)
        assert len(result) == 1
        assert result[0].id == "f1"

    def test_evidence_concrete_subtypes_dont_overlap(self) -> None:
        graph = _full_graph()
        slot_ev = graph.domain(SlotBindingEvidenceNode)
        sym_ev = graph.domain(SymbolGroundingEvidenceNode)
        assert len(slot_ev) == 1 and slot_ev[0].id == "ev1"
        assert len(sym_ev) == 1 and sym_ev[0].id == "ev2"
        assert {n.id for n in slot_ev}.isdisjoint({n.id for n in sym_ev})


class TestDomainAbstractSuperclasses:
    def test_context_node_covers_all_context_subtypes(self) -> None:
        graph = _full_graph()
        ids = {n.id for n in graph.domain(ContextNode)}
        assert ids == {"i1", "a1", "run1"}

    def test_claim_node_covers_all_claim_subtypes(self) -> None:
        graph = _full_graph()
        ids = {n.id for n in graph.domain(ClaimNode)}
        assert ids == {"f1", "r1", "p1", "mp1"}

    def test_reasoner_claim_node_covers_all_reasoner_claim_subtypes(self) -> None:
        graph = _full_graph()
        ids = {n.id for n in graph.domain(ReasonerClaimNode)}
        assert ids == {"f1", "r1", "p1", "mp1"}

    def test_evidence_node_covers_both_evidence_subtypes(self) -> None:
        graph = _full_graph()
        ids = {n.id for n in graph.domain(EvidenceNode)}
        assert ids == {"ev1", "ev2"}

    def test_sibling_branches_dont_cross_contaminate(self) -> None:
        graph = _full_graph()
        context_ids = {n.id for n in graph.domain(ContextNode)}
        claim_ids = {n.id for n in graph.domain(ClaimNode)}
        evidence_ids = {n.id for n in graph.domain(EvidenceNode)}
        assert context_ids.isdisjoint(claim_ids)
        assert context_ids.isdisjoint(evidence_ids)
        assert claim_ids.isdisjoint(evidence_ids)

    def test_frame_hypothesis_in_claim_but_not_context(self) -> None:
        graph = _full_graph()
        claim_ids = {n.id for n in graph.domain(ClaimNode)}
        context_ids = {n.id for n in graph.domain(ContextNode)}
        assert "f1" in claim_ids
        assert "f1" not in context_ids


class TestDomainUnknownType:
    def test_domain_of_unregistered_type_returns_empty(self) -> None:
        graph = _full_graph()

        class _Phantom:
            pass

        assert graph.domain(_Phantom) == []

    def test_domain_of_empty_graph_returns_empty(self) -> None:
        assert HypothesisGraph().domain(FrameHypothesisNode) == []


class TestMROIndexAfterRemoval:
    def test_remove_concrete_node_drops_from_concrete_domain(self) -> None:
        graph = _full_graph()
        graph.remove_node("f1")
        assert graph.domain(FrameHypothesisNode) == []

    def test_remove_concrete_node_drops_from_abstract_domain(self) -> None:
        graph = _full_graph()
        initial_count = len(graph.domain(ClaimNode))
        graph.remove_node("f1")
        assert len(graph.domain(ClaimNode)) == initial_count - 1
        assert all(n.id != "f1" for n in graph.domain(ClaimNode))

    def test_remove_one_sibling_does_not_affect_others(self) -> None:
        graph = _full_graph()
        graph.remove_node("r1")
        frames = graph.domain(FrameHypothesisNode)
        assert len(frames) == 1 and frames[0].id == "f1"

    def test_clear_empties_all_domains(self) -> None:
        graph = _full_graph()
        graph.clear()
        for cls in (ClaimNode, ContextNode, EvidenceNode, ReasonerClaimNode,
                    FrameHypothesisNode, FrameRoleHypothesisNode):
            assert graph.domain(cls) == [], f"domain({cls.__name__}) not empty after clear"


class TestMROInsertionOrder:
    def test_domain_preserves_insertion_order_for_concrete_type(self) -> None:
        graph = HypothesisGraph()
        for i in range(5):
            graph.add_node(FrameRoleHypothesisNode(
                id=f"r{i}", meta=_meta(), role_family="core",
                role_name=f"role{i}", filler_text=f"val{i}", filler_kind="entity",
            ))
        ids = [n.id for n in graph.domain(FrameRoleHypothesisNode)]
        assert ids == [f"r{i}" for i in range(5)]

    def test_abstract_domain_preserves_inter_type_insertion_order(self) -> None:
        graph = HypothesisGraph()
        graph.add_node(FrameHypothesisNode(
            id="f1", meta=_meta(), frame="A", lexical_unit="a.v",
            framenet_label="a", action_type="A", instruction_text=None,
        ))
        graph.add_node(FrameRoleHypothesisNode(
            id="r1", meta=_meta(), role_family="core",
            role_name="theme", filler_text="x", filler_kind="entity",
        ))
        graph.add_node(FrameHypothesisNode(
            id="f2", meta=_meta(), frame="B", lexical_unit="b.v",
            framenet_label="b", action_type="B", instruction_text=None,
        ))
        frame_ids = [n.id for n in graph.domain(FrameHypothesisNode)]
        assert frame_ids == ["f1", "f2"]
