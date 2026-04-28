"""Tests for llmr.hypotheses.algorithms — graph-algorithm-powered queries."""

from __future__ import annotations

import pytest

from llmr.hypotheses import (
    AboutActionEdge,
    ActionNode,
    ClaimStatus,
    EvokesFrameEdge,
    FrameHypothesisNode,
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
    conflicting_role_claims,
    hypothesis_closure,
    invalidate_from_symbol,
    reasoning_chain,
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
    reasoner: str = "framenet_reasoner",
) -> HypothesisMeta:
    return HypothesisMeta(
        source_reasoner=reasoner,
        status=status,
        grounding=grounding,
        run_id=run_id,
    )


def _build_sample_graph() -> tuple[HypothesisGraph, dict[str, object]]:
    """Build a graph with one grounded role for use across algorithm tests."""
    graph = HypothesisGraph()
    action = _FakeAction()
    milk = _FakeSymbol("milk")

    i1 = graph.add_node(
        InstructionNode(
            id="i1",
            meta=_meta(),
            text="pick up the milk",
            normalized_text="pick up the milk",
        )
    )
    a1 = graph.add_node(
        ActionNode(id="a1", meta=_meta(), action_ref=action, action_type="PickUpAction")
    )
    run1 = graph.add_node(
        ReasonerRunNode(
            id="run1",
            meta=_meta(run_id="run1"),
            reasoner_name="framenet_reasoner",
            run_id="run1",
            model_name=None,
            prompt_version=None,
            action_type="PickUpAction",
            instruction_text="pick up the milk",
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
            instruction_text="pick up the milk",
        )
    )
    r1 = graph.add_node(
        FrameRoleHypothesisNode(
            id="r1",
            meta=_meta(
                grounding=GroundingState.SYMBOL_GROUNDED,
                run_id="run1",
            ),
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
    ev1 = graph.add_node(
        SlotBindingEvidenceNode(
            id="ev1",
            meta=_meta(grounding=GroundingState.SLOT_ALIGNED, run_id="run1"),
            slot_name="object_designator",
            value_ref=milk,
            value_repr="milk",
        )
    )
    ev2 = graph.add_node(
        SymbolGroundingEvidenceNode(
            id="ev2",
            meta=_meta(grounding=GroundingState.SYMBOL_GROUNDED, run_id="run1"),
            query_text="milk",
            symbol_ref=milk,
            symbol_type="WorldBody",
            grounding_method="symbol_grounder",
        )
    )

    for edge in [
        EvokesFrameEdge(id="e1", meta=_meta(run_id="run1"), src_id="i1", dst_id="f1"),
        AboutActionEdge(id="e2", meta=_meta(run_id="run1"), src_id="f1", dst_id="a1"),
        ProducedClaimEdge(id="e3", meta=_meta(run_id="run1"), src_id="run1", dst_id="f1"),
        ProducedClaimEdge(id="e4", meta=_meta(run_id="run1"), src_id="run1", dst_id="r1"),
        ProducedClaimEdge(id="e5", meta=_meta(run_id="run1"), src_id="run1", dst_id="r2"),
        HasRoleEdge(id="e6", meta=_meta(run_id="run1"), src_id="f1", dst_id="r1"),
        HasRoleEdge(id="e7", meta=_meta(run_id="run1"), src_id="f1", dst_id="r2"),
        SupportedByEdge(id="e8", meta=_meta(run_id="run1"), src_id="r1", dst_id="ev1"),
        GroundedByEdge(id="e9", meta=_meta(run_id="run1"), src_id="r1", dst_id="ev2"),
    ]:
        graph.add_edge(edge)

    return graph, {"action": action, "milk": milk, "i1": i1, "f1": f1, "r1": r1}


# ---------------------------------------------------------------------------
# invalidate_from_symbol
# ---------------------------------------------------------------------------


class TestInvalidateFromSymbol:
    def test_marks_dependent_claim_nodes_as_refuted(self) -> None:
        graph, refs = _build_sample_graph()
        affected = invalidate_from_symbol(graph, refs["milk"])

        affected_ids = {n.id for n in affected}
        # r1 is grounded to milk; f1 is the frame that contains r1
        assert "r1" in affected_ids
        assert "f1" in affected_ids

    def test_does_not_affect_nodes_not_dependent_on_symbol(self) -> None:
        graph, refs = _build_sample_graph()
        invalidate_from_symbol(graph, refs["milk"])

        # r2 is not grounded to milk — must stay unchanged
        r2 = graph.get_node("r2")
        assert r2 is not None
        assert r2.meta.status == ClaimStatus.HYPOTHESIS

    def test_affected_nodes_have_new_status(self) -> None:
        graph, refs = _build_sample_graph()
        invalidate_from_symbol(graph, refs["milk"], new_status=ClaimStatus.SUPERSEDED)

        r1 = graph.get_node("r1")
        assert r1 is not None
        assert r1.meta.status == ClaimStatus.SUPERSEDED

    def test_unknown_symbol_returns_empty(self) -> None:
        graph, _ = _build_sample_graph()
        unrelated_symbol = _FakeSymbol("cup")
        affected = invalidate_from_symbol(graph, unrelated_symbol)
        assert affected == []

    def test_no_duplicate_affected_nodes(self) -> None:
        graph, refs = _build_sample_graph()
        affected = invalidate_from_symbol(graph, refs["milk"])
        assert len(affected) == len({n.id for n in affected})


# ---------------------------------------------------------------------------
# reasoning_chain
# ---------------------------------------------------------------------------


class TestReasoningChain:
    def test_chain_from_instruction_to_role(self) -> None:
        graph, _ = _build_sample_graph()
        chain = reasoning_chain(graph, "r1")
        chain_ids = [n.id for n in chain]
        # Must start at an instruction and end at r1
        assert chain_ids[0] == "i1"
        assert chain_ids[-1] == "r1"

    def test_chain_passes_through_frame(self) -> None:
        graph, _ = _build_sample_graph()
        chain = reasoning_chain(graph, "r1")
        chain_ids = [n.id for n in chain]
        # Shortest undirected path: i1 → f1 → r1
        assert "f1" in chain_ids

    def test_chain_to_frame_itself(self) -> None:
        graph, _ = _build_sample_graph()
        chain = reasoning_chain(graph, "f1")
        chain_ids = [n.id for n in chain]
        assert chain_ids[0] == "i1"
        assert chain_ids[-1] == "f1"

    def test_unknown_node_returns_empty(self) -> None:
        graph, _ = _build_sample_graph()
        assert reasoning_chain(graph, "nonexistent") == []

    def test_chain_is_shortest_path(self) -> None:
        graph, _ = _build_sample_graph()
        chain_r1 = reasoning_chain(graph, "r1")
        chain_f1 = reasoning_chain(graph, "f1")
        # r1 chain must be strictly longer than f1 chain
        assert len(chain_r1) > len(chain_f1)


# ---------------------------------------------------------------------------
# conflicting_role_claims
# ---------------------------------------------------------------------------


class TestConflictingRoleClaims:
    def test_no_conflicts_in_clean_graph(self) -> None:
        graph, refs = _build_sample_graph()
        assert conflicting_role_claims(graph, refs["action"]) == []

    def test_detects_same_role_different_fillers(self) -> None:
        graph, refs = _build_sample_graph()
        action = refs["action"]

        # Add a second run that claims "theme" is "cup" instead of "milk"
        r_conflict = graph.add_node(
            FrameRoleHypothesisNode(
                id="r_conflict",
                meta=_meta(run_id="run1"),
                role_family="core",
                role_name="theme",
                filler_text="cup",
                filler_kind="entity",
            )
        )
        graph.add_edge(
            HasRoleEdge(
                id="e_conflict",
                meta=_meta(run_id="run1"),
                src_id="f1",
                dst_id="r_conflict",
            )
        )

        conflicts = conflicting_role_claims(graph, action)
        assert len(conflicts) == 1
        conflict_ids = {c.id for pair in conflicts for c in pair}
        assert "r1" in conflict_ids
        assert "r_conflict" in conflict_ids

    def test_same_filler_is_not_a_conflict(self) -> None:
        graph, refs = _build_sample_graph()
        action = refs["action"]

        # Duplicate "theme: milk" — same filler, should not be flagged
        r_dup = graph.add_node(
            FrameRoleHypothesisNode(
                id="r_dup",
                meta=_meta(run_id="run1"),
                role_family="core",
                role_name="theme",
                filler_text="milk",
                filler_kind="entity",
            )
        )
        graph.add_edge(
            HasRoleEdge(
                id="e_dup",
                meta=_meta(run_id="run1"),
                src_id="f1",
                dst_id="r_dup",
            )
        )

        assert conflicting_role_claims(graph, action) == []

    def test_unknown_action_returns_empty(self) -> None:
        graph, _ = _build_sample_graph()
        assert conflicting_role_claims(graph, _FakeAction()) == []


# ---------------------------------------------------------------------------
# hypothesis_closure
# ---------------------------------------------------------------------------


class TestHypothesisClosure:
    def test_closure_includes_ancestors_and_descendants(self) -> None:
        graph, _ = _build_sample_graph()
        closure = hypothesis_closure(graph, "r1")
        closure_ids = {n.id for n in closure.iter_nodes()}

        # Descendants of r1: ev1, ev2
        assert {"ev1", "ev2"}.issubset(closure_ids)
        # Ancestors of r1: f1, run1, i1 (i1 via i1→f1→r1 and run1→r1)
        assert {"f1", "run1", "i1"}.issubset(closure_ids)
        # r1 itself
        assert "r1" in closure_ids

    def test_closure_excludes_unrelated_nodes(self) -> None:
        graph, _ = _build_sample_graph()
        closure = hypothesis_closure(graph, "r1")
        closure_ids = {n.id for n in closure.iter_nodes()}

        # a1 has no directed path to/from r1 (f1→a1 goes away from r1's subtree)
        # r2 is reachable from run1 and f1, so it IS an ancestor sibling — but
        # r2 is NOT a descendant of r1 and is NOT an ancestor of r1.
        # r2 is reachable FROM run1 (run1→r2) and FROM f1 (f1→r2), both of which
        # are ancestors of r1. So r2 is NOT in the ancestors/descendants of r1.
        assert "r2" not in closure_ids
        assert "a1" not in closure_ids

    def test_closure_edges_are_restricted_to_closure(self) -> None:
        graph, _ = _build_sample_graph()
        closure = hypothesis_closure(graph, "r1")
        for edge in closure.iter_edges():
            assert closure.has_node(edge.src_id)
            assert closure.has_node(edge.dst_id)

    def test_unknown_node_returns_empty_graph(self) -> None:
        graph, _ = _build_sample_graph()
        empty = hypothesis_closure(graph, "nonexistent")
        assert empty.node_count == 0
        assert empty.edge_count == 0

    def test_closure_of_instruction_contains_full_subgraph(self) -> None:
        graph, _ = _build_sample_graph()
        # i1 has no ancestors; its descendants should cover most of the graph
        closure = hypothesis_closure(graph, "i1")
        closure_ids = {n.id for n in closure.iter_nodes()}
        # f1, r1, r2, ev1, ev2 are all descendants of i1
        assert {"f1", "r1", "r2", "ev1", "ev2"}.issubset(closure_ids)
        assert "i1" in closure_ids
