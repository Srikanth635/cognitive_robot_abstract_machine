"""Tests for llmr.hypotheses concrete nodes and edges."""

from __future__ import annotations

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
    HypothesisMeta,
    InstructionNode,
    ProducedClaimEdge,
    ReasonerRunNode,
    SlotBindingEvidenceNode,
    SupportedByEdge,
    SymbolGroundingEvidenceNode,
)


class _FakeSymbol:
    def __init__(self, name: str) -> None:
        self.name = name


def _meta(
    *,
    status: ClaimStatus = ClaimStatus.HYPOTHESIS,
    grounding: GroundingState = GroundingState.TEXT_ONLY,
) -> HypothesisMeta:
    return HypothesisMeta(
        source_reasoner="framenet_reasoner",
        status=status,
        grounding=grounding,
    )


class TestNodes:
    def test_instruction_node(self) -> None:
        node = InstructionNode(
            id="i1",
            meta=_meta(),
            text="Pick up the milk",
            normalized_text="pick up the milk",
        )
        assert node.normalized_text == "pick up the milk"

    def test_frame_hypothesis_node(self) -> None:
        node = FrameHypothesisNode(
            id="f1",
            meta=_meta(),
            frame="Getting",
            lexical_unit="pick_up.v",
            framenet_label="picking_up_object",
            action_type="PickUpAction",
            instruction_text="pick up the milk",
        )
        assert node.frame == "Getting"

    def test_role_node_supports_grounded_case(self) -> None:
        node = FrameRoleHypothesisNode(
            id="r1",
            meta=_meta(
                status=ClaimStatus.SUPPORTED,
                grounding=GroundingState.SYMBOL_GROUNDED,
            ),
            role_family="core",
            role_name="theme",
            filler_text="milk",
            filler_kind="entity",
        )
        assert node.meta.status is ClaimStatus.SUPPORTED
        assert node.meta.grounding is GroundingState.SYMBOL_GROUNDED

    def test_role_node_supports_hypothesis_only_case(self) -> None:
        node = FrameRoleHypothesisNode(
            id="r2",
            meta=_meta(),
            role_family="core",
            role_name="goal",
            filler_text="robot grasp",
            filler_kind="abstract",
        )
        assert node.meta.status is ClaimStatus.HYPOTHESIS
        assert node.meta.grounding is GroundingState.TEXT_ONLY

    def test_symbol_grounding_evidence_node(self) -> None:
        symbol = _FakeSymbol("milk")
        node = SymbolGroundingEvidenceNode(
            id="g1",
            meta=_meta(grounding=GroundingState.SYMBOL_GROUNDED),
            query_text="milk",
            symbol_ref=symbol,
            symbol_type="WorldBody",
            grounding_method="symbol_grounder",
        )
        assert node.symbol_ref is symbol

    def test_reasoner_run_node(self) -> None:
        node = ReasonerRunNode(
            id="run1",
            meta=_meta(),
            reasoner_name="framenet_reasoner",
            run_id="run1",
            model_name="gpt-test",
            prompt_version="framenet_v1",
            action_type="PickUpAction",
            instruction_text="pick up the milk",
        )
        assert node.reasoner_name == "framenet_reasoner"


class TestEdges:
    def test_relation_names_are_stable(self) -> None:
        meta = _meta()
        produced = ProducedClaimEdge(id="e1", meta=meta, src_id="run1", dst_id="f1")
        evokes = EvokesFrameEdge(id="e2", meta=meta, src_id="i1", dst_id="f1")
        about = AboutActionEdge(id="e3", meta=meta, src_id="f1", dst_id="a1")
        has_role = HasRoleEdge(id="e4", meta=meta, src_id="f1", dst_id="r1")
        supported = SupportedByEdge(id="e5", meta=meta, src_id="r1", dst_id="ev1")
        grounded = GroundedByEdge(id="e6", meta=meta, src_id="r1", dst_id="ev2")

        assert produced.relation_name == "produced_claim"
        assert evokes.relation_name == "evokes_frame"
        assert about.relation_name == "about_action"
        assert has_role.relation_name == "has_role"
        assert supported.relation_name == "supported_by"
        assert grounded.relation_name == "grounded_by"
