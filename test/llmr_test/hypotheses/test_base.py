"""Tests for llmr.hypotheses.base."""

from __future__ import annotations

from dataclasses import dataclass

from llmr.hypotheses.elements import (
    ClaimStatus,
    GroundingState,
    HypothesisEdge,
    HypothesisMeta,
    HypothesisNode,
)


@dataclass
class _DummyNode(HypothesisNode):
    value: str


@dataclass
class _DummyEdge(HypothesisEdge):
    RELATION_NAME = "dummy"


class TestHypothesisMeta:
    def test_defaults(self) -> None:
        meta = HypothesisMeta(source_reasoner="framenet_reasoner")
        assert meta.status is ClaimStatus.HYPOTHESIS
        assert meta.grounding is GroundingState.TEXT_ONLY
        assert meta.created_at is not None

    def test_short_run_id_compacts_long_run_ids(self) -> None:
        meta = HypothesisMeta(
            source_reasoner="framenet_reasoner",
            run_id="1234567890abcdef1234567890abcdef",
        )
        assert meta.short_run_id == "12345678"


class TestEnums:
    def test_claim_status_values_are_stable(self) -> None:
        assert ClaimStatus.SUPPORTED.value == "supported"
        assert ClaimStatus.REFUTED.value == "refuted"

    def test_grounding_state_values_are_stable(self) -> None:
        assert GroundingState.SLOT_ALIGNED.value == "slot_aligned"
        assert GroundingState.SYMBOL_GROUNDED.value == "symbol_grounded"


class TestAbstractContracts:
    def test_dummy_node_instantiates(self) -> None:
        node = _DummyNode(
            id="n1",
            meta=HypothesisMeta(source_reasoner="dummy"),
            value="x",
        )
        assert node.id == "n1"
        assert node.display_id == "n1"

    def test_dummy_node_display_id_compacts_long_structured_ids(self) -> None:
        node = _DummyNode(
            id="framenet_reasoner:1234567890abcdef1234567890abcdef:node:frame",
            meta=HypothesisMeta(source_reasoner="dummy"),
            value="x",
        )
        assert node.display_id == "framenet_reasoner:12345678:node:frame"

    def test_dummy_edge_instantiates(self) -> None:
        edge = _DummyEdge(
            id="e1",
            meta=HypothesisMeta(source_reasoner="dummy"),
            src_id="n1",
            dst_id="n2",
        )
        assert edge.relation_name == "dummy"
