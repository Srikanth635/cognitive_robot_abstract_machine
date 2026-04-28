"""Generic graph view abstractions layered on top of HypothesisGraph."""

from __future__ import annotations

from abc import ABC
from dataclasses import dataclass
from typing import ClassVar

from typing_extensions import Optional, Tuple, TypeVar

from llmr.hypotheses.common.edges import AboutActionEdge
from llmr.hypotheses.elements import HypothesisEdge, HypothesisNode
from llmr.hypotheses.graph import HypothesisGraph

TNode = TypeVar("TNode", bound=HypothesisNode)
TEdge = TypeVar("TEdge", bound=HypothesisEdge)


@dataclass(frozen=True)
class HypothesisGraphView:
    """Thin typed facade over a generic HypothesisGraph."""

    graph: HypothesisGraph

    def nodes(self, node_type: type[TNode]) -> list[TNode]:
        """Return graph nodes of *node_type* preserving insertion order."""

        return self.graph.domain(node_type)

    def edges(self, edge_type: type[TEdge]) -> list[TEdge]:
        """Return graph edges of *edge_type* preserving insertion order."""

        return self.graph.edge_domain(edge_type)

    def get_node(self, node_id: str) -> Optional[HypothesisNode]:
        """Return the node with *node_id*, if present."""

        return self.graph.get_node(node_id)

    def get_edge(self, edge_id: str) -> Optional[HypothesisEdge]:
        """Return the edge with *edge_id*, if present."""

        return self.graph.get_edge(edge_id)

    def nodes_for_run(self, run_id: str) -> list[HypothesisNode]:
        """Return nodes tagged with *run_id*."""

        return self.graph.nodes_for_run(run_id)

    def nodes_from_reasoner(self, reasoner_name: str) -> list[HypothesisNode]:
        """Return nodes attributed to *reasoner_name*."""

        return self.graph.nodes_from_reasoner(reasoner_name)

    def subgraph_for_run(self, run_id: str) -> "HypothesisGraphView":
        """Return a same-view wrapper over the run-local hypothesis subgraph."""

        return type(self)(self.graph.subgraph_for_run(run_id))

    def subgraph_for_reasoner(self, reasoner_name: str) -> "HypothesisGraphView":
        """Return a same-view wrapper over the reasoner-local hypothesis subgraph."""

        return type(self)(self.graph.subgraph_for_reasoner(reasoner_name))


@dataclass(frozen=True)
class ReasonerGraphView(HypothesisGraphView, ABC):
    """Base view contract shared by reasoner-family query facades."""

    REASONER_NAME: ClassVar[str]
    ROOT_CLAIM_TYPES: ClassVar[Tuple[type[HypothesisNode], ...]] = ()
    CLAIM_TYPES: ClassVar[Tuple[type[HypothesisNode], ...]] = ()
    ACTION_EDGE_TYPE: ClassVar[type[HypothesisEdge]] = AboutActionEdge

    def claims(self) -> list[HypothesisNode]:
        """Return claim nodes belonging to this reasoner family."""

        return [
            node
            for cls in self.CLAIM_TYPES
            for node in self.graph.domain(cls)
            if node.meta.source_reasoner == self.REASONER_NAME
        ]

    def root_claims(self) -> list[HypothesisNode]:
        """Return top-level claim nodes belonging to this reasoner family."""

        return [
            node
            for cls in self.ROOT_CLAIM_TYPES
            for node in self.graph.domain(cls)
            if node.meta.source_reasoner == self.REASONER_NAME
        ]

    def claims_for_run(self, run_id: str) -> list[HypothesisNode]:
        """Return this family's claims tagged with *run_id*."""

        return [
            node
            for cls in self.CLAIM_TYPES
            for node in self.graph.domain(cls)
            if node.meta.source_reasoner == self.REASONER_NAME
            and node.meta.run_id == run_id
        ]

    def claims_for_action(self, action_ref: object) -> list[HypothesisNode]:
        """Return this family's claims in the action-local hypothesis closure."""

        return self.action_subgraph(action_ref).claims()

    def action_subgraph(self, action_ref: object) -> "ReasonerGraphView":
        """Return a same-view wrapper over the action-local hypothesis subgraph."""

        return type(self)(self.graph.subgraph_for_action(action_ref))

    def run_subgraph(self, run_id: str) -> "ReasonerGraphView":
        """Return a same-view wrapper over the run-local hypothesis subgraph."""

        return type(self)(self.graph.subgraph_for_run(run_id))
