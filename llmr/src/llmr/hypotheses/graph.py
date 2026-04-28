"""Graph-native repository for llmr hypothesis nodes and edges."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable, Iterable

from rustworkx import PyDiGraph
from typing_extensions import Any, Dict, List, Optional, Set, Tuple, TypeVar

from llmr.hypotheses.elements import (
    ClaimStatus,
    GroundingState,
    HypothesisEdge,
    HypothesisNode,
)
from llmr.hypotheses.common.nodes import (
    ActionNode,
    EvidenceNode,
    InstructionNode,
    ReasonerRunNode,
    SymbolGroundingEvidenceNode,
)
from llmr.hypotheses.common.edges import AboutActionEdge

if TYPE_CHECKING:
    from llmr.hypotheses.projection import HypothesisProjection

TNode = TypeVar("TNode", bound=HypothesisNode)
TEdge = TypeVar("TEdge", bound=HypothesisEdge)
EdgeCondition = Callable[[HypothesisEdge], bool]


def _normalize_instruction_text(text: str) -> str:
    """Normalize instruction text for deduplication and lookup."""

    return " ".join(text.split()).strip().lower()


@dataclass
class HypothesisGraph:
    """Graph-native repository of epistemic nodes and relations.

    Nodes and edges are stored directly inside a ``rustworkx.PyDiGraph`` while
    stable id/type indexes are maintained alongside it for EQL domains and
    metadata-based lookup.
    """

    _graph: PyDiGraph[HypothesisNode, HypothesisEdge] = field(
        default_factory=PyDiGraph, init=False, repr=False
    )
    _node_indices_by_id: Dict[str, int] = field(default_factory=dict, init=False)
    _edge_indices_by_id: Dict[str, int] = field(default_factory=dict, init=False)
    _nodes_by_id: Dict[str, HypothesisNode] = field(default_factory=dict, init=False)
    _edges_by_id: Dict[str, HypothesisEdge] = field(default_factory=dict, init=False)
    _nodes_by_type: Dict[type, List[HypothesisNode]] = field(
        default_factory=dict, init=False
    )
    _edges_by_type: Dict[type, List[HypothesisEdge]] = field(
        default_factory=dict, init=False
    )
    _instruction_nodes_by_normalized_text: Dict[str, InstructionNode] = field(
        default_factory=dict, init=False
    )
    _action_nodes_by_ref_id: Dict[int, ActionNode] = field(
        default_factory=dict, init=False
    )
    _nodes_by_status: Dict[ClaimStatus, List[HypothesisNode]] = field(
        default_factory=dict, init=False
    )
    _nodes_by_grounding: Dict[GroundingState, List[HypothesisNode]] = field(
        default_factory=dict, init=False
    )
    _nodes_by_run_id: Dict[str, List[HypothesisNode]] = field(
        default_factory=dict, init=False
    )
    _nodes_by_reasoner: Dict[str, List[HypothesisNode]] = field(
        default_factory=dict, init=False
    )
    _edges_by_run_id: Dict[str, List[HypothesisEdge]] = field(
        default_factory=dict, init=False
    )
    _edges_by_reasoner: Dict[str, List[HypothesisEdge]] = field(
        default_factory=dict, init=False
    )
    _edges_by_relation: Dict[Tuple[type, str, str], List[HypothesisEdge]] = field(
        default_factory=dict, init=False
    )
    _edges_by_relation_and_run: Dict[
        Tuple[type, str, str, Optional[str]], List[HypothesisEdge]
    ] = field(default_factory=dict, init=False)
    _symbol_groundings_by_symbol_ref_id: Dict[int, List[SymbolGroundingEvidenceNode]] = field(default_factory=dict, init=False)

    def add_node(self, node: TNode) -> TNode:
        """Insert *node* into the graph, reusing stable context nodes when possible."""

        existing = self._get_existing_context_node(node)
        if existing is not None:
            return existing  # type: ignore[return-value]

        existing = self._nodes_by_id.get(node.id)
        if existing is not None:
            if existing != node:
                raise ValueError(f"node id collision for {node.id!r}")
            return existing  # type: ignore[return-value]

        node_index = self._graph.add_node(node)
        self._node_indices_by_id[node.id] = node_index
        self._nodes_by_id[node.id] = node
        self._index_node(node)
        return node

    def add_edge(self, edge: TEdge) -> TEdge:
        """Insert *edge* into the graph after verifying endpoint integrity."""

        source_index = self._node_indices_by_id.get(edge.src_id)
        if source_index is None:
            raise KeyError(f"edge source node {edge.src_id!r} not found")
        target_index = self._node_indices_by_id.get(edge.dst_id)
        if target_index is None:
            raise KeyError(f"edge destination node {edge.dst_id!r} not found")

        existing = self._edges_by_id.get(edge.id)
        if existing is not None:
            if existing != edge:
                raise ValueError(f"edge id collision for {edge.id!r}")
            return existing  # type: ignore[return-value]

        edge_index = self._graph.add_edge(source_index, target_index, edge)
        self._edge_indices_by_id[edge.id] = edge_index
        self._edges_by_id[edge.id] = edge
        self._index_edge(edge)
        return edge

    def add_projection(self, projection: "HypothesisProjection") -> None:
        """Insert a typed projection into the graph."""

        for node in projection.nodes:
            self.add_node(node)
        for edge in projection.edges:
            self.add_edge(edge)

    def get_node(self, node_id: str) -> Optional[HypothesisNode]:
        """Return the node with *node_id*, if present."""

        return self._nodes_by_id.get(node_id)

    def get_edge(self, edge_id: str) -> Optional[HypothesisEdge]:
        """Return the edge with *edge_id*, if present."""

        return self._edges_by_id.get(edge_id)

    def has_node(self, node_id: str) -> bool:
        """Return whether a node with *node_id* exists."""

        return node_id in self._nodes_by_id

    def has_edge(self, edge_id: str) -> bool:
        """Return whether an edge with *edge_id* exists."""

        return edge_id in self._edges_by_id

    def iter_nodes(self) -> Iterable[HypothesisNode]:
        """Iterate over nodes preserving graph insertion order."""

        return iter(self._nodes_by_id.values())

    def iter_edges(self) -> Iterable[HypothesisEdge]:
        """Iterate over edges preserving graph insertion order."""

        return iter(self._edges_by_id.values())

    def out_edges(
        self,
        node_id: str,
        edge_type: Optional[type[TEdge]] = None,
    ) -> List[TEdge]:
        """Return outgoing edges for *node_id*, optionally filtered by type."""

        node_index = self._node_indices_by_id.get(node_id)
        if node_index is None:
            return []
        weighted_edges = sorted(
            self._graph.out_edges(node_index),
            key=lambda item: self._edge_indices_by_id[item[2].id],
        )
        edges = [edge for _, _, edge in weighted_edges]
        if edge_type is None:
            return list(edges)  # type: ignore[return-value]
        return [edge for edge in edges if isinstance(edge, edge_type)]  # type: ignore[return-value]

    def in_edges(
        self,
        node_id: str,
        edge_type: Optional[type[TEdge]] = None,
    ) -> List[TEdge]:
        """Return incoming edges for *node_id*, optionally filtered by type."""

        node_index = self._node_indices_by_id.get(node_id)
        if node_index is None:
            return []
        weighted_edges = sorted(
            self._graph.in_edges(node_index),
            key=lambda item: self._edge_indices_by_id[item[2].id],
        )
        edges = [edge for _, _, edge in weighted_edges]
        if edge_type is None:
            return list(edges)  # type: ignore[return-value]
        return [edge for edge in edges if isinstance(edge, edge_type)]  # type: ignore[return-value]

    def get_outgoing_edges_with_condition(
        self,
        node_id: str,
        condition: EdgeCondition,
        edge_type: Optional[type[TEdge]] = None,
    ) -> List[HypothesisEdge]:
        """Return outgoing edges for *node_id* that satisfy *condition*."""

        return [
            edge
            for edge in self.out_edges(node_id, edge_type=edge_type)
            if condition(edge)
        ]

    def get_incoming_edges_with_condition(
        self,
        node_id: str,
        condition: EdgeCondition,
        edge_type: Optional[type[TEdge]] = None,
    ) -> List[HypothesisEdge]:
        """Return incoming edges for *node_id* that satisfy *condition*."""

        return [
            edge
            for edge in self.in_edges(node_id, edge_type=edge_type)
            if condition(edge)
        ]

    def get_targets(
        self,
        node_id: str,
        edge_type: Optional[type[TEdge]] = None,
        node_type: Optional[type[TNode]] = None,
    ) -> List[TNode]:
        """Return target nodes reachable from *node_id* via outgoing edges."""

        node_index = self._node_indices_by_id.get(node_id)
        if node_index is None:
            return []
        targets: List[HypothesisNode] = []
        for _, target_index, edge in sorted(
            self._graph.out_edges(node_index),
            key=lambda item: self._edge_indices_by_id[item[2].id],
        ):
            if edge_type is not None and not isinstance(edge, edge_type):
                continue
            node = self._graph.get_node_data(target_index)
            if node_type is not None and not isinstance(node, node_type):
                continue
            targets.append(node)
        return targets  # type: ignore[return-value]

    def get_sources(
        self,
        node_id: str,
        edge_type: Optional[type[TEdge]] = None,
        node_type: Optional[type[TNode]] = None,
    ) -> List[TNode]:
        """Return source nodes that point to *node_id* via incoming edges."""

        node_index = self._node_indices_by_id.get(node_id)
        if node_index is None:
            return []
        sources: List[HypothesisNode] = []
        for source_index, _, edge in sorted(
            self._graph.in_edges(node_index),
            key=lambda item: self._edge_indices_by_id[item[2].id],
        ):
            if edge_type is not None and not isinstance(edge, edge_type):
                continue
            node = self._graph.get_node_data(source_index)
            if node_type is not None and not isinstance(node, node_type):
                continue
            sources.append(node)
        return sources  # type: ignore[return-value]

    def neighbors(
        self,
        node_id: str,
        edge_type: Optional[type[TEdge]] = None,
        node_type: Optional[type[TNode]] = None,
    ) -> List[TNode]:
        """Return incoming and outgoing neighboring nodes for *node_id*."""

        neighbors: List[HypothesisNode] = []
        seen_ids: set[str] = set()
        for node in self.get_targets(node_id, edge_type=edge_type, node_type=node_type):
            if node.id not in seen_ids:
                neighbors.append(node)
                seen_ids.add(node.id)
        for node in self.get_sources(node_id, edge_type=edge_type, node_type=node_type):
            if node.id not in seen_ids:
                neighbors.append(node)
                seen_ids.add(node.id)
        return neighbors  # type: ignore[return-value]

    def domain(self, node_type: type[TNode]) -> List[TNode]:
        """Return a query domain for *node_type* preserving insertion order."""

        exact = self._nodes_by_type.get(node_type)
        if exact is not None:
            return list(exact)  # type: ignore[return-value]
        return [
            node for node in self._nodes_by_id.values() if isinstance(node, node_type)
        ]

    def edge_domain(self, edge_type: type[TEdge]) -> List[TEdge]:
        """Return a query domain for *edge_type* preserving insertion order."""

        exact = self._edges_by_type.get(edge_type)
        if exact is not None:
            return list(exact)  # type: ignore[return-value]
        return [
            edge for edge in self._edges_by_id.values() if isinstance(edge, edge_type)
        ]

    @property
    def graph(self) -> PyDiGraph[HypothesisNode, HypothesisEdge]:
        """Expose the underlying graph backend for advanced inspection."""

        return self._graph

    @property
    def node_count(self) -> int:
        return len(self._nodes_by_id)

    @property
    def edge_count(self) -> int:
        return len(self._edges_by_id)

    @property
    def instructions(self) -> List[InstructionNode]:
        return self.domain(InstructionNode)

    @property
    def actions(self) -> List[ActionNode]:
        return self.domain(ActionNode)

    @property
    def reasoner_runs(self) -> List[ReasonerRunNode]:
        return self.domain(ReasonerRunNode)

    @property
    def evidences(self) -> List[EvidenceNode]:
        return self.domain(EvidenceNode)

    @property
    def edges(self) -> List[HypothesisEdge]:
        return list(self._edges_by_id.values())

    def get_instruction(self, text: str) -> Optional[InstructionNode]:
        """Return the instruction node matching *text* after normalization."""

        return self._instruction_nodes_by_normalized_text.get(
            _normalize_instruction_text(text)
        )

    def get_action(self, action_ref: Any) -> Optional[ActionNode]:
        """Return the action node anchored to *action_ref*."""

        return self._action_nodes_by_ref_id.get(id(action_ref))

    def nodes_by_status(self, status: ClaimStatus) -> List[HypothesisNode]:
        """Return all nodes with the given claim status."""

        return list(self._nodes_by_status.get(status, []))

    def nodes_by_grounding(self, grounding: GroundingState) -> List[HypothesisNode]:
        """Return all nodes with the given grounding state."""

        return list(self._nodes_by_grounding.get(grounding, []))

    def nodes_for_run(self, run_id: str) -> List[HypothesisNode]:
        """Return all nodes produced or tagged within *run_id*."""

        return list(self._nodes_by_run_id.get(run_id, []))

    def nodes_from_reasoner(self, reasoner_name: str) -> List[HypothesisNode]:
        """Return all nodes attributed to *reasoner_name*."""

        return list(self._nodes_by_reasoner.get(reasoner_name, []))

    def edges_for_run(self, run_id: str) -> List[HypothesisEdge]:
        """Return all edges produced or tagged within *run_id*."""

        return list(self._edges_by_run_id.get(run_id, []))

    def edges_from_reasoner(self, reasoner_name: str) -> List[HypothesisEdge]:
        """Return all edges attributed to *reasoner_name*."""

        return list(self._edges_by_reasoner.get(reasoner_name, []))

    def groundings_for_symbol(
        self, symbol_ref: Any
    ) -> List[SymbolGroundingEvidenceNode]:
        """Return all grounding evidence nodes attached to *symbol_ref*."""

        return list(self._symbol_groundings_by_symbol_ref_id.get(id(symbol_ref), []))

    def edge_exists(
        self,
        src_id: str,
        dst_id: str,
        edge_type: type[TEdge],
        *,
        run_id: Optional[str] = None,
    ) -> bool:
        """Return whether a typed edge exists between *src_id* and *dst_id*."""

        if run_id is None:
            return bool(self._edges_by_relation.get((edge_type, src_id, dst_id), []))
        return bool(
            self._edges_by_relation_and_run.get((edge_type, src_id, dst_id, run_id), [])
        )

    def remove_edge(self, edge_id: str) -> Optional[HypothesisEdge]:
        """Remove the edge with *edge_id*, if present."""

        edge = self._edges_by_id.get(edge_id)
        if edge is None:
            return None
        self._graph.remove_edge_from_index(self._edge_indices_by_id[edge_id])
        self._rebuild_indexes()
        return edge

    def remove_node(self, node_id: str) -> Optional[HypothesisNode]:
        """Remove the node with *node_id* and any incident edges, if present."""

        node = self._nodes_by_id.get(node_id)
        if node is None:
            return None
        self._graph.remove_node(self._node_indices_by_id[node_id])
        self._rebuild_indexes()
        return node

    def clear(self) -> None:
        """Remove every node and edge from the graph."""

        self._graph = PyDiGraph()
        self._reset_indexes()

    def prune_run(self, run_id: str) -> Tuple[int, int]:
        """Remove all run-scoped nodes for *run_id* and their incident edges."""

        return self._remove_node_ids({node.id for node in self.nodes_for_run(run_id)})

    def prune_action(self, action_ref: Any) -> Tuple[int, int]:
        """Remove the subgraph associated with *action_ref*.

        Shared instruction nodes are retained when they still connect to nodes
        outside the action-local hypothesis closure.
        """

        subgraph = self.subgraph_for_action(action_ref)
        candidate_ids = {node.id for node in subgraph.iter_nodes()}
        removable_ids = {
            node_id
            for node_id in candidate_ids
            if not self._has_external_incident_edges(node_id, candidate_ids)
        }
        return self._remove_node_ids(removable_ids)

    def subgraph_for_run(self, run_id: str) -> "HypothesisGraph":
        """Return a graph containing the hypothesis closure for *run_id*."""

        nodes = {node.id for node in self.nodes_for_run(run_id)}
        edge_ids = {edge.id for edge in self.edges_for_run(run_id)}
        for edge in self.edges_for_run(run_id):
            nodes.add(edge.src_id)
            nodes.add(edge.dst_id)
        return self._clone_subset(nodes, edge_ids)

    def subgraph_for_reasoner(self, reasoner_name: str) -> "HypothesisGraph":
        """Return a graph containing the hypothesis closure for *reasoner_name*."""

        nodes = {node.id for node in self.nodes_from_reasoner(reasoner_name)}
        edge_ids = {edge.id for edge in self.edges_from_reasoner(reasoner_name)}
        for edge in self.edges_from_reasoner(reasoner_name):
            nodes.add(edge.src_id)
            nodes.add(edge.dst_id)
        return self._clone_subset(nodes, edge_ids)

    def subgraph_for_action(self, action_ref: Any) -> "HypothesisGraph":
        """Return the structured hypothesis closure centered on *action_ref*."""

        action_node = self.get_action(action_ref)
        if action_node is None:
            return HypothesisGraph()

        node_ids: Set[str] = {action_node.id}
        edge_ids: Set[str] = set()

        root_claims = self.get_sources(action_node.id, AboutActionEdge)
        edge_ids.update(edge.id for edge in self.in_edges(action_node.id, AboutActionEdge))
        node_ids.update(claim.id for claim in root_claims)

        for claim in root_claims:
            if claim.meta.run_id is not None:
                run_subgraph = self.subgraph_for_run(claim.meta.run_id)
                node_ids.update(node.id for node in run_subgraph.iter_nodes())
                edge_ids.update(edge.id for edge in run_subgraph.iter_edges())
                continue

            for edge in self.in_edges(claim.id):
                edge_ids.add(edge.id)
                node_ids.add(edge.src_id)
                node_ids.add(edge.dst_id)
            for edge in self.out_edges(claim.id):
                edge_ids.add(edge.id)
                node_ids.add(edge.src_id)
                node_ids.add(edge.dst_id)

        return self._clone_subset(node_ids, edge_ids)

    def to_dot(self) -> str:
        """Return a DOT representation of the current hypothesis graph."""

        return self._graph.to_dot(
            lambda node: dict(
                label=f"{type(node).__name__}\\n{node.display_id}",
                color="black",
                fillcolor=self._node_fillcolor(node),
                style="filled",
            ),
            lambda edge: dict(
                color=self._edge_color(edge),
                style="solid",
                label=edge.relation_name,
            ),
            dict(rankdir="LR"),
        )

    def _get_existing_context_node(self, node: HypothesisNode) -> Optional[HypothesisNode]:
        """Return an existing stable context node for deduplicated node types."""

        if isinstance(node, InstructionNode):
            return self._instruction_nodes_by_normalized_text.get(node.normalized_text)
        if isinstance(node, ActionNode):
            return self._action_nodes_by_ref_id.get(id(node.action_ref))
        return None

    def _index_node(self, node: HypothesisNode) -> None:
        self._nodes_by_type.setdefault(type(node), []).append(node)
        self._nodes_by_status.setdefault(node.meta.status, []).append(node)
        self._nodes_by_grounding.setdefault(node.meta.grounding, []).append(node)
        self._nodes_by_reasoner.setdefault(node.meta.source_reasoner, []).append(node)

        if node.meta.run_id is not None:
            self._nodes_by_run_id.setdefault(node.meta.run_id, []).append(node)

        if isinstance(node, InstructionNode):
            self._instruction_nodes_by_normalized_text[node.normalized_text] = node
        elif isinstance(node, ActionNode):
            self._action_nodes_by_ref_id[id(node.action_ref)] = node
        elif isinstance(node, SymbolGroundingEvidenceNode):
            self._symbol_groundings_by_symbol_ref_id.setdefault(
                id(node.symbol_ref), []
            ).append(node)

    def _index_edge(self, edge: HypothesisEdge) -> None:
        self._edges_by_type.setdefault(type(edge), []).append(edge)
        self._edges_by_reasoner.setdefault(edge.meta.source_reasoner, []).append(edge)
        self._edges_by_relation.setdefault(
            (type(edge), edge.src_id, edge.dst_id), []
        ).append(edge)
        self._edges_by_relation_and_run.setdefault(
            (type(edge), edge.src_id, edge.dst_id, edge.meta.run_id), []
        ).append(edge)
        if edge.meta.run_id is not None:
            self._edges_by_run_id.setdefault(edge.meta.run_id, []).append(edge)

    def _reset_indexes(self) -> None:
        self._node_indices_by_id.clear()
        self._edge_indices_by_id.clear()
        self._nodes_by_id.clear()
        self._edges_by_id.clear()
        self._nodes_by_type.clear()
        self._edges_by_type.clear()
        self._instruction_nodes_by_normalized_text.clear()
        self._action_nodes_by_ref_id.clear()
        self._nodes_by_status.clear()
        self._nodes_by_grounding.clear()
        self._nodes_by_run_id.clear()
        self._nodes_by_reasoner.clear()
        self._edges_by_run_id.clear()
        self._edges_by_reasoner.clear()
        self._edges_by_relation.clear()
        self._edges_by_relation_and_run.clear()
        self._symbol_groundings_by_symbol_ref_id.clear()

    def _rebuild_indexes(self) -> None:
        self._reset_indexes()
        for node_index in self._graph.node_indexes():
            node = self._graph.get_node_data(node_index)
            self._node_indices_by_id[node.id] = node_index
            self._nodes_by_id[node.id] = node
            self._index_node(node)
        for edge_index in self._graph.edge_indices():
            edge = self._graph.get_edge_data_by_index(edge_index)
            self._edge_indices_by_id[edge.id] = edge_index
            self._edges_by_id[edge.id] = edge
            self._index_edge(edge)

    def _remove_node_ids(self, node_ids: Set[str]) -> Tuple[int, int]:
        existing_node_ids = [
            node_id for node_id in node_ids if node_id in self._node_indices_by_id
        ]
        if not existing_node_ids:
            return (0, 0)

        removed_edge_ids: Set[str] = set()
        for node_id in existing_node_ids:
            removed_edge_ids.update(edge.id for edge in self.in_edges(node_id))
            removed_edge_ids.update(edge.id for edge in self.out_edges(node_id))

        indexed_nodes = sorted(
            (
                (self._node_indices_by_id[node_id], node_id)
                for node_id in existing_node_ids
            ),
            reverse=True,
        )
        for node_index, _ in indexed_nodes:
            self._graph.remove_node(node_index)

        self._rebuild_indexes()
        return (len(existing_node_ids), len(removed_edge_ids))

    def _clone_subset(
        self,
        node_ids: Set[str],
        edge_ids: Set[str],
    ) -> "HypothesisGraph":
        subgraph = HypothesisGraph()
        for node in self.iter_nodes():
            if node.id in node_ids:
                subgraph.add_node(node)
        for edge in self.iter_edges():
            if (
                edge.id in edge_ids
                and edge.src_id in node_ids
                and edge.dst_id in node_ids
            ):
                subgraph.add_edge(edge)
        return subgraph

    def _has_external_incident_edges(
        self,
        node_id: str,
        candidate_node_ids: Set[str],
    ) -> bool:
        for edge in self.in_edges(node_id):
            if edge.src_id not in candidate_node_ids:
                return True
        for edge in self.out_edges(node_id):
            if edge.dst_id not in candidate_node_ids:
                return True
        return False

    @staticmethod
    def _node_fillcolor(node: HypothesisNode) -> str:
        if node.meta.grounding == GroundingState.SYMBOL_GROUNDED:
            return "lightgreen"
        if node.meta.grounding == GroundingState.SLOT_ALIGNED:
            return "khaki"
        if node.meta.status == ClaimStatus.SUPPORTED:
            return "lightblue"
        if node.meta.status == ClaimStatus.REFUTED:
            return "lightcoral"
        return "white"

    @staticmethod
    def _edge_color(edge: HypothesisEdge) -> str:
        if edge.meta.grounding == GroundingState.SYMBOL_GROUNDED:
            return "green"
        if edge.meta.grounding == GroundingState.SLOT_ALIGNED:
            return "goldenrod"
        return "black"
