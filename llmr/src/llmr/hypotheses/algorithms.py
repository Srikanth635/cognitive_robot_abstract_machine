"""Graph-algorithm-powered queries over the hypothesis graph.

These four functions are the load-bearing reason to keep rustworkx: they
cannot be replicated efficiently with a flat store or parallel dicts.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Set, Tuple

import rustworkx

from llmr.hypotheses.elements import ClaimStatus
from llmr.hypotheses.common.nodes import ClaimNode
from llmr.hypotheses.graph import HypothesisGraph
from llmr.hypotheses.projectors.framenet.nodes import FrameRoleHypothesisNode


def invalidate_from_symbol(
    graph: HypothesisGraph,
    symbol_ref: Any,
    *,
    new_status: ClaimStatus = ClaimStatus.REFUTED,
) -> List[ClaimNode]:
    """Mark all claims that depend on *symbol_ref* with *new_status*.

    Walks every ancestor of each SymbolGroundingEvidenceNode bound to
    *symbol_ref* via rustworkx.ancestors(). Any ancestor that is a ClaimNode
    has its meta.status updated in place and is appended to the return list.
    """
    affected: List[ClaimNode] = []
    seen_ids: Set[str] = set()

    for evidence in graph.groundings_for_symbol(symbol_ref):
        evidence_index = graph._node_indices_by_id[evidence.id]
        for ancestor_index in rustworkx.ancestors(graph._graph, evidence_index):
            node = graph._graph.get_node_data(ancestor_index)
            if isinstance(node, ClaimNode) and node.id not in seen_ids:
                node.meta.status = new_status
                affected.append(node)
                seen_ids.add(node.id)

    return affected


def reasoning_chain(graph: HypothesisGraph, node_id: str) -> List[Any]:
    """Return the shortest path from any instruction to *node_id*.

    Uses Dijkstra on the undirected view of the graph so that edges can be
    traversed in either direction. Returns an empty list when *node_id* is
    unknown or no instruction is reachable.
    """
    if node_id not in graph._node_indices_by_id:
        return []

    target_idx = graph._node_indices_by_id[node_id]
    best_path: Optional[List[int]] = None

    for instruction in graph.instructions:
        src_idx = graph._node_indices_by_id[instruction.id]
        paths = rustworkx.dijkstra_shortest_paths(
            graph._graph, src_idx, target=target_idx, as_undirected=True
        )
        candidate = list(paths[target_idx]) if target_idx in paths else None
        if candidate is not None and (
            best_path is None or len(candidate) < len(best_path)
        ):
            best_path = candidate

    if best_path is None:
        return []
    return [graph._graph.get_node_data(i) for i in best_path]


def conflicting_role_claims(
    graph: HypothesisGraph,
    action_ref: Any,
) -> List[Tuple[FrameRoleHypothesisNode, FrameRoleHypothesisNode]]:
    """Find FrameRoleHypothesisNode pairs with the same role name but different fillers.

    Only examines the hypothesis closure of *action_ref*, so roles from
    unrelated actions are never compared.
    """
    subgraph = graph.subgraph_for_action(action_ref)
    roles = subgraph.domain(FrameRoleHypothesisNode)

    by_role: Dict[str, List[FrameRoleHypothesisNode]] = {}
    for role in roles:
        by_role.setdefault(role.role_name, []).append(role)

    conflicts: List[Tuple[FrameRoleHypothesisNode, FrameRoleHypothesisNode]] = []
    for role_nodes in by_role.values():
        for i, r1 in enumerate(role_nodes):
            for r2 in role_nodes[i + 1 :]:
                if r1.filler_text != r2.filler_text:
                    conflicts.append((r1, r2))

    return conflicts


def hypothesis_closure(graph: HypothesisGraph, node_id: str) -> HypothesisGraph:
    """Return a new HypothesisGraph with all nodes reachable from *node_id*.

    Combines rustworkx.ancestors() and rustworkx.descendants() to build the
    full bidirectional reachability closure. The resulting subgraph retains
    only edges whose both endpoints are inside the closure.
    """
    if node_id not in graph._node_indices_by_id:
        return HypothesisGraph()

    index = graph._node_indices_by_id[node_id]
    all_indices = (
        rustworkx.ancestors(graph._graph, index)
        | rustworkx.descendants(graph._graph, index)
        | {index}
    )
    return graph._clone_subset_by_indices(all_indices)
