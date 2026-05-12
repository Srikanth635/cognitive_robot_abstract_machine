"""Benchmark for HypothesisGraph core operations.

Run with:
    python llmr/bench/hypothesis_graph_bench.py

Measures:
    - Bulk insert: N nodes, 3N edges
    - Domain query: domain(RoleClaimNode) on N-node graph
    - Abstract domain query: domain(ClaimNode) on N-node graph
    - Algorithm queries: hypothesis_closure, invalidate_from_symbol
    - Bulk removal: remove all claim nodes
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Generator

from llmr.hypotheses import (
    AboutActionEdge,
    ActionNode,
    ClaimStatus,
    EvokesFrameEdge,
    FrameClaimNode,
    RoleClaimNode,
    GroundedByEdge,
    GroundingState,
    HasRoleEdge,
    HypothesisGraph,
    HypothesisMeta,
    InstructionNode,
    ProducesClaimEdge,
    ReasonerRunNode,
    GroundingEvidenceNode,
    hypothesis_closure,
    invalidate_from_symbol,
)

# ---------------------------------------------------------------------------
# Benchmark helpers
# ---------------------------------------------------------------------------

@contextmanager
def _timer(label: str) -> Generator[None, None, None]:
    start = time.perf_counter()
    yield
    elapsed = time.perf_counter() - start
    print(f"  {label:<55} {elapsed * 1000:>8.2f} ms")


def _meta(reasoner: str = "framenet_reasoner", run_id: str | None = None) -> HypothesisMeta:
    return HypothesisMeta(
        source_reasoner=reasoner,
        status=ClaimStatus.HYPOTHESIS,
        grounding=GroundingState.TEXT_ONLY,
        run_id=run_id,
    )


# ---------------------------------------------------------------------------
# Graph builders
# ---------------------------------------------------------------------------

def _build_graph(n_actions: int) -> tuple[HypothesisGraph, list[object]]:
    """Build a graph with n_actions instruction/action/run/frame/2-role clusters.

    Per-action node count: 1 instruction (deduped) + 1 action + 1 run + 1 frame + 2 roles + 1 grounding evidence = 7
    Per-action edge count: evokes + about + 3 produced + 2 roles + 1 grounding = 8
    """
    graph = HypothesisGraph()
    symbol_refs: list[object] = []

    shared_instruction = InstructionNode(
        id="i_shared",
        meta=_meta(),
        text="benchmark instruction",
        normalized_text="benchmark instruction",
    )
    graph.add_node(shared_instruction)

    for i in range(n_actions):
        action_ref = object()
        symbol_ref = object()
        symbol_refs.append(symbol_ref)
        run_id = f"run{i}"

        graph.add_node(ActionNode(id=f"a{i}", meta=_meta(), action_ref=action_ref, action_type="PickUp"))
        graph.add_node(ReasonerRunNode(
            id=f"run{i}", meta=_meta(run_id=run_id), reasoner_name="framenet_reasoner",
            run_id=run_id, model_name=None, prompt_version=None,
            action_type="PickUp", instruction_text="benchmark instruction",
        ))
        graph.add_node(FrameClaimNode(
            id=f"f{i}", meta=_meta(run_id=run_id), frame="Getting",
            lexical_unit="pick_up.v", framenet_label="pick",
            action_type="PickUp", instruction_text="benchmark instruction",
        ))
        graph.add_node(RoleClaimNode(
            id=f"r{i}_theme", meta=_meta(run_id=run_id), role_family="core",
            role_name="theme", filler_text=f"obj{i}", filler_kind="entity",
        ))
        graph.add_node(RoleClaimNode(
            id=f"r{i}_goal", meta=_meta(run_id=run_id), role_family="core",
            role_name="goal", filler_text="robot hand", filler_kind="abstract",
        ))
        graph.add_node(GroundingEvidenceNode(
            id=f"ev{i}", meta=_meta(run_id=run_id), query_text=f"obj{i}",
            symbol_ref=symbol_ref, symbol_type="Body", grounding_method="name_match",
        ))
        for edge in [
            EvokesFrameEdge(id=f"e{i}_evokes", meta=_meta(run_id=run_id), src_id="i_shared", dst_id=f"f{i}"),
            AboutActionEdge(id=f"e{i}_about", meta=_meta(run_id=run_id), src_id=f"f{i}", dst_id=f"a{i}"),
            ProducesClaimEdge(id=f"e{i}_pc_f", meta=_meta(run_id=run_id), src_id=f"run{i}", dst_id=f"f{i}"),
            ProducesClaimEdge(id=f"e{i}_pc_rt", meta=_meta(run_id=run_id), src_id=f"run{i}", dst_id=f"r{i}_theme"),
            ProducesClaimEdge(id=f"e{i}_pc_rg", meta=_meta(run_id=run_id), src_id=f"run{i}", dst_id=f"r{i}_goal"),
            HasRoleEdge(id=f"e{i}_role_t", meta=_meta(run_id=run_id), src_id=f"f{i}", dst_id=f"r{i}_theme"),
            HasRoleEdge(id=f"e{i}_role_g", meta=_meta(run_id=run_id), src_id=f"f{i}", dst_id=f"r{i}_goal"),
            GroundedByEdge(id=f"e{i}_grounded", meta=_meta(run_id=run_id), src_id=f"r{i}_theme", dst_id=f"ev{i}"),
        ]:
            graph.add_edge(edge)

    return graph, symbol_refs


# ---------------------------------------------------------------------------
# Benchmark suite
# ---------------------------------------------------------------------------

def run_benchmarks(n_actions: int = 1000) -> None:
    n_nodes = 1 + n_actions * 6  # 1 shared instruction + 6 per action
    n_edges = n_actions * 8

    print(f"\n{'='*70}")
    print(f"HypothesisGraph benchmark  —  {n_actions} action clusters")
    print(f"  Expected: ~{n_nodes} nodes, ~{n_edges} edges")
    print(f"{'='*70}")

    # --- Insert ---
    print("\nInsert")
    with _timer(f"build graph ({n_actions} actions, ~{n_nodes} nodes, ~{n_edges} edges)"):
        graph, symbol_refs = _build_graph(n_actions)

    assert graph.node_count == n_nodes, f"expected {n_nodes} nodes, got {graph.node_count}"
    assert graph.edge_count == n_edges, f"expected {n_edges} edges, got {graph.edge_count}"

    # --- Domain queries ---
    print("\nDomain queries")
    REPS = 10

    t0 = time.perf_counter()
    for _ in range(REPS):
        _ = graph.domain(RoleClaimNode)
    elapsed = (time.perf_counter() - t0) / REPS
    print(f"  {'domain(RoleClaimNode) x10 avg':<55} {elapsed * 1000:>8.3f} ms")

    t0 = time.perf_counter()
    for _ in range(REPS):
        _ = graph.domain(FrameClaimNode)
    elapsed = (time.perf_counter() - t0) / REPS
    print(f"  {'domain(FrameClaimNode) x10 avg':<55} {elapsed * 1000:>8.3f} ms")

    from llmr.hypotheses.common.nodes import ClaimNode
    t0 = time.perf_counter()
    for _ in range(REPS):
        _ = graph.domain(ClaimNode)
    elapsed = (time.perf_counter() - t0) / REPS
    print(f"  {'domain(ClaimNode) [abstract, all subtypes] x10 avg':<55} {elapsed * 1000:>8.3f} ms")

    # --- Algorithm queries ---
    print("\nAlgorithm queries")

    mid = n_actions // 2
    with _timer(f"hypothesis_closure(f{mid})"):
        closure = hypothesis_closure(graph, f"f{mid}")
    print(f"    → closure has {closure.node_count} nodes, {closure.edge_count} edges")

    with _timer(f"invalidate_from_symbol(symbol_refs[{mid}])"):
        affected = invalidate_from_symbol(graph, symbol_refs[mid])
    print(f"    → {len(affected)} claims invalidated")

    with _timer("nodes_for_run('run0')"):
        _ = graph.nodes_for_run("run0")

    with _timer("subgraph_for_run('run0')"):
        _ = graph.subgraph_for_run("run0")

    # --- Removal ---
    print("\nRemoval")
    frame_ids = [f"f{i}" for i in range(n_actions)]
    with _timer(f"remove_node x{n_actions} (all frame nodes)"):
        for fid in frame_ids:
            graph.remove_node(fid)

    remaining_frames = graph.domain(FrameClaimNode)
    assert remaining_frames == [], f"expected 0 frames after bulk remove, got {len(remaining_frames)}"
    print(f"    → {n_actions} nodes removed, domain(FrameClaimNode) = 0 ✓")

    print(f"\n{'='*70}\n")


if __name__ == "__main__":
    import sys

    n = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
    run_benchmarks(n_actions=n)
