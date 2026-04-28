"""Side-by-side visualization of the llmr match-resolution pipeline.

Three plain-data trees power the visualizer:

  * :class:`MatchTree` — a KRROOD Match expression snapshot (before/after slot fill).
  * :class:`SymbolTree` — the SymbolGraph grouped by class.
  * :class:`MatchResolutionSnapshot` — bundles before/after/symbol trees, plus
    the set of slots newly filled during evaluation and the symbol→leaf
    bindings produced by the slot-filler.

Rendering lives in :mod:`llmr.visualization.render`; this module is matplotlib-free.
"""

from __future__ import annotations

from llmr.visualization.trees import (
    MatchResolutionSnapshot,
    MatchTree,
    SymbolTree,
    TreeNode,
    bindings_from_semantics,
    build_match_tree,
    build_symbol_tree,
    diff_match_trees,
)
from llmr.visualization.render import render_match_resolution, render_panels

__all__ = [
    "MatchResolutionSnapshot",
    "MatchTree",
    "SymbolTree",
    "TreeNode",
    "bindings_from_semantics",
    "build_match_tree",
    "build_symbol_tree",
    "diff_match_trees",
    "render_match_resolution",
    "render_panels",
]
