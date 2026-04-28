"""Matplotlib renderer for :mod:`llmr.visualization.trees`.

Draws one :class:`~llmr.visualization.trees.MatchTree` or
:class:`~llmr.visualization.trees.SymbolTree` on a supplied
:class:`matplotlib.axes.Axes`.  Node colours and label formatting discriminate
between action roots, branches, free leaves, filled leaves, symbol class
buckets, and symbol bodies.

Three-panel composition and cross-axis bindings are added by
``render_panels`` (next layer) — this module only owns the per-tree drawing.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing_extensions import Any, Dict, List, Optional, Set, Tuple

from llmr.visualization.trees import MatchTree, SymbolTree, TreeNode

if TYPE_CHECKING:
    from matplotlib.axes import Axes


# ── Colour scheme ─────────────────────────────────────────────────────────────

_COLORS: Dict[str, str] = {
    "action": "#cfe2ff",
    "branch": "#e9ecef",
    "leaf-free": "#ffffff",
    "leaf-filled": "#d1e7dd",
    "symbol-root": "#e2d4f0",
    "symbol-class-bucket": "#e9ecef",
    "symbol-body": "#ffe5b4",
}

_HIGHLIGHT_FACE = "#fff3bf"
_HIGHLIGHT_EDGE = "#e67700"
_DEFAULT_EDGE = "#495057"


# ── Layout ────────────────────────────────────────────────────────────────────


def _layout(
    tree: "MatchTree | SymbolTree",
) -> Dict[str, Tuple[float, float]]:
    """Compute (x, y) positions for every node in *tree*.

    Standard tidy layout: leaves get sequential integer x-values; internal
    nodes sit at the horizontal midpoint of their children; y is negative
    depth so the root is on top.
    """
    positions: Dict[str, Tuple[float, float]] = {}
    counter = [0.0]

    def _walk(node_id: str, depth: int) -> float:
        node = tree.nodes[node_id]
        if not node.children_ids:
            x = counter[0]
            counter[0] += 1.0
            positions[node_id] = (x, -float(depth))
            return x

        xs = [_walk(child, depth + 1) for child in node.children_ids]
        x = (xs[0] + xs[-1]) / 2.0
        positions[node_id] = (x, -float(depth))
        return x

    _walk(tree.root_id, 0)
    return positions


# ── Drawing ───────────────────────────────────────────────────────────────────


def _draw_tree(
    ax: "Axes",
    tree: "MatchTree | SymbolTree",
    positions: Optional[Dict[str, Tuple[float, float]]] = None,
    title: Optional[str] = None,
    highlight: Optional[Set[str]] = None,
) -> Dict[str, Tuple[float, float]]:
    """Render *tree* on *ax* and return the layout used.

    Returning the layout lets callers compose multi-axis decorations (e.g.
    :class:`~matplotlib.patches.ConnectionPatch` binding arrows) without
    recomputing coordinates.

    :param highlight: Node ids to outline with the newly-filled colour.
    """
    positions = positions if positions is not None else _layout(tree)
    highlight = highlight or set()

    _draw_edges(ax, tree, positions)
    for node_id, node in tree.nodes.items():
        x, y = positions[node_id]
        _draw_node(ax, node, x, y, is_highlighted=node_id in highlight)

    _finalize_axes(ax, positions, title=title)
    return positions


def _draw_edges(
    ax: "Axes",
    tree: "MatchTree | SymbolTree",
    positions: Dict[str, Tuple[float, float]],
) -> None:
    for node_id, node in tree.nodes.items():
        px, py = positions[node_id]
        for child_id in node.children_ids:
            cx, cy = positions[child_id]
            ax.plot([px, cx], [py, cy], color=_DEFAULT_EDGE, linewidth=0.8, zorder=1)


def _draw_node(
    ax: "Axes",
    node: TreeNode,
    x: float,
    y: float,
    is_highlighted: bool,
) -> None:
    face = _HIGHLIGHT_FACE if is_highlighted else _COLORS.get(node.kind, "#ffffff")
    edge = _HIGHLIGHT_EDGE if is_highlighted else _DEFAULT_EDGE
    linestyle = "--" if node.kind == "leaf-free" else "-"
    text = _node_label(node)

    ax.text(
        x,
        y,
        text,
        ha="center",
        va="center",
        fontsize=8,
        zorder=3,
        bbox={
            "boxstyle": "round,pad=0.35",
            "facecolor": face,
            "edgecolor": edge,
            "linestyle": linestyle,
            "linewidth": 1.2 if is_highlighted else 0.8,
        },
    )


def _node_label(node: TreeNode) -> str:
    if node.kind.startswith("leaf"):
        value_display = node.metadata.get("value_display") or ""
        if value_display:
            return f"{node.label} = {value_display}"
        return node.label
    if node.kind == "symbol-class-bucket":
        count = node.metadata.get("count")
        if isinstance(count, int):
            return f"{node.label} ({count})"
    return node.label


def _finalize_axes(
    ax: "Axes",
    positions: Dict[str, Tuple[float, float]],
    title: Optional[str],
) -> None:
    xs: List[float] = [x for x, _ in positions.values()]
    ys: List[float] = [y for _, y in positions.values()]
    if xs:
        ax.set_xlim(min(xs) - 0.6, max(xs) + 0.6)
    if ys:
        ax.set_ylim(min(ys) - 0.6, max(ys) + 0.6)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    if title:
        ax.set_title(title, fontsize=10, pad=8)


# ── 3-panel composition ───────────────────────────────────────────────────────


def render_panels(
    snapshot: "Any",
) -> "Any":
    """Render a :class:`~llmr.visualization.trees.MatchResolutionSnapshot` as a
    3-panel matplotlib figure.

    Panel layout (left → right):

    1. **Before** — initial (underspecified) MatchTree; free leaves have dashed borders.
    2. **World** — SymbolTree grouped by class; newly-bound bodies are highlighted.
    3. **After** — resolved MatchTree; newly-filled leaves are highlighted gold.

    Binding arrows connect each highlighted symbol body in the World panel to the
    matching resolved leaf in the After panel using
    :class:`matplotlib.patches.ConnectionPatch`.

    :param snapshot: A fully-populated
        :class:`~llmr.visualization.trees.MatchResolutionSnapshot` (i.e.
        ``record_after`` has been called).
    :returns: The :class:`matplotlib.figure.Figure` — display it with
        ``plt.show()`` or ``fig.savefig(...)``.
    """
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(16, 7))
    ax_before, ax_symbol, ax_after = axes

    # Panel 1 — Before
    _draw_tree(ax_before, snapshot.before, title="Before (underspecified)")

    # Panel 2 — World / SymbolTree
    sym_positions: Dict[str, Tuple[float, float]] = {}
    if snapshot.symbol_tree is not None:
        # Highlight symbol bodies that appear in bindings.
        bound_names = set(snapshot.bindings.values())
        sym_highlight = {
            nid
            for nid, node in snapshot.symbol_tree.nodes.items()
            if node.kind == "symbol-body"
            and node.metadata.get("display_name") in bound_names
        }
        sym_positions = _draw_tree(
            ax_symbol,
            snapshot.symbol_tree,
            title="World (SymbolGraph)",
            highlight=sym_highlight,
        )

    # Panel 3 — After
    after_positions: Dict[str, Tuple[float, float]] = {}
    if snapshot.after is not None:
        after_positions = _draw_tree(
            ax_after,
            snapshot.after,
            title="After (resolved)",
            highlight=snapshot.newly_filled,
        )

    # Binding arrows (World → After)
    if snapshot.after is not None and snapshot.symbol_tree is not None:
        _draw_bindings(fig, ax_symbol, ax_after, snapshot, sym_positions, after_positions)

    fig.tight_layout()
    return fig


def _draw_bindings(
    fig: "Any",
    ax_symbol: "Axes",
    ax_after: "Axes",
    snapshot: "Any",
    sym_positions: Dict[str, Tuple[float, float]],
    after_positions: Dict[str, Tuple[float, float]],
) -> None:
    """Draw :class:`~matplotlib.patches.ConnectionPatch` arrows from bound
    symbol bodies in the World panel to the corresponding resolved leaves in the
    After panel.
    """
    from matplotlib.patches import ConnectionPatch

    # Build reverse map: display_name → symbol body node_id
    name_to_sym_id: Dict[str, str] = {}
    if snapshot.symbol_tree is not None:
        for nid, node in snapshot.symbol_tree.nodes.items():
            if node.kind == "symbol-body":
                name_to_sym_id[node.metadata.get("display_name", "")] = nid

    for leaf_id, bound_name in snapshot.bindings.items():
        sym_id = name_to_sym_id.get(bound_name)
        if sym_id is None or sym_id not in sym_positions:
            continue
        if leaf_id not in after_positions:
            continue

        sx, sy = sym_positions[sym_id]
        ax, ay = after_positions[leaf_id]

        arrow = ConnectionPatch(
            xyA=(sx, sy),
            xyB=(ax, ay),
            coordsA="data",
            coordsB="data",
            axesA=ax_symbol,
            axesB=ax_after,
            arrowstyle="->",
            color=_HIGHLIGHT_EDGE,
            linewidth=1.2,
            linestyle="dashed",
            zorder=5,
        )
        fig.add_artist(arrow)


# ── Convenience entry point ───────────────────────────────────────────────────


def render_match_resolution(
    expression: "Any",
    backend: "Any",
    symbol_type: "Any" = None,
    symbol_graph: "Any" = None,
) -> "Any":
    """Capture a before/after match-resolution snapshot and render the 3-panel figure.

    Typical usage inside a notebook after calling ``backend.evaluate(match)``::

        from llmr.visualization.render import render_match_resolution
        fig = render_match_resolution(match, backend, symbol_type=WorldBody)
        fig.savefig("resolution.png")

    :param expression: The KRROOD ``Match`` expression to resolve (evaluated in place).
    :param backend: The :class:`~llmr.backend.LLMBackend` that performed
        (or will perform) evaluation. ``backend.semantics`` is read for bindings.
    :param symbol_type: Optional Symbol subclass passed to
        :func:`~llmr.visualization.trees.build_symbol_tree`.
    :param symbol_graph: Optional SymbolGraph override; defaults to the singleton.
    :returns: The rendered :class:`matplotlib.figure.Figure`.
    """
    from llmr.visualization.trees import MatchResolutionSnapshot

    snapshot = MatchResolutionSnapshot.from_match(
        expression,
        symbol_type=symbol_type,
        symbol_graph=symbol_graph,
    )
    snapshot.record_after(expression, backend=backend)
    return render_panels(snapshot)


__all__ = ["_layout", "_draw_tree", "render_panels", "render_match_resolution"]
