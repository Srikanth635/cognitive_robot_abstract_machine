"""Plain-data tree models for the llmr match-resolution visualizer.

Two trees describe the state of an llmr pipeline at a point in time:

  * :class:`MatchTree` — snapshot of a KRROOD Match expression.  The root is
    the action class; branches are nested-Match fields; leaves are the
    resolvable variables, each tagged with ``is_free``, ``value``, and
    :class:`~llmr.bridge.introspect.FieldKind`.
  * :class:`SymbolTree` — SymbolGraph bodies grouped by class under a synthetic
    ``World`` root.

A :class:`MatchResolutionSnapshot` bundles a *before* and *after* MatchTree
alongside the SymbolTree and the symbol→leaf bindings pulled from the
backend's :class:`~llmr.schemas.ActionAnnotationBundle`.

This module is matplotlib-free; see :mod:`llmr.visualization.render` for drawing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING
from typing_extensions import Any, Dict, List, Optional, Set, Type

from llmr.bridge.match_reader import MatchSnapshot, MatchField, snapshot_match
from llmr.bridge.world_reader import symbol_display_name

if TYPE_CHECKING:
    from krrood.entity_query_language.query.match import Match
    from krrood.symbol_graph.symbol_graph import Symbol, SymbolGraph

    from llmr.backend import LLMBackend
    from llmr.schemas import ActionAnnotationBundle as ActionSemantics


# ── Tree primitives ───────────────────────────────────────────────────────────


@dataclass
class TreeNode:
    """One node in a :class:`MatchTree` or :class:`SymbolTree`.

    ``node_id`` is stable across snapshots so diffing works by id.  ``metadata``
    carries renderer-specific fields (``field_kind``, ``is_free``, ``value``
    for match leaves; ``class_name``, ``instance_id`` for symbol nodes).
    """

    node_id: str
    label: str
    kind: str
    parent_id: Optional[str] = None
    children_ids: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MatchTree:
    """Tree view of a :class:`~llmr.bridge.match_reader.MatchSnapshot` snapshot."""

    root_id: str
    nodes: Dict[str, TreeNode]
    action_name: str

    @property
    def root(self) -> TreeNode:
        return self.nodes[self.root_id]

    def leaf_ids(self) -> List[str]:
        """Node ids whose ``kind`` starts with ``"leaf"``."""
        return [nid for nid, node in self.nodes.items() if node.kind.startswith("leaf")]


@dataclass
class SymbolTree:
    """Tree view of a SymbolGraph grouped by class name under a ``World`` root."""

    root_id: str
    nodes: Dict[str, TreeNode]

    @property
    def root(self) -> TreeNode:
        return self.nodes[self.root_id]

    def body_ids(self) -> List[str]:
        return [nid for nid, node in self.nodes.items() if node.kind == "symbol-body"]


# ── Match tree builder ────────────────────────────────────────────────────────


_ROOT_ID = ""


def build_match_tree(match_data: MatchSnapshot) -> MatchTree:
    """Build a :class:`MatchTree` from a :class:`MatchSnapshot` snapshot.

    Intermediate branch nodes (e.g. ``grasp_description``) are reconstructed
    from the dotted ``prompt_name`` of each leaf — KRROOD's
    :meth:`Match.matches_with_variables` yields only leaves.

    :param match_data: Snapshot produced by
        :func:`~llmr.bridge.match_reader.snapshot_match`.
    """
    root = TreeNode(
        node_id=_ROOT_ID,
        label=match_data.action_name,
        kind="action",
        metadata={"action_name": match_data.action_name},
    )
    nodes: Dict[str, TreeNode] = {_ROOT_ID: root}

    for slot in match_data.slots:
        _add_slot(nodes, slot)

    return MatchTree(
        root_id=_ROOT_ID,
        nodes=nodes,
        action_name=match_data.action_name,
    )


def _add_slot(nodes: Dict[str, TreeNode], slot: MatchField) -> None:
    """Insert one leaf + its ancestor branches into *nodes*."""
    parts = slot.prompt_name.split(".") if slot.prompt_name else [slot.attribute_name]

    parent_id = _ROOT_ID
    for depth, part in enumerate(parts[:-1]):
        branch_id = ".".join(parts[: depth + 1])
        if branch_id not in nodes:
            branch = TreeNode(
                node_id=branch_id,
                label=part,
                kind="branch",
                parent_id=parent_id,
                metadata={"path": branch_id},
            )
            nodes[branch_id] = branch
            nodes[parent_id].children_ids.append(branch_id)
        parent_id = branch_id

    leaf_id = slot.prompt_name or slot.attribute_name
    if leaf_id in nodes:
        # Defensive: a duplicate leaf from a malformed snapshot.  Last write wins.
        return
    leaf = TreeNode(
        node_id=leaf_id,
        label=parts[-1],
        kind="leaf-free" if slot.is_free else "leaf-filled",
        parent_id=parent_id,
        metadata={
            "attribute_name": slot.attribute_name,
            "prompt_name": slot.prompt_name,
            "field_kind": slot.field_kind.name,
            "field_type": _format_type(slot.field_type),
            "is_free": slot.is_free,
            "value": None if slot.is_free else slot.value,
            "value_display": "" if slot.is_free else _format_value(slot.value),
        },
    )
    nodes[leaf_id] = leaf
    nodes[parent_id].children_ids.append(leaf_id)


def _format_type(field_type: Any) -> str:
    return getattr(field_type, "__name__", str(field_type))


def _format_value(value: Any) -> str:
    """Render a slot value for display; keep it short and readable."""
    name = symbol_display_name(value) if value is not None else ""
    if name:
        return name
    if value is None:
        return ""
    enum_name = getattr(value, "name", None)
    if isinstance(enum_name, str):
        return enum_name
    text = repr(value)
    return text if len(text) <= 40 else text[:37] + "..."


# ── Symbol tree builder ───────────────────────────────────────────────────────


def build_symbol_tree(
    symbol_type: Optional[Type["Symbol"]] = None,
    symbol_graph: Optional["SymbolGraph"] = None,
) -> SymbolTree:
    """Walk a SymbolGraph and group its instances by class name under ``World``.

    The layout is deliberately shallow — one level of class-buckets under the
    root — so the panel stays legible even with a couple of dozen bodies.

    :param symbol_type: Symbol subclass to scope the walk.  Defaults to
        :class:`~krrood.symbol_graph.symbol_graph.Symbol` (all instances).
    :param symbol_graph: SymbolGraph to query; defaults to the singleton.
    """
    from krrood.symbol_graph.symbol_graph import Symbol, SymbolGraph

    scope = symbol_type or Symbol
    try:
        graph = symbol_graph or SymbolGraph()
    except Exception:
        graph = None

    root_id = "world"
    root = TreeNode(
        node_id=root_id,
        label="World",
        kind="symbol-root",
        metadata={"scope": scope.__name__},
    )
    nodes: Dict[str, TreeNode] = {root_id: root}

    if graph is None:
        return SymbolTree(root_id=root_id, nodes=nodes)

    buckets: Dict[str, List[Any]] = {}
    seen_bodies: Set[int] = set()
    try:
        instances = list(graph.get_instances_of_type(scope))
    except Exception:
        instances = []

    for body in instances:
        body_id = id(body)
        if body_id in seen_bodies:
            continue
        seen_bodies.add(body_id)
        buckets.setdefault(type(body).__name__, []).append(body)

    for class_name in sorted(buckets):
        bucket_id = f"type:{class_name}"
        bucket = TreeNode(
            node_id=bucket_id,
            label=class_name,
            kind="symbol-class-bucket",
            parent_id=root_id,
            metadata={"class_name": class_name, "count": len(buckets[class_name])},
        )
        nodes[bucket_id] = bucket
        root.children_ids.append(bucket_id)

        for body in sorted(
            buckets[class_name],
            key=lambda b: (symbol_display_name(b) or "").lower(),
        ):
            name = symbol_display_name(body) or f"{class_name}@{id(body):x}"
            body_node_id = f"body:{id(body)}"
            body_node = TreeNode(
                node_id=body_node_id,
                label=name,
                kind="symbol-body",
                parent_id=bucket_id,
                metadata={
                    "class_name": class_name,
                    "instance_id": id(body),
                    "display_name": name,
                },
            )
            nodes[body_node_id] = body_node
            bucket.children_ids.append(body_node_id)

    return SymbolTree(root_id=root_id, nodes=nodes)


# ── Diff and bindings ─────────────────────────────────────────────────────────


def diff_match_trees(before: MatchTree, after: MatchTree) -> Set[str]:
    """Leaf node-ids that transitioned from free (``before``) to filled (``after``)."""
    filled: Set[str] = set()
    for leaf_id in after.leaf_ids():
        after_node = after.nodes[leaf_id]
        if after_node.kind != "leaf-filled":
            continue
        before_node = before.nodes.get(leaf_id)
        if before_node is None or before_node.kind == "leaf-free":
            filled.add(leaf_id)
    return filled


def bindings_from_semantics(semantics: Optional["ActionSemantics"]) -> Dict[str, str]:
    """Map match-leaf node-id → symbol display-name for ENTITY slots.

    Uses ``semantics.slot_filling.slots[i].entity_description.name`` when
    present.  Slots without an entity description (ENUM, primitive, pose) are
    omitted — those produce no symbol-graph binding.
    """
    if semantics is None or semantics.slot_filling is None:
        return {}

    bindings: Dict[str, str] = {}
    for slot in semantics.slot_filling.slots:
        desc = slot.entity_description
        if desc is None or not desc.name:
            continue
        bindings[slot.field_name] = desc.name
    return bindings


# ── Snapshot container ────────────────────────────────────────────────────────


@dataclass
class MatchResolutionSnapshot:
    """Frozen state of a match-resolution cycle, ready for rendering."""

    before: MatchTree
    after: Optional[MatchTree] = None
    symbol_tree: Optional[SymbolTree] = None
    bindings: Dict[str, str] = field(default_factory=dict)
    newly_filled: Set[str] = field(default_factory=set)

    @classmethod
    def from_match(
        cls,
        expression: "Match[Any]",
        symbol_type: Optional[Type["Symbol"]] = None,
        symbol_graph: Optional["SymbolGraph"] = None,
    ) -> "MatchResolutionSnapshot":
        """Capture the *before* state: an initial MatchTree + current SymbolTree."""
        before = build_match_tree(snapshot_match(expression))
        symbol_tree = build_symbol_tree(
            symbol_type=symbol_type,
            symbol_graph=symbol_graph,
        )
        return cls(before=before, symbol_tree=symbol_tree)

    def record_after(
        self,
        expression: "Match[Any]",
        backend: Optional["LLMBackend"] = None,
    ) -> None:
        """Capture the *after* state and derive the diff + bindings in place."""
        self.after = build_match_tree(snapshot_match(expression))
        self.newly_filled = diff_match_trees(self.before, self.after)
        semantics = getattr(backend, "semantics", None) if backend else None
        self.bindings = bindings_from_semantics(semantics)
