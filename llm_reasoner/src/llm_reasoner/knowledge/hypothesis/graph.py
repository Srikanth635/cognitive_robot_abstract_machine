"""
Project a generated hypothesis into a rooted, visualizable graph.
"""

from __future__ import annotations

import _thread
import json
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum
from math import cos, pi, sin
from typing import TYPE_CHECKING, Any, cast

import rustworkx as rx
from typing_extensions import Self

from krrood.rustworkx_utils.graph_visualizer_base import GraphLayout
from krrood.rustworkx_utils.visualization.visnetwork_graph_visualizer import (
    VisEdge,
    VisNetworkGraphVisualizer,
    VisNode,
)
from llm_reasoner.knowledge.hypothesis.record import HypothesisRecord
from llm_reasoner.knowledge.hypothesis.scope import collect_reachable
from llm_reasoner.knowledge.hypothesis.presentation import (
    OntologyGraphPresentation,
)
from llm_reasoner.knowledge.ontological.decorators import (
    KnowledgeEntity,
    property_meta,
)

if TYPE_CHECKING:
    from flask import Flask, Response

# %% Projection model


class HypothesisGraphPresentation(Enum):
    """
    How ontology relationships are organized for presentation.
    """

    GROUPED = "grouped"
    """
    Place virtual category nodes between the root and its declarations.
    """

    COMPLETE = "complete"
    """
    Show every ontology relationship directly.
    """


class HypothesisGraphNodeKind(Enum):
    """
    The structural purpose of a node in a projected hypothesis.
    """

    INSTRUCTION = "instruction"
    ROOT = "root"
    GROUP = "group"
    FRAGMENT = "fragment"


@dataclass(frozen=True)
class HypothesisGraphNode:
    """
    A labelled node that retains its originating knowledge object.
    """

    identifier: str
    """
    Identity unique within the projected hypothesis.
    """

    label: str
    """
    Human-readable text shown in the visualization.
    """

    type_name: str
    """
    Name of the represented knowledge type.
    """

    kind: HypothesisGraphNodeKind
    """
    Structural purpose of this node in the projection.
    """

    source: Any = field(default=None, compare=False, repr=False)
    """
    Original knowledge object, or ``None`` for a synthetic presentation node.
    """

    def graph_label(self) -> str:
        """
        Return the text drawn on the node.
        """
        return self.label

    def graph_information(self) -> list[str]:
        """
        Return the details shown when the node is selected.
        """
        return [
            f"kind: {self.kind.value}",
            f"type: {self.type_name}",
            f"identifier: {self.identifier}",
        ]

    def graph_color(self) -> str:
        """
        Return a color distinguishing the node's structural purpose.
        """
        match self.kind:
            case HypothesisGraphNodeKind.INSTRUCTION:
                return "#e76f51"
            case HypothesisGraphNodeKind.ROOT:
                return "#e9c46a"
            case HypothesisGraphNodeKind.GROUP:
                return "#457b9d"
            case HypothesisGraphNodeKind.FRAGMENT:
                return "#2a9d8f"


@dataclass(frozen=True)
class HypothesisGraphRelation:
    """
    A labelled parent-to-child connection between two projected nodes.
    """

    source: HypothesisGraphNode
    """
    Node owning the ontological property.
    """

    predicate: str
    """
    Ontological field or synthetic relation naming the connection.
    """

    target: HypothesisGraphNode
    """
    Node referenced by the property.
    """


@dataclass
class HypothesisGraph:
    """
    A hypothesis projection rooted at a central instruction node.
    """

    graph: rx.PyDiGraph = field(default_factory=rx.PyDiGraph)
    """
    Directed graph containing projected nodes and labelled relations.
    """

    instruction_index: int = 0
    """
    Rustworkx index of the instruction node.
    """

    root_index: int = 0
    """
    Rustworkx index of the hypothesis's top-level knowledge fragment.
    """

    @property
    def instruction(self) -> HypothesisGraphNode:
        """:return: The central instruction node."""
        return self.graph[self.instruction_index]

    @property
    def root(self) -> HypothesisGraphNode:
        """:return: The top-level knowledge fragment node."""
        return self.graph[self.root_index]

    @property
    def nodes(self) -> tuple[HypothesisGraphNode, ...]:
        """:return: Every projected node in graph order."""
        return tuple(self.graph.nodes())

    @property
    def relations(self) -> tuple[HypothesisGraphRelation, ...]:
        """:return: Every labelled parent-to-child relation."""
        return tuple(self.graph.edges())


class HypothesisGraphProjectionError(ValueError):
    """
    Raised when a hypothesis has no projectable knowledge root.
    """


# %% Hypothesis projection


class SyntheticHypothesisRelation(Enum):
    """
    Relations introduced by the presentation layer.
    """

    DESCRIBES = "describes"
    GROUPS = "groups"


@dataclass(frozen=True)
class HypothesisGraphProjector:
    """
    Convert ontology-described hypothesis references into graph edges.
    """

    def project(
        self,
        record: HypothesisRecord,
        instruction: str | None = None,
        presentation: HypothesisGraphPresentation = HypothesisGraphPresentation.GROUPED,
    ) -> HypothesisGraph:
        """
        Project the fragments reachable from a hypothesis root.

        :param record: Hypothesis whose knowledge structure is projected.
        :param instruction: Text shown at the center of the graph. The action label is
            used when omitted.
        :param presentation: Relationship organization used for display.
        :return: A rooted graph containing every reachable knowledge fragment.
        :raises HypothesisGraphProjectionError: If the record has no knowledge entity as
            its root.
        """
        root = record.root
        if not isinstance(root, KnowledgeEntity):
            raise HypothesisGraphProjectionError(
                "the hypothesis record has no projectable knowledge root"
            )

        graph = rx.PyDiGraph(multigraph=True)
        instruction_node = self._instruction_node(record, instruction)
        instruction_index = graph.add_node(instruction_node)

        fragments = [
            fragment
            for fragment in collect_reachable(root)
            if isinstance(fragment, KnowledgeEntity)
        ]
        fragment_indices: dict[int, int] = {}
        for fragment in fragments:
            kind = (
                HypothesisGraphNodeKind.ROOT
                if fragment is root
                else HypothesisGraphNodeKind.FRAGMENT
            )
            node = HypothesisGraphNode(
                identifier=fragment.local_id,
                label=fragment.label or type(fragment).__name__,
                type_name=type(fragment).__name__,
                kind=kind,
                source=fragment,
            )
            fragment_indices[id(fragment)] = graph.add_node(node)

        root_index = fragment_indices[id(root)]
        self._add_relation(
            graph=graph,
            source_index=instruction_index,
            predicate=SyntheticHypothesisRelation.DESCRIBES.value,
            target_index=root_index,
        )
        self._add_fragment_relations(
            graph=graph,
            fragments=fragments,
            fragment_indices=fragment_indices,
            root=root,
            presentation=presentation,
        )
        return HypothesisGraph(
            graph=graph,
            instruction_index=instruction_index,
            root_index=root_index,
        )

    @staticmethod
    def _instruction_node(
        record: HypothesisRecord, instruction: str | None
    ) -> HypothesisGraphNode:
        """
        Build the central display node for a hypothesis's instruction.
        """
        supplied_label = instruction.strip() if instruction is not None else ""
        action_label = record.action_type.label or str(record.action_type)
        return HypothesisGraphNode(
            identifier=f"instruction:{record.record_id}",
            label=supplied_label or action_label,
            type_name="Instruction",
            kind=HypothesisGraphNodeKind.INSTRUCTION,
        )

    def _add_fragment_relations(
        self,
        graph: rx.PyDiGraph,
        fragments: list[KnowledgeEntity],
        fragment_indices: dict[int, int],
        root: KnowledgeEntity,
        presentation: HypothesisGraphPresentation,
    ) -> None:
        """
        Add every declared reference whose target is in the projection.
        """
        added_relations: set[tuple[int, str, int]] = set()
        for source in fragments:
            if source is root and presentation is HypothesisGraphPresentation.GROUPED:
                self._add_grouped_root_relations(
                    graph=graph,
                    root=root,
                    fragment_indices=fragment_indices,
                )
                continue
            source_identity = id(source)
            for predicate in property_meta(type(source)):
                value = vars(source)[predicate]
                for target in self._property_values(value):
                    target_identity = id(target)
                    if target_identity not in fragment_indices:
                        continue
                    relation_key = (source_identity, predicate, target_identity)
                    if relation_key in added_relations:
                        continue
                    added_relations.add(relation_key)
                    self._add_relation(
                        graph=graph,
                        source_index=fragment_indices[source_identity],
                        predicate=predicate,
                        target_index=fragment_indices[target_identity],
                    )

    def _add_grouped_root_relations(
        self,
        graph: rx.PyDiGraph,
        root: KnowledgeEntity,
        fragment_indices: dict[int, int],
    ) -> None:
        """
        Place category nodes between the root and its direct declarations.
        """
        root_index = fragment_indices[id(root)]
        for predicate in property_meta(type(root)):
            value = vars(root)[predicate]
            target_indices: list[int] = []
            seen_target_indices: set[int] = set()
            for target in self._property_values(value):
                target_identity = id(target)
                if target_identity not in fragment_indices:
                    continue
                target_index = fragment_indices[target_identity]
                if target_index in seen_target_indices:
                    continue
                seen_target_indices.add(target_index)
                target_indices.append(target_index)
            if not target_indices:
                continue

            group_node = HypothesisGraphNode(
                identifier=f"group:{root.local_id}:{predicate}",
                label=self._group_label(predicate, len(target_indices)),
                type_name="RelationshipGroup",
                kind=HypothesisGraphNodeKind.GROUP,
            )
            group_index = graph.add_node(group_node)
            self._add_relation(
                graph=graph,
                source_index=root_index,
                predicate=SyntheticHypothesisRelation.GROUPS.value,
                target_index=group_index,
            )
            for target_index in target_indices:
                self._add_relation(
                    graph=graph,
                    source_index=group_index,
                    predicate=predicate,
                    target_index=target_index,
                )

    @staticmethod
    def _group_label(predicate: str, member_count: int) -> str:
        """
        Build a readable plural category label with its member count.
        """
        category = predicate.removeprefix("defines_").removeprefix("has_")
        words = category.replace("_", " ")
        plural_words = words if words.endswith("s") else f"{words}s"
        return f"{plural_words.title()} ({member_count})"

    @staticmethod
    def _property_values(value: Any) -> tuple[Any, ...]:
        """
        Normalize a scalar or collection-valued property for traversal.
        """
        if isinstance(value, (list, tuple, set, frozenset)):
            return tuple(value)
        return (value,)

    @staticmethod
    def _add_relation(
        graph: rx.PyDiGraph,
        source_index: int,
        predicate: str,
        target_index: int,
    ) -> None:
        """
        Add one labelled relation between existing graph nodes.
        """
        relation = HypothesisGraphRelation(
            source=graph[source_index],
            predicate=predicate,
            target=graph[target_index],
        )
        graph.add_edge(source_index, target_index, relation)


# %% Visualization style


@dataclass(frozen=True)
class HypothesisGraphStyle:
    """
    Visual styling and separation settings for a hypothesis graph.
    """

    canvas_background_color: str = "#ffffff"
    """
    Background color of the graph canvas and details panel.
    """

    node_text_color: str = "#202124"
    """
    Color used for node labels.
    """

    edge_text_color: str = "#5f6368"
    """
    Color used for relationship labels.
    """

    edge_color: str = "#9aa0a6"
    """
    Color used for relationship lines.
    """

    font_family: str = "Arial"
    """
    Typeface used for graph labels.
    """

    node_font_size: int = 12
    """
    Font size used for node labels.
    """

    edge_font_size: int = 10
    """
    Font size used for relationship labels.
    """

    node_size: int = 13
    """
    Diameter setting used for node markers.
    """

    repulsion_node_distance: int = 220
    """
    Node separation requested from the repulsion solver.
    """

    def node_font_options(self) -> dict[str, Any]:
        """:return: Standard-weight vis-network node font options."""
        return {
            "color": self.node_text_color,
            "face": self.font_family,
            "size": self.node_font_size,
            "strokeWidth": 0,
        }

    def edge_font_options(self) -> dict[str, Any]:
        """:return: Standard-weight vis-network relationship font options."""
        return {
            "background": self.canvas_background_color,
            "color": self.edge_text_color,
            "face": self.font_family,
            "size": self.edge_font_size,
            "strokeWidth": 0,
        }


# %% Interactive visualization


@dataclass
class HypothesisGraphVisualizer(VisNetworkGraphVisualizer):
    """
    Render a hypothesis with its instruction fixed at the center.
    """

    center_index: int = 0
    """
    Rustworkx index placed at the center of the radial layout.
    """

    radial_spacing: float = 240.0
    """
    Distance between consecutive graph-depth rings.
    """

    layout: GraphLayout = GraphLayout.PHYSICS
    """
    Placement strategy, repulsion physics by default.
    """

    style: HypothesisGraphStyle = field(default_factory=HypothesisGraphStyle)
    """
    Typography, color, and spacing settings used by the browser visualization.
    """

    _expanded_indices: set[int] = field(default_factory=set, init=False, repr=False)
    """
    Nodes whose immediate children are visible.
    """

    _expansion_lock: _thread.LockType = field(
        default_factory=_thread.allocate_lock, init=False, repr=False
    )
    """
    Lock protecting expansion state shared with the visualization server.
    """

    @classmethod
    def for_graph(cls, projection: HypothesisGraph, **overrides: Any) -> Self:
        """
        Create a visualizer configured for a projected hypothesis.

        :param projection: Rooted hypothesis graph to display.
        :param overrides: Visualizer settings replacing the defaults.
        :return: A visualizer ready to run.
        """
        arguments: dict[str, Any] = {
            "graph": projection.graph,
            "center_index": projection.instruction_index,
            "label_getter": HypothesisGraphNode.graph_label,
            "information_getter": HypothesisGraphNode.graph_information,
            "color_getter": HypothesisGraphNode.graph_color,
            "layout": GraphLayout.PHYSICS,
            "title": "Hypothesis Knowledge Graph",
        }
        arguments.update(overrides)
        return cls(**arguments)

    @classmethod
    def for_ontology_graph(
        cls,
        projection: HypothesisGraph,
        presentation: OntologyGraphPresentation | None = None,
        **overrides: Any,
    ) -> Self:
        """
        Create a visualizer using concise ontology-derived labels.

        :param projection: Rooted hypothesis graph to display.
        :param presentation: Label and information policy to apply.
        :param overrides: Visualizer settings replacing the defaults.
        :return: A visualizer ready to run.
        """
        selected_presentation = presentation or OntologyGraphPresentation()
        arguments: dict[str, Any] = {
            "label_getter": selected_presentation.label,
            "information_getter": selected_presentation.information,
        }
        arguments.update(overrides)
        return cls.for_graph(projection, **arguments)

    def network_options(self) -> dict[str, Any]:
        """:return: Browser options for typography, edges, and graph layout."""
        options: dict[str, Any] = {
            "nodes": {
                "shape": "dot",
                "size": self.style.node_size,
                "font": self.style.node_font_options(),
            },
            "edges": {
                "color": self.style.edge_color,
                "arrows": "to",
                "font": self.style.edge_font_options(),
            },
        }
        options.update(self.layout_options.get_options(self.layout))
        match self.layout:
            case GraphLayout.PHYSICS:
                options["physics"] = {
                    "enabled": True,
                    "solver": "repulsion",
                    "repulsion": {
                        "nodeDistance": self.style.repulsion_node_distance,
                    },
                }
            case GraphLayout.FIXED:
                options["physics"] = {"enabled": False}
        return options

    def build_application(self) -> Flask:
        """:return: The progressive hypothesis graph browser application."""
        from flask import Flask, jsonify, render_template

        application = Flask(__name__)

        @application.route("/")
        def index() -> str:
            """:return: The interactive hypothesis graph page."""
            return render_template(
                "hypothesis_graph_visualizer.jinja",
                title=self.title,
                network_options_json=json.dumps(self.network_options()),
                canvas_background_color=self.style.canvas_background_color,
                interval_milliseconds=int(self.refresh_interval_seconds * 1000),
            )

        @application.route("/graph")
        def graph_data() -> Response:
            """:return: Currently visible graph nodes and edges."""
            return jsonify(nodes=self.graph_nodes(), edges=self.graph_edges())

        @application.route("/node/<int:node_index>")
        def node(node_index: int) -> Response:
            """:return: Selected node details after expanding its children."""
            return jsonify(details=self.node_details(node_index))

        @application.route("/collapse/<int:node_index>", methods=["POST"])
        def collapse_node(node_index: int) -> Response:
            """:return: Confirmation after collapsing a selected branch."""
            self.collapse_branch(node_index)
            return jsonify(collapsed=node_index)

        return application

    def node_details(self, node_index: int) -> list[str]:
        """:return: Node details after revealing the node's immediate children."""
        details = super().node_details(node_index)
        with self._expansion_lock:
            self._expanded_indices.add(node_index)
        return details

    def graph_nodes(self) -> list[VisNode]:
        """:return: Visible nodes seeded radially for the selected layout."""
        visible_indices = self._visible_node_indices()
        nodes = [
            node for node in super().graph_nodes() if int(node["id"]) in visible_indices
        ]
        if self.layout not in {GraphLayout.FIXED, GraphLayout.PHYSICS}:
            return nodes

        positions = self._radial_positions()
        for node in nodes:
            node_index = int(node["id"])
            horizontal, vertical = positions[node_index]
            is_fixed = (
                node_index == self.center_index or self.layout is GraphLayout.FIXED
            )
            node["x"] = horizontal
            node["y"] = vertical
            node["fixed"] = is_fixed
            node["physics"] = not is_fixed
        return nodes

    def graph_edges(self) -> list[VisEdge]:
        """:return: Visual edges carrying their ontological property labels."""
        expanded_indices = self._expanded_node_indices()
        visible_indices = self._visible_node_indices(expanded_indices)
        return [
            {
                "id": str(edge_number),
                "from": str(source),
                "to": str(target),
                "label": relation.predicate,
            }
            for edge_number, (source, target, relation) in enumerate(
                self.graph.weighted_edge_list()
            )
            if source in expanded_indices and target in visible_indices
        ]

    def collapse_all(self) -> None:
        """
        Hide every node except the central instruction.
        """
        with self._expansion_lock:
            self._expanded_indices.clear()

    def collapse_branch(self, node_index: int) -> None:
        """
        Hide the immediate children and expanded descendants of a node.

        :param node_index: Rustworkx index whose visible branch is collapsed.
        """
        directed_graph = cast(rx.PyDiGraph, self.graph)
        branch_indices = set(rx.descendants(directed_graph, node_index))
        branch_indices.add(node_index)
        with self._expansion_lock:
            self._expanded_indices.difference_update(branch_indices)

    def _expanded_node_indices(self) -> frozenset[int]:
        """
        Return a stable snapshot of nodes expanded by browser clicks.
        """
        with self._expansion_lock:
            return frozenset(self._expanded_indices)

    def _visible_node_indices(
        self, expanded_indices: frozenset[int] | None = None
    ) -> set[int]:
        """
        Return the center, expanded nodes, and their immediate children.
        """
        if expanded_indices is None:
            expanded_indices = self._expanded_node_indices()
        visible_indices = {self.center_index}
        directed_graph = cast(rx.PyDiGraph, self.graph)
        for expanded_index in expanded_indices:
            visible_indices.add(expanded_index)
            visible_indices.update(directed_graph.successor_indices(expanded_index))
        return visible_indices

    def _radial_positions(self) -> dict[int, tuple[float, float]]:
        """
        Place graph-depth layers on concentric rings around the center.
        """
        distances = self._distances_from_center()
        layers: dict[int, list[int]] = defaultdict(list)
        for node_index, distance in distances.items():
            layers[distance].append(node_index)

        positions = {self.center_index: (0.0, 0.0)}
        for distance, node_indices in layers.items():
            if distance == 0:
                continue
            ordered_indices = sorted(node_indices)
            radius = distance * self.radial_spacing
            for position, node_index in enumerate(ordered_indices):
                angle = (2 * pi * position / len(ordered_indices)) - (pi / 2)
                positions[node_index] = (
                    radius * cos(angle),
                    radius * sin(angle),
                )
        return positions

    def _distances_from_center(self) -> dict[int, int]:
        """
        Return shortest outgoing-edge distance from the center to each node.
        """
        directed_graph = cast(rx.PyDiGraph, self.graph)
        distances = {self.center_index: 0}
        frontier = deque([self.center_index])
        while frontier:
            current = frontier.popleft()
            for successor in directed_graph.successor_indices(current):
                if successor in distances:
                    continue
                distances[successor] = distances[current] + 1
                frontier.append(successor)

        if len(distances) == self.graph.num_nodes():
            return distances
        disconnected_distance = max(distances.values(), default=0) + 1
        for node_index in self.graph.node_indices():
            if node_index not in distances:
                distances[node_index] = disconnected_distance
        return distances


__all__ = [
    "HypothesisGraph",
    "HypothesisGraphNode",
    "HypothesisGraphNodeKind",
    "HypothesisGraphPresentation",
    "HypothesisGraphProjectionError",
    "HypothesisGraphProjector",
    "HypothesisGraphRelation",
    "HypothesisGraphStyle",
    "HypothesisGraphVisualizer",
    "SyntheticHypothesisRelation",
]
