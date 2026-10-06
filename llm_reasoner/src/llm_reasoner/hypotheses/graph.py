"""
Projecting a stored hypothesis into a graph centred on its instruction, with its
fragments linked by the ontology properties they state.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from enum import StrEnum
from operator import attrgetter

import rustworkx as rx
from typing_extensions import Any, Optional

from llm_reasoner.admission.citation import Citation
from llm_reasoner.generation.schema import is_inferred
from llm_reasoner.hypotheses.view import HypothesisView
from llm_reasoner.ontologies.terms import DescriptionEntity, PropertyTerm

GROUPED_FIELD_PREFIXES = ("defines_", "has_")
"""
The prefixes left out of a root field's name when it labels a group.
"""

# %% graph model


class HypothesisGraphPresentation(StrEnum):
    """
    How the root's links are drawn.
    """

    GROUPED = "grouped"
    """
    One group node per root field stands between the root and the fragments it links.
    """

    COMPLETE = "complete"
    """
    The root links each fragment directly.
    """


class HypothesisGraphNodeKind(StrEnum):
    """
    What a node of the graph stands for.
    """

    INSTRUCTION = "instruction"
    """
    The instruction the hypothesis answers.
    """

    ROOT = "root"
    """
    The fragment the hypothesis was generated for.
    """

    GROUP = "group"
    """
    The fragments one field of the root links, drawn as one node.
    """

    FRAGMENT = "fragment"
    """
    Any other fragment.
    """


class SyntheticHypothesisRelation(StrEnum):
    """
    Links the graph draws that no ontology property states.
    """

    DESCRIBES = "describes"
    """
    From the instruction to the root.
    """

    GROUPS = "groups"
    """
    From the root to one of its groups.
    """


@dataclass(frozen=True)
class HypothesisGraphNode:
    """
    A node of the graph.
    """

    identifier: str
    """
    The identity of the node; a fragment's is its citation's IRI.
    """

    label: str
    """
    The text drawn on the node.
    """

    type_name: str
    """
    The name of the vocabulary class of the fragment, or of what else the node is.
    """

    kind: HypothesisGraphNodeKind
    """
    What the node stands for.
    """

    source: Optional[DescriptionEntity] = field(default=None, compare=False, repr=False)
    """
    The fragment the node stands for, or ``None`` for the instruction and the groups.
    """


@dataclass(frozen=True)
class HypothesisGraphRelation:
    """
    A link of the graph.
    """

    source: HypothesisGraphNode
    """
    The node the link starts at.
    """

    predicate: str
    """
    The field stating the link, or the synthetic relation it is.
    """

    target: HypothesisGraphNode
    """
    The node the link ends at.
    """


@dataclass
class HypothesisGraph:
    """
    A hypothesis as a directed graph centred on its instruction.
    """

    graph: rx.PyDiGraph
    """
    The nodes and links, as node and edge payloads.
    """

    instruction_index: int
    """
    The index of the instruction node.
    """

    root_index: int
    """
    The index of the root's node.
    """

    @property
    def nodes(self) -> list[HypothesisGraphNode]:
        """
        Every node.
        """
        return list(self.graph.nodes())

    @property
    def relations(self) -> list[HypothesisGraphRelation]:
        """
        Every link.
        """
        return list(self.graph.edges())


# %% projection


@dataclass
class HypothesisGraphProjector:
    """
    Projects stored hypotheses into graphs.
    """

    def project(
        self,
        view: HypothesisView,
        presentation: HypothesisGraphPresentation = HypothesisGraphPresentation.GROUPED,
    ) -> HypothesisGraph:
        """
        :return: The graph of every fragment of the hypothesis, with a link for every
            value a fragment states of an ontology property, leaving out properties the
            ontology infers from sub-properties.
        """
        graph = rx.PyDiGraph(multigraph=True)
        instruction_index = graph.add_node(
            HypothesisGraphNode(
                identifier=Citation.namespace_of(view.record.record_id),
                label=view.record.instruction,
                type_name=HypothesisGraphNodeKind.INSTRUCTION.value,
                kind=HypothesisGraphNodeKind.INSTRUCTION,
            )
        )
        indices = {
            id(fragment): graph.add_node(self.fragment_node(view, fragment))
            for fragment in view.fragments
        }
        root_index = indices[id(view.root)]
        self.link(
            graph, instruction_index, SyntheticHypothesisRelation.DESCRIBES, root_index
        )
        for fragment in view.fragments:
            links = self.links_of(fragment, indices)
            if (
                fragment is view.root
                and presentation is HypothesisGraphPresentation.GROUPED
            ):
                self.link_through_groups(graph, root_index, links)
                continue
            for name, targets in links.items():
                for target_index in targets:
                    self.link(graph, indices[id(fragment)], name, target_index)
        return HypothesisGraph(
            graph=graph, instruction_index=instruction_index, root_index=root_index
        )

    @staticmethod
    def fragment_node(
        view: HypothesisView, fragment: DescriptionEntity
    ) -> HypothesisGraphNode:
        """
        :return: The node standing for a fragment.
        """
        return HypothesisGraphNode(
            identifier=view.citation_of(fragment).iri,
            label=fragment.label or type(fragment).__name__,
            type_name=type(fragment).__name__,
            kind=(
                HypothesisGraphNodeKind.ROOT
                if fragment is view.root
                else HypothesisGraphNodeKind.FRAGMENT
            ),
            source=fragment,
        )

    @staticmethod
    def links_of(
        fragment: DescriptionEntity, indices: dict[int, int]
    ) -> dict[str, list[int]]:
        """
        :return: The indices of the fragments each ontology property of a fragment
            links it to, by field name, leaving out properties the ontology infers.
        """
        links: dict[str, list[int]] = {}
        for declared in fields(fragment):
            if PropertyTerm.of_field(
                type(fragment), declared.name
            ) is None or is_inferred(type(fragment), declared.name):
                continue
            value: Any = attrgetter(declared.name)(fragment)
            targets = [
                indices[id(held)]
                for held in (value if isinstance(value, list) else [value])
                if id(held) in indices
            ]
            if targets:
                links[declared.name] = list(dict.fromkeys(targets))
        return links

    def link_through_groups(
        self, graph: rx.PyDiGraph, root_index: int, links: dict[str, list[int]]
    ) -> None:
        """
        Link the root to one group node per field, and each group to the fragments its
        field links.
        """
        root = graph[root_index]
        for name, targets in links.items():
            group_index = graph.add_node(
                HypothesisGraphNode(
                    identifier=f"{root.identifier}/{name}",
                    label=self.group_label(name, len(targets)),
                    type_name=HypothesisGraphNodeKind.GROUP.value,
                    kind=HypothesisGraphNodeKind.GROUP,
                )
            )
            self.link(
                graph, root_index, SyntheticHypothesisRelation.GROUPS, group_index
            )
            for target_index in targets:
                self.link(graph, group_index, name, target_index)

    @staticmethod
    def group_label(field_name: str, member_count: int) -> str:
        """
        :return: The plural, title-cased subject of a field, with its member count.
        """
        subject = field_name
        for prefix in GROUPED_FIELD_PREFIXES:
            subject = subject.removeprefix(prefix)
        words = subject.replace("_", " ")
        plural = words if words.endswith("s") else f"{words}s"
        return f"{plural.title()} ({member_count})"

    @staticmethod
    def link(
        graph: rx.PyDiGraph, source_index: int, predicate: str, target_index: int
    ) -> None:
        """
        Add a link between two nodes of the graph.
        """
        graph.add_edge(
            source_index,
            target_index,
            HypothesisGraphRelation(
                source=graph[source_index],
                predicate=predicate,
                target=graph[target_index],
            ),
        )
