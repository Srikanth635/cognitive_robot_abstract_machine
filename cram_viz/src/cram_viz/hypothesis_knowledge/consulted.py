"""
The hypotheses a resolution consulted beside the phase structure: the action core's
frame and each perspective on it, drawn in the same graph as the phase structure.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field

from llm_reasoner.knowledge.access import (
    HypothesisGraph,
    HypothesisGraphNode,
    HypothesisGraphPresentation,
    HypothesisGraphProjector,
    HypothesisView,
    SyntheticHypothesisRelation,
)

from cram_viz.hypothesis_knowledge.ontology_backing import OntologyBacking


@dataclass(frozen=True)
class ConsultedHypotheses:
    """
    The core hypothesis and the perspectives on it a resolution consulted.
    """

    views: tuple[HypothesisView, ...] = ()
    """
    The consulted hypotheses, the core first; a perspective's fragments link the core's
    slots and dimensions themselves.
    """

    backing: OntologyBacking = field(default_factory=OntologyBacking)
    """
    Names the module each hypothesis is written in.
    """

    def attach(self, projection: HypothesisGraph) -> HypothesisGraph:
        """
        Draw the consulted hypotheses in a projection, each root described by the
        projection's instruction and titled by the module its hypothesis is written in.

        :return: The same projection, unchanged when nothing was consulted.
        """
        if not self.views:
            return projection
        consulted = HypothesisGraphProjector().project_together(
            self.views, HypothesisGraphPresentation.COMPLETE
        )
        index_of = {
            index: projection.graph.add_node(self.titled(consulted.graph[index]))
            for index in consulted.graph.node_indices()
            if index != consulted.instruction_index
        }
        index_of[consulted.instruction_index] = projection.instruction_index
        for source, target, relation in consulted.graph.weighted_edge_list():
            HypothesisGraphProjector.link(
                projection.graph,
                index_of[source],
                relation.predicate,
                index_of[target],
                relation.property_iri,
            )
        return projection

    def titled(self, node: HypothesisGraphNode) -> HypothesisGraphNode:
        """
        :return: The node, a root labelled with the title of its module instead of its
            own sentence.
        """
        root_iris = {type(view.root).term.iri for view in self.views}
        if node.ontology_class not in root_iris or node.source not in [
            view.root for view in self.views
        ]:
            return node
        namespace = self.backing.namespace_of(node.ontology_class)
        if namespace is None:
            return node
        return dataclasses.replace(node, label=namespace.title)
