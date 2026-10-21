"""
The payloads of the three views the add-on's panels draw: the hypothesis graph, the
ontology graph behind it, and the designator the active resolution made.
"""

from __future__ import annotations

from enum import StrEnum

from llm_reasoner.knowledge.access import HypothesisGraphNode
from typing_extensions import Any, Sequence

from cramera_resolution import designator_knowledge, hypothesis_knowledge
from cramera_resolution.designator_description import DesignatorDescriber
from cramera_resolution.hypothesis_knowledge.ontology_backing import OntologyBacking
from cramera_resolution.hypothesis_knowledge.ontology_class_graph import (
    OntologyClassGraph,
)
from cramera_resolution.payload import PayloadKey
from cramera_resolution.scenes import SceneObject

# %% what a view is called and says


class ViewName(StrEnum):
    """
    The views the add-on serves, by the name a request asks for.
    """

    HYPOTHESIS = "hypothesis"
    """
    The projection of the active hypothesis, phase first.
    """

    ONTOLOGY = "ontology"
    """
    The classes behind the active hypothesis.
    """

    DESIGNATOR = "designator"
    """
    The designator the active resolution made.
    """


class ViewMessage(StrEnum):
    """
    What a view says instead of a graph.
    """

    NOTHING_GENERATED = (
        "Nothing generated yet — pick a designator below the search bar and click "
        "Generate."
    )
    """
    Before a designator has been resolved.
    """

    NO_HYPOTHESIS_GRAPH = "No hypothesis knowledge graph available."
    """
    When the active hypothesis projects to nothing.
    """

    NO_CLASSES = "No classes behind this hypothesis."
    """
    When the active hypothesis instantiates no class.
    """

    UNKNOWN_VIEW = "unknown view: %s"
    """
    For a name that is none of the views; filled with the name.
    """


# %% the views


def view_payload(name: str, objects: Sequence[SceneObject]) -> dict[str, Any]:
    """
    :param name: The view's name.
    :param objects: The objects of the scene the view is of.
    :return: The view's payload, or an error payload for a name that is none of the views.
    """
    if name == ViewName.HYPOTHESIS:
        return hypothesis_graph_view()
    if name == ViewName.ONTOLOGY:
        return ontology_graph_view()
    if name == ViewName.DESIGNATOR:
        return designator_view(objects)
    return {
        PayloadKey.OK: False,
        PayloadKey.ERROR: ViewMessage.UNKNOWN_VIEW % name,
    }


def designator_view(objects: Sequence[SceneObject]) -> dict[str, Any]:
    """
    The designator for the active instruction: the slots of coraplex's pouring action,
    reflected off the class, followed by the generated hypothesis described along its
    ontology properties.

    Once a predefined designator has been resolved, the slots hold the values the resolver
    chose, with why and what they rest on. Before that it is illustrative: a container slot
    names the scene object reading like the hypothesis's filler of its role.

    :param objects: The objects of the scene.
    """
    active_hypothesis = hypothesis_knowledge.active()
    view = active_hypothesis.phase_structure if active_hypothesis else None
    resolution = active_hypothesis.resolution if active_hypothesis else None
    knowledge = designator_knowledge.reflect_pouring_action(objects, view, resolution)
    return {
        PayloadKey.OK: True,
        PayloadKey.CRUMB: "%s designator" % knowledge.designator_name,
        PayloadKey.DESIGNATOR: DesignatorDescriber(view=view)
        .describe(knowledge)
        .payload(),
    }


def hypothesis_node_lines(node: HypothesisGraphNode) -> list[str]:
    """
    :param node: One projected node.
    :return: The lines of its detail panel.
    """
    lines = [
        "kind: " + node.kind.value,
        "type: " + node.type_name,
        "identifier: " + node.identifier,
    ]
    if node.source is not None and node.source.label:
        lines.append("says: " + node.source.label)
    return lines


def hypothesis_graph_view() -> dict[str, Any]:
    """
    The projection of the active pouring structure, presented phase first, with every
    node and edge of it. Empty until an instruction has been generated.
    """
    active_hypothesis = hypothesis_knowledge.active()
    if active_hypothesis is None:
        return {
            PayloadKey.OK: True,
            PayloadKey.CRUMB: "hypothesis graph",
            PayloadKey.ROOT: None,
            PayloadKey.NODES: [],
            PayloadKey.EDGES: [],
            PayloadKey.DETAILS: {},
            PayloadKey.EMPTY: ViewMessage.NOTHING_GENERATED,
        }
    projection = active_hypothesis.projection
    return {
        PayloadKey.OK: True,
        PayloadKey.CRUMB: "hypothesis graph",
        PayloadKey.NODES: [
            {
                PayloadKey.ID: node.identifier,
                PayloadKey.LABEL: node.label,
                PayloadKey.KIND: node.kind.value,
            }
            for node in projection.nodes
        ],
        PayloadKey.EDGES: [
            {
                PayloadKey.FROM: relation.source.identifier,
                PayloadKey.TO: relation.target.identifier,
                PayloadKey.LABEL: relation.predicate,
            }
            for relation in projection.relations
        ],
        PayloadKey.DETAILS: {
            node.identifier: {
                PayloadKey.LABEL: node.label,
                PayloadKey.GROUP: ViewName.HYPOTHESIS,
                PayloadKey.LINES: hypothesis_node_lines(node),
            }
            for node in projection.nodes
        },
        PayloadKey.ROOT: projection.instruction.identifier,
        PayloadKey.EMPTY: ViewMessage.NO_HYPOTHESIS_GRAPH,
    }


def ontology_graph_view() -> dict[str, Any]:
    """
    The ontology behind the active hypothesis at the level of classes: the microworld
    classes it instantiates, the classes they specialise up to CRAM-AICOR and DUL, the
    concepts it cites, and the links between them. Empty until an instruction has been
    generated.
    """
    active_hypothesis = hypothesis_knowledge.active()
    if active_hypothesis is None:
        return {
            PayloadKey.OK: True,
            PayloadKey.CRUMB: "ontology graph",
            PayloadKey.NODES: [],
            PayloadKey.EDGES: [],
            PayloadKey.DETAILS: {},
            PayloadKey.EMPTY: ViewMessage.NOTHING_GENERATED,
        }
    graph = OntologyClassGraph.of(active_hypothesis.view, *active_hypothesis.consulted)
    backing = OntologyBacking()
    return {
        PayloadKey.OK: True,
        PayloadKey.CRUMB: "ontology graph",
        PayloadKey.INSTRUCTION: active_hypothesis.view.record.instruction,
        PayloadKey.NODES: [
            {
                PayloadKey.ID: node.iri,
                PayloadKey.LABEL: node.label,
                PayloadKey.CATEGORY: node.category.value,
                PayloadKey.COUNT: node.instance_count,
            }
            for node in graph.nodes
        ],
        PayloadKey.EDGES: [
            {
                PayloadKey.SOURCE: edge.source,
                PayloadKey.TARGET: edge.target,
                PayloadKey.KIND: edge.kind.value,
                PayloadKey.LABEL: (
                    backing.prefixed(edge.property_iri) if edge.property_iri else ""
                ),
                PayloadKey.COUNT: edge.count,
            }
            for edge in graph.edges
        ],
        PayloadKey.DETAILS: {
            node.iri: {
                PayloadKey.LABEL: node.label,
                PayloadKey.IRI: node.iri,
                PayloadKey.PREFIXED: backing.prefixed(node.iri),
                PayloadKey.CATEGORY: node.category.value,
                PayloadKey.SOURCE: node.source.value,
                PayloadKey.DESCRIPTION: node.description,
                PayloadKey.EVIDENCE: node.evidence_status,
                PayloadKey.CLAIMS: [
                    {
                        PayloadKey.ID: backing.prefixed(claim.iri),
                        PayloadKey.TEXT: claim.text,
                        PayloadKey.EVIDENCE: claim.evidence_status,
                    }
                    for claim in node.claims
                ],
                PayloadKey.REQUIREMENTS: list(node.requirements),
                PayloadKey.DEFINITION: node.definition,
                PayloadKey.INSTANCES: list(node.instances),
                PayloadKey.COUNT: node.instance_count,
            }
            for node in graph.nodes
        },
        PayloadKey.EMPTY: ViewMessage.NO_CLASSES,
    }
