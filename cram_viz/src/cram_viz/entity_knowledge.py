"""
Static, hand-authored domain knowledge about scene entities (bench objects): their
ontology classification, world properties, reified qualities, affordances and relations.

Mirrors :mod:`cram_viz.action_knowledge` for the entity side of the knowledge graph and
reuses its graph-node-id and rendering building blocks, so an entity's classification
chain and category roots are drawn exactly the same way an action's are.

:data:`ENTITY_KNOWLEDGE` is the registry :mod:`cram_viz.kb` merges into the knowledge
graph, keyed by :class:`~cram_viz.kb.BenchObject` name;
:class:`EntityKnowledgeGraphFragment` renders one registered :class:`EntityKnowledge`
into the node/edge shape the graph payload uses, anchored on the matching object node.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing_extensions import Dict, List, Tuple

from cram_viz.action_knowledge import (
    CategorySpec,
    KnowledgeGraphEdge,
    KnowledgeGraphNode,
    KnowledgeRelation,
    OntologyClass,
    render_knowledge_anchor_edges,
    render_knowledge_fragment,
)


# %% domain model
@dataclass(unsafe_hash=True)
class Property:
    """
    An observed or illustrative world property of an entity, e.g. ``mass: 0.5 kg``.
    """

    name: str
    """
    Property and its value, as one label.
    """

    entity_name: str = field(default="", init=False)
    """
    Name of the entity this property belongs to, auto-stamped by
    :meth:`EntityKnowledge.__post_init__`.
    """


@dataclass(unsafe_hash=True)
class Quality:
    """
    A reified descriptive quality of an entity, e.g. ``material: ceramic``.
    """

    name: str
    """
    Quality and its value, as one label.
    """

    entity_name: str = field(default="", init=False)
    """
    Name of the entity this quality belongs to, auto-stamped by
    :meth:`EntityKnowledge.__post_init__`.
    """


@dataclass(unsafe_hash=True)
class Affordance:
    """
    An action an entity's structure and context are asserted to support.
    """

    name: str
    """
    Affordance identifier, e.g. ``GraspAffordance``.
    """

    entity_name: str = field(default="", init=False)
    """
    Name of the entity this affordance belongs to, auto-stamped by
    :meth:`EntityKnowledge.__post_init__`.
    """


@dataclass(unsafe_hash=True)
class EntityRelation:
    """
    A semantic connection from an entity to another world entity or process.
    """

    name: str
    """
    Relation and its target, as one label, e.g. ``patient_of: TransportAction``.
    """

    entity_name: str = field(default="", init=False)
    """
    Name of the entity this relation belongs to, auto-stamped by
    :meth:`EntityKnowledge.__post_init__`.
    """


@dataclass(unsafe_hash=True)
class EntityKnowledge:
    """
    The static ontology knowledge registered for one scene entity.
    """

    name: str
    """
    The :class:`~cram_viz.kb.BenchObject` name this knowledge is registered for.
    """

    classes: Tuple[OntologyClass, ...] = ()
    """
    The entity's ontology classification, broad to specific.
    """

    properties: Tuple[Property, ...] = ()
    """
    Direct world properties.
    """

    qualities: Tuple[Quality, ...] = ()
    """
    Reified descriptive qualities.
    """

    affordances: Tuple[Affordance, ...] = ()
    """
    Actions the entity's structure and context support.
    """

    relations: Tuple[EntityRelation, ...] = ()
    """
    Semantic connections to other world entities or processes.
    """

    def __post_init__(self) -> None:
        """
        Stamp this entry's entity name onto every property/quality/affordance/relation
        it owns, so each one carries its own back-reference without registry authors
        having to write it by hand.
        """
        for members in (
            self.properties,
            self.qualities,
            self.affordances,
            self.relations,
        ):
            for member in members:
                member.entity_name = self.name

    def matches(self, object_name: str) -> bool:
        """
        Whether this knowledge is registered for the given BenchObject name.
        """
        return object_name == self.name


class EntityKnowledgeGraphFragment:
    """
    Renders one :class:`EntityKnowledge` into knowledge-graph nodes and edges.

    Structurally identical to :class:`~cram_viz.action_knowledge.ActionKnowledgeGraphFragment`
    (both delegate to the shared ``render_knowledge_*`` functions); kept as its own class
    because it is anchored on a different node kind (a BenchObject, matched by exact
    name rather than an action verb's prefix match) and owns a different category set.
    """

    def __init__(self, knowledge: EntityKnowledge) -> None:
        """
        :param knowledge: The registered entity knowledge to render.
        """
        self._knowledge = knowledge

    def anchor_edges(self, anchor_node_id: str) -> List[KnowledgeGraphEdge]:
        """
        The ``has_knowledge`` edges from the matching object node to every category
        root.
        """
        return render_knowledge_anchor_edges(
            anchor_node_id,
            self._knowledge.name,
            self._knowledge.classes,
            self._categories(),
        )

    def nodes_and_edges(
        self,
    ) -> Tuple[List[KnowledgeGraphNode], List[KnowledgeGraphEdge]]:
        """
        The knowledge's own nodes and edges: its ontology-class chain and its
        properties/qualities/affordances/relations, each fanning out from its own
        category root.
        """
        return render_knowledge_fragment(
            self._knowledge.name, self._knowledge.classes, self._categories()
        )

    def _categories(self) -> List[CategorySpec]:
        """
        This knowledge's flat categories (everything but the ontology-class chain).
        """
        knowledge = self._knowledge
        return [
            (
                "properties",
                knowledge.properties,
                "a Property",
                KnowledgeRelation.OBSERVED,
            ),
            (
                "qualities",
                knowledge.qualities,
                "a Quality",
                KnowledgeRelation.HAS_QUALITY,
            ),
            (
                "affordances",
                knowledge.affordances,
                "an Affordance",
                KnowledgeRelation.AFFORDS,
            ),
            (
                "relations",
                knowledge.relations,
                "a Relation",
                KnowledgeRelation.RELATED_BY,
            ),
        ]


# %% registry
#: static, hand-authored registry: BenchObject name -> its entity knowledge
ENTITY_KNOWLEDGE: Dict[str, EntityKnowledge] = {
    "milk": EntityKnowledge(
        name="milk",
        classes=(
            OntologyClass("Entity"),
            OntologyClass("PhysicalObject"),
            OntologyClass("Container"),
        ),
        properties=(
            Property("world type: WorldBody (STL mesh)"),
            Property("shape: rectangular carton"),
            Property("mass: illustrative — approx. 1.0 kg (full carton)"),
            Property("size: illustrative — approx. 0.07 x 0.07 x 0.20 m"),
        ),
        qualities=(
            Quality("material: illustrative — cardboard/plastic composite"),
            Quality("contents: illustrative — liquid"),
            Quality("color: illustrative — white with brand markings"),
            Quality("temperature: illustrative — perishable, kept refrigerated"),
        ),
        affordances=(
            Affordance("GraspAffordance"),
            Affordance("PourableAffordance"),
            Affordance("ContainmentAffordance"),
            Affordance("PlaceableAffordance"),
            Affordance("OpenableAffordance"),
        ),
        relations=(
            EntityRelation("patient_of: PickUpAction"),
            EntityRelation("patient_of: PlaceAction"),
            EntityRelation("patient_of: TransportAction"),
            EntityRelation("supported_by: kitchen counter"),
            EntityRelation("contains: milk (liquid)"),
        ),
    ),
}
