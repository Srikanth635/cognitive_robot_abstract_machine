"""
Static, hand-authored domain knowledge about CRAM actions: what an action verb (e.g.
``transport``) *means* -- its ontology classification, participant roles, a sample
parameterization, preconditions, the phases it decomposes into, and its expected goals.

Unlike :mod:`cram_viz.kb`'s scene-driven entities (built from a recording) or its
architecture graph (scanned from the repository), the knowledge here is authored, not
derived. Every value that is not read from the recorded scene is a plausible
illustrative sample -- clearly not a verified fact -- the same honest convention used by
the ``cognitive_knowledge`` notebooks this module's shape was modelled on.

:data:`ACTION_PHASE_KNOWLEDGE` and :data:`ACTION_KNOWLEDGE` are the registries
:mod:`cram_viz.kb` merges into the knowledge graph, both keyed by action verb;
:class:`ActionPhaseGraphFragment` and :class:`ActionKnowledgeGraphFragment` render one
registered entry into the node/edge shape the graph payload uses, anchored directly on
the matching verb node. A phase/role/parameter/precondition/goal is a node shared
globally by name across every verb that has one with that name (see
:func:`member_node_id`), so e.g. ``pickup`` and ``transport`` both link straight to the
very same ``Grasp`` phase instead of each carrying a private copy.
:mod:`cram_viz.entity_knowledge` mirrors this module for scene entities and reuses its
graph-node-id and rendering building blocks.
"""

from __future__ import annotations

from enum import Enum
from dataclasses import dataclass, field
from typing_extensions import Any, Dict, List, Tuple


# %% matching a verb against a recorded episode's name
def verb_matches_episode(verb: str, episode_name: str) -> bool:
    """
    Whether an ActionEpisode's name was derived from an action verb.

    Episode step names are built as ``<verb>`` or ``<verb>_<object>`` (see
    :func:`cram_viz.onboard.demo.derive_segments`), so a prefix match identifies every
    episode a verb's knowledge applies to.
    """
    return episode_name == verb or episode_name.startswith(verb + "_")


# %% relations
class KnowledgeRelation(Enum):
    """
    Named relation types used by the static action/entity knowledge ontology.
    """

    HAS_KNOWLEDGE = "has_knowledge"
    CLASSIFIED_AS = "classified_as"
    SPECIALIZED_AS = "specialized_as"
    HAS_ROLE = "has_role"
    RESOLVED_TO = "resolved_to"
    REQUIRES = "requires"
    HAS_PHASE = "has_phase"
    MAINTAINS = "maintains"
    TARGETS = "targets"
    EXPECTS = "expects"
    OBSERVED = "observed"
    HAS_QUALITY = "has_quality"
    AFFORDS = "affords"
    RELATED_BY = "related_by"


# %% domain model — globally shared concepts and ontology classes
@dataclass(unsafe_hash=True)
class Concept:
    """
    A named domain concept a phase maintains or targets (e.g. a constraint or a pose).
    """

    name: str
    """
    Concept identifier, e.g. ``CollisionFreeMotion``.
    """


@dataclass(unsafe_hash=True)
class OntologyClass:
    """
    A named class in the action/entity ontology, e.g. ``TransportAction``.

    Ontology classes are shared graph nodes: the same class referenced from two
    different verbs or entities is drawn once and reused (see :func:`class_node_id`).
    """

    name: str
    """
    Class identifier, e.g. ``ManipulationAction``.
    """


# %% domain model — a verb's ordered phases
@dataclass(unsafe_hash=True)
class ActionPhase:
    """
    One phase of executing an action, e.g. the *reach* phase of a pickup.
    """

    name: str
    """
    Phase identifier, e.g. ``Reach``.
    """

    order: int
    """
    Position of this phase in its action's sequence (1-based).
    """

    maintains: Tuple[Concept, ...] = ()
    """
    Constraints the phase upholds while it executes.
    """

    targets: Tuple[Concept, ...] = ()
    """
    Goals the phase pursues.
    """

    action_verb: str = field(default="", init=False)
    """
    Verb of the action this phase belongs to, auto-stamped by
    :meth:`ActionPhaseSequence.__post_init__`.

    Phase names are reused across verbs (e.g. both ``transport`` and ``place`` have a
    ``Carry`` phase), so this is what disambiguates one EQL-queried phase from another.
    """


@dataclass(unsafe_hash=True)
class ActionPhaseSequence:
    """
    The ordered phase ("Phases") knowledge registered for one action verb.
    """

    name: str
    """
    The action verb this sequence describes, e.g. ``pickup``.
    """

    phases: Tuple[ActionPhase, ...]
    """
    The action's phases, in execution order.
    """

    def __post_init__(self) -> None:
        """
        Stamp this sequence's action verb onto every phase it owns, so each one
        carries its own back-reference without registry authors having to write it
        by hand.
        """
        for phase in self.phases:
            phase.action_verb = self.name

    def matches(self, episode_name: str) -> bool:
        """
        Whether an ActionEpisode's name was derived from this sequence's verb.
        """
        return verb_matches_episode(self.name, episode_name)


# %% domain model — the rest of a verb's knowledge
@dataclass(unsafe_hash=True)
class Role:
    """
    A participant role in an action, e.g. ``agent: pr2``.
    """

    name: str
    """
    Role and its filled-in participant, as one label.
    """

    action_verb: str = field(default="", init=False)
    """
    Verb of the action this role belongs to, auto-stamped by
    :meth:`ActionKnowledge.__post_init__`.
    """


@dataclass(unsafe_hash=True)
class Parameter:
    """
    One value of a sample concrete parameterization of an action, e.g. ``arm: left``.
    """

    name: str
    """
    Parameter and its sample value, as one label.
    """

    action_verb: str = field(default="", init=False)
    """
    Verb of the action this parameter belongs to, auto-stamped by
    :meth:`ActionKnowledge.__post_init__`.
    """


@dataclass(unsafe_hash=True)
class Precondition:
    """
    A fact that should hold before an action can begin.
    """

    name: str
    """
    The required fact, as one label.
    """

    action_verb: str = field(default="", init=False)
    """
    Verb of the action this precondition belongs to, auto-stamped by
    :meth:`ActionKnowledge.__post_init__`.
    """


@dataclass(unsafe_hash=True)
class Goal:
    """
    An expected postcondition of a successfully executed action.
    """

    name: str
    """
    The expected postcondition, as one label.
    """

    action_verb: str = field(default="", init=False)
    """
    Verb of the action this goal belongs to, auto-stamped by
    :meth:`ActionKnowledge.__post_init__`.
    """


@dataclass(unsafe_hash=True)
class ActionKnowledge:
    """
    The static ontology knowledge registered for one action verb, beyond its phases.
    """

    name: str
    """
    The action verb this knowledge describes, e.g. ``transport``.
    """

    classes: Tuple[OntologyClass, ...] = ()
    """
    The verb's ontology classification, broad to specific.
    """

    roles: Tuple[Role, ...] = ()
    """
    Participants in the action and their semantic roles.
    """

    parameters: Tuple[Parameter, ...] = ()
    """
    A sample concrete parameterization of the underspecified action.
    """

    preconditions: Tuple[Precondition, ...] = ()
    """
    Facts that should hold before execution can begin.
    """

    goals: Tuple[Goal, ...] = ()
    """
    Expected postconditions of successful execution.
    """

    def __post_init__(self) -> None:
        """
        Stamp this entry's action verb onto every role/parameter/precondition/goal it
        owns, so each one carries its own back-reference without registry authors
        having to write it by hand.
        """
        for members in (self.roles, self.parameters, self.preconditions, self.goals):
            for member in members:
                member.action_verb = self.name

    def matches(self, episode_name: str) -> bool:
        """
        Whether an ActionEpisode's name was derived from this knowledge's verb.
        """
        return verb_matches_episode(self.name, episode_name)


# %% graph node ids
# Single source of truth for how domain entities map onto graph node ids: both the
# GraphFragment renderers (building the graph) and cram_viz.kb (highlighting query
# results back onto it) key off these, so the two can never drift apart.
def concept_node_id(concept: Concept) -> str:
    """
    Id of a concept node, shared globally across every phase that references it.
    """
    return "concept:" + concept.name


def class_node_id(ontology_class: OntologyClass) -> str:
    """
    Id of an ontology class node, shared globally across every owner that references it.
    """
    return "class:" + ontology_class.name


def category_node_id(category: str, owner_name: str) -> str:
    """
    Id of one knowledge-category root node, e.g. ``roles:transport``.

    :param category: The category key, e.g. ``roles`` or ``properties``.
    :param owner_name: The action verb or entity name the category belongs to.
    """
    return "%s:%s" % (category, owner_name)


def member_node_id(category: str, member_name: str) -> str:
    """
    Id of one knowledge-category member, shared globally across every owner that
    registers a member with this name (e.g. two actions both having an
    ``agent: pr2`` role reference the very same graph node, rather than each owner
    getting its own private copy).
    """
    return "%s:%s" % (category, member_name)


def phase_node_id(phase: ActionPhase) -> str:
    """
    Id of a phase node, shared globally across every verb whose sequence includes it
    (e.g. both ``pickup`` and ``transport`` have a ``Grasp`` phase, and both link
    straight to the very same node — see :func:`member_node_id`).
    """
    return member_node_id("phase", phase.name)


# %% graph rendering
@dataclass(frozen=True)
class KnowledgeGraphNode:
    """
    One node a GraphFragment contributes to the graph payload.
    """

    identifier: str
    """
    Node id, unique across the whole knowledge graph.
    """

    label: str
    """
    Display label.
    """

    lines: Tuple[str, ...]
    """
    Detail-panel body lines.
    """

    group: str = "concept"
    """
    Colour group; ``concept`` is the graph's built-in "Problem / phase / fluent" bucket.
    """


@dataclass(frozen=True)
class KnowledgeGraphEdge:
    """
    One edge a GraphFragment contributes to the graph payload.
    """

    source: str
    """
    Id of the edge's source node.
    """

    target: str
    """
    Id of the edge's target node.
    """

    relation: KnowledgeRelation
    """
    The named relation this edge asserts.
    """

    kind: str = "prop"
    """
    Rendering kind (``prop`` solid / ``type`` dashed), as used throughout the graph.
    """


#: (category key, members, member kind label, relation) — the shape both
#: render_knowledge_fragment() and render_knowledge_anchor_edges() consume for every
#: "flat" knowledge category (i.e. every category except the ontology-class chain)
CategorySpec = Tuple[str, Tuple[Any, ...], str, KnowledgeRelation]


def owned_label(name: str, owner_name: str) -> str:
    """
    Display label for a node owned by exactly one verb/entity.

    Category roots and their leaves reuse the same names across different owners (e.g.
    both ``pickup`` and ``transport`` have a ``Grasp`` phase and an ``agent: pr2``
    role), so without the owner suffix a merged view of every owner at once -- the
    Domain Knowledge tab -- would show what looks like the same node repeated. Globally
    shared nodes (:class:`Concept`, :class:`OntologyClass`) do not use this: they have
    exactly one real owner-independent identity and are meant to be drawn once.
    """
    return "%s (%s)" % (name, owner_name)


def render_knowledge_fragment(
    owner_name: str,
    classes: Tuple[OntologyClass, ...],
    categories: List[CategorySpec],
) -> Tuple[List[KnowledgeGraphNode], List[KnowledgeGraphEdge]]:
    """
    Nodes/edges for an owner's ontology-class chain plus its flat knowledge categories.

    Shared by :class:`ActionKnowledgeGraphFragment` and
    :class:`cram_viz.entity_knowledge.EntityKnowledgeGraphFragment`. The classification
    chain still roots at its own per-owner node (two owners can share every class in
    the chain yet still need their own place to start it from), but a flat category has
    no such root: every member is a node shared globally by name across every owner
    that registers one with the same name (see :func:`member_node_id`) -- two actions
    with the same ``agent: pr2`` role point at the very same node, not a private copy
    each. The edge connecting an owner to a member lives in
    :func:`render_knowledge_anchor_edges`, since it is per-owner even when the member
    node itself is not.

    :param owner_name: The action verb or entity name the classification chain is
        namespaced by.
    :param classes: The owner's ontology classes, broad to specific (shared nodes).
    :param categories: One :data:`CategorySpec` per flat knowledge category.
    """
    nodes: List[KnowledgeGraphNode] = []
    edges: List[KnowledgeGraphEdge] = []
    if classes:
        root_id = category_node_id("classes", owner_name)
        nodes.append(
            KnowledgeGraphNode(
                root_id,
                owned_label("classes", owner_name),
                ("the ontology classification of '%s'" % owner_name,),
            )
        )
        previous_id = root_id
        relation = KnowledgeRelation.CLASSIFIED_AS
        for ontology_class in classes:
            class_id = class_node_id(ontology_class)
            nodes.append(
                KnowledgeGraphNode(class_id, ontology_class.name, ("an OntologyClass",))
            )
            edges.append(KnowledgeGraphEdge(previous_id, class_id, relation))
            previous_id = class_id
            relation = KnowledgeRelation.SPECIALIZED_AS
    for category, members, kind_label, _ in categories:
        for member in members:
            member_id = member_node_id(category, member.name)
            nodes.append(KnowledgeGraphNode(member_id, member.name, (kind_label,)))
    return nodes, edges


def render_knowledge_anchor_edges(
    anchor_node_id: str,
    owner_name: str,
    classes: Tuple[OntologyClass, ...],
    categories: List[CategorySpec],
) -> List[KnowledgeGraphEdge]:
    """
    Edges connecting an anchor straight to its knowledge: ``has_knowledge`` to its own
    classification root, and each flat category's own relation directly to every one
    of its members -- so e.g. both ``pickup`` and ``transport`` link straight to the
    very same shared ``Grasp`` phase node, and selecting either one highlights it.
    """
    edges = []
    if classes:
        edges.append(
            KnowledgeGraphEdge(
                anchor_node_id,
                category_node_id("classes", owner_name),
                KnowledgeRelation.HAS_KNOWLEDGE,
            )
        )
    for category, members, _, relation in categories:
        for member in members:
            edges.append(
                KnowledgeGraphEdge(
                    anchor_node_id, member_node_id(category, member.name), relation
                )
            )
    return edges


class ActionPhaseGraphFragment:
    """
    Renders one :class:`ActionPhaseSequence` into knowledge-graph nodes and edges.

    A phase is a node shared globally by name across every sequence that includes it
    (e.g. both ``pickup`` and ``transport`` have a ``Grasp`` phase, and both link
    straight to the very same node -- see :func:`phase_node_id`), so its own subgraph
    (the phase and the concepts it maintains/targets) is built once via
    :meth:`nodes_and_edges`; :meth:`anchor_edges` then links one matching verb directly
    to every phase in its sequence, so selecting that verb highlights all of them.
    """

    def __init__(self, sequence: ActionPhaseSequence) -> None:
        """
        :param sequence: The registered phase knowledge to render.
        """
        self._sequence = sequence

    def anchor_edges(self, anchor_node_id: str) -> List[KnowledgeGraphEdge]:
        """
        The ``has_phase`` edges from a matching verb straight to every phase in its
        sequence.
        """
        return [
            KnowledgeGraphEdge(
                anchor_node_id, phase_node_id(phase), KnowledgeRelation.HAS_PHASE
            )
            for phase in self._sequence.phases
        ]

    def nodes_and_edges(
        self,
    ) -> Tuple[List[KnowledgeGraphNode], List[KnowledgeGraphEdge]]:
        """
        Every phase's own node and the concepts it maintains or targets.
        """
        nodes: List[KnowledgeGraphNode] = []
        edges: List[KnowledgeGraphEdge] = []
        for phase in sorted(
            self._sequence.phases, key=lambda one_phase: one_phase.order
        ):
            phase_id = phase_node_id(phase)
            nodes.append(KnowledgeGraphNode(phase_id, phase.name, ("an ActionPhase",)))
            for concept, relation in self._phase_concept_relations(phase):
                concept_id = concept_node_id(concept)
                nodes.append(
                    KnowledgeGraphNode(concept_id, concept.name, ("a Concept",))
                )
                edges.append(KnowledgeGraphEdge(phase_id, concept_id, relation))
        return nodes, edges

    @staticmethod
    def _phase_concept_relations(
        phase: ActionPhase,
    ) -> List[Tuple[Concept, KnowledgeRelation]]:
        """
        Every (concept, relation) pair a phase asserts, in a fixed, readable order.
        """
        return [
            (concept, KnowledgeRelation.MAINTAINS) for concept in phase.maintains
        ] + [(concept, KnowledgeRelation.TARGETS) for concept in phase.targets]


class ActionKnowledgeGraphFragment:
    """
    Renders one :class:`ActionKnowledge` into knowledge-graph nodes and edges.

    Mirrors :class:`ActionPhaseGraphFragment`'s split between the shared subgraph
    (:meth:`nodes_and_edges`) and the per-verb anchor edges (:meth:`anchor_edges`): a
    role/parameter/precondition/goal is a node shared globally by name across every
    verb that has one with that name, and a matching verb links directly to each of
    its own, so selecting the verb highlights them too.
    """

    def __init__(self, knowledge: ActionKnowledge) -> None:
        """
        :param knowledge: The registered action knowledge to render.
        """
        self._knowledge = knowledge

    def anchor_edges(self, anchor_node_id: str) -> List[KnowledgeGraphEdge]:
        """
        The edges from a matching verb straight to its classification root and to
        every one of its roles/parameters/preconditions/goals.
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
        The knowledge's own nodes and edges: its ontology-class chain, plus every one
        of its roles/parameters/preconditions/goals as a globally shared node.
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
            ("roles", knowledge.roles, "a Role", KnowledgeRelation.HAS_ROLE),
            (
                "parameters",
                knowledge.parameters,
                "a Parameter",
                KnowledgeRelation.RESOLVED_TO,
            ),
            (
                "preconditions",
                knowledge.preconditions,
                "a Precondition",
                KnowledgeRelation.REQUIRES,
            ),
            ("goals", knowledge.goals, "a Goal", KnowledgeRelation.EXPECTS),
        ]


# %% registry
#: static, hand-authored registry: action verb -> its phase knowledge
ACTION_PHASE_KNOWLEDGE: Dict[str, ActionPhaseSequence] = {
    "pickup": ActionPhaseSequence(
        name="pickup",
        phases=(
            ActionPhase(
                name="Reach",
                order=1,
                maintains=(Concept("CollisionFreeMotion"),),
                targets=(Concept("PreGraspPose"),),
            ),
            ActionPhase(
                name="Grasp",
                order=2,
                maintains=(Concept("CollisionFreeMotion"),),
                targets=(Concept("StableGrasp"),),
            ),
            ActionPhase(
                name="Lift",
                order=3,
                maintains=(Concept("CollisionFreeMotion"), Concept("StableGrasp")),
                targets=(Concept("ClearOfSupport"),),
            ),
        ),
    ),
    "transport": ActionPhaseSequence(
        name="transport",
        phases=(
            ActionPhase(
                name="Reach",
                order=1,
                maintains=(Concept("CollisionFreeMotion"),),
                targets=(Concept("PreGraspPose"),),
            ),
            ActionPhase(
                name="Grasp",
                order=2,
                maintains=(Concept("CollisionFreeMotion"),),
                targets=(Concept("StableGrasp"),),
            ),
            ActionPhase(
                name="Lift",
                order=3,
                maintains=(Concept("CollisionFreeMotion"), Concept("StableGrasp")),
                targets=(Concept("ClearOfSupport"),),
            ),
            ActionPhase(
                name="Carry",
                order=4,
                maintains=(Concept("CollisionFreeMotion"), Concept("StableGrasp")),
                targets=(Concept("PlacePose"),),
            ),
            ActionPhase(
                name="Place",
                order=5,
                maintains=(Concept("CollisionFreeMotion"),),
                targets=(Concept("PlacePose"),),
            ),
        ),
    ),
    "place": ActionPhaseSequence(
        name="place",
        phases=(
            ActionPhase(
                name="Carry",
                order=1,
                maintains=(Concept("CollisionFreeMotion"), Concept("StableGrasp")),
                targets=(Concept("PlacePose"),),
            ),
            ActionPhase(
                name="Place",
                order=2,
                maintains=(Concept("CollisionFreeMotion"),),
                targets=(Concept("PlacePose"),),
            ),
        ),
    ),
}

#: static, hand-authored registry: action verb -> its classification/roles/parameters/
#: preconditions/goals knowledge (its phases are registered separately, above)
ACTION_KNOWLEDGE: Dict[str, ActionKnowledge] = {
    "transport": ActionKnowledge(
        name="transport",
        classes=(
            OntologyClass("Action"),
            OntologyClass("ManipulationAction"),
            OntologyClass("TransportAction"),
        ),
        roles=(
            Role("agent: pr2"),
            Role("patient: milk"),
            Role("destination: place area"),
        ),
        parameters=(
            Parameter("arm: left"),
            Parameter("grasp: power"),
            Parameter("approach: from above"),
        ),
        preconditions=(
            Precondition("object pose is known"),
            Precondition("gripper is empty"),
        ),
        goals=(
            Goal("object is at the destination"),
            Goal("object is no longer held by the robot"),
        ),
    ),
    "pickup": ActionKnowledge(
        name="pickup",
        classes=(
            OntologyClass("Action"),
            OntologyClass("ManipulationAction"),
            OntologyClass("PickUpAction"),
        ),
        roles=(
            Role("agent: pr2"),
            Role("patient: milk"),
            Role("source: kitchen counter"),
        ),
        parameters=(
            Parameter("arm: left"),
            Parameter("grasp: power"),
            Parameter("approach: from above"),
        ),
        preconditions=(
            Precondition("object pose is known"),
            Precondition("object is reachable"),
            Precondition("gripper is empty"),
        ),
        goals=(
            Goal("object is grasped"),
            Goal("object is no longer resting on its support"),
        ),
    ),
    "place": ActionKnowledge(
        name="place",
        classes=(
            OntologyClass("Action"),
            OntologyClass("ManipulationAction"),
            OntologyClass("PlaceAction"),
        ),
        roles=(
            Role("agent: pr2"),
            Role("patient: milk"),
            Role("destination: place area"),
        ),
        parameters=(
            Parameter("arm: left"),
            Parameter("approach: from above"),
            Parameter("release: open gripper"),
        ),
        preconditions=(
            # exactly pickup's own goals -- place picks up right where pickup leaves
            # off in the pickup -> place pipeline that transport runs end to end
            Precondition("object is grasped"),
            Precondition("destination is clear"),
        ),
        goals=(
            Goal("object is at the destination"),
            Goal("object is no longer held by the robot"),
        ),
    ),
}
