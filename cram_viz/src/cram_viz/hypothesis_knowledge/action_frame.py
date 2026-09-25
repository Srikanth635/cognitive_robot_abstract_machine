"""
The Minsky action-frame representation layered onto a hypothesis projection.

Ported verbatim in behaviour from ``llm_reasoner``'s ``knowledge_01_simple`` notebook,
where it lives as notebook-local cells rather than package code. It stays here because
``llm_reasoner`` is treated as read-only, and because the Hypothesis Graph tab needs the
overlay to run against a freshly generated structure, not only against a saved export.

The frame is display-only: its nodes carry no ``KnowledgeEntity`` source and therefore
never enter an EQL domain. Entities bind to the projected structure through
``RoleSlot.binds_as`` -- the role's *concept*, not its free-text label -- so the same
template attaches to any structure built from the same Flanagan role classes.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from llm_reasoner.knowledge.hypothesis.graph import (
    HypothesisGraph,
    HypothesisGraphNode,
    HypothesisGraphNodeKind,
    HypothesisGraphRelation,
)
from llm_reasoner.knowledge.microworlds.flanagan.core import RoleSlot


def role_type_name(role: RoleSlot) -> str | None:
    """
    The local name of a role slot's bound concept (e.g. ``'SourceContainer'``), or
    ``None`` if the slot names no concept.

    Derived from ``binds_as.iri`` rather than the slot's own free-text ``label``,
    which is optional and, when set, often describes the microworld's illustrative
    filler (e.g. ``'source container: cereal box'``) rather than the role itself.
    """
    if role.binds_as is None:
        return None
    return role.binds_as.iri.rsplit("#", 1)[-1].rsplit("/", 1)[-1]


# %% Frame vocabulary


class MinskyActionSlot(StrEnum):
    """Direct slots of the action frame."""

    ACTION = "Action"
    ENTITIES = "Entities"
    EXPECTED_OUTCOMES = "Expected Outcomes"


class FrameRoleKind(StrEnum):
    """Generic participant roles a Minsky action frame's entities can fill."""

    AGENT = "Agent"
    SOURCE_CONTAINER = "SourceContainer"
    DESTINATION_CONTAINER = "DestinationContainer"
    SUBSTANCE = "Substance"


#: type_name of a role's filler leaf when the instruction names it
ROLE_FILLER_TYPE_NAME = "RoleFiller"

#: type_name of a role's filler leaf when the instruction does not name it; shared
#: with :mod:`cram_viz.kb`, which reads it back to flag the node for the frontend
UNRESOLVED_ROLE_FILLER_TYPE_NAME = "UnresolvedRoleFiller"


class EntityPropertyKind(StrEnum):
    """Property groups used to describe a stereotypical entity."""

    AFFORDANCES = "Affordances"
    DISPOSITIONS = "Dispositions"
    VISUAL_PROPERTIES = "Visual Properties"
    GEOMETRIC_PROPERTIES = "Geometric Properties"


class MinskyFrameRelation(StrEnum):
    """Relationships introduced only by the frame visualization."""

    INTERPRETED_AS = "interpreted_as"
    HAS_SLOT = "has_slot"
    HAS_VALUE = "has_value"
    HAS_ENTITY = "has_entity"
    HAS_NAME = "has_name"
    DESCRIBED_BY = "described_by"
    HAS_PROPERTY = "has_property"
    PARTICIPATES_AS = "participates_as"


@dataclass(frozen=True, eq=False)
class EntityPropertyFrame:
    """A group of stereotypical properties for an entity.

    ``eq=False``: two property groups with the same kind and values are still
    distinct occurrences (one per entity), the same reasoning ``KnowledgeEntity``
    is built on. Value equality here would let an EQL ``in_()`` membership check
    match a group that merely looks alike, against a completely different entity.
    """

    kind: EntityPropertyKind
    """Kind of entity property represented by the group."""

    values: tuple[str, ...]
    """Prototype property descriptions in the group."""


@dataclass(frozen=True, eq=False)
class EntityFrame:
    """A stereotypical entity involved in an action.

    ``eq=False``, for the same reason as :class:`EntityPropertyFrame`.
    """

    role: FrameRoleKind
    """Generic participant role this entity fills; shown as the frame node's name."""

    concrete_name: str
    """This microworld's illustrative filler for the role (e.g. ``'Cereal Box'``),
    shown as a leaf under the role node only when the instruction names it."""

    property_frames: tuple[EntityPropertyFrame, ...]
    """Affordance, disposition, visual, and geometric descriptions."""

    role_type_names: tuple[str, ...]
    """Flanagan role types that this entity typically fills."""


@dataclass(frozen=True, eq=False)
class MinskyActionFrameTemplate:
    """The object-centered content of a stereotypical action frame.

    ``eq=False``, for the same reason as :class:`EntityPropertyFrame`.
    """

    label: str
    """Name shown on the action-frame node."""

    action: str
    """Canonical action represented by the frame."""

    entities: tuple[EntityFrame, ...]
    """Entities generally involved in the action."""

    expected_outcomes: tuple[str, ...]
    """States normally expected after successful performance."""

    def entity_for_role(self, role: RoleSlot) -> EntityFrame | None:
        """Return the entity frame corresponding to a Flanagan role slot."""
        role_type = role_type_name(role)
        if role_type is None:
            return None
        for entity in self.entities:
            if role_type in entity.role_type_names:
                return entity
        return None

    @classmethod
    def cereal_pouring(cls) -> "MinskyActionFrameTemplate":
        """Return an object-centered cereal-pouring stereotype."""
        return cls(
            label="FrameModel",
            action="Pouring",
            entities=(
                EntityFrame(
                    role=FrameRoleKind.AGENT,
                    concrete_name="Gripper",
                    role_type_names=("GraspingEffector", "Actor"),
                    property_frames=(
                        EntityPropertyFrame(
                            EntityPropertyKind.AFFORDANCES,
                            (
                                "Reach and grasp the cereal box",
                                "Lift and orient the cereal box",
                                "Regulate cereal release",
                            ),
                        ),
                        EntityPropertyFrame(
                            EntityPropertyKind.DISPOSITIONS,
                            (
                                "Moves under intentional control",
                                "Adjusts grip to the changing box load",
                            ),
                        ),
                        EntityPropertyFrame(
                            EntityPropertyKind.VISUAL_PROPERTIES,
                            (
                                "Effector pose",
                                "Contact configuration",
                                "Visibility of the grasp",
                            ),
                        ),
                        EntityPropertyFrame(
                            EntityPropertyKind.GEOMETRIC_PROPERTIES,
                            (
                                "Reachable workspace",
                                "Grasp span",
                                "Approach direction",
                            ),
                        ),
                    ),
                ),
                EntityFrame(
                    role=FrameRoleKind.SOURCE_CONTAINER,
                    concrete_name="Cereal Box",
                    role_type_names=("Manipulandum", "SourceContainer"),
                    property_frames=(
                        EntityPropertyFrame(
                            EntityPropertyKind.AFFORDANCES,
                            (
                                "Graspable",
                                "Liftable",
                                "Tiltable",
                                "Supports controlled cereal pouring",
                            ),
                        ),
                        EntityPropertyFrame(
                            EntityPropertyKind.DISPOSITIONS,
                            (
                                "Retains cereal while upright or closed",
                                "Releases cereal through its opening when tilted",
                                "May scatter cereal when misaligned",
                            ),
                        ),
                        EntityPropertyFrame(
                            EntityPropertyKind.VISUAL_PROPERTIES,
                            (
                                "Visible package opening or top flaps",
                                "Printed rectangular package surfaces",
                                "Observable grasp region on the box panels",
                            ),
                        ),
                        EntityPropertyFrame(
                            EntityPropertyKind.GEOMETRIC_PROPERTIES,
                            (
                                "Cuboid package dimensions",
                                "Opening size",
                                "Interior capacity",
                                "Center of mass",
                            ),
                        ),
                    ),
                ),
                EntityFrame(
                    role=FrameRoleKind.SUBSTANCE,
                    concrete_name="Cereal",
                    role_type_names=("PouredSubstance",),
                    property_frames=(
                        EntityPropertyFrame(
                            EntityPropertyKind.AFFORDANCES,
                            (
                                "Transferable between containers",
                                "Pourable through the box opening",
                            ),
                        ),
                        EntityPropertyFrame(
                            EntityPropertyKind.DISPOSITIONS,
                            (
                                "Flows as a granular collective under gravity",
                                "Forms a pile in the bowl",
                                "Scatters when spilled",
                            ),
                        ),
                        EntityPropertyFrame(
                            EntityPropertyKind.VISUAL_PROPERTIES,
                            (
                                "Discrete pieces are visible",
                                "Piece color and shape are observable",
                                "A cereal stream is visible during transfer",
                            ),
                        ),
                        EntityPropertyFrame(
                            EntityPropertyKind.GEOMETRIC_PROPERTIES,
                            (
                                "Bulk volume",
                                "Piece size and shape",
                                "Pile footprint",
                                "No fixed bulk shape",
                            ),
                        ),
                    ),
                ),
                EntityFrame(
                    role=FrameRoleKind.DESTINATION_CONTAINER,
                    concrete_name="Bowl",
                    role_type_names=("TargetReceptacle",),
                    property_frames=(
                        EntityPropertyFrame(
                            EntityPropertyKind.AFFORDANCES,
                            (
                                "Receives cereal",
                                "Contains transferred cereal",
                                "Supports portion observation",
                            ),
                        ),
                        EntityPropertyFrame(
                            EntityPropertyKind.DISPOSITIONS,
                            (
                                "Retains cereal when stable",
                                "Fills as cereal enters",
                                "May overflow when overfilled",
                            ),
                        ),
                        EntityPropertyFrame(
                            EntityPropertyKind.VISUAL_PROPERTIES,
                            (
                                "Visible open rim",
                                "Observable concave interior",
                                "Observable cereal level",
                            ),
                        ),
                        EntityPropertyFrame(
                            EntityPropertyKind.GEOMETRIC_PROPERTIES,
                            (
                                "Opening diameter",
                                "Concave interior",
                                "Interior capacity",
                                "Pose and orientation",
                            ),
                        ),
                    ),
                ),
            ),
            expected_outcomes=(
                "Cereal is contained by the bowl",
                "The intended cereal portion has been transferred",
                "Cereal flow has stopped",
                "No unintended cereal spill has occurred",
            ),
        )


# %% Frame overlay


@dataclass(frozen=True)
class MinskyActionFrameOverlay:
    """Add an independent object-centered frame to a graph projection."""

    template: MinskyActionFrameTemplate
    """Prototype content rendered by the overlay."""

    def apply(self, projection: HypothesisGraph) -> HypothesisGraph:
        """Add the frame branch and return the same projection."""
        frame_identifier = f"minsky-frame:{projection.instruction.identifier}"
        if any(node.identifier == frame_identifier for node in projection.nodes):
            return projection

        frame_index = self._add_node(
            projection,
            identifier=frame_identifier,
            label=self.template.label,
            type_name="MinskyActionFrame",
            kind=HypothesisGraphNodeKind.ROOT,
            source=self.template,
        )
        self._connect(
            projection,
            projection.instruction_index,
            MinskyFrameRelation.INTERPRETED_AS,
            frame_index,
        )
        self._add_action_slot(projection, frame_index)
        self._add_entity_slot(projection, frame_index)
        self._add_outcome_slot(projection, frame_index)
        return projection

    def _add_action_slot(
        self,
        projection: HypothesisGraph,
        frame_index: int,
    ) -> None:
        """Add the canonical action as a direct frame slot."""
        slot_index = self._add_slot(
            projection,
            frame_index,
            MinskyActionSlot.ACTION,
            member_count=1,
        )
        value_index = self._add_node(
            projection,
            identifier=f"minsky-action:{projection.instruction.identifier}",
            label=self.template.action,
            type_name="ActionFrameValue",
            kind=HypothesisGraphNodeKind.FRAGMENT,
            source=self.template,
        )
        self._connect(
            projection,
            slot_index,
            MinskyFrameRelation.HAS_VALUE,
            value_index,
        )

    def _add_entity_slot(
        self,
        projection: HypothesisGraph,
        frame_index: int,
    ) -> None:
        """Add entity frames and their object-centered properties."""
        slot_index = self._add_slot(
            projection,
            frame_index,
            MinskyActionSlot.ENTITIES,
            member_count=len(self.template.entities),
        )
        for entity_position, entity in enumerate(self.template.entities):
            entity_index = self._add_node(
                projection,
                identifier=(
                    f"minsky-entity:{projection.instruction.identifier}:"
                    f"{entity_position}"
                ),
                label=entity.role.value,
                type_name="EntityFrame",
                kind=HypothesisGraphNodeKind.FRAGMENT,
                source=entity,
            )
            self._connect(
                projection,
                slot_index,
                MinskyFrameRelation.HAS_ENTITY,
                entity_index,
            )
            for role_index in self._role_node_indices(projection, entity):
                self._connect(
                    projection,
                    entity_index,
                    MinskyFrameRelation.PARTICIPATES_AS,
                    role_index,
                )
            self._add_role_filler(projection, entity_index, entity_position, entity)
            self._add_entity_properties(
                projection,
                entity_index,
                entity_position,
                entity,
            )

    def _add_role_filler(
        self,
        projection: HypothesisGraph,
        entity_index: int,
        entity_position: int,
        entity: EntityFrame,
    ) -> None:
        """
        Add the role's concrete filler as a leaf, or an unresolved placeholder when
        the instruction never names it.
        """
        grounded = self._is_named_in_instruction(projection, entity.concrete_name)
        filler_index = self._add_node(
            projection,
            identifier=f"minsky-name:{projection.instruction.identifier}:{entity_position}",
            label=entity.concrete_name if grounded else "?",
            type_name=(
                ROLE_FILLER_TYPE_NAME if grounded else UNRESOLVED_ROLE_FILLER_TYPE_NAME
            ),
            kind=HypothesisGraphNodeKind.FRAGMENT,
            source=entity,
        )
        self._connect(
            projection,
            entity_index,
            MinskyFrameRelation.HAS_NAME,
            filler_index,
        )

    @staticmethod
    def _is_named_in_instruction(
        projection: HypothesisGraph, concrete_name: str
    ) -> bool:
        """Whether the instruction text mentions a role's illustrative filler."""
        return concrete_name.lower() in projection.instruction.label.lower()

    def _role_node_indices(
        self,
        projection: HypothesisGraph,
        entity: EntityFrame,
    ) -> tuple[int, ...]:
        """Return projected Flanagan role slots corresponding to an entity."""
        matched_indices: list[int] = []
        for node_index in projection.graph.node_indices():
            node_source = projection.graph[node_index].source
            if not isinstance(node_source, RoleSlot):
                continue
            if self.template.entity_for_role(node_source) is entity:
                matched_indices.append(node_index)
        return tuple(matched_indices)

    def _add_entity_properties(
        self,
        projection: HypothesisGraph,
        entity_index: int,
        entity_position: int,
        entity: EntityFrame,
    ) -> None:
        """Add the property groups and values of one entity frame."""
        for property_position, property_frame in enumerate(entity.property_frames):
            property_index = self._add_node(
                projection,
                identifier=(
                    f"minsky-property-group:{projection.instruction.identifier}:"
                    f"{entity_position}:{property_position}"
                ),
                label=(
                    f"{property_frame.kind.value} " f"({len(property_frame.values)})"
                ),
                type_name="EntityPropertyGroup",
                kind=HypothesisGraphNodeKind.GROUP,
                source=property_frame,
            )
            self._connect(
                projection,
                entity_index,
                MinskyFrameRelation.DESCRIBED_BY,
                property_index,
            )
            for value_position, value in enumerate(property_frame.values):
                value_index = self._add_node(
                    projection,
                    identifier=(
                        f"minsky-property:{projection.instruction.identifier}:"
                        f"{entity_position}:{property_position}:{value_position}"
                    ),
                    label=value,
                    type_name="EntityProperty",
                    kind=HypothesisGraphNodeKind.FRAGMENT,
                    source=property_frame,
                )
                self._connect(
                    projection,
                    property_index,
                    MinskyFrameRelation.HAS_PROPERTY,
                    value_index,
                )

    def _add_outcome_slot(
        self,
        projection: HypothesisGraph,
        frame_index: int,
    ) -> None:
        """Add expected successful outcomes as a direct frame slot."""
        slot_index = self._add_slot(
            projection,
            frame_index,
            MinskyActionSlot.EXPECTED_OUTCOMES,
            member_count=len(self.template.expected_outcomes),
        )
        for outcome_position, outcome in enumerate(self.template.expected_outcomes):
            outcome_index = self._add_node(
                projection,
                identifier=(
                    f"minsky-outcome:{projection.instruction.identifier}:"
                    f"{outcome_position}"
                ),
                label=outcome,
                type_name="ExpectedOutcome",
                kind=HypothesisGraphNodeKind.FRAGMENT,
                source=self.template,
            )
            self._connect(
                projection,
                slot_index,
                MinskyFrameRelation.HAS_VALUE,
                outcome_index,
            )

    def _add_slot(
        self,
        projection: HypothesisGraph,
        frame_index: int,
        slot: MinskyActionSlot,
        member_count: int,
    ) -> int:
        """Add one direct slot beneath the action frame."""
        slot_index = self._add_node(
            projection,
            identifier=(
                f"minsky-slot:{projection.instruction.identifier}:"
                f"{slot.name.lower()}"
            ),
            label=f"{slot.value} ({member_count})",
            type_name="MinskyActionSlot",
            kind=HypothesisGraphNodeKind.GROUP,
            source=slot,
        )
        self._connect(
            projection,
            frame_index,
            MinskyFrameRelation.HAS_SLOT,
            slot_index,
        )
        return slot_index

    @staticmethod
    def _add_node(
        projection: HypothesisGraph,
        identifier: str,
        label: str,
        type_name: str,
        kind: HypothesisGraphNodeKind,
        source: object,
    ) -> int:
        """Add one synthetic frame node."""
        return projection.graph.add_node(
            HypothesisGraphNode(
                identifier=identifier,
                label=label,
                type_name=type_name,
                kind=kind,
                source=source,
            )
        )

    @staticmethod
    def _connect(
        projection: HypothesisGraph,
        source_index: int,
        relation: MinskyFrameRelation,
        target_index: int,
    ) -> None:
        """Add one visualization-only frame relationship."""
        projection.graph.add_edge(
            source_index,
            target_index,
            HypothesisGraphRelation(
                source=projection.graph[source_index],
                predicate=relation.value,
                target=projection.graph[target_index],
            ),
        )
