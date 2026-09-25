"""
The image-schema profile layered onto a hypothesis projection.

Ported in behaviour from ``llm_reasoner``'s ``knowledge_01_simple`` notebook, for the
same reasons as :mod:`cram_viz.hypothesis_knowledge.action_frame`.

.. note::
   Apply this *after* :class:`~cram_viz.hypothesis_knowledge.action_frame.MinskyActionFrameOverlay`.
   Entity-valued spatial primitives are grounded by reusing the action frame's own
   entity nodes, so an entity filler with no frame entity raises
   :class:`ImageSchemaGroundingError` rather than silently drawing a second copy.
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

from cram_viz.hypothesis_knowledge.action_frame import FrameRoleKind

# %% Image-schema vocabulary


class ImageSchemaKind(StrEnum):
    """Reusable image-schema types represented by this profile."""

    CONTAINMENT = "CONTAINMENT"
    SOURCE_PATH_GOAL = "SOURCE-PATH-GOAL"


class SpatialPrimitiveKind(StrEnum):
    """Semantic roles internal to the represented image schemas."""

    CONTAINER = "CONTAINER"
    CONTAINED = "CONTAINED"
    INTERIOR = "INTERIOR"
    BOUNDARY = "BOUNDARY"
    EXTERIOR = "EXTERIOR"
    TRAJECTOR = "TRAJECTOR"
    SOURCE = "SOURCE"
    PATH = "PATH"
    GOAL = "GOAL"
    DIRECTION = "DIRECTION"


class ImageSchemaFillerKind(StrEnum):
    """Ontological category of a spatial-primitive filler."""

    ENTITY = "EntityFrame"
    SPATIAL_REGION = "SpatialRegion"
    SPATIAL_PATH = "SpatialPath"
    SPATIAL_DIRECTION = "SpatialDirection"


class ImageSchemaRelation(StrEnum):
    """Relationships introduced by the image-schema visualization."""

    HAS_PROFILE = "has_image_schema_profile"
    HAS_SCHEMA_INSTANCE = "has_schema_instance"
    INSTANCE_OF = "instance_of"
    NECESSARY_SPATIAL_PRIMITIVE = "necessary_spatial_primitive"
    FILLED_BY = "filled_by"
    PRECEDES = "precedes"


class ImageSchemaGroundingError(ValueError):
    """Raised when an entity-valued primitive has no frame entity filler."""


@dataclass(frozen=True, eq=False)
class SpatialPrimitiveBinding:
    """Bind one spatial-primitive role to a participant or spatial value.

    ``eq=False``: two bindings can share every field (e.g. both the Initial
    Containment and Result Containment instances bind CONTAINED to Substance)
    while belonging to different schema instances. Value equality would let an
    EQL ``in_()`` membership check treat those as interchangeable -- the same
    reasoning ``KnowledgeEntity`` is built on.
    """

    primitive: SpatialPrimitiveKind
    """Spatial-primitive role played within an image schema."""

    filler_label: str
    """For a spatial value, the label drawn on its own node. For an entity, the
    :class:`~cram_viz.hypothesis_knowledge.action_frame.FrameRoleKind` value of the
    action-frame role sharing this filler -- never drawn directly, since the shared
    node already carries that text as its own label."""

    filler_kind: ImageSchemaFillerKind
    """Ontological category assigned to the filler."""


@dataclass(frozen=True, eq=False)
class ImageSchemaInstance:
    """A situation-specific occurrence of a reusable image schema.

    ``eq=False``, for the same reason as :class:`SpatialPrimitiveBinding`.
    """

    label: str
    """Situation-specific name shown in the graph."""

    schema_kind: ImageSchemaKind
    """Reusable image-schema type instantiated by this occurrence."""

    primitive_bindings: tuple[SpatialPrimitiveBinding, ...]
    """Required spatial-primitive roles and their fillers."""


@dataclass(frozen=True, eq=False)
class ImageSchemaProfile:
    """The ordered image-schematic characterization of an instruction.

    ``eq=False``, for the same reason as :class:`SpatialPrimitiveBinding`.
    """

    label: str
    """Name shown on the profile node."""

    schema_instances: tuple[ImageSchemaInstance, ...]
    """Image-schema occurrences activated by the instruction."""

    @classmethod
    def cereal_pouring(cls) -> ImageSchemaProfile:
        """Return the image-schema profile for pouring cereal into a bowl."""
        return cls(
            label="ImageSchemaModel",
            schema_instances=(
                ImageSchemaInstance(
                    label="Initial Containment",
                    schema_kind=ImageSchemaKind.CONTAINMENT,
                    primitive_bindings=(
                        SpatialPrimitiveBinding(
                            SpatialPrimitiveKind.CONTAINER,
                            FrameRoleKind.SOURCE_CONTAINER.value,
                            ImageSchemaFillerKind.ENTITY,
                        ),
                        SpatialPrimitiveBinding(
                            SpatialPrimitiveKind.CONTAINED,
                            FrameRoleKind.SUBSTANCE.value,
                            ImageSchemaFillerKind.ENTITY,
                        ),
                        SpatialPrimitiveBinding(
                            SpatialPrimitiveKind.INTERIOR,
                            "SourceContainer Interior",
                            ImageSchemaFillerKind.SPATIAL_REGION,
                        ),
                        SpatialPrimitiveBinding(
                            SpatialPrimitiveKind.BOUNDARY,
                            "SourceContainer Boundary",
                            ImageSchemaFillerKind.SPATIAL_REGION,
                        ),
                        SpatialPrimitiveBinding(
                            SpatialPrimitiveKind.EXTERIOR,
                            "Outside SourceContainer",
                            ImageSchemaFillerKind.SPATIAL_REGION,
                        ),
                    ),
                ),
                ImageSchemaInstance(
                    label="Substance Transfer",
                    schema_kind=ImageSchemaKind.SOURCE_PATH_GOAL,
                    primitive_bindings=(
                        SpatialPrimitiveBinding(
                            SpatialPrimitiveKind.TRAJECTOR,
                            FrameRoleKind.SUBSTANCE.value,
                            ImageSchemaFillerKind.ENTITY,
                        ),
                        SpatialPrimitiveBinding(
                            SpatialPrimitiveKind.SOURCE,
                            FrameRoleKind.SOURCE_CONTAINER.value,
                            ImageSchemaFillerKind.ENTITY,
                        ),
                        SpatialPrimitiveBinding(
                            SpatialPrimitiveKind.PATH,
                            "Downward Transfer Path",
                            ImageSchemaFillerKind.SPATIAL_PATH,
                        ),
                        SpatialPrimitiveBinding(
                            SpatialPrimitiveKind.GOAL,
                            FrameRoleKind.DESTINATION_CONTAINER.value,
                            ImageSchemaFillerKind.ENTITY,
                        ),
                        SpatialPrimitiveBinding(
                            SpatialPrimitiveKind.DIRECTION,
                            "Downward",
                            ImageSchemaFillerKind.SPATIAL_DIRECTION,
                        ),
                    ),
                ),
                ImageSchemaInstance(
                    label="Result Containment",
                    schema_kind=ImageSchemaKind.CONTAINMENT,
                    primitive_bindings=(
                        SpatialPrimitiveBinding(
                            SpatialPrimitiveKind.CONTAINER,
                            FrameRoleKind.DESTINATION_CONTAINER.value,
                            ImageSchemaFillerKind.ENTITY,
                        ),
                        SpatialPrimitiveBinding(
                            SpatialPrimitiveKind.CONTAINED,
                            FrameRoleKind.SUBSTANCE.value,
                            ImageSchemaFillerKind.ENTITY,
                        ),
                        SpatialPrimitiveBinding(
                            SpatialPrimitiveKind.INTERIOR,
                            "DestinationContainer Interior",
                            ImageSchemaFillerKind.SPATIAL_REGION,
                        ),
                        SpatialPrimitiveBinding(
                            SpatialPrimitiveKind.BOUNDARY,
                            "DestinationContainer Boundary",
                            ImageSchemaFillerKind.SPATIAL_REGION,
                        ),
                        SpatialPrimitiveBinding(
                            SpatialPrimitiveKind.EXTERIOR,
                            "Outside DestinationContainer",
                            ImageSchemaFillerKind.SPATIAL_REGION,
                        ),
                    ),
                ),
            ),
        )


# %% Image-schema graph projection


@dataclass(frozen=True)
class ImageSchemaProfileOverlay:
    """Add an image-schema profile without changing the hypothesis record."""

    profile: ImageSchemaProfile
    """Situation-specific image-schema profile to project."""

    def apply(self, projection: HypothesisGraph) -> HypothesisGraph:
        """Add the image-schema branch and return the same projection."""
        profile_identifier = f"image-schema-profile:{projection.instruction.identifier}"
        if any(node.identifier == profile_identifier for node in projection.nodes):
            return projection

        profile_index = self._add_node(
            projection,
            identifier=profile_identifier,
            label=f"{self.profile.label} ({len(self.profile.schema_instances)})",
            type_name="ImageSchemaProfile",
            kind=HypothesisGraphNodeKind.ROOT,
            source=self.profile,
        )
        self._connect(
            projection,
            projection.instruction_index,
            ImageSchemaRelation.HAS_PROFILE,
            profile_index,
        )

        entity_indices = self._entity_indices(projection)
        filler_indices = dict(entity_indices)
        schema_type_indices: dict[ImageSchemaKind, int] = {}
        instance_indices: list[int] = []
        for instance_position, schema_instance in enumerate(
            self.profile.schema_instances
        ):
            instance_index = self._add_schema_instance(
                projection,
                profile_index,
                instance_position,
                schema_instance,
            )
            instance_indices.append(instance_index)
            schema_type_index = self._schema_type_index(
                projection,
                schema_instance.schema_kind,
                schema_type_indices,
            )
            self._connect(
                projection,
                instance_index,
                ImageSchemaRelation.INSTANCE_OF,
                schema_type_index,
            )
            self._add_spatial_primitives(
                projection,
                instance_index,
                instance_position,
                schema_instance,
                filler_indices,
            )

        for earlier_index, later_index in zip(
            instance_indices,
            instance_indices[1:],
            strict=False,
        ):
            self._connect(
                projection,
                earlier_index,
                ImageSchemaRelation.PRECEDES,
                later_index,
            )
        return projection

    def _add_schema_instance(
        self,
        projection: HypothesisGraph,
        profile_index: int,
        instance_position: int,
        schema_instance: ImageSchemaInstance,
    ) -> int:
        """Add one situation-specific image-schema occurrence."""
        instance_index = self._add_node(
            projection,
            identifier=(
                f"image-schema-instance:{projection.instruction.identifier}:"
                f"{instance_position}"
            ),
            label=schema_instance.label,
            type_name="ImageSchemaInstance",
            kind=HypothesisGraphNodeKind.GROUP,
            source=schema_instance,
        )
        self._connect(
            projection,
            profile_index,
            ImageSchemaRelation.HAS_SCHEMA_INSTANCE,
            instance_index,
        )
        return instance_index

    def _schema_type_index(
        self,
        projection: HypothesisGraph,
        schema_kind: ImageSchemaKind,
        schema_type_indices: dict[ImageSchemaKind, int],
    ) -> int:
        """Return the shared node for a reusable image-schema type."""
        if schema_kind in schema_type_indices:
            return schema_type_indices[schema_kind]

        schema_type_index = self._add_node(
            projection,
            identifier=f"image-schema-type:{schema_kind.name.lower()}",
            label=schema_kind.value,
            type_name="ImageSchema",
            kind=HypothesisGraphNodeKind.FRAGMENT,
            source=schema_kind,
        )
        schema_type_indices[schema_kind] = schema_type_index
        return schema_type_index

    def _add_spatial_primitives(
        self,
        projection: HypothesisGraph,
        instance_index: int,
        instance_position: int,
        schema_instance: ImageSchemaInstance,
        filler_indices: dict[str, int],
    ) -> None:
        """Add semantic-role nodes and bind them to shared fillers."""
        for primitive_position, binding in enumerate(
            schema_instance.primitive_bindings
        ):
            primitive_index = self._add_node(
                projection,
                identifier=(
                    f"spatial-primitive:{projection.instruction.identifier}:"
                    f"{instance_position}:{primitive_position}"
                ),
                label=binding.primitive.value,
                type_name="SpatialPrimitive",
                kind=HypothesisGraphNodeKind.GROUP,
                source=binding,
            )
            self._connect(
                projection,
                instance_index,
                ImageSchemaRelation.NECESSARY_SPATIAL_PRIMITIVE,
                primitive_index,
            )
            filler_index = self._filler_index(
                projection,
                binding,
                filler_indices,
            )
            self._connect(
                projection,
                primitive_index,
                ImageSchemaRelation.FILLED_BY,
                filler_index,
            )

    def _filler_index(
        self,
        projection: HypothesisGraph,
        binding: SpatialPrimitiveBinding,
        filler_indices: dict[str, int],
    ) -> int:
        """Return a shared entity or synthetic spatial-value node."""
        if binding.filler_label in filler_indices:
            return filler_indices[binding.filler_label]
        if binding.filler_kind is ImageSchemaFillerKind.ENTITY:
            raise ImageSchemaGroundingError(
                f"No frame entity is available for {binding.filler_label!r}."
            )

        filler_index = self._add_node(
            projection,
            identifier=(
                f"image-schema-filler:{projection.instruction.identifier}:"
                f"{binding.filler_kind.name.lower()}:"
                f"{binding.filler_label.lower().replace(' ', '-')}"
            ),
            label=binding.filler_label,
            type_name=binding.filler_kind.value,
            kind=HypothesisGraphNodeKind.FRAGMENT,
            source=binding,
        )
        filler_indices[binding.filler_label] = filler_index
        return filler_index

    @staticmethod
    def _entity_indices(projection: HypothesisGraph) -> dict[str, int]:
        """
        Return existing action-frame entities indexed by their generic role name.

        An entity node's label *is* its role's name (see
        :class:`~cram_viz.hypothesis_knowledge.action_frame.FrameRoleKind`), which is
        also what this profile's own entity-kind ``filler_label``\\ s use, so the two
        representations share one node per role rather than each drawing its own copy.
        """
        return {
            projection.graph[node_index].label: node_index
            for node_index in projection.graph.node_indices()
            if projection.graph[node_index].type_name == "EntityFrame"
        }

    @staticmethod
    def _add_node(
        projection: HypothesisGraph,
        identifier: str,
        label: str,
        type_name: str,
        kind: HypothesisGraphNodeKind,
        source: object,
    ) -> int:
        """Add one image-schema projection node."""
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
        relation: ImageSchemaRelation,
        target_index: int,
    ) -> None:
        """Add one image-schema relationship."""
        projection.graph.add_edge(
            source_index,
            target_index,
            HypothesisGraphRelation(
                source=projection.graph[source_index],
                predicate=relation.value,
                target=projection.graph[target_index],
            ),
        )
