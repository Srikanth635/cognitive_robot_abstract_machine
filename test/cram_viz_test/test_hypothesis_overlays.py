"""
Tests for the Minsky action-frame and image-schema overlays.

These are the two informal representations layered on top of a projected Flanagan
structure. They must bind to whatever structure they are given -- hand-authored or
model-generated -- which is what makes the Hypothesis Graph tab work for a freshly
generated hypothesis rather than only for the checked-in sample.
"""

from __future__ import annotations

import pytest
from llm_reasoner.knowledge.microworlds.flanagan.core import RoleSlot

from cram_viz.hypothesis_knowledge import (
    DEFAULT_INSTRUCTION,
    build_hypothesis_record,
    build_projection,
)
from cram_viz.hypothesis_knowledge.action_frame import (
    ROLE_FILLER_TYPE_NAME,
    UNRESOLVED_ROLE_FILLER_TYPE_NAME,
    FrameRoleKind,
    MinskyActionFrameOverlay,
    MinskyActionFrameTemplate,
    MinskyFrameRelation,
)
from cram_viz.hypothesis_knowledge.image_schema import (
    ImageSchemaProfile,
    ImageSchemaProfileOverlay,
    ImageSchemaRelation,
)
from cram_viz.hypothesis_knowledge.pouring_structure import (
    build_pouring_phase_structure,
)


def bare_projection_for(instruction=DEFAULT_INSTRUCTION):
    """A projection of a freshly built structure, with no overlay applied yet."""
    structure = build_pouring_phase_structure()
    record = build_hypothesis_record(structure, instruction=instruction)
    return build_projection(record, instruction=instruction, apply_overlays=False)


@pytest.fixture
def bare_projection():
    return bare_projection_for()


def name_leaf(projection, role):
    """The single ``HAS_NAME`` child hanging off the entity node for one role."""
    role_node_identifier = next(
        node.identifier
        for node in projection.nodes
        if node.type_name == "EntityFrame" and node.source.role is role
    )
    (leaf,) = [
        relation.target
        for relation in projection.relations
        if relation.source.identifier == role_node_identifier
        and relation.predicate == MinskyFrameRelation.HAS_NAME.value
    ]
    return leaf


def predicates_from(projection, source_identifier):
    """Every predicate leaving the node with the given identifier."""
    return {
        relation.predicate
        for relation in projection.relations
        if relation.source.identifier == source_identifier
    }


def node_labels_by_type(projection, type_name):
    """Labels of every projected node of one type."""
    return {node.label for node in projection.nodes if node.type_name == type_name}


# %% Minsky action frame
class TestMinskyActionFrameOverlay:
    def test_frame_root_hangs_off_the_instruction(self, bare_projection):
        MinskyActionFrameOverlay(
            template=MinskyActionFrameTemplate.cereal_pouring()
        ).apply(bare_projection)

        assert MinskyFrameRelation.INTERPRETED_AS.value in predicates_from(
            bare_projection, bare_projection.instruction.identifier
        )

    def test_entity_frames_are_projected_with_generic_role_names(self, bare_projection):
        template = MinskyActionFrameTemplate.cereal_pouring()
        MinskyActionFrameOverlay(template=template).apply(bare_projection)

        assert node_labels_by_type(bare_projection, "EntityFrame") == {
            entity.role.value for entity in template.entities
        }

    def test_root_label_is_generic(self):
        assert MinskyActionFrameTemplate.cereal_pouring().label == "FrameModel"

    def test_named_filler_becomes_a_grounded_leaf(self, bare_projection):
        MinskyActionFrameOverlay(
            template=MinskyActionFrameTemplate.cereal_pouring()
        ).apply(bare_projection)

        leaf = name_leaf(bare_projection, FrameRoleKind.SOURCE_CONTAINER)

        assert leaf.type_name == ROLE_FILLER_TYPE_NAME
        assert leaf.label == "Cereal Box"

    def test_unnamed_filler_becomes_an_unresolved_leaf(self, bare_projection):
        """
        The instruction never names an agent/effector, so that role's filler stays
        unresolved regardless of which structure or instruction it is asked about.
        """
        MinskyActionFrameOverlay(
            template=MinskyActionFrameTemplate.cereal_pouring()
        ).apply(bare_projection)

        leaf = name_leaf(bare_projection, FrameRoleKind.AGENT)

        assert leaf.type_name == UNRESOLVED_ROLE_FILLER_TYPE_NAME
        assert leaf.label == "?"

    def test_grounding_follows_the_instruction_not_the_role(self):
        """
        The same role is grounded or not purely based on whether its illustrative
        filler name is actually mentioned in the instruction that produced this
        projection -- not a fixed property of the role itself.
        """
        omitting_the_bowl = bare_projection_for("Pour cereal from the cereal box")
        MinskyActionFrameOverlay(
            template=MinskyActionFrameTemplate.cereal_pouring()
        ).apply(omitting_the_bowl)

        leaf = name_leaf(omitting_the_bowl, FrameRoleKind.DESTINATION_CONTAINER)

        assert leaf.type_name == UNRESOLVED_ROLE_FILLER_TYPE_NAME
        assert leaf.label == "?"

    def test_entities_bind_to_the_structures_own_role_slots(self, bare_projection):
        """
        The overlay binds by the role's ``binds_as`` concept, not by free text, so it
        attaches to any structure built from the same Flanagan role classes.
        """
        template = MinskyActionFrameTemplate.cereal_pouring()
        MinskyActionFrameOverlay(template=template).apply(bare_projection)

        participation_targets = [
            relation.target
            for relation in bare_projection.relations
            if relation.predicate == MinskyFrameRelation.PARTICIPATES_AS.value
        ]
        assert participation_targets
        assert all(
            isinstance(target.source, RoleSlot) for target in participation_targets
        )

    def test_applying_twice_adds_nothing(self, bare_projection):
        overlay = MinskyActionFrameOverlay(
            template=MinskyActionFrameTemplate.cereal_pouring()
        )
        overlay.apply(bare_projection)
        node_count = len(bare_projection.nodes)

        overlay.apply(bare_projection)

        assert len(bare_projection.nodes) == node_count


# %% image schema profile
class TestImageSchemaProfileOverlay:
    def test_root_label_is_generic(self):
        assert ImageSchemaProfile.cereal_pouring().label == "ImageSchemaModel"

    def test_profile_root_hangs_off_the_instruction(self, bare_projection):
        MinskyActionFrameOverlay(
            template=MinskyActionFrameTemplate.cereal_pouring()
        ).apply(bare_projection)
        ImageSchemaProfileOverlay(profile=ImageSchemaProfile.cereal_pouring()).apply(
            bare_projection
        )

        assert ImageSchemaRelation.HAS_PROFILE.value in predicates_from(
            bare_projection, bare_projection.instruction.identifier
        )

    def test_entity_fillers_reuse_the_action_frame_entity_nodes(self, bare_projection):
        """
        An entity-valued spatial primitive points at the *same* node the action frame
        already created, so the two representations share one entity rather than
        each drawing its own copy.
        """
        MinskyActionFrameOverlay(
            template=MinskyActionFrameTemplate.cereal_pouring()
        ).apply(bare_projection)
        entity_identifiers = {
            node.identifier
            for node in bare_projection.nodes
            if node.type_name == "EntityFrame"
        }

        ImageSchemaProfileOverlay(profile=ImageSchemaProfile.cereal_pouring()).apply(
            bare_projection
        )

        filled_by_entities = {
            relation.target.identifier
            for relation in bare_projection.relations
            if relation.predicate == ImageSchemaRelation.FILLED_BY.value
            and relation.target.type_name == "EntityFrame"
        }
        assert filled_by_entities
        assert filled_by_entities <= entity_identifiers

    def test_schema_instances_are_ordered_by_precedes(self, bare_projection):
        profile = ImageSchemaProfile.cereal_pouring()
        MinskyActionFrameOverlay(
            template=MinskyActionFrameTemplate.cereal_pouring()
        ).apply(bare_projection)

        ImageSchemaProfileOverlay(profile=profile).apply(bare_projection)

        precedes_count = sum(
            1
            for relation in bare_projection.relations
            if relation.predicate == ImageSchemaRelation.PRECEDES.value
        )
        assert precedes_count == len(profile.schema_instances) - 1
