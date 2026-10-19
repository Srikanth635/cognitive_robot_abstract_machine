"""
An action's frame can be said to instantiate image schemas (source-path-goal, containment, verticality, support, contact). Each schema states which of the frame's slots take part and in which schema role, what relations it requires between those slots before, after or throughout the action, which of the core's parameter dimensions each relation constrains, and which kinds of quantity about which slots decide whether the relation holds, without giving any quantity a value.

Generated from ``image-schemas.ttl`` by ``python -m llm_reasoner.knowledge.ontology_translation``. Regenerate rather than edit.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

from typing_extensions import ClassVar, List, Optional

from llm_reasoner.knowledge.ontologies.action_perspectives import action_core
from llm_reasoner.knowledge.ontologies.properties import ObjectPropertyDescriptor
from llm_reasoner.knowledge.ontologies.terms import (
    ClassTerm,
    ComplementOf,
    ConceptReference,
    DescriptionEntity,
    EnumeratedClass,
    IntersectionOf,
    NamedClass,
    PropertyRestriction,
    PropertyTerm,
    Quantifier,
    UnionOf,
)

# %% value partitions


class QuantityAspect(StrEnum):
    """
    The aspects of a slot's filler that a deciding quantity can measure.
    """

    CENTRE = "http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/image-schemas#Centre"
    """
    The centre of the filler or of its opening.
    """

    EXTENT = "http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/image-schemas#Extent"
    """
    The size of the filler or of its opening.
    """

    LOCATION = "http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/image-schemas#Location"
    """
    Where the filler or a part of it is.
    """


class RelationKind(StrEnum):
    """
    The kinds of relation a schema can require between two slots.
    """

    ABOVE = "http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/image-schemas#Above"
    """
    The subject is higher than the reference along the vertical axis.
    """

    BELOW = "http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/image-schemas#Below"
    """
    The subject is lower than the reference along the vertical axis.
    """

    VERTICALLY_OVER = "http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/image-schemas#VerticallyOver"
    """
    The subject is above the reference and within its horizontal extent.
    """

    WITHIN_INTERIOR = "http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/image-schemas#WithinInterior"
    """
    The subject is within the interior of the reference.
    """

    OUTSIDE_INTERIOR = "http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/image-schemas#OutsideInterior"
    """
    The subject is not within the interior of the reference.
    """

    IN_CONTACT_WITH = "http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/image-schemas#InContactWith"
    """
    The subject touches the reference.
    """

    HOLDS_UP = "http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/image-schemas#HoldsUp"
    """
    The subject holds up the reference.
    """


class RequirementPhase(StrEnum):
    """
    The phases of an action in which a requirement holds.
    """

    BEFORE_ACTION = "http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/image-schemas#BeforeAction"
    """
    Holds before the action starts.
    """

    AFTER_ACTION = "http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/image-schemas#AfterAction"
    """
    Holds once the action has ended.
    """

    THROUGHOUT_ACTION = "http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/image-schemas#ThroughoutAction"
    """
    Holds during the whole action.
    """


class SchemaKind(StrEnum):
    """
    The kinds of image schema an action's frame can instantiate.
    """

    SOURCE_PATH_GOAL = "http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/image-schemas#SourcePathGoal"
    """
    Movement along a path from a source to a goal.
    """

    CONTAINMENT = "http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/image-schemas#Containment"
    """
    Being inside, entering or leaving a bounded interior.
    """

    VERTICALITY = "http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/image-schemas#Verticality"
    """
    Being above or below along the vertical axis.
    """

    SUPPORT_PATTERN = "http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/image-schemas#SupportPattern"
    """
    Holding something up against gravity.
    """

    CONTACT_PATTERN = "http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/image-schemas#ContactPattern"
    """
    Touching without a gap.
    """


class SchemaRoleKind(StrEnum):
    """
    The roles a slot can take within an image schema.
    """

    TRAJECTOR = "http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/image-schemas#Trajector"
    """
    The thing that moves or is located relative to another.
    """

    LANDMARK = "http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/image-schemas#Landmark"
    """
    The thing against which a trajector is located or moves.
    """

    PATH_START = "http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/image-schemas#PathStart"
    """
    Where the path begins.
    """

    PATH_END = "http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/image-schemas#PathEnd"
    """
    Where the path ends.
    """

    ENTRANCE = "http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/image-schemas#Entrance"
    """
    The way in to a bounded interior.
    """

    CONTAINING = "http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/image-schemas#Containing"
    """
    The bounded thing that has an interior.
    """

    CONTENT = "http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/image-schemas#Content"
    """
    What is inside the containing thing.
    """

    HIGHER = "http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/image-schemas#Higher"
    """
    The thing that is above in a vertical relation.
    """

    LOWER = "http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/image-schemas#Lower"
    """
    The thing that is below in a vertical relation.
    """

    SUPPORTER = "http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/image-schemas#Supporter"
    """
    The thing that holds up another.
    """

    SUPPORTED = "http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/image-schemas#Supported"
    """
    The thing that is held up.
    """

    TOUCHER = "http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/image-schemas#Toucher"
    """
    A thing in contact with another.
    """


# %% classes


@dataclass(eq=False)
class ImageSchema(DescriptionEntity):
    """
    A recurring spatial or force pattern, such as a path, a containment or a support, that the frame of an action instantiates; it says which slots take part and what it requires of them.
    """

    term: ClassVar[ClassTerm] = ClassTerm(
        iri="http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/image-schemas#ImageSchema",
        linked_terms=(
            "http://www.ontologydesignpatterns.org/ont/dul/DUL.owl#Description",
        ),
        evidence_status="",
        claims=(),
        restrictions=(
            PropertyRestriction(
                property_iri="http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/image-schemas#hasParticipation",
                quantifier=Quantifier.SOME,
                filler=NamedClass(
                    iri="http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/image-schemas#SchemaParticipation"
                ),
            ),
        ),
        definition=None,
    )
    """
    The ontology class this class stands for.
    """

    has_participation: List[SchemaParticipation] = field(
        default_factory=list,
        metadata=PropertyTerm(
            iri="http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/image-schemas#hasParticipation",
            functional=False,
            range="http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/image-schemas#SchemaParticipation",
        ).as_dict(),
    )
    """
    A participation of a slot of the frame in the schema.
    """

    has_requirement: List[SchemaRequirement] = field(
        default_factory=list,
        metadata=PropertyTerm(
            iri="http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/image-schemas#hasRequirement",
            functional=False,
            range="http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/image-schemas#SchemaRequirement",
        ).as_dict(),
    )
    """
    A relation between slots that the schema requires.
    """

    schema_kind: Optional[SchemaKind] = field(
        default=None,
        metadata=PropertyTerm(
            iri="http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/image-schemas#schemaKind",
            functional=True,
        ).as_dict(),
    )
    """
    Which kind of image schema this is, such as source-path-goal or containment.
    """


@dataclass(eq=False)
class ContactSchema(ImageSchema):
    """
    The image schema of two things touching, with no gap between them.
    """

    term: ClassVar[ClassTerm] = ClassTerm(
        iri="http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/image-schemas#ContactSchema",
        linked_terms=(),
        evidence_status="",
        claims=(),
        restrictions=(
            PropertyRestriction(
                property_iri="http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/image-schemas#schemaKind",
                quantifier=Quantifier.VALUE,
                value="http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/image-schemas#ContactPattern",
            ),
        ),
        definition=None,
    )
    """
    The ontology class this class stands for.
    """


@dataclass(eq=False)
class ContainmentSchema(ImageSchema):
    """
    The image schema of something being inside, entering or leaving a bounded interior.
    """

    term: ClassVar[ClassTerm] = ClassTerm(
        iri="http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/image-schemas#ContainmentSchema",
        linked_terms=(),
        evidence_status="",
        claims=(),
        restrictions=(
            PropertyRestriction(
                property_iri="http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/image-schemas#schemaKind",
                quantifier=Quantifier.VALUE,
                value="http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/image-schemas#Containment",
            ),
        ),
        definition=None,
    )
    """
    The ontology class this class stands for.
    """


@dataclass(eq=False)
class DecidingQuantity(DescriptionEntity):
    """
    A kind of quantity about a slot, such as the centre or size of an opening or the position of a lip, that decides whether a required relation holds; it carries no value.
    """

    term: ClassVar[ClassTerm] = ClassTerm(
        iri="http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/image-schemas#DecidingQuantity",
        linked_terms=(
            "http://www.ontologydesignpatterns.org/ont/dul/DUL.owl#Description",
        ),
        evidence_status="",
        claims=(),
        restrictions=(
            PropertyRestriction(
                property_iri="http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/action-core#hasQuantityKind",
                quantifier=Quantifier.SOME,
                filler=NamedClass(
                    iri="http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/action-core#QuantityKind"
                ),
            ),
            PropertyRestriction(
                property_iri="http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/image-schemas#quantityOfSlot",
                quantifier=Quantifier.SOME,
                filler=NamedClass(
                    iri="http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/action-core#FrameSlot"
                ),
            ),
        ),
        definition=None,
    )
    """
    The ontology class this class stands for.
    """

    has_quantity_kind: Optional[action_core.QuantityKind] = field(
        default=None,
        metadata=PropertyTerm(
            iri="http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/action-core#hasQuantityKind",
            functional=True,
        ).as_dict(),
    )
    """
    What kind of quantity something about an action is about, such as the quantity a choice sets, a condition constrains or a change changes.
    """

    quantity_aspect: Optional[QuantityAspect] = field(
        default=None,
        metadata=PropertyTerm(
            iri="http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/image-schemas#quantityAspect",
            functional=True,
        ).as_dict(),
    )
    """
    Which aspect of the slot's filler the quantity measures, such as its centre or its extent.
    """

    quantity_of_slot: Optional[action_core.FrameSlot] = field(
        default=None,
        metadata=PropertyTerm(
            iri="http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/image-schemas#quantityOfSlot",
            functional=True,
            range="http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/action-core#FrameSlot",
        ).as_dict(),
    )
    """
    The slot whose filler the deciding quantity is about.
    """


@dataclass(eq=False)
class SchemaParticipation(DescriptionEntity):
    """
    The taking part of one slot of the frame in a schema, in one role of that schema.
    """

    term: ClassVar[ClassTerm] = ClassTerm(
        iri="http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/image-schemas#SchemaParticipation",
        linked_terms=(
            "http://www.ontologydesignpatterns.org/ont/dul/DUL.owl#Description",
        ),
        evidence_status="",
        claims=(),
        restrictions=(
            PropertyRestriction(
                property_iri="http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/image-schemas#participatingSlot",
                quantifier=Quantifier.SOME,
                filler=NamedClass(
                    iri="http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/action-core#FrameSlot"
                ),
            ),
        ),
        definition=None,
    )
    """
    The ontology class this class stands for.
    """

    in_schema_role: List[SchemaRoleKind] = field(
        default_factory=list,
        metadata=PropertyTerm(
            iri="http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/image-schemas#inSchemaRole",
            functional=False,
        ).as_dict(),
    )
    """
    The role of the schema that the slot takes, such as where the path starts or the way in.
    """

    participating_slot: Optional[action_core.FrameSlot] = field(
        default=None,
        metadata=PropertyTerm(
            iri="http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/image-schemas#participatingSlot",
            functional=True,
            range="http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/action-core#FrameSlot",
        ).as_dict(),
    )
    """
    The slot of the frame that takes part in the schema.
    """


@dataclass(eq=False)
class SchemaRequirement(DescriptionEntity):
    """
    A relation a schema requires between two slots of the frame during one phase of the action, such as the lip of the source being above the opening of the goal before the substance leaves.
    """

    term: ClassVar[ClassTerm] = ClassTerm(
        iri="http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/image-schemas#SchemaRequirement",
        linked_terms=(
            "http://www.ontologydesignpatterns.org/ont/dul/DUL.owl#Description",
        ),
        evidence_status="",
        claims=(),
        restrictions=(
            PropertyRestriction(
                property_iri="http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/image-schemas#referenceSlot",
                quantifier=Quantifier.SOME,
                filler=NamedClass(
                    iri="http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/action-core#FrameSlot"
                ),
            ),
            PropertyRestriction(
                property_iri="http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/image-schemas#subjectSlot",
                quantifier=Quantifier.SOME,
                filler=NamedClass(
                    iri="http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/action-core#FrameSlot"
                ),
            ),
        ),
        definition=None,
    )
    """
    The ontology class this class stands for.
    """

    constrains_dimension: List[action_core.ParameterDimension] = field(
        default_factory=list,
        metadata=PropertyTerm(
            iri="http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/image-schemas#constrainsDimension",
            functional=False,
            range="http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/action-core#ParameterDimension",
        ).as_dict(),
    )
    """
    A parameter dimension of the core that the required relation constrains.
    """

    decided_by: List[DecidingQuantity] = field(
        default_factory=list,
        metadata=PropertyTerm(
            iri="http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/image-schemas#decidedBy",
            functional=False,
            range="http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/image-schemas#DecidingQuantity",
        ).as_dict(),
    )
    """
    A kind of quantity that decides whether the required relation holds.
    """

    holds_in_phase: Optional[RequirementPhase] = field(
        default=None,
        metadata=PropertyTerm(
            iri="http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/image-schemas#holdsInPhase",
            functional=True,
        ).as_dict(),
    )
    """
    The phase of the action in which the requirement holds.
    """

    names_spatial_relation: List[ConceptReference] = field(
        default_factory=list,
        metadata=PropertyTerm(
            iri="http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/image-schemas#namesSpatialRelation",
            functional=False,
        ).as_dict(),
    )
    """
    The CRAM-AICOR spatial relation concept, such as aicor:Inside or aicor:On, that the requirement names by its IRI.
    """

    names_state: List[ConceptReference] = field(
        default_factory=list,
        metadata=PropertyTerm(
            iri="http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/image-schemas#namesState",
            functional=False,
        ).as_dict(),
    )
    """
    The CRAM-AICOR state type, such as aicor:ContainedState, that the requirement amounts to, named by its IRI.
    """

    reference_slot: Optional[action_core.FrameSlot] = field(
        default=None,
        metadata=PropertyTerm(
            iri="http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/image-schemas#referenceSlot",
            functional=True,
            range="http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/action-core#FrameSlot",
        ).as_dict(),
    )
    """
    The slot that the required relation is measured against.
    """

    required_relation_kind: Optional[RelationKind] = field(
        default=None,
        metadata=PropertyTerm(
            iri="http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/image-schemas#requiredRelationKind",
            functional=True,
        ).as_dict(),
    )
    """
    The kind of relation required between the subject slot and the reference slot.
    """

    subject_slot: Optional[action_core.FrameSlot] = field(
        default=None,
        metadata=PropertyTerm(
            iri="http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/image-schemas#subjectSlot",
            functional=True,
            range="http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/action-core#FrameSlot",
        ).as_dict(),
    )
    """
    The slot that the required relation is about.
    """


@dataclass(eq=False)
class SchemaStereotype(DescriptionEntity):
    """
    The image schemas an action's frame normally instantiates: for each, the slots that take part and what it requires of them.
    """

    term: ClassVar[ClassTerm] = ClassTerm(
        iri="http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/image-schemas#SchemaStereotype",
        linked_terms=(
            "http://www.ontologydesignpatterns.org/ont/dul/DUL.owl#Description",
        ),
        evidence_status="",
        claims=(),
        restrictions=(
            PropertyRestriction(
                property_iri="http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/image-schemas#schemaStereotypeOf",
                quantifier=Quantifier.SOME,
                filler=NamedClass(
                    iri="http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/action-core#ActionFrame"
                ),
            ),
        ),
        definition=None,
    )
    """
    The ontology class this class stands for.
    """

    has_schema: List[ImageSchema] = field(
        default_factory=list,
        metadata=PropertyTerm(
            iri="http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/image-schemas#hasSchema",
            functional=False,
            range="http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/image-schemas#ImageSchema",
        ).as_dict(),
    )
    """
    An image schema that the action's frame instantiates.
    """

    schema_stereotype_of: Optional[action_core.ActionFrame] = field(
        default=None,
        metadata=PropertyTerm(
            iri="http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/image-schemas#schemaStereotypeOf",
            functional=True,
            range="http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/action-core#ActionFrame",
        ).as_dict(),
    )
    """
    The action frame of the action core the stereotype is of.
    """


@dataclass(eq=False)
class SourcePathGoalSchema(ImageSchema):
    """
    The image schema of something moving along a path from a source to a goal.
    """

    term: ClassVar[ClassTerm] = ClassTerm(
        iri="http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/image-schemas#SourcePathGoalSchema",
        linked_terms=(),
        evidence_status="",
        claims=(),
        restrictions=(
            PropertyRestriction(
                property_iri="http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/image-schemas#schemaKind",
                quantifier=Quantifier.VALUE,
                value="http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/image-schemas#SourcePathGoal",
            ),
        ),
        definition=None,
    )
    """
    The ontology class this class stands for.
    """


@dataclass(eq=False)
class SupportSchema(ImageSchema):
    """
    The image schema of one thing holding up another against gravity.
    """

    term: ClassVar[ClassTerm] = ClassTerm(
        iri="http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/image-schemas#SupportSchema",
        linked_terms=(),
        evidence_status="",
        claims=(),
        restrictions=(
            PropertyRestriction(
                property_iri="http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/image-schemas#schemaKind",
                quantifier=Quantifier.VALUE,
                value="http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/image-schemas#SupportPattern",
            ),
        ),
        definition=None,
    )
    """
    The ontology class this class stands for.
    """


@dataclass(eq=False)
class VerticalitySchema(ImageSchema):
    """
    The image schema of one thing being above or below another along the vertical axis.
    """

    term: ClassVar[ClassTerm] = ClassTerm(
        iri="http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/image-schemas#VerticalitySchema",
        linked_terms=(),
        evidence_status="",
        claims=(),
        restrictions=(
            PropertyRestriction(
                property_iri="http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/image-schemas#schemaKind",
                quantifier=Quantifier.VALUE,
                value="http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/image-schemas#Verticality",
            ),
        ),
        definition=None,
    )
    """
    The ontology class this class stands for.
    """


# %% property descriptors


@dataclass
class ConstrainsDimension(ObjectPropertyDescriptor):
    """
    A parameter dimension of the core that the required relation constrains.
    """


@dataclass
class DecidedBy(ObjectPropertyDescriptor):
    """
    A kind of quantity that decides whether the required relation holds.
    """


@dataclass
class HasParticipation(ObjectPropertyDescriptor):
    """
    A participation of a slot of the frame in the schema.
    """


@dataclass
class HasRequirement(ObjectPropertyDescriptor):
    """
    A relation between slots that the schema requires.
    """


@dataclass
class HasSchema(ObjectPropertyDescriptor):
    """
    An image schema that the action's frame instantiates.
    """


@dataclass
class ParticipatingSlot(ObjectPropertyDescriptor):
    """
    The slot of the frame that takes part in the schema.
    """


@dataclass
class QuantityOfSlot(ObjectPropertyDescriptor):
    """
    The slot whose filler the deciding quantity is about.
    """


@dataclass
class ReferenceSlot(ObjectPropertyDescriptor):
    """
    The slot that the required relation is measured against.
    """


@dataclass
class SchemaStereotypeOf(ObjectPropertyDescriptor):
    """
    The action frame of the action core the stereotype is of.
    """


@dataclass
class SubjectSlot(ObjectPropertyDescriptor):
    """
    The slot that the required relation is about.
    """


# %% descriptor attachments

SchemaRequirement.constrains_dimension = ConstrainsDimension(
    SchemaRequirement, "constrains_dimension"
)
SchemaRequirement.decided_by = DecidedBy(SchemaRequirement, "decided_by")
ImageSchema.has_participation = HasParticipation(ImageSchema, "has_participation")
ImageSchema.has_requirement = HasRequirement(ImageSchema, "has_requirement")
SchemaStereotype.has_schema = HasSchema(SchemaStereotype, "has_schema")
SchemaParticipation.participating_slot = ParticipatingSlot(
    SchemaParticipation, "participating_slot"
)
DecidingQuantity.quantity_of_slot = QuantityOfSlot(DecidingQuantity, "quantity_of_slot")
SchemaRequirement.reference_slot = ReferenceSlot(SchemaRequirement, "reference_slot")
SchemaStereotype.schema_stereotype_of = SchemaStereotypeOf(
    SchemaStereotype, "schema_stereotype_of"
)
SchemaRequirement.subject_slot = SubjectSlot(SchemaRequirement, "subject_slot")
