"""
For any action frame, a knowledge base can say, per slot, what kind of thing normally fills it and which checkable conditions its filler is expected to meet: having a part, affording something, being in a state or posture. It can also state the defaults that hold when nothing says otherwise, and the conditions that must hold before the action starts. It can add changes beyond the core's consequences, each with the quantity changed and its direction. Finally, it can give for each parameter dimension the core names a rationale: which slots it concerns and which outcome condition it decides. Everything is stated in terms of the core's slots, parts, quantity kinds and parameter dimensions, so it holds for kinds of action and names no world.

Generated from ``action-frame.ttl`` by ``python -m llm_reasoner.knowledge.ontology_translation``. Regenerate rather than edit.
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


class ChangeDirection(StrEnum):
    """
    The direction in which a quantity changes.
    """

    INCREASES = "http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/action-frame#Increases"
    """
    The quantity becomes larger.
    """

    DECREASES = "http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/action-frame#Decreases"
    """
    The quantity becomes smaller.
    """


class Posture(StrEnum):
    """
    How a thing stands relative to the vertical, as a qualitative value of its orientation.
    """

    UPRIGHT = "http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/action-frame#Upright"
    """
    Standing in its usual position, with an opening facing up.
    """

    TILTED = "http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/action-frame#Tilted"
    """
    Leaning away from its upright position, short of upside down.
    """

    INVERTED = "http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/action-frame#Inverted"
    """
    Upside down, with an opening facing down.
    """


class SlotConditionPolarity(StrEnum):
    """
    Whether a slot condition is required to hold or required not to hold.
    """

    HOLDS = "http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/action-frame#Holds"
    """
    The condition is required to hold.
    """

    DOES_NOT_HOLD = "http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/action-frame#DoesNotHold"
    """
    The condition is required not to hold.
    """


# %% classes


@dataclass(eq=False)
class ActionDefault(DescriptionEntity):
    """
    A default of the frame that holds when nothing says otherwise, such as pouring until the goal holds the wanted amount rather than until the source is empty.
    """

    term: ClassVar[ClassTerm] = ClassTerm(
        iri="http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/action-frame#ActionDefault",
        linked_terms=(
            "http://www.ontologydesignpatterns.org/ont/dul/DUL.owl#Description",
        ),
        evidence_status="",
        claims=(),
        restrictions=(
            PropertyRestriction(
                property_iri="http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/action-frame#defaultCondition",
                quantifier=Quantifier.SOME,
                filler=NamedClass(
                    iri="http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/action-frame#SlotCondition"
                ),
            ),
        ),
        definition=None,
    )
    """
    The ontology class this class stands for.
    """

    default_condition: List[SlotCondition] = field(
        default_factory=list,
        metadata=PropertyTerm(
            iri="http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/action-frame#defaultCondition",
            functional=False,
            range="http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/action-frame#SlotCondition",
        ).as_dict(),
    )
    """
    The condition that holds by default, such as the goal holding the wanted amount.
    """

    default_for_slot: List[action_core.FrameSlot] = field(
        default_factory=list,
        metadata=PropertyTerm(
            iri="http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/action-frame#defaultForSlot",
            functional=False,
            range="http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/action-core#FrameSlot",
        ).as_dict(),
    )
    """
    The slot whose filler the default is about.
    """

    default_instead_of: List[SlotCondition] = field(
        default_factory=list,
        metadata=PropertyTerm(
            iri="http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/action-frame#defaultInsteadOf",
            functional=False,
            range="http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/action-frame#SlotCondition",
        ).as_dict(),
    )
    """
    The condition that does not hold by default although it could, such as the source being empty.
    """

    default_settles_dimension: List[action_core.ParameterDimension] = field(
        default_factory=list,
        metadata=PropertyTerm(
            iri="http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/action-frame#defaultSettlesDimension",
            functional=False,
            range="http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/action-core#ParameterDimension",
        ).as_dict(),
    )
    """
    The parameter dimension whose choice the default settles when nothing says otherwise.
    """


@dataclass(eq=False)
class AdditionalConsequence(action_core.Consequence):
    """
    A change an action makes beyond the consequences the core lists, stated as the quantity of a slot's filler that rises or falls, such as the source becoming lighter.
    """

    term: ClassVar[ClassTerm] = ClassTerm(
        iri="http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/action-frame#AdditionalConsequence",
        linked_terms=(),
        evidence_status="",
        claims=(),
        restrictions=(
            PropertyRestriction(
                property_iri="http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/action-core#affectsSlot",
                quantifier=Quantifier.SOME,
                filler=NamedClass(
                    iri="http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/action-core#FrameSlot"
                ),
            ),
            PropertyRestriction(
                property_iri="http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/action-core#hasQuantityKind",
                quantifier=Quantifier.SOME,
                filler=NamedClass(
                    iri="http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/action-core#QuantityKind"
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

    changes_in_direction: Optional[ChangeDirection] = field(
        default=None,
        metadata=PropertyTerm(
            iri="http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/action-frame#changesInDirection",
            functional=True,
        ).as_dict(),
    )
    """
    Whether the changed quantity rises or falls.
    """


@dataclass(eq=False)
class SlotCondition(DescriptionEntity):
    """
    A condition a world can check about the filler of a slot of an action frame, such as having a part, affording something or being in a state, rather than membership of a class.
    """

    term: ClassVar[ClassTerm] = ClassTerm(
        iri="http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/action-frame#SlotCondition",
        linked_terms=(
            "http://www.ontologydesignpatterns.org/ont/dul/DUL.owl#Description",
        ),
        evidence_status="",
        claims=(),
        restrictions=(
            PropertyRestriction(
                property_iri="http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/action-frame#conditionOnSlot",
                quantifier=Quantifier.SOME,
                filler=NamedClass(
                    iri="http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/action-core#FrameSlot"
                ),
            ),
            PropertyRestriction(
                property_iri="http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/action-frame#hasPolarity",
                quantifier=Quantifier.SOME,
                filler=NamedClass(
                    iri="http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/action-frame#SlotConditionPolarity"
                ),
            ),
        ),
        definition=None,
    )
    """
    The ontology class this class stands for.
    """

    condition_has_state: List[ConceptReference] = field(
        default_factory=list,
        metadata=PropertyTerm(
            iri="http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/action-frame#conditionHasState",
            functional=False,
        ).as_dict(),
    )
    """
    The state the filler must be in for the condition to hold, as the IRI of a CRAM-AICOR state type, such as aicor:GraspedState.
    """

    condition_involves_slot: List[action_core.FrameSlot] = field(
        default_factory=list,
        metadata=PropertyTerm(
            iri="http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/action-frame#conditionInvolvesSlot",
            functional=False,
            range="http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/action-core#FrameSlot",
        ).as_dict(),
    )
    """
    Another slot whose filler the condition relates the first filler to, such as the agent in the condition that the source is held.
    """

    condition_on_slot: Optional[action_core.FrameSlot] = field(
        default=None,
        metadata=PropertyTerm(
            iri="http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/action-frame#conditionOnSlot",
            functional=True,
            range="http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/action-core#FrameSlot",
        ).as_dict(),
    )
    """
    The slot whose filler the condition is about.
    """

    condition_requires_part: List[action_core.PartSlot] = field(
        default_factory=list,
        metadata=PropertyTerm(
            iri="http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/action-frame#conditionRequiresPart",
            functional=False,
            range="http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/action-core#PartSlot",
        ).as_dict(),
    )
    """
    The part slot that a part of the filler must fill for the condition to hold.
    """

    has_polarity: Optional[SlotConditionPolarity] = field(
        default=None,
        metadata=PropertyTerm(
            iri="http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/action-frame#hasPolarity",
            functional=True,
        ).as_dict(),
    )
    """
    Whether the condition is required to hold or required not to hold.
    """


@dataclass(eq=False)
class AffordanceCondition(SlotCondition):
    """
    A condition that the filler of a slot affords an action, such as the source being graspable or tiltable, or can be brought into a state, possibly with the filler of another slot, such as the goal being able to contain the substance.
    """

    term: ClassVar[ClassTerm] = ClassTerm(
        iri="http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/action-frame#AffordanceCondition",
        linked_terms=(),
        evidence_status="",
        claims=(),
        restrictions=(),
        definition=None,
    )
    """
    The ontology class this class stands for.
    """

    condition_affords_action: List[ConceptReference] = field(
        default_factory=list,
        metadata=PropertyTerm(
            iri="http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/action-frame#conditionAffordsAction",
            functional=False,
        ).as_dict(),
    )
    """
    A kind of action the filler must afford for the condition to hold, as the IRI of a CRAM-AICOR action category, such as aicor:Grasping or aicor:Tilting.
    """

    condition_affords_state: List[ConceptReference] = field(
        default_factory=list,
        metadata=PropertyTerm(
            iri="http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/action-frame#conditionAffordsState",
            functional=False,
        ).as_dict(),
    )
    """
    A state the filler must be able to be brought into for the condition to hold, as the IRI of a CRAM-AICOR state type, such as aicor:ContainedState for a goal that can contain the substance.
    """


@dataclass(eq=False)
class DimensionRationale(DescriptionEntity):
    """
    Why a parameter dimension matters to the action: the slots it concerns and the outcome condition it decides, such as the position of the source's lip relative to the goal's opening deciding whether the substance lands in the goal.
    """

    term: ClassVar[ClassTerm] = ClassTerm(
        iri="http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/action-frame#DimensionRationale",
        linked_terms=(
            "http://www.ontologydesignpatterns.org/ont/dul/DUL.owl#Description",
        ),
        evidence_status="",
        claims=(),
        restrictions=(
            PropertyRestriction(
                property_iri="http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/action-frame#explainsDimension",
                quantifier=Quantifier.SOME,
                filler=NamedClass(
                    iri="http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/action-core#ParameterDimension"
                ),
            ),
            PropertyRestriction(
                property_iri="http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/action-frame#rationaleConcernsSlot",
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

    decides_whether: List[SlotCondition] = field(
        default_factory=list,
        metadata=PropertyTerm(
            iri="http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/action-frame#decidesWhether",
            functional=False,
            range="http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/action-frame#SlotCondition",
        ).as_dict(),
    )
    """
    The outcome condition whose holding the dimension's choice decides.
    """

    explains_dimension: Optional[action_core.ParameterDimension] = field(
        default=None,
        metadata=PropertyTerm(
            iri="http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/action-frame#explainsDimension",
            functional=True,
            range="http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/action-core#ParameterDimension",
        ).as_dict(),
    )
    """
    The parameter dimension the rationale is about.
    """

    rationale_concerns_slot: List[action_core.FrameSlot] = field(
        default_factory=list,
        metadata=PropertyTerm(
            iri="http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/action-frame#rationaleConcernsSlot",
            functional=False,
            range="http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/action-core#FrameSlot",
        ).as_dict(),
    )
    """
    A slot whose filler the dimension matters for.
    """


@dataclass(eq=False)
class FeatureCondition(SlotCondition):
    """
    A condition that the filler of a slot has a part that fills a part slot of the frame, such as the source having a lip or the goal having an opening.
    """

    term: ClassVar[ClassTerm] = ClassTerm(
        iri="http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/action-frame#FeatureCondition",
        linked_terms=(),
        evidence_status="",
        claims=(),
        restrictions=(
            PropertyRestriction(
                property_iri="http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/action-frame#conditionRequiresPart",
                quantifier=Quantifier.SOME,
                filler=NamedClass(
                    iri="http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/action-core#PartSlot"
                ),
            ),
        ),
        definition=None,
    )
    """
    The ontology class this class stands for.
    """


@dataclass(eq=False)
class FrameStereotype(DescriptionEntity):
    """
    What an action frame normally expects, as Minsky's frames describe a stereotyped situation: what each slot's filler is expected to be, have or afford, the defaults, what must hold before the action and why each parameter dimension matters.
    """

    term: ClassVar[ClassTerm] = ClassTerm(
        iri="http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/action-frame#FrameStereotype",
        linked_terms=(
            "http://www.ontologydesignpatterns.org/ont/dul/DUL.owl#Description",
        ),
        evidence_status="",
        claims=(),
        restrictions=(
            PropertyRestriction(
                property_iri="http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/action-frame#stereotypeOf",
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

    gives_rationale: List[DimensionRationale] = field(
        default_factory=list,
        metadata=PropertyTerm(
            iri="http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/action-frame#givesRationale",
            functional=False,
            range="http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/action-frame#DimensionRationale",
        ).as_dict(),
    )
    """
    An account of why a parameter dimension of the action matters.
    """

    has_default: List[ActionDefault] = field(
        default_factory=list,
        metadata=PropertyTerm(
            iri="http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/action-frame#hasDefault",
            functional=False,
            range="http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/action-frame#ActionDefault",
        ).as_dict(),
    )
    """
    A default that holds in the frame when nothing says otherwise.
    """

    has_slot_expectation: List[SlotExpectation] = field(
        default_factory=list,
        metadata=PropertyTerm(
            iri="http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/action-frame#hasSlotExpectation",
            functional=False,
            range="http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/action-frame#SlotExpectation",
        ).as_dict(),
    )
    """
    An expectation the frame holds of one of its slots.
    """

    requires_before_start: List[SlotCondition] = field(
        default_factory=list,
        metadata=PropertyTerm(
            iri="http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/action-frame#requiresBeforeStart",
            functional=False,
            range="http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/action-frame#SlotCondition",
        ).as_dict(),
    )
    """
    A condition that must hold before the action starts.
    """

    stereotype_of: Optional[action_core.ActionFrame] = field(
        default=None,
        metadata=PropertyTerm(
            iri="http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/action-frame#stereotypeOf",
            functional=True,
            range="http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/action-core#ActionFrame",
        ).as_dict(),
    )
    """
    The action frame of the action core the stereotype is of.
    """


@dataclass(eq=False)
class PostureCondition(SlotCondition):
    """
    A condition that the filler of a slot stands in a given posture, such as the source standing upright before pouring.
    """

    term: ClassVar[ClassTerm] = ClassTerm(
        iri="http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/action-frame#PostureCondition",
        linked_terms=(),
        evidence_status="",
        claims=(),
        restrictions=(
            PropertyRestriction(
                property_iri="http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/action-core#hasQuantityKind",
                quantifier=Quantifier.VALUE,
                value="http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/action-core#Orientation",
            ),
            PropertyRestriction(
                property_iri="http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/action-frame#conditionRequiresPosture",
                quantifier=Quantifier.SOME,
                filler=NamedClass(
                    iri="http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/action-frame#Posture"
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

    condition_requires_posture: Optional[Posture] = field(
        default=None,
        metadata=PropertyTerm(
            iri="http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/action-frame#conditionRequiresPosture",
            functional=True,
        ).as_dict(),
    )
    """
    The posture the filler must stand in for the condition to hold.
    """


@dataclass(eq=False)
class SlotExpectation(DescriptionEntity):
    """
    What a frame expects of one of its slots: the kind of thing that normally fills it and the conditions its filler is expected to meet. It holds by default and can be overridden.
    """

    term: ClassVar[ClassTerm] = ClassTerm(
        iri="http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/action-frame#SlotExpectation",
        linked_terms=(
            "http://www.ontologydesignpatterns.org/ont/dul/DUL.owl#Description",
        ),
        evidence_status="",
        claims=(),
        restrictions=(
            PropertyRestriction(
                property_iri="http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/action-frame#expectsOfSlot",
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

    expects_filler_to_satisfy: List[SlotCondition] = field(
        default_factory=list,
        metadata=PropertyTerm(
            iri="http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/action-frame#expectsFillerToSatisfy",
            functional=False,
            range="http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/action-frame#SlotCondition",
        ).as_dict(),
    )
    """
    A condition the filler of the slot is expected to meet.
    """

    expects_of_slot: Optional[action_core.FrameSlot] = field(
        default=None,
        metadata=PropertyTerm(
            iri="http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/action-frame#expectsOfSlot",
            functional=True,
            range="http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/action-core#FrameSlot",
        ).as_dict(),
    )
    """
    The slot of the frame that the expectation is about.
    """

    normally_filled_by: List[ConceptReference] = field(
        default_factory=list,
        metadata=PropertyTerm(
            iri="http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/action-frame#normallyFilledBy",
            functional=False,
        ).as_dict(),
    )
    """
    A kind of thing that normally fills the slot, as the IRI of a CRAM-AICOR class, such as aicor:Container for the source of a pour.
    """


@dataclass(eq=False)
class StateCondition(SlotCondition):
    """
    A condition that the filler of a slot, possibly in relation to the filler of another slot, is in a given discrete state, such as the agent grasping the source or the substance being contained in the goal; with the polarity that it does not hold, that it is not in that state, such as the source being empty: the substance not contained in it.
    """

    term: ClassVar[ClassTerm] = ClassTerm(
        iri="http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/action-frame#StateCondition",
        linked_terms=(),
        evidence_status="",
        claims=(),
        restrictions=(
            PropertyRestriction(
                property_iri="http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/action-frame#conditionHasState",
                quantifier=Quantifier.SOME,
                filler=NamedClass(iri="http://www.w3.org/2002/07/owl#Thing"),
            ),
        ),
        definition=None,
    )
    """
    The ontology class this class stands for.
    """


# %% property descriptors


@dataclass
class ConditionInvolvesSlot(ObjectPropertyDescriptor):
    """
    Another slot whose filler the condition relates the first filler to, such as the agent in the condition that the source is held.
    """


@dataclass
class ConditionOnSlot(ObjectPropertyDescriptor):
    """
    The slot whose filler the condition is about.
    """


@dataclass
class ConditionRequiresPart(ObjectPropertyDescriptor):
    """
    The part slot that a part of the filler must fill for the condition to hold.
    """


@dataclass
class DecidesWhether(ObjectPropertyDescriptor):
    """
    The outcome condition whose holding the dimension's choice decides.
    """


@dataclass
class DefaultCondition(ObjectPropertyDescriptor):
    """
    The condition that holds by default, such as the goal holding the wanted amount.
    """


@dataclass
class DefaultForSlot(ObjectPropertyDescriptor):
    """
    The slot whose filler the default is about.
    """


@dataclass
class DefaultInsteadOf(ObjectPropertyDescriptor):
    """
    The condition that does not hold by default although it could, such as the source being empty.
    """


@dataclass
class DefaultSettlesDimension(ObjectPropertyDescriptor):
    """
    The parameter dimension whose choice the default settles when nothing says otherwise.
    """


@dataclass
class ExpectsFillerToSatisfy(ObjectPropertyDescriptor):
    """
    A condition the filler of the slot is expected to meet.
    """


@dataclass
class ExpectsOfSlot(ObjectPropertyDescriptor):
    """
    The slot of the frame that the expectation is about.
    """


@dataclass
class ExplainsDimension(ObjectPropertyDescriptor):
    """
    The parameter dimension the rationale is about.
    """


@dataclass
class GivesRationale(ObjectPropertyDescriptor):
    """
    An account of why a parameter dimension of the action matters.
    """


@dataclass
class HasDefault(ObjectPropertyDescriptor):
    """
    A default that holds in the frame when nothing says otherwise.
    """


@dataclass
class HasSlotExpectation(ObjectPropertyDescriptor):
    """
    An expectation the frame holds of one of its slots.
    """


@dataclass
class RationaleConcernsSlot(ObjectPropertyDescriptor):
    """
    A slot whose filler the dimension matters for.
    """


@dataclass
class RequiresBeforeStart(ObjectPropertyDescriptor):
    """
    A condition that must hold before the action starts.
    """


@dataclass
class StereotypeOf(ObjectPropertyDescriptor):
    """
    The action frame of the action core the stereotype is of.
    """


# %% descriptor attachments

SlotCondition.condition_involves_slot = ConditionInvolvesSlot(
    SlotCondition, "condition_involves_slot"
)
SlotCondition.condition_on_slot = ConditionOnSlot(SlotCondition, "condition_on_slot")
SlotCondition.condition_requires_part = ConditionRequiresPart(
    SlotCondition, "condition_requires_part"
)
DimensionRationale.decides_whether = DecidesWhether(
    DimensionRationale, "decides_whether"
)
ActionDefault.default_condition = DefaultCondition(ActionDefault, "default_condition")
ActionDefault.default_for_slot = DefaultForSlot(ActionDefault, "default_for_slot")
ActionDefault.default_instead_of = DefaultInsteadOf(ActionDefault, "default_instead_of")
ActionDefault.default_settles_dimension = DefaultSettlesDimension(
    ActionDefault, "default_settles_dimension"
)
SlotExpectation.expects_filler_to_satisfy = ExpectsFillerToSatisfy(
    SlotExpectation, "expects_filler_to_satisfy"
)
SlotExpectation.expects_of_slot = ExpectsOfSlot(SlotExpectation, "expects_of_slot")
DimensionRationale.explains_dimension = ExplainsDimension(
    DimensionRationale, "explains_dimension"
)
FrameStereotype.gives_rationale = GivesRationale(FrameStereotype, "gives_rationale")
FrameStereotype.has_default = HasDefault(FrameStereotype, "has_default")
FrameStereotype.has_slot_expectation = HasSlotExpectation(
    FrameStereotype, "has_slot_expectation"
)
DimensionRationale.rationale_concerns_slot = RationaleConcernsSlot(
    DimensionRationale, "rationale_concerns_slot"
)
FrameStereotype.requires_before_start = RequiresBeforeStart(
    FrameStereotype, "requires_before_start"
)
FrameStereotype.stereotype_of = StereotypeOf(FrameStereotype, "stereotype_of")
