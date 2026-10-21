"""
The hub every perspective on an action attaches to: the frame of a kind of action, the slots its participants fill, what it is for, what it brings about and the kinds of choice it leaves open.

A perspective, such as the action frame, the image schemas or predictive control, describes the action in terms of the core's slots and parameter dimensions and never in terms of another perspective. The resolver grounds a slot in an individual of the world through a CRAM-AICOR role binding, and maps the slots and parameter dimensions to the fields of an action designator.

Each concept is linked to CRAM-AICOR and to DUL: frames are descriptions, slots are CRAM-AICOR role slots specialising its participant roles, an objective is a CRAM-AICOR telos, a consequence a CRAM-AICOR effect, and a parameter dimension a DUL parameter in CRAM-AICOR's action parameter space. Axioms are read under the open-world assumption.

Generated from ``action-core.ttl`` by ``python -m llm_reasoner.knowledge.ontology_translation``. Regenerate rather than edit.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

from typing_extensions import ClassVar, List, Optional

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
    Whether something becomes larger or smaller, such as a quantity an action changes, a choice a correction moves or an estimate evidence revises. A direction that can also take values of its own lists these two among its members.
    """

    INCREASES = "http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/action-core#Increases"
    """
    Becomes larger.
    """

    DECREASES = "http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/action-core#Decreases"
    """
    Becomes smaller, or grows more slowly.
    """


class ConditionPolarity(StrEnum):
    """
    Whether a condition holds or does not hold. The property relating a condition to its polarity says whether that is demanded, such as a requirement of a phase or a frame, or observed, such as the condition a failure is noticed by.
    """

    HOLDS = (
        "http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/action-core#Holds"
    )
    """
    The condition holds.
    """

    DOES_NOT_HOLD = "http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/action-core#DoesNotHold"
    """
    The condition does not hold.
    """


class MaterialRegime(StrEnum):
    """
    The physical flow behaviour of a transferred substance.
    """

    LIQUID = "http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/action-core#Liquid"
    """
    A low-shear material that flows freely and may slosh.
    """

    GRANULAR = "http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/action-core#Granular"
    """
    Discrete particles whose bulk flow is not characterised by viscosity.
    """

    COHESIVE = "http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/action-core#Cohesive"
    """
    A highly viscous or cohesive material with delayed flow.
    """


class QuantityKind(StrEnum):
    """
    The kind of quantity a choice an action leaves open sets.
    """

    AMOUNT = "http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/action-core#Amount"
    """
    How much of something, such as the volume poured.
    """

    RATE = (
        "http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/action-core#Rate"
    )
    """
    How fast something changes, such as the flow or the tilting.
    """

    POSITION = "http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/action-core#Position"
    """
    Where something is, such as where the lip is over the goal.
    """

    ORIENTATION = "http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/action-core#Orientation"
    """
    Which way something faces, such as how far the source is tilted.
    """

    DISTANCE = "http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/action-core#Distance"
    """
    How far apart two things are, such as the height of the lip above the goal.
    """

    DURATION = "http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/action-core#Duration"
    """
    How long something lasts.
    """

    SELECTION = "http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/action-core#Selection"
    """
    Which of several alternatives is taken, such as the hand that holds the source.
    """


class TransferTerminationReason(StrEnum):
    """
    The condition that intentionally or exceptionally ends a transfer.
    """

    TARGET_QUANTITY_ATTAINED = "http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/action-core#TargetQuantityAttained"
    """
    The requested transferred quantity has been reached.
    """

    SOURCE_EXHAUSTED = "http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/action-core#SourceExhausted"
    """
    No transferable material remains in the source.
    """

    INSTRUCTED_STOP = "http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/action-core#InstructedStop"
    """
    An external instruction requests that the transfer stop.
    """

    TRANSFER_FAILURE = "http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/action-core#TransferFailure"
    """
    A spill or another failure forces the transfer to stop.
    """


# %% classes


@dataclass(eq=False)
class ActionFrame(DescriptionEntity):
    """
    The frame of a kind of action: the slots its participants fill, what it is for, what it brings about and the kinds of choice it leaves open. Every perspective on the action refers to the slots its frame defines.
    """

    term: ClassVar[ClassTerm] = ClassTerm(
        iri="http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/action-core#ActionFrame",
        linked_terms=(
            "http://aicor.knowledge/l2-ontology.owl#dul_Description",
            "http://www.ontologydesignpatterns.org/ont/dul/DUL.owl#Description",
        ),
        evidence_status="",
        claims=(),
        restrictions=(
            PropertyRestriction(
                property_iri="http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/action-core#definesSlot",
                quantifier=Quantifier.SOME,
                filler=NamedClass(
                    iri="http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/action-core#ParticipantSlot"
                ),
            ),
            PropertyRestriction(
                property_iri="http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/action-core#statesObjective",
                quantifier=Quantifier.SOME,
                filler=NamedClass(
                    iri="http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/action-core#Objective"
                ),
            ),
        ),
        definition=None,
    )
    """
    The ontology class this class stands for.
    """

    brings_about: List[Consequence] = field(
        default_factory=list,
        metadata=PropertyTerm(
            iri="http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/action-core#bringsAbout",
            functional=False,
            range="http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/action-core#Consequence",
        ).as_dict(),
    )
    """
    A change the action makes to one of its participants.
    """

    defines_slot: List[FrameSlot] = field(
        default_factory=list,
        metadata=PropertyTerm(
            iri="http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/action-core#definesSlot",
            functional=False,
            range="http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/action-core#FrameSlot",
        ).as_dict(),
    )
    """
    A slot of the frame, which a participant of the action, or a part of one, fills.
    """

    frames_action: Optional[ConceptReference] = field(
        default=None,
        metadata=PropertyTerm(
            iri="http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/action-core#framesAction",
            functional=True,
        ).as_dict(),
    )
    """
    The kind of action the frame describes, as the IRI of a CRAM-AICOR action category, such as aicor:Pouring.
    """

    leaves_open: List[ParameterDimension] = field(
        default_factory=list,
        metadata=PropertyTerm(
            iri="http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/action-core#leavesOpen",
            functional=False,
            range="http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/action-core#ParameterDimension",
        ).as_dict(),
    )
    """
    A kind of choice the frame does not settle, which whoever performs the action makes.
    """

    states_objective: List[Objective] = field(
        default_factory=list,
        metadata=PropertyTerm(
            iri="http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/action-core#statesObjective",
            functional=False,
            range="http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/action-core#Objective",
        ).as_dict(),
    )
    """
    What the action is for.
    """


@dataclass(eq=False)
class FrameSlot(DescriptionEntity):
    """
    A place in a frame that something taking part in the action fills; a CRAM-AICOR role binding binds an individual of the world to it.
    """

    term: ClassVar[ClassTerm] = ClassTerm(
        iri="http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/action-core#FrameSlot",
        linked_terms=("http://aicor.knowledge/l2-ontology.owl#RoleSlot",),
        evidence_status="",
        claims=(),
        restrictions=(),
        definition=None,
    )
    """
    The ontology class this class stands for.
    """


@dataclass(eq=False)
class ParticipantSlot(FrameSlot):
    """
    A slot that a participant of the action fills as a whole, such as the source of a pour.
    """

    term: ClassVar[ClassTerm] = ClassTerm(
        iri="http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/action-core#ParticipantSlot",
        linked_terms=(),
        evidence_status="",
        claims=(),
        restrictions=(),
        definition=None,
    )
    """
    The ontology class this class stands for.
    """


@dataclass(eq=False)
class AgentSlot(ParticipantSlot):
    """
    The slot of the one who performs the action.
    """

    term: ClassVar[ClassTerm] = ClassTerm(
        iri="http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/action-core#AgentSlot",
        linked_terms=("http://aicor.knowledge/l2-ontology.owl#AgentRole",),
        evidence_status="",
        claims=(),
        restrictions=(),
        definition=None,
    )
    """
    The ontology class this class stands for.
    """


@dataclass(eq=False)
class ManipulationActionFrame(ActionFrame):
    """
    The frame of an action that manipulates objects, which CRAM-AICOR calls a manipulation frame; an action that manipulates nothing, such as moving the robot's base, has a frame that is not one.
    """

    term: ClassVar[ClassTerm] = ClassTerm(
        iri="http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/action-core#ManipulationActionFrame",
        linked_terms=("http://aicor.knowledge/l2-ontology.owl#ManipulationFrame",),
        evidence_status="",
        claims=(),
        restrictions=(),
        definition=None,
    )
    """
    The ontology class this class stands for.
    """


@dataclass(eq=False)
class PouringFrame(ManipulationActionFrame):
    """
    The frame of pouring, CRAM-AICOR's aicor:Pouring: an agent makes a substance leave a source over a part of it, such as its lip, and enter a goal through a part of it, such as its opening, so that the substance accumulates in the goal, as aicor:PouringIntention and its telos say. It names the agent, the source, the substance and the goal.
    """

    term: ClassVar[ClassTerm] = ClassTerm(
        iri="http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/action-core#PouringFrame",
        linked_terms=(),
        evidence_status="",
        claims=(),
        restrictions=(
            PropertyRestriction(
                property_iri="http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/action-core#bringsAbout",
                quantifier=Quantifier.SOME,
                filler=NamedClass(
                    iri="http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/action-core#Consequence"
                ),
            ),
            PropertyRestriction(
                property_iri="http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/action-core#definesSlot",
                quantifier=Quantifier.SOME,
                filler=IntersectionOf(
                    operands=(
                        NamedClass(
                            iri="http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/action-core#PartSlot"
                        ),
                        PropertyRestriction(
                            property_iri="http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/action-core#isPartOfSlot",
                            quantifier=Quantifier.SOME,
                            filler=NamedClass(
                                iri="http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/action-core#GoalSlot"
                            ),
                        ),
                    )
                ),
            ),
            PropertyRestriction(
                property_iri="http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/action-core#definesSlot",
                quantifier=Quantifier.SOME,
                filler=IntersectionOf(
                    operands=(
                        NamedClass(
                            iri="http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/action-core#PartSlot"
                        ),
                        PropertyRestriction(
                            property_iri="http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/action-core#isPartOfSlot",
                            quantifier=Quantifier.SOME,
                            filler=NamedClass(
                                iri="http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/action-core#SourceSlot"
                            ),
                        ),
                    )
                ),
            ),
            PropertyRestriction(
                property_iri="http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/action-core#definesSlot",
                quantifier=Quantifier.SOME,
                filler=NamedClass(
                    iri="http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/action-core#AgentSlot"
                ),
            ),
            PropertyRestriction(
                property_iri="http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/action-core#definesSlot",
                quantifier=Quantifier.SOME,
                filler=NamedClass(
                    iri="http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/action-core#GoalSlot"
                ),
            ),
            PropertyRestriction(
                property_iri="http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/action-core#definesSlot",
                quantifier=Quantifier.SOME,
                filler=NamedClass(
                    iri="http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/action-core#SourceSlot"
                ),
            ),
            PropertyRestriction(
                property_iri="http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/action-core#definesSlot",
                quantifier=Quantifier.SOME,
                filler=NamedClass(
                    iri="http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/action-core#SubstanceSlot"
                ),
            ),
            PropertyRestriction(
                property_iri="http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/action-core#framesAction",
                quantifier=Quantifier.VALUE,
                value="http://aicor.knowledge/l2-ontology.owl#Pouring",
            ),
            PropertyRestriction(
                property_iri="http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/action-core#hasMaterialRegime",
                quantifier=Quantifier.ONLY,
                filler=NamedClass(
                    iri="http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/action-core#MaterialRegime"
                ),
            ),
            PropertyRestriction(
                property_iri="http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/action-core#hasTerminationReason",
                quantifier=Quantifier.ONLY,
                filler=NamedClass(
                    iri="http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/action-core#TransferTerminationReason"
                ),
            ),
            PropertyRestriction(
                property_iri="http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/action-core#leavesOpen",
                quantifier=Quantifier.SOME,
                filler=NamedClass(
                    iri="http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/action-core#ParameterDimension"
                ),
            ),
            PropertyRestriction(
                property_iri="http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/action-core#statesObjective",
                quantifier=Quantifier.SOME,
                filler=IntersectionOf(
                    operands=(
                        NamedClass(
                            iri="http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/action-core#Objective"
                        ),
                        PropertyRestriction(
                            property_iri="http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/action-core#aimsAtState",
                            quantifier=Quantifier.VALUE,
                            value="http://aicor.knowledge/l2-ontology.owl#AccumulatedState",
                        ),
                    )
                ),
            ),
        ),
        definition=None,
    )
    """
    The ontology class this class stands for.
    """

    has_material_regime: Optional[MaterialRegime] = field(
        default=None,
        metadata=PropertyTerm(
            iri="http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/action-core#hasMaterialRegime",
            functional=True,
        ).as_dict(),
    )
    """
    The flow behaviour of the transferred substance.
    """

    has_termination_reason: Optional[TransferTerminationReason] = field(
        default=None,
        metadata=PropertyTerm(
            iri="http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/action-core#hasTerminationReason",
            functional=True,
        ).as_dict(),
    )
    """
    The condition that ends the transfer.
    """


@dataclass(eq=False)
class CohesivePouringFrame(PouringFrame):
    """
    The frame of pouring a cohesive material.
    """

    term: ClassVar[ClassTerm] = ClassTerm(
        iri="http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/action-core#CohesivePouringFrame",
        linked_terms=(),
        evidence_status="",
        claims=(),
        restrictions=(),
        definition=IntersectionOf(
            operands=(
                NamedClass(
                    iri="http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/action-core#PouringFrame"
                ),
                PropertyRestriction(
                    property_iri="http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/action-core#hasMaterialRegime",
                    quantifier=Quantifier.VALUE,
                    value="http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/action-core#Cohesive",
                ),
            )
        ),
    )
    """
    The ontology class this class stands for.
    """


@dataclass(eq=False)
class Consequence(DescriptionEntity):
    """
    A change an action makes to one of its participants, intended or not, such as the source of a pour holding less, which CRAM-AICOR calls an effect.
    """

    term: ClassVar[ClassTerm] = ClassTerm(
        iri="http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/action-core#Consequence",
        linked_terms=(
            "http://aicor.knowledge/l2-ontology.owl#Effect",
            "http://www.ontologydesignpatterns.org/ont/dul/DUL.owl#Description",
        ),
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
        ),
        definition=None,
    )
    """
    The ontology class this class stands for.
    """

    affects_slot: List[FrameSlot] = field(
        default_factory=list,
        metadata=PropertyTerm(
            iri="http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/action-core#affectsSlot",
            functional=False,
            range="http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/action-core#FrameSlot",
        ).as_dict(),
    )
    """
    A slot whose filler the consequence changes.
    """

    results_in_state: Optional[ConceptReference] = field(
        default=None,
        metadata=PropertyTerm(
            iri="http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/action-core#resultsInState",
            functional=True,
        ).as_dict(),
    )
    """
    The state the filler is in afterwards, as the IRI of a CRAM-AICOR state type, such as aicor:ContainedState.
    """


@dataclass(eq=False)
class GoalSlot(ParticipantSlot):
    """
    The slot of where what the action moves ends up, such as the container poured into.
    """

    term: ClassVar[ClassTerm] = ClassTerm(
        iri="http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/action-core#GoalSlot",
        linked_terms=("http://aicor.knowledge/l2-ontology.owl#GoalRole",),
        evidence_status="",
        claims=(),
        restrictions=(),
        definition=None,
    )
    """
    The ontology class this class stands for.
    """


@dataclass(eq=False)
class GranularPouringFrame(PouringFrame):
    """
    The frame of pouring a granular material.
    """

    term: ClassVar[ClassTerm] = ClassTerm(
        iri="http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/action-core#GranularPouringFrame",
        linked_terms=(),
        evidence_status="",
        claims=(),
        restrictions=(),
        definition=IntersectionOf(
            operands=(
                NamedClass(
                    iri="http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/action-core#PouringFrame"
                ),
                PropertyRestriction(
                    property_iri="http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/action-core#hasMaterialRegime",
                    quantifier=Quantifier.VALUE,
                    value="http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/action-core#Granular",
                ),
            )
        ),
    )
    """
    The ontology class this class stands for.
    """


@dataclass(eq=False)
class InstructedStopPouringFrame(PouringFrame):
    """
    The frame of a pour that ends on an instruction.
    """

    term: ClassVar[ClassTerm] = ClassTerm(
        iri="http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/action-core#InstructedStopPouringFrame",
        linked_terms=(),
        evidence_status="",
        claims=(),
        restrictions=(),
        definition=IntersectionOf(
            operands=(
                NamedClass(
                    iri="http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/action-core#PouringFrame"
                ),
                PropertyRestriction(
                    property_iri="http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/action-core#hasTerminationReason",
                    quantifier=Quantifier.VALUE,
                    value="http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/action-core#InstructedStop",
                ),
            )
        ),
    )
    """
    The ontology class this class stands for.
    """


@dataclass(eq=False)
class InstrumentSlot(ParticipantSlot):
    """
    The slot of a tool the agent acts with on something else.
    """

    term: ClassVar[ClassTerm] = ClassTerm(
        iri="http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/action-core#InstrumentSlot",
        linked_terms=("http://aicor.knowledge/l2-ontology.owl#InstrumentRole",),
        evidence_status="",
        claims=(),
        restrictions=(),
        definition=None,
    )
    """
    The ontology class this class stands for.
    """


@dataclass(eq=False)
class LiquidPouringFrame(PouringFrame):
    """
    The frame of pouring a liquid.
    """

    term: ClassVar[ClassTerm] = ClassTerm(
        iri="http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/action-core#LiquidPouringFrame",
        linked_terms=(),
        evidence_status="",
        claims=(),
        restrictions=(),
        definition=IntersectionOf(
            operands=(
                NamedClass(
                    iri="http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/action-core#PouringFrame"
                ),
                PropertyRestriction(
                    property_iri="http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/action-core#hasMaterialRegime",
                    quantifier=Quantifier.VALUE,
                    value="http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/action-core#Liquid",
                ),
            )
        ),
    )
    """
    The ontology class this class stands for.
    """


@dataclass(eq=False)
class Objective(DescriptionEntity):
    """
    What an action is for: a state of its participants it aims at, which CRAM-AICOR calls a telos.
    """

    term: ClassVar[ClassTerm] = ClassTerm(
        iri="http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/action-core#Objective",
        linked_terms=("http://aicor.knowledge/l2-ontology.owl#Telos",),
        evidence_status="",
        claims=(),
        restrictions=(
            PropertyRestriction(
                property_iri="http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/action-core#concernsSlot",
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

    aims_at_state: Optional[ConceptReference] = field(
        default=None,
        metadata=PropertyTerm(
            iri="http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/action-core#aimsAtState",
            functional=True,
        ).as_dict(),
    )
    """
    The state the objective aims at, as the IRI of a CRAM-AICOR state type, such as aicor:AccumulatedState.
    """

    concerns_slot: List[FrameSlot] = field(
        default_factory=list,
        metadata=PropertyTerm(
            iri="http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/action-core#concernsSlot",
            functional=False,
            range="http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/action-core#FrameSlot",
        ).as_dict(),
    )
    """
    A slot whose filler the objective is about.
    """


@dataclass(eq=False)
class SlotQuantity(DescriptionEntity):
    """
    A kind of quantity about the filler of a slot, possibly measured from the filler of another slot, such as the tilt of the source or the height of the lip above the goal; it carries no value. What it is for is said by its kind: a choice the action leaves open, a quantity deciding whether a relation holds, or one a failure is noticed by.
    """

    term: ClassVar[ClassTerm] = ClassTerm(
        iri="http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/action-core#SlotQuantity",
        linked_terms=(
            "http://aicor.knowledge/l2-ontology.owl#dul_Concept",
            "http://www.ontologydesignpatterns.org/ont/dul/DUL.owl#Concept",
        ),
        evidence_status="",
        claims=(),
        restrictions=(
            PropertyRestriction(
                property_iri="http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/action-core#aboutSlot",
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

    about_slot: Optional[FrameSlot] = field(
        default=None,
        metadata=PropertyTerm(
            iri="http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/action-core#aboutSlot",
            functional=True,
            range="http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/action-core#FrameSlot",
        ).as_dict(),
    )
    """
    The slot whose filler the quantity is about, such as the source whose orientation is chosen.
    """

    has_quantity_kind: Optional[QuantityKind] = field(
        default=None,
        metadata=PropertyTerm(
            iri="http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/action-core#hasQuantityKind",
            functional=True,
        ).as_dict(),
    )
    """
    What kind of quantity something about an action is about, such as the quantity a choice sets, a condition constrains or a change changes.
    """

    relative_to_slot: Optional[FrameSlot] = field(
        default=None,
        metadata=PropertyTerm(
            iri="http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/action-core#relativeToSlot",
            functional=True,
            range="http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/action-core#FrameSlot",
        ).as_dict(),
    )
    """
    The slot whose filler the quantity is measured from, such as the goal a position above which is chosen, or the opening the lip ends beside; absent for a quantity measured from nothing in particular.
    """


@dataclass(eq=False)
class ParameterDimension(SlotQuantity):
    """
    A kind of choice an action leaves open, independent of any designator: how much, how fast, where relative to another participant, which way something faces, how far or how long, or which of several. A designator's field is one way of making the choice.
    """

    term: ClassVar[ClassTerm] = ClassTerm(
        iri="http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/action-core#ParameterDimension",
        linked_terms=(
            "http://aicor.knowledge/l2-ontology.owl#ActionParameterSpace",
            "http://www.ontologydesignpatterns.org/ont/dul/DUL.owl#Parameter",
        ),
        evidence_status="",
        claims=(),
        restrictions=(),
        definition=None,
    )
    """
    The ontology class this class stands for.
    """


@dataclass(eq=False)
class PartSlot(FrameSlot):
    """
    A slot that a part of a participant fills, which the action involves on its own, such as the lip of the source of a pour or the opening of its goal.
    """

    term: ClassVar[ClassTerm] = ClassTerm(
        iri="http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/action-core#PartSlot",
        linked_terms=(),
        evidence_status="",
        claims=(),
        restrictions=(
            PropertyRestriction(
                property_iri="http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/action-core#isPartOfSlot",
                quantifier=Quantifier.SOME,
                filler=NamedClass(
                    iri="http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/action-core#ParticipantSlot"
                ),
            ),
        ),
        definition=None,
    )
    """
    The ontology class this class stands for.
    """

    is_part_of_slot: Optional[ParticipantSlot] = field(
        default=None,
        metadata=PropertyTerm(
            iri="http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/action-core#isPartOfSlot",
            functional=True,
            range="http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/action-core#ParticipantSlot",
        ).as_dict(),
    )
    """
    The participant slot whose filler the part belongs to.
    """


@dataclass(eq=False)
class PatientSlot(ParticipantSlot):
    """
    The slot of what the action acts on and changes.
    """

    term: ClassVar[ClassTerm] = ClassTerm(
        iri="http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/action-core#PatientSlot",
        linked_terms=("http://aicor.knowledge/l2-ontology.owl#PatientRole",),
        evidence_status="",
        claims=(),
        restrictions=(),
        definition=None,
    )
    """
    The ontology class this class stands for.
    """


@dataclass(eq=False)
class SlotCondition(DescriptionEntity):
    """
    A condition a world can check about the filler of a slot, possibly in relation to the filler of another slot, such as having a part, affording something or being in a state, rather than membership of a class. Whether it is required or observed is said by the property relating it to what uses it.
    """

    term: ClassVar[ClassTerm] = ClassTerm(
        iri="http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/action-core#SlotCondition",
        linked_terms=(
            "http://aicor.knowledge/l2-ontology.owl#dul_Description",
            "http://www.ontologydesignpatterns.org/ont/dul/DUL.owl#Description",
        ),
        evidence_status="",
        claims=(),
        restrictions=(
            PropertyRestriction(
                property_iri="http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/action-core#conditionOnSlot",
                quantifier=Quantifier.SOME,
                filler=NamedClass(
                    iri="http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/action-core#FrameSlot"
                ),
            ),
            PropertyRestriction(
                property_iri="http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/action-core#hasPolarity",
                quantifier=Quantifier.SOME,
                filler=NamedClass(
                    iri="http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/action-core#ConditionPolarity"
                ),
            ),
        ),
        definition=None,
    )
    """
    The ontology class this class stands for.
    """

    condition_involves_slot: List[FrameSlot] = field(
        default_factory=list,
        metadata=PropertyTerm(
            iri="http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/action-core#conditionInvolvesSlot",
            functional=False,
            range="http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/action-core#FrameSlot",
        ).as_dict(),
    )
    """
    Another slot whose filler the condition relates the first filler to, such as the agent in the condition that the source is held, or the opening in the condition that the lip is over it.
    """

    condition_on_slot: Optional[FrameSlot] = field(
        default=None,
        metadata=PropertyTerm(
            iri="http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/action-core#conditionOnSlot",
            functional=True,
            range="http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/action-core#FrameSlot",
        ).as_dict(),
    )
    """
    The slot whose filler the condition is about.
    """

    has_polarity: Optional[ConditionPolarity] = field(
        default=None,
        metadata=PropertyTerm(
            iri="http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/action-core#hasPolarity",
            functional=True,
        ).as_dict(),
    )
    """
    Whether the condition holds or does not hold.
    """


@dataclass(eq=False)
class SourceExhaustedPouringFrame(PouringFrame):
    """
    The frame of a pour that ends once the source is empty.
    """

    term: ClassVar[ClassTerm] = ClassTerm(
        iri="http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/action-core#SourceExhaustedPouringFrame",
        linked_terms=(),
        evidence_status="",
        claims=(),
        restrictions=(),
        definition=IntersectionOf(
            operands=(
                NamedClass(
                    iri="http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/action-core#PouringFrame"
                ),
                PropertyRestriction(
                    property_iri="http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/action-core#hasTerminationReason",
                    quantifier=Quantifier.VALUE,
                    value="http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/action-core#SourceExhausted",
                ),
            )
        ),
    )
    """
    The ontology class this class stands for.
    """


@dataclass(eq=False)
class SourceSlot(ParticipantSlot):
    """
    The slot of where something the action moves comes from, such as the container poured from.
    """

    term: ClassVar[ClassTerm] = ClassTerm(
        iri="http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/action-core#SourceSlot",
        linked_terms=("http://aicor.knowledge/l2-ontology.owl#SourceRole",),
        evidence_status="",
        claims=(),
        restrictions=(),
        definition=None,
    )
    """
    The ontology class this class stands for.
    """


@dataclass(eq=False)
class StateCondition(SlotCondition):
    """
    A condition that the filler of a slot, possibly in relation to the filler of another slot, is in a given discrete state, such as the agent grasping the source or the substance being contained in the goal; with the polarity that it does not hold, that it is not in that state, such as the source being empty: the substance not contained in it.
    """

    term: ClassVar[ClassTerm] = ClassTerm(
        iri="http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/action-core#StateCondition",
        linked_terms=(),
        evidence_status="",
        claims=(),
        restrictions=(
            PropertyRestriction(
                property_iri="http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/action-core#conditionHasState",
                quantifier=Quantifier.SOME,
                filler=NamedClass(iri="http://www.w3.org/2002/07/owl#Thing"),
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
            iri="http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/action-core#conditionHasState",
            functional=False,
        ).as_dict(),
    )
    """
    The state the filler is in when the condition holds, as the IRI of a CRAM-AICOR state type, such as aicor:GraspedState or aicor:FlowingState.
    """


@dataclass(eq=False)
class SubstanceSlot(ParticipantSlot):
    """
    The slot of the material the action moves, as distinct from what holds it.
    """

    term: ClassVar[ClassTerm] = ClassTerm(
        iri="http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/action-core#SubstanceSlot",
        linked_terms=("http://aicor.knowledge/l2-ontology.owl#SubstanceRole",),
        evidence_status="",
        claims=(),
        restrictions=(),
        definition=None,
    )
    """
    The ontology class this class stands for.
    """


@dataclass(eq=False)
class TargetQuantityAttainedPouringFrame(PouringFrame):
    """
    The frame of a pour that ends once the wanted quantity is in the goal.
    """

    term: ClassVar[ClassTerm] = ClassTerm(
        iri="http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/action-core#TargetQuantityAttainedPouringFrame",
        linked_terms=(),
        evidence_status="",
        claims=(),
        restrictions=(),
        definition=IntersectionOf(
            operands=(
                NamedClass(
                    iri="http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/action-core#PouringFrame"
                ),
                PropertyRestriction(
                    property_iri="http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/action-core#hasTerminationReason",
                    quantifier=Quantifier.VALUE,
                    value="http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/action-core#TargetQuantityAttained",
                ),
            )
        ),
    )
    """
    The ontology class this class stands for.
    """


@dataclass(eq=False)
class TransferFailurePouringFrame(PouringFrame):
    """
    The frame of a pour that ends because it fails.
    """

    term: ClassVar[ClassTerm] = ClassTerm(
        iri="http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/action-core#TransferFailurePouringFrame",
        linked_terms=(),
        evidence_status="",
        claims=(),
        restrictions=(),
        definition=IntersectionOf(
            operands=(
                NamedClass(
                    iri="http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/action-core#PouringFrame"
                ),
                PropertyRestriction(
                    property_iri="http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/action-core#hasTerminationReason",
                    quantifier=Quantifier.VALUE,
                    value="http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/action-core#TransferFailure",
                ),
            )
        ),
    )
    """
    The ontology class this class stands for.
    """


# %% property descriptors


@dataclass
class AboutSlot(ObjectPropertyDescriptor):
    """
    The slot whose filler the quantity is about, such as the source whose orientation is chosen.
    """


@dataclass
class AffectsSlot(ObjectPropertyDescriptor):
    """
    A slot whose filler the consequence changes.
    """


@dataclass
class BringsAbout(ObjectPropertyDescriptor):
    """
    A change the action makes to one of its participants.
    """


@dataclass
class ConcernsSlot(ObjectPropertyDescriptor):
    """
    A slot whose filler the objective is about.
    """


@dataclass
class ConditionInvolvesSlot(ObjectPropertyDescriptor):
    """
    Another slot whose filler the condition relates the first filler to, such as the agent in the condition that the source is held, or the opening in the condition that the lip is over it.
    """


@dataclass
class ConditionOnSlot(ObjectPropertyDescriptor):
    """
    The slot whose filler the condition is about.
    """


@dataclass
class DefinesSlot(ObjectPropertyDescriptor):
    """
    A slot of the frame, which a participant of the action, or a part of one, fills.
    """


@dataclass
class IsPartOfSlot(ObjectPropertyDescriptor):
    """
    The participant slot whose filler the part belongs to.
    """


@dataclass
class LeavesOpen(ObjectPropertyDescriptor):
    """
    A kind of choice the frame does not settle, which whoever performs the action makes.
    """


@dataclass
class RelativeToSlot(ObjectPropertyDescriptor):
    """
    The slot whose filler the quantity is measured from, such as the goal a position above which is chosen, or the opening the lip ends beside; absent for a quantity measured from nothing in particular.
    """


@dataclass
class StatesObjective(ObjectPropertyDescriptor):
    """
    What the action is for.
    """


# %% descriptor attachments

SlotQuantity.about_slot = AboutSlot(SlotQuantity, "about_slot")
Consequence.affects_slot = AffectsSlot(Consequence, "affects_slot")
ActionFrame.brings_about = BringsAbout(ActionFrame, "brings_about")
Objective.concerns_slot = ConcernsSlot(Objective, "concerns_slot")
SlotCondition.condition_involves_slot = ConditionInvolvesSlot(
    SlotCondition, "condition_involves_slot"
)
SlotCondition.condition_on_slot = ConditionOnSlot(SlotCondition, "condition_on_slot")
ActionFrame.defines_slot = DefinesSlot(ActionFrame, "defines_slot")
PartSlot.is_part_of_slot = IsPartOfSlot(PartSlot, "is_part_of_slot")
ActionFrame.leaves_open = LeavesOpen(ActionFrame, "leaves_open")
SlotQuantity.relative_to_slot = RelativeToSlot(SlotQuantity, "relative_to_slot")
ActionFrame.states_objective = StatesObjective(ActionFrame, "states_objective")
