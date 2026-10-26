"""
The action core's frame of pouring: the participants and parts a pour names, what it is for, and the situations it is performed in, as the material poured and the reason the pour ends.

It specialises the action core's frames, slots and objectives for one kind of action; the core itself names no action of its own.

Generated from ``pouring-frame.ttl`` by ``python -m llm_reasoner.knowledge.ontology_translation``. Regenerate rather than edit.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from typing_extensions import ClassVar, List, Optional

from llm_reasoner.knowledge import ontologies
from llm_reasoner.knowledge.ontologies.action_perspectives import action_core
from llm_reasoner.knowledge.ontologies.terms import (
    ClassTerm,
    ComplementOf,
    ConceptReference,
    DescriptionEntity,
    EnumeratedClass,
    IntersectionOf,
    NamedClass,
    OpenValuePartition,
    PropertyRestriction,
    PropertyTerm,
    Quantifier,
    UnionOf,
)

# %% value partitions


class MaterialRegime(OpenValuePartition):
    """
    The physical flow behaviour of a transferred substance.
    """

    LIQUID = "http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/pouring-frame#Liquid"
    """
    A low-shear material that flows freely and may slosh.
    """

    GRANULAR = "http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/pouring-frame#Granular"
    """
    Discrete particles whose bulk flow is not characterised by viscosity.
    """

    COHESIVE = "http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/pouring-frame#Cohesive"
    """
    A highly viscous or cohesive material with delayed flow.
    """

    @property
    def declaring_module(self) -> ontologies.OntologyModuleFile:
        return {
            MaterialRegime.LIQUID: ontologies.ActionPerspectiveFile.POURING_FRAME,
            MaterialRegime.GRANULAR: ontologies.ActionPerspectiveFile.POURING_FRAME,
            MaterialRegime.COHESIVE: ontologies.ActionPerspectiveFile.POURING_FRAME,
        }[self]


class TransferTerminationReason(OpenValuePartition):
    """
    The condition that intentionally or exceptionally ends a transfer.
    """

    TARGET_QUANTITY_ATTAINED = "http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/pouring-frame#TargetQuantityAttained"
    """
    The requested transferred quantity has been reached.
    """

    SOURCE_EXHAUSTED = "http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/pouring-frame#SourceExhausted"
    """
    No transferable material remains in the source.
    """

    INSTRUCTED_STOP = "http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/pouring-frame#InstructedStop"
    """
    An external instruction requests that the transfer stop.
    """

    TRANSFER_FAILURE = "http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/pouring-frame#TransferFailure"
    """
    A spill or another failure forces the transfer to stop.
    """

    @property
    def declaring_module(self) -> ontologies.OntologyModuleFile:
        return {
            TransferTerminationReason.TARGET_QUANTITY_ATTAINED: ontologies.ActionPerspectiveFile.POURING_FRAME,
            TransferTerminationReason.SOURCE_EXHAUSTED: ontologies.ActionPerspectiveFile.POURING_FRAME,
            TransferTerminationReason.INSTRUCTED_STOP: ontologies.ActionPerspectiveFile.POURING_FRAME,
            TransferTerminationReason.TRANSFER_FAILURE: ontologies.ActionPerspectiveFile.POURING_FRAME,
        }[self]


# %% classes


@dataclass(eq=False)
class PouringFrame(action_core.ManipulationActionFrame):
    """
    The frame of pouring, CRAM-AICOR's aicor:Pouring: an agent makes a substance leave a source over a part of it, such as its lip, and enter a goal through a part of it, such as its opening, so that the substance accumulates in the goal, as aicor:PouringIntention and its telos say. It names the agent, the source, the substance and the goal.
    """

    term: ClassVar[ClassTerm] = ClassTerm(
        iri="http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/pouring-frame#PouringFrame",
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
                quantifier=Quantifier.SOME,
                filler=NamedClass(iri="http://aicor.knowledge/l2-ontology.owl#Pouring"),
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
                            quantifier=Quantifier.SOME,
                            filler=NamedClass(
                                iri="http://aicor.knowledge/l2-ontology.owl#AccumulatedState"
                            ),
                        ),
                    )
                ),
            ),
            PropertyRestriction(
                property_iri="http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/pouring-frame#hasMaterialRegime",
                quantifier=Quantifier.ONLY,
                filler=NamedClass(
                    iri="http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/pouring-frame#MaterialRegime"
                ),
            ),
            PropertyRestriction(
                property_iri="http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/pouring-frame#hasTerminationReason",
                quantifier=Quantifier.ONLY,
                filler=NamedClass(
                    iri="http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/pouring-frame#TransferTerminationReason"
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
            iri="http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/pouring-frame#hasMaterialRegime",
            functional=True,
        ).as_dict(),
    )
    """
    The flow behaviour of the transferred substance.
    """

    has_termination_reason: Optional[TransferTerminationReason] = field(
        default=None,
        metadata=PropertyTerm(
            iri="http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/pouring-frame#hasTerminationReason",
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
        iri="http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/pouring-frame#CohesivePouringFrame",
        linked_terms=(),
        evidence_status="",
        claims=(),
        restrictions=(),
        definition=IntersectionOf(
            operands=(
                NamedClass(
                    iri="http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/pouring-frame#PouringFrame"
                ),
                PropertyRestriction(
                    property_iri="http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/pouring-frame#hasMaterialRegime",
                    quantifier=Quantifier.VALUE,
                    value="http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/pouring-frame#Cohesive",
                ),
            )
        ),
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
        iri="http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/pouring-frame#GranularPouringFrame",
        linked_terms=(),
        evidence_status="",
        claims=(),
        restrictions=(),
        definition=IntersectionOf(
            operands=(
                NamedClass(
                    iri="http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/pouring-frame#PouringFrame"
                ),
                PropertyRestriction(
                    property_iri="http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/pouring-frame#hasMaterialRegime",
                    quantifier=Quantifier.VALUE,
                    value="http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/pouring-frame#Granular",
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
        iri="http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/pouring-frame#InstructedStopPouringFrame",
        linked_terms=(),
        evidence_status="",
        claims=(),
        restrictions=(),
        definition=IntersectionOf(
            operands=(
                NamedClass(
                    iri="http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/pouring-frame#PouringFrame"
                ),
                PropertyRestriction(
                    property_iri="http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/pouring-frame#hasTerminationReason",
                    quantifier=Quantifier.VALUE,
                    value="http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/pouring-frame#InstructedStop",
                ),
            )
        ),
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
        iri="http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/pouring-frame#LiquidPouringFrame",
        linked_terms=(),
        evidence_status="",
        claims=(),
        restrictions=(),
        definition=IntersectionOf(
            operands=(
                NamedClass(
                    iri="http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/pouring-frame#PouringFrame"
                ),
                PropertyRestriction(
                    property_iri="http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/pouring-frame#hasMaterialRegime",
                    quantifier=Quantifier.VALUE,
                    value="http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/pouring-frame#Liquid",
                ),
            )
        ),
    )
    """
    The ontology class this class stands for.
    """


@dataclass(eq=False)
class SourceExhaustedPouringFrame(PouringFrame):
    """
    The frame of a pour that ends once the source is empty.
    """

    term: ClassVar[ClassTerm] = ClassTerm(
        iri="http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/pouring-frame#SourceExhaustedPouringFrame",
        linked_terms=(),
        evidence_status="",
        claims=(),
        restrictions=(),
        definition=IntersectionOf(
            operands=(
                NamedClass(
                    iri="http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/pouring-frame#PouringFrame"
                ),
                PropertyRestriction(
                    property_iri="http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/pouring-frame#hasTerminationReason",
                    quantifier=Quantifier.VALUE,
                    value="http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/pouring-frame#SourceExhausted",
                ),
            )
        ),
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
        iri="http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/pouring-frame#TargetQuantityAttainedPouringFrame",
        linked_terms=(),
        evidence_status="",
        claims=(),
        restrictions=(),
        definition=IntersectionOf(
            operands=(
                NamedClass(
                    iri="http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/pouring-frame#PouringFrame"
                ),
                PropertyRestriction(
                    property_iri="http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/pouring-frame#hasTerminationReason",
                    quantifier=Quantifier.VALUE,
                    value="http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/pouring-frame#TargetQuantityAttained",
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
        iri="http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/pouring-frame#TransferFailurePouringFrame",
        linked_terms=(),
        evidence_status="",
        claims=(),
        restrictions=(),
        definition=IntersectionOf(
            operands=(
                NamedClass(
                    iri="http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/pouring-frame#PouringFrame"
                ),
                PropertyRestriction(
                    property_iri="http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/pouring-frame#hasTerminationReason",
                    quantifier=Quantifier.VALUE,
                    value="http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/pouring-frame#TransferFailure",
                ),
            )
        ),
    )
    """
    The ontology class this class stands for.
    """
