"""
Substance transfer as an engineering extension, tied to CRAM-AICOR's pouring model: aicor:Pouring, the roles of aicor:PouringIntention, the states of aicor:PouringTemplate and aicor:EffluxOnsetEvent.

Generated from ``pouring.ttl`` by ``python -m llm_reasoner.knowledge.ontology_translation``. Regenerate rather than edit.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

from typing_extensions import ClassVar, List, Optional

from llm_reasoner.knowledge.ontologies.predictive_manipulation import core
from llm_reasoner.knowledge.ontologies.predictive_manipulation import prehension
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


class MaterialRegime(StrEnum):
    """
    The physical flow behaviour of a transferred substance.
    """

    LIQUID = "http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/pouring#Liquid"
    """
    A low-shear material that flows freely and may slosh.
    """

    GRANULAR = "http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/pouring#Granular"
    """
    Discrete particles whose bulk flow is not characterised by viscosity.
    """

    COHESIVE = "http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/pouring#Cohesive"
    """
    A highly viscous or cohesive material with delayed flow.
    """


class TransferTerminationReason(StrEnum):
    """
    The condition that intentionally or exceptionally ends the transfer.
    """

    TARGET_QUANTITY_ATTAINED = "http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/pouring#TargetQuantityAttained"
    """
    The requested transferred quantity has been reached.
    """

    SOURCE_EXHAUSTED = "http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/pouring#SourceExhausted"
    """
    No transferable material remains in the source.
    """

    INSTRUCTED_STOP = "http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/pouring#InstructedStop"
    """
    An external instruction requests that the transfer stop.
    """

    TRANSFER_FAILURE = "http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/pouring#TransferFailure"
    """
    A spill or another failure forces the transfer to stop.
    """


# %% classes


@dataclass(eq=False)
class AlignmentErrorEstimate(core.AngleParameter, core.ObjectStateEstimate):
    """
    The angular deviation between the pouring lip and the receptacle opening.
    """

    term: ClassVar[ClassTerm] = ClassTerm(
        iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/pouring#AlignmentErrorEstimate",
        linked_terms=(),
        evidence_status="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#EngineeringExtension",
        claims=(),
        restrictions=(),
        definition=None,
    )
    """
    The ontology class this class stands for.
    """


@dataclass(eq=False)
class AlignmentPhase(core.ActionPhase):
    """
    Positioning of the lip over the receptacle before tilting. Optional when the source already starts aligned.
    """

    term: ClassVar[ClassTerm] = ClassTerm(
        iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/pouring#AlignmentPhase",
        linked_terms=(),
        evidence_status="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#EngineeringExtension",
        claims=(),
        restrictions=(
            PropertyRestriction(
                property_iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#hasCouplingRegime",
                quantifier=Quantifier.VALUE,
                value="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#Coupled",
            ),
        ),
        definition=None,
    )
    """
    The ontology class this class stands for.
    """


@dataclass(eq=False)
class CenterOfMassShiftEstimate(core.DynamicsEstimate, core.LengthParameter):
    """
    The predicted displacement of the centre of mass as substance leaves the source.
    """

    term: ClassVar[ClassTerm] = ClassTerm(
        iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/pouring#CenterOfMassShiftEstimate",
        linked_terms=(),
        evidence_status="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#EngineeringExtension",
        claims=(),
        restrictions=(),
        definition=None,
    )
    """
    The ontology class this class stands for.
    """


@dataclass(eq=False)
class CohesiveFlowDynamicsEstimate(core.CategoricalParameter, core.DynamicsEstimate):
    """
    A categorical model of delayed or intermittent cohesive flow.
    """

    term: ClassVar[ClassTerm] = ClassTerm(
        iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/pouring#CohesiveFlowDynamicsEstimate",
        linked_terms=(),
        evidence_status="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#EngineeringExtension",
        claims=(),
        restrictions=(),
        definition=None,
    )
    """
    The ontology class this class stands for.
    """


@dataclass(eq=False)
class PouringPhaseStructure(prehension.PrehensionPhaseStructure):
    """
    A substance-transfer phase structure composed with prehension: the source is reached, grasped, loaded, lifted and transported, aligned and tilted until the substance flows, regulated and stopped, returned upright, put back and released. Unloading and withdrawing the hand are optional. Grip force is coupled to the load, and gaze is fixed while aligning the lip and while regulating the flow.
    """

    term: ClassVar[ClassTerm] = ClassTerm(
        iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/pouring#PouringPhaseStructure",
        linked_terms=(),
        evidence_status="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#EngineeringExtension",
        claims=(),
        restrictions=(
            PropertyRestriction(
                property_iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#decomposesTask",
                quantifier=Quantifier.VALUE,
                value="http://aicor.knowledge/l2-ontology.owl#Pouring",
            ),
            PropertyRestriction(
                property_iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#definesPhase",
                quantifier=Quantifier.SOME,
                filler=NamedClass(
                    iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/pouring#AlignmentPhase"
                ),
            ),
            PropertyRestriction(
                property_iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#definesPhase",
                quantifier=Quantifier.SOME,
                filler=NamedClass(
                    iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/pouring#FlowRegulationPhase"
                ),
            ),
            PropertyRestriction(
                property_iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#definesPhase",
                quantifier=Quantifier.SOME,
                filler=NamedClass(
                    iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/pouring#FlowTerminationPhase"
                ),
            ),
            PropertyRestriction(
                property_iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#definesPhase",
                quantifier=Quantifier.SOME,
                filler=NamedClass(
                    iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/pouring#ReturnUprightPhase"
                ),
            ),
            PropertyRestriction(
                property_iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#definesPhase",
                quantifier=Quantifier.SOME,
                filler=NamedClass(
                    iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/pouring#TiltToFlowPhase"
                ),
            ),
            PropertyRestriction(
                property_iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#definesPhase",
                quantifier=Quantifier.SOME,
                filler=NamedClass(
                    iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/prehension#ContactEstablishmentPhase"
                ),
            ),
            PropertyRestriction(
                property_iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#definesPhase",
                quantifier=Quantifier.SOME,
                filler=NamedClass(
                    iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/prehension#ContactReleasePhase"
                ),
            ),
            PropertyRestriction(
                property_iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#definesPhase",
                quantifier=Quantifier.SOME,
                filler=NamedClass(
                    iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/prehension#ForceLoadingPhase"
                ),
            ),
            PropertyRestriction(
                property_iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#definesPhase",
                quantifier=Quantifier.SOME,
                filler=NamedClass(
                    iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/prehension#LiftPhase"
                ),
            ),
            PropertyRestriction(
                property_iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#definesPhase",
                quantifier=Quantifier.SOME,
                filler=NamedClass(
                    iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/prehension#ReachPhase"
                ),
            ),
            PropertyRestriction(
                property_iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#definesPhase",
                quantifier=Quantifier.SOME,
                filler=NamedClass(
                    iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/prehension#ReplacePhase"
                ),
            ),
            PropertyRestriction(
                property_iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#definesPhase",
                quantifier=Quantifier.SOME,
                filler=NamedClass(
                    iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/prehension#TransportPhase"
                ),
            ),
            PropertyRestriction(
                property_iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#definesRole",
                quantifier=Quantifier.SOME,
                filler=NamedClass(
                    iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/pouring#PouredSubstance"
                ),
            ),
            PropertyRestriction(
                property_iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#definesRole",
                quantifier=Quantifier.SOME,
                filler=NamedClass(
                    iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/pouring#SourceContainer"
                ),
            ),
            PropertyRestriction(
                property_iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#definesRole",
                quantifier=Quantifier.SOME,
                filler=NamedClass(
                    iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/pouring#TargetReceptacle"
                ),
            ),
            PropertyRestriction(
                property_iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/gaze#definesGazeFixation",
                quantifier=Quantifier.SOME,
                filler=IntersectionOf(
                    operands=(
                        NamedClass(
                            iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/gaze#GazeFixation"
                        ),
                        PropertyRestriction(
                            property_iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/gaze#servesPhase",
                            quantifier=Quantifier.SOME,
                            filler=NamedClass(
                                iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/pouring#AlignmentPhase"
                            ),
                        ),
                    )
                ),
            ),
            PropertyRestriction(
                property_iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/gaze#definesGazeFixation",
                quantifier=Quantifier.SOME,
                filler=IntersectionOf(
                    operands=(
                        NamedClass(
                            iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/gaze#GazeFixation"
                        ),
                        PropertyRestriction(
                            property_iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/gaze#servesPhase",
                            quantifier=Quantifier.SOME,
                            filler=NamedClass(
                                iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/pouring#FlowRegulationPhase"
                            ),
                        ),
                    )
                ),
            ),
            PropertyRestriction(
                property_iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/pouring#hasMaterialRegime",
                quantifier=Quantifier.SOME,
                filler=NamedClass(
                    iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/pouring#MaterialRegime"
                ),
            ),
            PropertyRestriction(
                property_iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/pouring#hasTerminationReason",
                quantifier=Quantifier.SOME,
                filler=NamedClass(
                    iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/pouring#TransferTerminationReason"
                ),
            ),
            PropertyRestriction(
                property_iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/prehension#definesForceCoupling",
                quantifier=Quantifier.SOME,
                filler=NamedClass(
                    iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/prehension#GripLoadCoupling"
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
            iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/pouring#hasMaterialRegime",
            functional=True,
        ).as_dict(),
    )
    """
    The flow behaviour of the transferred substance.
    """

    has_termination_reason: Optional[TransferTerminationReason] = field(
        default=None,
        metadata=PropertyTerm(
            iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/pouring#hasTerminationReason",
            functional=True,
        ).as_dict(),
    )
    """
    The condition that ends the transfer in this task variant.
    """


@dataclass(eq=False)
class CohesivePouringStructure(PouringPhaseStructure):
    """
    A pouring structure transferring a cohesive material; it declares no slosh or granular flow dynamics.
    """

    term: ClassVar[ClassTerm] = ClassTerm(
        iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/pouring#CohesivePouringStructure",
        linked_terms=(),
        evidence_status="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#EngineeringExtension",
        claims=(),
        restrictions=(
            PropertyRestriction(
                property_iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#definesParameter",
                quantifier=Quantifier.ONLY,
                filler=ComplementOf(
                    operand=UnionOf(
                        operands=(
                            NamedClass(
                                iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/pouring#SloshDynamicsEstimate"
                            ),
                            NamedClass(
                                iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/pouring#GranularFlowDynamicsEstimate"
                            ),
                        )
                    )
                ),
            ),
        ),
        definition=IntersectionOf(
            operands=(
                NamedClass(
                    iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/pouring#PouringPhaseStructure"
                ),
                PropertyRestriction(
                    property_iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/pouring#hasMaterialRegime",
                    quantifier=Quantifier.VALUE,
                    value="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/pouring#Cohesive",
                ),
            )
        ),
    )
    """
    The ontology class this class stands for.
    """


@dataclass(eq=False)
class DynamicViscosityEstimate(
    core.DynamicViscosityParameter, core.ObjectPropertyEstimate
):
    """
    The anticipated dynamic viscosity of a liquid or cohesive substance. Not applicable to granular material.
    """

    term: ClassVar[ClassTerm] = ClassTerm(
        iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/pouring#DynamicViscosityEstimate",
        linked_terms=(),
        evidence_status="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#EngineeringExtension",
        claims=(),
        restrictions=(),
        definition=None,
    )
    """
    The ontology class this class stands for.
    """


@dataclass(eq=False)
class FlowEventType(core.MechanicalEventType):
    """
    A mechanical event constituted by a change in substance motion.
    """

    term: ClassVar[ClassTerm] = ClassTerm(
        iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/pouring#FlowEventType",
        linked_terms=(),
        evidence_status="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#EngineeringExtension",
        claims=(),
        restrictions=(),
        definition=None,
    )
    """
    The ontology class this class stands for.
    """


@dataclass(eq=False)
class FlowCessation(FlowEventType):
    """
    Substance stops crossing the pouring lip.
    """

    term: ClassVar[ClassTerm] = ClassTerm(
        iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/pouring#FlowCessation",
        linked_terms=(),
        evidence_status="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#EngineeringExtension",
        claims=(),
        restrictions=(
            PropertyRestriction(
                property_iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#terminatesStateType",
                quantifier=Quantifier.SOME,
                filler=NamedClass(
                    iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/pouring#SubstanceFlowing"
                ),
            ),
        ),
        definition=None,
    )
    """
    The ontology class this class stands for.
    """


@dataclass(eq=False)
class FlowOnset(FlowEventType, core.ThresholdCrossingEventType):
    """
    Substance first crosses the lip once the onset threshold is crossed: its weight overcomes the containment by the source.
    """

    term: ClassVar[ClassTerm] = ClassTerm(
        iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/pouring#FlowOnset",
        linked_terms=(),
        evidence_status="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#EngineeringExtension",
        claims=(),
        restrictions=(
            PropertyRestriction(
                property_iri="http://aicor.knowledge/l2-ontology.owl#classifies",
                quantifier=Quantifier.ONLY,
                filler=NamedClass(
                    iri="http://aicor.knowledge/l2-ontology.owl#EffluxOnsetEvent"
                ),
            ),
            PropertyRestriction(
                property_iri="http://aicor.knowledge/l2-ontology.owl#initiatesStateType",
                quantifier=Quantifier.SOME,
                filler=NamedClass(
                    iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/pouring#SubstanceFlowing"
                ),
            ),
            PropertyRestriction(
                property_iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#crossesThreshold",
                quantifier=Quantifier.SOME,
                filler=NamedClass(
                    iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/pouring#FlowOnsetThreshold"
                ),
            ),
            PropertyRestriction(
                property_iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#resolvesOpposition",
                quantifier=Quantifier.SOME,
                filler=IntersectionOf(
                    operands=(
                        NamedClass(
                            iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#ForceOpposition"
                        ),
                        PropertyRestriction(
                            property_iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#hasDrivingForce",
                            quantifier=Quantifier.SOME,
                            filler=PropertyRestriction(
                                property_iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#hasForceKind",
                                quantifier=Quantifier.VALUE,
                                value="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#Weight",
                            ),
                        ),
                        PropertyRestriction(
                            property_iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#hasResistingForce",
                            quantifier=Quantifier.SOME,
                            filler=PropertyRestriction(
                                property_iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#hasForceKind",
                                quantifier=Quantifier.VALUE,
                                value="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#Containment",
                            ),
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


@dataclass(eq=False)
class FlowOnsetThreshold(core.AngleParameter, core.MechanicalThreshold):
    """
    The tilt angle at which the substance begins crossing the lip.
    """

    term: ClassVar[ClassTerm] = ClassTerm(
        iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/pouring#FlowOnsetThreshold",
        linked_terms=(),
        evidence_status="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#EngineeringExtension",
        claims=(),
        restrictions=(),
        definition=None,
    )
    """
    The ontology class this class stands for.
    """


@dataclass(eq=False)
class FlowRateSetpoint(core.TaskSetpoint):
    """
    The desired mass or volume transferred per unit time, a task target and not a motor command.
    """

    term: ClassVar[ClassTerm] = ClassTerm(
        iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/pouring#FlowRateSetpoint",
        linked_terms=(),
        evidence_status="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#EngineeringExtension",
        claims=(),
        restrictions=(
            UnionOf(
                operands=(
                    NamedClass(
                        iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#MassFlowRateParameter"
                    ),
                    NamedClass(
                        iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#VolumetricFlowRateParameter"
                    ),
                )
            ),
        ),
        definition=None,
    )
    """
    The ontology class this class stands for.
    """


@dataclass(eq=False)
class FlowRegulationPhase(core.ActionPhase):
    """
    Continuous modulation of tilt and grip while substance flows, against the quantity reaching the target.
    """

    term: ClassVar[ClassTerm] = ClassTerm(
        iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/pouring#FlowRegulationPhase",
        linked_terms=(),
        evidence_status="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#EngineeringExtension",
        claims=(),
        restrictions=(
            PropertyRestriction(
                property_iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#controlledBy",
                quantifier=Quantifier.SOME,
                filler=IntersectionOf(
                    operands=(
                        NamedClass(
                            iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#FeedbackRegulatedController"
                        ),
                        PropertyRestriction(
                            property_iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#hasContinuousPrediction",
                            quantifier=Quantifier.SOME,
                            filler=NamedClass(
                                iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#ContinuousSignalPrediction"
                            ),
                        ),
                    )
                ),
            ),
            PropertyRestriction(
                property_iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#hasCouplingRegime",
                quantifier=Quantifier.VALUE,
                value="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#Coupled",
            ),
        ),
        definition=None,
    )
    """
    The ontology class this class stands for.
    """


@dataclass(eq=False)
class FlowTerminationPhase(core.ActionPhase):
    """
    Counter-rotation that brings the substance back below the lip. Ends at flow cessation.
    """

    term: ClassVar[ClassTerm] = ClassTerm(
        iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/pouring#FlowTerminationPhase",
        linked_terms=(),
        evidence_status="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#EngineeringExtension",
        claims=(),
        restrictions=(
            PropertyRestriction(
                property_iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#hasCouplingRegime",
                quantifier=Quantifier.VALUE,
                value="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#Coupled",
            ),
        ),
        definition=None,
    )
    """
    The ontology class this class stands for.
    """


@dataclass(eq=False)
class GranularFlowDynamicsEstimate(core.CategoricalParameter, core.DynamicsEstimate):
    """
    A categorical model of particle flow, jamming and avalanching.
    """

    term: ClassVar[ClassTerm] = ClassTerm(
        iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/pouring#GranularFlowDynamicsEstimate",
        linked_terms=(),
        evidence_status="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#EngineeringExtension",
        claims=(),
        restrictions=(),
        definition=None,
    )
    """
    The ontology class this class stands for.
    """


@dataclass(eq=False)
class GranularPouringStructure(PouringPhaseStructure):
    """
    A pouring structure transferring a granular material; it declares no viscosity, slosh or cohesive flow dynamics.
    """

    term: ClassVar[ClassTerm] = ClassTerm(
        iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/pouring#GranularPouringStructure",
        linked_terms=(),
        evidence_status="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#EngineeringExtension",
        claims=(),
        restrictions=(
            PropertyRestriction(
                property_iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#definesParameter",
                quantifier=Quantifier.ONLY,
                filler=ComplementOf(
                    operand=UnionOf(
                        operands=(
                            NamedClass(
                                iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/pouring#DynamicViscosityEstimate"
                            ),
                            NamedClass(
                                iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/pouring#SloshDynamicsEstimate"
                            ),
                            NamedClass(
                                iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/pouring#CohesiveFlowDynamicsEstimate"
                            ),
                        )
                    )
                ),
            ),
        ),
        definition=IntersectionOf(
            operands=(
                NamedClass(
                    iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/pouring#PouringPhaseStructure"
                ),
                PropertyRestriction(
                    property_iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/pouring#hasMaterialRegime",
                    quantifier=Quantifier.VALUE,
                    value="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/pouring#Granular",
                ),
            )
        ),
    )
    """
    The ontology class this class stands for.
    """


@dataclass(eq=False)
class InstructedStopPouringStructure(PouringPhaseStructure):
    """
    A pouring structure whose transfer stops on an instruction; it declares the external event that ends it.
    """

    term: ClassVar[ClassTerm] = ClassTerm(
        iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/pouring#InstructedStopPouringStructure",
        linked_terms=(),
        evidence_status="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#EngineeringExtension",
        claims=(),
        restrictions=(
            PropertyRestriction(
                property_iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#definesEventType",
                quantifier=Quantifier.SOME,
                filler=NamedClass(
                    iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#ExternalTerminationEventType"
                ),
            ),
        ),
        definition=IntersectionOf(
            operands=(
                NamedClass(
                    iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/pouring#PouringPhaseStructure"
                ),
                PropertyRestriction(
                    property_iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/pouring#hasTerminationReason",
                    quantifier=Quantifier.VALUE,
                    value="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/pouring#InstructedStop",
                ),
            )
        ),
    )
    """
    The ontology class this class stands for.
    """


@dataclass(eq=False)
class LipAligned(core.PoseAttainmentEventType):
    """
    The commanded pose placing the lip over the receptacle opening has been reached.
    """

    term: ClassVar[ClassTerm] = ClassTerm(
        iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/pouring#LipAligned",
        linked_terms=(),
        evidence_status="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#EngineeringExtension",
        claims=(),
        restrictions=(
            PropertyRestriction(
                property_iri="http://aicor.knowledge/l2-ontology.owl#initiatesStateType",
                quantifier=Quantifier.SOME,
                filler=NamedClass(
                    iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/pouring#LipOverReceptacle"
                ),
            ),
            PropertyRestriction(
                property_iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#attainsSetpoint",
                quantifier=Quantifier.SOME,
                filler=NamedClass(
                    iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/pouring#LipAlignmentSetpoint"
                ),
            ),
        ),
        definition=None,
    )
    """
    The ontology class this class stands for.
    """


@dataclass(eq=False)
class LipAlignmentSetpoint(core.AngleParameter, core.MotorSetpoint):
    """
    The commanded angular alignment of the pouring lip with the receptacle.
    """

    term: ClassVar[ClassTerm] = ClassTerm(
        iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/pouring#LipAlignmentSetpoint",
        linked_terms=(),
        evidence_status="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#EngineeringExtension",
        claims=(),
        restrictions=(),
        definition=None,
    )
    """
    The ontology class this class stands for.
    """


@dataclass(eq=False)
class LipOverReceptacle(core.PhaseConditionType):
    """
    The pouring lip is positioned over the opening of the target receptacle.
    """

    term: ClassVar[ClassTerm] = ClassTerm(
        iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/pouring#LipOverReceptacle",
        linked_terms=(),
        evidence_status="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#EngineeringExtension",
        claims=(),
        restrictions=(),
        definition=None,
    )
    """
    The ontology class this class stands for.
    """


@dataclass(eq=False)
class LiquidPouringStructure(PouringPhaseStructure):
    """
    A pouring structure transferring a liquid; it declares no granular or cohesive flow dynamics.
    """

    term: ClassVar[ClassTerm] = ClassTerm(
        iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/pouring#LiquidPouringStructure",
        linked_terms=(),
        evidence_status="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#EngineeringExtension",
        claims=(),
        restrictions=(
            PropertyRestriction(
                property_iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#definesParameter",
                quantifier=Quantifier.ONLY,
                filler=ComplementOf(
                    operand=UnionOf(
                        operands=(
                            NamedClass(
                                iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/pouring#GranularFlowDynamicsEstimate"
                            ),
                            NamedClass(
                                iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/pouring#CohesiveFlowDynamicsEstimate"
                            ),
                        )
                    )
                ),
            ),
        ),
        definition=IntersectionOf(
            operands=(
                NamedClass(
                    iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/pouring#PouringPhaseStructure"
                ),
                PropertyRestriction(
                    property_iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/pouring#hasMaterialRegime",
                    quantifier=Quantifier.VALUE,
                    value="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/pouring#Liquid",
                ),
            )
        ),
    )
    """
    The ontology class this class stands for.
    """


@dataclass(eq=False)
class PourPoseSetpoint(core.CategoricalParameter, core.MotorSetpoint):
    """
    A categorical reference to the desired pouring pose.
    """

    term: ClassVar[ClassTerm] = ClassTerm(
        iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/pouring#PourPoseSetpoint",
        linked_terms=(),
        evidence_status="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#EngineeringExtension",
        claims=(),
        restrictions=(),
        definition=None,
    )
    """
    The ontology class this class stands for.
    """


@dataclass(eq=False)
class PouredSubstance(core.ManipulationRole):
    """
    The substance transferred, as distinct from the container that is manipulated.
    """

    term: ClassVar[ClassTerm] = ClassTerm(
        iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/pouring#PouredSubstance",
        linked_terms=("http://aicor.knowledge/l2-ontology.owl#SubstanceRole",),
        evidence_status="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#EngineeringExtension",
        claims=(),
        restrictions=(),
        definition=None,
    )
    """
    The ontology class this class stands for.
    """


@dataclass(eq=False)
class PouredSubstanceBinding(DescriptionEntity):
    """
    An execution-time binding of the poured substance slot, whose filler is a CRAM-AICOR substance.
    """

    term: ClassVar[ClassTerm] = ClassTerm(
        iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/pouring#PouredSubstanceBinding",
        linked_terms=("http://aicor.knowledge/l2-ontology.owl#RoleBinding",),
        evidence_status="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#EngineeringExtension",
        claims=(),
        restrictions=(
            PropertyRestriction(
                property_iri="http://aicor.knowledge/l2-ontology.owl#bindsFiller",
                quantifier=Quantifier.ONLY,
                filler=NamedClass(
                    iri="http://aicor.knowledge/l2-ontology.owl#Substance"
                ),
            ),
        ),
        definition=IntersectionOf(
            operands=(
                NamedClass(iri="http://aicor.knowledge/l2-ontology.owl#RoleBinding"),
                PropertyRestriction(
                    property_iri="http://aicor.knowledge/l2-ontology.owl#bindsSlot",
                    quantifier=Quantifier.SOME,
                    filler=NamedClass(
                        iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/pouring#PouredSubstance"
                    ),
                ),
            )
        ),
    )
    """
    The ontology class this class stands for.
    """


@dataclass(eq=False)
class PouringLip(core.ManipulationRole):
    """
    The edge of the source container over which the substance leaves it.
    """

    term: ClassVar[ClassTerm] = ClassTerm(
        iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/pouring#PouringLip",
        linked_terms=(),
        evidence_status="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#EngineeringExtension",
        claims=(),
        restrictions=(),
        definition=None,
    )
    """
    The ontology class this class stands for.
    """


@dataclass(eq=False)
class RemainingSourceQuantityEstimate(core.ObjectStateEstimate):
    """
    The anticipated mass or volume remaining in the source.
    """

    term: ClassVar[ClassTerm] = ClassTerm(
        iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/pouring#RemainingSourceQuantityEstimate",
        linked_terms=(),
        evidence_status="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#EngineeringExtension",
        claims=(),
        restrictions=(
            UnionOf(
                operands=(
                    NamedClass(
                        iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#MassParameter"
                    ),
                    NamedClass(
                        iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#VolumeParameter"
                    ),
                )
            ),
        ),
        definition=None,
    )
    """
    The ontology class this class stands for.
    """


@dataclass(eq=False)
class ReturnUprightPhase(core.ActionPhase):
    """
    Restoration of the container to a stable orientation after the flow has stopped. Optional.
    """

    term: ClassVar[ClassTerm] = ClassTerm(
        iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/pouring#ReturnUprightPhase",
        linked_terms=(),
        evidence_status="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#EngineeringExtension",
        claims=(),
        restrictions=(
            PropertyRestriction(
                property_iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#hasCouplingRegime",
                quantifier=Quantifier.VALUE,
                value="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#Coupled",
            ),
        ),
        definition=None,
    )
    """
    The ontology class this class stands for.
    """


@dataclass(eq=False)
class SloshDynamicsEstimate(core.CategoricalParameter, core.DynamicsEstimate):
    """
    A categorical model of internal liquid motion under acceleration.
    """

    term: ClassVar[ClassTerm] = ClassTerm(
        iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/pouring#SloshDynamicsEstimate",
        linked_terms=(),
        evidence_status="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#EngineeringExtension",
        claims=(),
        restrictions=(),
        definition=None,
    )
    """
    The ontology class this class stands for.
    """


@dataclass(eq=False)
class SourceContainer(prehension.PrehensileManipulandum):
    """
    The vessel that is grasped and tilted, from which the substance flows.
    """

    term: ClassVar[ClassTerm] = ClassTerm(
        iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/pouring#SourceContainer",
        linked_terms=("http://aicor.knowledge/l2-ontology.owl#SourceRole",),
        evidence_status="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#EngineeringExtension",
        claims=(),
        restrictions=(),
        definition=None,
    )
    """
    The ontology class this class stands for.
    """


@dataclass(eq=False)
class SourceContainerBinding(DescriptionEntity):
    """
    An execution-time binding of the source container slot, whose filler is a CRAM-AICOR container.
    """

    term: ClassVar[ClassTerm] = ClassTerm(
        iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/pouring#SourceContainerBinding",
        linked_terms=("http://aicor.knowledge/l2-ontology.owl#RoleBinding",),
        evidence_status="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#EngineeringExtension",
        claims=(),
        restrictions=(
            PropertyRestriction(
                property_iri="http://aicor.knowledge/l2-ontology.owl#bindsFiller",
                quantifier=Quantifier.ONLY,
                filler=NamedClass(
                    iri="http://aicor.knowledge/l2-ontology.owl#Container"
                ),
            ),
        ),
        definition=IntersectionOf(
            operands=(
                NamedClass(iri="http://aicor.knowledge/l2-ontology.owl#RoleBinding"),
                PropertyRestriction(
                    property_iri="http://aicor.knowledge/l2-ontology.owl#bindsSlot",
                    quantifier=Quantifier.SOME,
                    filler=NamedClass(
                        iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/pouring#SourceContainer"
                    ),
                ),
            )
        ),
    )
    """
    The ontology class this class stands for.
    """


@dataclass(eq=False)
class SourceEmpty(core.PhaseBoundaryEventType):
    """
    The transferable quantity remaining in the source reaches zero.
    """

    term: ClassVar[ClassTerm] = ClassTerm(
        iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/pouring#SourceEmpty",
        linked_terms=(),
        evidence_status="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#EngineeringExtension",
        claims=(),
        restrictions=(
            PropertyRestriction(
                property_iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#terminatesStateType",
                quantifier=Quantifier.SOME,
                filler=NamedClass(
                    iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/pouring#SubstanceContainedInSource"
                ),
            ),
        ),
        definition=None,
    )
    """
    The ontology class this class stands for.
    """


@dataclass(eq=False)
class SourceExhaustedPouringStructure(PouringPhaseStructure):
    """
    A pouring structure whose transfer stops once the source is empty; it declares the event detecting that.
    """

    term: ClassVar[ClassTerm] = ClassTerm(
        iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/pouring#SourceExhaustedPouringStructure",
        linked_terms=(),
        evidence_status="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#EngineeringExtension",
        claims=(),
        restrictions=(
            PropertyRestriction(
                property_iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#definesEventType",
                quantifier=Quantifier.SOME,
                filler=NamedClass(
                    iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/pouring#SourceEmpty"
                ),
            ),
        ),
        definition=IntersectionOf(
            operands=(
                NamedClass(
                    iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/pouring#PouringPhaseStructure"
                ),
                PropertyRestriction(
                    property_iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/pouring#hasTerminationReason",
                    quantifier=Quantifier.VALUE,
                    value="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/pouring#SourceExhausted",
                ),
            )
        ),
    )
    """
    The ontology class this class stands for.
    """


@dataclass(eq=False)
class SourceUpright(core.PhaseConditionType):
    """
    The source container stands in a stable upright orientation.
    """

    term: ClassVar[ClassTerm] = ClassTerm(
        iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/pouring#SourceUpright",
        linked_terms=(),
        evidence_status="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#EngineeringExtension",
        claims=(),
        restrictions=(),
        definition=None,
    )
    """
    The ontology class this class stands for.
    """


@dataclass(eq=False)
class SpillCorrection(core.CorrectiveAction):
    """
    Reversing the tilt to bring the substance back below the lip after a spill or overly fast flow.
    """

    term: ClassVar[ClassTerm] = ClassTerm(
        iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/pouring#SpillCorrection",
        linked_terms=(),
        evidence_status="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#EngineeringExtension",
        claims=(),
        restrictions=(
            PropertyRestriction(
                property_iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#hasCorrectionDirection",
                quantifier=Quantifier.VALUE,
                value="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#ReverseChange",
            ),
            PropertyRestriction(
                property_iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#targetsSetpoint",
                quantifier=Quantifier.ONLY,
                filler=UnionOf(
                    operands=(
                        NamedClass(
                            iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/pouring#TiltAngleSetpoint"
                        ),
                        NamedClass(
                            iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/pouring#TiltRateSetpoint"
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


@dataclass(eq=False)
class SpillEvent(core.FailureEventType, FlowEventType, core.ThresholdCrossingEventType):
    """
    Substance leaves the intended flow path outside the target receptacle.
    """

    term: ClassVar[ClassTerm] = ClassTerm(
        iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/pouring#SpillEvent",
        linked_terms=(),
        evidence_status="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#EngineeringExtension",
        claims=(),
        restrictions=(
            PropertyRestriction(
                property_iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#crossesThreshold",
                quantifier=Quantifier.SOME,
                filler=NamedClass(
                    iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/pouring#SpillThreshold"
                ),
            ),
        ),
        definition=None,
    )
    """
    The ontology class this class stands for.
    """


@dataclass(eq=False)
class SpillThreshold(core.MechanicalThreshold):
    """
    The flow rate beyond which the substance is no longer contained by the target.
    """

    term: ClassVar[ClassTerm] = ClassTerm(
        iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/pouring#SpillThreshold",
        linked_terms=(),
        evidence_status="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#EngineeringExtension",
        claims=(),
        restrictions=(
            UnionOf(
                operands=(
                    NamedClass(
                        iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#MassFlowRateParameter"
                    ),
                    NamedClass(
                        iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#VolumetricFlowRateParameter"
                    ),
                )
            ),
        ),
        definition=None,
    )
    """
    The ontology class this class stands for.
    """


@dataclass(eq=False)
class SubstanceAccumulatedInTarget(core.PhaseConditionType):
    """
    Transferred substance has accumulated in the target receptacle.
    """

    term: ClassVar[ClassTerm] = ClassTerm(
        iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/pouring#SubstanceAccumulatedInTarget",
        linked_terms=("http://aicor.knowledge/l2-ontology.owl#AccumulatedState",),
        evidence_status="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#EngineeringExtension",
        claims=(),
        restrictions=(),
        definition=None,
    )
    """
    The ontology class this class stands for.
    """


@dataclass(eq=False)
class SubstanceContainedInSource(core.PhaseConditionType):
    """
    Transferable substance remains inside the source container.
    """

    term: ClassVar[ClassTerm] = ClassTerm(
        iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/pouring#SubstanceContainedInSource",
        linked_terms=("http://aicor.knowledge/l2-ontology.owl#ContainedState",),
        evidence_status="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#EngineeringExtension",
        claims=(),
        restrictions=(),
        definition=None,
    )
    """
    The ontology class this class stands for.
    """


@dataclass(eq=False)
class SubstanceFlowing(core.PhaseConditionType):
    """
    Substance is crossing the pouring lip toward the target.
    """

    term: ClassVar[ClassTerm] = ClassTerm(
        iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/pouring#SubstanceFlowing",
        linked_terms=("http://aicor.knowledge/l2-ontology.owl#FlowingState",),
        evidence_status="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#EngineeringExtension",
        claims=(),
        restrictions=(),
        definition=None,
    )
    """
    The ontology class this class stands for.
    """


@dataclass(eq=False)
class TargetFillStateEstimate(core.DimensionlessParameter, core.ObjectStateEstimate):
    """
    The anticipated filled fraction of the target receptacle.
    """

    term: ClassVar[ClassTerm] = ClassTerm(
        iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/pouring#TargetFillStateEstimate",
        linked_terms=(),
        evidence_status="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#EngineeringExtension",
        claims=(),
        restrictions=(),
        definition=None,
    )
    """
    The ontology class this class stands for.
    """


@dataclass(eq=False)
class TargetQuantityAttainedPouringStructure(PouringPhaseStructure):
    """
    A pouring structure whose transfer stops once the target quantity is reached; it declares the event detecting that.
    """

    term: ClassVar[ClassTerm] = ClassTerm(
        iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/pouring#TargetQuantityAttainedPouringStructure",
        linked_terms=(),
        evidence_status="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#EngineeringExtension",
        claims=(),
        restrictions=(
            PropertyRestriction(
                property_iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#definesEventType",
                quantifier=Quantifier.SOME,
                filler=NamedClass(
                    iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/pouring#TargetQuantityReached"
                ),
            ),
        ),
        definition=IntersectionOf(
            operands=(
                NamedClass(
                    iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/pouring#PouringPhaseStructure"
                ),
                PropertyRestriction(
                    property_iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/pouring#hasTerminationReason",
                    quantifier=Quantifier.VALUE,
                    value="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/pouring#TargetQuantityAttained",
                ),
            )
        ),
    )
    """
    The ontology class this class stands for.
    """


@dataclass(eq=False)
class TargetQuantityReached(core.SetpointAttainmentEventType):
    """
    The transferred mass or volume meets the requested task quantity.
    """

    term: ClassVar[ClassTerm] = ClassTerm(
        iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/pouring#TargetQuantityReached",
        linked_terms=(),
        evidence_status="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#EngineeringExtension",
        claims=(),
        restrictions=(
            PropertyRestriction(
                property_iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#attainsSetpoint",
                quantifier=Quantifier.SOME,
                filler=NamedClass(
                    iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/pouring#TransferredQuantitySetpoint"
                ),
            ),
        ),
        definition=None,
    )
    """
    The ontology class this class stands for.
    """


@dataclass(eq=False)
class TargetReceptacle(core.ManipulationRole):
    """
    The vessel or surface receiving the substance.
    """

    term: ClassVar[ClassTerm] = ClassTerm(
        iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/pouring#TargetReceptacle",
        linked_terms=("http://aicor.knowledge/l2-ontology.owl#GoalRole",),
        evidence_status="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#EngineeringExtension",
        claims=(),
        restrictions=(),
        definition=None,
    )
    """
    The ontology class this class stands for.
    """


@dataclass(eq=False)
class TiltAngleSetpoint(core.AngleParameter, core.MotorSetpoint):
    """
    The commanded orientation of the source container about its pouring axis.
    """

    term: ClassVar[ClassTerm] = ClassTerm(
        iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/pouring#TiltAngleSetpoint",
        linked_terms=(),
        evidence_status="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#EngineeringExtension",
        claims=(),
        restrictions=(),
        definition=None,
    )
    """
    The ontology class this class stands for.
    """


@dataclass(eq=False)
class TiltRateSetpoint(core.AngularRateParameter, core.MotorSetpoint):
    """
    The commanded angular velocity of tilting the source container.
    """

    term: ClassVar[ClassTerm] = ClassTerm(
        iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/pouring#TiltRateSetpoint",
        linked_terms=(),
        evidence_status="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#EngineeringExtension",
        claims=(),
        restrictions=(),
        definition=None,
    )
    """
    The ontology class this class stands for.
    """


@dataclass(eq=False)
class TiltToFlowPhase(core.ActionPhase):
    """
    Rotation of the source container until flow begins.
    """

    term: ClassVar[ClassTerm] = ClassTerm(
        iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/pouring#TiltToFlowPhase",
        linked_terms=("http://aicor.knowledge/l2-ontology.owl#PouringPhase",),
        evidence_status="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#EngineeringExtension",
        claims=(),
        restrictions=(
            PropertyRestriction(
                property_iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#hasCouplingRegime",
                quantifier=Quantifier.VALUE,
                value="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#Coupled",
            ),
            PropertyRestriction(
                property_iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#realizesActionCategory",
                quantifier=Quantifier.VALUE,
                value="http://aicor.knowledge/l2-ontology.owl#Tilting",
            ),
        ),
        definition=None,
    )
    """
    The ontology class this class stands for.
    """


@dataclass(eq=False)
class TorqueSetpoint(core.MotorSetpoint, core.TorqueParameter):
    """
    The commanded moment about the grasp.
    """

    term: ClassVar[ClassTerm] = ClassTerm(
        iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/pouring#TorqueSetpoint",
        linked_terms=(),
        evidence_status="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#EngineeringExtension",
        claims=(),
        restrictions=(),
        definition=None,
    )
    """
    The ontology class this class stands for.
    """


@dataclass(eq=False)
class TransferFailurePouringStructure(PouringPhaseStructure):
    """
    A pouring structure whose transfer stops because it fails; it declares the failure event that ends it.
    """

    term: ClassVar[ClassTerm] = ClassTerm(
        iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/pouring#TransferFailurePouringStructure",
        linked_terms=(),
        evidence_status="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#EngineeringExtension",
        claims=(),
        restrictions=(
            PropertyRestriction(
                property_iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#definesEventType",
                quantifier=Quantifier.SOME,
                filler=NamedClass(
                    iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#FailureEventType"
                ),
            ),
        ),
        definition=IntersectionOf(
            operands=(
                NamedClass(
                    iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/pouring#PouringPhaseStructure"
                ),
                PropertyRestriction(
                    property_iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/pouring#hasTerminationReason",
                    quantifier=Quantifier.VALUE,
                    value="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/pouring#TransferFailure",
                ),
            )
        ),
    )
    """
    The ontology class this class stands for.
    """


@dataclass(eq=False)
class TransferredQuantitySetpoint(core.TaskSetpoint):
    """
    The desired mass or volume of substance to transfer, a task target and not a motor command.
    """

    term: ClassVar[ClassTerm] = ClassTerm(
        iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/pouring#TransferredQuantitySetpoint",
        linked_terms=(),
        evidence_status="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#EngineeringExtension",
        claims=(),
        restrictions=(
            UnionOf(
                operands=(
                    NamedClass(
                        iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#MassParameter"
                    ),
                    NamedClass(
                        iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#VolumeParameter"
                    ),
                )
            ),
        ),
        definition=None,
    )
    """
    The ontology class this class stands for.
    """


@dataclass(eq=False)
class UprightAngleSetpoint(core.AngleParameter, core.MotorSetpoint):
    """
    The commanded upright orientation of the source container.
    """

    term: ClassVar[ClassTerm] = ClassTerm(
        iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/pouring#UprightAngleSetpoint",
        linked_terms=(),
        evidence_status="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#EngineeringExtension",
        claims=(),
        restrictions=(),
        definition=None,
    )
    """
    The ontology class this class stands for.
    """


@dataclass(eq=False)
class UprightRestored(core.PoseAttainmentEventType):
    """
    The source container has been returned to a stable orientation.
    """

    term: ClassVar[ClassTerm] = ClassTerm(
        iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/pouring#UprightRestored",
        linked_terms=(),
        evidence_status="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#EngineeringExtension",
        claims=(),
        restrictions=(
            PropertyRestriction(
                property_iri="http://aicor.knowledge/l2-ontology.owl#initiatesStateType",
                quantifier=Quantifier.SOME,
                filler=NamedClass(
                    iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/pouring#SourceUpright"
                ),
            ),
            PropertyRestriction(
                property_iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#attainsSetpoint",
                quantifier=Quantifier.SOME,
                filler=NamedClass(
                    iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/pouring#UprightAngleSetpoint"
                ),
            ),
        ),
        definition=None,
    )
    """
    The ontology class this class stands for.
    """
