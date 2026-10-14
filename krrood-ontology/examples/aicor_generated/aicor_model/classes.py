"""Structural ontology classes; generated fields do not validate OWL constraints."""

from __future__ import annotations

from dataclasses import dataclass as _dataclass
from dataclasses import field as _field
from decimal import Decimal as _Decimal
from typing import ClassVar as _ClassVar
from typing import Any as _Any

from krrood.entity_query_language.predicate import Symbol as _KrroodSymbol


@_dataclass(eq=False, kw_only=True)
class _OntologyEntity(_KrroodSymbol):
    """Private storage base for shared native relation associations."""

    co_refers_with: set[_OntologyEntity] = _field(
        default_factory=set,
        init=False,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#coRefersWith",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    force_dynamically_prior_to: set[_OntologyEntity] = _field(
        default_factory=set,
        init=False,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#forceDynamicallyPriorTo",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    has_precondition: set[_OntologyEntity] = _field(
        default_factory=set,
        init=False,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#hasPrecondition",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    has_subaction: set[_OntologyEntity] = _field(
        default_factory=set,
        init=False,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#hasSubaction",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    part_of: set[_OntologyEntity] = _field(
        default_factory=set,
        init=False,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#partOf",
            "ontology_range": "",
            "property_kind": "object",
        },
    )


@_dataclass(eq=False, kw_only=True)
class Abstain(_OntologyEntity):
    """Structural representation of Abstain."""

    __ontology_iri__: _ClassVar[str] = "http://aicor.knowledge/l2-ontology.owl#Abstain"


@_dataclass(eq=False, kw_only=True)
class Abstention(_OntologyEntity):
    """Structural representation of Abstention."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#Abstention"
    )


@_dataclass(eq=False, kw_only=True)
class Entity(_OntologyEntity):
    """Structural representation of Entity."""

    __ontology_iri__: _ClassVar[str] = "http://aicor.knowledge/l2-ontology.owl#Entity"


@_dataclass(eq=False, kw_only=True)
class AccountabilityQuestion(_OntologyEntity):
    """Structural representation of AccountabilityQuestion."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#AccountabilityQuestion"
    )


@_dataclass(eq=False, kw_only=True)
class AccountableDimension(_OntologyEntity):
    """Structural representation of AccountableDimension."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#AccountableDimension"
    )


@_dataclass(eq=False, kw_only=True)
class FailureReason(_OntologyEntity):
    """Structural representation of FailureReason."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#FailureReason"
    )


@_dataclass(eq=False, kw_only=True)
class Action(_OntologyEntity):
    """Structural representation of Action."""

    __ontology_iri__: _ClassVar[str] = "http://aicor.knowledge/l2-ontology.owl#Action"


@_dataclass(eq=False, kw_only=True)
class ActionParameterSpace(_OntologyEntity):
    """Structural representation of ActionParameterSpace."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#ActionParameterSpace"
    )


@_dataclass(eq=False, kw_only=True)
class DesignatorTarget(_OntologyEntity):
    """Structural representation of DesignatorTarget."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#DesignatorTarget"
    )


@_dataclass(eq=False, kw_only=True)
class ActionRelatedForceCorridor(_OntologyEntity):
    """Structural representation of ActionRelatedForceCorridor."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#ActionRelatedForceCorridor"
    )


@_dataclass(eq=False, kw_only=True)
class ActionRelatedGrasp(_OntologyEntity):
    """Structural representation of ActionRelatedGrasp."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#ActionRelatedGrasp"
    )


@_dataclass(eq=False, kw_only=True)
class ActionRelatedMotion(_OntologyEntity):
    """Structural representation of ActionRelatedMotion."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#ActionRelatedMotion"
    )


@_dataclass(eq=False, kw_only=True)
class ActionRelatedPlace(_OntologyEntity):
    """Structural representation of ActionRelatedPlace."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#ActionRelatedPlace"
    )


@_dataclass(eq=False, kw_only=True)
class ActionRelatedPour(_OntologyEntity):
    """Structural representation of ActionRelatedPour."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#ActionRelatedPour"
    )


@_dataclass(eq=False, kw_only=True)
class ActionToken(_OntologyEntity):
    """Structural representation of ActionToken."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#ActionToken"
    )


@_dataclass(eq=False, kw_only=True)
class CommitmentTarget(_OntologyEntity):
    """Structural representation of CommitmentTarget."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#CommitmentTarget"
    )


@_dataclass(eq=False, kw_only=True)
class ActivityEpisode(_OntologyEntity):
    """Structural representation of ActivityEpisode."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#ActivityEpisode"
    )


@_dataclass(eq=False, kw_only=True)
class ActuationNuisance(_OntologyEntity):
    """Structural representation of ActuationNuisance."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#ActuationNuisance"
    )


@_dataclass(eq=False, kw_only=True)
class ActuatorInput(_OntologyEntity):
    """Structural representation of ActuatorInput."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#ActuatorInput"
    )


@_dataclass(eq=False, kw_only=True)
class AddSupport(_OntologyEntity):
    """Structural representation of AddSupport."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#AddSupport"
    )


@_dataclass(eq=False, kw_only=True)
class AffectsExecution(_OntologyEntity):
    """Structural representation of AffectsExecution."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#AffectsExecution"
    )


@_dataclass(eq=False, kw_only=True)
class AffordanceEstimation(_OntologyEntity):
    """Structural representation of AffordanceEstimation."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#AffordanceEstimation"
    )


@_dataclass(eq=False, kw_only=True)
class ParticipantRole(_OntologyEntity):
    """Structural representation of ParticipantRole."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#ParticipantRole"
    )


@_dataclass(eq=False, kw_only=True)
class Appearance(_OntologyEntity):
    """Structural representation of Appearance."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#Appearance"
    )


@_dataclass(eq=False, kw_only=True)
class ArrangementRole(_OntologyEntity):
    """Structural representation of ArrangementRole."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#ArrangementRole"
    )


@_dataclass(eq=False, kw_only=True)
class AssessmentAndClaim(_OntologyEntity):
    """Structural representation of AssessmentAndClaim."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#AssessmentAndClaim"
    )


@_dataclass(eq=False, kw_only=True)
class BodyDynamics(_OntologyEntity):
    """Structural representation of BodyDynamics."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#BodyDynamics"
    )


@_dataclass(eq=False, kw_only=True)
class DulEvent(_OntologyEntity):
    """Structural representation of DulEvent."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#dul_Event"
    )


@_dataclass(eq=False, kw_only=True)
class BodyState(_OntologyEntity):
    """Structural representation of BodyState."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#BodyState"
    )


@_dataclass(eq=False, kw_only=True)
class BodyStateSpace(_OntologyEntity):
    """Structural representation of BodyStateSpace."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#BodyStateSpace"
    )


@_dataclass(eq=False, kw_only=True)
class DulProcess(_OntologyEntity):
    """Structural representation of DulProcess."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#dul_Process"
    )


@_dataclass(eq=False, kw_only=True)
class SomaAction(_OntologyEntity):
    """Structural representation of SomaAction."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#soma_Action"
    )


@_dataclass(eq=False, kw_only=True)
class CandidateSet(_OntologyEntity):
    """Structural representation of CandidateSet."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#CandidateSet"
    )


@_dataclass(eq=False, kw_only=True)
class DulEntity(_OntologyEntity):
    """Structural representation of DulEntity."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#dul_Entity"
    )


@_dataclass(eq=False, kw_only=True)
class CausalEffectiveness(_OntologyEntity):
    """Structural representation of CausalEffectiveness."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#CausalEffectiveness"
    )


@_dataclass(eq=False, kw_only=True)
class Check(_OntologyEntity):
    """Structural representation of Check."""

    __ontology_iri__: _ClassVar[str] = "http://aicor.knowledge/l2-ontology.owl#Check"


@_dataclass(eq=False, kw_only=True)
class Clarify(_OntologyEntity):
    """Structural representation of Clarify."""

    __ontology_iri__: _ClassVar[str] = "http://aicor.knowledge/l2-ontology.owl#Clarify"


@_dataclass(eq=False, kw_only=True)
class CodeReplace(_OntologyEntity):
    """Structural representation of CodeReplace."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#CodeReplace"
    )


@_dataclass(eq=False, kw_only=True)
class Commit(_OntologyEntity):
    """Structural representation of Commit."""

    __ontology_iri__: _ClassVar[str] = "http://aicor.knowledge/l2-ontology.owl#Commit"


@_dataclass(eq=False, kw_only=True)
class CommitmentCriterion(_OntologyEntity):
    """Structural representation of CommitmentCriterion."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#CommitmentCriterion"
    )


@_dataclass(eq=False, kw_only=True)
class CommitmentRelevance(_OntologyEntity):
    """Structural representation of CommitmentRelevance."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#CommitmentRelevance"
    )


@_dataclass(eq=False, kw_only=True)
class CommitmentRelevantStructure(_OntologyEntity):
    """Structural representation of CommitmentRelevantStructure."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#CommitmentRelevantStructure"
    )


@_dataclass(eq=False, kw_only=True)
class DulQuality(_OntologyEntity):
    """Structural representation of DulQuality."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#dul_Quality"
    )


@_dataclass(eq=False, kw_only=True)
class ConceptFamilyEnum(_OntologyEntity):
    """Structural representation of ConceptFamilyEnum."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#ConceptFamilyEnum"
    )


@_dataclass(eq=False, kw_only=True)
class Configuration(_OntologyEntity):
    """Structural representation of Configuration."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#Configuration"
    )


@_dataclass(eq=False, kw_only=True)
class CouplingType(_OntologyEntity):
    """Structural representation of CouplingType."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#CouplingType"
    )


@_dataclass(eq=False, kw_only=True)
class ConsumableScope(_OntologyEntity):
    """Structural representation of ConsumableScope."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#ConsumableScope"
    )


@_dataclass(eq=False, kw_only=True)
class Continuous(_OntologyEntity):
    """Structural representation of Continuous."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#Continuous"
    )


@_dataclass(eq=False, kw_only=True)
class ContinuousState(_OntologyEntity):
    """Structural representation of ContinuousState."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#ContinuousState"
    )


@_dataclass(eq=False, kw_only=True)
class Continuum(_OntologyEntity):
    """Structural representation of Continuum."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#Continuum"
    )


@_dataclass(eq=False, kw_only=True)
class ContributesTraceEvidence(_OntologyEntity):
    """Structural representation of ContributesTraceEvidence."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#ContributesTraceEvidence"
    )


@_dataclass(eq=False, kw_only=True)
class ControlCandidate(_OntologyEntity):
    """Structural representation of ControlCandidate."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#ControlCandidate"
    )


@_dataclass(eq=False, kw_only=True)
class ControlCounterpart(_OntologyEntity):
    """Structural representation of ControlCounterpart."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#ControlCounterpart"
    )


@_dataclass(eq=False, kw_only=True)
class ControlCycle(_OntologyEntity):
    """Structural representation of ControlCycle."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#ControlCycle"
    )


@_dataclass(eq=False, kw_only=True)
class ControlSequence(_OntologyEntity):
    """Structural representation of ControlSequence."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#ControlSequence"
    )


@_dataclass(eq=False, kw_only=True)
class CouplingRelation(_OntologyEntity):
    """Structural representation of CouplingRelation."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#CouplingRelation"
    )


@_dataclass(eq=False, kw_only=True)
class CrackOrientation(_OntologyEntity):
    """Structural representation of CrackOrientation."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#CrackOrientation"
    )


@_dataclass(eq=False, kw_only=True)
class CriterionRelative(_OntologyEntity):
    """Structural representation of CriterionRelative."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#CriterionRelative"
    )


@_dataclass(eq=False, kw_only=True)
class CriticHook(_OntologyEntity):
    """Structural representation of CriticHook."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#CriticHook"
    )


@_dataclass(eq=False, kw_only=True)
class DesignatorResolution(_OntologyEntity):
    """Structural representation of DesignatorResolution."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#DesignatorResolution"
    )


@_dataclass(eq=False, kw_only=True)
class DegreeOfFreedom(_OntologyEntity):
    """Structural representation of DegreeOfFreedom."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#DegreeOfFreedom"
    )


@_dataclass(eq=False, kw_only=True)
class Deictic(_OntologyEntity):
    """Structural representation of Deictic."""

    __ontology_iri__: _ClassVar[str] = "http://aicor.knowledge/l2-ontology.owl#Deictic"


@_dataclass(eq=False, kw_only=True)
class Descriptive(_OntologyEntity):
    """Structural representation of Descriptive."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#Descriptive"
    )


@_dataclass(eq=False, kw_only=True)
class DesignatorReferent(_OntologyEntity):
    """Structural representation of DesignatorReferent."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#DesignatorReferent"
    )


@_dataclass(eq=False, kw_only=True)
class Destination(_OntologyEntity):
    """Structural representation of Destination."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#Destination"
    )


@_dataclass(eq=False, kw_only=True)
class DiagnosisCounterpart(_OntologyEntity):
    """Structural representation of DiagnosisCounterpart."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#DiagnosisCounterpart"
    )


@_dataclass(eq=False, kw_only=True)
class DiagnosticForce(_OntologyEntity):
    """Structural representation of DiagnosticForce."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#DiagnosticForce"
    )


@_dataclass(eq=False, kw_only=True)
class DistributionalPlausibility(_OntologyEntity):
    """Structural representation of DistributionalPlausibility."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#DistributionalPlausibility"
    )


@_dataclass(eq=False, kw_only=True)
class DistributionalPlausible(_OntologyEntity):
    """Structural representation of DistributionalPlausible."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#DistributionalPlausible"
    )


@_dataclass(eq=False, kw_only=True)
class DisturbanceNuisance(_OntologyEntity):
    """Structural representation of DisturbanceNuisance."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#DisturbanceNuisance"
    )


@_dataclass(eq=False, kw_only=True)
class DownCrack(_OntologyEntity):
    """Structural representation of DownCrack."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#DownCrack"
    )


@_dataclass(eq=False, kw_only=True)
class Drop(_OntologyEntity):
    """Structural representation of Drop."""

    __ontology_iri__: _ClassVar[str] = "http://aicor.knowledge/l2-ontology.owl#Drop"


@_dataclass(eq=False, kw_only=True)
class Dynamic(_OntologyEntity):
    """Structural representation of Dynamic."""

    __ontology_iri__: _ClassVar[str] = "http://aicor.knowledge/l2-ontology.owl#Dynamic"


@_dataclass(eq=False, kw_only=True)
class Effect(_OntologyEntity):
    """Structural representation of Effect."""

    __ontology_iri__: _ClassVar[str] = "http://aicor.knowledge/l2-ontology.owl#Effect"


@_dataclass(eq=False, kw_only=True)
class Embodiment(_OntologyEntity):
    """Structural representation of Embodiment."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#Embodiment"
    )


@_dataclass(eq=False, kw_only=True)
class Environment(_OntologyEntity):
    """Structural representation of Environment."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#Environment"
    )


@_dataclass(eq=False, kw_only=True)
class EnvironmentModel(_OntologyEntity):
    """Structural representation of EnvironmentModel."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#EnvironmentModel"
    )


@_dataclass(eq=False, kw_only=True)
class EpisodeAndTrace(_OntologyEntity):
    """Structural representation of EpisodeAndTrace."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#EpisodeAndTrace"
    )


@_dataclass(eq=False, kw_only=True)
class EverydayTerm(_OntologyEntity):
    """Structural representation of EverydayTerm."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#EverydayTerm"
    )


@_dataclass(eq=False, kw_only=True)
class EvidenceCounterpart(_OntologyEntity):
    """Structural representation of EvidenceCounterpart."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#EvidenceCounterpart"
    )


@_dataclass(eq=False, kw_only=True)
class EvidenceLogging(_OntologyEntity):
    """Structural representation of EvidenceLogging."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#EvidenceLogging"
    )


@_dataclass(eq=False, kw_only=True)
class EvidenceRequirement(_OntologyEntity):
    """Structural representation of EvidenceRequirement."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#EvidenceRequirement"
    )


@_dataclass(eq=False, kw_only=True)
class EvidentialSufficiency(_OntologyEntity):
    """Structural representation of EvidentialSufficiency."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#EvidentialSufficiency"
    )


@_dataclass(eq=False, kw_only=True)
class DulSituation(_OntologyEntity):
    """Structural representation of DulSituation."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#dul_Situation"
    )


@_dataclass(eq=False, kw_only=True)
class Executive(_OntologyEntity):
    """Structural representation of Executive."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#Executive"
    )


@_dataclass(eq=False, kw_only=True)
class ExecutiveArchitecture(_OntologyEntity):
    """Structural representation of ExecutiveArchitecture."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#ExecutiveArchitecture"
    )


@_dataclass(eq=False, kw_only=True)
class ExecutiveCounterpart(_OntologyEntity):
    """Structural representation of ExecutiveCounterpart."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#ExecutiveCounterpart"
    )


@_dataclass(eq=False, kw_only=True)
class ExecutiveEstimate(_OntologyEntity):
    """Structural representation of ExecutiveEstimate."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#ExecutiveEstimate"
    )


@_dataclass(eq=False, kw_only=True)
class ExecutiveResponsibility(_OntologyEntity):
    """Structural representation of ExecutiveResponsibility."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#ExecutiveResponsibility"
    )


@_dataclass(eq=False, kw_only=True)
class Expectation(_OntologyEntity):
    """Structural representation of Expectation."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#Expectation"
    )


@_dataclass(eq=False, kw_only=True)
class FailureConcept(_OntologyEntity):
    """Structural representation of FailureConcept."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#FailureConcept"
    )


@_dataclass(eq=False, kw_only=True)
class FeasibilityAssessment(_OntologyEntity):
    """Structural representation of FeasibilityAssessment."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#FeasibilityAssessment"
    )


@_dataclass(eq=False, kw_only=True)
class FeasibilityEnvelope(_OntologyEntity):
    """Structural representation of FeasibilityEnvelope."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#FeasibilityEnvelope"
    )


@_dataclass(eq=False, kw_only=True)
class FeedbackInterface(_OntologyEntity):
    """Structural representation of FeedbackInterface."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#FeedbackInterface"
    )


@_dataclass(eq=False, kw_only=True)
class Finite(_OntologyEntity):
    """Structural representation of Finite."""

    __ontology_iri__: _ClassVar[str] = "http://aicor.knowledge/l2-ontology.owl#Finite"


@_dataclass(eq=False, kw_only=True)
class Fixed(_OntologyEntity):
    """Structural representation of Fixed."""

    __ontology_iri__: _ClassVar[str] = "http://aicor.knowledge/l2-ontology.owl#Fixed"


@_dataclass(eq=False, kw_only=True)
class ForceInteraction(_OntologyEntity):
    """Structural representation of ForceInteraction."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#ForceInteraction"
    )


@_dataclass(eq=False, kw_only=True)
class ForceProfile(_OntologyEntity):
    """Structural representation of ForceProfile."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#ForceProfile"
    )


@_dataclass(eq=False, kw_only=True)
class Generate(_OntologyEntity):
    """Structural representation of Generate."""

    __ontology_iri__: _ClassVar[str] = "http://aicor.knowledge/l2-ontology.owl#Generate"


@_dataclass(eq=False, kw_only=True)
class Generative(_OntologyEntity):
    """Structural representation of Generative."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#Generative"
    )


@_dataclass(eq=False, kw_only=True)
class Geometry(_OntologyEntity):
    """Structural representation of Geometry."""

    __ontology_iri__: _ClassVar[str] = "http://aicor.knowledge/l2-ontology.owl#Geometry"


@_dataclass(eq=False, kw_only=True)
class GraspRef(_OntologyEntity):
    """Structural representation of GraspRef."""

    __ontology_iri__: _ClassVar[str] = "http://aicor.knowledge/l2-ontology.owl#GraspRef"


@_dataclass(eq=False, kw_only=True)
class Ground(_OntologyEntity):
    """Structural representation of Ground."""

    __ontology_iri__: _ClassVar[str] = "http://aicor.knowledge/l2-ontology.owl#Ground"


@_dataclass(eq=False, kw_only=True)
class GroundedBy(_OntologyEntity):
    """Structural representation of GroundedBy."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#GroundedBy"
    )


@_dataclass(eq=False, kw_only=True)
class GroundedSchema(_OntologyEntity):
    """Structural representation of GroundedSchema."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#GroundedSchema"
    )


@_dataclass(eq=False, kw_only=True)
class GroundingField(_OntologyEntity):
    """Structural representation of GroundingField."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#GroundingField"
    )


@_dataclass(eq=False, kw_only=True)
class HelpRequest(_OntologyEntity):
    """Structural representation of HelpRequest."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#HelpRequest"
    )


@_dataclass(eq=False, kw_only=True)
class HumanNuisance(_OntologyEntity):
    """Structural representation of HumanNuisance."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#HumanNuisance"
    )


@_dataclass(eq=False, kw_only=True)
class HumanReadableAndMachineInterpretable(_OntologyEntity):
    """Structural representation of HumanReadableAndMachineInterpretable."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#HumanReadableAndMachineInterpretable"
    )


@_dataclass(eq=False, kw_only=True)
class Individual(_OntologyEntity):
    """Structural representation of Individual."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#Individual"
    )


@_dataclass(eq=False, kw_only=True)
class InertiaMatrix(_OntologyEntity):
    """Structural representation of InertiaMatrix."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#InertiaMatrix"
    )


@_dataclass(eq=False, kw_only=True)
class InertialSpec(_OntologyEntity):
    """Structural representation of InertialSpec."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#InertialSpec"
    )


@_dataclass(eq=False, kw_only=True)
class InitNuisance(_OntologyEntity):
    """Structural representation of InitNuisance."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#InitNuisance"
    )


@_dataclass(eq=False, kw_only=True)
class InstanceOfLink(_OntologyEntity):
    """Structural representation of InstanceOfLink."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#InstanceOfLink"
    )


@_dataclass(eq=False, kw_only=True)
class InstantaneousHorizon(_OntologyEntity):
    """Structural representation of InstantaneousHorizon."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#InstantaneousHorizon"
    )


@_dataclass(eq=False, kw_only=True)
class InstantiateMonitor(_OntologyEntity):
    """Structural representation of InstantiateMonitor."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#InstantiateMonitor"
    )


@_dataclass(eq=False, kw_only=True)
class Jam(_OntologyEntity):
    """Structural representation of Jam."""

    __ontology_iri__: _ClassVar[str] = "http://aicor.knowledge/l2-ontology.owl#Jam"


@_dataclass(eq=False, kw_only=True)
class JointElement(_OntologyEntity):
    """Structural representation of JointElement."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#JointElement"
    )


@_dataclass(eq=False, kw_only=True)
class JointType(_OntologyEntity):
    """Structural representation of JointType."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#JointType"
    )


@_dataclass(eq=False, kw_only=True)
class KrrHypothesis(_OntologyEntity):
    """Structural representation of KrrHypothesis."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#KRRHypothesis"
    )


@_dataclass(eq=False, kw_only=True)
class Latent(_OntologyEntity):
    """Structural representation of Latent."""

    __ontology_iri__: _ClassVar[str] = "http://aicor.knowledge/l2-ontology.owl#Latent"


@_dataclass(eq=False, kw_only=True)
class Law(_OntologyEntity):
    """Structural representation of Law."""

    __ontology_iri__: _ClassVar[str] = "http://aicor.knowledge/l2-ontology.owl#Law"


@_dataclass(eq=False, kw_only=True)
class LayoutNuisance(_OntologyEntity):
    """Structural representation of LayoutNuisance."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#LayoutNuisance"
    )


@_dataclass(eq=False, kw_only=True)
class LifecycleEvent(_OntologyEntity):
    """Structural representation of LifecycleEvent."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#LifecycleEvent"
    )


@_dataclass(eq=False, kw_only=True)
class LocationRef(_OntologyEntity):
    """Structural representation of LocationRef."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#LocationRef"
    )


@_dataclass(eq=False, kw_only=True)
class Lock(_OntologyEntity):
    """Structural representation of Lock."""

    __ontology_iri__: _ClassVar[str] = "http://aicor.knowledge/l2-ontology.owl#Lock"


@_dataclass(eq=False, kw_only=True)
class MjcfFormat(_OntologyEntity):
    """Structural representation of MjcfFormat."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#MJCFFormat"
    )


@_dataclass(eq=False, kw_only=True)
class ManipulableInstance(_OntologyEntity):
    """Structural representation of ManipulableInstance."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#ManipulableInstance"
    )


@_dataclass(eq=False, kw_only=True)
class ManipulationAspect(_OntologyEntity):
    """Structural representation of ManipulationAspect."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#ManipulationAspect"
    )


@_dataclass(eq=False, kw_only=True)
class DulAgent(_OntologyEntity):
    """Structural representation of DulAgent."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#dul_Agent"
    )


@_dataclass(eq=False, kw_only=True)
class MasteryCriterion(_OntologyEntity):
    """Structural representation of MasteryCriterion."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#MasteryCriterion"
    )


@_dataclass(eq=False, kw_only=True)
class MaterialProperty(_OntologyEntity):
    """Structural representation of MaterialProperty."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#MaterialProperty"
    )


@_dataclass(eq=False, kw_only=True)
class Metareasoning(_OntologyEntity):
    """Structural representation of Metareasoning."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#Metareasoning"
    )


@_dataclass(eq=False, kw_only=True)
class MisdirectedForce(_OntologyEntity):
    """Structural representation of MisdirectedForce."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#MisdirectedForce"
    )


@_dataclass(eq=False, kw_only=True)
class Modality(_OntologyEntity):
    """Structural representation of Modality."""

    __ontology_iri__: _ClassVar[str] = "http://aicor.knowledge/l2-ontology.owl#Modality"


@_dataclass(eq=False, kw_only=True)
class MonitorAttachment(_OntologyEntity):
    """Structural representation of MonitorAttachment."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#MonitorAttachment"
    )


@_dataclass(eq=False, kw_only=True)
class MonitorCounterpart(_OntologyEntity):
    """Structural representation of MonitorCounterpart."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#MonitorCounterpart"
    )


@_dataclass(eq=False, kw_only=True)
class Monitorability(_OntologyEntity):
    """Structural representation of Monitorability."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#Monitorability"
    )


@_dataclass(eq=False, kw_only=True)
class Motion(_OntologyEntity):
    """Structural representation of Motion."""

    __ontology_iri__: _ClassVar[str] = "http://aicor.knowledge/l2-ontology.owl#Motion"


@_dataclass(eq=False, kw_only=True)
class MotionRef(_OntologyEntity):
    """Structural representation of MotionRef."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#MotionRef"
    )


@_dataclass(eq=False, kw_only=True)
class None_(_OntologyEntity):
    """Structural representation of None_."""

    __ontology_iri__: _ClassVar[str] = "http://aicor.knowledge/l2-ontology.owl#None"


@_dataclass(eq=False, kw_only=True)
class NuisanceComponent(_OntologyEntity):
    """Structural representation of NuisanceComponent."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#NuisanceComponent"
    )


@_dataclass(eq=False, kw_only=True)
class ObjectNuisance(_OntologyEntity):
    """Structural representation of ObjectNuisance."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#ObjectNuisance"
    )


@_dataclass(eq=False, kw_only=True)
class ObjectRef(_OntologyEntity):
    """Structural representation of ObjectRef."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#ObjectRef"
    )


@_dataclass(eq=False, kw_only=True)
class ObjectScope(_OntologyEntity):
    """Structural representation of ObjectScope."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#ObjectScope"
    )


@_dataclass(eq=False, kw_only=True)
class ObjectState(_OntologyEntity):
    """Structural representation of ObjectState."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#ObjectState"
    )


@_dataclass(eq=False, kw_only=True)
class ObservationFunctions(_OntologyEntity):
    """Structural representation of ObservationFunctions."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#ObservationFunctions"
    )


@_dataclass(eq=False, kw_only=True)
class ObservationRef(_OntologyEntity):
    """Structural representation of ObservationRef."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#ObservationRef"
    )


@_dataclass(eq=False, kw_only=True)
class ObservationRequest(_OntologyEntity):
    """Structural representation of ObservationRequest."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#ObservationRequest"
    )


@_dataclass(eq=False, kw_only=True)
class ObservationToPerceptService(_OntologyEntity):
    """Structural representation of ObservationToPerceptService."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#ObservationToPerceptService"
    )


@_dataclass(eq=False, kw_only=True)
class OntologicalConcept(_OntologyEntity):
    """Structural representation of OntologicalConcept."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#OntologicalConcept"
    )


@_dataclass(eq=False, kw_only=True)
class OperationalizationCriterion(_OntologyEntity):
    """Structural representation of OperationalizationCriterion."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#OperationalizationCriterion"
    )


@_dataclass(eq=False, kw_only=True)
class Outcome(_OntologyEntity):
    """Structural representation of Outcome."""

    __ontology_iri__: _ClassVar[str] = "http://aicor.knowledge/l2-ontology.owl#Outcome"


@_dataclass(eq=False, kw_only=True)
class OutcomeDistribution(_OntologyEntity):
    """Structural representation of OutcomeDistribution."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#OutcomeDistribution"
    )


@_dataclass(eq=False, kw_only=True)
class Parameter(_OntologyEntity):
    """Structural representation of Parameter."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#Parameter"
    )


@_dataclass(eq=False, kw_only=True)
class ParameterInterval(_OntologyEntity):
    """Structural representation of ParameterInterval."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#ParameterInterval"
    )


@_dataclass(eq=False, kw_only=True)
class Percept(_OntologyEntity):
    """Structural representation of Percept."""

    __ontology_iri__: _ClassVar[str] = "http://aicor.knowledge/l2-ontology.owl#Percept"


@_dataclass(eq=False, kw_only=True)
class Perception(_OntologyEntity):
    """Structural representation of Perception."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#Perception"
    )


@_dataclass(eq=False, kw_only=True)
class Performance(_OntologyEntity):
    """Structural representation of Performance."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#Performance"
    )


@_dataclass(eq=False, kw_only=True)
class PhysicalCausal(_OntologyEntity):
    """Structural representation of PhysicalCausal."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#PhysicalCausal"
    )


@_dataclass(eq=False, kw_only=True)
class PhysicalContact(_OntologyEntity):
    """Structural representation of PhysicalContact."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#PhysicalContact"
    )


@_dataclass(eq=False, kw_only=True)
class PhysicalDeviation(_OntologyEntity):
    """Structural representation of PhysicalDeviation."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#PhysicalDeviation"
    )


@_dataclass(eq=False, kw_only=True)
class PhysicalNuisance(_OntologyEntity):
    """Structural representation of PhysicalNuisance."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#PhysicalNuisance"
    )


@_dataclass(eq=False, kw_only=True)
class Physics(_OntologyEntity):
    """Structural representation of Physics."""

    __ontology_iri__: _ClassVar[str] = "http://aicor.knowledge/l2-ontology.owl#Physics"


@_dataclass(eq=False, kw_only=True)
class PhysicsModel(_OntologyEntity):
    """Structural representation of PhysicsModel."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#PhysicsModel"
    )


@_dataclass(eq=False, kw_only=True)
class Prediction(_OntologyEntity):
    """Structural representation of Prediction."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#Prediction"
    )


@_dataclass(eq=False, kw_only=True)
class PredictiveForce(_OntologyEntity):
    """Structural representation of PredictiveForce."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#PredictiveForce"
    )


@_dataclass(eq=False, kw_only=True)
class Preference(_OntologyEntity):
    """Structural representation of Preference."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#Preference"
    )


@_dataclass(eq=False, kw_only=True)
class PreferenceOrder(_OntologyEntity):
    """Structural representation of PreferenceOrder."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#PreferenceOrder"
    )


@_dataclass(eq=False, kw_only=True)
class PrimaryActivity(_OntologyEntity):
    """Structural representation of PrimaryActivity."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#PrimaryActivity"
    )


@_dataclass(eq=False, kw_only=True)
class Prismatic(_OntologyEntity):
    """Structural representation of Prismatic."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#Prismatic"
    )


@_dataclass(eq=False, kw_only=True)
class Probability(_OntologyEntity):
    """Structural representation of Probability."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#Probability"
    )


@_dataclass(eq=False, kw_only=True)
class ProcessEscape(_OntologyEntity):
    """Structural representation of ProcessEscape."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#ProcessEscape"
    )


@_dataclass(eq=False, kw_only=True)
class ProcessModule(_OntologyEntity):
    """Structural representation of ProcessModule."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#ProcessModule"
    )


@_dataclass(eq=False, kw_only=True)
class ProcessOccurrence(_OntologyEntity):
    """Structural representation of ProcessOccurrence."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#ProcessOccurrence"
    )


@_dataclass(eq=False, kw_only=True)
class ProcessStall(_OntologyEntity):
    """Structural representation of ProcessStall."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#ProcessStall"
    )


@_dataclass(eq=False, kw_only=True)
class ProcessingTrace(_OntologyEntity):
    """Structural representation of ProcessingTrace."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#ProcessingTrace"
    )
    trace_produced_by: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#traceProducedBy",
            "ontology_range": "",
            "property_kind": "object",
        },
    )


@_dataclass(eq=False, kw_only=True)
class ProcessingVerdict(_OntologyEntity):
    """Structural representation of ProcessingVerdict."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#ProcessingVerdict"
    )
    assesses_trace: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#assessesTrace",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    relative_to_purpose: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#relativeToPurpose",
            "ontology_range": "",
            "property_kind": "object",
        },
    )


@_dataclass(eq=False, kw_only=True)
class Progress(_OntologyEntity):
    """Structural representation of Progress."""

    __ontology_iri__: _ClassVar[str] = "http://aicor.knowledge/l2-ontology.owl#Progress"


@_dataclass(eq=False, kw_only=True)
class PrologSemantics(_OntologyEntity):
    """Structural representation of PrologSemantics."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#PrologSemantics"
    )


@_dataclass(eq=False, kw_only=True)
class Proposal(_OntologyEntity):
    """Structural representation of Proposal."""

    __ontology_iri__: _ClassVar[str] = "http://aicor.knowledge/l2-ontology.owl#Proposal"


@_dataclass(eq=False, kw_only=True)
class ProposalCommitment(_OntologyEntity):
    """Structural representation of ProposalCommitment."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#ProposalCommitment"
    )


@_dataclass(eq=False, kw_only=True)
class ProposalMechanism(_OntologyEntity):
    """Structural representation of ProposalMechanism."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#ProposalMechanism"
    )


@_dataclass(eq=False, kw_only=True)
class ProposalNotCommitment(_OntologyEntity):
    """Structural representation of ProposalNotCommitment."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#ProposalNotCommitment"
    )


@_dataclass(eq=False, kw_only=True)
class QualityFunctional(_OntologyEntity):
    """Structural representation of QualityFunctional."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#QualityFunctional"
    )


@_dataclass(eq=False, kw_only=True)
class DulAction(_OntologyEntity):
    """Structural representation of DulAction."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#dul_Action"
    )


@_dataclass(eq=False, kw_only=True)
class Realization(_OntologyEntity):
    """Structural representation of Realization."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#Realization"
    )


@_dataclass(eq=False, kw_only=True)
class RealizationForm(_OntologyEntity):
    """Structural representation of RealizationForm."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#RealizationForm"
    )


@_dataclass(eq=False, kw_only=True)
class Reasoner(_OntologyEntity):
    """Structural representation of Reasoner."""

    __ontology_iri__: _ClassVar[str] = "http://aicor.knowledge/l2-ontology.owl#Reasoner"


@_dataclass(eq=False, kw_only=True)
class Record(_OntologyEntity):
    """Structural representation of Record."""

    __ontology_iri__: _ClassVar[str] = "http://aicor.knowledge/l2-ontology.owl#Record"


@_dataclass(eq=False, kw_only=True)
class Recoverability(_OntologyEntity):
    """Structural representation of Recoverability."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#Recoverability"
    )


@_dataclass(eq=False, kw_only=True)
class ReferentDistribution(_OntologyEntity):
    """Structural representation of ReferentDistribution."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#ReferentDistribution"
    )


@_dataclass(eq=False, kw_only=True)
class DulRelation(_OntologyEntity):
    """Structural representation of DulRelation."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#dul_Relation"
    )


@_dataclass(eq=False, kw_only=True)
class RelationFamily(_OntologyEntity):
    """Structural representation of RelationFamily."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#RelationFamily"
    )


@_dataclass(eq=False, kw_only=True)
class Relational(_OntologyEntity):
    """Structural representation of Relational."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#Relational"
    )


@_dataclass(eq=False, kw_only=True)
class RepairCandidateRef(_OntologyEntity):
    """Structural representation of RepairCandidateRef."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#RepairCandidateRef"
    )


@_dataclass(eq=False, kw_only=True)
class RepairOption(_OntologyEntity):
    """Structural representation of RepairOption."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#RepairOption"
    )


@_dataclass(eq=False, kw_only=True)
class Reparameterize(_OntologyEntity):
    """Structural representation of Reparameterize."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#Reparameterize"
    )


@_dataclass(eq=False, kw_only=True)
class Represents(_OntologyEntity):
    """Structural representation of Represents."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#Represents"
    )


@_dataclass(eq=False, kw_only=True)
class RequestHelp(_OntologyEntity):
    """Structural representation of RequestHelp."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#RequestHelp"
    )


@_dataclass(eq=False, kw_only=True)
class ResolutionMode(_OntologyEntity):
    """Structural representation of ResolutionMode."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#ResolutionMode"
    )


@_dataclass(eq=False, kw_only=True)
class ResponsePolicy(_OntologyEntity):
    """Structural representation of ResponsePolicy."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#ResponsePolicy"
    )


@_dataclass(eq=False, kw_only=True)
class ResponsibleClaimScope(_OntologyEntity):
    """Structural representation of ResponsibleClaimScope."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#ResponsibleClaimScope"
    )


@_dataclass(eq=False, kw_only=True)
class ResponsibleNonAccomplishment(_OntologyEntity):
    """Structural representation of ResponsibleNonAccomplishment."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#ResponsibleNonAccomplishment"
    )


@_dataclass(eq=False, kw_only=True)
class RevisionCounterpart(_OntologyEntity):
    """Structural representation of RevisionCounterpart."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#RevisionCounterpart"
    )


@_dataclass(eq=False, kw_only=True)
class Revolute(_OntologyEntity):
    """Structural representation of Revolute."""

    __ontology_iri__: _ClassVar[str] = "http://aicor.knowledge/l2-ontology.owl#Revolute"


@_dataclass(eq=False, kw_only=True)
class RiskEstimation(_OntologyEntity):
    """Structural representation of RiskEstimation."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#RiskEstimation"
    )


@_dataclass(eq=False, kw_only=True)
class Robustness(_OntologyEntity):
    """Structural representation of Robustness."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#Robustness"
    )


@_dataclass(eq=False, kw_only=True)
class RoleAssignmentLayer(_OntologyEntity):
    """Structural representation of RoleAssignmentLayer."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#RoleAssignmentLayer"
    )


@_dataclass(eq=False, kw_only=True)
class SdfFormat(_OntologyEntity):
    """Structural representation of SdfFormat."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#SDFFormat"
    )


@_dataclass(eq=False, kw_only=True)
class SdfWorldFormat(_OntologyEntity):
    """Structural representation of SdfWorldFormat."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#SDFWorldFormat"
    )


@_dataclass(eq=False, kw_only=True)
class SdtNode(_OntologyEntity):
    """Structural representation of SdtNode."""

    __ontology_iri__: _ClassVar[str] = "http://aicor.knowledge/l2-ontology.owl#SDTNode"
    encodes_attribute: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#encodesAttribute",
            "ontology_range": "",
            "property_kind": "object",
        },
    )


@_dataclass(eq=False, kw_only=True)
class SafetyAssessment(_OntologyEntity):
    """Structural representation of SafetyAssessment."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#SafetyAssessment"
    )


@_dataclass(eq=False, kw_only=True)
class SafetyCheck(_OntologyEntity):
    """Structural representation of SafetyCheck."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#SafetyCheck"
    )


@_dataclass(eq=False, kw_only=True)
class Scope(_OntologyEntity):
    """Structural representation of Scope."""

    __ontology_iri__: _ClassVar[str] = "http://aicor.knowledge/l2-ontology.owl#Scope"


@_dataclass(eq=False, kw_only=True)
class Semantic(_OntologyEntity):
    """Structural representation of Semantic."""

    __ontology_iri__: _ClassVar[str] = "http://aicor.knowledge/l2-ontology.owl#Semantic"


@_dataclass(eq=False, kw_only=True)
class SensingNuisance(_OntologyEntity):
    """Structural representation of SensingNuisance."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#SensingNuisance"
    )


@_dataclass(eq=False, kw_only=True)
class SensorOutput(_OntologyEntity):
    """Structural representation of SensorOutput."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#SensorOutput"
    )


@_dataclass(eq=False, kw_only=True)
class SkillInvocation(_OntologyEntity):
    """Structural representation of SkillInvocation."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#SkillInvocation"
    )


@_dataclass(eq=False, kw_only=True)
class Slip(_OntologyEntity):
    """Structural representation of Slip."""

    __ontology_iri__: _ClassVar[str] = "http://aicor.knowledge/l2-ontology.owl#Slip"


@_dataclass(eq=False, kw_only=True)
class SomaOntology(_OntologyEntity):
    """Structural representation of SomaOntology."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#SomaOntology"
    )


@_dataclass(eq=False, kw_only=True)
class Source(_OntologyEntity):
    """Structural representation of Source."""

    __ontology_iri__: _ClassVar[str] = "http://aicor.knowledge/l2-ontology.owl#Source"


@_dataclass(eq=False, kw_only=True)
class SourceAndFamily(_OntologyEntity):
    """Structural representation of SourceAndFamily."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#SourceAndFamily"
    )


@_dataclass(eq=False, kw_only=True)
class SparqlQueryAnsweringModule(_OntologyEntity):
    """Structural representation of SparqlQueryAnsweringModule."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#SparqlQueryAnsweringModule"
    )


@_dataclass(eq=False, kw_only=True)
class SpatialPose(_OntologyEntity):
    """Structural representation of SpatialPose."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#SpatialPose"
    )


@_dataclass(eq=False, kw_only=True)
class SpecificationField(_OntologyEntity):
    """Structural representation of SpecificationField."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#SpecificationField"
    )


@_dataclass(eq=False, kw_only=True)
class Spherical(_OntologyEntity):
    """Structural representation of Spherical."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#Spherical"
    )


@_dataclass(eq=False, kw_only=True)
class Static(_OntologyEntity):
    """Structural representation of Static."""

    __ontology_iri__: _ClassVar[str] = "http://aicor.knowledge/l2-ontology.owl#Static"


@_dataclass(eq=False, kw_only=True)
class Structural(_OntologyEntity):
    """Structural representation of Structural."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#Structural"
    )


@_dataclass(eq=False, kw_only=True)
class Support(_OntologyEntity):
    """Structural representation of Support."""

    __ontology_iri__: _ClassVar[str] = "http://aicor.knowledge/l2-ontology.owl#Support"


@_dataclass(eq=False, kw_only=True)
class SupportsReasoning(_OntologyEntity):
    """Structural representation of SupportsReasoning."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#SupportsReasoning"
    )


@_dataclass(eq=False, kw_only=True)
class SwitchMode(_OntologyEntity):
    """Structural representation of SwitchMode."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#SwitchMode"
    )


@_dataclass(eq=False, kw_only=True)
class SymbolicDiagnosis(_OntologyEntity):
    """Structural representation of SymbolicDiagnosis."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#SymbolicDiagnosis"
    )


@_dataclass(eq=False, kw_only=True)
class SymbolicQuery(_OntologyEntity):
    """Structural representation of SymbolicQuery."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#SymbolicQuery"
    )


@_dataclass(eq=False, kw_only=True)
class System(_OntologyEntity):
    """Structural representation of System."""

    __ontology_iri__: _ClassVar[str] = "http://aicor.knowledge/l2-ontology.owl#System"


@_dataclass(eq=False, kw_only=True)
class System1Output(_OntologyEntity):
    """Structural representation of System1Output."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#System1Output"
    )


@_dataclass(eq=False, kw_only=True)
class TaskComponent(_OntologyEntity):
    """Structural representation of TaskComponent."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#TaskComponent"
    )


@_dataclass(eq=False, kw_only=True)
class TaskGenerator(_OntologyEntity):
    """Structural representation of TaskGenerator."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#TaskGenerator"
    )


@_dataclass(eq=False, kw_only=True)
class TaskManagementService(_OntologyEntity):
    """Structural representation of TaskManagementService."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#TaskManagementService"
    )


@_dataclass(eq=False, kw_only=True)
class TaskNuisance(_OntologyEntity):
    """Structural representation of TaskNuisance."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#TaskNuisance"
    )


@_dataclass(eq=False, kw_only=True)
class TaskRelativeClass(_OntologyEntity):
    """Structural representation of TaskRelativeClass."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#TaskRelativeClass"
    )


@_dataclass(eq=False, kw_only=True)
class TaskSource(_OntologyEntity):
    """Structural representation of TaskSource."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#TaskSource"
    )


@_dataclass(eq=False, kw_only=True)
class TaskType(_OntologyEntity):
    """Structural representation of TaskType."""

    __ontology_iri__: _ClassVar[str] = "http://aicor.knowledge/l2-ontology.owl#TaskType"


@_dataclass(eq=False, kw_only=True)
class TerminalDisposition(_OntologyEntity):
    """Structural representation of TerminalDisposition."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#TerminalDisposition"
    )


@_dataclass(eq=False, kw_only=True)
class TraceRecording(_OntologyEntity):
    """Structural representation of TraceRecording."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#TraceRecording"
    )


@_dataclass(eq=False, kw_only=True)
class Trajectory(_OntologyEntity):
    """Structural representation of Trajectory."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#Trajectory"
    )


@_dataclass(eq=False, kw_only=True)
class TrajectoryRef(_OntologyEntity):
    """Structural representation of TrajectoryRef."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#TrajectoryRef"
    )


@_dataclass(eq=False, kw_only=True)
class Translate(_OntologyEntity):
    """Structural representation of Translate."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#Translate"
    )


@_dataclass(eq=False, kw_only=True)
class Tree(_OntologyEntity):
    """Structural representation of Tree."""

    __ontology_iri__: _ClassVar[str] = "http://aicor.knowledge/l2-ontology.owl#Tree"


@_dataclass(eq=False, kw_only=True)
class Triple(_OntologyEntity):
    """Structural representation of Triple."""

    __ontology_iri__: _ClassVar[str] = "http://aicor.knowledge/l2-ontology.owl#Triple"


@_dataclass(eq=False, kw_only=True)
class Type_(_OntologyEntity):
    """Structural representation of Type_."""

    __ontology_iri__: _ClassVar[str] = "http://aicor.knowledge/l2-ontology.owl#Type"


@_dataclass(eq=False, kw_only=True)
class UrdfFormat(_OntologyEntity):
    """Structural representation of UrdfFormat."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#URDFFormat"
    )


@_dataclass(eq=False, kw_only=True)
class Usd(_OntologyEntity):
    """Structural representation of Usd."""

    __ontology_iri__: _ClassVar[str] = "http://aicor.knowledge/l2-ontology.owl#USD"


@_dataclass(eq=False, kw_only=True)
class UsdFormat(_OntologyEntity):
    """Structural representation of UsdFormat."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#USDFormat"
    )


@_dataclass(eq=False, kw_only=True)
class Uncertainty(_OntologyEntity):
    """Structural representation of Uncertainty."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#Uncertainty"
    )


@_dataclass(eq=False, kw_only=True)
class Unique(_OntologyEntity):
    """Structural representation of Unique."""

    __ontology_iri__: _ClassVar[str] = "http://aicor.knowledge/l2-ontology.owl#Unique"


@_dataclass(eq=False, kw_only=True)
class UniqueReferent(_OntologyEntity):
    """Structural representation of UniqueReferent."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#UniqueReferent"
    )


@_dataclass(eq=False, kw_only=True)
class Unresolved(_OntologyEntity):
    """Structural representation of Unresolved."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#Unresolved"
    )


@_dataclass(eq=False, kw_only=True)
class UnstableRelease(_OntologyEntity):
    """Structural representation of UnstableRelease."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#UnstableRelease"
    )


@_dataclass(eq=False, kw_only=True)
class UpCrack(_OntologyEntity):
    """Structural representation of UpCrack."""

    __ontology_iri__: _ClassVar[str] = "http://aicor.knowledge/l2-ontology.owl#UpCrack"


@_dataclass(eq=False, kw_only=True)
class VlaModel(_OntologyEntity):
    """Structural representation of VlaModel."""

    __ontology_iri__: _ClassVar[str] = "http://aicor.knowledge/l2-ontology.owl#VLAModel"


@_dataclass(eq=False, kw_only=True)
class Variable(_OntologyEntity):
    """Structural representation of Variable."""

    __ontology_iri__: _ClassVar[str] = "http://aicor.knowledge/l2-ontology.owl#Variable"


@_dataclass(eq=False, kw_only=True)
class WarrantRecord(_OntologyEntity):
    """Structural representation of WarrantRecord."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#WarrantRecord"
    )


@_dataclass(eq=False, kw_only=True)
class WithdrawAndAbstain(_OntologyEntity):
    """Structural representation of WithdrawAndAbstain."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#WithdrawAndAbstain"
    )


@_dataclass(eq=False, kw_only=True)
class AbstractEntity(Entity):
    """Structural representation of AbstractEntity."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#AbstractEntity"
    )


@_dataclass(eq=False, kw_only=True)
class PhysicalEntity(Entity):
    """Structural representation of PhysicalEntity."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#PhysicalEntity"
    )


@_dataclass(eq=False, kw_only=True)
class AchievementMiss(FailureReason):
    """Structural representation of AchievementMiss."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#AchievementMiss"
    )


@_dataclass(eq=False, kw_only=True)
class MaintenanceBreach(FailureReason):
    """Structural representation of MaintenanceBreach."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#MaintenanceBreach"
    )


@_dataclass(eq=False, kw_only=True)
class PreventionViolation(FailureReason):
    """Structural representation of PreventionViolation."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#PreventionViolation"
    )


@_dataclass(eq=False, kw_only=True)
class ActivityCommitment(CommitmentTarget):
    """Structural representation of ActivityCommitment."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#ActivityCommitment"
    )
    commitment_kind: list[str] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#commitmentKind",
            "ontology_range": "http://www.w3.org/2001/XMLSchema#string",
            "property_kind": "datatype",
        },
    )


@_dataclass(eq=False, kw_only=True)
class AgentCommitment(CommitmentTarget):
    """Structural representation of AgentCommitment."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#AgentCommitment"
    )
    commitment_kind: list[str] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#commitmentKind",
            "ontology_range": "http://www.w3.org/2001/XMLSchema#string",
            "property_kind": "datatype",
        },
    )


@_dataclass(eq=False, kw_only=True)
class ArrangementCommitment(CommitmentTarget):
    """Structural representation of ArrangementCommitment."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#ArrangementCommitment"
    )
    commitment_kind: list[str] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#commitmentKind",
            "ontology_range": "http://www.w3.org/2001/XMLSchema#string",
            "property_kind": "datatype",
        },
    )


@_dataclass(eq=False, kw_only=True)
class AgentRole(ParticipantRole):
    """Structural representation of AgentRole."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#AgentRole"
    )


@_dataclass(eq=False, kw_only=True)
class GoalRole(ParticipantRole):
    """Structural representation of GoalRole."""

    __ontology_iri__: _ClassVar[str] = "http://aicor.knowledge/l2-ontology.owl#GoalRole"


@_dataclass(eq=False, kw_only=True)
class InstrumentRole(ParticipantRole):
    """Structural representation of InstrumentRole."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#InstrumentRole"
    )


@_dataclass(eq=False, kw_only=True)
class PatientRole(ParticipantRole):
    """Structural representation of PatientRole."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#PatientRole"
    )


@_dataclass(eq=False, kw_only=True)
class SourceRole(ParticipantRole):
    """Structural representation of SourceRole."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#SourceRole"
    )


@_dataclass(eq=False, kw_only=True)
class SubstanceRole(ParticipantRole):
    """Structural representation of SubstanceRole."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#SubstanceRole"
    )


@_dataclass(eq=False, kw_only=True)
class BodyWorldTrajectory(DulEvent):
    """Structural representation of BodyWorldTrajectory."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#BodyWorldTrajectory"
    )


@_dataclass(eq=False, kw_only=True)
class BodyWorldTrajectorySegment(DulEvent):
    """Structural representation of BodyWorldTrajectorySegment."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#BodyWorldTrajectorySegment"
    )
    authored_by: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#authoredBy",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    is_result_of: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#isResultOf",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    over_interval: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#overInterval",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    segment_of: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#segmentOf",
            "ontology_range": "",
            "property_kind": "object",
        },
    )


@_dataclass(eq=False, kw_only=True)
class TaskRelativeEffect(DulEvent):
    """Structural representation of TaskRelativeEffect."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#TaskRelativeEffect"
    )


@_dataclass(eq=False, kw_only=True)
class TaskRelevantStateChange(DulEvent):
    """Structural representation of TaskRelevantStateChange."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#TaskRelevantStateChange"
    )
    within_interval: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#withinInterval",
            "ontology_range": "",
            "property_kind": "object",
        },
    )


@_dataclass(eq=False, kw_only=True)
class TwoLevelFailure(DulEvent):
    """Structural representation of TwoLevelFailure."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#TwoLevelFailure"
    )


@_dataclass(eq=False, kw_only=True)
class BodyWorldPhysicalProcess(DulProcess):
    """Structural representation of BodyWorldPhysicalProcess."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#BodyWorldPhysicalProcess"
    )
    is_abstract_process: list[bool] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#isAbstractProcess",
            "ontology_range": "http://www.w3.org/2001/XMLSchema#boolean",
            "property_kind": "datatype",
        },
    )


@_dataclass(eq=False, kw_only=True)
class CramOperation(DulProcess):
    """Structural representation of CramOperation."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#CRAMOperation"
    )


@_dataclass(eq=False, kw_only=True)
class ComputationalProcess(DulProcess):
    """Structural representation of ComputationalProcess."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#ComputationalProcess"
    )
    operates_on: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#operatesOn",
            "ontology_range": "",
            "property_kind": "object",
        },
    )


@_dataclass(eq=False, kw_only=True)
class HouseholdActivity(DulProcess):
    """Structural representation of HouseholdActivity."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#HouseholdActivity"
    )


@_dataclass(eq=False, kw_only=True)
class PhysicalProcess(DulProcess):
    """Structural representation of PhysicalProcess."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#PhysicalProcess"
    )


@_dataclass(eq=False, kw_only=True)
class CandidateAction(SomaAction):
    """Structural representation of CandidateAction."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#CandidateAction"
    )


@_dataclass(eq=False, kw_only=True)
class CaseLevelEntity(DulEntity):
    """Structural representation of CaseLevelEntity."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#CaseLevelEntity"
    )
    at_case_level: list[bool] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#atCaseLevel",
            "ontology_range": "http://www.w3.org/2001/XMLSchema#boolean",
            "property_kind": "datatype",
        },
    )


@_dataclass(eq=False, kw_only=True)
class MetaLevelEntity(DulEntity):
    """Structural representation of MetaLevelEntity."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#MetaLevelEntity"
    )
    at_case_level: list[bool] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#atCaseLevel",
            "ontology_range": "http://www.w3.org/2001/XMLSchema#boolean",
            "property_kind": "datatype",
        },
    )


@_dataclass(eq=False, kw_only=True)
class Recordability(CommitmentCriterion):
    """Structural representation of Recordability."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#Recordability"
    )


@_dataclass(eq=False, kw_only=True)
class ComponentCapability(DulQuality):
    """Structural representation of ComponentCapability."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#ComponentCapability"
    )
    capability_of: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#capabilityOf",
            "ontology_range": "",
            "property_kind": "object",
        },
    )


@_dataclass(eq=False, kw_only=True)
class DiscreteModeStructure(DulQuality):
    """Structural representation of DiscreteModeStructure."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#DiscreteModeStructure"
    )


@_dataclass(eq=False, kw_only=True)
class EdgeReversibility(DulQuality):
    """Structural representation of EdgeReversibility."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#EdgeReversibility"
    )


@_dataclass(eq=False, kw_only=True)
class ExecutiveCapability(DulQuality):
    """Structural representation of ExecutiveCapability."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#ExecutiveCapability"
    )
    determined_by: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#determinedBy",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    determines_performance_on: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#determinesPerformanceOn",
            "ontology_range": "",
            "property_kind": "object",
        },
    )


@_dataclass(eq=False, kw_only=True)
class HybridBodyWorldState(DulQuality):
    """Structural representation of HybridBodyWorldState."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#HybridBodyWorldState"
    )
    evolves_by: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#evolvesBy",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    has_continuous_state: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#hasContinuousState",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    has_discrete_mode: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#hasDiscreteMode",
            "ontology_range": "",
            "property_kind": "object",
        },
    )


@_dataclass(eq=False, kw_only=True)
class KinematicRedundancy(DulQuality):
    """Structural representation of KinematicRedundancy."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#KinematicRedundancy"
    )


@_dataclass(eq=False, kw_only=True)
class KinematicSingularity(DulQuality):
    """Structural representation of KinematicSingularity."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#KinematicSingularity"
    )


@_dataclass(eq=False, kw_only=True)
class Constraint(CouplingType):
    """Structural representation of Constraint."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#Constraint"
    )
    characteristic_failure: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#characteristicFailure",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    governed_by: list[_Any] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#governedBy",
            "ontology_range": "",
            "property_kind": "datatype",
        },
    )


@_dataclass(eq=False, kw_only=True)
class ForceClosure(CouplingType):
    """Structural representation of ForceClosure."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#ForceClosure"
    )
    characteristic_failure: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#characteristicFailure",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    governed_by: list[_Any] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#governedBy",
            "ontology_range": "",
            "property_kind": "datatype",
        },
    )


@_dataclass(eq=False, kw_only=True)
class Process(CouplingType):
    """Structural representation of Process."""

    __ontology_iri__: _ClassVar[str] = "http://aicor.knowledge/l2-ontology.owl#Process"
    characteristic_failure: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#characteristicFailure",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    governed_by: list[_Any] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#governedBy",
            "ontology_range": "",
            "property_kind": "datatype",
        },
    )


@_dataclass(eq=False, kw_only=True)
class GiveUpFailure(FailureConcept):
    """Structural representation of GiveUpFailure."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#GiveUpFailure"
    )


@_dataclass(eq=False, kw_only=True)
class ImpossibilityFailure(FailureConcept):
    """Structural representation of ImpossibilityFailure."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#ImpossibilityFailure"
    )


@_dataclass(eq=False, kw_only=True)
class PurposeFailure(FailureConcept):
    """Structural representation of PurposeFailure."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#PurposeFailure"
    )


@_dataclass(eq=False, kw_only=True)
class QueryInterfaceOperation(DulAction):
    """Structural representation of QueryInterfaceOperation."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#QueryInterfaceOperation"
    )


@_dataclass(eq=False, kw_only=True)
class RelationEdge(DulRelation):
    """Structural representation of RelationEdge."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#RelationEdge"
    )


@_dataclass(eq=False, kw_only=True)
class ActionReferent(AbstractEntity):
    """Structural representation of ActionReferent."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#ActionReferent"
    )


@_dataclass(eq=False, kw_only=True)
class Occasion(AbstractEntity, DulSituation):
    """Structural representation of Occasion."""

    __ontology_iri__: _ClassVar[str] = "http://aicor.knowledge/l2-ontology.owl#Occasion"
    holds_interval: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#holdsInterval",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    of_fluent_type: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#ofFluentType",
            "ontology_range": "",
            "property_kind": "object",
        },
    )


@_dataclass(eq=False, kw_only=True)
class Occurrence(AbstractEntity, DulSituation):
    """Structural representation of Occurrence."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#Occurrence"
    )
    of_event_type: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#ofEventType",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    over_interval: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#overInterval",
            "ontology_range": "",
            "property_kind": "object",
        },
    )


@_dataclass(eq=False, kw_only=True)
class Phase(AbstractEntity):
    """Structural representation of Phase."""

    __ontology_iri__: _ClassVar[str] = "http://aicor.knowledge/l2-ontology.owl#Phase"


@_dataclass(eq=False, kw_only=True)
class DulConcept(AbstractEntity):
    """Structural representation of DulConcept."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#dul_Concept"
    )


@_dataclass(eq=False, kw_only=True)
class DulDescription(AbstractEntity):
    """Structural representation of DulDescription."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#dul_Description"
    )


@_dataclass(eq=False, kw_only=True)
class DulInformationObject(AbstractEntity):
    """Structural representation of DulInformationObject."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#dul_InformationObject"
    )


@_dataclass(eq=False, kw_only=True)
class DulRegion(AbstractEntity):
    """Structural representation of DulRegion."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#dul_Region"
    )


@_dataclass(eq=False, kw_only=True)
class BodyMotion(PhysicalEntity, DulEvent):
    """Structural representation of BodyMotion."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#BodyMotion"
    )
    consequential_via: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#consequentialVia",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    driven_by: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#drivenBy",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    generated_by: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#generatedBy",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    free_space: list[bool] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#freeSpace",
            "ontology_range": "http://www.w3.org/2001/XMLSchema#boolean",
            "property_kind": "datatype",
        },
    )
    generator_agnostic: list[bool] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#generatorAgnostic",
            "ontology_range": "http://www.w3.org/2001/XMLSchema#boolean",
            "property_kind": "datatype",
        },
    )


@_dataclass(eq=False, kw_only=True)
class Contact(PhysicalEntity):
    """Structural representation of Contact."""

    __ontology_iri__: _ClassVar[str] = "http://aicor.knowledge/l2-ontology.owl#Contact"


@_dataclass(eq=False, kw_only=True)
class Force(PhysicalEntity):
    """Structural representation of Force."""

    __ontology_iri__: _ClassVar[str] = "http://aicor.knowledge/l2-ontology.owl#Force"


@_dataclass(eq=False, kw_only=True)
class PhysicalObject(PhysicalEntity):
    """Structural representation of PhysicalObject."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#PhysicalObject"
    )


@_dataclass(eq=False, kw_only=True)
class DulPhysicalAgent(PhysicalEntity):
    """Structural representation of DulPhysicalAgent."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#dul_PhysicalAgent"
    )


@_dataclass(eq=False, kw_only=True)
class DulPhysicalObject(PhysicalEntity):
    """Structural representation of DulPhysicalObject."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#dul_PhysicalObject"
    )


@_dataclass(eq=False, kw_only=True)
class AuthoredStateChange(TaskRelevantStateChange):
    """Structural representation of AuthoredStateChange."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#AuthoredStateChange"
    )
    caused_by_perform: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#causedByPerform",
            "ontology_range": "",
            "property_kind": "object",
        },
    )


@_dataclass(eq=False, kw_only=True)
class DisturbanceStateChange(TaskRelevantStateChange):
    """Structural representation of DisturbanceStateChange."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#DisturbanceStateChange"
    )


@_dataclass(eq=False, kw_only=True)
class BindingLevelFailure(TwoLevelFailure):
    """Structural representation of BindingLevelFailure."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#BindingLevelFailure"
    )


@_dataclass(eq=False, kw_only=True)
class SchemaLevelFailure(TwoLevelFailure):
    """Structural representation of SchemaLevelFailure."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#SchemaLevelFailure"
    )


@_dataclass(eq=False, kw_only=True)
class Consolidation(ComputationalProcess):
    """Structural representation of Consolidation."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#Consolidation"
    )
    partitions_into: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#partitionsInto",
            "ontology_range": "",
            "property_kind": "object",
        },
    )


@_dataclass(eq=False, kw_only=True)
class Critic(ComputationalProcess):
    """Structural representation of Critic."""

    __ontology_iri__: _ClassVar[str] = "http://aicor.knowledge/l2-ontology.owl#Critic"
    yields_verdict: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#yieldsVerdict",
            "ontology_range": "",
            "property_kind": "object",
        },
    )


@_dataclass(eq=False, kw_only=True)
class DeferredResolution(ComputationalProcess, DesignatorResolution):
    """Structural representation of DeferredResolution."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#DeferredResolution"
    )
    has_resolution_mode: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#hasResolutionMode",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    is_lazy: list[_Any] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#isLazy",
            "ontology_range": "",
            "property_kind": "datatype",
        },
    )


@_dataclass(eq=False, kw_only=True)
class ExecutiveInformationProcess(ComputationalProcess):
    """Structural representation of ExecutiveInformationProcess."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#ExecutiveInformationProcess"
    )
    is_abstract_process: list[bool] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#isAbstractProcess",
            "ontology_range": "http://www.w3.org/2001/XMLSchema#boolean",
            "property_kind": "datatype",
        },
    )


@_dataclass(eq=False, kw_only=True)
class InternalSimulation(ComputationalProcess):
    """Structural representation of InternalSimulation."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#InternalSimulation"
    )
    simulates_over: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#simulatesOver",
            "ontology_range": "",
            "property_kind": "object",
        },
    )


@_dataclass(eq=False, kw_only=True)
class LifePathSelection(ComputationalProcess):
    """Structural representation of LifePathSelection."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#LifePathSelection"
    )
    computes_risk: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#computesRisk",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    selects_by_min_risk: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#selectsByMinRisk",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    uses_loss_function: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#usesLossFunction",
            "ontology_range": "",
            "property_kind": "object",
        },
    )


@_dataclass(eq=False, kw_only=True)
class Provisioning(ComputationalProcess):
    """Structural representation of Provisioning."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#Provisioning"
    )
    acquires: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#acquires",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    incurs_obligation: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#incursObligation",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    ranked_by_norm: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#rankedByNorm",
            "ontology_range": "",
            "property_kind": "object",
        },
    )


@_dataclass(eq=False, kw_only=True)
class Refinement(ComputationalProcess, CramOperation):
    """Structural representation of Refinement."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#Refinement"
    )
    refines_representation: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#refinesRepresentation",
            "ontology_range": "",
            "property_kind": "object",
        },
    )


@_dataclass(eq=False, kw_only=True)
class RequestCoercion(ComputationalProcess):
    """Structural representation of RequestCoercion."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#RequestCoercion"
    )
    coerces_object_to: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#coercesObjectTo",
            "ontology_range": "",
            "property_kind": "object",
        },
    )


@_dataclass(eq=False, kw_only=True)
class RequestResolution(ComputationalProcess):
    """Structural representation of RequestResolution."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#RequestResolution"
    )
    resolves_request: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#resolvesRequest",
            "ontology_range": "",
            "property_kind": "object",
        },
    )


@_dataclass(eq=False, kw_only=True)
class StoryTelling(ComputationalProcess):
    """Structural representation of StoryTelling."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#StoryTelling"
    )
    produces: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#produces",
            "ontology_range": "",
            "property_kind": "object",
        },
    )


@_dataclass(eq=False, kw_only=True)
class Drinking(HouseholdActivity):
    """Structural representation of Drinking."""

    __ontology_iri__: _ClassVar[str] = "http://aicor.knowledge/l2-ontology.owl#Drinking"


@_dataclass(eq=False, kw_only=True)
class Eating(HouseholdActivity):
    """Structural representation of Eating."""

    __ontology_iri__: _ClassVar[str] = "http://aicor.knowledge/l2-ontology.owl#Eating"


@_dataclass(eq=False, kw_only=True)
class Execution(PhysicalProcess, CramOperation):
    """Structural representation of Execution."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#Execution"
    )
    generates_body_motion: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#generatesBodyMotion",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    realizes_designator: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#realizesDesignator",
            "ontology_range": "",
            "property_kind": "object",
        },
    )


@_dataclass(eq=False, kw_only=True)
class AdmissibleAction(CandidateAction):
    """Structural representation of AdmissibleAction."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#AdmissibleAction"
    )


@_dataclass(eq=False, kw_only=True)
class Coupling(CaseLevelEntity):
    """Structural representation of Coupling."""

    __ontology_iri__: _ClassVar[str] = "http://aicor.knowledge/l2-ontology.owl#Coupling"


@_dataclass(eq=False, kw_only=True)
class InitialCondition(CaseLevelEntity):
    """Structural representation of InitialCondition."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#InitialCondition"
    )


@_dataclass(eq=False, kw_only=True)
class ManipulationCase(CaseLevelEntity):
    """Structural representation of ManipulationCase."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#ManipulationCase"
    )
    has_problem: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#hasProblem",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    has_rwms: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#hasRWMS",
            "ontology_range": "",
            "property_kind": "object",
        },
    )


@_dataclass(eq=False, kw_only=True)
class ManipulationEpisode(CaseLevelEntity, AbstractEntity, DulSituation):
    """Structural representation of ManipulationEpisode."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#ManipulationEpisode"
    )
    conforms_to_scope: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#conformsToScope",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    has_commitment_record: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#hasCommitmentRecord",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    has_lifecycle_events: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#hasLifecycleEvents",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    has_physical_core: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#hasPhysicalCore",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    has_recording: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#hasRecording",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    has_request: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#hasRequest",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    has_state_estimate: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#hasStateEstimate",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    has_task_reading: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#hasTaskReading",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    has_terminal_disposition: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#hasTerminalDisposition",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    has_trace: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#hasTrace",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    has_verdict_component: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#hasVerdictComponent",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    is_run_of: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#isRunOf",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    satisfies: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#satisfies",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    is_bounded: list[bool] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#isBounded",
            "ontology_range": "http://www.w3.org/2001/XMLSchema#boolean",
            "property_kind": "datatype",
        },
    )
    is_hierarchical: list[bool] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#isHierarchical",
            "ontology_range": "http://www.w3.org/2001/XMLSchema#boolean",
            "property_kind": "datatype",
        },
    )


@_dataclass(eq=False, kw_only=True)
class ManipulationExecutive(CaseLevelEntity, AbstractEntity, DulAgent):
    """Structural representation of ManipulationExecutive."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#ManipulationExecutive"
    )
    controls: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#controls",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    discharges: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#discharges",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    generates: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#generates",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    issues: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#issues",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    is_accountable: list[bool] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#isAccountable",
            "ontology_range": "http://www.w3.org/2001/XMLSchema#boolean",
            "property_kind": "datatype",
        },
    )


@_dataclass(eq=False, kw_only=True)
class TaskRelevantVariable(CaseLevelEntity):
    """Structural representation of TaskRelevantVariable."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#TaskRelevantVariable"
    )


@_dataclass(eq=False, kw_only=True)
class AssessmentEvent(MetaLevelEntity):
    """Structural representation of AssessmentEvent."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#AssessmentEvent"
    )
    assesses_case: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#assessesCase",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    has_verdict: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#hasVerdict",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    uses_regime: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#usesRegime",
            "ontology_range": "",
            "property_kind": "object",
        },
    )


@_dataclass(eq=False, kw_only=True)
class AxiomSet(MetaLevelEntity):
    """Structural representation of AxiomSet."""

    __ontology_iri__: _ClassVar[str] = "http://aicor.knowledge/l2-ontology.owl#AxiomSet"


@_dataclass(eq=False, kw_only=True)
class CaseFamily(MetaLevelEntity):
    """Structural representation of CaseFamily."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#CaseFamily"
    )
    has_case_member: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#hasCaseMember",
            "ontology_range": "",
            "property_kind": "object",
        },
    )


@_dataclass(eq=False, kw_only=True)
class Distribution(MetaLevelEntity):
    """Structural representation of Distribution."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#Distribution"
    )
    has_support: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#hasSupport",
            "ontology_range": "",
            "property_kind": "object",
        },
    )


@_dataclass(eq=False, kw_only=True)
class EvidenceSet(MetaLevelEntity):
    """Structural representation of EvidenceSet."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#EvidenceSet"
    )
    evidence_derived_from: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#evidenceDerivedFrom",
            "ontology_range": "",
            "property_kind": "object",
        },
    )


@_dataclass(eq=False, kw_only=True)
class VerdictComponent(MetaLevelEntity, AbstractEntity, DulQuality):
    """Structural representation of VerdictComponent."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#VerdictComponent"
    )
    along_dimension: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#alongDimension",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    assesses: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#assesses",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    grounded_in: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#groundedIn",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    claim_scope: list[str] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#claimScope",
            "ontology_range": "http://www.w3.org/2001/XMLSchema#string",
            "property_kind": "datatype",
        },
    )
    is_critic_assigned: list[bool] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#isCriticAssigned",
            "ontology_range": "http://www.w3.org/2001/XMLSchema#boolean",
            "property_kind": "datatype",
        },
    )


@_dataclass(eq=False, kw_only=True)
class Irreversible(EdgeReversibility):
    """Structural representation of Irreversible."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#Irreversible"
    )


@_dataclass(eq=False, kw_only=True)
class Recoverable(EdgeReversibility):
    """Structural representation of Recoverable."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#Recoverable"
    )


@_dataclass(eq=False, kw_only=True)
class ArrangementCompetence(ExecutiveCapability):
    """Structural representation of ArrangementCompetence."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#ArrangementCompetence"
    )


@_dataclass(eq=False, kw_only=True)
class CognitiveCapability(ExecutiveCapability):
    """Structural representation of CognitiveCapability."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#CognitiveCapability"
    )


@_dataclass(eq=False, kw_only=True)
class PlacementCompetence(ExecutiveCapability):
    """Structural representation of PlacementCompetence."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#PlacementCompetence"
    )
    scoped_to_class: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#scopedToClass",
            "ontology_range": "",
            "property_kind": "object",
        },
    )


@_dataclass(eq=False, kw_only=True)
class RealizationCompetence(ExecutiveCapability):
    """Structural representation of RealizationCompetence."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#RealizationCompetence"
    )


@_dataclass(eq=False, kw_only=True)
class RestorationCompetence(ExecutiveCapability):
    """Structural representation of RestorationCompetence."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#RestorationCompetence"
    )


@_dataclass(eq=False, kw_only=True)
class StabilizationCompetence(ExecutiveCapability):
    """Structural representation of StabilizationCompetence."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#StabilizationCompetence"
    )


@_dataclass(eq=False, kw_only=True)
class BeliefOccasion(Occasion):
    """Structural representation of BeliefOccasion."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#BeliefOccasion"
    )


@_dataclass(eq=False, kw_only=True)
class WorldOccasion(Occasion):
    """Structural representation of WorldOccasion."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#WorldOccasion"
    )


@_dataclass(eq=False, kw_only=True)
class BeliefOccurrence(Occurrence):
    """Structural representation of BeliefOccurrence."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#BeliefOccurrence"
    )


@_dataclass(eq=False, kw_only=True)
class PerformOccurrence(Occurrence):
    """Structural representation of PerformOccurrence."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#PerformOccurrence"
    )
    answers_request: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#answersRequest",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    has_information_process: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#hasInformationProcess",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    has_performer: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#hasPerformer",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    has_physical_process: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#hasPhysicalProcess",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    has_task_status: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#hasTaskStatus",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    occupies_interval: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#occupiesInterval",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    over_system: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#overSystem",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    produces_segment: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#producesSegment",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    realizes_via: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#realizesVia",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    renders_via: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#rendersVia",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    via_chain: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#viaChain",
            "ontology_range": "",
            "property_kind": "object",
        },
    )


@_dataclass(eq=False, kw_only=True)
class WorldOccurrence(Occurrence):
    """Structural representation of WorldOccurrence."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#WorldOccurrence"
    )


@_dataclass(eq=False, kw_only=True)
class ConceptFamily(DulConcept):
    """Structural representation of ConceptFamily."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#ConceptFamily"
    )
    answers_question: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#answersQuestion",
            "ontology_range": "",
            "property_kind": "object",
        },
    )


@_dataclass(eq=False, kw_only=True)
class EpistemicStatus(DulConcept):
    """Structural representation of EpistemicStatus."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#EpistemicStatus"
    )


@_dataclass(eq=False, kw_only=True)
class FoundationModelInsertionPoint(DulConcept):
    """Structural representation of FoundationModelInsertionPoint."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#FoundationModelInsertionPoint"
    )


@_dataclass(eq=False, kw_only=True)
class ManipulationDimension(DulConcept):
    """Structural representation of ManipulationDimension."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#ManipulationDimension"
    )
    develops_scientific_object: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#developsScientificObject",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    has_state_of_understanding: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#hasStateOfUnderstanding",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    pairs_concept_family: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#pairsConceptFamily",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    capability_axis: list[str] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#capabilityAxis",
            "ontology_range": "http://www.w3.org/2001/XMLSchema#string",
            "property_kind": "datatype",
        },
    )


@_dataclass(eq=False, kw_only=True)
class RoleSlot(DulConcept):
    """Structural representation of RoleSlot."""

    __ontology_iri__: _ClassVar[str] = "http://aicor.knowledge/l2-ontology.owl#RoleSlot"
    physically_typed_by: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#physicallyTypedBy",
            "ontology_range": "",
            "property_kind": "object",
        },
    )


@_dataclass(eq=False, kw_only=True)
class AccountableImplementationStructure(DulDescription):
    """Structural representation of AccountableImplementationStructure."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#AccountableImplementationStructure"
    )
    answers_question: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#answersQuestion",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    characterizes: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#characterizes",
            "ontology_range": "",
            "property_kind": "object",
        },
    )


@_dataclass(eq=False, kw_only=True)
class ActionProposalResource(DulDescription):
    """Structural representation of ActionProposalResource."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#ActionProposalResource"
    )
    proposes: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#proposes",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    requires_before_commit: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#requiresBeforeCommit",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    near_action_interface: list[_Any] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#nearActionInterface",
            "ontology_range": "",
            "property_kind": "datatype",
        },
    )


@_dataclass(eq=False, kw_only=True)
class ActionRelatedConcept(DulDescription):
    """Structural representation of ActionRelatedConcept."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#ActionRelatedConcept"
    )
    conditioned_on: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#conditionedOn",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    estimates: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#estimates",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    over_parameter_space: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#overParameterSpace",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    vocabulary_for: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#vocabularyFor",
            "ontology_range": "",
            "property_kind": "object",
        },
    )


@_dataclass(eq=False, kw_only=True)
class ActivityUnderstanding(DulDescription, ComputationalProcess):
    """Structural representation of ActivityUnderstanding."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#ActivityUnderstanding"
    )
    perceives: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#perceives",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    produces_episode: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#producesEpisode",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    reconstructs_phase: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#reconstructsPhase",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    validated_by: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#validatedBy",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    yields_designator: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#yieldsDesignator",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    from_video: list[_Any] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#fromVideo",
            "ontology_range": "",
            "property_kind": "datatype",
        },
    )
    physics_validated: list[_Any] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#physicsValidated",
            "ontology_range": "",
            "property_kind": "datatype",
        },
    )


@_dataclass(eq=False, kw_only=True)
class ApplicationContext(DulDescription):
    """Structural representation of ApplicationContext."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#ApplicationContext"
    )


@_dataclass(eq=False, kw_only=True)
class AssessmentRegime(DulDescription, MetaLevelEntity):
    """Structural representation of AssessmentRegime."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#AssessmentRegime"
    )
    applies_critic: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#appliesCritic",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    governs_claim: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#governsClaim",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    produces_verdict: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#producesVerdict",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    is_executive_independent: list[bool] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#isExecutiveIndependent",
            "ontology_range": "http://www.w3.org/2001/XMLSchema#boolean",
            "property_kind": "datatype",
        },
    )


@_dataclass(eq=False, kw_only=True)
class AssumptionSet(DulDescription):
    """Structural representation of AssumptionSet."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#AssumptionSet"
    )
    scopes_projection: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#scopesProjection",
            "ontology_range": "",
            "property_kind": "object",
        },
    )


@_dataclass(eq=False, kw_only=True)
class BehaviorCritic(DulDescription, ComputationalProcess):
    """Structural representation of BehaviorCritic."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#BehaviorCritic"
    )
    examines: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#examines",
            "ontology_range": "",
            "property_kind": "object",
        },
    )


@_dataclass(eq=False, kw_only=True)
class BindingWorkspace(DulDescription):
    """Structural representation of BindingWorkspace."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#BindingWorkspace"
    )
    has_part: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#hasPart",
            "ontology_range": "",
            "property_kind": "object",
        },
    )


@_dataclass(eq=False, kw_only=True)
class BodyWorldConstraint(DulDescription):
    """Structural representation of BodyWorldConstraint."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#BodyWorldConstraint"
    )
    constrains: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#constrains",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    unlocks: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#unlocks",
            "ontology_range": "",
            "property_kind": "object",
        },
    )


@_dataclass(eq=False, kw_only=True)
class CapabilityInteraction(DulDescription):
    """Structural representation of CapabilityInteraction."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#CapabilityInteraction"
    )
    interacts_among: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#interactsAmong",
            "ontology_range": "",
            "property_kind": "object",
        },
    )


@_dataclass(eq=False, kw_only=True)
class CausalAdmissibility(DulDescription):
    """Structural representation of CausalAdmissibility."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#CausalAdmissibility"
    )


@_dataclass(eq=False, kw_only=True)
class ChannelIndependence(DulDescription):
    """Structural representation of ChannelIndependence."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#ChannelIndependence"
    )
    constrains: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#constrains",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    is_defeasible: list[_Any] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#isDefeasible",
            "ontology_range": "",
            "property_kind": "datatype",
        },
    )
    is_testable: list[_Any] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#isTestable",
            "ontology_range": "",
            "property_kind": "datatype",
        },
    )


@_dataclass(eq=False, kw_only=True)
class ClaimWarrant(DulDescription):
    """Structural representation of ClaimWarrant."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#ClaimWarrant"
    )
    warrant_ranges_over: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#warrantRangesOver",
            "ontology_range": "",
            "property_kind": "object",
        },
    )


@_dataclass(eq=False, kw_only=True)
class CoReference(DulDescription, ComputationalProcess):
    """Structural representation of CoReference."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#CoReference"
    )
    equates_with: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#equatesWith",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    asserts_same_entity: list[_Any] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#assertsSameEntity",
            "ontology_range": "",
            "property_kind": "datatype",
        },
    )


@_dataclass(eq=False, kw_only=True)
class CommitmentChecking(DulDescription):
    """Structural representation of CommitmentChecking."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#CommitmentChecking"
    )
    against: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#against",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    takes_proposal: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#takesProposal",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    tests: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#tests",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    uses: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#uses",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    yields: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#yields",
            "ontology_range": "",
            "property_kind": "object",
        },
    )


@_dataclass(eq=False, kw_only=True)
class CommitmentRecord(DulDescription):
    """Structural representation of CommitmentRecord."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#CommitmentRecord"
    )
    contains: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#contains",
            "ontology_range": "",
            "property_kind": "object",
        },
    )


@_dataclass(eq=False, kw_only=True)
class CompetenceQuestion(DulDescription):
    """Structural representation of CompetenceQuestion."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#CompetenceQuestion"
    )
    answered_by: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#answeredBy",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    raised_by: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#raisedBy",
            "ontology_range": "",
            "property_kind": "object",
        },
    )


@_dataclass(eq=False, kw_only=True)
class CompetencyQuestion(DulDescription):
    """Structural representation of CompetencyQuestion."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#CompetencyQuestion"
    )
    answered_by: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#answeredBy",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    required_for_accomplishing: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#requiredForAccomplishing",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    enables_successful_action: list[_Any] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#enablesSuccessfulAction",
            "ontology_range": "",
            "property_kind": "datatype",
        },
    )


@_dataclass(eq=False, kw_only=True)
class ConceptualFramework(DulDescription):
    """Structural representation of ConceptualFramework."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#ConceptualFramework"
    )
    frames_object: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#framesObject",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    makes_statable: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#makesStatable",
            "ontology_range": "",
            "property_kind": "object",
        },
    )


@_dataclass(eq=False, kw_only=True)
class ConcurrentPolicy(DulDescription):
    """Structural representation of ConcurrentPolicy."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#ConcurrentPolicy"
    )
    maintains_condition_for: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#maintainsConditionFor",
            "ontology_range": "",
            "property_kind": "object",
        },
    )


@_dataclass(eq=False, kw_only=True)
class ConstraintExploitationPrinciple(DulDescription):
    """Structural representation of ConstraintExploitationPrinciple."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#ConstraintExploitationPrinciple"
    )
    characterizes_method: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#characterizesMethod",
            "ontology_range": "",
            "property_kind": "object",
        },
    )


@_dataclass(eq=False, kw_only=True)
class ConstraintSet(DulDescription):
    """Structural representation of ConstraintSet."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#ConstraintSet"
    )


@_dataclass(eq=False, kw_only=True)
class ContactPair(DulDescription):
    """Structural representation of ContactPair."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#ContactPair"
    )
    target_part: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#targetPart",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    tool_part: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#toolPart",
            "ontology_range": "",
            "property_kind": "object",
        },
    )


@_dataclass(eq=False, kw_only=True)
class ContextualizationStep(DulDescription, ComputationalProcess):
    """Structural representation of ContextualizationStep."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#ContextualizationStep"
    )


@_dataclass(eq=False, kw_only=True)
class ControlProgram(DulDescription):
    """Structural representation of ControlProgram."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#ControlProgram"
    )
    gives_inspectable_structure_to: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#givesInspectableStructureTo",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    has_part: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#hasPart",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    may_call: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#mayCall",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    organizes_control_of: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#organizesControlOf",
            "ontology_range": "",
            "property_kind": "object",
        },
    )


@_dataclass(eq=False, kw_only=True)
class CostFunction(DulDescription):
    """Structural representation of CostFunction."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#CostFunction"
    )


@_dataclass(eq=False, kw_only=True)
class CouplingTerm(DulDescription):
    """Structural representation of CouplingTerm."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#CouplingTerm"
    )
    has_coupling_type: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#hasCouplingType",
            "ontology_range": "",
            "property_kind": "object",
        },
    )


@_dataclass(eq=False, kw_only=True)
class CriticizeReviseCycle(DulDescription, ComputationalProcess):
    """Structural representation of CriticizeReviseCycle."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#CriticizeReviseCycle"
    )
    criticizes_with: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#criticizesWith",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    projects: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#projects",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    revises_via: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#revisesVia",
            "ontology_range": "",
            "property_kind": "object",
        },
    )


@_dataclass(eq=False, kw_only=True)
class DeclaredBoundary(DulDescription, CaseLevelEntity):
    """Structural representation of DeclaredBoundary."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#DeclaredBoundary"
    )
    declares_body_world_membership: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#declaresBodyWorldMembership",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    declares_system_environment_split: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#declaresSystemEnvironmentSplit",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    declares_task_relevant_variable: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#declaresTaskRelevantVariable",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    declares_validity_interval: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#declaresValidityInterval",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    boundary_is_declared: list[bool] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#boundaryIsDeclared",
            "ontology_range": "http://www.w3.org/2001/XMLSchema#boolean",
            "property_kind": "datatype",
        },
    )


@_dataclass(eq=False, kw_only=True)
class DeliberationScheduling(DulDescription, ComputationalProcess):
    """Structural representation of DeliberationScheduling."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#DeliberationScheduling"
    )
    costs_over: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#costsOver",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    drives: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#drives",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    schedules: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#schedules",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    is_metareasoning: list[_Any] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#isMetareasoning",
            "ontology_range": "",
            "property_kind": "datatype",
        },
    )
    is_open_frontier: list[_Any] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#isOpenFrontier",
            "ontology_range": "",
            "property_kind": "datatype",
        },
    )


@_dataclass(eq=False, kw_only=True)
class DependencyEdge(DulDescription):
    """Structural representation of DependencyEdge."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#DependencyEdge"
    )


@_dataclass(eq=False, kw_only=True)
class Designator(DulDescription):
    """Structural representation of Designator."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#Designator"
    )
    bound_to: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#boundTo",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    for_role_slot: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#forRoleSlot",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    references: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#references",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    resolves_to: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#resolvesTo",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    resolves_via: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#resolvesVia",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    lazy: list[bool] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#lazy",
            "ontology_range": "http://www.w3.org/2001/XMLSchema#boolean",
            "property_kind": "datatype",
        },
    )


@_dataclass(eq=False, kw_only=True)
class DesignatorDrift(DulDescription):
    """Structural representation of DesignatorDrift."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#DesignatorDrift"
    )
    drifts_from: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#driftsFrom",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    drifts_to: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#driftsTo",
            "ontology_range": "",
            "property_kind": "object",
        },
    )


@_dataclass(eq=False, kw_only=True)
class DownstreamModel(DulDescription):
    """Structural representation of DownstreamModel."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#DownstreamModel"
    )
    carries_metadata: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#carriesMetadata",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    integrated_into: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#integratedInto",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    output_supports: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#outputSupports",
            "ontology_range": "",
            "property_kind": "object",
        },
    )


@_dataclass(eq=False, kw_only=True)
class DualProcessManipulationExecutive(DulDescription):
    """Structural representation of DualProcessManipulationExecutive."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#DualProcessManipulationExecutive"
    )
    architecture_of: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#architectureOf",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    has_feedback: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#hasFeedback",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    has_system1: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#hasSystem1",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    has_system2: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#hasSystem2",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    has_translation_interface: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#hasTranslationInterface",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    logs_to: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#logsTo",
            "ontology_range": "",
            "property_kind": "object",
        },
    )


@_dataclass(eq=False, kw_only=True)
class DynamicModel(DulDescription):
    """Structural representation of DynamicModel."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#DynamicModel"
    )
    has_inertia_term: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#hasInertiaTerm",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    model_of: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#modelOf",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    relates_torque_to_acceleration: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#relatesTorqueToAcceleration",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    structures: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#structures",
            "ontology_range": "",
            "property_kind": "object",
        },
    )


@_dataclass(eq=False, kw_only=True)
class EngineeringDiscipline(DulDescription):
    """Structural representation of EngineeringDiscipline."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#EngineeringDiscipline"
    )
    designs_artifact_for: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#designsArtifactFor",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    investigates_solution_of: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#investigatesSolutionOf",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    studies_object: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#studiesObject",
            "ontology_range": "",
            "property_kind": "object",
        },
    )


@_dataclass(eq=False, kw_only=True)
class Envisioning(DulDescription):
    """Structural representation of Envisioning."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#Envisioning"
    )
    composes_simulate: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#composesSimulate",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    composes_translate: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#composesTranslate",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    envisioned_by: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#envisionedBy",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    over_parameter_space: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#overParameterSpace",
            "ontology_range": "",
            "property_kind": "object",
        },
    )


@_dataclass(eq=False, kw_only=True)
class EpistemicObject(DulDescription):
    """Structural representation of EpistemicObject."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#EpistemicObject"
    )
    concerns_object: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#concernsObject",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    has_epistemic_status: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#hasEpistemicStatus",
            "ontology_range": "",
            "property_kind": "object",
        },
    )


@_dataclass(eq=False, kw_only=True)
class EpistemicStatusTransition(DulDescription):
    """Structural representation of EpistemicStatusTransition."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#EpistemicStatusTransition"
    )
    from_status: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#fromStatus",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    to_status: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#toStatus",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    transition_of: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#transitionOf",
            "ontology_range": "",
            "property_kind": "object",
        },
    )


@_dataclass(eq=False, kw_only=True)
class ExcludedClaimType(DulDescription):
    """Structural representation of ExcludedClaimType."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#ExcludedClaimType"
    )


@_dataclass(eq=False, kw_only=True)
class ExecutiveBehaviouralSpecification(DulDescription):
    """Structural representation of ExecutiveBehaviouralSpecification."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#ExecutiveBehaviouralSpecification"
    )
    has_property: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#hasProperty",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    relative_to_problem_family: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#relativeToProblemFamily",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    specifies: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#specifies",
            "ontology_range": "",
            "property_kind": "object",
        },
    )


@_dataclass(eq=False, kw_only=True)
class ExecutiveComponent(DulDescription):
    """Structural representation of ExecutiveComponent."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#ExecutiveComponent"
    )
    has_capability: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#hasCapability",
            "ontology_range": "",
            "property_kind": "object",
        },
    )


@_dataclass(eq=False, kw_only=True)
class ExecutiveMechanism(DulDescription):
    """Structural representation of ExecutiveMechanism."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#ExecutiveMechanism"
    )
    mechanism_input: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#mechanismInput",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    mechanism_output: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#mechanismOutput",
            "ontology_range": "",
            "property_kind": "object",
        },
    )


@_dataclass(eq=False, kw_only=True)
class ExecutiveProperty(DulDescription):
    """Structural representation of ExecutiveProperty."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#ExecutiveProperty"
    )


@_dataclass(eq=False, kw_only=True)
class ExecutiveRealization(DulDescription):
    """Structural representation of ExecutiveRealization."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#ExecutiveRealization"
    )
    distinct_from: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#distinctFrom",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    has_processes: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#hasProcesses",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    has_substrate: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#hasSubstrate",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    realization_of: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#realizationOf",
            "ontology_range": "",
            "property_kind": "object",
        },
    )


@_dataclass(eq=False, kw_only=True)
class ExploitableConstraint(DulDescription):
    """Structural representation of ExploitableConstraint."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#ExploitableConstraint"
    )
    exposed_by_language: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#exposedByLanguage",
            "ontology_range": "",
            "property_kind": "object",
        },
    )


@_dataclass(eq=False, kw_only=True)
class FailureDescription(DulDescription):
    """Structural representation of FailureDescription."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#FailureDescription"
    )
    has_failed_task: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#hasFailedTask",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    has_failure_id: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#hasFailureId",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    has_severity: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#hasSeverity",
            "ontology_range": "",
            "property_kind": "object",
        },
    )


@_dataclass(eq=False, kw_only=True)
class FailureSignature(DulDescription):
    """Structural representation of FailureSignature."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#FailureSignature"
    )
    bridges: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#bridges",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    detected_by: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#detectedBy",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    enumerated_by: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#enumeratedBy",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    triggers: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#triggers",
            "ontology_range": "",
            "property_kind": "object",
        },
    )


@_dataclass(eq=False, kw_only=True)
class ForceDynamicDependency(DulDescription):
    """Structural representation of ForceDynamicDependency."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#ForceDynamicDependency"
    )
    posterior_schema: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#posteriorSchema",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    prior_schema: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#priorSchema",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    satisfied_by_event: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#satisfiedByEvent",
            "ontology_range": "",
            "property_kind": "object",
        },
    )


@_dataclass(eq=False, kw_only=True)
class ForceDynamicSchema(DulDescription):
    """Structural representation of ForceDynamicSchema."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#ForceDynamicSchema"
    )
    applies_to: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#appliesTo",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    defines_condition_on: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#definesConditionOn",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    has_role_slot: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#hasRoleSlot",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    is_event_description: list[bool] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#isEventDescription",
            "ontology_range": "http://www.w3.org/2001/XMLSchema#boolean",
            "property_kind": "datatype",
        },
    )


@_dataclass(eq=False, kw_only=True)
class ForwardKinematics(DulDescription):
    """Structural representation of ForwardKinematics."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#ForwardKinematics"
    )
    defined_by: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#definedBy",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    has_codomain: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#hasCodomain",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    has_domain: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#hasDomain",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    single_valued: list[bool] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#singleValued",
            "ontology_range": "http://www.w3.org/2001/XMLSchema#boolean",
            "property_kind": "datatype",
        },
    )
    total: list[bool] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#total",
            "ontology_range": "http://www.w3.org/2001/XMLSchema#boolean",
            "property_kind": "datatype",
        },
    )


@_dataclass(eq=False, kw_only=True)
class GenerativeDependencyStructure(DulDescription):
    """Structural representation of GenerativeDependencyStructure."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#GenerativeDependencyStructure"
    )
    has_dependency_edge: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#hasDependencyEdge",
            "ontology_range": "",
            "property_kind": "object",
        },
    )


@_dataclass(eq=False, kw_only=True)
class GlobalStandard(DulDescription):
    """Structural representation of GlobalStandard."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#GlobalStandard"
    )
    judges_system: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#judgesSystem",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    standard_over_family: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#standardOverFamily",
            "ontology_range": "",
            "property_kind": "object",
        },
    )


@_dataclass(eq=False, kw_only=True)
class Goal(DulDescription):
    """Structural representation of Goal."""

    __ontology_iri__: _ClassVar[str] = "http://aicor.knowledge/l2-ontology.owl#Goal"
    goal_scope: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#goalScope",
            "ontology_range": "",
            "property_kind": "object",
        },
    )


@_dataclass(eq=False, kw_only=True)
class GoalClause(DulDescription):
    """Structural representation of GoalClause."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#GoalClause"
    )
    checked_against: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#checkedAgainst",
            "ontology_range": "",
            "property_kind": "object",
        },
    )


@_dataclass(eq=False, kw_only=True)
class GrammarOfAction(DulDescription):
    """Structural representation of GrammarOfAction."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#GrammarOfAction"
    )
    composes_schema: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#composesSchema",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    constituent_bounded_by: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#constituentBoundedBy",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    has_dependency: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#hasDependency",
            "ontology_range": "",
            "property_kind": "object",
        },
    )


@_dataclass(eq=False, kw_only=True)
class GroundedManipulationProblem(DulDescription, CaseLevelEntity):
    """Structural representation of GroundedManipulationProblem."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#GroundedManipulationProblem"
    )
    generates_trace: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#generatesTrace",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    has_body_world_system: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#hasBodyWorldSystem",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    has_nuisance_condition: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#hasNuisanceCondition",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    has_task_specification: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#hasTaskSpecification",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    member_of: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#memberOf",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    is_assessment_input: list[bool] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#isAssessmentInput",
            "ontology_range": "http://www.w3.org/2001/XMLSchema#boolean",
            "property_kind": "datatype",
        },
    )
    is_rwms_part: list[bool] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#isRWMSPart",
            "ontology_range": "http://www.w3.org/2001/XMLSchema#boolean",
            "property_kind": "datatype",
        },
    )


@_dataclass(eq=False, kw_only=True)
class GroundingCommitment(DulDescription):
    """Structural representation of GroundingCommitment."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#GroundingCommitment"
    )
    collected_into: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#collectedInto",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    commits_that: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#commitsThat",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    constrains: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#constrains",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    constructed_by: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#constructedBy",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    is_revisable: list[_Any] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#isRevisable",
            "ontology_range": "",
            "property_kind": "datatype",
        },
    )


@_dataclass(eq=False, kw_only=True)
class ImplicitToExplicitManipulation(DulDescription):
    """Structural representation of ImplicitToExplicitManipulation."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#ImplicitToExplicitManipulation"
    )
    evaluates_in: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#evaluatesIn",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    maps_request: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#mapsRequest",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    to_body_motion: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#toBodyMotion",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    evaluates_in_perceptual_space: list[_Any] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#evaluatesInPerceptualSpace",
            "ontology_range": "",
            "property_kind": "datatype",
        },
    )
    propagates_constraints_downward: list[_Any] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#propagatesConstraintsDownward",
            "ontology_range": "",
            "property_kind": "datatype",
        },
    )


@_dataclass(eq=False, kw_only=True)
class InverseKinematics(DulDescription):
    """Structural representation of InverseKinematics."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#InverseKinematics"
    )
    has_codomain: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#hasCodomain",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    has_domain: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#hasDomain",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    has_solution_cardinality: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#hasSolutionCardinality",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    inverse_of: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#inverseOf",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    redundant: list[bool] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#redundant",
            "ontology_range": "http://www.w3.org/2001/XMLSchema#boolean",
            "property_kind": "datatype",
        },
    )


@_dataclass(eq=False, kw_only=True)
class KrrResource(DulDescription):
    """Structural representation of KrrResource."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#KRRResource"
    )
    makes_concrete: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#makesConcrete",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    resource_for: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#resourceFor",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    is_explicit: list[_Any] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#isExplicit",
            "ontology_range": "",
            "property_kind": "datatype",
        },
    )
    is_revisable: list[_Any] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#isRevisable",
            "ontology_range": "",
            "property_kind": "datatype",
        },
    )


@_dataclass(eq=False, kw_only=True)
class KinematicJacobian(DulDescription):
    """Structural representation of KinematicJacobian."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#KinematicJacobian"
    )
    differential_of: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#differentialOf",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    indexed_by: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#indexedBy",
            "ontology_range": "",
            "property_kind": "object",
        },
    )


@_dataclass(eq=False, kw_only=True)
class KinematicModel(DulDescription):
    """Structural representation of KinematicModel."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#KinematicModel"
    )
    defines_map: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#definesMap",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    has_differential: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#hasDifferential",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    model_of: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#modelOf",
            "ontology_range": "",
            "property_kind": "object",
        },
    )


@_dataclass(eq=False, kw_only=True)
class LearningLoop(DulDescription, ComputationalProcess):
    """Structural representation of LearningLoop."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#LearningLoop"
    )
    drives: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#drives",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    has_part: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#hasPart",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    improves_model: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#improvesModel",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    learns_from: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#learnsFrom",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    revises_via: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#revisesVia",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    is_closed_cycle: list[_Any] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#isClosedCycle",
            "ontology_range": "",
            "property_kind": "datatype",
        },
    )
    representation_edge_open: list[_Any] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#representationEdgeOpen",
            "ontology_range": "",
            "property_kind": "datatype",
        },
    )


@_dataclass(eq=False, kw_only=True)
class ManipulationControlProgram(DulDescription):
    """Structural representation of ManipulationControlProgram."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#ManipulationControlProgram"
    )
    consumes: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#consumes",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    emits: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#emits",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    realized_as: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#realizedAs",
            "ontology_range": "",
            "property_kind": "object",
        },
    )


@_dataclass(eq=False, kw_only=True)
class ManipulationExecutiveOperatingSystem(DulDescription):
    """Structural representation of ManipulationExecutiveOperatingSystem."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#ManipulationExecutiveOperatingSystem"
    )
    provides: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#provides",
            "ontology_range": "",
            "property_kind": "object",
        },
    )


@_dataclass(eq=False, kw_only=True)
class ManipulationFrame(DulDescription):
    """Structural representation of ManipulationFrame."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#ManipulationFrame"
    )
    frame_of: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#frameOf",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    has_frame_element: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#hasFrameElement",
            "ontology_range": "",
            "property_kind": "object",
        },
    )


@_dataclass(eq=False, kw_only=True)
class ManipulationOntologyAxiomatization(DulDescription):
    """Structural representation of ManipulationOntologyAxiomatization."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#ManipulationOntologyAxiomatization"
    )
    answers: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#answers",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    constrains: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#constrains",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    satisfied_by: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#satisfiedBy",
            "ontology_range": "",
            "property_kind": "object",
        },
    )


@_dataclass(eq=False, kw_only=True)
class ManipulationProblem(DulDescription, CaseLevelEntity):
    """Structural representation of ManipulationProblem."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#ManipulationProblem"
    )
    assigns_class: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#assignsClass",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    classifies: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#classifies",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    has_goal: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#hasGoal",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    relative_to: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#relativeTo",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    is_grounded: list[bool] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#isGrounded",
            "ontology_range": "http://www.w3.org/2001/XMLSchema#boolean",
            "property_kind": "datatype",
        },
    )


@_dataclass(eq=False, kw_only=True)
class ManipulationProblemClass(DulDescription, MetaLevelEntity):
    """Structural representation of ManipulationProblemClass."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#ManipulationProblemClass"
    )
    constrains_rung: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#constrainsRung",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    generates: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#generates",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    has_critic_set: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#hasCriticSet",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    has_mastery_criterion: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#hasMasteryCriterion",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    has_nuisance_model: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#hasNuisanceModel",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    has_rwms_family: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#hasRWMSFamily",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    has_task_scope: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#hasTaskScope",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    specified_by: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#specifiedBy",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    is_generator: list[bool] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#isGenerator",
            "ontology_range": "http://www.w3.org/2001/XMLSchema#boolean",
            "property_kind": "datatype",
        },
    )
    is_rwms_component: list[bool] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#isRWMSComponent",
            "ontology_range": "http://www.w3.org/2001/XMLSchema#boolean",
            "property_kind": "datatype",
        },
    )


@_dataclass(eq=False, kw_only=True)
class ManipulationTaskSpecification(DulDescription):
    """Structural representation of ManipulationTaskSpecification."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#ManipulationTaskSpecification"
    )
    has_goal: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#hasGoal",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    has_quality_functional: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#hasQualityFunctional",
            "ontology_range": "",
            "property_kind": "object",
        },
    )


@_dataclass(eq=False, kw_only=True)
class MicrotheoryTemplate(DulDescription):
    """Structural representation of MicrotheoryTemplate."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#MicrotheoryTemplate"
    )
    has_question: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#hasQuestion",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    template_for: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#templateFor",
            "ontology_range": "",
            "property_kind": "object",
        },
    )


@_dataclass(eq=False, kw_only=True)
class ModeIndexedDynamics(DulDescription):
    """Structural representation of ModeIndexedDynamics."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#ModeIndexedDynamics"
    )
    local_model_per_mode: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#localModelPerMode",
            "ontology_range": "",
            "property_kind": "object",
        },
    )


@_dataclass(eq=False, kw_only=True)
class ModelComponent(DulDescription):
    """Structural representation of ModelComponent."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#ModelComponent"
    )
    trained_on_distribution: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#trainedOnDistribution",
            "ontology_range": "",
            "property_kind": "object",
        },
    )


@_dataclass(eq=False, kw_only=True)
class ModelGuarantee(DulDescription):
    """Structural representation of ModelGuarantee."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#ModelGuarantee"
    )
    conditional_on: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#conditionalOn",
            "ontology_range": "",
            "property_kind": "object",
        },
    )


@_dataclass(eq=False, kw_only=True)
class ModelLearning(DulDescription, ComputationalProcess):
    """Structural representation of ModelLearning."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#ModelLearning"
    )


@_dataclass(eq=False, kw_only=True)
class Monitor(DulDescription, ComputationalProcess):
    """Structural representation of Monitor."""

    __ontology_iri__: _ClassVar[str] = "http://aicor.knowledge/l2-ontology.owl#Monitor"


@_dataclass(eq=False, kw_only=True)
class NecessityClaim(DulDescription):
    """Structural representation of NecessityClaim."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#NecessityClaim"
    )
    over_executives_of: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#overExecutivesOf",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    requires_property: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#requiresProperty",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    claim_scope: list[str] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#claimScope",
            "ontology_range": "http://www.w3.org/2001/XMLSchema#string",
            "property_kind": "datatype",
        },
    )


@_dataclass(eq=False, kw_only=True)
class NuisanceModel(DulDescription):
    """Structural representation of NuisanceModel."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#NuisanceModel"
    )
    defines_robustness_range: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#definesRobustnessRange",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    has_component: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#hasComponent",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    has_nuisance_kind: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#hasNuisanceKind",
            "ontology_range": "",
            "property_kind": "object",
        },
    )


@_dataclass(eq=False, kw_only=True)
class ObjectLevelRequirement(DulDescription):
    """Structural representation of ObjectLevelRequirement."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#ObjectLevelRequirement"
    )
    checked_by: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#checkedBy",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    constrains: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#constrains",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    half_of: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#halfOf",
            "ontology_range": "",
            "property_kind": "object",
        },
    )


@_dataclass(eq=False, kw_only=True)
class ObjectTransformation(DulDescription, ComputationalProcess):
    """Structural representation of ObjectTransformation."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#ObjectTransformation"
    )
    transforms_entity: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#transformsEntity",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    creates_or_destroys_objects: list[_Any] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#createsOrDestroysObjects",
            "ontology_range": "",
            "property_kind": "datatype",
        },
    )
    over_mereology: list[_Any] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#overMereology",
            "ontology_range": "",
            "property_kind": "datatype",
        },
    )


@_dataclass(eq=False, kw_only=True)
class ObservationChannelModel(DulDescription):
    """Structural representation of ObservationChannelModel."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#ObservationChannelModel"
    )


@_dataclass(eq=False, kw_only=True)
class ObservationFunction(DulDescription):
    """Structural representation of ObservationFunction."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#ObservationFunction"
    )
    surfaces: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#surfaces",
            "ontology_range": "",
            "property_kind": "object",
        },
    )


@_dataclass(eq=False, kw_only=True)
class ObservationModel(DulDescription):
    """Structural representation of ObservationModel."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#ObservationModel"
    )
    conditioned_on_state: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#conditionedOnState",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    produces_observation: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#producesObservation",
            "ontology_range": "",
            "property_kind": "object",
        },
    )


@_dataclass(eq=False, kw_only=True)
class PerceptionMethod(DulDescription, ComputationalProcess):
    """Structural representation of PerceptionMethod."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#PerceptionMethod"
    )


@_dataclass(eq=False, kw_only=True)
class Phenomenon(DulDescription):
    """Structural representation of Phenomenon."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#Phenomenon"
    )
    anchored_by_episode: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#anchoredByEpisode",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    explained_by_discipline: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#explainedByDiscipline",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    step_reached: list[int] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#stepReached",
            "ontology_range": "http://www.w3.org/2001/XMLSchema#integer",
            "property_kind": "datatype",
        },
    )


@_dataclass(eq=False, kw_only=True)
class PlanInterpreter(DulDescription, ComputationalProcess):
    """Structural representation of PlanInterpreter."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#PlanInterpreter"
    )


@_dataclass(eq=False, kw_only=True)
class PlanLearning(DulDescription, ComputationalProcess):
    """Structural representation of PlanLearning."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#PlanLearning"
    )


@_dataclass(eq=False, kw_only=True)
class PlanManagement(DulDescription, ComputationalProcess):
    """Structural representation of PlanManagement."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#PlanManagement"
    )
    monitors: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#monitors",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    relative_to: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#relativeTo",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    is_concurrent: list[_Any] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#isConcurrent",
            "ontology_range": "",
            "property_kind": "datatype",
        },
    )


@_dataclass(eq=False, kw_only=True)
class PlanTransformation(DulDescription, ComputationalProcess):
    """Structural representation of PlanTransformation."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#PlanTransformation"
    )
    edits: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#edits",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    fired_by: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#firedBy",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    transforms: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#transforms",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    is_interruptible: list[_Any] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#isInterruptible",
            "ontology_range": "",
            "property_kind": "datatype",
        },
    )


@_dataclass(eq=False, kw_only=True)
class ProbabilisticHybridProcessModel(DulDescription):
    """Structural representation of ProbabilisticHybridProcessModel."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#ProbabilisticHybridProcessModel"
    )
    formalizes: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#formalizes",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    has_continuous_state: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#hasContinuousState",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    has_mode_set: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#hasModeSet",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    has_successor_distribution: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#hasSuccessorDistribution",
            "ontology_range": "",
            "property_kind": "object",
        },
    )


@_dataclass(eq=False, kw_only=True)
class ProblemClassSpecificationTemplate(DulDescription):
    """Structural representation of ProblemClassSpecificationTemplate."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#ProblemClassSpecificationTemplate"
    )
    has_field: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#hasField",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    requires_mastery_criterion: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#requiresMasteryCriterion",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    template_for: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#templateFor",
            "ontology_range": "",
            "property_kind": "object",
        },
    )


@_dataclass(eq=False, kw_only=True)
class ProblemDomain(DulDescription, MetaLevelEntity):
    """Structural representation of ProblemDomain."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#ProblemDomain"
    )
    evidence_base_for: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#evidenceBaseFor",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    generated_by: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#generatedBy",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    has_member: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#hasMember",
            "ontology_range": "",
            "property_kind": "object",
        },
    )


@_dataclass(eq=False, kw_only=True)
class ProcessControlAlignment(DulDescription):
    """Structural representation of ProcessControlAlignment."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#ProcessControlAlignment"
    )
    constrains: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#constrains",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    requires_counterpart: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#requiresCounterpart",
            "ontology_range": "",
            "property_kind": "object",
        },
    )


@_dataclass(eq=False, kw_only=True)
class ProcessInterpretation(DulDescription):
    """Structural representation of ProcessInterpretation."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#ProcessInterpretation"
    )
    applies_model: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#appliesModel",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    confers_force: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#confersForce",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    interprets: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#interprets",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    yields: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#yields",
            "ontology_range": "",
            "property_kind": "object",
        },
    )


@_dataclass(eq=False, kw_only=True)
class ProcessMonitor(DulDescription):
    """Structural representation of ProcessMonitor."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#ProcessMonitor"
    )
    checks: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#checks",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    monitor_for: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#monitorFor",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    operationalizes: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#operationalizes",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    produces_record_in: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#producesRecordIn",
            "ontology_range": "",
            "property_kind": "object",
        },
    )


@_dataclass(eq=False, kw_only=True)
class ProposalToCommitmentPipeline(DulDescription):
    """Structural representation of ProposalToCommitmentPipeline."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#ProposalToCommitmentPipeline"
    )
    has_stage: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#hasStage",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    operationalizes: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#operationalizes",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    pipeline_of: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#pipelineOf",
            "ontology_range": "",
            "property_kind": "object",
        },
    )


@_dataclass(eq=False, kw_only=True)
class QueryTemplate(DulDescription):
    """Structural representation of QueryTemplate."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#QueryTemplate"
    )
    has_parameter: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#hasParameter",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    ranges_over: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#rangesOver",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    template_of: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#templateOf",
            "ontology_range": "",
            "property_kind": "object",
        },
    )


@_dataclass(eq=False, kw_only=True)
class RwmsFamily(DulDescription, MetaLevelEntity):
    """Structural representation of RwmsFamily."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#RWMSFamily"
    )
    has_body_world_family: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#hasBodyWorldFamily",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    has_coupling_relation: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#hasCouplingRelation",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    has_executive_family: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#hasExecutiveFamily",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    makes_family_relative: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#makesFamilyRelative",
            "ontology_range": "",
            "property_kind": "object",
        },
    )


@_dataclass(eq=False, kw_only=True)
class RealizationChain(DulDescription):
    """Structural representation of RealizationChain."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#RealizationChain"
    )
    has_first_link: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#hasFirstLink",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    has_link: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#hasLink",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    realizes_commitment: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#realizesCommitment",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    forbids_shortcut: list[bool] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#forbidsShortcut",
            "ontology_range": "http://www.w3.org/2001/XMLSchema#boolean",
            "property_kind": "datatype",
        },
    )


@_dataclass(eq=False, kw_only=True)
class RealizationInterface(DulDescription):
    """Structural representation of RealizationInterface."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#RealizationInterface"
    )


@_dataclass(eq=False, kw_only=True)
class RealizationLink(DulDescription):
    """Structural representation of RealizationLink."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#RealizationLink"
    )
    failure_routes_to: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#failureRoutesTo",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    next_link: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#nextLink",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    can_depart_from_prediction: list[bool] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#canDepartFromPrediction",
            "ontology_range": "http://www.w3.org/2001/XMLSchema#boolean",
            "property_kind": "datatype",
        },
    )


@_dataclass(eq=False, kw_only=True)
class ReasoningEngine(DulDescription):
    """Structural representation of ReasoningEngine."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#ReasoningEngine"
    )


@_dataclass(eq=False, kw_only=True)
class ReferentialPersistence(DulDescription, ComputationalProcess):
    """Structural representation of ReferentialPersistence."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#ReferentialPersistence"
    )
    tracks_referent_of: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#tracksReferentOf",
            "ontology_range": "",
            "property_kind": "object",
        },
    )


@_dataclass(eq=False, kw_only=True)
class ReflectiveRequirement(DulDescription):
    """Structural representation of ReflectiveRequirement."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#ReflectiveRequirement"
    )
    constrains: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#constrains",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    enables: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#enables",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    half_of: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#halfOf",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    requires: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#requires",
            "ontology_range": "",
            "property_kind": "object",
        },
    )


@_dataclass(eq=False, kw_only=True)
class RelationalGrammar(DulDescription):
    """Structural representation of RelationalGrammar."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#RelationalGrammar"
    )
    has_relation_family: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#hasRelationFamily",
            "ontology_range": "",
            "property_kind": "object",
        },
    )


@_dataclass(eq=False, kw_only=True)
class RenderingInterface(DulDescription):
    """Structural representation of RenderingInterface."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#RenderingInterface"
    )


@_dataclass(eq=False, kw_only=True)
class Repair(DulDescription):
    """Structural representation of Repair."""

    __ontology_iri__: _ClassVar[str] = "http://aicor.knowledge/l2-ontology.owl#Repair"
    accountable_via: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#accountableVia",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    has_form: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#hasForm",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    responds_to: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#respondsTo",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    transforms: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#transforms",
            "ontology_range": "",
            "property_kind": "object",
        },
    )


@_dataclass(eq=False, kw_only=True)
class RepresentationLanguage(DulDescription):
    """Structural representation of RepresentationLanguage."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#RepresentationLanguage"
    )
    expresses_projection: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#expressesProjection",
            "ontology_range": "",
            "property_kind": "object",
        },
    )


@_dataclass(eq=False, kw_only=True)
class RepresentationProcessCapabilityTriad(DulDescription):
    """Structural representation of RepresentationProcessCapabilityTriad."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#RepresentationProcessCapabilityTriad"
    )
    has_process: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#hasProcess",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    has_representation: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#hasRepresentation",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    yields_capability: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#yieldsCapability",
            "ontology_range": "",
            "property_kind": "object",
        },
    )


@_dataclass(eq=False, kw_only=True)
class RepresentationalObligation(DulDescription):
    """Structural representation of RepresentationalObligation."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#RepresentationalObligation"
    )
    derived_from: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#derivedFrom",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    obligates: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#obligates",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    scoped_by: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#scopedBy",
            "ontology_range": "",
            "property_kind": "object",
        },
    )


@_dataclass(eq=False, kw_only=True)
class RequestOutcome(DulDescription):
    """Structural representation of RequestOutcome."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#RequestOutcome"
    )
    assesses_episode: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#assessesEpisode",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    relative_to_problem: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#relativeToProblem",
            "ontology_range": "",
            "property_kind": "object",
        },
    )


@_dataclass(eq=False, kw_only=True)
class RobotControlLaw(DulDescription):
    """Structural representation of RobotControlLaw."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#RobotControlLaw"
    )
    directed_by: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#directedBy",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    produces_input: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#producesInput",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    reads_state: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#readsState",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    realizes: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#realizes",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    uses: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#uses",
            "ontology_range": "",
            "property_kind": "object",
        },
    )


@_dataclass(eq=False, kw_only=True)
class RoleBinding(DulDescription):
    """Structural representation of RoleBinding."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#RoleBinding"
    )
    binds_filler: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#bindsFiller",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    binds_slot: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#bindsSlot",
            "ontology_range": "",
            "property_kind": "object",
        },
    )


@_dataclass(eq=False, kw_only=True)
class RuntimePlanning(DulDescription, ComputationalProcess):
    """Structural representation of RuntimePlanning."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#RuntimePlanning"
    )
    applies_interruptibly: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#appliesInterruptibly",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    outputs_via: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#outputsVia",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    runs_as_control_process: list[_Any] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#runsAsControlProcess",
            "ontology_range": "",
            "property_kind": "datatype",
        },
    )


@_dataclass(eq=False, kw_only=True)
class RuntimeReasonablePlan(DulDescription):
    """Structural representation of RuntimeReasonablePlan."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#RuntimeReasonablePlan"
    )
    enables_local_repair: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#enablesLocalRepair",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    is_executable: list[_Any] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#isExecutable",
            "ontology_range": "",
            "property_kind": "datatype",
        },
    )
    is_reasonable_at_runtime: list[_Any] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#isReasonableAtRuntime",
            "ontology_range": "",
            "property_kind": "datatype",
        },
    )


@_dataclass(eq=False, kw_only=True)
class Scenario(DulDescription):
    """Structural representation of Scenario."""

    __ontology_iri__: _ClassVar[str] = "http://aicor.knowledge/l2-ontology.owl#Scenario"
    is_about: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#isAbout",
            "ontology_range": "",
            "property_kind": "object",
        },
    )


@_dataclass(eq=False, kw_only=True)
class SceneGraphToKnowledgeGraph(DulDescription):
    """Structural representation of SceneGraphToKnowledgeGraph."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#SceneGraphToKnowledgeGraph"
    )
    applies_step: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#appliesStep",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    consumes: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#consumes",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    pivots_on: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#pivotsOn",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    produces: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#produces",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    uses: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#uses",
            "ontology_range": "",
            "property_kind": "object",
        },
    )


@_dataclass(eq=False, kw_only=True)
class ScienceDiscipline(DulDescription):
    """Structural representation of ScienceDiscipline."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#ScienceDiscipline"
    )
    explains_phenomenon: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#explainsPhenomenon",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    studies_object: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#studiesObject",
            "ontology_range": "",
            "property_kind": "object",
        },
    )


@_dataclass(eq=False, kw_only=True)
class ScientificObject(DulDescription):
    """Structural representation of ScientificObject."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#ScientificObject"
    )
    about_referent: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#aboutReferent",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    studied_by: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#studiedBy",
            "ontology_range": "",
            "property_kind": "object",
        },
    )


@_dataclass(eq=False, kw_only=True)
class ScopeCard(DulDescription):
    """Structural representation of ScopeCard."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#ScopeCard"
    )
    assumes_embodiment_class: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#assumesEmbodimentClass",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    attested_by: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#attestedBy",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    covers_problem_class: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#coversProblemClass",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    declares_exclusion: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#declaresExclusion",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    declares_nuisance_range: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#declaresNuisanceRange",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    excludes_claim_type: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#excludesClaimType",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    has_scope_condition: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#hasScopeCondition",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    models_channel: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#modelsChannel",
            "ontology_range": "",
            "property_kind": "object",
        },
    )


@_dataclass(eq=False, kw_only=True)
class ScopeCondition(DulDescription):
    """Structural representation of ScopeCondition."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#ScopeCondition"
    )


@_dataclass(eq=False, kw_only=True)
class ScopeExclusion(DulDescription):
    """Structural representation of ScopeExclusion."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#ScopeExclusion"
    )


@_dataclass(eq=False, kw_only=True)
class ScopedClaim(DulDescription):
    """Structural representation of ScopedClaim."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#ScopedClaim"
    )
    established_by: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#establishedBy",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    is_universally_quantified: list[bool] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#isUniversallyQuantified",
            "ontology_range": "http://www.w3.org/2001/XMLSchema#boolean",
            "property_kind": "datatype",
        },
    )


@_dataclass(eq=False, kw_only=True)
class SemanticTagging(DulDescription):
    """Structural representation of SemanticTagging."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#SemanticTagging"
    )
    grounds: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#grounds",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    links_prim_to: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#linksPrimTo",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    produces: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#produces",
            "ontology_range": "",
            "property_kind": "object",
        },
    )


@_dataclass(eq=False, kw_only=True)
class SolutionMethod(DulDescription):
    """Structural representation of SolutionMethod."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#SolutionMethod"
    )
    exploits_constraint: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#exploitsConstraint",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    produces_solution_for: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#producesSolutionFor",
            "ontology_range": "",
            "property_kind": "object",
        },
    )


@_dataclass(eq=False, kw_only=True)
class SolvabilityClaim(DulDescription):
    """Structural representation of SolvabilityClaim."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#SolvabilityClaim"
    )
    asserts_existence_over: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#assertsExistenceOver",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    claim_scope: list[str] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#claimScope",
            "ontology_range": "http://www.w3.org/2001/XMLSchema#string",
            "property_kind": "datatype",
        },
    )


@_dataclass(eq=False, kw_only=True)
class SolvabilityFailureMode(DulDescription):
    """Structural representation of SolvabilityFailureMode."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#SolvabilityFailureMode"
    )


@_dataclass(eq=False, kw_only=True)
class SolvabilityRung(DulDescription):
    """Structural representation of SolvabilityRung."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#SolvabilityRung"
    )
    grounded_in_layer: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#groundedInLayer",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    has_rung_failure_mode: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#hasRungFailureMode",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    rung_of: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#rungOf",
            "ontology_range": "",
            "property_kind": "object",
        },
    )


@_dataclass(eq=False, kw_only=True)
class SolvabilityStructure(DulDescription):
    """Structural representation of SolvabilityStructure."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#SolvabilityStructure"
    )
    for_problem: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#forProblem",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    has_rung: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#hasRung",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    structure_of: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#structureOf",
            "ontology_range": "",
            "property_kind": "object",
        },
    )


@_dataclass(eq=False, kw_only=True)
class StateOfUnderstanding(DulDescription):
    """Structural representation of StateOfUnderstanding."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#StateOfUnderstanding"
    )
    collects_claim: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#collectsClaim",
            "ontology_range": "",
            "property_kind": "object",
        },
    )


@_dataclass(eq=False, kw_only=True)
class StrengtheningCost(DulDescription):
    """Structural representation of StrengtheningCost."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#StrengtheningCost"
    )


@_dataclass(eq=False, kw_only=True)
class StrengtheningModule(DulDescription):
    """Structural representation of StrengtheningModule."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#StrengtheningModule"
    )
    has_cost: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#hasCost",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    unlocks: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#unlocks",
            "ontology_range": "",
            "property_kind": "object",
        },
    )


@_dataclass(eq=False, kw_only=True)
class SubstrateRelativeOptimality(DulDescription):
    """Structural representation of SubstrateRelativeOptimality."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#SubstrateRelativeOptimality"
    )
    optimal_design: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#optimalDesign",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    optimal_over_family: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#optimalOverFamily",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    presupposes_rationality: list[bool] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#presupposesRationality",
            "ontology_range": "http://www.w3.org/2001/XMLSchema#boolean",
            "property_kind": "datatype",
        },
    )


@_dataclass(eq=False, kw_only=True)
class SubstrateSufficiency(DulDescription):
    """Structural representation of SubstrateSufficiency."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#SubstrateSufficiency"
    )
    grounds_property: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#groundsProperty",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    maps_to_rung: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#mapsToRung",
            "ontology_range": "",
            "property_kind": "object",
        },
    )


@_dataclass(eq=False, kw_only=True)
class SynergisticModelCombination(DulDescription):
    """Structural representation of SynergisticModelCombination."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#SynergisticModelCombination"
    )
    combines_statistical_model: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#combinesStatisticalModel",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    combines_structural_model: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#combinesStructuralModel",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    operationalized_by: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#operationalizedBy",
            "ontology_range": "",
            "property_kind": "object",
        },
    )


@_dataclass(eq=False, kw_only=True)
class TaskEnvironmentEmbodimentClass(DulDescription, MetaLevelEntity):
    """Structural representation of TaskEnvironmentEmbodimentClass."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#TaskEnvironmentEmbodimentClass"
    )
    admits_embodiment: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#admitsEmbodiment",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    admits_environment_model: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#admitsEnvironmentModel",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    admits_task_type: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#admitsTaskType",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    governed_by_physics: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#governedByPhysics",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    has_validity_interval: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#hasValidityInterval",
            "ontology_range": "",
            "property_kind": "object",
        },
    )


@_dataclass(eq=False, kw_only=True)
class TaskGrounding(DulDescription):
    """Structural representation of TaskGrounding."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#TaskGrounding"
    )
    consumes: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#consumes",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    grounds: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#grounds",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    has_outcome: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#hasOutcome",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    produces: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#produces",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    relative_to: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#relativeTo",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    yields: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#yields",
            "ontology_range": "",
            "property_kind": "object",
        },
    )


@_dataclass(eq=False, kw_only=True)
class TaskInstance(DulDescription):
    """Structural representation of TaskInstance."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#TaskInstance"
    )
    grounds_to: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#groundsTo",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    sampled_from: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#sampledFrom",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    underspecified: list[bool] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#underspecified",
            "ontology_range": "http://www.w3.org/2001/XMLSchema#boolean",
            "property_kind": "datatype",
        },
    )


@_dataclass(eq=False, kw_only=True)
class TaskReading(DulDescription):
    """Structural representation of TaskReading."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#TaskReading"
    )


@_dataclass(eq=False, kw_only=True)
class TaskScope(DulDescription):
    """Structural representation of TaskScope."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#TaskScope"
    )
    generates_instance: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#generatesInstance",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    has_admissibility_constraint: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#hasAdmissibilityConstraint",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    has_interpretation_modality: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#hasInterpretationModality",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    has_preference: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#hasPreference",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    has_task_generator: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#hasTaskGenerator",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    has_task_source: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#hasTaskSource",
            "ontology_range": "",
            "property_kind": "object",
        },
    )


@_dataclass(eq=False, kw_only=True)
class TrainingDistribution(DulDescription):
    """Structural representation of TrainingDistribution."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#TrainingDistribution"
    )


@_dataclass(eq=False, kw_only=True)
class Transferability(DulDescription):
    """Structural representation of Transferability."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#Transferability"
    )
    achieved_by: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#achievedBy",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    asserted_by: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#assertedBy",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    transfers_across: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#transfersAcross",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    requires_limited_reprogramming: list[_Any] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#requiresLimitedReprogramming",
            "ontology_range": "",
            "property_kind": "datatype",
        },
    )


@_dataclass(eq=False, kw_only=True)
class TranslationInterface(DulDescription):
    """Structural representation of TranslationInterface."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#TranslationInterface"
    )
    interface_of: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#interfaceOf",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    maps_from: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#mapsFrom",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    produces_target: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#producesTarget",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    is_commitment_sensitive: list[_Any] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#isCommitmentSensitive",
            "ontology_range": "",
            "property_kind": "datatype",
        },
    )


@_dataclass(eq=False, kw_only=True)
class TranslationProgram(DulDescription):
    """Structural representation of TranslationProgram."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#TranslationProgram"
    )
    into_problem_language: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#intoProblemLanguage",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    translates_episode: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#translatesEpisode",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    via_projection: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#viaProjection",
            "ontology_range": "",
            "property_kind": "object",
        },
    )


@_dataclass(eq=False, kw_only=True)
class ValidityCondition(DulDescription):
    """Structural representation of ValidityCondition."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#ValidityCondition"
    )


@_dataclass(eq=False, kw_only=True)
class WarrantedExecutiveCommitment(DulDescription):
    """Structural representation of WarrantedExecutiveCommitment."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#WarrantedExecutiveCommitment"
    )
    adopts_candidate: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#adoptsCandidate",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    has_response_policy: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#hasResponsePolicy",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    has_validity_condition: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#hasValidityCondition",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    under_warrant: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#underWarrant",
            "ontology_range": "",
            "property_kind": "object",
        },
    )


@_dataclass(eq=False, kw_only=True)
class WorkingMemory(DulDescription):
    """Structural representation of WorkingMemory."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#WorkingMemory"
    )
    has_part: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#hasPart",
            "ontology_range": "",
            "property_kind": "object",
        },
    )


@_dataclass(eq=False, kw_only=True)
class WorldStateRepresentation(DulDescription):
    """Structural representation of WorldStateRepresentation."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#WorldStateRepresentation"
    )


@_dataclass(eq=False, kw_only=True)
class AnswerSubstitution(DulInformationObject):
    """Structural representation of AnswerSubstitution."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#AnswerSubstitution"
    )
    answers_query: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#answersQuery",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    has_binding: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#hasBinding",
            "ontology_range": "",
            "property_kind": "object",
        },
    )


@_dataclass(eq=False, kw_only=True)
class AttributeConstraint(DulInformationObject):
    """Structural representation of AttributeConstraint."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#AttributeConstraint"
    )


@_dataclass(eq=False, kw_only=True)
class Binding(DulInformationObject):
    """Structural representation of Binding."""

    __ontology_iri__: _ClassVar[str] = "http://aicor.knowledge/l2-ontology.owl#Binding"
    binds_variable: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#bindsVariable",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    to_value: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#toValue",
            "ontology_range": "",
            "property_kind": "object",
        },
    )


@_dataclass(eq=False, kw_only=True)
class ComputationalMedium(DulInformationObject):
    """Structural representation of ComputationalMedium."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#ComputationalMedium"
    )
    characterized_by: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#characterizedBy",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    makes_available: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#makesAvailable",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    realizes_via: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#realizesVia",
            "ontology_range": "",
            "property_kind": "object",
        },
    )


@_dataclass(eq=False, kw_only=True)
class DataGenerationEngine(DulInformationObject, MetaLevelEntity):
    """Structural representation of DataGenerationEngine."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#DataGenerationEngine"
    )
    uses_model: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#usesModel",
            "ontology_range": "",
            "property_kind": "object",
        },
    )


@_dataclass(eq=False, kw_only=True)
class DigitalBodyWorldSystem(DulInformationObject):
    """Structural representation of DigitalBodyWorldSystem."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#DigitalBodyWorldSystem"
    )
    has_expected_observation: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#hasExpectedObservation",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    has_predictor: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#hasPredictor",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    has_scope_card: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#hasScopeCard",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    has_structural_core: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#hasStructuralCore",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    is_model_of: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#isModelOf",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    is_physical: list[bool] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#isPhysical",
            "ontology_range": "http://www.w3.org/2001/XMLSchema#boolean",
            "property_kind": "datatype",
        },
    )
    is_specification: list[bool] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#isSpecification",
            "ontology_range": "http://www.w3.org/2001/XMLSchema#boolean",
            "property_kind": "datatype",
        },
    )


@_dataclass(eq=False, kw_only=True)
class DigitalResearchLaboratory(DulInformationObject, MetaLevelEntity):
    """Structural representation of DigitalResearchLaboratory."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#DigitalResearchLaboratory"
    )
    uses_model: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#usesModel",
            "ontology_range": "",
            "property_kind": "object",
        },
    )


@_dataclass(eq=False, kw_only=True)
class EntityQuery(DulInformationObject):
    """Structural representation of EntityQuery."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#EntityQuery"
    )
    evaluated_against: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#evaluatedAgainst",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    expressed_in: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#expressedIn",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    has_partial_description: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#hasPartialDescription",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    has_query_variable: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#hasQueryVariable",
            "ontology_range": "",
            "property_kind": "object",
        },
    )


@_dataclass(eq=False, kw_only=True)
class EntityRelationQueryLanguage(DulInformationObject):
    """Structural representation of EntityRelationQueryLanguage."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#EntityRelationQueryLanguage"
    )
    describes_entity_via: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#describesEntityVia",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    queries_model_of: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#queriesModelOf",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    realizes: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#realizes",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    supports_nesting: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#supportsNesting",
            "ontology_range": "",
            "property_kind": "object",
        },
    )


@_dataclass(eq=False, kw_only=True)
class EnvisionedTimeline(DulInformationObject):
    """Structural representation of EnvisionedTimeline."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#EnvisionedTimeline"
    )
    grounds_predicate: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#groundsPredicate",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    produced_by_envisioning: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#producedByEnvisioning",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    projection_of_branch: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#projectionOfBranch",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    relative_to_scenario: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#relativeToScenario",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    translated_from: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#translatedFrom",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    is_grounded_in_simulation: list[bool] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#isGroundedInSimulation",
            "ontology_range": "http://www.w3.org/2001/XMLSchema#boolean",
            "property_kind": "datatype",
        },
    )


@_dataclass(eq=False, kw_only=True)
class EvidenceBearingTrace(DulInformationObject, CaseLevelEntity):
    """Structural representation of EvidenceBearingTrace."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#EvidenceBearingTrace"
    )
    interprets: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#interprets",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    part_of_scientific_object: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#partOfScientificObject",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    records: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#records",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    relative_to: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#relativeTo",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    is_critic_relative: list[bool] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#isCriticRelative",
            "ontology_range": "http://www.w3.org/2001/XMLSchema#boolean",
            "property_kind": "datatype",
        },
    )
    is_raw_log: list[bool] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#isRawLog",
            "ontology_range": "http://www.w3.org/2001/XMLSchema#boolean",
            "property_kind": "datatype",
        },
    )


@_dataclass(eq=False, kw_only=True)
class ExecutableMachineryLayer(DulInformationObject):
    """Structural representation of ExecutableMachineryLayer."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#ExecutableMachineryLayer"
    )


@_dataclass(eq=False, kw_only=True)
class ExecutablePrediction(DulInformationObject):
    """Structural representation of ExecutablePrediction."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#ExecutablePrediction"
    )
    answers_query: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#answersQuery",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    applies_to_coupling: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#appliesToCoupling",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    predicts_evolution_of: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#predictsEvolutionOf",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    is_model_quantity: list[bool] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#isModelQuantity",
            "ontology_range": "http://www.w3.org/2001/XMLSchema#boolean",
            "property_kind": "datatype",
        },
    )


@_dataclass(eq=False, kw_only=True)
class ExecutionStateBelief(DulInformationObject):
    """Structural representation of ExecutionStateBelief."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#ExecutionStateBelief"
    )
    about_execution_of: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#aboutExecutionOf",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    records_binding: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#recordsBinding",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    records_status: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#recordsStatus",
            "ontology_range": "",
            "property_kind": "object",
        },
    )


@_dataclass(eq=False, kw_only=True)
class ExpectedObservation(DulInformationObject):
    """Structural representation of ExpectedObservation."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#ExpectedObservation"
    )
    generated_by: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#generatedBy",
            "ontology_range": "",
            "property_kind": "object",
        },
    )


@_dataclass(eq=False, kw_only=True)
class FailureId(DulInformationObject):
    """Structural representation of FailureId."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#FailureId"
    )


@_dataclass(eq=False, kw_only=True)
class FailureSeverity(DulInformationObject):
    """Structural representation of FailureSeverity."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#FailureSeverity"
    )
    severity_value: list[int] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#severityValue",
            "ontology_range": "http://www.w3.org/2001/XMLSchema#integer",
            "property_kind": "datatype",
        },
    )


@_dataclass(eq=False, kw_only=True)
class FoundationModel(DulInformationObject):
    """Structural representation of FoundationModel."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#FoundationModel"
    )
    contributes_at: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#contributesAt",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    emits_hypothesis: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#emitsHypothesis",
            "ontology_range": "",
            "property_kind": "object",
        },
    )


@_dataclass(eq=False, kw_only=True)
class GroundingRecord(DulInformationObject):
    """Structural representation of GroundingRecord."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#GroundingRecord"
    )
    component_of: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#componentOf",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    has_field: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#hasField",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    produced_by: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#producedBy",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    is_revisable: list[_Any] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#isRevisable",
            "ontology_range": "",
            "property_kind": "datatype",
        },
    )


@_dataclass(eq=False, kw_only=True)
class HierarchicalSceneStructure(DulInformationObject):
    """Structural representation of HierarchicalSceneStructure."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#HierarchicalSceneStructure"
    )
    has_node: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#hasNode",
            "ontology_range": "",
            "property_kind": "object",
        },
    )


@_dataclass(eq=False, kw_only=True)
class IntervalTemporalLogic(DulInformationObject):
    """Structural representation of IntervalTemporalLogic."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#IntervalTemporalLogic"
    )
    has_predicate: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#hasPredicate",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    reasons_over: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#reasonsOver",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    reifies: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#reifies",
            "ontology_range": "",
            "property_kind": "object",
        },
    )


@_dataclass(eq=False, kw_only=True)
class KnowledgeGraph(DulInformationObject):
    """Structural representation of KnowledgeGraph."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#KnowledgeGraph"
    )
    classified_by: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#classifiedBy",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    grounded_in: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#groundedIn",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    has_triple: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#hasTriple",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    is_about: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#isAbout",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    produced_from: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#producedFrom",
            "ontology_range": "",
            "property_kind": "object",
        },
    )


@_dataclass(eq=False, kw_only=True)
class KnowledgeQueryLanguage(DulInformationObject):
    """Structural representation of KnowledgeQueryLanguage."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#KnowledgeQueryLanguage"
    )
    documents: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#documents",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    provides_implementation: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#providesImplementation",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    queries_over: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#queriesOver",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    realizes: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#realizes",
            "ontology_range": "",
            "property_kind": "object",
        },
    )


@_dataclass(eq=False, kw_only=True)
class LinkElement(DulInformationObject):
    """Structural representation of LinkElement."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#LinkElement"
    )
    has_inertial: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#hasInertial",
            "ontology_range": "",
            "property_kind": "object",
        },
    )


@_dataclass(eq=False, kw_only=True)
class Observation(DulInformationObject):
    """Structural representation of Observation."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#Observation"
    )


@_dataclass(eq=False, kw_only=True)
class PartialDescription(DulInformationObject):
    """Structural representation of PartialDescription."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#PartialDescription"
    )
    describes_as: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#describesAs",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    has_type_constraint: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#hasTypeConstraint",
            "ontology_range": "",
            "property_kind": "object",
        },
    )


@_dataclass(eq=False, kw_only=True)
class PhysicsEngine(DulInformationObject):
    """Structural representation of PhysicsEngine."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#PhysicsEngine"
    )
    consumes: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#consumes",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    integrates: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#integrates",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    realizes: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#realizes",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    resolves_contact_by: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#resolvesContactBy",
            "ontology_range": "",
            "property_kind": "object",
        },
    )


@_dataclass(eq=False, kw_only=True)
class ProspectiveBelief(DulInformationObject):
    """Structural representation of ProspectiveBelief."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#ProspectiveBelief"
    )
    about_future_execution_of: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#aboutFutureExecutionOf",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    computed_by: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#computedBy",
            "ontology_range": "",
            "property_kind": "object",
        },
    )


@_dataclass(eq=False, kw_only=True)
class QueryConnective(DulInformationObject):
    """Structural representation of QueryConnective."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#QueryConnective"
    )
    combines: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#combines",
            "ontology_range": "",
            "property_kind": "object",
        },
    )


@_dataclass(eq=False, kw_only=True)
class QueryVariable(DulInformationObject):
    """Structural representation of QueryVariable."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#QueryVariable"
    )
    is_bound: list[bool] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#isBound",
            "ontology_range": "http://www.w3.org/2001/XMLSchema#boolean",
            "property_kind": "datatype",
        },
    )


@_dataclass(eq=False, kw_only=True)
class Recording(DulInformationObject):
    """Structural representation of Recording."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#Recording"
    )
    is_about: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#isAbout",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    is_partial: list[bool] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#isPartial",
            "ontology_range": "http://www.w3.org/2001/XMLSchema#boolean",
            "property_kind": "datatype",
        },
    )


@_dataclass(eq=False, kw_only=True)
class ReflectiveRepresentation(DulInformationObject):
    """Structural representation of ReflectiveRepresentation."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#ReflectiveRepresentation"
    )
    enables: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#enables",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    represents_self_state_of: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#representsSelfStateOf",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    supports: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#supports",
            "ontology_range": "",
            "property_kind": "object",
        },
    )


@_dataclass(eq=False, kw_only=True)
class ReifiedTemporalEntity(DulInformationObject):
    """Structural representation of ReifiedTemporalEntity."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#ReifiedTemporalEntity"
    )


@_dataclass(eq=False, kw_only=True)
class Renderer(DulInformationObject):
    """Structural representation of Renderer."""

    __ontology_iri__: _ClassVar[str] = "http://aicor.knowledge/l2-ontology.owl#Renderer"
    consumes: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#consumes",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    instantiates: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#instantiates",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    produces: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#produces",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    realizes: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#realizes",
            "ontology_range": "",
            "property_kind": "object",
        },
    )


@_dataclass(eq=False, kw_only=True)
class ResearchInfrastructureLayer(DulInformationObject):
    """Structural representation of ResearchInfrastructureLayer."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#ResearchInfrastructureLayer"
    )


@_dataclass(eq=False, kw_only=True)
class ResearchSemanticDigitalTwin(DulInformationObject, MetaLevelEntity):
    """Structural representation of ResearchSemanticDigitalTwin."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#ResearchSemanticDigitalTwin"
    )
    instantiates_family: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#instantiatesFamily",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    supports_regime: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#supportsRegime",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    varies_nuisance: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#variesNuisance",
            "ontology_range": "",
            "property_kind": "object",
        },
    )


@_dataclass(eq=False, kw_only=True)
class RobotDescription(DulInformationObject):
    """Structural representation of RobotDescription."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#RobotDescription"
    )
    describes: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#describes",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    has_format: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#hasFormat",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    serializes: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#serializes",
            "ontology_range": "",
            "property_kind": "object",
        },
    )


@_dataclass(eq=False, kw_only=True)
class RobotSimulator(DulInformationObject):
    """Structural representation of RobotSimulator."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#RobotSimulator"
    )
    functions_as: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#functionsAs",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    has_part: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#hasPart",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    has_scope_card: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#hasScopeCard",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    loads: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#loads",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    realizes: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#realizes",
            "ontology_range": "",
            "property_kind": "object",
        },
    )


@_dataclass(eq=False, kw_only=True)
class SceneDescription(DulInformationObject):
    """Structural representation of SceneDescription."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#SceneDescription"
    )
    has_format: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#hasFormat",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    has_robot_subtree: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#hasRobotSubtree",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    references_asset: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#referencesAsset",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    serializes: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#serializes",
            "ontology_range": "",
            "property_kind": "object",
        },
    )


@_dataclass(eq=False, kw_only=True)
class SceneNode(DulInformationObject):
    """Structural representation of SceneNode."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#SceneNode"
    )


@_dataclass(eq=False, kw_only=True)
class SemanticDigitalTwin(DulInformationObject, CaseLevelEntity):
    """Structural representation of SemanticDigitalTwin."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#SemanticDigitalTwin"
    )
    has_node_structure: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#hasNodeStructure",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    is_about: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#isAbout",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    supports_operation: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#supportsOperation",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    is_belief_state: list[bool] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#isBeliefState",
            "ontology_range": "http://www.w3.org/2001/XMLSchema#boolean",
            "property_kind": "datatype",
        },
    )
    is_ground_truth: list[bool] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#isGroundTruth",
            "ontology_range": "http://www.w3.org/2001/XMLSchema#boolean",
            "property_kind": "datatype",
        },
    )


@_dataclass(eq=False, kw_only=True)
class SemanticReport(DulInformationObject):
    """Structural representation of SemanticReport."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#SemanticReport"
    )
    identifies: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#identifies",
            "ontology_range": "",
            "property_kind": "object",
        },
    )


@_dataclass(eq=False, kw_only=True)
class SensorModel(DulInformationObject):
    """Structural representation of SensorModel."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#SensorModel"
    )
    driven_by: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#drivenBy",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    model_of: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#modelOf",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    produces_synthetic: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#producesSynthetic",
            "ontology_range": "",
            "property_kind": "object",
        },
    )


@_dataclass(eq=False, kw_only=True)
class Signal(DulInformationObject):
    """Structural representation of Signal."""

    __ontology_iri__: _ClassVar[str] = "http://aicor.knowledge/l2-ontology.owl#Signal"


@_dataclass(eq=False, kw_only=True)
class SimulationAsset(DulInformationObject):
    """Structural representation of SimulationAsset."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#SimulationAsset"
    )
    has_asset_role: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#hasAssetRole",
            "ontology_range": "",
            "property_kind": "object",
        },
    )


@_dataclass(eq=False, kw_only=True)
class SimulationLog(DulInformationObject):
    """Structural representation of SimulationLog."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#SimulationLog"
    )


@_dataclass(eq=False, kw_only=True)
class StructuralCoreLayer(DulInformationObject):
    """Structural representation of StructuralCoreLayer."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#StructuralCoreLayer"
    )


@_dataclass(eq=False, kw_only=True)
class StructuralRepresentation(DulInformationObject):
    """Structural representation of StructuralRepresentation."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#StructuralRepresentation"
    )


@_dataclass(eq=False, kw_only=True)
class TaskRequest(DulInformationObject):
    """Structural representation of TaskRequest."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#TaskRequest"
    )


@_dataclass(eq=False, kw_only=True)
class TemporalPredicate(DulInformationObject):
    """Structural representation of TemporalPredicate."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#TemporalPredicate"
    )


@_dataclass(eq=False, kw_only=True)
class TripleBasedQuery(DulInformationObject):
    """Structural representation of TripleBasedQuery."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#TripleBasedQuery"
    )
    executed_by: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#executedBy",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    has_triple_pattern: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#hasTriplePattern",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    implementation_of: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#implementationOf",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    uses_predicate_from: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#usesPredicateFrom",
            "ontology_range": "",
            "property_kind": "object",
        },
    )


@_dataclass(eq=False, kw_only=True)
class TriplePattern(DulInformationObject):
    """Structural representation of TriplePattern."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#TriplePattern"
    )


@_dataclass(eq=False, kw_only=True)
class TypeConstraint(DulInformationObject):
    """Structural representation of TypeConstraint."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#TypeConstraint"
    )


@_dataclass(eq=False, kw_only=True)
class UsdOntology(DulInformationObject):
    """Structural representation of UsdOntology."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#USDOntology"
    )


@_dataclass(eq=False, kw_only=True)
class VrbContainer(DulInformationObject):
    """Structural representation of VrbContainer."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#VRBContainer"
    )
    has_container_layer: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#hasContainerLayer",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    has_scope_card: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#hasScopeCard",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    packages: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#packages",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    is_shareable_research_object: list[bool] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#isShareableResearchObject",
            "ontology_range": "http://www.w3.org/2001/XMLSchema#boolean",
            "property_kind": "datatype",
        },
    )


@_dataclass(eq=False, kw_only=True)
class ValidationRecord(DulInformationObject):
    """Structural representation of ValidationRecord."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#ValidationRecord"
    )
    has_validity_window: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#hasValidityWindow",
            "ontology_range": "",
            "property_kind": "object",
        },
    )


@_dataclass(eq=False, kw_only=True)
class WhatIfQuery(DulInformationObject):
    """Structural representation of WhatIfQuery."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#WhatIfQuery"
    )


@_dataclass(eq=False, kw_only=True)
class Xacro(DulInformationObject):
    """Structural representation of Xacro."""

    __ontology_iri__: _ClassVar[str] = "http://aicor.knowledge/l2-ontology.owl#Xacro"
    generates: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#generates",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    has_parameter: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#hasParameter",
            "ontology_range": "",
            "property_kind": "object",
        },
    )


@_dataclass(eq=False, kw_only=True)
class Chronicle(DulRegion):
    """Structural representation of Chronicle."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#Chronicle"
    )
    totally_orders: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#totallyOrders",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    forward_branching: list[bool] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#forwardBranching",
            "ontology_range": "http://www.w3.org/2001/XMLSchema#boolean",
            "property_kind": "datatype",
        },
    )
    maximal: list[bool] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#maximal",
            "ontology_range": "http://www.w3.org/2001/XMLSchema#boolean",
            "property_kind": "datatype",
        },
    )


@_dataclass(eq=False, kw_only=True)
class CompetenceDegree(DulRegion):
    """Structural representation of CompetenceDegree."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#CompetenceDegree"
    )


@_dataclass(eq=False, kw_only=True)
class ConfigurationSpace(DulRegion):
    """Structural representation of ConfigurationSpace."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#ConfigurationSpace"
    )
    configuration_space_of: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#configurationSpaceOf",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    structures: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#structures",
            "ontology_range": "",
            "property_kind": "object",
        },
    )


@_dataclass(eq=False, kw_only=True)
class FilmThickness(DulRegion):
    """Structural representation of FilmThickness."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#FilmThickness"
    )
    thins_as_radius_grows: list[bool] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#thinsAsRadiusGrows",
            "ontology_range": "http://www.w3.org/2001/XMLSchema#boolean",
            "property_kind": "datatype",
        },
    )


@_dataclass(eq=False, kw_only=True)
class ForceCorridor(DulRegion):
    """Structural representation of ForceCorridor."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#ForceCorridor"
    )
    corridor_of: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#corridorOf",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    lower_edge: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#lowerEdge",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    upper_edge: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#upperEdge",
            "ontology_range": "",
            "property_kind": "object",
        },
    )


@_dataclass(eq=False, kw_only=True)
class ParameterSample(DulRegion):
    """Structural representation of ParameterSample."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#ParameterSample"
    )
    drawn_from: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#drawnFrom",
            "ontology_range": "",
            "property_kind": "object",
        },
    )


@_dataclass(eq=False, kw_only=True)
class ParameterSpace(DulRegion):
    """Structural representation of ParameterSpace."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#ParameterSpace"
    )


@_dataclass(eq=False, kw_only=True)
class RecedingHorizon(DulRegion):
    """Structural representation of RecedingHorizon."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#RecedingHorizon"
    )


@_dataclass(eq=False, kw_only=True)
class SpreadRadius(DulRegion):
    """Structural representation of SpreadRadius."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#SpreadRadius"
    )


@_dataclass(eq=False, kw_only=True)
class TaskSpace(DulRegion):
    """Structural representation of TaskSpace."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#TaskSpace"
    )


@_dataclass(eq=False, kw_only=True)
class TimeInstant(DulRegion):
    """Structural representation of TimeInstant."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#TimeInstant"
    )


@_dataclass(eq=False, kw_only=True)
class TimeInterval(DulRegion):
    """Structural representation of TimeInterval."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#TimeInterval"
    )
    has_end: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#hasEnd",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    has_start: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#hasStart",
            "ontology_range": "",
            "property_kind": "object",
        },
    )


@_dataclass(eq=False, kw_only=True)
class TaskAchievingBodyMotion(BodyMotion):
    """Structural representation of TaskAchievingBodyMotion."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#TaskAchievingBodyMotion"
    )


@_dataclass(eq=False, kw_only=True)
class Robot(DulPhysicalAgent):
    """Structural representation of Robot."""

    __ontology_iri__: _ClassVar[str] = "http://aicor.knowledge/l2-ontology.owl#Robot"
    has_proper_part: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#hasProperPart",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    hosts: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#hosts",
            "ontology_range": "",
            "property_kind": "object",
        },
    )


@_dataclass(eq=False, kw_only=True)
class ArticulatedObject(DulPhysicalObject):
    """Structural representation of ArticulatedObject."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#ArticulatedObject"
    )
    has_joint: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#hasJoint",
            "ontology_range": "",
            "property_kind": "object",
        },
    )


@_dataclass(eq=False, kw_only=True)
class BodyWorldSystem(DulPhysicalObject, CaseLevelEntity):
    """Structural representation of BodyWorldSystem."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#BodyWorldSystem"
    )
    has_coupling: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#hasCoupling",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    has_declared_boundary: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#hasDeclaredBoundary",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    has_proper_part: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#hasProperPart",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    boundary_task_relative: list[bool] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#boundaryTaskRelative",
            "ontology_range": "http://www.w3.org/2001/XMLSchema#boolean",
            "property_kind": "datatype",
        },
    )


@_dataclass(eq=False, kw_only=True)
class Joint(DulPhysicalObject):
    """Structural representation of Joint."""

    __ontology_iri__: _ClassVar[str] = "http://aicor.knowledge/l2-ontology.owl#Joint"
    actuated_by: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#actuatedBy",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    connects: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#connects",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    has_joint_type: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#hasJointType",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    provides_dof: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#providesDOF",
            "ontology_range": "",
            "property_kind": "object",
        },
    )


@_dataclass(eq=False, kw_only=True)
class Link(DulPhysicalObject):
    """Structural representation of Link."""

    __ontology_iri__: _ClassVar[str] = "http://aicor.knowledge/l2-ontology.owl#Link"
    connected_by: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#connectedBy",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    has_pose: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#hasPose",
            "ontology_range": "",
            "property_kind": "object",
        },
    )


@_dataclass(eq=False, kw_only=True)
class MaterialEntity(DulPhysicalObject):
    """Structural representation of MaterialEntity."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#MaterialEntity"
    )


@_dataclass(eq=False, kw_only=True)
class Motor(DulPhysicalObject):
    """Structural representation of Motor."""

    __ontology_iri__: _ClassVar[str] = "http://aicor.knowledge/l2-ontology.owl#Motor"
    actuates: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#actuates",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    consumes: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#consumes",
            "ontology_range": "",
            "property_kind": "object",
        },
    )


@_dataclass(eq=False, kw_only=True)
class RobotWorldManipulationSystem(DulPhysicalObject, CaseLevelEntity):
    """Structural representation of RobotWorldManipulationSystem."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#RobotWorldManipulationSystem"
    )
    assessed_against: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#assessedAgainst",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    has_observation_action_coupling: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#hasObservationActionCoupling",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    has_proper_part: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#hasProperPart",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    has_solvability_structure: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#hasSolvabilityStructure",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    studied_as: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#studiedAs",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    is_closed_loop: list[bool] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#isClosedLoop",
            "ontology_range": "http://www.w3.org/2001/XMLSchema#boolean",
            "property_kind": "datatype",
        },
    )
    problem_is_proper_part: list[bool] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#problemIsProperPart",
            "ontology_range": "http://www.w3.org/2001/XMLSchema#boolean",
            "property_kind": "datatype",
        },
    )


@_dataclass(eq=False, kw_only=True)
class Sensor(DulPhysicalObject):
    """Structural representation of Sensor."""

    __ontology_iri__: _ClassVar[str] = "http://aicor.knowledge/l2-ontology.owl#Sensor"
    mounted_on: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#mountedOn",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    produces: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#produces",
            "ontology_range": "",
            "property_kind": "object",
        },
    )


@_dataclass(eq=False, kw_only=True)
class WorldPhysicalEntity(DulPhysicalObject):
    """Structural representation of WorldPhysicalEntity."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#WorldPhysicalEntity"
    )


@_dataclass(eq=False, kw_only=True)
class WorldState(DulPhysicalObject):
    """Structural representation of WorldState."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#WorldState"
    )
    has_continuous_state: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#hasContinuousState",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    has_discrete_mode: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#hasDiscreteMode",
            "ontology_range": "",
            "property_kind": "object",
        },
    )


@_dataclass(eq=False, kw_only=True)
class AccessibilityCritic(Critic):
    """Structural representation of AccessibilityCritic."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#AccessibilityCritic"
    )
    critic_dimension: list[str] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#criticDimension",
            "ontology_range": "http://www.w3.org/2001/XMLSchema#string",
            "property_kind": "datatype",
        },
    )


@_dataclass(eq=False, kw_only=True)
class CompletenessCritic(Critic):
    """Structural representation of CompletenessCritic."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#CompletenessCritic"
    )
    critic_dimension: list[str] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#criticDimension",
            "ontology_range": "http://www.w3.org/2001/XMLSchema#string",
            "property_kind": "datatype",
        },
    )


@_dataclass(eq=False, kw_only=True)
class ConventionCritic(Critic):
    """Structural representation of ConventionCritic."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#ConventionCritic"
    )
    critic_dimension: list[str] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#criticDimension",
            "ontology_range": "http://www.w3.org/2001/XMLSchema#string",
            "property_kind": "datatype",
        },
    )


@_dataclass(eq=False, kw_only=True)
class PhysicalCritic(Critic):
    """Structural representation of PhysicalCritic."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#PhysicalCritic"
    )
    critic_dimension: list[str] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#criticDimension",
            "ontology_range": "http://www.w3.org/2001/XMLSchema#string",
            "property_kind": "datatype",
        },
    )


@_dataclass(eq=False, kw_only=True)
class RelationalCritic(Critic):
    """Structural representation of RelationalCritic."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#RelationalCritic"
    )
    critic_dimension: list[str] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#criticDimension",
            "ontology_range": "http://www.w3.org/2001/XMLSchema#string",
            "property_kind": "datatype",
        },
    )


@_dataclass(eq=False, kw_only=True)
class SafetyCritic(Critic):
    """Structural representation of SafetyCritic."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#SafetyCritic"
    )
    critic_dimension: list[str] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#criticDimension",
            "ontology_range": "http://www.w3.org/2001/XMLSchema#string",
            "property_kind": "datatype",
        },
    )


@_dataclass(eq=False, kw_only=True)
class WorkflowCritic(Critic):
    """Structural representation of WorkflowCritic."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#WorkflowCritic"
    )
    critic_dimension: list[str] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#criticDimension",
            "ontology_range": "http://www.w3.org/2001/XMLSchema#string",
            "property_kind": "datatype",
        },
    )


@_dataclass(eq=False, kw_only=True)
class GenerativePerception(InternalSimulation, DulDescription):
    """Structural representation of GenerativePerception."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#GenerativePerception"
    )
    deepens: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#deepens",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    generates_world_model: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#generatesWorldModel",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    infers: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#infers",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    simulates_via: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#simulatesVia",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    is_causal: list[_Any] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#isCausal",
            "ontology_range": "",
            "property_kind": "datatype",
        },
    )
    is_white_box: list[_Any] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#isWhiteBox",
            "ontology_range": "",
            "property_kind": "datatype",
        },
    )


@_dataclass(eq=False, kw_only=True)
class DesignatorRefinement(Refinement):
    """Structural representation of DesignatorRefinement."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#DesignatorRefinement"
    )


@_dataclass(eq=False, kw_only=True)
class PlanInterpretationOp(Refinement):
    """Structural representation of PlanInterpretationOp."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#PlanInterpretationOp"
    )
    builds_task_network: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#buildsTaskNetwork",
            "ontology_range": "",
            "property_kind": "object",
        },
    )


@_dataclass(eq=False, kw_only=True)
class ProspectiveCase(ManipulationCase):
    """Structural representation of ProspectiveCase."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#ProspectiveCase"
    )
    has_initial_condition: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#hasInitialCondition",
            "ontology_range": "",
            "property_kind": "object",
        },
    )


@_dataclass(eq=False, kw_only=True)
class RealizedCase(ManipulationCase):
    """Structural representation of RealizedCase."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#RealizedCase"
    )
    has_episode: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#hasEpisode",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    has_record: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#hasRecord",
            "ontology_range": "",
            "property_kind": "object",
        },
    )


@_dataclass(eq=False, kw_only=True)
class AicorReferenceManipulationExecutive(ManipulationExecutive):
    """Structural representation of AicorReferenceManipulationExecutive."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#AICORReferenceManipulationExecutive"
    )
    has_proper_part: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#hasProperPart",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    is_architecture_neutral: list[bool] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#isArchitectureNeutral",
            "ontology_range": "http://www.w3.org/2001/XMLSchema#boolean",
            "property_kind": "datatype",
        },
    )


@_dataclass(eq=False, kw_only=True)
class RationalExecutive(ManipulationExecutive):
    """Structural representation of RationalExecutive."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#RationalExecutive"
    )
    assigns_status: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#assignsStatus",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    holds_active_while_believed_succeedable: list[bool] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#holdsActiveWhileBelievedSucceedable",
            "ontology_range": "http://www.w3.org/2001/XMLSchema#boolean",
            "property_kind": "datatype",
        },
    )
    stops_when_believed_unsucceedable_or_irrelevant: list[bool] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#stopsWhenBelievedUnsucceedableOrIrrelevant",
            "ontology_range": "http://www.w3.org/2001/XMLSchema#boolean",
            "property_kind": "datatype",
        },
    )


@_dataclass(eq=False, kw_only=True)
class Verdict(VerdictComponent):
    """Structural representation of Verdict."""

    __ontology_iri__: _ClassVar[str] = "http://aicor.knowledge/l2-ontology.owl#Verdict"
    comprises_verdict_component: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#comprisesVerdictComponent",
            "ontology_range": "",
            "property_kind": "object",
        },
    )


@_dataclass(eq=False, kw_only=True)
class Accounting(CognitiveCapability):
    """Structural representation of Accounting."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#Accounting"
    )


@_dataclass(eq=False, kw_only=True)
class CausalPrediction(CognitiveCapability):
    """Structural representation of CausalPrediction."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#CausalPrediction"
    )


@_dataclass(eq=False, kw_only=True)
class CounterfactualReasoning(CognitiveCapability):
    """Structural representation of CounterfactualReasoning."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#CounterfactualReasoning"
    )


@_dataclass(eq=False, kw_only=True)
class Diagnosis(CognitiveCapability):
    """Structural representation of Diagnosis."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#Diagnosis"
    )


@_dataclass(eq=False, kw_only=True)
class Evaluation(CognitiveCapability):
    """Structural representation of Evaluation."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#Evaluation"
    )


@_dataclass(eq=False, kw_only=True)
class Grounding(CognitiveCapability):
    """Structural representation of Grounding."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#Grounding"
    )


@_dataclass(eq=False, kw_only=True)
class Prospection(CognitiveCapability, InternalSimulation):
    """Structural representation of Prospection."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#Prospection"
    )


@_dataclass(eq=False, kw_only=True)
class RepairAbstention(CognitiveCapability):
    """Structural representation of RepairAbstention."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#RepairAbstention"
    )


@_dataclass(eq=False, kw_only=True)
class RequestConditionedPlacementCompetence(PlacementCompetence):
    """Structural representation of RequestConditionedPlacementCompetence."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#RequestConditionedPlacementCompetence"
    )


@_dataclass(eq=False, kw_only=True)
class MonitorGeneratorRole(FoundationModelInsertionPoint):
    """Structural representation of MonitorGeneratorRole."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#MonitorGeneratorRole"
    )


@_dataclass(eq=False, kw_only=True)
class PredictorRole(FoundationModelInsertionPoint):
    """Structural representation of PredictorRole."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#PredictorRole"
    )


@_dataclass(eq=False, kw_only=True)
class ProposalGeneratorRole(FoundationModelInsertionPoint):
    """Structural representation of ProposalGeneratorRole."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#ProposalGeneratorRole"
    )


@_dataclass(eq=False, kw_only=True)
class ForbiddenRegardlessCondition(ActionRelatedConcept):
    """Structural representation of ForbiddenRegardlessCondition."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#ForbiddenRegardlessCondition"
    )


@_dataclass(eq=False, kw_only=True)
class Purpose(ActionRelatedConcept):
    """Structural representation of Purpose."""

    __ontology_iri__: _ClassVar[str] = "http://aicor.knowledge/l2-ontology.owl#Purpose"


@_dataclass(eq=False, kw_only=True)
class NarrativeUnderstanding(ActivityUnderstanding):
    """Structural representation of NarrativeUnderstanding."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#NarrativeUnderstanding"
    )
    produces: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#produces",
            "ontology_range": "",
            "property_kind": "object",
        },
    )


@_dataclass(eq=False, kw_only=True)
class FederationWideAssessment(AssessmentRegime):
    """Structural representation of FederationWideAssessment."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#FederationWideAssessment"
    )
    comparable_under: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#comparableUnder",
            "ontology_range": "",
            "property_kind": "object",
        },
    )


@_dataclass(eq=False, kw_only=True)
class CapabilityAssertion(BodyWorldConstraint):
    """Structural representation of CapabilityAssertion."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#CapabilityAssertion"
    )


@_dataclass(eq=False, kw_only=True)
class IncapabilityAssertion(BodyWorldConstraint):
    """Structural representation of IncapabilityAssertion."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#IncapabilityAssertion"
    )


@_dataclass(eq=False, kw_only=True)
class DistributionalWarrant(ClaimWarrant):
    """Structural representation of DistributionalWarrant."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#DistributionalWarrant"
    )
    carries_margin: list[bool] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#carriesMargin",
            "ontology_range": "http://www.w3.org/2001/XMLSchema#boolean",
            "property_kind": "datatype",
        },
    )
    family_declared_independently_of_outcome: list[bool] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#familyDeclaredIndependentlyOfOutcome",
            "ontology_range": "http://www.w3.org/2001/XMLSchema#boolean",
            "property_kind": "datatype",
        },
    )


@_dataclass(eq=False, kw_only=True)
class FamilyRangingWarrant(ClaimWarrant):
    """Structural representation of FamilyRangingWarrant."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#FamilyRangingWarrant"
    )
    family_declared_independently_of_outcome: list[bool] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#familyDeclaredIndependentlyOfOutcome",
            "ontology_range": "http://www.w3.org/2001/XMLSchema#boolean",
            "property_kind": "datatype",
        },
    )


@_dataclass(eq=False, kw_only=True)
class ObservedConditionsWarrant(ClaimWarrant):
    """Structural representation of ObservedConditionsWarrant."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#ObservedConditionsWarrant"
    )


@_dataclass(eq=False, kw_only=True)
class SingleCaseWarrant(ClaimWarrant):
    """Structural representation of SingleCaseWarrant."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#SingleCaseWarrant"
    )


@_dataclass(eq=False, kw_only=True)
class CommitmentQuestion(CompetenceQuestion):
    """Structural representation of CommitmentQuestion."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#CommitmentQuestion"
    )
    principle_tag: list[str] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#principleTag",
            "ontology_range": "http://www.w3.org/2001/XMLSchema#string",
            "property_kind": "datatype",
        },
    )


@_dataclass(eq=False, kw_only=True)
class EvidenceQuestion(CompetenceQuestion):
    """Structural representation of EvidenceQuestion."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#EvidenceQuestion"
    )
    principle_tag: list[str] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#principleTag",
            "ontology_range": "http://www.w3.org/2001/XMLSchema#string",
            "property_kind": "datatype",
        },
    )


@_dataclass(eq=False, kw_only=True)
class RegulationQuestion(CompetenceQuestion):
    """Structural representation of RegulationQuestion."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#RegulationQuestion"
    )
    principle_tag: list[str] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#principleTag",
            "ontology_range": "http://www.w3.org/2001/XMLSchema#string",
            "property_kind": "datatype",
        },
    )


@_dataclass(eq=False, kw_only=True)
class WarrantQuestion(CompetenceQuestion):
    """Structural representation of WarrantQuestion."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#WarrantQuestion"
    )
    principle_tag: list[str] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#principleTag",
            "ontology_range": "http://www.w3.org/2001/XMLSchema#string",
            "property_kind": "datatype",
        },
    )


@_dataclass(eq=False, kw_only=True)
class OperationalGrammar(ConceptualFramework):
    """Structural representation of OperationalGrammar."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#OperationalGrammar"
    )
    disambiguates: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#disambiguates",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    has_dependency_structure: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#hasDependencyStructure",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    has_relational_grammar: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#hasRelationalGrammar",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    organized_by: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#organizedBy",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    presupposes: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#presupposes",
            "ontology_range": "",
            "property_kind": "object",
        },
    )


@_dataclass(eq=False, kw_only=True)
class MentalSimulation(Envisioning, InternalSimulation):
    """Structural representation of MentalSimulation."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#MentalSimulation"
    )
    reruns_sensorimotor_loop: list[_Any] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#rerunsSensorimotorLoop",
            "ontology_range": "",
            "property_kind": "datatype",
        },
    )


@_dataclass(eq=False, kw_only=True)
class Projection(Envisioning, InternalSimulation):
    """Structural representation of Projection."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#Projection"
    )
    abstracts_away: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#abstractsAway",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    drives: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#drives",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    produces_timeline: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#producesTimeline",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    projects_from: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#projectsFrom",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    is_monte_carlo: list[_Any] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#isMonteCarlo",
            "ontology_range": "",
            "property_kind": "datatype",
        },
    )


@_dataclass(eq=False, kw_only=True)
class Definition(EpistemicObject):
    """Structural representation of Definition."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#Definition"
    )
    defines_concept: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#definesConcept",
            "ontology_range": "",
            "property_kind": "object",
        },
    )


@_dataclass(eq=False, kw_only=True)
class Experiment(EpistemicObject):
    """Structural representation of Experiment."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#Experiment"
    )
    tests_hypothesis: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#testsHypothesis",
            "ontology_range": "",
            "property_kind": "object",
        },
    )


@_dataclass(eq=False, kw_only=True)
class Hypothesis(EpistemicObject):
    """Structural representation of Hypothesis."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#Hypothesis"
    )


@_dataclass(eq=False, kw_only=True)
class ScientificConcept(EpistemicObject):
    """Structural representation of ScientificConcept."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#ScientificConcept"
    )
    fixed_by_definition: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#fixedByDefinition",
            "ontology_range": "",
            "property_kind": "object",
        },
    )


@_dataclass(eq=False, kw_only=True)
class Theorem(EpistemicObject, MetaLevelEntity):
    """Structural representation of Theorem."""

    __ontology_iri__: _ClassVar[str] = "http://aicor.knowledge/l2-ontology.owl#Theorem"
    entailed_by: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#entailedBy",
            "ontology_range": "",
            "property_kind": "object",
        },
    )


@_dataclass(eq=False, kw_only=True)
class Theory(EpistemicObject, MetaLevelEntity):
    """Structural representation of Theory."""

    __ontology_iri__: _ClassVar[str] = "http://aicor.knowledge/l2-ontology.owl#Theory"
    about_scientific_object: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#aboutScientificObject",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    comprises_definition: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#comprisesDefinition",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    comprises_theorem: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#comprisesTheorem",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    explains: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#explains",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    has_claim: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#hasClaim",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    has_vocabulary: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#hasVocabulary",
            "ontology_range": "",
            "property_kind": "object",
        },
    )


@_dataclass(eq=False, kw_only=True)
class WorldModel(EpistemicObject):
    """Structural representation of WorldModel."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#WorldModel"
    )
    carries_guarantee: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#carriesGuarantee",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    enables_action_on: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#enablesActionOn",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    models_world: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#modelsWorld",
            "ontology_range": "",
            "property_kind": "object",
        },
    )


@_dataclass(eq=False, kw_only=True)
class ForbiddenTransition(EpistemicStatusTransition):
    """Structural representation of ForbiddenTransition."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#ForbiddenTransition"
    )


@_dataclass(eq=False, kw_only=True)
class PermittedTransition(EpistemicStatusTransition):
    """Structural representation of PermittedTransition."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#PermittedTransition"
    )


@_dataclass(eq=False, kw_only=True)
class ActionExecutive(ExecutiveComponent, ComputationalProcess):
    """Structural representation of ActionExecutive."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#ActionExecutive"
    )
    executes_motion: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#executesMotion",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    generates_action_token: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#generatesActionToken",
            "ontology_range": "",
            "property_kind": "object",
        },
    )


@_dataclass(eq=False, kw_only=True)
class KnowledgeExecutive(ExecutiveComponent, ComputationalProcess):
    """Structural representation of KnowledgeExecutive."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#KnowledgeExecutive"
    )
    acquires_via: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#acquiresVia",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    curates: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#curates",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    reasons_about: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#reasonsAbout",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    reasons_with: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#reasonsWith",
            "ontology_range": "",
            "property_kind": "object",
        },
    )


@_dataclass(eq=False, kw_only=True)
class MetaReasoningExecutive(ExecutiveComponent, ComputationalProcess):
    """Structural representation of MetaReasoningExecutive."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#MetaReasoningExecutive"
    )
    runs_scheduler: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#runsScheduler",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    specializes_via: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#specializesVia",
            "ontology_range": "",
            "property_kind": "object",
        },
    )


@_dataclass(eq=False, kw_only=True)
class PerceptionExecutive(ExecutiveComponent, ComputationalProcess):
    """Structural representation of PerceptionExecutive."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#PerceptionExecutive"
    )
    composes_method: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#composesMethod",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    maintains: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#maintains",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    resolves_designator: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#resolvesDesignator",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    verifies_against: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#verifiesAgainst",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    uses_imagination: list[_Any] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#usesImagination",
            "ontology_range": "",
            "property_kind": "datatype",
        },
    )


@_dataclass(eq=False, kw_only=True)
class PlanExecutive(ExecutiveComponent, ComputationalProcess):
    """Structural representation of PlanExecutive."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#PlanExecutive"
    )
    achieves: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#achieves",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    executes_plan: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#executesPlan",
            "ontology_range": "",
            "property_kind": "object",
        },
    )


@_dataclass(eq=False, kw_only=True)
class ActingMechanism(ExecutiveMechanism):
    """Structural representation of ActingMechanism."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#ActingMechanism"
    )


@_dataclass(eq=False, kw_only=True)
class EvidentialMechanism(ExecutiveMechanism):
    """Structural representation of EvidentialMechanism."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#EvidentialMechanism"
    )


@_dataclass(eq=False, kw_only=True)
class HonestyProperty(ExecutiveProperty):
    """Structural representation of HonestyProperty."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#HonestyProperty"
    )


@_dataclass(eq=False, kw_only=True)
class RegimeDependentProperty(ExecutiveProperty):
    """Structural representation of RegimeDependentProperty."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#RegimeDependentProperty"
    )


@_dataclass(eq=False, kw_only=True)
class CompositeFailure(FailureDescription):
    """Structural representation of CompositeFailure."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#CompositeFailure"
    )
    has_component: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#hasComponent",
            "ontology_range": "",
            "property_kind": "object",
        },
    )


@_dataclass(eq=False, kw_only=True)
class IntractableFailure(FailureDescription):
    """Structural representation of IntractableFailure."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#IntractableFailure"
    )


@_dataclass(eq=False, kw_only=True)
class TrivialFailure(FailureDescription):
    """Structural representation of TrivialFailure."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#TrivialFailure"
    )


@_dataclass(eq=False, kw_only=True)
class SchemaEmbedding(ForceDynamicDependency):
    """Structural representation of SchemaEmbedding."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#SchemaEmbedding"
    )
    embedded_within_phase: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#embeddedWithinPhase",
            "ontology_range": "",
            "property_kind": "object",
        },
    )


@_dataclass(eq=False, kw_only=True)
class CompetenceStandard(GlobalStandard):
    """Structural representation of CompetenceStandard."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#CompetenceStandard"
    )
    is_hard_constraint: list[bool] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#isHardConstraint",
            "ontology_range": "http://www.w3.org/2001/XMLSchema#boolean",
            "property_kind": "datatype",
        },
    )
    standard_code: list[str] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#standardCode",
            "ontology_range": "http://www.w3.org/2001/XMLSchema#string",
            "property_kind": "datatype",
        },
    )


@_dataclass(eq=False, kw_only=True)
class CompletenessStandard(GlobalStandard):
    """Structural representation of CompletenessStandard."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#CompletenessStandard"
    )
    is_hard_constraint: list[bool] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#isHardConstraint",
            "ontology_range": "http://www.w3.org/2001/XMLSchema#boolean",
            "property_kind": "datatype",
        },
    )
    standard_code: list[str] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#standardCode",
            "ontology_range": "http://www.w3.org/2001/XMLSchema#string",
            "property_kind": "datatype",
        },
    )


@_dataclass(eq=False, kw_only=True)
class CorrectnessStandard(GlobalStandard):
    """Structural representation of CorrectnessStandard."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#CorrectnessStandard"
    )
    is_hard_constraint: list[bool] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#isHardConstraint",
            "ontology_range": "http://www.w3.org/2001/XMLSchema#boolean",
            "property_kind": "datatype",
        },
    )
    standard_code: list[str] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#standardCode",
            "ontology_range": "http://www.w3.org/2001/XMLSchema#string",
            "property_kind": "datatype",
        },
    )


@_dataclass(eq=False, kw_only=True)
class SafetyStandard(GlobalStandard):
    """Structural representation of SafetyStandard."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#SafetyStandard"
    )
    is_hard_constraint: list[bool] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#isHardConstraint",
            "ontology_range": "http://www.w3.org/2001/XMLSchema#boolean",
            "property_kind": "datatype",
        },
    )
    standard_code: list[str] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#standardCode",
            "ontology_range": "http://www.w3.org/2001/XMLSchema#string",
            "property_kind": "datatype",
        },
    )


@_dataclass(eq=False, kw_only=True)
class AchievementGoal(Goal):
    """Structural representation of AchievementGoal."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#AchievementGoal"
    )
    has_condition_type: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#hasConditionType",
            "ontology_range": "",
            "property_kind": "object",
        },
    )


@_dataclass(eq=False, kw_only=True)
class MaintenanceGoal(Goal):
    """Structural representation of MaintenanceGoal."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#MaintenanceGoal"
    )
    has_condition_type: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#hasConditionType",
            "ontology_range": "",
            "property_kind": "object",
        },
    )


@_dataclass(eq=False, kw_only=True)
class PreventionGoal(Goal):
    """Structural representation of PreventionGoal."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#PreventionGoal"
    )
    has_condition_type: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#hasConditionType",
            "ontology_range": "",
            "property_kind": "object",
        },
    )


@_dataclass(eq=False, kw_only=True)
class StructuredReactiveController(ManipulationControlProgram):
    """Structural representation of StructuredReactiveController."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#StructuredReactiveController"
    )
    calls: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#calls",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    has_policy: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#hasPolicy",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    has_primary_activity: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#hasPrimaryActivity",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    mirrors: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#mirrors",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    reads: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#reads",
            "ontology_range": "",
            "property_kind": "object",
        },
    )


@_dataclass(eq=False, kw_only=True)
class ArrangementProblem(ManipulationProblem):
    """Structural representation of ArrangementProblem."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#ArrangementProblem"
    )


@_dataclass(eq=False, kw_only=True)
class BwsRelativeTaskSpecification(ManipulationProblem):
    """Structural representation of BwsRelativeTaskSpecification."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#BWSRelativeTaskSpecification"
    )


@_dataclass(eq=False, kw_only=True)
class PlacementProblem(ManipulationProblem):
    """Structural representation of PlacementProblem."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#PlacementProblem"
    )


@_dataclass(eq=False, kw_only=True)
class RealizationProblem(ManipulationProblem):
    """Structural representation of RealizationProblem."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#RealizationProblem"
    )


@_dataclass(eq=False, kw_only=True)
class RestorationProblem(ManipulationProblem):
    """Structural representation of RestorationProblem."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#RestorationProblem"
    )


@_dataclass(eq=False, kw_only=True)
class StabilizationProblem(ManipulationProblem):
    """Structural representation of StabilizationProblem."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#StabilizationProblem"
    )


@_dataclass(eq=False, kw_only=True)
class AxiomConditionalGuarantee(ModelGuarantee):
    """Structural representation of AxiomConditionalGuarantee."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#AxiomConditionalGuarantee"
    )


@_dataclass(eq=False, kw_only=True)
class PacGuarantee(ModelGuarantee):
    """Structural representation of PacGuarantee."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#PACGuarantee"
    )


@_dataclass(eq=False, kw_only=True)
class BindingConfirmationFunction(ObservationFunction):
    """Structural representation of BindingConfirmationFunction."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#BindingConfirmationFunction"
    )
    confirms: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#confirms",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    function_code: list[str] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#functionCode",
            "ontology_range": "http://www.w3.org/2001/XMLSchema#string",
            "property_kind": "datatype",
        },
    )


@_dataclass(eq=False, kw_only=True)
class ProcessMonitoringFunction(ObservationFunction):
    """Structural representation of ProcessMonitoringFunction."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#ProcessMonitoringFunction"
    )
    detects: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#detects",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    function_code: list[str] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#functionCode",
            "ontology_range": "http://www.w3.org/2001/XMLSchema#string",
            "property_kind": "datatype",
        },
    )


@_dataclass(eq=False, kw_only=True)
class Contextualization(ProposalToCommitmentPipeline, ComputationalProcess):
    """Structural representation of Contextualization."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#Contextualization"
    )
    has_step: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#hasStep",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    instantiates: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#instantiates",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    produces: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#produces",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    queries: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#queries",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    queries_model: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#queriesModel",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    resolves: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#resolves",
            "ontology_range": "",
            "property_kind": "object",
        },
    )


@_dataclass(eq=False, kw_only=True)
class BodyMotionLink(RealizationLink):
    """Structural representation of BodyMotionLink."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#BodyMotionLink"
    )
    realizes_motion: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#realizesMotion",
            "ontology_range": "",
            "property_kind": "object",
        },
    )


@_dataclass(eq=False, kw_only=True)
class ForceGenerationLink(RealizationLink):
    """Structural representation of ForceGenerationLink."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#ForceGenerationLink"
    )
    realizes_interaction: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#realizesInteraction",
            "ontology_range": "",
            "property_kind": "object",
        },
    )


@_dataclass(eq=False, kw_only=True)
class StateTransitionLink(RealizationLink):
    """Structural representation of StateTransitionLink."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#StateTransitionLink"
    )
    produces_effect: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#producesEffect",
            "ontology_range": "",
            "property_kind": "object",
        },
    )


@_dataclass(eq=False, kw_only=True)
class TokenEmissionLink(RealizationLink):
    """Structural representation of TokenEmissionLink."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#TokenEmissionLink"
    )
    emits_token: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#emitsToken",
            "ontology_range": "",
            "property_kind": "object",
        },
    )


@_dataclass(eq=False, kw_only=True)
class PhysicalRepair(Repair):
    """Structural representation of PhysicalRepair."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#PhysicalRepair"
    )


@_dataclass(eq=False, kw_only=True)
class FailedRequest(RequestOutcome):
    """Structural representation of FailedRequest."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#FailedRequest"
    )
    has_failure_reason: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#hasFailureReason",
            "ontology_range": "",
            "property_kind": "object",
        },
    )


@_dataclass(eq=False, kw_only=True)
class SolvedRequest(RequestOutcome):
    """Structural representation of SolvedRequest."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#SolvedRequest"
    )


@_dataclass(eq=False, kw_only=True)
class ConstraintBasedControl(RobotControlLaw):
    """Structural representation of ConstraintBasedControl."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#ConstraintBasedControl"
    )
    expresses_task_as: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#expressesTaskAs",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    solves_for: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#solvesFor",
            "ontology_range": "",
            "property_kind": "object",
        },
    )


@_dataclass(eq=False, kw_only=True)
class OptimizationBasedControl(RobotControlLaw):
    """Structural representation of OptimizationBasedControl."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#OptimizationBasedControl"
    )
    has_horizon: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#hasHorizon",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    minimizes: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#minimizes",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    subject_to: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#subjectTo",
            "ontology_range": "",
            "property_kind": "object",
        },
    )


@_dataclass(eq=False, kw_only=True)
class DomainScientificObject(ScientificObject):
    """Structural representation of DomainScientificObject."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#DomainScientificObject"
    )
    requires_methodology: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#requiresMethodology",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    is_methodological_object: list[bool] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#isMethodologicalObject",
            "ontology_range": "http://www.w3.org/2001/XMLSchema#boolean",
            "property_kind": "datatype",
        },
    )


@_dataclass(eq=False, kw_only=True)
class MethodologicalScientificObject(ScientificObject):
    """Structural representation of MethodologicalScientificObject."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#MethodologicalScientificObject"
    )
    is_methodological_object: list[bool] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#isMethodologicalObject",
            "ontology_range": "http://www.w3.org/2001/XMLSchema#boolean",
            "property_kind": "datatype",
        },
    )


@_dataclass(eq=False, kw_only=True)
class SubfieldScientificObject(ScientificObject):
    """Structural representation of SubfieldScientificObject."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#SubfieldScientificObject"
    )
    declares_assumption_set: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#declaresAssumptionSet",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    has_assessment_regime: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#hasAssessmentRegime",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    has_problem_family: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#hasProblemFamily",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    offers_method: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#offersMethod",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    ranges_over_projection: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#rangesOverProjection",
            "ontology_range": "",
            "property_kind": "object",
        },
    )


@_dataclass(eq=False, kw_only=True)
class CapabilityClaim(ScopedClaim):
    """Structural representation of CapabilityClaim."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#CapabilityClaim"
    )


@_dataclass(eq=False, kw_only=True)
class CompetenceClaim(ScopedClaim, MetaLevelEntity):
    """Structural representation of CompetenceClaim."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#CompetenceClaim"
    )
    asserts_standard: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#assertsStandard",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    at_degree: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#atDegree",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    claims_over: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#claimsOver",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    has_scope: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#hasScope",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    held_by: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#heldBy",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    warranted_by: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#warrantedBy",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    claim_scope: list[str] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#claimScope",
            "ontology_range": "http://www.w3.org/2001/XMLSchema#string",
            "property_kind": "datatype",
        },
    )
    is_dispositional: list[bool] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#isDispositional",
            "ontology_range": "http://www.w3.org/2001/XMLSchema#boolean",
            "property_kind": "datatype",
        },
    )
    is_rate_bearing: list[bool] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#isRateBearing",
            "ontology_range": "http://www.w3.org/2001/XMLSchema#boolean",
            "property_kind": "datatype",
        },
    )
    licensed_by_single_episode: list[bool] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#licensedBySingleEpisode",
            "ontology_range": "http://www.w3.org/2001/XMLSchema#boolean",
            "property_kind": "datatype",
        },
    )


@_dataclass(eq=False, kw_only=True)
class EpisodeClaim(ScopedClaim):
    """Structural representation of EpisodeClaim."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#EpisodeClaim"
    )
    claims_about_episode: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#claimsAboutEpisode",
            "ontology_range": "",
            "property_kind": "object",
        },
    )


@_dataclass(eq=False, kw_only=True)
class MasteryClaim(ScopedClaim, MetaLevelEntity):
    """Structural representation of MasteryClaim."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#MasteryClaim"
    )
    about_system_family: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#aboutSystemFamily",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    over_problem_family: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#overProblemFamily",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    satisfies: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#satisfies",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    strengthens: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#strengthens",
            "ontology_range": "",
            "property_kind": "object",
        },
    )


@_dataclass(eq=False, kw_only=True)
class ControlBasisTooWeak(SolvabilityFailureMode):
    """Structural representation of ControlBasisTooWeak."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#ControlBasisTooWeak"
    )
    failure_code: list[str] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#failureCode",
            "ontology_range": "http://www.w3.org/2001/XMLSchema#string",
            "property_kind": "datatype",
        },
    )


@_dataclass(eq=False, kw_only=True)
class InadequateConduct(SolvabilityFailureMode):
    """Structural representation of InadequateConduct."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#InadequateConduct"
    )
    failure_code: list[str] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#failureCode",
            "ontology_range": "http://www.w3.org/2001/XMLSchema#string",
            "property_kind": "datatype",
        },
    )


@_dataclass(eq=False, kw_only=True)
class NoAdmissibleSolution(SolvabilityFailureMode):
    """Structural representation of NoAdmissibleSolution."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#NoAdmissibleSolution"
    )
    failure_code: list[str] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#failureCode",
            "ontology_range": "http://www.w3.org/2001/XMLSchema#string",
            "property_kind": "datatype",
        },
    )


@_dataclass(eq=False, kw_only=True)
class SignalUnavailable(SolvabilityFailureMode):
    """Structural representation of SignalUnavailable."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#SignalUnavailable"
    )
    failure_code: list[str] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#failureCode",
            "ontology_range": "http://www.w3.org/2001/XMLSchema#string",
            "property_kind": "datatype",
        },
    )


@_dataclass(eq=False, kw_only=True)
class SignalUnrepresented(SolvabilityFailureMode):
    """Structural representation of SignalUnrepresented."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#SignalUnrepresented"
    )
    failure_code: list[str] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#failureCode",
            "ontology_range": "http://www.w3.org/2001/XMLSchema#string",
            "property_kind": "datatype",
        },
    )


@_dataclass(eq=False, kw_only=True)
class DesignRung(SolvabilityRung):
    """Structural representation of DesignRung."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#DesignRung"
    )
    designer_controllable: list[bool] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#designerControllable",
            "ontology_range": "http://www.w3.org/2001/XMLSchema#boolean",
            "property_kind": "datatype",
        },
    )


@_dataclass(eq=False, kw_only=True)
class SubstrateRung(SolvabilityRung):
    """Structural representation of SubstrateRung."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#SubstrateRung"
    )
    designer_controllable: list[bool] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#designerControllable",
            "ontology_range": "http://www.w3.org/2001/XMLSchema#boolean",
            "property_kind": "datatype",
        },
    )


@_dataclass(eq=False, kw_only=True)
class ActionSufficiency(SubstrateSufficiency):
    """Structural representation of ActionSufficiency."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#ActionSufficiency"
    )
    sufficiency_kind: list[str] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#sufficiencyKind",
            "ontology_range": "http://www.w3.org/2001/XMLSchema#string",
            "property_kind": "datatype",
        },
    )


@_dataclass(eq=False, kw_only=True)
class InformationalSufficiency(SubstrateSufficiency):
    """Structural representation of InformationalSufficiency."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#InformationalSufficiency"
    )
    sufficiency_kind: list[str] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#sufficiencyKind",
            "ontology_range": "http://www.w3.org/2001/XMLSchema#string",
            "property_kind": "datatype",
        },
    )


@_dataclass(eq=False, kw_only=True)
class TemporalSufficiency(SubstrateSufficiency):
    """Structural representation of TemporalSufficiency."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#TemporalSufficiency"
    )
    sufficiency_kind: list[str] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#sufficiencyKind",
            "ontology_range": "http://www.w3.org/2001/XMLSchema#string",
            "property_kind": "datatype",
        },
    )


@_dataclass(eq=False, kw_only=True)
class ComparisonFrame(ComputationalMedium):
    """Structural representation of ComparisonFrame."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#ComparisonFrame"
    )
    frame_fixes: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#frameFixes",
            "ontology_range": "",
            "property_kind": "object",
        },
    )


@_dataclass(eq=False, kw_only=True)
class FixedCondition(ComputationalMedium):
    """Structural representation of FixedCondition."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#FixedCondition"
    )


@_dataclass(eq=False, kw_only=True)
class LaboratoryAccessTerms(ComputationalMedium):
    """Structural representation of LaboratoryAccessTerms."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#LaboratoryAccessTerms"
    )


@_dataclass(eq=False, kw_only=True)
class LaboratoryFederation(ComputationalMedium):
    """Structural representation of LaboratoryFederation."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#LaboratoryFederation"
    )
    federation_member: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#federationMember",
            "ontology_range": "",
            "property_kind": "object",
        },
    )


@_dataclass(eq=False, kw_only=True)
class Representation(ComputationalMedium):
    """Structural representation of Representation."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#Representation"
    )


@_dataclass(eq=False, kw_only=True)
class FederatedResearchLaboratory(DigitalResearchLaboratory):
    """Structural representation of FederatedResearchLaboratory."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#FederatedResearchLaboratory"
    )
    entry_governed_by: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#entryGovernedBy",
            "ontology_range": "",
            "property_kind": "object",
        },
    )


@_dataclass(eq=False, kw_only=True)
class DerivedCaseReconstruction(EvidenceBearingTrace):
    """Structural representation of DerivedCaseReconstruction."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#DerivedCaseReconstruction"
    )
    reconstructed_by_model: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#reconstructedByModel",
            "ontology_range": "",
            "property_kind": "object",
        },
    )


@_dataclass(eq=False, kw_only=True)
class DualProcessTrace(EvidenceBearingTrace):
    """Structural representation of DualProcessTrace."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#DualProcessTrace"
    )
    inspectable_by: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#inspectableBy",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    records_commitment: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#recordsCommitment",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    records_grounding_and_checks: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#recordsGroundingAndChecks",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    records_monitor_and_repair: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#recordsMonitorAndRepair",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    records_system1_output: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#recordsSystem1Output",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    records_translation: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#recordsTranslation",
            "ontology_range": "",
            "property_kind": "object",
        },
    )


@_dataclass(eq=False, kw_only=True)
class PortableResult(EvidenceBearingTrace):
    """Structural representation of PortableResult."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#PortableResult"
    )
    result_obtained_in: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#resultObtainedIn",
            "ontology_range": "",
            "property_kind": "object",
        },
    )


@_dataclass(eq=False, kw_only=True)
class BindingConfirmationExpectation(ExpectedObservation):
    """Structural representation of BindingConfirmationExpectation."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#BindingConfirmationExpectation"
    )
    confirms_binding_in: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#confirmsBindingIn",
            "ontology_range": "",
            "property_kind": "object",
        },
    )


@_dataclass(eq=False, kw_only=True)
class ProcessMonitoringExpectation(ExpectedObservation):
    """Structural representation of ProcessMonitoringExpectation."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#ProcessMonitoringExpectation"
    )
    monitors_phase_of: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#monitorsPhaseOf",
            "ontology_range": "",
            "property_kind": "object",
        },
    )


@_dataclass(eq=False, kw_only=True)
class ComponentDescription(PartialDescription):
    """Structural representation of ComponentDescription."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#ComponentDescription"
    )


@_dataclass(eq=False, kw_only=True)
class Conjunction(QueryConnective):
    """Structural representation of Conjunction."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#Conjunction"
    )


@_dataclass(eq=False, kw_only=True)
class Disjunction(QueryConnective):
    """Structural representation of Disjunction."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#Disjunction"
    )


@_dataclass(eq=False, kw_only=True)
class BoundVariable(QueryVariable):
    """Structural representation of BoundVariable."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#BoundVariable"
    )


@_dataclass(eq=False, kw_only=True)
class UnboundVariable(QueryVariable):
    """Structural representation of UnboundVariable."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#UnboundVariable"
    )


@_dataclass(eq=False, kw_only=True)
class EventType(ReifiedTemporalEntity, DulDescription):
    """Structural representation of EventType."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#EventType"
    )
    occurs_over: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#occursOver",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    homogeneous: list[bool] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#homogeneous",
            "ontology_range": "http://www.w3.org/2001/XMLSchema#boolean",
            "property_kind": "datatype",
        },
    )


@_dataclass(eq=False, kw_only=True)
class FluentType(ReifiedTemporalEntity, DulDescription):
    """Structural representation of FluentType."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#FluentType"
    )
    holds_over: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#holdsOver",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    homogeneous: list[bool] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#homogeneous",
            "ontology_range": "http://www.w3.org/2001/XMLSchema#boolean",
            "property_kind": "datatype",
        },
    )


@_dataclass(eq=False, kw_only=True)
class Urdf(RobotDescription):
    """Structural representation of Urdf."""

    __ontology_iri__: _ClassVar[str] = "http://aicor.knowledge/l2-ontology.owl#URDF"
    has_joint_element: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#hasJointElement",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    has_link_element: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#hasLinkElement",
            "ontology_range": "",
            "property_kind": "object",
        },
    )


@_dataclass(eq=False, kw_only=True)
class BeliefStateSemanticDigitalTwin(SemanticDigitalTwin):
    """Structural representation of BeliefStateSemanticDigitalTwin."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#BeliefStateSemanticDigitalTwin"
    )
    represents_for_executive: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#representsForExecutive",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    represents_selected_aspects_of: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#representsSelectedAspectsOf",
            "ontology_range": "",
            "property_kind": "object",
        },
    )


@_dataclass(eq=False, kw_only=True)
class CramSemanticDigitalTwin(SemanticDigitalTwin):
    """Structural representation of CramSemanticDigitalTwin."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#CRAMSemanticDigitalTwin"
    )
    has_observation_model: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#hasObservationModel",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    has_physics_model: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#hasPhysicsModel",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    transitions_on: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#transitionsOn",
            "ontology_range": "",
            "property_kind": "object",
        },
    )


@_dataclass(eq=False, kw_only=True)
class AbstentionSignal(Signal):
    """Structural representation of AbstentionSignal."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#AbstentionSignal"
    )


@_dataclass(eq=False, kw_only=True)
class CompletionSignal(Signal):
    """Structural representation of CompletionSignal."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#CompletionSignal"
    )


@_dataclass(eq=False, kw_only=True)
class FailureSignal(Signal):
    """Structural representation of FailureSignal."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#FailureSignal"
    )


@_dataclass(eq=False, kw_only=True)
class SuccessSignal(Signal):
    """Structural representation of SuccessSignal."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#SuccessSignal"
    )


@_dataclass(eq=False, kw_only=True)
class CollisionGeometry(SimulationAsset):
    """Structural representation of CollisionGeometry."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#CollisionGeometry"
    )
    consumed_by: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#consumedBy",
            "ontology_range": "",
            "property_kind": "object",
        },
    )


@_dataclass(eq=False, kw_only=True)
class VisualGeometry(SimulationAsset):
    """Structural representation of VisualGeometry."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#VisualGeometry"
    )
    consumed_by: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#consumedBy",
            "ontology_range": "",
            "property_kind": "object",
        },
    )


@_dataclass(eq=False, kw_only=True)
class SceneGraph(StructuralRepresentation):
    """Structural representation of SceneGraph."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#SceneGraph"
    )
    has_node: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#hasNode",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    has_relation_edge: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#hasRelationEdge",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    has_role_assignment_layer: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#hasRoleAssignmentLayer",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    partitioned_into: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#partitionedInto",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    is_data_structure_realization: list[bool] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#isDataStructureRealization",
            "ontology_range": "http://www.w3.org/2001/XMLSchema#boolean",
            "property_kind": "datatype",
        },
    )


@_dataclass(eq=False, kw_only=True)
class RobotBody(ArticulatedObject):
    """Structural representation of RobotBody."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#RobotBody"
    )
    generates: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#generates",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    has_end_effector: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#hasEndEffector",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    has_link: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#hasLink",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    has_sensor: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#hasSensor",
            "ontology_range": "",
            "property_kind": "object",
        },
    )


@_dataclass(eq=False, kw_only=True)
class EndEffector(Link):
    """Structural representation of EndEffector."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#EndEffector"
    )
    binds_role: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#bindsRole",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    establishes_contact: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#establishesContact",
            "ontology_range": "",
            "property_kind": "object",
        },
    )


@_dataclass(eq=False, kw_only=True)
class PhysicalIndividual(MaterialEntity):
    """Structural representation of PhysicalIndividual."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#PhysicalIndividual"
    )
    matter_kind: list[str] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#matterKind",
            "ontology_range": "http://www.w3.org/2001/XMLSchema#string",
            "property_kind": "datatype",
        },
    )


@_dataclass(eq=False, kw_only=True)
class Substance(MaterialEntity):
    """Structural representation of Substance."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#Substance"
    )
    matter_kind: list[str] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#matterKind",
            "ontology_range": "http://www.w3.org/2001/XMLSchema#string",
            "property_kind": "datatype",
        },
    )


@_dataclass(eq=False, kw_only=True)
class RiskAvoidanceReasoning(Evaluation):
    """Structural representation of RiskAvoidanceReasoning."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#RiskAvoidanceReasoning"
    )


@_dataclass(eq=False, kw_only=True)
class TradeOffReasoning(Evaluation):
    """Structural representation of TradeOffReasoning."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#TradeOffReasoning"
    )


@_dataclass(eq=False, kw_only=True)
class ActionPlanning(Prospection):
    """Structural representation of ActionPlanning."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#ActionPlanning"
    )
    produces: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#produces",
            "ontology_range": "",
            "property_kind": "object",
        },
    )


@_dataclass(eq=False, kw_only=True)
class EpisodicSimulation(Prospection):
    """Structural representation of EpisodicSimulation."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#EpisodicSimulation"
    )
    produces: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#produces",
            "ontology_range": "",
            "property_kind": "object",
        },
    )


@_dataclass(eq=False, kw_only=True)
class GoalSetting(Prospection):
    """Structural representation of GoalSetting."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#GoalSetting"
    )
    produces: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#produces",
            "ontology_range": "",
            "property_kind": "object",
        },
    )


@_dataclass(eq=False, kw_only=True)
class OutcomePrediction(Prospection):
    """Structural representation of OutcomePrediction."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#OutcomePrediction"
    )
    produces: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#produces",
            "ontology_range": "",
            "property_kind": "object",
        },
    )


@_dataclass(eq=False, kw_only=True)
class FoundationModelOutput(Hypothesis):
    """Structural representation of FoundationModelOutput."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#FoundationModelOutput"
    )
    inserted_into: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#insertedInto",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    is_ground_truth: list[bool] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#isGroundTruth",
            "ontology_range": "http://www.w3.org/2001/XMLSchema#boolean",
            "property_kind": "datatype",
        },
    )
    is_warranted_commitment: list[bool] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#isWarrantedCommitment",
            "ontology_range": "http://www.w3.org/2001/XMLSchema#boolean",
            "property_kind": "datatype",
        },
    )


@_dataclass(eq=False, kw_only=True)
class ManipulationHypothesis(Hypothesis, MetaLevelEntity):
    """Structural representation of ManipulationHypothesis."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#ManipulationHypothesis"
    )
    evaluated_by: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#evaluatedBy",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    ranges_over_family: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#rangesOverFamily",
            "ontology_range": "",
            "property_kind": "object",
        },
    )


@_dataclass(eq=False, kw_only=True)
class ArtifactCertification(Theorem):
    """Structural representation of ArtifactCertification."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#ArtifactCertification"
    )
    certifies_artifact: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#certifiesArtifact",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    over_problem_class: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#overProblemClass",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    has_certification_kind: list[str] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#hasCertificationKind",
            "ontology_range": "http://www.w3.org/2001/XMLSchema#string",
            "property_kind": "datatype",
        },
    )


@_dataclass(eq=False, kw_only=True)
class StatisticalWorldModel(WorldModel):
    """Structural representation of StatisticalWorldModel."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#StatisticalWorldModel"
    )
    has_model_component: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#hasModelComponent",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    fit_from_data: list[bool] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#fitFromData",
            "ontology_range": "http://www.w3.org/2001/XMLSchema#boolean",
            "property_kind": "datatype",
        },
    )


@_dataclass(eq=False, kw_only=True)
class WorldStructureModel(WorldModel):
    """Structural representation of WorldStructureModel."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#WorldStructureModel"
    )
    enables_solution_of: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#enablesSolutionOf",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    models_structure_of: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#modelsStructureOf",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    fit_from_data: list[bool] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#fitFromData",
            "ontology_range": "http://www.w3.org/2001/XMLSchema#boolean",
            "property_kind": "datatype",
        },
    )


@_dataclass(eq=False, kw_only=True)
class Actuating(ActingMechanism):
    """Structural representation of Actuating."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#Actuating"
    )
    mechanism_realized_by: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#mechanismRealizedBy",
            "ontology_range": "",
            "property_kind": "object",
        },
    )


@_dataclass(eq=False, kw_only=True)
class Disambiguating(ActingMechanism):
    """Structural representation of Disambiguating."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#Disambiguating"
    )
    mechanism_realized_by: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#mechanismRealizedBy",
            "ontology_range": "",
            "property_kind": "object",
        },
    )


@_dataclass(eq=False, kw_only=True)
class Proposing(ActingMechanism):
    """Structural representation of Proposing."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#Proposing"
    )
    mechanism_realized_by: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#mechanismRealizedBy",
            "ontology_range": "",
            "property_kind": "object",
        },
    )


@_dataclass(eq=False, kw_only=True)
class Rationalizing(ActingMechanism):
    """Structural representation of Rationalizing."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#Rationalizing"
    )
    mechanism_realized_by: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#mechanismRealizedBy",
            "ontology_range": "",
            "property_kind": "object",
        },
    )


@_dataclass(eq=False, kw_only=True)
class Realizing(ActingMechanism):
    """Structural representation of Realizing."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#Realizing"
    )
    mechanism_realized_by: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#mechanismRealizedBy",
            "ontology_range": "",
            "property_kind": "object",
        },
    )


@_dataclass(eq=False, kw_only=True)
class Assessing(EvidentialMechanism):
    """Structural representation of Assessing."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#Assessing"
    )
    mechanism_realized_by: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#mechanismRealizedBy",
            "ontology_range": "",
            "property_kind": "object",
        },
    )


@_dataclass(eq=False, kw_only=True)
class Monitoring(EvidentialMechanism, CognitiveCapability):
    """Structural representation of Monitoring."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#Monitoring"
    )
    mechanism_realized_by: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#mechanismRealizedBy",
            "ontology_range": "",
            "property_kind": "object",
        },
    )


@_dataclass(eq=False, kw_only=True)
class Perceiving(EvidentialMechanism):
    """Structural representation of Perceiving."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#Perceiving"
    )
    mechanism_realized_by: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#mechanismRealizedBy",
            "ontology_range": "",
            "property_kind": "object",
        },
    )


@_dataclass(eq=False, kw_only=True)
class Regulating(EvidentialMechanism):
    """Structural representation of Regulating."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#Regulating"
    )
    mechanism_realized_by: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#mechanismRealizedBy",
            "ontology_range": "",
            "property_kind": "object",
        },
    )


@_dataclass(eq=False, kw_only=True)
class CramStructuredReactiveController(
    StructuredReactiveController, ComputationalProcess
):
    """Structural representation of CramStructuredReactiveController."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#CRAMStructuredReactiveController"
    )
    adapts_by: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#adaptsBy",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    has_part: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#hasPart",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    interpreted_by: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#interpretedBy",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    produces: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#produces",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    runs: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#runs",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    is_concurrent: list[_Any] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#isConcurrent",
            "ontology_range": "",
            "property_kind": "datatype",
        },
    )
    replans_during_execution: list[_Any] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#replansDuringExecution",
            "ontology_range": "",
            "property_kind": "datatype",
        },
    )


@_dataclass(eq=False, kw_only=True)
class RequestedPlacementProblem(PlacementProblem):
    """Structural representation of RequestedPlacementProblem."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#RequestedPlacementProblem"
    )


@_dataclass(eq=False, kw_only=True)
class PlacementPipeline(Contextualization):
    """Structural representation of PlacementPipeline."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#PlacementPipeline"
    )
    commits_target: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#commitsTarget",
            "ontology_range": "",
            "property_kind": "object",
        },
    )


@_dataclass(eq=False, kw_only=True)
class SolvedEggPickup(SolvedRequest):
    """Structural representation of SolvedEggPickup."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#SolvedEggPickup"
    )
    entails_terminal_state: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#entailsTerminalState",
            "ontology_range": "",
            "property_kind": "object",
        },
    )


@_dataclass(eq=False, kw_only=True)
class ModelPredictiveControl(OptimizationBasedControl):
    """Structural representation of ModelPredictiveControl."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#ModelPredictiveControl"
    )
    applies_first_of: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#appliesFirstOf",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    optimizes: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#optimizes",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    resolves_each: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#resolvesEach",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    rolls_out: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#rollsOut",
            "ontology_range": "",
            "property_kind": "object",
        },
    )


@_dataclass(eq=False, kw_only=True)
class CompetenceProperty(CompetenceClaim):
    """Structural representation of CompetenceProperty."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#CompetenceProperty"
    )
    conditioned_on: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#conditionedOn",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    reads_aspect: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#readsAspect",
            "ontology_range": "",
            "property_kind": "object",
        },
    )


@_dataclass(eq=False, kw_only=True)
class ComponentCompetenceClaim(CompetenceClaim):
    """Structural representation of ComponentCompetenceClaim."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#ComponentCompetenceClaim"
    )
    held_by_component: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#heldByComponent",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    requires_declared_scope: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#requiresDeclaredScope",
            "ontology_range": "",
            "property_kind": "object",
        },
    )


@_dataclass(eq=False, kw_only=True)
class FederationWarrantedCompetenceClaim(CompetenceClaim):
    """Structural representation of FederationWarrantedCompetenceClaim."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#FederationWarrantedCompetenceClaim"
    )
    warranted_by_portable_result: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#warrantedByPortableResult",
            "ontology_range": "",
            "property_kind": "object",
        },
    )


@_dataclass(eq=False, kw_only=True)
class PurposeIndexedCompetenceClaim(CompetenceClaim):
    """Structural representation of PurposeIndexedCompetenceClaim."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#PurposeIndexedCompetenceClaim"
    )
    scoped_by_purpose: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#scopedByPurpose",
            "ontology_range": "",
            "property_kind": "object",
        },
    )


@_dataclass(eq=False, kw_only=True)
class ExecutiveAdequacy(DesignRung):
    """Structural representation of ExecutiveAdequacy."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#ExecutiveAdequacy"
    )
    rung_level: list[int] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#rungLevel",
            "ontology_range": "http://www.w3.org/2001/XMLSchema#integer",
            "property_kind": "datatype",
        },
    )


@_dataclass(eq=False, kw_only=True)
class RepresentationalAccessibility(DesignRung):
    """Structural representation of RepresentationalAccessibility."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#RepresentationalAccessibility"
    )
    rung_level: list[int] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#rungLevel",
            "ontology_range": "http://www.w3.org/2001/XMLSchema#integer",
            "property_kind": "datatype",
        },
    )


@_dataclass(eq=False, kw_only=True)
class ControlRealizability(SubstrateRung):
    """Structural representation of ControlRealizability."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#ControlRealizability"
    )
    rung_level: list[int] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#rungLevel",
            "ontology_range": "http://www.w3.org/2001/XMLSchema#integer",
            "property_kind": "datatype",
        },
    )


@_dataclass(eq=False, kw_only=True)
class InformationalAvailability(SubstrateRung):
    """Structural representation of InformationalAvailability."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#InformationalAvailability"
    )
    rung_level: list[int] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#rungLevel",
            "ontology_range": "http://www.w3.org/2001/XMLSchema#integer",
            "property_kind": "datatype",
        },
    )


@_dataclass(eq=False, kw_only=True)
class PhysicalSolvability(SubstrateRung):
    """Structural representation of PhysicalSolvability."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#PhysicalSolvability"
    )
    rung_level: list[int] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#rungLevel",
            "ontology_range": "http://www.w3.org/2001/XMLSchema#integer",
            "property_kind": "datatype",
        },
    )


@_dataclass(eq=False, kw_only=True)
class OpenEntryTerms(LaboratoryAccessTerms):
    """Structural representation of OpenEntryTerms."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#OpenEntryTerms"
    )
    open_entry_value: list[int] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#openEntryValue",
            "ontology_range": "http://www.w3.org/2001/XMLSchema#integer",
            "property_kind": "datatype",
        },
    )


@_dataclass(eq=False, kw_only=True)
class RestrictedEntryTerms(LaboratoryAccessTerms):
    """Structural representation of RestrictedEntryTerms."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#RestrictedEntryTerms"
    )
    open_entry_value: list[int] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#openEntryValue",
            "ontology_range": "http://www.w3.org/2001/XMLSchema#integer",
            "property_kind": "datatype",
        },
    )


@_dataclass(eq=False, kw_only=True)
class ActionManifold(Representation):
    """Structural representation of ActionManifold."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#ActionManifold"
    )


@_dataclass(eq=False, kw_only=True)
class ActionRef(Representation, DesignatorTarget):
    """Structural representation of ActionRef."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#ActionRef"
    )


@_dataclass(eq=False, kw_only=True)
class ActionRelatedPlaceModel(Representation, ActionRelatedPlace):
    """Structural representation of ActionRelatedPlaceModel."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#ActionRelatedPlaceModel"
    )
    computed_from: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#computedFrom",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    predicts_success: list[_Any] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#predictsSuccess",
            "ontology_range": "",
            "property_kind": "datatype",
        },
    )


@_dataclass(eq=False, kw_only=True)
class ActionTokenDesignator(Representation, Designator):
    """Structural representation of ActionTokenDesignator."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#ActionTokenDesignator"
    )


@_dataclass(eq=False, kw_only=True)
class ActualCourse(Representation, DulDescription):
    """Structural representation of ActualCourse."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#ActualCourse"
    )


@_dataclass(eq=False, kw_only=True)
class Affordance(Representation, ActionRelatedConcept):
    """Structural representation of Affordance."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#Affordance"
    )
    affords_action: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#affordsAction",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    is_context_conditioned: list[_Any] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#isContextConditioned",
            "ontology_range": "",
            "property_kind": "datatype",
        },
    )
    requires_compatible_disposition: list[_Any] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#requiresCompatibleDisposition",
            "ontology_range": "",
            "property_kind": "datatype",
        },
    )


@_dataclass(eq=False, kw_only=True)
class Arrangement(Representation):
    """Structural representation of Arrangement."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#Arrangement"
    )
    has_arrangement_role: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#hasArrangementRole",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    requires_relation: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#requiresRelation",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    serves_activity: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#servesActivity",
            "ontology_range": "",
            "property_kind": "object",
        },
    )


@_dataclass(eq=False, kw_only=True)
class AuthorityTier(Representation):
    """Structural representation of AuthorityTier."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#AuthorityTier"
    )


@_dataclass(eq=False, kw_only=True)
class BayesianRisk(Representation):
    """Structural representation of BayesianRisk."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#BayesianRisk"
    )


@_dataclass(eq=False, kw_only=True)
class BehaviorFlaw(Representation, VerdictComponent):
    """Structural representation of BehaviorFlaw."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#BehaviorFlaw"
    )
    drives: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#drives",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    drives_revision: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#drivesRevision",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    escalates_to: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#escalatesTo",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    has_severity: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#hasSeverity",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    matched_over: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#matchedOver",
            "ontology_range": "",
            "property_kind": "object",
        },
    )


@_dataclass(eq=False, kw_only=True)
class BeliefState(Representation):
    """Structural representation of BeliefState."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#BeliefState"
    )
    has_part: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#hasPart",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    includes: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#includes",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    is_about: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#isAbout",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    resource_for: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#resourceFor",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    has_ambiguity: list[_Any] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#hasAmbiguity",
            "ontology_range": "",
            "property_kind": "datatype",
        },
    )
    has_inaccuracy: list[_Any] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#hasInaccuracy",
            "ontology_range": "",
            "property_kind": "datatype",
        },
    )
    is_continuously_updated: list[_Any] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#isContinuouslyUpdated",
            "ontology_range": "",
            "property_kind": "datatype",
        },
    )
    is_probabilistic: list[_Any] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#isProbabilistic",
            "ontology_range": "",
            "property_kind": "datatype",
        },
    )


@_dataclass(eq=False, kw_only=True)
class BelongingNorm(Representation):
    """Structural representation of BelongingNorm."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#BelongingNorm"
    )
    establishes_stability: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#establishesStability",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    governs_object_scope: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#governsObjectScope",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    prescribes_destination: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#prescribesDestination",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    prescribes_relation: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#prescribesRelation",
            "ontology_range": "",
            "property_kind": "object",
        },
    )


@_dataclass(eq=False, kw_only=True)
class BodyWorldConfiguration(Representation):
    """Structural representation of BodyWorldConfiguration."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#BodyWorldConfiguration"
    )


@_dataclass(eq=False, kw_only=True)
class Bug(Representation, DulDescription):
    """Structural representation of Bug."""

    __ontology_iri__: _ClassVar[str] = "http://aicor.knowledge/l2-ontology.owl#Bug"
    discovered_by: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#discoveredBy",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    indexes: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#indexes",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    specifies: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#specifies",
            "ontology_range": "",
            "property_kind": "object",
        },
    )


@_dataclass(eq=False, kw_only=True)
class CramDesignator(Representation, Designator):
    """Structural representation of CramDesignator."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#CRAMDesignator"
    )
    has_part: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#hasPart",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    has_resolution: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#hasResolution",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    is_immutable: list[_Any] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#isImmutable",
            "ontology_range": "",
            "property_kind": "datatype",
        },
    )


@_dataclass(eq=False, kw_only=True)
class CramPartialDescription(Representation, PartialDescription):
    """Structural representation of CramPartialDescription."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#CRAMPartialDescription"
    )


@_dataclass(eq=False, kw_only=True)
class CodeNode(Representation):
    """Structural representation of CodeNode."""

    __ontology_iri__: _ClassVar[str] = "http://aicor.knowledge/l2-ontology.owl#CodeNode"
    has_code_path: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#hasCodePath",
            "ontology_range": "",
            "property_kind": "object",
        },
    )


@_dataclass(eq=False, kw_only=True)
class CodePath(Representation):
    """Structural representation of CodePath."""

    __ontology_iri__: _ClassVar[str] = "http://aicor.knowledge/l2-ontology.owl#CodePath"


@_dataclass(eq=False, kw_only=True)
class CodeTree(Representation):
    """Structural representation of CodeTree."""

    __ontology_iri__: _ClassVar[str] = "http://aicor.knowledge/l2-ontology.owl#CodeTree"
    has_node: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#hasNode",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    has_tag_table: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#hasTagTable",
            "ontology_range": "",
            "property_kind": "object",
        },
    )


@_dataclass(eq=False, kw_only=True)
class Commitment(Representation):
    """Structural representation of Commitment."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#Commitment"
    )
    binds_instance: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#bindsInstance",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    reserved_for: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#reservedFor",
            "ontology_range": "",
            "property_kind": "object",
        },
    )


@_dataclass(eq=False, kw_only=True)
class ConceptPattern(Representation):
    """Structural representation of ConceptPattern."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#ConceptPattern"
    )
    pattern_in: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#patternIn",
            "ontology_range": "",
            "property_kind": "object",
        },
    )


@_dataclass(eq=False, kw_only=True)
class ConcurrentProcessInterference(Representation, DulDescription):
    """Structural representation of ConcurrentProcessInterference."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#ConcurrentProcessInterference"
    )
    composes_flow_in: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#composesFlowIn",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    is_symbolic: list[_Any] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#isSymbolic",
            "ontology_range": "",
            "property_kind": "datatype",
        },
    )


@_dataclass(eq=False, kw_only=True)
class ConstituentPart(Representation):
    """Structural representation of ConstituentPart."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#ConstituentPart"
    )


@_dataclass(eq=False, kw_only=True)
class ContinuousFlow(Representation, DulDescription):
    """Structural representation of ContinuousFlow."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#ContinuousFlow"
    )


@_dataclass(eq=False, kw_only=True)
class DesignatorConstraintSet(Representation, DulDescription):
    """Structural representation of DesignatorConstraintSet."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#DesignatorConstraintSet"
    )
    has_part: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#hasPart",
            "ontology_range": "",
            "property_kind": "object",
        },
    )


@_dataclass(eq=False, kw_only=True)
class DesignatorParameter(Representation, DulDescription):
    """Structural representation of DesignatorParameter."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#DesignatorParameter"
    )


@_dataclass(eq=False, kw_only=True)
class DiscreteMode(Representation, DulDescription):
    """Structural representation of DiscreteMode."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#DiscreteMode"
    )


@_dataclass(eq=False, kw_only=True)
class Divergence(Representation, DulDescription):
    """Structural representation of Divergence."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#Divergence"
    )


@_dataclass(eq=False, kw_only=True)
class ExecutionScenario(Representation, DulSituation):
    """Structural representation of ExecutionScenario."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#ExecutionScenario"
    )
    interpreted_by: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#interpretedBy",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    synchronized_with: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#synchronizedWith",
            "ontology_range": "",
            "property_kind": "object",
        },
    )


@_dataclass(eq=False, kw_only=True)
class ForceDynamicEvent(Representation, DulEvent):
    """Structural representation of ForceDynamicEvent."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#ForceDynamicEvent"
    )
    between_role_fillers: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#betweenRoleFillers",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    has_force_profile: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#hasForceProfile",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    involves_contact: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#involvesContact",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    is_transition_of: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#isTransitionOf",
            "ontology_range": "",
            "property_kind": "object",
        },
    )


@_dataclass(eq=False, kw_only=True)
class GeneralizedSuccessModel(Representation, DulDescription):
    """Structural representation of GeneralizedSuccessModel."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#GeneralizedSuccessModel"
    )


@_dataclass(eq=False, kw_only=True)
class GenerativeModel(Representation, GenerativeDependencyStructure):
    """Structural representation of GenerativeModel."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#GenerativeModel"
    )
    has_variant: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#hasVariant",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    maps_outcome_to: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#mapsOutcomeTo",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    maximizes: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#maximizes",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    is_joint_distribution: list[_Any] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#isJointDistribution",
            "ontology_range": "",
            "property_kind": "datatype",
        },
    )
    realized_by_krr: list[_Any] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#realizedByKRR",
            "ontology_range": "",
            "property_kind": "datatype",
        },
    )


@_dataclass(eq=False, kw_only=True)
class HumanActivityObservation(Representation):
    """Structural representation of HumanActivityObservation."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#HumanActivityObservation"
    )


@_dataclass(eq=False, kw_only=True)
class Intention(Representation, DulDescription):
    """Structural representation of Intention."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#Intention"
    )


@_dataclass(eq=False, kw_only=True)
class IntentionalStructure(Representation):
    """Structural representation of IntentionalStructure."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#IntentionalStructure"
    )
    has_participant_role: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#hasParticipantRole",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    has_telos: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#hasTelos",
            "ontology_range": "",
            "property_kind": "object",
        },
    )


@_dataclass(eq=False, kw_only=True)
class LossFunction(Representation):
    """Structural representation of LossFunction."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#LossFunction"
    )


@_dataclass(eq=False, kw_only=True)
class ManipulationPhase(Representation, DulDescription):
    """Structural representation of ManipulationPhase."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#ManipulationPhase"
    )


@_dataclass(eq=False, kw_only=True)
class MotionPhase(Representation, DulDescription):
    """Structural representation of MotionPhase."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#MotionPhase"
    )
    has_goal: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#hasGoal",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    has_phase_goal: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#hasPhaseGoal",
            "ontology_range": "",
            "property_kind": "object",
        },
    )


@_dataclass(eq=False, kw_only=True)
class MotionStatechart(Representation, ConstraintBasedControl):
    """Structural representation of MotionStatechart."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#MotionStatechart"
    )
    composed_of: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#composedOf",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    controlled_by: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#controlledBy",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    executes: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#executes",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    has_monitor: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#hasMonitor",
            "ontology_range": "",
            "property_kind": "object",
        },
    )


@_dataclass(eq=False, kw_only=True)
class MotionTask(Representation, DulDescription):
    """Structural representation of MotionTask."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#MotionTask"
    )
    is_single_purpose: list[_Any] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#isSinglePurpose",
            "ontology_range": "",
            "property_kind": "datatype",
        },
    )


@_dataclass(eq=False, kw_only=True)
class Narrative(Representation):
    """Structural representation of Narrative."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#Narrative"
    )


@_dataclass(eq=False, kw_only=True)
class ObjectTyping(Representation, DulDescription):
    """Structural representation of ObjectTyping."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#ObjectTyping"
    )


@_dataclass(eq=False, kw_only=True)
class Peam(Representation, TaskEnvironmentEmbodimentClass):
    """Structural representation of Peam."""

    __ontology_iri__: _ClassVar[str] = "http://aicor.knowledge/l2-ontology.owl#PEAM"
    comprises_manifold: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#comprisesManifold",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    constrained_by: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#constrainedBy",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    constrains: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#constrains",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    renders_tractable: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#rendersTractable",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    scopes_model: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#scopesModel",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    is_low_dimensional: list[_Any] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#isLowDimensional",
            "ontology_range": "",
            "property_kind": "datatype",
        },
    )
    is_open_frontier: list[_Any] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#isOpenFrontier",
            "ontology_range": "",
            "property_kind": "datatype",
        },
    )
    is_specialization: list[_Any] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#isSpecialization",
            "ontology_range": "",
            "property_kind": "datatype",
        },
    )


@_dataclass(eq=False, kw_only=True)
class PlaceProperty(Representation):
    """Structural representation of PlaceProperty."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#PlaceProperty"
    )


@_dataclass(eq=False, kw_only=True)
class PlacementRequest(Representation):
    """Structural representation of PlacementRequest."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#PlacementRequest"
    )
    has_authority: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#hasAuthority",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    has_request_constraint: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#hasRequestConstraint",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    has_request_scope: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#hasRequestScope",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    requests_object_scope: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#requestsObjectScope",
            "ontology_range": "",
            "property_kind": "object",
        },
    )


@_dataclass(eq=False, kw_only=True)
class PlacementTarget(Representation):
    """Structural representation of PlacementTarget."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#PlacementTarget"
    )
    has_destination: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#hasDestination",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    has_placement_region: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#hasPlacementRegion",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    has_spatial_relation: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#hasSpatialRelation",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    serves_activity: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#servesActivity",
            "ontology_range": "",
            "property_kind": "object",
        },
    )


@_dataclass(eq=False, kw_only=True)
class PlanFailure(Representation, DulDescription):
    """Structural representation of PlanFailure."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#PlanFailure"
    )
    escalates_to: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#escalatesTo",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    signals_description: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#signalsDescription",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    is_signalled: list[_Any] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#isSignalled",
            "ontology_range": "",
            "property_kind": "datatype",
        },
    )


@_dataclass(eq=False, kw_only=True)
class Policy(Representation, ConcurrentPolicy):
    """Structural representation of Policy."""

    __ontology_iri__: _ClassVar[str] = "http://aicor.knowledge/l2-ontology.owl#Policy"
    constrains: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#constrains",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    interrupts_primary: list[_Any] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#interruptsPrimary",
            "ontology_range": "",
            "property_kind": "datatype",
        },
    )
    is_concurrent: list[_Any] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#isConcurrent",
            "ontology_range": "",
            "property_kind": "datatype",
        },
    )


@_dataclass(eq=False, kw_only=True)
class Portion(Representation):
    """Structural representation of Portion."""

    __ontology_iri__: _ClassVar[str] = "http://aicor.knowledge/l2-ontology.owl#Portion"
    individuates: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#individuates",
            "ontology_range": "",
            "property_kind": "object",
        },
    )


@_dataclass(eq=False, kw_only=True)
class PossibleWorld(Representation, DulEntity):
    """Structural representation of PossibleWorld."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#PossibleWorld"
    )


@_dataclass(eq=False, kw_only=True)
class ProbabilisticHybridActionModel(Representation, ProbabilisticHybridProcessModel):
    """Structural representation of ProbabilisticHybridActionModel."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#ProbabilisticHybridActionModel"
    )
    has_part: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#hasPart",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    has_non_deterministic_effects: list[_Any] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#hasNonDeterministicEffects",
            "ontology_range": "",
            "property_kind": "datatype",
        },
    )
    is_symbolic: list[_Any] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#isSymbolic",
            "ontology_range": "",
            "property_kind": "datatype",
        },
    )


@_dataclass(eq=False, kw_only=True)
class Program(Representation):
    """Structural representation of Program."""

    __ontology_iri__: _ClassVar[str] = "http://aicor.knowledge/l2-ontology.owl#Program"
    specifies_process: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#specifiesProcess",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    is_executable: list[_Any] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#isExecutable",
            "ontology_range": "",
            "property_kind": "datatype",
        },
    )


@_dataclass(eq=False, kw_only=True)
class QualiaStructure(Representation):
    """Structural representation of QualiaStructure."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#QualiaStructure"
    )
    has_agentive_role: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#hasAgentiveRole",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    has_constitutive_role: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#hasConstitutiveRole",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    has_telic_role: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#hasTelicRole",
            "ontology_range": "",
            "property_kind": "object",
        },
    )


@_dataclass(eq=False, kw_only=True)
class ReferenceOutcome(Representation, DulDescription):
    """Structural representation of ReferenceOutcome."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#ReferenceOutcome"
    )


@_dataclass(eq=False, kw_only=True)
class RelationalRepresentation(Representation, DulDescription):
    """Structural representation of RelationalRepresentation."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#RelationalRepresentation"
    )


@_dataclass(eq=False, kw_only=True)
class RequestConstraint(Representation):
    """Structural representation of RequestConstraint."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#RequestConstraint"
    )


@_dataclass(eq=False, kw_only=True)
class RequestScope(Representation):
    """Structural representation of RequestScope."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#RequestScope"
    )


@_dataclass(eq=False, kw_only=True)
class RestorationObligation(Representation):
    """Structural representation of RestorationObligation."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#RestorationObligation"
    )
    authorized_by_activity: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#authorizedByActivity",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    disturbs_stability: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#disturbsStability",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    has_discharge_alternative: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#hasDischargeAlternative",
            "ontology_range": "",
            "property_kind": "object",
        },
    )


@_dataclass(eq=False, kw_only=True)
class RestorationType(Representation):
    """Structural representation of RestorationType."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#RestorationType"
    )


@_dataclass(eq=False, kw_only=True)
class SelectionNorm(Representation):
    """Structural representation of SelectionNorm."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#SelectionNorm"
    )


@_dataclass(eq=False, kw_only=True)
class ServingNorm(Representation):
    """Structural representation of ServingNorm."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#ServingNorm"
    )
    governs_consumable: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#governsConsumable",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    prescribes_portion: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#prescribesPortion",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    prescribes_vessel: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#prescribesVessel",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    serves_purpose: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#servesPurpose",
            "ontology_range": "",
            "property_kind": "object",
        },
    )


@_dataclass(eq=False, kw_only=True)
class SpatialRelation(Representation):
    """Structural representation of SpatialRelation."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#SpatialRelation"
    )


@_dataclass(eq=False, kw_only=True)
class StabilityCondition(Representation):
    """Structural representation of StabilityCondition."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#StabilityCondition"
    )
    preferred_configuration: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#preferredConfiguration",
            "ontology_range": "",
            "property_kind": "object",
        },
    )


@_dataclass(eq=False, kw_only=True)
class StateEstimate(Representation, DulDescription):
    """Structural representation of StateEstimate."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#StateEstimate"
    )


@_dataclass(eq=False, kw_only=True)
class StatisticalRelationalModel(Representation, KrrResource):
    """Structural representation of StatisticalRelationalModel."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#StatisticalRelationalModel"
    )
    assigns_probability_over: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#assignsProbabilityOver",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    expressed_in: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#expressedIn",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    realizes_model: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#realizesModel",
            "ontology_range": "",
            "property_kind": "object",
        },
    )


@_dataclass(eq=False, kw_only=True)
class SubstanceReference(Representation):
    """Structural representation of SubstanceReference."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#SubstanceReference"
    )
    reference_rides_on: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#referenceRidesOn",
            "ontology_range": "",
            "property_kind": "object",
        },
    )


@_dataclass(eq=False, kw_only=True)
class TagTable(Representation):
    """Structural representation of TagTable."""

    __ontology_iri__: _ClassVar[str] = "http://aicor.knowledge/l2-ontology.owl#TagTable"


@_dataclass(eq=False, kw_only=True)
class Task(Representation, DulDescription):
    """Structural representation of Task."""

    __ontology_iri__: _ClassVar[str] = "http://aicor.knowledge/l2-ontology.owl#Task"
    has_status: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#hasStatus",
            "ontology_range": "",
            "property_kind": "object",
        },
    )


@_dataclass(eq=False, kw_only=True)
class TaskNetwork(Representation, DulDescription):
    """Structural representation of TaskNetwork."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#TaskNetwork"
    )
    has_task: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#hasTask",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    has_top_task: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#hasTopTask",
            "ontology_range": "",
            "property_kind": "object",
        },
    )


@_dataclass(eq=False, kw_only=True)
class Telos(Representation):
    """Structural representation of Telos."""

    __ontology_iri__: _ClassVar[str] = "http://aicor.knowledge/l2-ontology.owl#Telos"
    defined_over_state: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#definedOverState",
            "ontology_range": "",
            "property_kind": "object",
        },
    )


@_dataclass(eq=False, kw_only=True)
class UnresolvedResolution(Representation, DesignatorResolution):
    """Structural representation of UnresolvedResolution."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#UnresolvedResolution"
    )


@_dataclass(eq=False, kw_only=True)
class Vessel(Representation):
    """Structural representation of Vessel."""

    __ontology_iri__: _ClassVar[str] = "http://aicor.knowledge/l2-ontology.owl#Vessel"


@_dataclass(eq=False, kw_only=True)
class CommensurableLaboratory(FederatedResearchLaboratory):
    """Structural representation of CommensurableLaboratory."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#CommensurableLaboratory"
    )
    comparable_under: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#comparableUnder",
            "ontology_range": "",
            "property_kind": "object",
        },
    )


@_dataclass(eq=False, kw_only=True)
class ActionEventType(EventType):
    """Structural representation of ActionEventType."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#ActionEventType"
    )


@_dataclass(eq=False, kw_only=True)
class CrackEvent(EventType):
    """Structural representation of CrackEvent."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#CrackEvent"
    )
    at_contact_pair: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#atContactPair",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    initiates_state_type: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#initiatesStateType",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    requires_harder_tool: list[bool] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#requiresHarderTool",
            "ontology_range": "http://www.w3.org/2001/XMLSchema#boolean",
            "property_kind": "datatype",
        },
    )


@_dataclass(eq=False, kw_only=True)
class FailPrimitive(EventType):
    """Structural representation of FailPrimitive."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#FailPrimitive"
    )
    produces_description: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#producesDescription",
            "ontology_range": "",
            "property_kind": "object",
        },
    )


@_dataclass(eq=False, kw_only=True)
class ForceDynamicEventType(EventType):
    """Structural representation of ForceDynamicEventType."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#ForceDynamicEventType"
    )


@_dataclass(eq=False, kw_only=True)
class InformationProcessingEvent(EventType):
    """Structural representation of InformationProcessingEvent."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#InformationProcessingEvent"
    )


@_dataclass(eq=False, kw_only=True)
class ObservationEvent(EventType):
    """Structural representation of ObservationEvent."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#ObservationEvent"
    )


@_dataclass(eq=False, kw_only=True)
class OpenUpEvent(EventType):
    """Structural representation of OpenUpEvent."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#OpenUpEvent"
    )
    acts_on_state: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#actsOnState",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    initiates_state_type: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#initiatesStateType",
            "ontology_range": "",
            "property_kind": "object",
        },
    )


@_dataclass(eq=False, kw_only=True)
class SpreadArrestEvent(EventType):
    """Structural representation of SpreadArrestEvent."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#SpreadArrestEvent"
    )


@_dataclass(eq=False, kw_only=True)
class ContainerClosureState(FluentType):
    """Structural representation of ContainerClosureState."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#ContainerClosureState"
    )


@_dataclass(eq=False, kw_only=True)
class Fluent(FluentType, Representation):
    """Structural representation of Fluent."""

    __ontology_iri__: _ClassVar[str] = "http://aicor.knowledge/l2-ontology.owl#Fluent"
    listens_to: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#listensTo",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    is_pulsed: list[_Any] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#isPulsed",
            "ontology_range": "",
            "property_kind": "datatype",
        },
    )


@_dataclass(eq=False, kw_only=True)
class TaskStatus(FluentType):
    """Structural representation of TaskStatus."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#TaskStatus"
    )
    status_code: list[str] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#statusCode",
            "ontology_range": "http://www.w3.org/2001/XMLSchema#string",
            "property_kind": "datatype",
        },
    )


@_dataclass(eq=False, kw_only=True)
class CouplingAlignedSubtree(SceneGraph):
    """Structural representation of CouplingAlignedSubtree."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#CouplingAlignedSubtree"
    )
    simulatable_under: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#simulatableUnder",
            "ontology_range": "",
            "property_kind": "object",
        },
    )


@_dataclass(eq=False, kw_only=True)
class SemanticObjectMap(SceneGraph, Representation):
    """Structural representation of SemanticObjectMap."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#SemanticObjectMap"
    )
    has_part: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#hasPart",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    resolved_against_by: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#resolvedAgainstBy",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    has_functional_typing: list[_Any] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#hasFunctionalTyping",
            "ontology_range": "",
            "property_kind": "datatype",
        },
    )


@_dataclass(eq=False, kw_only=True)
class Carrier(PhysicalIndividual):
    """Structural representation of Carrier."""

    __ontology_iri__: _ClassVar[str] = "http://aicor.knowledge/l2-ontology.owl#Carrier"


@_dataclass(eq=False, kw_only=True)
class Container_(PhysicalIndividual):
    """Structural representation of Container_."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#Container"
    )
    contains: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#contains",
            "ontology_range": "",
            "property_kind": "object",
        },
    )


@_dataclass(eq=False, kw_only=True)
class Stack(PhysicalIndividual):
    """Structural representation of Stack."""

    __ontology_iri__: _ClassVar[str] = "http://aicor.knowledge/l2-ontology.owl#Stack"
    composed_of: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#composedOf",
            "ontology_range": "",
            "property_kind": "object",
        },
    )


@_dataclass(eq=False, kw_only=True)
class CompetenceHypothesis(ManipulationHypothesis):
    """Structural representation of CompetenceHypothesis."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#CompetenceHypothesis"
    )
    predicts_competence: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#predictsCompetence",
            "ontology_range": "",
            "property_kind": "object",
        },
    )


@_dataclass(eq=False, kw_only=True)
class NecessityHypothesis(ManipulationHypothesis):
    """Structural representation of NecessityHypothesis."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#NecessityHypothesis"
    )
    predicts_required_capability: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#predictsRequiredCapability",
            "ontology_range": "",
            "property_kind": "object",
        },
    )


@_dataclass(eq=False, kw_only=True)
class BodyWorldProcessMicrotheory(WorldStructureModel):
    """Structural representation of BodyWorldProcessMicrotheory."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#BodyWorldProcessMicrotheory"
    )
    formalized_by: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#formalizedBy",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    has_scope_condition: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#hasScopeCondition",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    has_template: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#hasTemplate",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    interprets_via: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#interpretsVia",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    meets: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#meets",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    resource_for: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#resourceFor",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    is_scoped: list[bool] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#isScoped",
            "ontology_range": "http://www.w3.org/2001/XMLSchema#boolean",
            "property_kind": "datatype",
        },
    )


@_dataclass(eq=False, kw_only=True)
class AccountabilityClaim(CompetenceProperty):
    """Structural representation of AccountabilityClaim."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#AccountabilityClaim"
    )
    property_kind: list[str] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#propertyKind",
            "ontology_range": "http://www.w3.org/2001/XMLSchema#string",
            "property_kind": "datatype",
        },
    )


@_dataclass(eq=False, kw_only=True)
class AwarenessClaim(CompetenceProperty):
    """Structural representation of AwarenessClaim."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#AwarenessClaim"
    )
    property_kind: list[str] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#propertyKind",
            "ontology_range": "http://www.w3.org/2001/XMLSchema#string",
            "property_kind": "datatype",
        },
    )


@_dataclass(eq=False, kw_only=True)
class CompletenessClaim(CompetenceProperty):
    """Structural representation of CompletenessClaim."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#CompletenessClaim"
    )
    property_kind: list[str] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#propertyKind",
            "ontology_range": "http://www.w3.org/2001/XMLSchema#string",
            "property_kind": "datatype",
        },
    )


@_dataclass(eq=False, kw_only=True)
class CorrectnessClaim(CompetenceProperty):
    """Structural representation of CorrectnessClaim."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#CorrectnessClaim"
    )
    property_kind: list[str] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#propertyKind",
            "ontology_range": "http://www.w3.org/2001/XMLSchema#string",
            "property_kind": "datatype",
        },
    )


@_dataclass(eq=False, kw_only=True)
class PerformanceClaim(CompetenceProperty):
    """Structural representation of PerformanceClaim."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#PerformanceClaim"
    )
    property_kind: list[str] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#propertyKind",
            "ontology_range": "http://www.w3.org/2001/XMLSchema#string",
            "property_kind": "datatype",
        },
    )


@_dataclass(eq=False, kw_only=True)
class PotencyClaim(CompetenceProperty):
    """Structural representation of PotencyClaim."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#PotencyClaim"
    )
    property_kind: list[str] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#propertyKind",
            "ontology_range": "http://www.w3.org/2001/XMLSchema#string",
            "property_kind": "datatype",
        },
    )


@_dataclass(eq=False, kw_only=True)
class SafetyClaim(CompetenceProperty):
    """Structural representation of SafetyClaim."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#SafetyClaim"
    )
    property_kind: list[str] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#propertyKind",
            "ontology_range": "http://www.w3.org/2001/XMLSchema#string",
            "property_kind": "datatype",
        },
    )


@_dataclass(eq=False, kw_only=True)
class RequestedActionRef(ActionRef):
    """Structural representation of RequestedActionRef."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#RequestedActionRef"
    )


@_dataclass(eq=False, kw_only=True)
class ExecutedActionDesignator(ActionTokenDesignator):
    """Structural representation of ExecutedActionDesignator."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#ExecutedActionDesignator"
    )


@_dataclass(eq=False, kw_only=True)
class RequestDesignator(ActionTokenDesignator):
    """Structural representation of RequestDesignator."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#RequestDesignator"
    )


@_dataclass(eq=False, kw_only=True)
class Load(Arrangement):
    """Structural representation of Load."""

    __ontology_iri__: _ClassVar[str] = "http://aicor.knowledge/l2-ontology.owl#Load"
    carried_by: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#carriedBy",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    has_payload: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#hasPayload",
            "ontology_range": "",
            "property_kind": "object",
        },
    )


@_dataclass(eq=False, kw_only=True)
class GuestAuthority(AuthorityTier):
    """Structural representation of GuestAuthority."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#GuestAuthority"
    )
    authority_kind: list[str] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#authorityKind",
            "ontology_range": "http://www.w3.org/2001/XMLSchema#string",
            "property_kind": "datatype",
        },
    )


@_dataclass(eq=False, kw_only=True)
class InferredAuthority(AuthorityTier):
    """Structural representation of InferredAuthority."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#InferredAuthority"
    )
    authority_kind: list[str] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#authorityKind",
            "ontology_range": "http://www.w3.org/2001/XMLSchema#string",
            "property_kind": "datatype",
        },
    )


@_dataclass(eq=False, kw_only=True)
class ResidentAuthority(AuthorityTier):
    """Structural representation of ResidentAuthority."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#ResidentAuthority"
    )
    authority_kind: list[str] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#authorityKind",
            "ontology_range": "http://www.w3.org/2001/XMLSchema#string",
            "property_kind": "datatype",
        },
    )


@_dataclass(eq=False, kw_only=True)
class ServedConfiguration(BodyWorldConfiguration):
    """Structural representation of ServedConfiguration."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#ServedConfiguration"
    )
    for_purpose: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#forPurpose",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    has_portion: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#hasPortion",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    in_vessel: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#inVessel",
            "ontology_range": "",
            "property_kind": "object",
        },
    )


@_dataclass(eq=False, kw_only=True)
class AttributiveDesignator(CramDesignator):
    """Structural representation of AttributiveDesignator."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#AttributiveDesignator"
    )


@_dataclass(eq=False, kw_only=True)
class GroundedDesignator(CramDesignator):
    """Structural representation of GroundedDesignator."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#GroundedDesignator"
    )
    all_parameters_bound: list[bool] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#allParametersBound",
            "ontology_range": "http://www.w3.org/2001/XMLSchema#boolean",
            "property_kind": "datatype",
        },
    )


@_dataclass(eq=False, kw_only=True)
class LocationDesignator(CramDesignator):
    """Structural representation of LocationDesignator."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#LocationDesignator"
    )
    resolves_to_state: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#resolvesToState",
            "ontology_range": "",
            "property_kind": "object",
        },
    )


@_dataclass(eq=False, kw_only=True)
class MotionDesignator(CramDesignator):
    """Structural representation of MotionDesignator."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#MotionDesignator"
    )


@_dataclass(eq=False, kw_only=True)
class ObjectDesignator(CramDesignator):
    """Structural representation of ObjectDesignator."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#ObjectDesignator"
    )
    resolved_against: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#resolvedAgainst",
            "ontology_range": "",
            "property_kind": "object",
        },
    )


@_dataclass(eq=False, kw_only=True)
class ReferentialDesignator(CramDesignator):
    """Structural representation of ReferentialDesignator."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#ReferentialDesignator"
    )


@_dataclass(eq=False, kw_only=True)
class ForceDynamicState(DiscreteMode):
    """Structural representation of ForceDynamicState."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#ForceDynamicState"
    )


@_dataclass(eq=False, kw_only=True)
class EffluxOnsetEvent(ForceDynamicEvent):
    """Structural representation of EffluxOnsetEvent."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#EffluxOnsetEvent"
    )


@_dataclass(eq=False, kw_only=True)
class FractureEvent(ForceDynamicEvent):
    """Structural representation of FractureEvent."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#FractureEvent"
    )


@_dataclass(eq=False, kw_only=True)
class GraspEvent(ForceDynamicEvent):
    """Structural representation of GraspEvent."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#GraspEvent"
    )


@_dataclass(eq=False, kw_only=True)
class CuttingIntention(IntentionalStructure):
    """Structural representation of CuttingIntention."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#CuttingIntention"
    )


@_dataclass(eq=False, kw_only=True)
class PouringIntention(IntentionalStructure):
    """Structural representation of PouringIntention."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#PouringIntention"
    )


@_dataclass(eq=False, kw_only=True)
class CuttingPhase(MotionPhase):
    """Structural representation of CuttingPhase."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#CuttingPhase"
    )


@_dataclass(eq=False, kw_only=True)
class PouringPhase(MotionPhase):
    """Structural representation of PouringPhase."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#PouringPhase"
    )


@_dataclass(eq=False, kw_only=True)
class ElaborationGraph(Narrative):
    """Structural representation of ElaborationGraph."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#ElaborationGraph"
    )
    elaborates: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#elaborates",
            "ontology_range": "",
            "property_kind": "object",
        },
    )


@_dataclass(eq=False, kw_only=True)
class Neem(Narrative):
    """Structural representation of Neem."""

    __ontology_iri__: _ClassVar[str] = "http://aicor.knowledge/l2-ontology.owl#NEEM"
    derived_from: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#derivedFrom",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    drives: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#drives",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    has_part: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#hasPart",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    is_about: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#isAbout",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    is_derived: list[_Any] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#isDerived",
            "ontology_range": "",
            "property_kind": "datatype",
        },
    )
    records_intention_and_actual: list[_Any] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#recordsIntentionAndActual",
            "ontology_range": "",
            "property_kind": "datatype",
        },
    )


@_dataclass(eq=False, kw_only=True)
class OverridingRequest(PlacementRequest):
    """Structural representation of OverridingRequest."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#OverridingRequest"
    )
    defeats: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#defeats",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    licenses_disturbance_of: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#licensesDisturbanceOf",
            "ontology_range": "",
            "property_kind": "object",
        },
    )


@_dataclass(eq=False, kw_only=True)
class StandingRequest(PlacementRequest):
    """Structural representation of StandingRequest."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#StandingRequest"
    )
    proposes_amendment_of: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#proposesAmendmentOf",
            "ontology_range": "",
            "property_kind": "object",
        },
    )


@_dataclass(eq=False, kw_only=True)
class CramControlProgram(Program, ManipulationControlProgram):
    """Structural representation of CramControlProgram."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#CRAMControlProgram"
    )
    executed_on: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#executedOn",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    generates_body_motion: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#generatesBodyMotion",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    hosted_on: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#hostedOn",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    moves_body: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#movesBody",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    produces: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#produces",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    realized_by: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#realizedBy",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    satisfies_request: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#satisfiesRequest",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    steers: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#steers",
            "ontology_range": "",
            "property_kind": "object",
        },
    )


@_dataclass(eq=False, kw_only=True)
class GeneralizedActionPlan(Program, ManipulationControlProgram):
    """Structural representation of GeneralizedActionPlan."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#GeneralizedActionPlan"
    )
    contextualized_by: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#contextualizedBy",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    embeds: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#embeds",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    expands_into: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#expandsInto",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    has_part: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#hasPart",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    is_underdetermined: list[_Any] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#isUnderdetermined",
            "ontology_range": "",
            "property_kind": "datatype",
        },
    )


@_dataclass(eq=False, kw_only=True)
class MotionPlan(Program, ControlProgram):
    """Structural representation of MotionPlan."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#MotionPlan"
    )
    has_motion_phase: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#hasMotionPhase",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    parameterized_by: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#parameterizedBy",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    produces: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#produces",
            "ontology_range": "",
            "property_kind": "object",
        },
    )


@_dataclass(eq=False, kw_only=True)
class Plan(Program, ControlProgram):
    """Structural representation of Plan."""

    __ontology_iri__: _ClassVar[str] = "http://aicor.knowledge/l2-ontology.owl#Plan"
    manipulated_at_runtime_by: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#manipulatedAtRuntimeBy",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    reasoned_about_by: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#reasonedAboutBy",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    can_be_reasoned_about: list[_Any] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#canBeReasonedAbout",
            "ontology_range": "",
            "property_kind": "datatype",
        },
    )
    is_manipulable_at_runtime: list[_Any] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#isManipulableAtRuntime",
            "ontology_range": "",
            "property_kind": "datatype",
        },
    )


@_dataclass(eq=False, kw_only=True)
class Diverges(ReferenceOutcome):
    """Structural representation of Diverges."""

    __ontology_iri__: _ClassVar[str] = "http://aicor.knowledge/l2-ontology.owl#Diverges"


@_dataclass(eq=False, kw_only=True)
class Fails(ReferenceOutcome):
    """Structural representation of Fails."""

    __ontology_iri__: _ClassVar[str] = "http://aicor.knowledge/l2-ontology.owl#Fails"


@_dataclass(eq=False, kw_only=True)
class Realizes(ReferenceOutcome):
    """Structural representation of Realizes."""

    __ontology_iri__: _ClassVar[str] = "http://aicor.knowledge/l2-ontology.owl#Realizes"


@_dataclass(eq=False, kw_only=True)
class DestinationConstraint(RequestConstraint):
    """Structural representation of DestinationConstraint."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#DestinationConstraint"
    )
    constrains_destination: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#constrainsDestination",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    constraint_kind: list[str] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#constraintKind",
            "ontology_range": "http://www.w3.org/2001/XMLSchema#string",
            "property_kind": "datatype",
        },
    )


@_dataclass(eq=False, kw_only=True)
class ExclusionConstraint(RequestConstraint):
    """Structural representation of ExclusionConstraint."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#ExclusionConstraint"
    )
    exempts_object_scope: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#exemptsObjectScope",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    constraint_kind: list[str] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#constraintKind",
            "ontology_range": "http://www.w3.org/2001/XMLSchema#string",
            "property_kind": "datatype",
        },
    )


@_dataclass(eq=False, kw_only=True)
class PropertyConstraint(RequestConstraint):
    """Structural representation of PropertyConstraint."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#PropertyConstraint"
    )
    requires_property: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#requiresProperty",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    constraint_kind: list[str] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#constraintKind",
            "ontology_range": "http://www.w3.org/2001/XMLSchema#string",
            "property_kind": "datatype",
        },
    )


@_dataclass(eq=False, kw_only=True)
class RelationConstraint(RequestConstraint):
    """Structural representation of RelationConstraint."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#RelationConstraint"
    )
    constrains_relation: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#constrainsRelation",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    relates_to_description: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#relatesToDescription",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    constraint_kind: list[str] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#constraintKind",
            "ontology_range": "http://www.w3.org/2001/XMLSchema#string",
            "property_kind": "datatype",
        },
    )


@_dataclass(eq=False, kw_only=True)
class ActivityBoundScope(RequestScope):
    """Structural representation of ActivityBoundScope."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#ActivityBoundScope"
    )
    scope_kind: list[str] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#scopeKind",
            "ontology_range": "http://www.w3.org/2001/XMLSchema#string",
            "property_kind": "datatype",
        },
    )


@_dataclass(eq=False, kw_only=True)
class EpisodicScope(RequestScope):
    """Structural representation of EpisodicScope."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#EpisodicScope"
    )
    scope_kind: list[str] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#scopeKind",
            "ontology_range": "http://www.w3.org/2001/XMLSchema#string",
            "property_kind": "datatype",
        },
    )


@_dataclass(eq=False, kw_only=True)
class StandingScope(RequestScope):
    """Structural representation of StandingScope."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#StandingScope"
    )
    scope_kind: list[str] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#scopeKind",
            "ontology_range": "http://www.w3.org/2001/XMLSchema#string",
            "property_kind": "datatype",
        },
    )


@_dataclass(eq=False, kw_only=True)
class ConfigurationRestoration(RestorationType):
    """Structural representation of ConfigurationRestoration."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#ConfigurationRestoration"
    )
    restoration_kind: list[str] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#restorationKind",
            "ontology_range": "http://www.w3.org/2001/XMLSchema#string",
            "property_kind": "datatype",
        },
    )


@_dataclass(eq=False, kw_only=True)
class ExactReturn(RestorationType):
    """Structural representation of ExactReturn."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#ExactReturn"
    )
    restoration_kind: list[str] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#restorationKind",
            "ontology_range": "http://www.w3.org/2001/XMLSchema#string",
            "property_kind": "datatype",
        },
    )


@_dataclass(eq=False, kw_only=True)
class FunctionalReturn(RestorationType):
    """Structural representation of FunctionalReturn."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#FunctionalReturn"
    )
    restoration_kind: list[str] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#restorationKind",
            "ontology_range": "http://www.w3.org/2001/XMLSchema#string",
            "property_kind": "datatype",
        },
    )


@_dataclass(eq=False, kw_only=True)
class StateMediatedReturn(RestorationType):
    """Structural representation of StateMediatedReturn."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#StateMediatedReturn"
    )
    restoration_kind: list[str] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#restorationKind",
            "ontology_range": "http://www.w3.org/2001/XMLSchema#string",
            "property_kind": "datatype",
        },
    )


@_dataclass(eq=False, kw_only=True)
class SuccessorStateResolution(RestorationType):
    """Structural representation of SuccessorStateResolution."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#SuccessorStateResolution"
    )
    restoration_kind: list[str] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#restorationKind",
            "ontology_range": "http://www.w3.org/2001/XMLSchema#string",
            "property_kind": "datatype",
        },
    )


@_dataclass(eq=False, kw_only=True)
class AccessibleFrom(SpatialRelation):
    """Structural representation of AccessibleFrom."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#AccessibleFrom"
    )
    relation_kind: list[str] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#relationKind",
            "ontology_range": "http://www.w3.org/2001/XMLSchema#string",
            "property_kind": "datatype",
        },
    )


@_dataclass(eq=False, kw_only=True)
class AlignedWith(SpatialRelation):
    """Structural representation of AlignedWith."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#AlignedWith"
    )
    relation_kind: list[str] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#relationKind",
            "ontology_range": "http://www.w3.org/2001/XMLSchema#string",
            "property_kind": "datatype",
        },
    )


@_dataclass(eq=False, kw_only=True)
class AttachedTo(SpatialRelation):
    """Structural representation of AttachedTo."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#AttachedTo"
    )
    relation_kind: list[str] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#relationKind",
            "ontology_range": "http://www.w3.org/2001/XMLSchema#string",
            "property_kind": "datatype",
        },
    )


@_dataclass(eq=False, kw_only=True)
class Inside(SpatialRelation):
    """Structural representation of Inside."""

    __ontology_iri__: _ClassVar[str] = "http://aicor.knowledge/l2-ontology.owl#Inside"
    relation_kind: list[str] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#relationKind",
            "ontology_range": "http://www.w3.org/2001/XMLSchema#string",
            "property_kind": "datatype",
        },
    )


@_dataclass(eq=False, kw_only=True)
class On(SpatialRelation):
    """Structural representation of On."""

    __ontology_iri__: _ClassVar[str] = "http://aicor.knowledge/l2-ontology.owl#On"
    relation_kind: list[str] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#relationKind",
            "ontology_range": "http://www.w3.org/2001/XMLSchema#string",
            "property_kind": "datatype",
        },
    )


@_dataclass(eq=False, kw_only=True)
class ControlStructure(Task):
    """Structural representation of ControlStructure."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#ControlStructure"
    )
    has_subtask: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#hasSubtask",
            "ontology_range": "",
            "property_kind": "object",
        },
    )


@_dataclass(eq=False, kw_only=True)
class CuttingTelos(Telos):
    """Structural representation of CuttingTelos."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#CuttingTelos"
    )


@_dataclass(eq=False, kw_only=True)
class PouringTelos(Telos):
    """Structural representation of PouringTelos."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#PouringTelos"
    )


@_dataclass(eq=False, kw_only=True)
class Bowl(Vessel):
    """Structural representation of Bowl."""

    __ontology_iri__: _ClassVar[str] = "http://aicor.knowledge/l2-ontology.owl#Bowl"
    vessel_kind: list[str] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#vesselKind",
            "ontology_range": "http://www.w3.org/2001/XMLSchema#string",
            "property_kind": "datatype",
        },
    )


@_dataclass(eq=False, kw_only=True)
class Cup(Vessel):
    """Structural representation of Cup."""

    __ontology_iri__: _ClassVar[str] = "http://aicor.knowledge/l2-ontology.owl#Cup"
    vessel_kind: list[str] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#vesselKind",
            "ontology_range": "http://www.w3.org/2001/XMLSchema#string",
            "property_kind": "datatype",
        },
    )


@_dataclass(eq=False, kw_only=True)
class DrinkingVessel(Vessel):
    """Structural representation of DrinkingVessel."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#DrinkingVessel"
    )
    vessel_kind: list[str] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#vesselKind",
            "ontology_range": "http://www.w3.org/2001/XMLSchema#string",
            "property_kind": "datatype",
        },
    )


@_dataclass(eq=False, kw_only=True)
class Plate(Vessel):
    """Structural representation of Plate."""

    __ontology_iri__: _ClassVar[str] = "http://aicor.knowledge/l2-ontology.owl#Plate"
    vessel_kind: list[str] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#vesselKind",
            "ontology_range": "http://www.w3.org/2001/XMLSchema#string",
            "property_kind": "datatype",
        },
    )


@_dataclass(eq=False, kw_only=True)
class RuptureEventType(ForceDynamicEventType):
    """Structural representation of RuptureEventType."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#RuptureEventType"
    )
    crossing_is: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#crossingIs",
            "ontology_range": "",
            "property_kind": "object",
        },
    )


@_dataclass(eq=False, kw_only=True)
class SlipEventType(ForceDynamicEventType):
    """Structural representation of SlipEventType."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#SlipEventType"
    )
    crossing_is: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#crossingIs",
            "ontology_range": "",
            "property_kind": "object",
        },
    )


@_dataclass(eq=False, kw_only=True)
class PlanInterpretationEvent(InformationProcessingEvent):
    """Structural representation of PlanInterpretationEvent."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#PlanInterpretationEvent"
    )


@_dataclass(eq=False, kw_only=True)
class ThermalSetArrest(SpreadArrestEvent):
    """Structural representation of ThermalSetArrest."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#ThermalSetArrest"
    )


@_dataclass(eq=False, kw_only=True)
class YieldEquilibriumArrest(SpreadArrestEvent):
    """Structural representation of YieldEquilibriumArrest."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#YieldEquilibriumArrest"
    )


@_dataclass(eq=False, kw_only=True)
class ClosedContainerState(ContainerClosureState):
    """Structural representation of ClosedContainerState."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#ClosedContainerState"
    )


@_dataclass(eq=False, kw_only=True)
class OpenContainerState(ContainerClosureState):
    """Structural representation of OpenContainerState."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#OpenContainerState"
    )


@_dataclass(eq=False, kw_only=True)
class ActiveStatus(TaskStatus):
    """Structural representation of ActiveStatus."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#ActiveStatus"
    )


@_dataclass(eq=False, kw_only=True)
class CreatedStatus(TaskStatus):
    """Structural representation of CreatedStatus."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#CreatedStatus"
    )


@_dataclass(eq=False, kw_only=True)
class NonterminalStatus(TaskStatus):
    """Structural representation of NonterminalStatus."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#NonterminalStatus"
    )


@_dataclass(eq=False, kw_only=True)
class TerminalStatus(TaskStatus):
    """Structural representation of TerminalStatus."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#TerminalStatus"
    )


@_dataclass(eq=False, kw_only=True)
class MonotonicityHypothesis(CompetenceHypothesis):
    """Structural representation of MonotonicityHypothesis."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#MonotonicityHypothesis"
    )
    assessed_over: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#assessedOver",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    predicts_degree_trend: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#predictsDegreeTrend",
            "ontology_range": "",
            "property_kind": "object",
        },
    )


@_dataclass(eq=False, kw_only=True)
class TransferHypothesis(CompetenceHypothesis):
    """Structural representation of TransferHypothesis."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#TransferHypothesis"
    )
    predicts_transfer_to: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#predictsTransferTo",
            "ontology_range": "",
            "property_kind": "object",
        },
    )


@_dataclass(eq=False, kw_only=True)
class ApplicationCentricMicrotheory(BodyWorldProcessMicrotheory):
    """Structural representation of ApplicationCentricMicrotheory."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#ApplicationCentricMicrotheory"
    )
    focuses_on: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#focusesOn",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    microtheory_focus: list[str] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#microtheoryFocus",
            "ontology_range": "http://www.w3.org/2001/XMLSchema#string",
            "property_kind": "datatype",
        },
    )


@_dataclass(eq=False, kw_only=True)
class BodyCentricMicrotheory(BodyWorldProcessMicrotheory):
    """Structural representation of BodyCentricMicrotheory."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#BodyCentricMicrotheory"
    )
    focuses_on: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#focusesOn",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    microtheory_focus: list[str] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#microtheoryFocus",
            "ontology_range": "http://www.w3.org/2001/XMLSchema#string",
            "property_kind": "datatype",
        },
    )


@_dataclass(eq=False, kw_only=True)
class OpenSurfaceSpreading(BodyWorldProcessMicrotheory):
    """Structural representation of OpenSurfaceSpreading."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#OpenSurfaceSpreading"
    )
    arrested_by_event: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#arrestedByEvent",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    has_coverage_target: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#hasCoverageTarget",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    has_film_thickness: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#hasFilmThickness",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    has_spread_radius: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#hasSpreadRadius",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    on_flat_object: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#onFlatObject",
            "ontology_range": "",
            "property_kind": "object",
        },
    )


@_dataclass(eq=False, kw_only=True)
class ProcessCentricMicrotheory(BodyWorldProcessMicrotheory):
    """Structural representation of ProcessCentricMicrotheory."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#ProcessCentricMicrotheory"
    )
    focuses_on: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#focusesOn",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    microtheory_focus: list[str] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#microtheoryFocus",
            "ontology_range": "http://www.w3.org/2001/XMLSchema#string",
            "property_kind": "datatype",
        },
    )


@_dataclass(eq=False, kw_only=True)
class DivergentDesignator(ExecutedActionDesignator):
    """Structural representation of DivergentDesignator."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#DivergentDesignator"
    )
    satisfies_reference: list[bool] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#satisfiesReference",
            "ontology_range": "http://www.w3.org/2001/XMLSchema#boolean",
            "property_kind": "datatype",
        },
    )


@_dataclass(eq=False, kw_only=True)
class RealizingDesignator(ExecutedActionDesignator):
    """Structural representation of RealizingDesignator."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#RealizingDesignator"
    )
    satisfies_reference: list[bool] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#satisfiesReference",
            "ontology_range": "http://www.w3.org/2001/XMLSchema#boolean",
            "property_kind": "datatype",
        },
    )


@_dataclass(eq=False, kw_only=True)
class ActionDesignator(AttributiveDesignator):
    """Structural representation of ActionDesignator."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#ActionDesignator"
    )


@_dataclass(eq=False, kw_only=True)
class EntityDesignator(ReferentialDesignator, ObjectDesignator):
    """Structural representation of EntityDesignator."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#EntityDesignator"
    )


@_dataclass(eq=False, kw_only=True)
class AccumulatedState(ForceDynamicState):
    """Structural representation of AccumulatedState."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#AccumulatedState"
    )


@_dataclass(eq=False, kw_only=True)
class ContainedState(ForceDynamicState):
    """Structural representation of ContainedState."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#ContainedState"
    )


@_dataclass(eq=False, kw_only=True)
class FlowingState(ForceDynamicState):
    """Structural representation of FlowingState."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#FlowingState"
    )


@_dataclass(eq=False, kw_only=True)
class GraspedState(ForceDynamicState):
    """Structural representation of GraspedState."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#GraspedState"
    )


@_dataclass(eq=False, kw_only=True)
class SeparatedState(ForceDynamicState):
    """Structural representation of SeparatedState."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#SeparatedState"
    )


@_dataclass(eq=False, kw_only=True)
class StressedState(ForceDynamicState):
    """Structural representation of StressedState."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#StressedState"
    )


@_dataclass(eq=False, kw_only=True)
class ArrangementTemplate(GeneralizedActionPlan):
    """Structural representation of ArrangementTemplate."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#ArrangementTemplate"
    )
    schema_of: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#schemaOf",
            "ontology_range": "",
            "property_kind": "object",
        },
    )


@_dataclass(eq=False, kw_only=True)
class CourseOfAction(GeneralizedActionPlan):
    """Structural representation of CourseOfAction."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#CourseOfAction"
    )


@_dataclass(eq=False, kw_only=True)
class ManipulationActionCategory(GeneralizedActionPlan):
    """Structural representation of ManipulationActionCategory."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#ManipulationActionCategory"
    )
    has_causal_structure: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#hasCausalStructure",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    has_intentional_structure: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#hasIntentionalStructure",
            "ontology_range": "",
            "property_kind": "object",
        },
    )


@_dataclass(eq=False, kw_only=True)
class PreparationMethod(GeneralizedActionPlan):
    """Structural representation of PreparationMethod."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#PreparationMethod"
    )
    produces_configuration: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#producesConfiguration",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    realized_by_category: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#realizedByCategory",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    requires_ingredient: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#requiresIngredient",
            "ontology_range": "",
            "property_kind": "object",
        },
    )


@_dataclass(eq=False, kw_only=True)
class ForceDynamicTemplate(MotionPlan, ProbabilisticHybridActionModel):
    """Structural representation of ForceDynamicTemplate."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#ForceDynamicTemplate"
    )
    atlas_of: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#atlasOf",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    has_fd_event: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#hasFDEvent",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    has_fd_state: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#hasFDState",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    realizes_intention: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#realizesIntention",
            "ontology_range": "",
            "property_kind": "object",
        },
    )


@_dataclass(eq=False, kw_only=True)
class StructuredReactivePlan(Plan):
    """Structural representation of StructuredReactivePlan."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#StructuredReactivePlan"
    )
    executes_into: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#executesInto",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    has_code_tree: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#hasCodeTree",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    interpreted_by: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#interpretedBy",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    is_dual_with: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#isDualWith",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    is_dual: list[_Any] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#isDual",
            "ontology_range": "",
            "property_kind": "datatype",
        },
    )
    is_transparent: list[_Any] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#isTransparent",
            "ontology_range": "",
            "property_kind": "datatype",
        },
    )


@_dataclass(eq=False, kw_only=True)
class Par(ControlStructure):
    """Structural representation of Par."""

    __ontology_iri__: _ClassVar[str] = "http://aicor.knowledge/l2-ontology.owl#Par"
    orders_subtasks: list[bool] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#ordersSubtasks",
            "ontology_range": "http://www.w3.org/2001/XMLSchema#boolean",
            "property_kind": "datatype",
        },
    )


@_dataclass(eq=False, kw_only=True)
class PerformTask(ControlStructure):
    """Structural representation of PerformTask."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#PerformTask"
    )
    has_goal_clause: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#hasGoalClause",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    performs_designator: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#performsDesignator",
            "ontology_range": "",
            "property_kind": "object",
        },
    )


@_dataclass(eq=False, kw_only=True)
class Seq(ControlStructure):
    """Structural representation of Seq."""

    __ontology_iri__: _ClassVar[str] = "http://aicor.knowledge/l2-ontology.owl#Seq"
    orders_subtasks: list[bool] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#ordersSubtasks",
            "ontology_range": "http://www.w3.org/2001/XMLSchema#boolean",
            "property_kind": "datatype",
        },
    )


@_dataclass(eq=False, kw_only=True)
class TryAll(ControlStructure):
    """Structural representation of TryAll."""

    __ontology_iri__: _ClassVar[str] = "http://aicor.knowledge/l2-ontology.owl#TryAll"
    catches_failure: list[bool] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#catchesFailure",
            "ontology_range": "http://www.w3.org/2001/XMLSchema#boolean",
            "property_kind": "datatype",
        },
    )
    orders_subtasks: list[bool] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#ordersSubtasks",
            "ontology_range": "http://www.w3.org/2001/XMLSchema#boolean",
            "property_kind": "datatype",
        },
    )


@_dataclass(eq=False, kw_only=True)
class TryInOrder(ControlStructure):
    """Structural representation of TryInOrder."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#TryInOrder"
    )
    catches_failure: list[bool] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#catchesFailure",
            "ontology_range": "http://www.w3.org/2001/XMLSchema#boolean",
            "property_kind": "datatype",
        },
    )
    orders_subtasks: list[bool] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#ordersSubtasks",
            "ontology_range": "http://www.w3.org/2001/XMLSchema#boolean",
            "property_kind": "datatype",
        },
    )


@_dataclass(eq=False, kw_only=True)
class Whenever(ControlStructure):
    """Structural representation of Whenever."""

    __ontology_iri__: _ClassVar[str] = "http://aicor.knowledge/l2-ontology.owl#Whenever"


@_dataclass(eq=False, kw_only=True)
class WithPolicy(ControlStructure):
    """Structural representation of WithPolicy."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#WithPolicy"
    )
    has_policy_task: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#hasPolicyTask",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    has_primary_task: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#hasPrimaryTask",
            "ontology_range": "",
            "property_kind": "object",
        },
    )


@_dataclass(eq=False, kw_only=True)
class BeginTask(PlanInterpretationEvent):
    """Structural representation of BeginTask."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#BeginTask"
    )
    marks_transition_to: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#marksTransitionTo",
            "ontology_range": "",
            "property_kind": "object",
        },
    )


@_dataclass(eq=False, kw_only=True)
class EndTask(PlanInterpretationEvent):
    """Structural representation of EndTask."""

    __ontology_iri__: _ClassVar[str] = "http://aicor.knowledge/l2-ontology.owl#EndTask"
    marks_transition_to: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#marksTransitionTo",
            "ontology_range": "",
            "property_kind": "object",
        },
    )


@_dataclass(eq=False, kw_only=True)
class ResolveEvent(PlanInterpretationEvent):
    """Structural representation of ResolveEvent."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#ResolveEvent"
    )
    queries_engine: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#queriesEngine",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    resolves_designator: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#resolvesDesignator",
            "ontology_range": "",
            "property_kind": "object",
        },
    )


@_dataclass(eq=False, kw_only=True)
class CrackedContainerState(ClosedContainerState):
    """Structural representation of CrackedContainerState."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#CrackedContainerState"
    )
    has_crack_orientation: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#hasCrackOrientation",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    initiated_by_event: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#initiatedByEvent",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    resolves_to_state: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#resolvesToState",
            "ontology_range": "",
            "property_kind": "object",
        },
    )


@_dataclass(eq=False, kw_only=True)
class IntactContainerState(ClosedContainerState):
    """Structural representation of IntactContainerState."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#IntactContainerState"
    )


@_dataclass(eq=False, kw_only=True)
class Active(NonterminalStatus):
    """Structural representation of Active."""

    __ontology_iri__: _ClassVar[str] = "http://aicor.knowledge/l2-ontology.owl#Active"


@_dataclass(eq=False, kw_only=True)
class Created(NonterminalStatus):
    """Structural representation of Created."""

    __ontology_iri__: _ClassVar[str] = "http://aicor.knowledge/l2-ontology.owl#Created"


@_dataclass(eq=False, kw_only=True)
class Enabled(NonterminalStatus):
    """Structural representation of Enabled."""

    __ontology_iri__: _ClassVar[str] = "http://aicor.knowledge/l2-ontology.owl#Enabled"


@_dataclass(eq=False, kw_only=True)
class Evaporated(TerminalStatus):
    """Structural representation of Evaporated."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#Evaporated"
    )


@_dataclass(eq=False, kw_only=True)
class EvaporatedStatus(TerminalStatus):
    """Structural representation of EvaporatedStatus."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#EvaporatedStatus"
    )


@_dataclass(eq=False, kw_only=True)
class Failed(TerminalStatus):
    """Structural representation of Failed."""

    __ontology_iri__: _ClassVar[str] = "http://aicor.knowledge/l2-ontology.owl#Failed"
    described_by: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#describedBy",
            "ontology_range": "",
            "property_kind": "object",
        },
    )


@_dataclass(eq=False, kw_only=True)
class FailedStatus(TerminalStatus):
    """Structural representation of FailedStatus."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#FailedStatus"
    )


@_dataclass(eq=False, kw_only=True)
class Finished(TerminalStatus):
    """Structural representation of Finished."""

    __ontology_iri__: _ClassVar[str] = "http://aicor.knowledge/l2-ontology.owl#Finished"


@_dataclass(eq=False, kw_only=True)
class SucceededStatus(TerminalStatus):
    """Structural representation of SucceededStatus."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#SucceededStatus"
    )


@_dataclass(eq=False, kw_only=True)
class ConfigurationChange(ManipulationActionCategory):
    """Structural representation of ConfigurationChange."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#ConfigurationChange"
    )
    effect_dimension: list[str] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#effectDimension",
            "ontology_range": "http://www.w3.org/2001/XMLSchema#string",
            "property_kind": "datatype",
        },
    )


@_dataclass(eq=False, kw_only=True)
class ConnectingAction(ManipulationActionCategory):
    """Structural representation of ConnectingAction."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#ConnectingAction"
    )
    effect_dimension: list[str] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#effectDimension",
            "ontology_range": "http://www.w3.org/2001/XMLSchema#string",
            "property_kind": "datatype",
        },
    )


@_dataclass(eq=False, kw_only=True)
class InteractionAction(ManipulationActionCategory):
    """Structural representation of InteractionAction."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#InteractionAction"
    )
    effect_dimension: list[str] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#effectDimension",
            "ontology_range": "http://www.w3.org/2001/XMLSchema#string",
            "property_kind": "datatype",
        },
    )


@_dataclass(eq=False, kw_only=True)
class MaintenanceAction(ManipulationActionCategory):
    """Structural representation of MaintenanceAction."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#MaintenanceAction"
    )
    effect_dimension: list[str] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#effectDimension",
            "ontology_range": "http://www.w3.org/2001/XMLSchema#string",
            "property_kind": "datatype",
        },
    )


@_dataclass(eq=False, kw_only=True)
class ObjectDisplacement(ManipulationActionCategory):
    """Structural representation of ObjectDisplacement."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#ObjectDisplacement"
    )
    effect_dimension: list[str] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#effectDimension",
            "ontology_range": "http://www.w3.org/2001/XMLSchema#string",
            "property_kind": "datatype",
        },
    )


@_dataclass(eq=False, kw_only=True)
class PrimitiveAction(ManipulationActionCategory):
    """Structural representation of PrimitiveAction."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#PrimitiveAction"
    )


@_dataclass(eq=False, kw_only=True)
class PurposeConstrainedCategory(ManipulationActionCategory):
    """Structural representation of PurposeConstrainedCategory."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#PurposeConstrainedCategory"
    )
    has_forbidden_regardless: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#hasForbiddenRegardless",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    success_constrained_by: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#successConstrainedBy",
            "ontology_range": "",
            "property_kind": "object",
        },
    )


@_dataclass(eq=False, kw_only=True)
class SubstanceManipulation(ManipulationActionCategory):
    """Structural representation of SubstanceManipulation."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#SubstanceManipulation"
    )
    effect_dimension: list[str] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#effectDimension",
            "ontology_range": "http://www.w3.org/2001/XMLSchema#string",
            "property_kind": "datatype",
        },
    )


@_dataclass(eq=False, kw_only=True)
class SurfaceTreatment(ManipulationActionCategory):
    """Structural representation of SurfaceTreatment."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#SurfaceTreatment"
    )
    effect_dimension: list[str] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#effectDimension",
            "ontology_range": "http://www.w3.org/2001/XMLSchema#string",
            "property_kind": "datatype",
        },
    )


@_dataclass(eq=False, kw_only=True)
class CuttingTemplate(ForceDynamicTemplate):
    """Structural representation of CuttingTemplate."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#CuttingTemplate"
    )
    has_phase: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#hasPhase",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    has_terminal_state: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#hasTerminalState",
            "ontology_range": "",
            "property_kind": "object",
        },
    )


@_dataclass(eq=False, kw_only=True)
class PouringTemplate(ForceDynamicTemplate):
    """Structural representation of PouringTemplate."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#PouringTemplate"
    )
    has_phase: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#hasPhase",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    has_terminal_state: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#hasTerminalState",
            "ontology_range": "",
            "property_kind": "object",
        },
    )


@_dataclass(eq=False, kw_only=True)
class CramArchitecture(StructuredReactivePlan):
    """Structural representation of CramArchitecture."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#CRAMArchitecture"
    )
    adopts_principle: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#adoptsPrinciple",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    has_belief_state: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#hasBeliefState",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    has_capability: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#hasCapability",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    has_component: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#hasComponent",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    has_element: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#hasElement",
            "ontology_range": "",
            "property_kind": "object",
        },
    )
    specifies: list[_KrroodSymbol] = _field(
        default_factory=list,
        init=True,
        metadata={
            "ontology_iri": "http://aicor.knowledge/l2-ontology.owl#specifies",
            "ontology_range": "",
            "property_kind": "object",
        },
    )


@_dataclass(eq=False, kw_only=True)
class HeldIntactEgg(IntactContainerState):
    """Structural representation of HeldIntactEgg."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#HeldIntactEgg"
    )


@_dataclass(eq=False, kw_only=True)
class Transporting(ObjectDisplacement):
    """Structural representation of Transporting."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#Transporting"
    )


@_dataclass(eq=False, kw_only=True)
class Grasping(PrimitiveAction):
    """Structural representation of Grasping."""

    __ontology_iri__: _ClassVar[str] = "http://aicor.knowledge/l2-ontology.owl#Grasping"


@_dataclass(eq=False, kw_only=True)
class Lifting(PrimitiveAction):
    """Structural representation of Lifting."""

    __ontology_iri__: _ClassVar[str] = "http://aicor.knowledge/l2-ontology.owl#Lifting"


@_dataclass(eq=False, kw_only=True)
class Reaching(PrimitiveAction):
    """Structural representation of Reaching."""

    __ontology_iri__: _ClassVar[str] = "http://aicor.knowledge/l2-ontology.owl#Reaching"


@_dataclass(eq=False, kw_only=True)
class Stroking(PrimitiveAction):
    """Structural representation of Stroking."""

    __ontology_iri__: _ClassVar[str] = "http://aicor.knowledge/l2-ontology.owl#Stroking"


@_dataclass(eq=False, kw_only=True)
class Tilting(PrimitiveAction):
    """Structural representation of Tilting."""

    __ontology_iri__: _ClassVar[str] = "http://aicor.knowledge/l2-ontology.owl#Tilting"


@_dataclass(eq=False, kw_only=True)
class Dividing(SubstanceManipulation):
    """Structural representation of Dividing."""

    __ontology_iri__: _ClassVar[str] = "http://aicor.knowledge/l2-ontology.owl#Dividing"


@_dataclass(eq=False, kw_only=True)
class Transferring(SubstanceManipulation):
    """Structural representation of Transferring."""

    __ontology_iri__: _ClassVar[str] = (
        "http://aicor.knowledge/l2-ontology.owl#Transferring"
    )


@_dataclass(eq=False, kw_only=True)
class Wiping(SurfaceTreatment):
    """Structural representation of Wiping."""

    __ontology_iri__: _ClassVar[str] = "http://aicor.knowledge/l2-ontology.owl#Wiping"


@_dataclass(eq=False, kw_only=True)
class Cutting(Dividing):
    """Structural representation of Cutting."""

    __ontology_iri__: _ClassVar[str] = "http://aicor.knowledge/l2-ontology.owl#Cutting"


@_dataclass(eq=False, kw_only=True)
class Pouring(Transferring):
    """Structural representation of Pouring."""

    __ontology_iri__: _ClassVar[str] = "http://aicor.knowledge/l2-ontology.owl#Pouring"


@_dataclass(eq=False, kw_only=True)
class Slicing(Cutting):
    """Structural representation of Slicing."""

    __ontology_iri__: _ClassVar[str] = "http://aicor.knowledge/l2-ontology.owl#Slicing"


CLASS_BY_IRI = {
    "http://aicor.knowledge/l2-ontology.owl#Abstain": Abstain,
    "http://aicor.knowledge/l2-ontology.owl#Abstention": Abstention,
    "http://aicor.knowledge/l2-ontology.owl#Entity": Entity,
    "http://aicor.knowledge/l2-ontology.owl#AccountabilityQuestion": AccountabilityQuestion,
    "http://aicor.knowledge/l2-ontology.owl#AccountableDimension": AccountableDimension,
    "http://aicor.knowledge/l2-ontology.owl#FailureReason": FailureReason,
    "http://aicor.knowledge/l2-ontology.owl#Action": Action,
    "http://aicor.knowledge/l2-ontology.owl#ActionParameterSpace": ActionParameterSpace,
    "http://aicor.knowledge/l2-ontology.owl#DesignatorTarget": DesignatorTarget,
    "http://aicor.knowledge/l2-ontology.owl#ActionRelatedForceCorridor": ActionRelatedForceCorridor,
    "http://aicor.knowledge/l2-ontology.owl#ActionRelatedGrasp": ActionRelatedGrasp,
    "http://aicor.knowledge/l2-ontology.owl#ActionRelatedMotion": ActionRelatedMotion,
    "http://aicor.knowledge/l2-ontology.owl#ActionRelatedPlace": ActionRelatedPlace,
    "http://aicor.knowledge/l2-ontology.owl#ActionRelatedPour": ActionRelatedPour,
    "http://aicor.knowledge/l2-ontology.owl#ActionToken": ActionToken,
    "http://aicor.knowledge/l2-ontology.owl#CommitmentTarget": CommitmentTarget,
    "http://aicor.knowledge/l2-ontology.owl#ActivityEpisode": ActivityEpisode,
    "http://aicor.knowledge/l2-ontology.owl#ActuationNuisance": ActuationNuisance,
    "http://aicor.knowledge/l2-ontology.owl#ActuatorInput": ActuatorInput,
    "http://aicor.knowledge/l2-ontology.owl#AddSupport": AddSupport,
    "http://aicor.knowledge/l2-ontology.owl#AffectsExecution": AffectsExecution,
    "http://aicor.knowledge/l2-ontology.owl#AffordanceEstimation": AffordanceEstimation,
    "http://aicor.knowledge/l2-ontology.owl#ParticipantRole": ParticipantRole,
    "http://aicor.knowledge/l2-ontology.owl#Appearance": Appearance,
    "http://aicor.knowledge/l2-ontology.owl#ArrangementRole": ArrangementRole,
    "http://aicor.knowledge/l2-ontology.owl#AssessmentAndClaim": AssessmentAndClaim,
    "http://aicor.knowledge/l2-ontology.owl#BodyDynamics": BodyDynamics,
    "http://aicor.knowledge/l2-ontology.owl#dul_Event": DulEvent,
    "http://aicor.knowledge/l2-ontology.owl#BodyState": BodyState,
    "http://aicor.knowledge/l2-ontology.owl#BodyStateSpace": BodyStateSpace,
    "http://aicor.knowledge/l2-ontology.owl#dul_Process": DulProcess,
    "http://aicor.knowledge/l2-ontology.owl#soma_Action": SomaAction,
    "http://aicor.knowledge/l2-ontology.owl#CandidateSet": CandidateSet,
    "http://aicor.knowledge/l2-ontology.owl#dul_Entity": DulEntity,
    "http://aicor.knowledge/l2-ontology.owl#CausalEffectiveness": CausalEffectiveness,
    "http://aicor.knowledge/l2-ontology.owl#Check": Check,
    "http://aicor.knowledge/l2-ontology.owl#Clarify": Clarify,
    "http://aicor.knowledge/l2-ontology.owl#CodeReplace": CodeReplace,
    "http://aicor.knowledge/l2-ontology.owl#Commit": Commit,
    "http://aicor.knowledge/l2-ontology.owl#CommitmentCriterion": CommitmentCriterion,
    "http://aicor.knowledge/l2-ontology.owl#CommitmentRelevance": CommitmentRelevance,
    "http://aicor.knowledge/l2-ontology.owl#CommitmentRelevantStructure": CommitmentRelevantStructure,
    "http://aicor.knowledge/l2-ontology.owl#dul_Quality": DulQuality,
    "http://aicor.knowledge/l2-ontology.owl#ConceptFamilyEnum": ConceptFamilyEnum,
    "http://aicor.knowledge/l2-ontology.owl#Configuration": Configuration,
    "http://aicor.knowledge/l2-ontology.owl#CouplingType": CouplingType,
    "http://aicor.knowledge/l2-ontology.owl#ConsumableScope": ConsumableScope,
    "http://aicor.knowledge/l2-ontology.owl#Continuous": Continuous,
    "http://aicor.knowledge/l2-ontology.owl#ContinuousState": ContinuousState,
    "http://aicor.knowledge/l2-ontology.owl#Continuum": Continuum,
    "http://aicor.knowledge/l2-ontology.owl#ContributesTraceEvidence": ContributesTraceEvidence,
    "http://aicor.knowledge/l2-ontology.owl#ControlCandidate": ControlCandidate,
    "http://aicor.knowledge/l2-ontology.owl#ControlCounterpart": ControlCounterpart,
    "http://aicor.knowledge/l2-ontology.owl#ControlCycle": ControlCycle,
    "http://aicor.knowledge/l2-ontology.owl#ControlSequence": ControlSequence,
    "http://aicor.knowledge/l2-ontology.owl#CouplingRelation": CouplingRelation,
    "http://aicor.knowledge/l2-ontology.owl#CrackOrientation": CrackOrientation,
    "http://aicor.knowledge/l2-ontology.owl#CriterionRelative": CriterionRelative,
    "http://aicor.knowledge/l2-ontology.owl#CriticHook": CriticHook,
    "http://aicor.knowledge/l2-ontology.owl#DesignatorResolution": DesignatorResolution,
    "http://aicor.knowledge/l2-ontology.owl#DegreeOfFreedom": DegreeOfFreedom,
    "http://aicor.knowledge/l2-ontology.owl#Deictic": Deictic,
    "http://aicor.knowledge/l2-ontology.owl#Descriptive": Descriptive,
    "http://aicor.knowledge/l2-ontology.owl#DesignatorReferent": DesignatorReferent,
    "http://aicor.knowledge/l2-ontology.owl#Destination": Destination,
    "http://aicor.knowledge/l2-ontology.owl#DiagnosisCounterpart": DiagnosisCounterpart,
    "http://aicor.knowledge/l2-ontology.owl#DiagnosticForce": DiagnosticForce,
    "http://aicor.knowledge/l2-ontology.owl#DistributionalPlausibility": DistributionalPlausibility,
    "http://aicor.knowledge/l2-ontology.owl#DistributionalPlausible": DistributionalPlausible,
    "http://aicor.knowledge/l2-ontology.owl#DisturbanceNuisance": DisturbanceNuisance,
    "http://aicor.knowledge/l2-ontology.owl#DownCrack": DownCrack,
    "http://aicor.knowledge/l2-ontology.owl#Drop": Drop,
    "http://aicor.knowledge/l2-ontology.owl#Dynamic": Dynamic,
    "http://aicor.knowledge/l2-ontology.owl#Effect": Effect,
    "http://aicor.knowledge/l2-ontology.owl#Embodiment": Embodiment,
    "http://aicor.knowledge/l2-ontology.owl#Environment": Environment,
    "http://aicor.knowledge/l2-ontology.owl#EnvironmentModel": EnvironmentModel,
    "http://aicor.knowledge/l2-ontology.owl#EpisodeAndTrace": EpisodeAndTrace,
    "http://aicor.knowledge/l2-ontology.owl#EverydayTerm": EverydayTerm,
    "http://aicor.knowledge/l2-ontology.owl#EvidenceCounterpart": EvidenceCounterpart,
    "http://aicor.knowledge/l2-ontology.owl#EvidenceLogging": EvidenceLogging,
    "http://aicor.knowledge/l2-ontology.owl#EvidenceRequirement": EvidenceRequirement,
    "http://aicor.knowledge/l2-ontology.owl#EvidentialSufficiency": EvidentialSufficiency,
    "http://aicor.knowledge/l2-ontology.owl#dul_Situation": DulSituation,
    "http://aicor.knowledge/l2-ontology.owl#Executive": Executive,
    "http://aicor.knowledge/l2-ontology.owl#ExecutiveArchitecture": ExecutiveArchitecture,
    "http://aicor.knowledge/l2-ontology.owl#ExecutiveCounterpart": ExecutiveCounterpart,
    "http://aicor.knowledge/l2-ontology.owl#ExecutiveEstimate": ExecutiveEstimate,
    "http://aicor.knowledge/l2-ontology.owl#ExecutiveResponsibility": ExecutiveResponsibility,
    "http://aicor.knowledge/l2-ontology.owl#Expectation": Expectation,
    "http://aicor.knowledge/l2-ontology.owl#FailureConcept": FailureConcept,
    "http://aicor.knowledge/l2-ontology.owl#FeasibilityAssessment": FeasibilityAssessment,
    "http://aicor.knowledge/l2-ontology.owl#FeasibilityEnvelope": FeasibilityEnvelope,
    "http://aicor.knowledge/l2-ontology.owl#FeedbackInterface": FeedbackInterface,
    "http://aicor.knowledge/l2-ontology.owl#Finite": Finite,
    "http://aicor.knowledge/l2-ontology.owl#Fixed": Fixed,
    "http://aicor.knowledge/l2-ontology.owl#ForceInteraction": ForceInteraction,
    "http://aicor.knowledge/l2-ontology.owl#ForceProfile": ForceProfile,
    "http://aicor.knowledge/l2-ontology.owl#Generate": Generate,
    "http://aicor.knowledge/l2-ontology.owl#Generative": Generative,
    "http://aicor.knowledge/l2-ontology.owl#Geometry": Geometry,
    "http://aicor.knowledge/l2-ontology.owl#GraspRef": GraspRef,
    "http://aicor.knowledge/l2-ontology.owl#Ground": Ground,
    "http://aicor.knowledge/l2-ontology.owl#GroundedBy": GroundedBy,
    "http://aicor.knowledge/l2-ontology.owl#GroundedSchema": GroundedSchema,
    "http://aicor.knowledge/l2-ontology.owl#GroundingField": GroundingField,
    "http://aicor.knowledge/l2-ontology.owl#HelpRequest": HelpRequest,
    "http://aicor.knowledge/l2-ontology.owl#HumanNuisance": HumanNuisance,
    "http://aicor.knowledge/l2-ontology.owl#HumanReadableAndMachineInterpretable": HumanReadableAndMachineInterpretable,
    "http://aicor.knowledge/l2-ontology.owl#Individual": Individual,
    "http://aicor.knowledge/l2-ontology.owl#InertiaMatrix": InertiaMatrix,
    "http://aicor.knowledge/l2-ontology.owl#InertialSpec": InertialSpec,
    "http://aicor.knowledge/l2-ontology.owl#InitNuisance": InitNuisance,
    "http://aicor.knowledge/l2-ontology.owl#InstanceOfLink": InstanceOfLink,
    "http://aicor.knowledge/l2-ontology.owl#InstantaneousHorizon": InstantaneousHorizon,
    "http://aicor.knowledge/l2-ontology.owl#InstantiateMonitor": InstantiateMonitor,
    "http://aicor.knowledge/l2-ontology.owl#Jam": Jam,
    "http://aicor.knowledge/l2-ontology.owl#JointElement": JointElement,
    "http://aicor.knowledge/l2-ontology.owl#JointType": JointType,
    "http://aicor.knowledge/l2-ontology.owl#KRRHypothesis": KrrHypothesis,
    "http://aicor.knowledge/l2-ontology.owl#Latent": Latent,
    "http://aicor.knowledge/l2-ontology.owl#Law": Law,
    "http://aicor.knowledge/l2-ontology.owl#LayoutNuisance": LayoutNuisance,
    "http://aicor.knowledge/l2-ontology.owl#LifecycleEvent": LifecycleEvent,
    "http://aicor.knowledge/l2-ontology.owl#LocationRef": LocationRef,
    "http://aicor.knowledge/l2-ontology.owl#Lock": Lock,
    "http://aicor.knowledge/l2-ontology.owl#MJCFFormat": MjcfFormat,
    "http://aicor.knowledge/l2-ontology.owl#ManipulableInstance": ManipulableInstance,
    "http://aicor.knowledge/l2-ontology.owl#ManipulationAspect": ManipulationAspect,
    "http://aicor.knowledge/l2-ontology.owl#dul_Agent": DulAgent,
    "http://aicor.knowledge/l2-ontology.owl#MasteryCriterion": MasteryCriterion,
    "http://aicor.knowledge/l2-ontology.owl#MaterialProperty": MaterialProperty,
    "http://aicor.knowledge/l2-ontology.owl#Metareasoning": Metareasoning,
    "http://aicor.knowledge/l2-ontology.owl#MisdirectedForce": MisdirectedForce,
    "http://aicor.knowledge/l2-ontology.owl#Modality": Modality,
    "http://aicor.knowledge/l2-ontology.owl#MonitorAttachment": MonitorAttachment,
    "http://aicor.knowledge/l2-ontology.owl#MonitorCounterpart": MonitorCounterpart,
    "http://aicor.knowledge/l2-ontology.owl#Monitorability": Monitorability,
    "http://aicor.knowledge/l2-ontology.owl#Motion": Motion,
    "http://aicor.knowledge/l2-ontology.owl#MotionRef": MotionRef,
    "http://aicor.knowledge/l2-ontology.owl#None": None_,
    "http://aicor.knowledge/l2-ontology.owl#NuisanceComponent": NuisanceComponent,
    "http://aicor.knowledge/l2-ontology.owl#ObjectNuisance": ObjectNuisance,
    "http://aicor.knowledge/l2-ontology.owl#ObjectRef": ObjectRef,
    "http://aicor.knowledge/l2-ontology.owl#ObjectScope": ObjectScope,
    "http://aicor.knowledge/l2-ontology.owl#ObjectState": ObjectState,
    "http://aicor.knowledge/l2-ontology.owl#ObservationFunctions": ObservationFunctions,
    "http://aicor.knowledge/l2-ontology.owl#ObservationRef": ObservationRef,
    "http://aicor.knowledge/l2-ontology.owl#ObservationRequest": ObservationRequest,
    "http://aicor.knowledge/l2-ontology.owl#ObservationToPerceptService": ObservationToPerceptService,
    "http://aicor.knowledge/l2-ontology.owl#OntologicalConcept": OntologicalConcept,
    "http://aicor.knowledge/l2-ontology.owl#OperationalizationCriterion": OperationalizationCriterion,
    "http://aicor.knowledge/l2-ontology.owl#Outcome": Outcome,
    "http://aicor.knowledge/l2-ontology.owl#OutcomeDistribution": OutcomeDistribution,
    "http://aicor.knowledge/l2-ontology.owl#Parameter": Parameter,
    "http://aicor.knowledge/l2-ontology.owl#ParameterInterval": ParameterInterval,
    "http://aicor.knowledge/l2-ontology.owl#Percept": Percept,
    "http://aicor.knowledge/l2-ontology.owl#Perception": Perception,
    "http://aicor.knowledge/l2-ontology.owl#Performance": Performance,
    "http://aicor.knowledge/l2-ontology.owl#PhysicalCausal": PhysicalCausal,
    "http://aicor.knowledge/l2-ontology.owl#PhysicalContact": PhysicalContact,
    "http://aicor.knowledge/l2-ontology.owl#PhysicalDeviation": PhysicalDeviation,
    "http://aicor.knowledge/l2-ontology.owl#PhysicalNuisance": PhysicalNuisance,
    "http://aicor.knowledge/l2-ontology.owl#Physics": Physics,
    "http://aicor.knowledge/l2-ontology.owl#PhysicsModel": PhysicsModel,
    "http://aicor.knowledge/l2-ontology.owl#Prediction": Prediction,
    "http://aicor.knowledge/l2-ontology.owl#PredictiveForce": PredictiveForce,
    "http://aicor.knowledge/l2-ontology.owl#Preference": Preference,
    "http://aicor.knowledge/l2-ontology.owl#PreferenceOrder": PreferenceOrder,
    "http://aicor.knowledge/l2-ontology.owl#PrimaryActivity": PrimaryActivity,
    "http://aicor.knowledge/l2-ontology.owl#Prismatic": Prismatic,
    "http://aicor.knowledge/l2-ontology.owl#Probability": Probability,
    "http://aicor.knowledge/l2-ontology.owl#ProcessEscape": ProcessEscape,
    "http://aicor.knowledge/l2-ontology.owl#ProcessModule": ProcessModule,
    "http://aicor.knowledge/l2-ontology.owl#ProcessOccurrence": ProcessOccurrence,
    "http://aicor.knowledge/l2-ontology.owl#ProcessStall": ProcessStall,
    "http://aicor.knowledge/l2-ontology.owl#ProcessingTrace": ProcessingTrace,
    "http://aicor.knowledge/l2-ontology.owl#ProcessingVerdict": ProcessingVerdict,
    "http://aicor.knowledge/l2-ontology.owl#Progress": Progress,
    "http://aicor.knowledge/l2-ontology.owl#PrologSemantics": PrologSemantics,
    "http://aicor.knowledge/l2-ontology.owl#Proposal": Proposal,
    "http://aicor.knowledge/l2-ontology.owl#ProposalCommitment": ProposalCommitment,
    "http://aicor.knowledge/l2-ontology.owl#ProposalMechanism": ProposalMechanism,
    "http://aicor.knowledge/l2-ontology.owl#ProposalNotCommitment": ProposalNotCommitment,
    "http://aicor.knowledge/l2-ontology.owl#QualityFunctional": QualityFunctional,
    "http://aicor.knowledge/l2-ontology.owl#dul_Action": DulAction,
    "http://aicor.knowledge/l2-ontology.owl#Realization": Realization,
    "http://aicor.knowledge/l2-ontology.owl#RealizationForm": RealizationForm,
    "http://aicor.knowledge/l2-ontology.owl#Reasoner": Reasoner,
    "http://aicor.knowledge/l2-ontology.owl#Record": Record,
    "http://aicor.knowledge/l2-ontology.owl#Recoverability": Recoverability,
    "http://aicor.knowledge/l2-ontology.owl#ReferentDistribution": ReferentDistribution,
    "http://aicor.knowledge/l2-ontology.owl#dul_Relation": DulRelation,
    "http://aicor.knowledge/l2-ontology.owl#RelationFamily": RelationFamily,
    "http://aicor.knowledge/l2-ontology.owl#Relational": Relational,
    "http://aicor.knowledge/l2-ontology.owl#RepairCandidateRef": RepairCandidateRef,
    "http://aicor.knowledge/l2-ontology.owl#RepairOption": RepairOption,
    "http://aicor.knowledge/l2-ontology.owl#Reparameterize": Reparameterize,
    "http://aicor.knowledge/l2-ontology.owl#Represents": Represents,
    "http://aicor.knowledge/l2-ontology.owl#RequestHelp": RequestHelp,
    "http://aicor.knowledge/l2-ontology.owl#ResolutionMode": ResolutionMode,
    "http://aicor.knowledge/l2-ontology.owl#ResponsePolicy": ResponsePolicy,
    "http://aicor.knowledge/l2-ontology.owl#ResponsibleClaimScope": ResponsibleClaimScope,
    "http://aicor.knowledge/l2-ontology.owl#ResponsibleNonAccomplishment": ResponsibleNonAccomplishment,
    "http://aicor.knowledge/l2-ontology.owl#RevisionCounterpart": RevisionCounterpart,
    "http://aicor.knowledge/l2-ontology.owl#Revolute": Revolute,
    "http://aicor.knowledge/l2-ontology.owl#RiskEstimation": RiskEstimation,
    "http://aicor.knowledge/l2-ontology.owl#Robustness": Robustness,
    "http://aicor.knowledge/l2-ontology.owl#RoleAssignmentLayer": RoleAssignmentLayer,
    "http://aicor.knowledge/l2-ontology.owl#SDFFormat": SdfFormat,
    "http://aicor.knowledge/l2-ontology.owl#SDFWorldFormat": SdfWorldFormat,
    "http://aicor.knowledge/l2-ontology.owl#SDTNode": SdtNode,
    "http://aicor.knowledge/l2-ontology.owl#SafetyAssessment": SafetyAssessment,
    "http://aicor.knowledge/l2-ontology.owl#SafetyCheck": SafetyCheck,
    "http://aicor.knowledge/l2-ontology.owl#Scope": Scope,
    "http://aicor.knowledge/l2-ontology.owl#Semantic": Semantic,
    "http://aicor.knowledge/l2-ontology.owl#SensingNuisance": SensingNuisance,
    "http://aicor.knowledge/l2-ontology.owl#SensorOutput": SensorOutput,
    "http://aicor.knowledge/l2-ontology.owl#SkillInvocation": SkillInvocation,
    "http://aicor.knowledge/l2-ontology.owl#Slip": Slip,
    "http://aicor.knowledge/l2-ontology.owl#SomaOntology": SomaOntology,
    "http://aicor.knowledge/l2-ontology.owl#Source": Source,
    "http://aicor.knowledge/l2-ontology.owl#SourceAndFamily": SourceAndFamily,
    "http://aicor.knowledge/l2-ontology.owl#SparqlQueryAnsweringModule": SparqlQueryAnsweringModule,
    "http://aicor.knowledge/l2-ontology.owl#SpatialPose": SpatialPose,
    "http://aicor.knowledge/l2-ontology.owl#SpecificationField": SpecificationField,
    "http://aicor.knowledge/l2-ontology.owl#Spherical": Spherical,
    "http://aicor.knowledge/l2-ontology.owl#Static": Static,
    "http://aicor.knowledge/l2-ontology.owl#Structural": Structural,
    "http://aicor.knowledge/l2-ontology.owl#Support": Support,
    "http://aicor.knowledge/l2-ontology.owl#SupportsReasoning": SupportsReasoning,
    "http://aicor.knowledge/l2-ontology.owl#SwitchMode": SwitchMode,
    "http://aicor.knowledge/l2-ontology.owl#SymbolicDiagnosis": SymbolicDiagnosis,
    "http://aicor.knowledge/l2-ontology.owl#SymbolicQuery": SymbolicQuery,
    "http://aicor.knowledge/l2-ontology.owl#System": System,
    "http://aicor.knowledge/l2-ontology.owl#System1Output": System1Output,
    "http://aicor.knowledge/l2-ontology.owl#TaskComponent": TaskComponent,
    "http://aicor.knowledge/l2-ontology.owl#TaskGenerator": TaskGenerator,
    "http://aicor.knowledge/l2-ontology.owl#TaskManagementService": TaskManagementService,
    "http://aicor.knowledge/l2-ontology.owl#TaskNuisance": TaskNuisance,
    "http://aicor.knowledge/l2-ontology.owl#TaskRelativeClass": TaskRelativeClass,
    "http://aicor.knowledge/l2-ontology.owl#TaskSource": TaskSource,
    "http://aicor.knowledge/l2-ontology.owl#TaskType": TaskType,
    "http://aicor.knowledge/l2-ontology.owl#TerminalDisposition": TerminalDisposition,
    "http://aicor.knowledge/l2-ontology.owl#TraceRecording": TraceRecording,
    "http://aicor.knowledge/l2-ontology.owl#Trajectory": Trajectory,
    "http://aicor.knowledge/l2-ontology.owl#TrajectoryRef": TrajectoryRef,
    "http://aicor.knowledge/l2-ontology.owl#Translate": Translate,
    "http://aicor.knowledge/l2-ontology.owl#Tree": Tree,
    "http://aicor.knowledge/l2-ontology.owl#Triple": Triple,
    "http://aicor.knowledge/l2-ontology.owl#Type": Type_,
    "http://aicor.knowledge/l2-ontology.owl#URDFFormat": UrdfFormat,
    "http://aicor.knowledge/l2-ontology.owl#USD": Usd,
    "http://aicor.knowledge/l2-ontology.owl#USDFormat": UsdFormat,
    "http://aicor.knowledge/l2-ontology.owl#Uncertainty": Uncertainty,
    "http://aicor.knowledge/l2-ontology.owl#Unique": Unique,
    "http://aicor.knowledge/l2-ontology.owl#UniqueReferent": UniqueReferent,
    "http://aicor.knowledge/l2-ontology.owl#Unresolved": Unresolved,
    "http://aicor.knowledge/l2-ontology.owl#UnstableRelease": UnstableRelease,
    "http://aicor.knowledge/l2-ontology.owl#UpCrack": UpCrack,
    "http://aicor.knowledge/l2-ontology.owl#VLAModel": VlaModel,
    "http://aicor.knowledge/l2-ontology.owl#Variable": Variable,
    "http://aicor.knowledge/l2-ontology.owl#WarrantRecord": WarrantRecord,
    "http://aicor.knowledge/l2-ontology.owl#WithdrawAndAbstain": WithdrawAndAbstain,
    "http://aicor.knowledge/l2-ontology.owl#AbstractEntity": AbstractEntity,
    "http://aicor.knowledge/l2-ontology.owl#PhysicalEntity": PhysicalEntity,
    "http://aicor.knowledge/l2-ontology.owl#AchievementMiss": AchievementMiss,
    "http://aicor.knowledge/l2-ontology.owl#MaintenanceBreach": MaintenanceBreach,
    "http://aicor.knowledge/l2-ontology.owl#PreventionViolation": PreventionViolation,
    "http://aicor.knowledge/l2-ontology.owl#ActivityCommitment": ActivityCommitment,
    "http://aicor.knowledge/l2-ontology.owl#AgentCommitment": AgentCommitment,
    "http://aicor.knowledge/l2-ontology.owl#ArrangementCommitment": ArrangementCommitment,
    "http://aicor.knowledge/l2-ontology.owl#AgentRole": AgentRole,
    "http://aicor.knowledge/l2-ontology.owl#GoalRole": GoalRole,
    "http://aicor.knowledge/l2-ontology.owl#InstrumentRole": InstrumentRole,
    "http://aicor.knowledge/l2-ontology.owl#PatientRole": PatientRole,
    "http://aicor.knowledge/l2-ontology.owl#SourceRole": SourceRole,
    "http://aicor.knowledge/l2-ontology.owl#SubstanceRole": SubstanceRole,
    "http://aicor.knowledge/l2-ontology.owl#BodyWorldTrajectory": BodyWorldTrajectory,
    "http://aicor.knowledge/l2-ontology.owl#BodyWorldTrajectorySegment": BodyWorldTrajectorySegment,
    "http://aicor.knowledge/l2-ontology.owl#TaskRelativeEffect": TaskRelativeEffect,
    "http://aicor.knowledge/l2-ontology.owl#TaskRelevantStateChange": TaskRelevantStateChange,
    "http://aicor.knowledge/l2-ontology.owl#TwoLevelFailure": TwoLevelFailure,
    "http://aicor.knowledge/l2-ontology.owl#BodyWorldPhysicalProcess": BodyWorldPhysicalProcess,
    "http://aicor.knowledge/l2-ontology.owl#CRAMOperation": CramOperation,
    "http://aicor.knowledge/l2-ontology.owl#ComputationalProcess": ComputationalProcess,
    "http://aicor.knowledge/l2-ontology.owl#HouseholdActivity": HouseholdActivity,
    "http://aicor.knowledge/l2-ontology.owl#PhysicalProcess": PhysicalProcess,
    "http://aicor.knowledge/l2-ontology.owl#CandidateAction": CandidateAction,
    "http://aicor.knowledge/l2-ontology.owl#CaseLevelEntity": CaseLevelEntity,
    "http://aicor.knowledge/l2-ontology.owl#MetaLevelEntity": MetaLevelEntity,
    "http://aicor.knowledge/l2-ontology.owl#Recordability": Recordability,
    "http://aicor.knowledge/l2-ontology.owl#ComponentCapability": ComponentCapability,
    "http://aicor.knowledge/l2-ontology.owl#DiscreteModeStructure": DiscreteModeStructure,
    "http://aicor.knowledge/l2-ontology.owl#EdgeReversibility": EdgeReversibility,
    "http://aicor.knowledge/l2-ontology.owl#ExecutiveCapability": ExecutiveCapability,
    "http://aicor.knowledge/l2-ontology.owl#HybridBodyWorldState": HybridBodyWorldState,
    "http://aicor.knowledge/l2-ontology.owl#KinematicRedundancy": KinematicRedundancy,
    "http://aicor.knowledge/l2-ontology.owl#KinematicSingularity": KinematicSingularity,
    "http://aicor.knowledge/l2-ontology.owl#Constraint": Constraint,
    "http://aicor.knowledge/l2-ontology.owl#ForceClosure": ForceClosure,
    "http://aicor.knowledge/l2-ontology.owl#Process": Process,
    "http://aicor.knowledge/l2-ontology.owl#GiveUpFailure": GiveUpFailure,
    "http://aicor.knowledge/l2-ontology.owl#ImpossibilityFailure": ImpossibilityFailure,
    "http://aicor.knowledge/l2-ontology.owl#PurposeFailure": PurposeFailure,
    "http://aicor.knowledge/l2-ontology.owl#QueryInterfaceOperation": QueryInterfaceOperation,
    "http://aicor.knowledge/l2-ontology.owl#RelationEdge": RelationEdge,
    "http://aicor.knowledge/l2-ontology.owl#ActionReferent": ActionReferent,
    "http://aicor.knowledge/l2-ontology.owl#Occasion": Occasion,
    "http://aicor.knowledge/l2-ontology.owl#Occurrence": Occurrence,
    "http://aicor.knowledge/l2-ontology.owl#Phase": Phase,
    "http://aicor.knowledge/l2-ontology.owl#dul_Concept": DulConcept,
    "http://aicor.knowledge/l2-ontology.owl#dul_Description": DulDescription,
    "http://aicor.knowledge/l2-ontology.owl#dul_InformationObject": DulInformationObject,
    "http://aicor.knowledge/l2-ontology.owl#dul_Region": DulRegion,
    "http://aicor.knowledge/l2-ontology.owl#BodyMotion": BodyMotion,
    "http://aicor.knowledge/l2-ontology.owl#Contact": Contact,
    "http://aicor.knowledge/l2-ontology.owl#Force": Force,
    "http://aicor.knowledge/l2-ontology.owl#PhysicalObject": PhysicalObject,
    "http://aicor.knowledge/l2-ontology.owl#dul_PhysicalAgent": DulPhysicalAgent,
    "http://aicor.knowledge/l2-ontology.owl#dul_PhysicalObject": DulPhysicalObject,
    "http://aicor.knowledge/l2-ontology.owl#AuthoredStateChange": AuthoredStateChange,
    "http://aicor.knowledge/l2-ontology.owl#DisturbanceStateChange": DisturbanceStateChange,
    "http://aicor.knowledge/l2-ontology.owl#BindingLevelFailure": BindingLevelFailure,
    "http://aicor.knowledge/l2-ontology.owl#SchemaLevelFailure": SchemaLevelFailure,
    "http://aicor.knowledge/l2-ontology.owl#Consolidation": Consolidation,
    "http://aicor.knowledge/l2-ontology.owl#Critic": Critic,
    "http://aicor.knowledge/l2-ontology.owl#DeferredResolution": DeferredResolution,
    "http://aicor.knowledge/l2-ontology.owl#ExecutiveInformationProcess": ExecutiveInformationProcess,
    "http://aicor.knowledge/l2-ontology.owl#InternalSimulation": InternalSimulation,
    "http://aicor.knowledge/l2-ontology.owl#LifePathSelection": LifePathSelection,
    "http://aicor.knowledge/l2-ontology.owl#Provisioning": Provisioning,
    "http://aicor.knowledge/l2-ontology.owl#Refinement": Refinement,
    "http://aicor.knowledge/l2-ontology.owl#RequestCoercion": RequestCoercion,
    "http://aicor.knowledge/l2-ontology.owl#RequestResolution": RequestResolution,
    "http://aicor.knowledge/l2-ontology.owl#StoryTelling": StoryTelling,
    "http://aicor.knowledge/l2-ontology.owl#Drinking": Drinking,
    "http://aicor.knowledge/l2-ontology.owl#Eating": Eating,
    "http://aicor.knowledge/l2-ontology.owl#Execution": Execution,
    "http://aicor.knowledge/l2-ontology.owl#AdmissibleAction": AdmissibleAction,
    "http://aicor.knowledge/l2-ontology.owl#Coupling": Coupling,
    "http://aicor.knowledge/l2-ontology.owl#InitialCondition": InitialCondition,
    "http://aicor.knowledge/l2-ontology.owl#ManipulationCase": ManipulationCase,
    "http://aicor.knowledge/l2-ontology.owl#ManipulationEpisode": ManipulationEpisode,
    "http://aicor.knowledge/l2-ontology.owl#ManipulationExecutive": ManipulationExecutive,
    "http://aicor.knowledge/l2-ontology.owl#TaskRelevantVariable": TaskRelevantVariable,
    "http://aicor.knowledge/l2-ontology.owl#AssessmentEvent": AssessmentEvent,
    "http://aicor.knowledge/l2-ontology.owl#AxiomSet": AxiomSet,
    "http://aicor.knowledge/l2-ontology.owl#CaseFamily": CaseFamily,
    "http://aicor.knowledge/l2-ontology.owl#Distribution": Distribution,
    "http://aicor.knowledge/l2-ontology.owl#EvidenceSet": EvidenceSet,
    "http://aicor.knowledge/l2-ontology.owl#VerdictComponent": VerdictComponent,
    "http://aicor.knowledge/l2-ontology.owl#Irreversible": Irreversible,
    "http://aicor.knowledge/l2-ontology.owl#Recoverable": Recoverable,
    "http://aicor.knowledge/l2-ontology.owl#ArrangementCompetence": ArrangementCompetence,
    "http://aicor.knowledge/l2-ontology.owl#CognitiveCapability": CognitiveCapability,
    "http://aicor.knowledge/l2-ontology.owl#PlacementCompetence": PlacementCompetence,
    "http://aicor.knowledge/l2-ontology.owl#RealizationCompetence": RealizationCompetence,
    "http://aicor.knowledge/l2-ontology.owl#RestorationCompetence": RestorationCompetence,
    "http://aicor.knowledge/l2-ontology.owl#StabilizationCompetence": StabilizationCompetence,
    "http://aicor.knowledge/l2-ontology.owl#BeliefOccasion": BeliefOccasion,
    "http://aicor.knowledge/l2-ontology.owl#WorldOccasion": WorldOccasion,
    "http://aicor.knowledge/l2-ontology.owl#BeliefOccurrence": BeliefOccurrence,
    "http://aicor.knowledge/l2-ontology.owl#PerformOccurrence": PerformOccurrence,
    "http://aicor.knowledge/l2-ontology.owl#WorldOccurrence": WorldOccurrence,
    "http://aicor.knowledge/l2-ontology.owl#ConceptFamily": ConceptFamily,
    "http://aicor.knowledge/l2-ontology.owl#EpistemicStatus": EpistemicStatus,
    "http://aicor.knowledge/l2-ontology.owl#FoundationModelInsertionPoint": FoundationModelInsertionPoint,
    "http://aicor.knowledge/l2-ontology.owl#ManipulationDimension": ManipulationDimension,
    "http://aicor.knowledge/l2-ontology.owl#RoleSlot": RoleSlot,
    "http://aicor.knowledge/l2-ontology.owl#AccountableImplementationStructure": AccountableImplementationStructure,
    "http://aicor.knowledge/l2-ontology.owl#ActionProposalResource": ActionProposalResource,
    "http://aicor.knowledge/l2-ontology.owl#ActionRelatedConcept": ActionRelatedConcept,
    "http://aicor.knowledge/l2-ontology.owl#ActivityUnderstanding": ActivityUnderstanding,
    "http://aicor.knowledge/l2-ontology.owl#ApplicationContext": ApplicationContext,
    "http://aicor.knowledge/l2-ontology.owl#AssessmentRegime": AssessmentRegime,
    "http://aicor.knowledge/l2-ontology.owl#AssumptionSet": AssumptionSet,
    "http://aicor.knowledge/l2-ontology.owl#BehaviorCritic": BehaviorCritic,
    "http://aicor.knowledge/l2-ontology.owl#BindingWorkspace": BindingWorkspace,
    "http://aicor.knowledge/l2-ontology.owl#BodyWorldConstraint": BodyWorldConstraint,
    "http://aicor.knowledge/l2-ontology.owl#CapabilityInteraction": CapabilityInteraction,
    "http://aicor.knowledge/l2-ontology.owl#CausalAdmissibility": CausalAdmissibility,
    "http://aicor.knowledge/l2-ontology.owl#ChannelIndependence": ChannelIndependence,
    "http://aicor.knowledge/l2-ontology.owl#ClaimWarrant": ClaimWarrant,
    "http://aicor.knowledge/l2-ontology.owl#CoReference": CoReference,
    "http://aicor.knowledge/l2-ontology.owl#CommitmentChecking": CommitmentChecking,
    "http://aicor.knowledge/l2-ontology.owl#CommitmentRecord": CommitmentRecord,
    "http://aicor.knowledge/l2-ontology.owl#CompetenceQuestion": CompetenceQuestion,
    "http://aicor.knowledge/l2-ontology.owl#CompetencyQuestion": CompetencyQuestion,
    "http://aicor.knowledge/l2-ontology.owl#ConceptualFramework": ConceptualFramework,
    "http://aicor.knowledge/l2-ontology.owl#ConcurrentPolicy": ConcurrentPolicy,
    "http://aicor.knowledge/l2-ontology.owl#ConstraintExploitationPrinciple": ConstraintExploitationPrinciple,
    "http://aicor.knowledge/l2-ontology.owl#ConstraintSet": ConstraintSet,
    "http://aicor.knowledge/l2-ontology.owl#ContactPair": ContactPair,
    "http://aicor.knowledge/l2-ontology.owl#ContextualizationStep": ContextualizationStep,
    "http://aicor.knowledge/l2-ontology.owl#ControlProgram": ControlProgram,
    "http://aicor.knowledge/l2-ontology.owl#CostFunction": CostFunction,
    "http://aicor.knowledge/l2-ontology.owl#CouplingTerm": CouplingTerm,
    "http://aicor.knowledge/l2-ontology.owl#CriticizeReviseCycle": CriticizeReviseCycle,
    "http://aicor.knowledge/l2-ontology.owl#DeclaredBoundary": DeclaredBoundary,
    "http://aicor.knowledge/l2-ontology.owl#DeliberationScheduling": DeliberationScheduling,
    "http://aicor.knowledge/l2-ontology.owl#DependencyEdge": DependencyEdge,
    "http://aicor.knowledge/l2-ontology.owl#Designator": Designator,
    "http://aicor.knowledge/l2-ontology.owl#DesignatorDrift": DesignatorDrift,
    "http://aicor.knowledge/l2-ontology.owl#DownstreamModel": DownstreamModel,
    "http://aicor.knowledge/l2-ontology.owl#DualProcessManipulationExecutive": DualProcessManipulationExecutive,
    "http://aicor.knowledge/l2-ontology.owl#DynamicModel": DynamicModel,
    "http://aicor.knowledge/l2-ontology.owl#EngineeringDiscipline": EngineeringDiscipline,
    "http://aicor.knowledge/l2-ontology.owl#Envisioning": Envisioning,
    "http://aicor.knowledge/l2-ontology.owl#EpistemicObject": EpistemicObject,
    "http://aicor.knowledge/l2-ontology.owl#EpistemicStatusTransition": EpistemicStatusTransition,
    "http://aicor.knowledge/l2-ontology.owl#ExcludedClaimType": ExcludedClaimType,
    "http://aicor.knowledge/l2-ontology.owl#ExecutiveBehaviouralSpecification": ExecutiveBehaviouralSpecification,
    "http://aicor.knowledge/l2-ontology.owl#ExecutiveComponent": ExecutiveComponent,
    "http://aicor.knowledge/l2-ontology.owl#ExecutiveMechanism": ExecutiveMechanism,
    "http://aicor.knowledge/l2-ontology.owl#ExecutiveProperty": ExecutiveProperty,
    "http://aicor.knowledge/l2-ontology.owl#ExecutiveRealization": ExecutiveRealization,
    "http://aicor.knowledge/l2-ontology.owl#ExploitableConstraint": ExploitableConstraint,
    "http://aicor.knowledge/l2-ontology.owl#FailureDescription": FailureDescription,
    "http://aicor.knowledge/l2-ontology.owl#FailureSignature": FailureSignature,
    "http://aicor.knowledge/l2-ontology.owl#ForceDynamicDependency": ForceDynamicDependency,
    "http://aicor.knowledge/l2-ontology.owl#ForceDynamicSchema": ForceDynamicSchema,
    "http://aicor.knowledge/l2-ontology.owl#ForwardKinematics": ForwardKinematics,
    "http://aicor.knowledge/l2-ontology.owl#GenerativeDependencyStructure": GenerativeDependencyStructure,
    "http://aicor.knowledge/l2-ontology.owl#GlobalStandard": GlobalStandard,
    "http://aicor.knowledge/l2-ontology.owl#Goal": Goal,
    "http://aicor.knowledge/l2-ontology.owl#GoalClause": GoalClause,
    "http://aicor.knowledge/l2-ontology.owl#GrammarOfAction": GrammarOfAction,
    "http://aicor.knowledge/l2-ontology.owl#GroundedManipulationProblem": GroundedManipulationProblem,
    "http://aicor.knowledge/l2-ontology.owl#GroundingCommitment": GroundingCommitment,
    "http://aicor.knowledge/l2-ontology.owl#ImplicitToExplicitManipulation": ImplicitToExplicitManipulation,
    "http://aicor.knowledge/l2-ontology.owl#InverseKinematics": InverseKinematics,
    "http://aicor.knowledge/l2-ontology.owl#KRRResource": KrrResource,
    "http://aicor.knowledge/l2-ontology.owl#KinematicJacobian": KinematicJacobian,
    "http://aicor.knowledge/l2-ontology.owl#KinematicModel": KinematicModel,
    "http://aicor.knowledge/l2-ontology.owl#LearningLoop": LearningLoop,
    "http://aicor.knowledge/l2-ontology.owl#ManipulationControlProgram": ManipulationControlProgram,
    "http://aicor.knowledge/l2-ontology.owl#ManipulationExecutiveOperatingSystem": ManipulationExecutiveOperatingSystem,
    "http://aicor.knowledge/l2-ontology.owl#ManipulationFrame": ManipulationFrame,
    "http://aicor.knowledge/l2-ontology.owl#ManipulationOntologyAxiomatization": ManipulationOntologyAxiomatization,
    "http://aicor.knowledge/l2-ontology.owl#ManipulationProblem": ManipulationProblem,
    "http://aicor.knowledge/l2-ontology.owl#ManipulationProblemClass": ManipulationProblemClass,
    "http://aicor.knowledge/l2-ontology.owl#ManipulationTaskSpecification": ManipulationTaskSpecification,
    "http://aicor.knowledge/l2-ontology.owl#MicrotheoryTemplate": MicrotheoryTemplate,
    "http://aicor.knowledge/l2-ontology.owl#ModeIndexedDynamics": ModeIndexedDynamics,
    "http://aicor.knowledge/l2-ontology.owl#ModelComponent": ModelComponent,
    "http://aicor.knowledge/l2-ontology.owl#ModelGuarantee": ModelGuarantee,
    "http://aicor.knowledge/l2-ontology.owl#ModelLearning": ModelLearning,
    "http://aicor.knowledge/l2-ontology.owl#Monitor": Monitor,
    "http://aicor.knowledge/l2-ontology.owl#NecessityClaim": NecessityClaim,
    "http://aicor.knowledge/l2-ontology.owl#NuisanceModel": NuisanceModel,
    "http://aicor.knowledge/l2-ontology.owl#ObjectLevelRequirement": ObjectLevelRequirement,
    "http://aicor.knowledge/l2-ontology.owl#ObjectTransformation": ObjectTransformation,
    "http://aicor.knowledge/l2-ontology.owl#ObservationChannelModel": ObservationChannelModel,
    "http://aicor.knowledge/l2-ontology.owl#ObservationFunction": ObservationFunction,
    "http://aicor.knowledge/l2-ontology.owl#ObservationModel": ObservationModel,
    "http://aicor.knowledge/l2-ontology.owl#PerceptionMethod": PerceptionMethod,
    "http://aicor.knowledge/l2-ontology.owl#Phenomenon": Phenomenon,
    "http://aicor.knowledge/l2-ontology.owl#PlanInterpreter": PlanInterpreter,
    "http://aicor.knowledge/l2-ontology.owl#PlanLearning": PlanLearning,
    "http://aicor.knowledge/l2-ontology.owl#PlanManagement": PlanManagement,
    "http://aicor.knowledge/l2-ontology.owl#PlanTransformation": PlanTransformation,
    "http://aicor.knowledge/l2-ontology.owl#ProbabilisticHybridProcessModel": ProbabilisticHybridProcessModel,
    "http://aicor.knowledge/l2-ontology.owl#ProblemClassSpecificationTemplate": ProblemClassSpecificationTemplate,
    "http://aicor.knowledge/l2-ontology.owl#ProblemDomain": ProblemDomain,
    "http://aicor.knowledge/l2-ontology.owl#ProcessControlAlignment": ProcessControlAlignment,
    "http://aicor.knowledge/l2-ontology.owl#ProcessInterpretation": ProcessInterpretation,
    "http://aicor.knowledge/l2-ontology.owl#ProcessMonitor": ProcessMonitor,
    "http://aicor.knowledge/l2-ontology.owl#ProposalToCommitmentPipeline": ProposalToCommitmentPipeline,
    "http://aicor.knowledge/l2-ontology.owl#QueryTemplate": QueryTemplate,
    "http://aicor.knowledge/l2-ontology.owl#RWMSFamily": RwmsFamily,
    "http://aicor.knowledge/l2-ontology.owl#RealizationChain": RealizationChain,
    "http://aicor.knowledge/l2-ontology.owl#RealizationInterface": RealizationInterface,
    "http://aicor.knowledge/l2-ontology.owl#RealizationLink": RealizationLink,
    "http://aicor.knowledge/l2-ontology.owl#ReasoningEngine": ReasoningEngine,
    "http://aicor.knowledge/l2-ontology.owl#ReferentialPersistence": ReferentialPersistence,
    "http://aicor.knowledge/l2-ontology.owl#ReflectiveRequirement": ReflectiveRequirement,
    "http://aicor.knowledge/l2-ontology.owl#RelationalGrammar": RelationalGrammar,
    "http://aicor.knowledge/l2-ontology.owl#RenderingInterface": RenderingInterface,
    "http://aicor.knowledge/l2-ontology.owl#Repair": Repair,
    "http://aicor.knowledge/l2-ontology.owl#RepresentationLanguage": RepresentationLanguage,
    "http://aicor.knowledge/l2-ontology.owl#RepresentationProcessCapabilityTriad": RepresentationProcessCapabilityTriad,
    "http://aicor.knowledge/l2-ontology.owl#RepresentationalObligation": RepresentationalObligation,
    "http://aicor.knowledge/l2-ontology.owl#RequestOutcome": RequestOutcome,
    "http://aicor.knowledge/l2-ontology.owl#RobotControlLaw": RobotControlLaw,
    "http://aicor.knowledge/l2-ontology.owl#RoleBinding": RoleBinding,
    "http://aicor.knowledge/l2-ontology.owl#RuntimePlanning": RuntimePlanning,
    "http://aicor.knowledge/l2-ontology.owl#RuntimeReasonablePlan": RuntimeReasonablePlan,
    "http://aicor.knowledge/l2-ontology.owl#Scenario": Scenario,
    "http://aicor.knowledge/l2-ontology.owl#SceneGraphToKnowledgeGraph": SceneGraphToKnowledgeGraph,
    "http://aicor.knowledge/l2-ontology.owl#ScienceDiscipline": ScienceDiscipline,
    "http://aicor.knowledge/l2-ontology.owl#ScientificObject": ScientificObject,
    "http://aicor.knowledge/l2-ontology.owl#ScopeCard": ScopeCard,
    "http://aicor.knowledge/l2-ontology.owl#ScopeCondition": ScopeCondition,
    "http://aicor.knowledge/l2-ontology.owl#ScopeExclusion": ScopeExclusion,
    "http://aicor.knowledge/l2-ontology.owl#ScopedClaim": ScopedClaim,
    "http://aicor.knowledge/l2-ontology.owl#SemanticTagging": SemanticTagging,
    "http://aicor.knowledge/l2-ontology.owl#SolutionMethod": SolutionMethod,
    "http://aicor.knowledge/l2-ontology.owl#SolvabilityClaim": SolvabilityClaim,
    "http://aicor.knowledge/l2-ontology.owl#SolvabilityFailureMode": SolvabilityFailureMode,
    "http://aicor.knowledge/l2-ontology.owl#SolvabilityRung": SolvabilityRung,
    "http://aicor.knowledge/l2-ontology.owl#SolvabilityStructure": SolvabilityStructure,
    "http://aicor.knowledge/l2-ontology.owl#StateOfUnderstanding": StateOfUnderstanding,
    "http://aicor.knowledge/l2-ontology.owl#StrengtheningCost": StrengtheningCost,
    "http://aicor.knowledge/l2-ontology.owl#StrengtheningModule": StrengtheningModule,
    "http://aicor.knowledge/l2-ontology.owl#SubstrateRelativeOptimality": SubstrateRelativeOptimality,
    "http://aicor.knowledge/l2-ontology.owl#SubstrateSufficiency": SubstrateSufficiency,
    "http://aicor.knowledge/l2-ontology.owl#SynergisticModelCombination": SynergisticModelCombination,
    "http://aicor.knowledge/l2-ontology.owl#TaskEnvironmentEmbodimentClass": TaskEnvironmentEmbodimentClass,
    "http://aicor.knowledge/l2-ontology.owl#TaskGrounding": TaskGrounding,
    "http://aicor.knowledge/l2-ontology.owl#TaskInstance": TaskInstance,
    "http://aicor.knowledge/l2-ontology.owl#TaskReading": TaskReading,
    "http://aicor.knowledge/l2-ontology.owl#TaskScope": TaskScope,
    "http://aicor.knowledge/l2-ontology.owl#TrainingDistribution": TrainingDistribution,
    "http://aicor.knowledge/l2-ontology.owl#Transferability": Transferability,
    "http://aicor.knowledge/l2-ontology.owl#TranslationInterface": TranslationInterface,
    "http://aicor.knowledge/l2-ontology.owl#TranslationProgram": TranslationProgram,
    "http://aicor.knowledge/l2-ontology.owl#ValidityCondition": ValidityCondition,
    "http://aicor.knowledge/l2-ontology.owl#WarrantedExecutiveCommitment": WarrantedExecutiveCommitment,
    "http://aicor.knowledge/l2-ontology.owl#WorkingMemory": WorkingMemory,
    "http://aicor.knowledge/l2-ontology.owl#WorldStateRepresentation": WorldStateRepresentation,
    "http://aicor.knowledge/l2-ontology.owl#AnswerSubstitution": AnswerSubstitution,
    "http://aicor.knowledge/l2-ontology.owl#AttributeConstraint": AttributeConstraint,
    "http://aicor.knowledge/l2-ontology.owl#Binding": Binding,
    "http://aicor.knowledge/l2-ontology.owl#ComputationalMedium": ComputationalMedium,
    "http://aicor.knowledge/l2-ontology.owl#DataGenerationEngine": DataGenerationEngine,
    "http://aicor.knowledge/l2-ontology.owl#DigitalBodyWorldSystem": DigitalBodyWorldSystem,
    "http://aicor.knowledge/l2-ontology.owl#DigitalResearchLaboratory": DigitalResearchLaboratory,
    "http://aicor.knowledge/l2-ontology.owl#EntityQuery": EntityQuery,
    "http://aicor.knowledge/l2-ontology.owl#EntityRelationQueryLanguage": EntityRelationQueryLanguage,
    "http://aicor.knowledge/l2-ontology.owl#EnvisionedTimeline": EnvisionedTimeline,
    "http://aicor.knowledge/l2-ontology.owl#EvidenceBearingTrace": EvidenceBearingTrace,
    "http://aicor.knowledge/l2-ontology.owl#ExecutableMachineryLayer": ExecutableMachineryLayer,
    "http://aicor.knowledge/l2-ontology.owl#ExecutablePrediction": ExecutablePrediction,
    "http://aicor.knowledge/l2-ontology.owl#ExecutionStateBelief": ExecutionStateBelief,
    "http://aicor.knowledge/l2-ontology.owl#ExpectedObservation": ExpectedObservation,
    "http://aicor.knowledge/l2-ontology.owl#FailureId": FailureId,
    "http://aicor.knowledge/l2-ontology.owl#FailureSeverity": FailureSeverity,
    "http://aicor.knowledge/l2-ontology.owl#FoundationModel": FoundationModel,
    "http://aicor.knowledge/l2-ontology.owl#GroundingRecord": GroundingRecord,
    "http://aicor.knowledge/l2-ontology.owl#HierarchicalSceneStructure": HierarchicalSceneStructure,
    "http://aicor.knowledge/l2-ontology.owl#IntervalTemporalLogic": IntervalTemporalLogic,
    "http://aicor.knowledge/l2-ontology.owl#KnowledgeGraph": KnowledgeGraph,
    "http://aicor.knowledge/l2-ontology.owl#KnowledgeQueryLanguage": KnowledgeQueryLanguage,
    "http://aicor.knowledge/l2-ontology.owl#LinkElement": LinkElement,
    "http://aicor.knowledge/l2-ontology.owl#Observation": Observation,
    "http://aicor.knowledge/l2-ontology.owl#PartialDescription": PartialDescription,
    "http://aicor.knowledge/l2-ontology.owl#PhysicsEngine": PhysicsEngine,
    "http://aicor.knowledge/l2-ontology.owl#ProspectiveBelief": ProspectiveBelief,
    "http://aicor.knowledge/l2-ontology.owl#QueryConnective": QueryConnective,
    "http://aicor.knowledge/l2-ontology.owl#QueryVariable": QueryVariable,
    "http://aicor.knowledge/l2-ontology.owl#Recording": Recording,
    "http://aicor.knowledge/l2-ontology.owl#ReflectiveRepresentation": ReflectiveRepresentation,
    "http://aicor.knowledge/l2-ontology.owl#ReifiedTemporalEntity": ReifiedTemporalEntity,
    "http://aicor.knowledge/l2-ontology.owl#Renderer": Renderer,
    "http://aicor.knowledge/l2-ontology.owl#ResearchInfrastructureLayer": ResearchInfrastructureLayer,
    "http://aicor.knowledge/l2-ontology.owl#ResearchSemanticDigitalTwin": ResearchSemanticDigitalTwin,
    "http://aicor.knowledge/l2-ontology.owl#RobotDescription": RobotDescription,
    "http://aicor.knowledge/l2-ontology.owl#RobotSimulator": RobotSimulator,
    "http://aicor.knowledge/l2-ontology.owl#SceneDescription": SceneDescription,
    "http://aicor.knowledge/l2-ontology.owl#SceneNode": SceneNode,
    "http://aicor.knowledge/l2-ontology.owl#SemanticDigitalTwin": SemanticDigitalTwin,
    "http://aicor.knowledge/l2-ontology.owl#SemanticReport": SemanticReport,
    "http://aicor.knowledge/l2-ontology.owl#SensorModel": SensorModel,
    "http://aicor.knowledge/l2-ontology.owl#Signal": Signal,
    "http://aicor.knowledge/l2-ontology.owl#SimulationAsset": SimulationAsset,
    "http://aicor.knowledge/l2-ontology.owl#SimulationLog": SimulationLog,
    "http://aicor.knowledge/l2-ontology.owl#StructuralCoreLayer": StructuralCoreLayer,
    "http://aicor.knowledge/l2-ontology.owl#StructuralRepresentation": StructuralRepresentation,
    "http://aicor.knowledge/l2-ontology.owl#TaskRequest": TaskRequest,
    "http://aicor.knowledge/l2-ontology.owl#TemporalPredicate": TemporalPredicate,
    "http://aicor.knowledge/l2-ontology.owl#TripleBasedQuery": TripleBasedQuery,
    "http://aicor.knowledge/l2-ontology.owl#TriplePattern": TriplePattern,
    "http://aicor.knowledge/l2-ontology.owl#TypeConstraint": TypeConstraint,
    "http://aicor.knowledge/l2-ontology.owl#USDOntology": UsdOntology,
    "http://aicor.knowledge/l2-ontology.owl#VRBContainer": VrbContainer,
    "http://aicor.knowledge/l2-ontology.owl#ValidationRecord": ValidationRecord,
    "http://aicor.knowledge/l2-ontology.owl#WhatIfQuery": WhatIfQuery,
    "http://aicor.knowledge/l2-ontology.owl#Xacro": Xacro,
    "http://aicor.knowledge/l2-ontology.owl#Chronicle": Chronicle,
    "http://aicor.knowledge/l2-ontology.owl#CompetenceDegree": CompetenceDegree,
    "http://aicor.knowledge/l2-ontology.owl#ConfigurationSpace": ConfigurationSpace,
    "http://aicor.knowledge/l2-ontology.owl#FilmThickness": FilmThickness,
    "http://aicor.knowledge/l2-ontology.owl#ForceCorridor": ForceCorridor,
    "http://aicor.knowledge/l2-ontology.owl#ParameterSample": ParameterSample,
    "http://aicor.knowledge/l2-ontology.owl#ParameterSpace": ParameterSpace,
    "http://aicor.knowledge/l2-ontology.owl#RecedingHorizon": RecedingHorizon,
    "http://aicor.knowledge/l2-ontology.owl#SpreadRadius": SpreadRadius,
    "http://aicor.knowledge/l2-ontology.owl#TaskSpace": TaskSpace,
    "http://aicor.knowledge/l2-ontology.owl#TimeInstant": TimeInstant,
    "http://aicor.knowledge/l2-ontology.owl#TimeInterval": TimeInterval,
    "http://aicor.knowledge/l2-ontology.owl#TaskAchievingBodyMotion": TaskAchievingBodyMotion,
    "http://aicor.knowledge/l2-ontology.owl#Robot": Robot,
    "http://aicor.knowledge/l2-ontology.owl#ArticulatedObject": ArticulatedObject,
    "http://aicor.knowledge/l2-ontology.owl#BodyWorldSystem": BodyWorldSystem,
    "http://aicor.knowledge/l2-ontology.owl#Joint": Joint,
    "http://aicor.knowledge/l2-ontology.owl#Link": Link,
    "http://aicor.knowledge/l2-ontology.owl#MaterialEntity": MaterialEntity,
    "http://aicor.knowledge/l2-ontology.owl#Motor": Motor,
    "http://aicor.knowledge/l2-ontology.owl#RobotWorldManipulationSystem": RobotWorldManipulationSystem,
    "http://aicor.knowledge/l2-ontology.owl#Sensor": Sensor,
    "http://aicor.knowledge/l2-ontology.owl#WorldPhysicalEntity": WorldPhysicalEntity,
    "http://aicor.knowledge/l2-ontology.owl#WorldState": WorldState,
    "http://aicor.knowledge/l2-ontology.owl#AccessibilityCritic": AccessibilityCritic,
    "http://aicor.knowledge/l2-ontology.owl#CompletenessCritic": CompletenessCritic,
    "http://aicor.knowledge/l2-ontology.owl#ConventionCritic": ConventionCritic,
    "http://aicor.knowledge/l2-ontology.owl#PhysicalCritic": PhysicalCritic,
    "http://aicor.knowledge/l2-ontology.owl#RelationalCritic": RelationalCritic,
    "http://aicor.knowledge/l2-ontology.owl#SafetyCritic": SafetyCritic,
    "http://aicor.knowledge/l2-ontology.owl#WorkflowCritic": WorkflowCritic,
    "http://aicor.knowledge/l2-ontology.owl#GenerativePerception": GenerativePerception,
    "http://aicor.knowledge/l2-ontology.owl#DesignatorRefinement": DesignatorRefinement,
    "http://aicor.knowledge/l2-ontology.owl#PlanInterpretationOp": PlanInterpretationOp,
    "http://aicor.knowledge/l2-ontology.owl#ProspectiveCase": ProspectiveCase,
    "http://aicor.knowledge/l2-ontology.owl#RealizedCase": RealizedCase,
    "http://aicor.knowledge/l2-ontology.owl#AICORReferenceManipulationExecutive": AicorReferenceManipulationExecutive,
    "http://aicor.knowledge/l2-ontology.owl#RationalExecutive": RationalExecutive,
    "http://aicor.knowledge/l2-ontology.owl#Verdict": Verdict,
    "http://aicor.knowledge/l2-ontology.owl#Accounting": Accounting,
    "http://aicor.knowledge/l2-ontology.owl#CausalPrediction": CausalPrediction,
    "http://aicor.knowledge/l2-ontology.owl#CounterfactualReasoning": CounterfactualReasoning,
    "http://aicor.knowledge/l2-ontology.owl#Diagnosis": Diagnosis,
    "http://aicor.knowledge/l2-ontology.owl#Evaluation": Evaluation,
    "http://aicor.knowledge/l2-ontology.owl#Grounding": Grounding,
    "http://aicor.knowledge/l2-ontology.owl#Prospection": Prospection,
    "http://aicor.knowledge/l2-ontology.owl#RepairAbstention": RepairAbstention,
    "http://aicor.knowledge/l2-ontology.owl#RequestConditionedPlacementCompetence": RequestConditionedPlacementCompetence,
    "http://aicor.knowledge/l2-ontology.owl#MonitorGeneratorRole": MonitorGeneratorRole,
    "http://aicor.knowledge/l2-ontology.owl#PredictorRole": PredictorRole,
    "http://aicor.knowledge/l2-ontology.owl#ProposalGeneratorRole": ProposalGeneratorRole,
    "http://aicor.knowledge/l2-ontology.owl#ForbiddenRegardlessCondition": ForbiddenRegardlessCondition,
    "http://aicor.knowledge/l2-ontology.owl#Purpose": Purpose,
    "http://aicor.knowledge/l2-ontology.owl#NarrativeUnderstanding": NarrativeUnderstanding,
    "http://aicor.knowledge/l2-ontology.owl#FederationWideAssessment": FederationWideAssessment,
    "http://aicor.knowledge/l2-ontology.owl#CapabilityAssertion": CapabilityAssertion,
    "http://aicor.knowledge/l2-ontology.owl#IncapabilityAssertion": IncapabilityAssertion,
    "http://aicor.knowledge/l2-ontology.owl#DistributionalWarrant": DistributionalWarrant,
    "http://aicor.knowledge/l2-ontology.owl#FamilyRangingWarrant": FamilyRangingWarrant,
    "http://aicor.knowledge/l2-ontology.owl#ObservedConditionsWarrant": ObservedConditionsWarrant,
    "http://aicor.knowledge/l2-ontology.owl#SingleCaseWarrant": SingleCaseWarrant,
    "http://aicor.knowledge/l2-ontology.owl#CommitmentQuestion": CommitmentQuestion,
    "http://aicor.knowledge/l2-ontology.owl#EvidenceQuestion": EvidenceQuestion,
    "http://aicor.knowledge/l2-ontology.owl#RegulationQuestion": RegulationQuestion,
    "http://aicor.knowledge/l2-ontology.owl#WarrantQuestion": WarrantQuestion,
    "http://aicor.knowledge/l2-ontology.owl#OperationalGrammar": OperationalGrammar,
    "http://aicor.knowledge/l2-ontology.owl#MentalSimulation": MentalSimulation,
    "http://aicor.knowledge/l2-ontology.owl#Projection": Projection,
    "http://aicor.knowledge/l2-ontology.owl#Definition": Definition,
    "http://aicor.knowledge/l2-ontology.owl#Experiment": Experiment,
    "http://aicor.knowledge/l2-ontology.owl#Hypothesis": Hypothesis,
    "http://aicor.knowledge/l2-ontology.owl#ScientificConcept": ScientificConcept,
    "http://aicor.knowledge/l2-ontology.owl#Theorem": Theorem,
    "http://aicor.knowledge/l2-ontology.owl#Theory": Theory,
    "http://aicor.knowledge/l2-ontology.owl#WorldModel": WorldModel,
    "http://aicor.knowledge/l2-ontology.owl#ForbiddenTransition": ForbiddenTransition,
    "http://aicor.knowledge/l2-ontology.owl#PermittedTransition": PermittedTransition,
    "http://aicor.knowledge/l2-ontology.owl#ActionExecutive": ActionExecutive,
    "http://aicor.knowledge/l2-ontology.owl#KnowledgeExecutive": KnowledgeExecutive,
    "http://aicor.knowledge/l2-ontology.owl#MetaReasoningExecutive": MetaReasoningExecutive,
    "http://aicor.knowledge/l2-ontology.owl#PerceptionExecutive": PerceptionExecutive,
    "http://aicor.knowledge/l2-ontology.owl#PlanExecutive": PlanExecutive,
    "http://aicor.knowledge/l2-ontology.owl#ActingMechanism": ActingMechanism,
    "http://aicor.knowledge/l2-ontology.owl#EvidentialMechanism": EvidentialMechanism,
    "http://aicor.knowledge/l2-ontology.owl#HonestyProperty": HonestyProperty,
    "http://aicor.knowledge/l2-ontology.owl#RegimeDependentProperty": RegimeDependentProperty,
    "http://aicor.knowledge/l2-ontology.owl#CompositeFailure": CompositeFailure,
    "http://aicor.knowledge/l2-ontology.owl#IntractableFailure": IntractableFailure,
    "http://aicor.knowledge/l2-ontology.owl#TrivialFailure": TrivialFailure,
    "http://aicor.knowledge/l2-ontology.owl#SchemaEmbedding": SchemaEmbedding,
    "http://aicor.knowledge/l2-ontology.owl#CompetenceStandard": CompetenceStandard,
    "http://aicor.knowledge/l2-ontology.owl#CompletenessStandard": CompletenessStandard,
    "http://aicor.knowledge/l2-ontology.owl#CorrectnessStandard": CorrectnessStandard,
    "http://aicor.knowledge/l2-ontology.owl#SafetyStandard": SafetyStandard,
    "http://aicor.knowledge/l2-ontology.owl#AchievementGoal": AchievementGoal,
    "http://aicor.knowledge/l2-ontology.owl#MaintenanceGoal": MaintenanceGoal,
    "http://aicor.knowledge/l2-ontology.owl#PreventionGoal": PreventionGoal,
    "http://aicor.knowledge/l2-ontology.owl#StructuredReactiveController": StructuredReactiveController,
    "http://aicor.knowledge/l2-ontology.owl#ArrangementProblem": ArrangementProblem,
    "http://aicor.knowledge/l2-ontology.owl#BWSRelativeTaskSpecification": BwsRelativeTaskSpecification,
    "http://aicor.knowledge/l2-ontology.owl#PlacementProblem": PlacementProblem,
    "http://aicor.knowledge/l2-ontology.owl#RealizationProblem": RealizationProblem,
    "http://aicor.knowledge/l2-ontology.owl#RestorationProblem": RestorationProblem,
    "http://aicor.knowledge/l2-ontology.owl#StabilizationProblem": StabilizationProblem,
    "http://aicor.knowledge/l2-ontology.owl#AxiomConditionalGuarantee": AxiomConditionalGuarantee,
    "http://aicor.knowledge/l2-ontology.owl#PACGuarantee": PacGuarantee,
    "http://aicor.knowledge/l2-ontology.owl#BindingConfirmationFunction": BindingConfirmationFunction,
    "http://aicor.knowledge/l2-ontology.owl#ProcessMonitoringFunction": ProcessMonitoringFunction,
    "http://aicor.knowledge/l2-ontology.owl#Contextualization": Contextualization,
    "http://aicor.knowledge/l2-ontology.owl#BodyMotionLink": BodyMotionLink,
    "http://aicor.knowledge/l2-ontology.owl#ForceGenerationLink": ForceGenerationLink,
    "http://aicor.knowledge/l2-ontology.owl#StateTransitionLink": StateTransitionLink,
    "http://aicor.knowledge/l2-ontology.owl#TokenEmissionLink": TokenEmissionLink,
    "http://aicor.knowledge/l2-ontology.owl#PhysicalRepair": PhysicalRepair,
    "http://aicor.knowledge/l2-ontology.owl#FailedRequest": FailedRequest,
    "http://aicor.knowledge/l2-ontology.owl#SolvedRequest": SolvedRequest,
    "http://aicor.knowledge/l2-ontology.owl#ConstraintBasedControl": ConstraintBasedControl,
    "http://aicor.knowledge/l2-ontology.owl#OptimizationBasedControl": OptimizationBasedControl,
    "http://aicor.knowledge/l2-ontology.owl#DomainScientificObject": DomainScientificObject,
    "http://aicor.knowledge/l2-ontology.owl#MethodologicalScientificObject": MethodologicalScientificObject,
    "http://aicor.knowledge/l2-ontology.owl#SubfieldScientificObject": SubfieldScientificObject,
    "http://aicor.knowledge/l2-ontology.owl#CapabilityClaim": CapabilityClaim,
    "http://aicor.knowledge/l2-ontology.owl#CompetenceClaim": CompetenceClaim,
    "http://aicor.knowledge/l2-ontology.owl#EpisodeClaim": EpisodeClaim,
    "http://aicor.knowledge/l2-ontology.owl#MasteryClaim": MasteryClaim,
    "http://aicor.knowledge/l2-ontology.owl#ControlBasisTooWeak": ControlBasisTooWeak,
    "http://aicor.knowledge/l2-ontology.owl#InadequateConduct": InadequateConduct,
    "http://aicor.knowledge/l2-ontology.owl#NoAdmissibleSolution": NoAdmissibleSolution,
    "http://aicor.knowledge/l2-ontology.owl#SignalUnavailable": SignalUnavailable,
    "http://aicor.knowledge/l2-ontology.owl#SignalUnrepresented": SignalUnrepresented,
    "http://aicor.knowledge/l2-ontology.owl#DesignRung": DesignRung,
    "http://aicor.knowledge/l2-ontology.owl#SubstrateRung": SubstrateRung,
    "http://aicor.knowledge/l2-ontology.owl#ActionSufficiency": ActionSufficiency,
    "http://aicor.knowledge/l2-ontology.owl#InformationalSufficiency": InformationalSufficiency,
    "http://aicor.knowledge/l2-ontology.owl#TemporalSufficiency": TemporalSufficiency,
    "http://aicor.knowledge/l2-ontology.owl#ComparisonFrame": ComparisonFrame,
    "http://aicor.knowledge/l2-ontology.owl#FixedCondition": FixedCondition,
    "http://aicor.knowledge/l2-ontology.owl#LaboratoryAccessTerms": LaboratoryAccessTerms,
    "http://aicor.knowledge/l2-ontology.owl#LaboratoryFederation": LaboratoryFederation,
    "http://aicor.knowledge/l2-ontology.owl#Representation": Representation,
    "http://aicor.knowledge/l2-ontology.owl#FederatedResearchLaboratory": FederatedResearchLaboratory,
    "http://aicor.knowledge/l2-ontology.owl#DerivedCaseReconstruction": DerivedCaseReconstruction,
    "http://aicor.knowledge/l2-ontology.owl#DualProcessTrace": DualProcessTrace,
    "http://aicor.knowledge/l2-ontology.owl#PortableResult": PortableResult,
    "http://aicor.knowledge/l2-ontology.owl#BindingConfirmationExpectation": BindingConfirmationExpectation,
    "http://aicor.knowledge/l2-ontology.owl#ProcessMonitoringExpectation": ProcessMonitoringExpectation,
    "http://aicor.knowledge/l2-ontology.owl#ComponentDescription": ComponentDescription,
    "http://aicor.knowledge/l2-ontology.owl#Conjunction": Conjunction,
    "http://aicor.knowledge/l2-ontology.owl#Disjunction": Disjunction,
    "http://aicor.knowledge/l2-ontology.owl#BoundVariable": BoundVariable,
    "http://aicor.knowledge/l2-ontology.owl#UnboundVariable": UnboundVariable,
    "http://aicor.knowledge/l2-ontology.owl#EventType": EventType,
    "http://aicor.knowledge/l2-ontology.owl#FluentType": FluentType,
    "http://aicor.knowledge/l2-ontology.owl#URDF": Urdf,
    "http://aicor.knowledge/l2-ontology.owl#BeliefStateSemanticDigitalTwin": BeliefStateSemanticDigitalTwin,
    "http://aicor.knowledge/l2-ontology.owl#CRAMSemanticDigitalTwin": CramSemanticDigitalTwin,
    "http://aicor.knowledge/l2-ontology.owl#AbstentionSignal": AbstentionSignal,
    "http://aicor.knowledge/l2-ontology.owl#CompletionSignal": CompletionSignal,
    "http://aicor.knowledge/l2-ontology.owl#FailureSignal": FailureSignal,
    "http://aicor.knowledge/l2-ontology.owl#SuccessSignal": SuccessSignal,
    "http://aicor.knowledge/l2-ontology.owl#CollisionGeometry": CollisionGeometry,
    "http://aicor.knowledge/l2-ontology.owl#VisualGeometry": VisualGeometry,
    "http://aicor.knowledge/l2-ontology.owl#SceneGraph": SceneGraph,
    "http://aicor.knowledge/l2-ontology.owl#RobotBody": RobotBody,
    "http://aicor.knowledge/l2-ontology.owl#EndEffector": EndEffector,
    "http://aicor.knowledge/l2-ontology.owl#PhysicalIndividual": PhysicalIndividual,
    "http://aicor.knowledge/l2-ontology.owl#Substance": Substance,
    "http://aicor.knowledge/l2-ontology.owl#RiskAvoidanceReasoning": RiskAvoidanceReasoning,
    "http://aicor.knowledge/l2-ontology.owl#TradeOffReasoning": TradeOffReasoning,
    "http://aicor.knowledge/l2-ontology.owl#ActionPlanning": ActionPlanning,
    "http://aicor.knowledge/l2-ontology.owl#EpisodicSimulation": EpisodicSimulation,
    "http://aicor.knowledge/l2-ontology.owl#GoalSetting": GoalSetting,
    "http://aicor.knowledge/l2-ontology.owl#OutcomePrediction": OutcomePrediction,
    "http://aicor.knowledge/l2-ontology.owl#FoundationModelOutput": FoundationModelOutput,
    "http://aicor.knowledge/l2-ontology.owl#ManipulationHypothesis": ManipulationHypothesis,
    "http://aicor.knowledge/l2-ontology.owl#ArtifactCertification": ArtifactCertification,
    "http://aicor.knowledge/l2-ontology.owl#StatisticalWorldModel": StatisticalWorldModel,
    "http://aicor.knowledge/l2-ontology.owl#WorldStructureModel": WorldStructureModel,
    "http://aicor.knowledge/l2-ontology.owl#Actuating": Actuating,
    "http://aicor.knowledge/l2-ontology.owl#Disambiguating": Disambiguating,
    "http://aicor.knowledge/l2-ontology.owl#Proposing": Proposing,
    "http://aicor.knowledge/l2-ontology.owl#Rationalizing": Rationalizing,
    "http://aicor.knowledge/l2-ontology.owl#Realizing": Realizing,
    "http://aicor.knowledge/l2-ontology.owl#Assessing": Assessing,
    "http://aicor.knowledge/l2-ontology.owl#Monitoring": Monitoring,
    "http://aicor.knowledge/l2-ontology.owl#Perceiving": Perceiving,
    "http://aicor.knowledge/l2-ontology.owl#Regulating": Regulating,
    "http://aicor.knowledge/l2-ontology.owl#CRAMStructuredReactiveController": CramStructuredReactiveController,
    "http://aicor.knowledge/l2-ontology.owl#RequestedPlacementProblem": RequestedPlacementProblem,
    "http://aicor.knowledge/l2-ontology.owl#PlacementPipeline": PlacementPipeline,
    "http://aicor.knowledge/l2-ontology.owl#SolvedEggPickup": SolvedEggPickup,
    "http://aicor.knowledge/l2-ontology.owl#ModelPredictiveControl": ModelPredictiveControl,
    "http://aicor.knowledge/l2-ontology.owl#CompetenceProperty": CompetenceProperty,
    "http://aicor.knowledge/l2-ontology.owl#ComponentCompetenceClaim": ComponentCompetenceClaim,
    "http://aicor.knowledge/l2-ontology.owl#FederationWarrantedCompetenceClaim": FederationWarrantedCompetenceClaim,
    "http://aicor.knowledge/l2-ontology.owl#PurposeIndexedCompetenceClaim": PurposeIndexedCompetenceClaim,
    "http://aicor.knowledge/l2-ontology.owl#ExecutiveAdequacy": ExecutiveAdequacy,
    "http://aicor.knowledge/l2-ontology.owl#RepresentationalAccessibility": RepresentationalAccessibility,
    "http://aicor.knowledge/l2-ontology.owl#ControlRealizability": ControlRealizability,
    "http://aicor.knowledge/l2-ontology.owl#InformationalAvailability": InformationalAvailability,
    "http://aicor.knowledge/l2-ontology.owl#PhysicalSolvability": PhysicalSolvability,
    "http://aicor.knowledge/l2-ontology.owl#OpenEntryTerms": OpenEntryTerms,
    "http://aicor.knowledge/l2-ontology.owl#RestrictedEntryTerms": RestrictedEntryTerms,
    "http://aicor.knowledge/l2-ontology.owl#ActionManifold": ActionManifold,
    "http://aicor.knowledge/l2-ontology.owl#ActionRef": ActionRef,
    "http://aicor.knowledge/l2-ontology.owl#ActionRelatedPlaceModel": ActionRelatedPlaceModel,
    "http://aicor.knowledge/l2-ontology.owl#ActionTokenDesignator": ActionTokenDesignator,
    "http://aicor.knowledge/l2-ontology.owl#ActualCourse": ActualCourse,
    "http://aicor.knowledge/l2-ontology.owl#Affordance": Affordance,
    "http://aicor.knowledge/l2-ontology.owl#Arrangement": Arrangement,
    "http://aicor.knowledge/l2-ontology.owl#AuthorityTier": AuthorityTier,
    "http://aicor.knowledge/l2-ontology.owl#BayesianRisk": BayesianRisk,
    "http://aicor.knowledge/l2-ontology.owl#BehaviorFlaw": BehaviorFlaw,
    "http://aicor.knowledge/l2-ontology.owl#BeliefState": BeliefState,
    "http://aicor.knowledge/l2-ontology.owl#BelongingNorm": BelongingNorm,
    "http://aicor.knowledge/l2-ontology.owl#BodyWorldConfiguration": BodyWorldConfiguration,
    "http://aicor.knowledge/l2-ontology.owl#Bug": Bug,
    "http://aicor.knowledge/l2-ontology.owl#CRAMDesignator": CramDesignator,
    "http://aicor.knowledge/l2-ontology.owl#CRAMPartialDescription": CramPartialDescription,
    "http://aicor.knowledge/l2-ontology.owl#CodeNode": CodeNode,
    "http://aicor.knowledge/l2-ontology.owl#CodePath": CodePath,
    "http://aicor.knowledge/l2-ontology.owl#CodeTree": CodeTree,
    "http://aicor.knowledge/l2-ontology.owl#Commitment": Commitment,
    "http://aicor.knowledge/l2-ontology.owl#ConceptPattern": ConceptPattern,
    "http://aicor.knowledge/l2-ontology.owl#ConcurrentProcessInterference": ConcurrentProcessInterference,
    "http://aicor.knowledge/l2-ontology.owl#ConstituentPart": ConstituentPart,
    "http://aicor.knowledge/l2-ontology.owl#ContinuousFlow": ContinuousFlow,
    "http://aicor.knowledge/l2-ontology.owl#DesignatorConstraintSet": DesignatorConstraintSet,
    "http://aicor.knowledge/l2-ontology.owl#DesignatorParameter": DesignatorParameter,
    "http://aicor.knowledge/l2-ontology.owl#DiscreteMode": DiscreteMode,
    "http://aicor.knowledge/l2-ontology.owl#Divergence": Divergence,
    "http://aicor.knowledge/l2-ontology.owl#ExecutionScenario": ExecutionScenario,
    "http://aicor.knowledge/l2-ontology.owl#ForceDynamicEvent": ForceDynamicEvent,
    "http://aicor.knowledge/l2-ontology.owl#GeneralizedSuccessModel": GeneralizedSuccessModel,
    "http://aicor.knowledge/l2-ontology.owl#GenerativeModel": GenerativeModel,
    "http://aicor.knowledge/l2-ontology.owl#HumanActivityObservation": HumanActivityObservation,
    "http://aicor.knowledge/l2-ontology.owl#Intention": Intention,
    "http://aicor.knowledge/l2-ontology.owl#IntentionalStructure": IntentionalStructure,
    "http://aicor.knowledge/l2-ontology.owl#LossFunction": LossFunction,
    "http://aicor.knowledge/l2-ontology.owl#ManipulationPhase": ManipulationPhase,
    "http://aicor.knowledge/l2-ontology.owl#MotionPhase": MotionPhase,
    "http://aicor.knowledge/l2-ontology.owl#MotionStatechart": MotionStatechart,
    "http://aicor.knowledge/l2-ontology.owl#MotionTask": MotionTask,
    "http://aicor.knowledge/l2-ontology.owl#Narrative": Narrative,
    "http://aicor.knowledge/l2-ontology.owl#ObjectTyping": ObjectTyping,
    "http://aicor.knowledge/l2-ontology.owl#PEAM": Peam,
    "http://aicor.knowledge/l2-ontology.owl#PlaceProperty": PlaceProperty,
    "http://aicor.knowledge/l2-ontology.owl#PlacementRequest": PlacementRequest,
    "http://aicor.knowledge/l2-ontology.owl#PlacementTarget": PlacementTarget,
    "http://aicor.knowledge/l2-ontology.owl#PlanFailure": PlanFailure,
    "http://aicor.knowledge/l2-ontology.owl#Policy": Policy,
    "http://aicor.knowledge/l2-ontology.owl#Portion": Portion,
    "http://aicor.knowledge/l2-ontology.owl#PossibleWorld": PossibleWorld,
    "http://aicor.knowledge/l2-ontology.owl#ProbabilisticHybridActionModel": ProbabilisticHybridActionModel,
    "http://aicor.knowledge/l2-ontology.owl#Program": Program,
    "http://aicor.knowledge/l2-ontology.owl#QualiaStructure": QualiaStructure,
    "http://aicor.knowledge/l2-ontology.owl#ReferenceOutcome": ReferenceOutcome,
    "http://aicor.knowledge/l2-ontology.owl#RelationalRepresentation": RelationalRepresentation,
    "http://aicor.knowledge/l2-ontology.owl#RequestConstraint": RequestConstraint,
    "http://aicor.knowledge/l2-ontology.owl#RequestScope": RequestScope,
    "http://aicor.knowledge/l2-ontology.owl#RestorationObligation": RestorationObligation,
    "http://aicor.knowledge/l2-ontology.owl#RestorationType": RestorationType,
    "http://aicor.knowledge/l2-ontology.owl#SelectionNorm": SelectionNorm,
    "http://aicor.knowledge/l2-ontology.owl#ServingNorm": ServingNorm,
    "http://aicor.knowledge/l2-ontology.owl#SpatialRelation": SpatialRelation,
    "http://aicor.knowledge/l2-ontology.owl#StabilityCondition": StabilityCondition,
    "http://aicor.knowledge/l2-ontology.owl#StateEstimate": StateEstimate,
    "http://aicor.knowledge/l2-ontology.owl#StatisticalRelationalModel": StatisticalRelationalModel,
    "http://aicor.knowledge/l2-ontology.owl#SubstanceReference": SubstanceReference,
    "http://aicor.knowledge/l2-ontology.owl#TagTable": TagTable,
    "http://aicor.knowledge/l2-ontology.owl#Task": Task,
    "http://aicor.knowledge/l2-ontology.owl#TaskNetwork": TaskNetwork,
    "http://aicor.knowledge/l2-ontology.owl#Telos": Telos,
    "http://aicor.knowledge/l2-ontology.owl#UnresolvedResolution": UnresolvedResolution,
    "http://aicor.knowledge/l2-ontology.owl#Vessel": Vessel,
    "http://aicor.knowledge/l2-ontology.owl#CommensurableLaboratory": CommensurableLaboratory,
    "http://aicor.knowledge/l2-ontology.owl#ActionEventType": ActionEventType,
    "http://aicor.knowledge/l2-ontology.owl#CrackEvent": CrackEvent,
    "http://aicor.knowledge/l2-ontology.owl#FailPrimitive": FailPrimitive,
    "http://aicor.knowledge/l2-ontology.owl#ForceDynamicEventType": ForceDynamicEventType,
    "http://aicor.knowledge/l2-ontology.owl#InformationProcessingEvent": InformationProcessingEvent,
    "http://aicor.knowledge/l2-ontology.owl#ObservationEvent": ObservationEvent,
    "http://aicor.knowledge/l2-ontology.owl#OpenUpEvent": OpenUpEvent,
    "http://aicor.knowledge/l2-ontology.owl#SpreadArrestEvent": SpreadArrestEvent,
    "http://aicor.knowledge/l2-ontology.owl#ContainerClosureState": ContainerClosureState,
    "http://aicor.knowledge/l2-ontology.owl#Fluent": Fluent,
    "http://aicor.knowledge/l2-ontology.owl#TaskStatus": TaskStatus,
    "http://aicor.knowledge/l2-ontology.owl#CouplingAlignedSubtree": CouplingAlignedSubtree,
    "http://aicor.knowledge/l2-ontology.owl#SemanticObjectMap": SemanticObjectMap,
    "http://aicor.knowledge/l2-ontology.owl#Carrier": Carrier,
    "http://aicor.knowledge/l2-ontology.owl#Container": Container_,
    "http://aicor.knowledge/l2-ontology.owl#Stack": Stack,
    "http://aicor.knowledge/l2-ontology.owl#CompetenceHypothesis": CompetenceHypothesis,
    "http://aicor.knowledge/l2-ontology.owl#NecessityHypothesis": NecessityHypothesis,
    "http://aicor.knowledge/l2-ontology.owl#BodyWorldProcessMicrotheory": BodyWorldProcessMicrotheory,
    "http://aicor.knowledge/l2-ontology.owl#AccountabilityClaim": AccountabilityClaim,
    "http://aicor.knowledge/l2-ontology.owl#AwarenessClaim": AwarenessClaim,
    "http://aicor.knowledge/l2-ontology.owl#CompletenessClaim": CompletenessClaim,
    "http://aicor.knowledge/l2-ontology.owl#CorrectnessClaim": CorrectnessClaim,
    "http://aicor.knowledge/l2-ontology.owl#PerformanceClaim": PerformanceClaim,
    "http://aicor.knowledge/l2-ontology.owl#PotencyClaim": PotencyClaim,
    "http://aicor.knowledge/l2-ontology.owl#SafetyClaim": SafetyClaim,
    "http://aicor.knowledge/l2-ontology.owl#RequestedActionRef": RequestedActionRef,
    "http://aicor.knowledge/l2-ontology.owl#ExecutedActionDesignator": ExecutedActionDesignator,
    "http://aicor.knowledge/l2-ontology.owl#RequestDesignator": RequestDesignator,
    "http://aicor.knowledge/l2-ontology.owl#Load": Load,
    "http://aicor.knowledge/l2-ontology.owl#GuestAuthority": GuestAuthority,
    "http://aicor.knowledge/l2-ontology.owl#InferredAuthority": InferredAuthority,
    "http://aicor.knowledge/l2-ontology.owl#ResidentAuthority": ResidentAuthority,
    "http://aicor.knowledge/l2-ontology.owl#ServedConfiguration": ServedConfiguration,
    "http://aicor.knowledge/l2-ontology.owl#AttributiveDesignator": AttributiveDesignator,
    "http://aicor.knowledge/l2-ontology.owl#GroundedDesignator": GroundedDesignator,
    "http://aicor.knowledge/l2-ontology.owl#LocationDesignator": LocationDesignator,
    "http://aicor.knowledge/l2-ontology.owl#MotionDesignator": MotionDesignator,
    "http://aicor.knowledge/l2-ontology.owl#ObjectDesignator": ObjectDesignator,
    "http://aicor.knowledge/l2-ontology.owl#ReferentialDesignator": ReferentialDesignator,
    "http://aicor.knowledge/l2-ontology.owl#ForceDynamicState": ForceDynamicState,
    "http://aicor.knowledge/l2-ontology.owl#EffluxOnsetEvent": EffluxOnsetEvent,
    "http://aicor.knowledge/l2-ontology.owl#FractureEvent": FractureEvent,
    "http://aicor.knowledge/l2-ontology.owl#GraspEvent": GraspEvent,
    "http://aicor.knowledge/l2-ontology.owl#CuttingIntention": CuttingIntention,
    "http://aicor.knowledge/l2-ontology.owl#PouringIntention": PouringIntention,
    "http://aicor.knowledge/l2-ontology.owl#CuttingPhase": CuttingPhase,
    "http://aicor.knowledge/l2-ontology.owl#PouringPhase": PouringPhase,
    "http://aicor.knowledge/l2-ontology.owl#ElaborationGraph": ElaborationGraph,
    "http://aicor.knowledge/l2-ontology.owl#NEEM": Neem,
    "http://aicor.knowledge/l2-ontology.owl#OverridingRequest": OverridingRequest,
    "http://aicor.knowledge/l2-ontology.owl#StandingRequest": StandingRequest,
    "http://aicor.knowledge/l2-ontology.owl#CRAMControlProgram": CramControlProgram,
    "http://aicor.knowledge/l2-ontology.owl#GeneralizedActionPlan": GeneralizedActionPlan,
    "http://aicor.knowledge/l2-ontology.owl#MotionPlan": MotionPlan,
    "http://aicor.knowledge/l2-ontology.owl#Plan": Plan,
    "http://aicor.knowledge/l2-ontology.owl#Diverges": Diverges,
    "http://aicor.knowledge/l2-ontology.owl#Fails": Fails,
    "http://aicor.knowledge/l2-ontology.owl#Realizes": Realizes,
    "http://aicor.knowledge/l2-ontology.owl#DestinationConstraint": DestinationConstraint,
    "http://aicor.knowledge/l2-ontology.owl#ExclusionConstraint": ExclusionConstraint,
    "http://aicor.knowledge/l2-ontology.owl#PropertyConstraint": PropertyConstraint,
    "http://aicor.knowledge/l2-ontology.owl#RelationConstraint": RelationConstraint,
    "http://aicor.knowledge/l2-ontology.owl#ActivityBoundScope": ActivityBoundScope,
    "http://aicor.knowledge/l2-ontology.owl#EpisodicScope": EpisodicScope,
    "http://aicor.knowledge/l2-ontology.owl#StandingScope": StandingScope,
    "http://aicor.knowledge/l2-ontology.owl#ConfigurationRestoration": ConfigurationRestoration,
    "http://aicor.knowledge/l2-ontology.owl#ExactReturn": ExactReturn,
    "http://aicor.knowledge/l2-ontology.owl#FunctionalReturn": FunctionalReturn,
    "http://aicor.knowledge/l2-ontology.owl#StateMediatedReturn": StateMediatedReturn,
    "http://aicor.knowledge/l2-ontology.owl#SuccessorStateResolution": SuccessorStateResolution,
    "http://aicor.knowledge/l2-ontology.owl#AccessibleFrom": AccessibleFrom,
    "http://aicor.knowledge/l2-ontology.owl#AlignedWith": AlignedWith,
    "http://aicor.knowledge/l2-ontology.owl#AttachedTo": AttachedTo,
    "http://aicor.knowledge/l2-ontology.owl#Inside": Inside,
    "http://aicor.knowledge/l2-ontology.owl#On": On,
    "http://aicor.knowledge/l2-ontology.owl#ControlStructure": ControlStructure,
    "http://aicor.knowledge/l2-ontology.owl#CuttingTelos": CuttingTelos,
    "http://aicor.knowledge/l2-ontology.owl#PouringTelos": PouringTelos,
    "http://aicor.knowledge/l2-ontology.owl#Bowl": Bowl,
    "http://aicor.knowledge/l2-ontology.owl#Cup": Cup,
    "http://aicor.knowledge/l2-ontology.owl#DrinkingVessel": DrinkingVessel,
    "http://aicor.knowledge/l2-ontology.owl#Plate": Plate,
    "http://aicor.knowledge/l2-ontology.owl#RuptureEventType": RuptureEventType,
    "http://aicor.knowledge/l2-ontology.owl#SlipEventType": SlipEventType,
    "http://aicor.knowledge/l2-ontology.owl#PlanInterpretationEvent": PlanInterpretationEvent,
    "http://aicor.knowledge/l2-ontology.owl#ThermalSetArrest": ThermalSetArrest,
    "http://aicor.knowledge/l2-ontology.owl#YieldEquilibriumArrest": YieldEquilibriumArrest,
    "http://aicor.knowledge/l2-ontology.owl#ClosedContainerState": ClosedContainerState,
    "http://aicor.knowledge/l2-ontology.owl#OpenContainerState": OpenContainerState,
    "http://aicor.knowledge/l2-ontology.owl#ActiveStatus": ActiveStatus,
    "http://aicor.knowledge/l2-ontology.owl#CreatedStatus": CreatedStatus,
    "http://aicor.knowledge/l2-ontology.owl#NonterminalStatus": NonterminalStatus,
    "http://aicor.knowledge/l2-ontology.owl#TerminalStatus": TerminalStatus,
    "http://aicor.knowledge/l2-ontology.owl#MonotonicityHypothesis": MonotonicityHypothesis,
    "http://aicor.knowledge/l2-ontology.owl#TransferHypothesis": TransferHypothesis,
    "http://aicor.knowledge/l2-ontology.owl#ApplicationCentricMicrotheory": ApplicationCentricMicrotheory,
    "http://aicor.knowledge/l2-ontology.owl#BodyCentricMicrotheory": BodyCentricMicrotheory,
    "http://aicor.knowledge/l2-ontology.owl#OpenSurfaceSpreading": OpenSurfaceSpreading,
    "http://aicor.knowledge/l2-ontology.owl#ProcessCentricMicrotheory": ProcessCentricMicrotheory,
    "http://aicor.knowledge/l2-ontology.owl#DivergentDesignator": DivergentDesignator,
    "http://aicor.knowledge/l2-ontology.owl#RealizingDesignator": RealizingDesignator,
    "http://aicor.knowledge/l2-ontology.owl#ActionDesignator": ActionDesignator,
    "http://aicor.knowledge/l2-ontology.owl#EntityDesignator": EntityDesignator,
    "http://aicor.knowledge/l2-ontology.owl#AccumulatedState": AccumulatedState,
    "http://aicor.knowledge/l2-ontology.owl#ContainedState": ContainedState,
    "http://aicor.knowledge/l2-ontology.owl#FlowingState": FlowingState,
    "http://aicor.knowledge/l2-ontology.owl#GraspedState": GraspedState,
    "http://aicor.knowledge/l2-ontology.owl#SeparatedState": SeparatedState,
    "http://aicor.knowledge/l2-ontology.owl#StressedState": StressedState,
    "http://aicor.knowledge/l2-ontology.owl#ArrangementTemplate": ArrangementTemplate,
    "http://aicor.knowledge/l2-ontology.owl#CourseOfAction": CourseOfAction,
    "http://aicor.knowledge/l2-ontology.owl#ManipulationActionCategory": ManipulationActionCategory,
    "http://aicor.knowledge/l2-ontology.owl#PreparationMethod": PreparationMethod,
    "http://aicor.knowledge/l2-ontology.owl#ForceDynamicTemplate": ForceDynamicTemplate,
    "http://aicor.knowledge/l2-ontology.owl#StructuredReactivePlan": StructuredReactivePlan,
    "http://aicor.knowledge/l2-ontology.owl#Par": Par,
    "http://aicor.knowledge/l2-ontology.owl#PerformTask": PerformTask,
    "http://aicor.knowledge/l2-ontology.owl#Seq": Seq,
    "http://aicor.knowledge/l2-ontology.owl#TryAll": TryAll,
    "http://aicor.knowledge/l2-ontology.owl#TryInOrder": TryInOrder,
    "http://aicor.knowledge/l2-ontology.owl#Whenever": Whenever,
    "http://aicor.knowledge/l2-ontology.owl#WithPolicy": WithPolicy,
    "http://aicor.knowledge/l2-ontology.owl#BeginTask": BeginTask,
    "http://aicor.knowledge/l2-ontology.owl#EndTask": EndTask,
    "http://aicor.knowledge/l2-ontology.owl#ResolveEvent": ResolveEvent,
    "http://aicor.knowledge/l2-ontology.owl#CrackedContainerState": CrackedContainerState,
    "http://aicor.knowledge/l2-ontology.owl#IntactContainerState": IntactContainerState,
    "http://aicor.knowledge/l2-ontology.owl#Active": Active,
    "http://aicor.knowledge/l2-ontology.owl#Created": Created,
    "http://aicor.knowledge/l2-ontology.owl#Enabled": Enabled,
    "http://aicor.knowledge/l2-ontology.owl#Evaporated": Evaporated,
    "http://aicor.knowledge/l2-ontology.owl#EvaporatedStatus": EvaporatedStatus,
    "http://aicor.knowledge/l2-ontology.owl#Failed": Failed,
    "http://aicor.knowledge/l2-ontology.owl#FailedStatus": FailedStatus,
    "http://aicor.knowledge/l2-ontology.owl#Finished": Finished,
    "http://aicor.knowledge/l2-ontology.owl#SucceededStatus": SucceededStatus,
    "http://aicor.knowledge/l2-ontology.owl#ConfigurationChange": ConfigurationChange,
    "http://aicor.knowledge/l2-ontology.owl#ConnectingAction": ConnectingAction,
    "http://aicor.knowledge/l2-ontology.owl#InteractionAction": InteractionAction,
    "http://aicor.knowledge/l2-ontology.owl#MaintenanceAction": MaintenanceAction,
    "http://aicor.knowledge/l2-ontology.owl#ObjectDisplacement": ObjectDisplacement,
    "http://aicor.knowledge/l2-ontology.owl#PrimitiveAction": PrimitiveAction,
    "http://aicor.knowledge/l2-ontology.owl#PurposeConstrainedCategory": PurposeConstrainedCategory,
    "http://aicor.knowledge/l2-ontology.owl#SubstanceManipulation": SubstanceManipulation,
    "http://aicor.knowledge/l2-ontology.owl#SurfaceTreatment": SurfaceTreatment,
    "http://aicor.knowledge/l2-ontology.owl#CuttingTemplate": CuttingTemplate,
    "http://aicor.knowledge/l2-ontology.owl#PouringTemplate": PouringTemplate,
    "http://aicor.knowledge/l2-ontology.owl#CRAMArchitecture": CramArchitecture,
    "http://aicor.knowledge/l2-ontology.owl#HeldIntactEgg": HeldIntactEgg,
    "http://aicor.knowledge/l2-ontology.owl#Transporting": Transporting,
    "http://aicor.knowledge/l2-ontology.owl#Grasping": Grasping,
    "http://aicor.knowledge/l2-ontology.owl#Lifting": Lifting,
    "http://aicor.knowledge/l2-ontology.owl#Reaching": Reaching,
    "http://aicor.knowledge/l2-ontology.owl#Stroking": Stroking,
    "http://aicor.knowledge/l2-ontology.owl#Tilting": Tilting,
    "http://aicor.knowledge/l2-ontology.owl#Dividing": Dividing,
    "http://aicor.knowledge/l2-ontology.owl#Transferring": Transferring,
    "http://aicor.knowledge/l2-ontology.owl#Wiping": Wiping,
    "http://aicor.knowledge/l2-ontology.owl#Cutting": Cutting,
    "http://aicor.knowledge/l2-ontology.owl#Pouring": Pouring,
    "http://aicor.knowledge/l2-ontology.owl#Slicing": Slicing,
}
PROPERTY_FIELD_NAMES = {
    "http://aicor.knowledge/l2-ontology.owl#aboutExecutionOf": "about_execution_of",
    "http://aicor.knowledge/l2-ontology.owl#aboutFutureExecutionOf": "about_future_execution_of",
    "http://aicor.knowledge/l2-ontology.owl#aboutReferent": "about_referent",
    "http://aicor.knowledge/l2-ontology.owl#aboutScientificObject": "about_scientific_object",
    "http://aicor.knowledge/l2-ontology.owl#aboutSystemFamily": "about_system_family",
    "http://aicor.knowledge/l2-ontology.owl#abstainsOn": "abstains_on",
    "http://aicor.knowledge/l2-ontology.owl#abstractsAway": "abstracts_away",
    "http://aicor.knowledge/l2-ontology.owl#accountableVia": "accountable_via",
    "http://aicor.knowledge/l2-ontology.owl#achievedBy": "achieved_by",
    "http://aicor.knowledge/l2-ontology.owl#achieves": "achieves",
    "http://aicor.knowledge/l2-ontology.owl#acquires": "acquires",
    "http://aicor.knowledge/l2-ontology.owl#acquiresVia": "acquires_via",
    "http://aicor.knowledge/l2-ontology.owl#actsOnState": "acts_on_state",
    "http://aicor.knowledge/l2-ontology.owl#actuatedBy": "actuated_by",
    "http://aicor.knowledge/l2-ontology.owl#actuates": "actuates",
    "http://aicor.knowledge/l2-ontology.owl#adaptsBy": "adapts_by",
    "http://aicor.knowledge/l2-ontology.owl#admitsEmbodiment": "admits_embodiment",
    "http://aicor.knowledge/l2-ontology.owl#admitsEnvironmentModel": "admits_environment_model",
    "http://aicor.knowledge/l2-ontology.owl#admitsTaskType": "admits_task_type",
    "http://aicor.knowledge/l2-ontology.owl#adoptsCandidate": "adopts_candidate",
    "http://aicor.knowledge/l2-ontology.owl#adoptsPrinciple": "adopts_principle",
    "http://aicor.knowledge/l2-ontology.owl#affordsAction": "affords_action",
    "http://aicor.knowledge/l2-ontology.owl#against": "against",
    "http://aicor.knowledge/l2-ontology.owl#alongDimension": "along_dimension",
    "http://aicor.knowledge/l2-ontology.owl#anchoredByEpisode": "anchored_by_episode",
    "http://aicor.knowledge/l2-ontology.owl#answeredBy": "answered_by",
    "http://aicor.knowledge/l2-ontology.owl#answers": "answers",
    "http://aicor.knowledge/l2-ontology.owl#answersQuery": "answers_query",
    "http://aicor.knowledge/l2-ontology.owl#answersQuestion": "answers_question",
    "http://aicor.knowledge/l2-ontology.owl#answersRequest": "answers_request",
    "http://aicor.knowledge/l2-ontology.owl#appliesCritic": "applies_critic",
    "http://aicor.knowledge/l2-ontology.owl#appliesFirstOf": "applies_first_of",
    "http://aicor.knowledge/l2-ontology.owl#appliesInState": "applies_in_state",
    "http://aicor.knowledge/l2-ontology.owl#appliesInterruptibly": "applies_interruptibly",
    "http://aicor.knowledge/l2-ontology.owl#appliesModel": "applies_model",
    "http://aicor.knowledge/l2-ontology.owl#appliesStep": "applies_step",
    "http://aicor.knowledge/l2-ontology.owl#appliesTo": "applies_to",
    "http://aicor.knowledge/l2-ontology.owl#appliesToCoupling": "applies_to_coupling",
    "http://aicor.knowledge/l2-ontology.owl#architectureOf": "architecture_of",
    "http://aicor.knowledge/l2-ontology.owl#arrestedByEvent": "arrested_by_event",
    "http://aicor.knowledge/l2-ontology.owl#assertedBy": "asserted_by",
    "http://aicor.knowledge/l2-ontology.owl#assertsExistenceOver": "asserts_existence_over",
    "http://aicor.knowledge/l2-ontology.owl#assertsStandard": "asserts_standard",
    "http://aicor.knowledge/l2-ontology.owl#assessedAgainst": "assessed_against",
    "http://aicor.knowledge/l2-ontology.owl#assessedBy": "assessed_by",
    "http://aicor.knowledge/l2-ontology.owl#assessedOver": "assessed_over",
    "http://aicor.knowledge/l2-ontology.owl#assesses": "assesses",
    "http://aicor.knowledge/l2-ontology.owl#assessesCase": "assesses_case",
    "http://aicor.knowledge/l2-ontology.owl#assessesDimension": "assesses_dimension",
    "http://aicor.knowledge/l2-ontology.owl#assessesEpisode": "assesses_episode",
    "http://aicor.knowledge/l2-ontology.owl#assessesTrace": "assesses_trace",
    "http://aicor.knowledge/l2-ontology.owl#assignsClass": "assigns_class",
    "http://aicor.knowledge/l2-ontology.owl#assignsProbabilityOver": "assigns_probability_over",
    "http://aicor.knowledge/l2-ontology.owl#assignsStatus": "assigns_status",
    "http://aicor.knowledge/l2-ontology.owl#assumesEmbodimentClass": "assumes_embodiment_class",
    "http://aicor.knowledge/l2-ontology.owl#atContactPair": "at_contact_pair",
    "http://aicor.knowledge/l2-ontology.owl#atDegree": "at_degree",
    "http://aicor.knowledge/l2-ontology.owl#atlasOf": "atlas_of",
    "http://aicor.knowledge/l2-ontology.owl#attestedBy": "attested_by",
    "http://aicor.knowledge/l2-ontology.owl#authoredBy": "authored_by",
    "http://aicor.knowledge/l2-ontology.owl#authorizedByActivity": "authorized_by_activity",
    "http://aicor.knowledge/l2-ontology.owl#bearerOfClaim": "bearer_of_claim",
    "http://aicor.knowledge/l2-ontology.owl#betweenRoleFillers": "between_role_fillers",
    "http://aicor.knowledge/l2-ontology.owl#bindsFiller": "binds_filler",
    "http://aicor.knowledge/l2-ontology.owl#bindsInstance": "binds_instance",
    "http://aicor.knowledge/l2-ontology.owl#bindsRole": "binds_role",
    "http://aicor.knowledge/l2-ontology.owl#bindsSlot": "binds_slot",
    "http://aicor.knowledge/l2-ontology.owl#bindsVariable": "binds_variable",
    "http://aicor.knowledge/l2-ontology.owl#boundTo": "bound_to",
    "http://aicor.knowledge/l2-ontology.owl#boundaryRelativeTo": "boundary_relative_to",
    "http://aicor.knowledge/l2-ontology.owl#branchesInto": "branches_into",
    "http://aicor.knowledge/l2-ontology.owl#bridges": "bridges",
    "http://aicor.knowledge/l2-ontology.owl#buildsTaskNetwork": "builds_task_network",
    "http://aicor.knowledge/l2-ontology.owl#calls": "calls",
    "http://aicor.knowledge/l2-ontology.owl#capabilityOf": "capability_of",
    "http://aicor.knowledge/l2-ontology.owl#carriedBy": "carried_by",
    "http://aicor.knowledge/l2-ontology.owl#carriesGuarantee": "carries_guarantee",
    "http://aicor.knowledge/l2-ontology.owl#carriesMetadata": "carries_metadata",
    "http://aicor.knowledge/l2-ontology.owl#causedByPerform": "caused_by_perform",
    "http://aicor.knowledge/l2-ontology.owl#certifiesArtifact": "certifies_artifact",
    "http://aicor.knowledge/l2-ontology.owl#characteristicFailure": "characteristic_failure",
    "http://aicor.knowledge/l2-ontology.owl#characterizedBy": "characterized_by",
    "http://aicor.knowledge/l2-ontology.owl#characterizes": "characterizes",
    "http://aicor.knowledge/l2-ontology.owl#characterizesMethod": "characterizes_method",
    "http://aicor.knowledge/l2-ontology.owl#checkedAgainst": "checked_against",
    "http://aicor.knowledge/l2-ontology.owl#checkedBy": "checked_by",
    "http://aicor.knowledge/l2-ontology.owl#checks": "checks",
    "http://aicor.knowledge/l2-ontology.owl#claimsAboutEpisode": "claims_about_episode",
    "http://aicor.knowledge/l2-ontology.owl#claimsOver": "claims_over",
    "http://aicor.knowledge/l2-ontology.owl#classifiedBy": "classified_by",
    "http://aicor.knowledge/l2-ontology.owl#classifies": "classifies",
    "http://aicor.knowledge/l2-ontology.owl#coRefersWith": "co_refers_with",
    "http://aicor.knowledge/l2-ontology.owl#coercesObjectTo": "coerces_object_to",
    "http://aicor.knowledge/l2-ontology.owl#collectedInto": "collected_into",
    "http://aicor.knowledge/l2-ontology.owl#collectsClaim": "collects_claim",
    "http://aicor.knowledge/l2-ontology.owl#combines": "combines",
    "http://aicor.knowledge/l2-ontology.owl#combinesStatisticalModel": "combines_statistical_model",
    "http://aicor.knowledge/l2-ontology.owl#combinesStructuralModel": "combines_structural_model",
    "http://aicor.knowledge/l2-ontology.owl#commitsTarget": "commits_target",
    "http://aicor.knowledge/l2-ontology.owl#commitsThat": "commits_that",
    "http://aicor.knowledge/l2-ontology.owl#comparableUnder": "comparable_under",
    "http://aicor.knowledge/l2-ontology.owl#componentOf": "component_of",
    "http://aicor.knowledge/l2-ontology.owl#composedOf": "composed_of",
    "http://aicor.knowledge/l2-ontology.owl#composesFlowIn": "composes_flow_in",
    "http://aicor.knowledge/l2-ontology.owl#composesMethod": "composes_method",
    "http://aicor.knowledge/l2-ontology.owl#composesSchema": "composes_schema",
    "http://aicor.knowledge/l2-ontology.owl#composesSimulate": "composes_simulate",
    "http://aicor.knowledge/l2-ontology.owl#composesTranslate": "composes_translate",
    "http://aicor.knowledge/l2-ontology.owl#comprisesDefinition": "comprises_definition",
    "http://aicor.knowledge/l2-ontology.owl#comprisesManifold": "comprises_manifold",
    "http://aicor.knowledge/l2-ontology.owl#comprisesTheorem": "comprises_theorem",
    "http://aicor.knowledge/l2-ontology.owl#comprisesVerdictComponent": "comprises_verdict_component",
    "http://aicor.knowledge/l2-ontology.owl#computedBy": "computed_by",
    "http://aicor.knowledge/l2-ontology.owl#computedFrom": "computed_from",
    "http://aicor.knowledge/l2-ontology.owl#computesRisk": "computes_risk",
    "http://aicor.knowledge/l2-ontology.owl#concernsObject": "concerns_object",
    "http://aicor.knowledge/l2-ontology.owl#conditionalOn": "conditional_on",
    "http://aicor.knowledge/l2-ontology.owl#conditionedOn": "conditioned_on",
    "http://aicor.knowledge/l2-ontology.owl#conditionedOnState": "conditioned_on_state",
    "http://aicor.knowledge/l2-ontology.owl#confersForce": "confers_force",
    "http://aicor.knowledge/l2-ontology.owl#configurationSpaceOf": "configuration_space_of",
    "http://aicor.knowledge/l2-ontology.owl#confirms": "confirms",
    "http://aicor.knowledge/l2-ontology.owl#confirmsBindingIn": "confirms_binding_in",
    "http://aicor.knowledge/l2-ontology.owl#confirmsHypothesis": "confirms_hypothesis",
    "http://aicor.knowledge/l2-ontology.owl#conformsTo": "conforms_to",
    "http://aicor.knowledge/l2-ontology.owl#conformsToScope": "conforms_to_scope",
    "http://aicor.knowledge/l2-ontology.owl#connectedBy": "connected_by",
    "http://aicor.knowledge/l2-ontology.owl#connects": "connects",
    "http://aicor.knowledge/l2-ontology.owl#consequentialVia": "consequential_via",
    "http://aicor.knowledge/l2-ontology.owl#constituentBoundedBy": "constituent_bounded_by",
    "http://aicor.knowledge/l2-ontology.owl#constrainedBy": "constrained_by",
    "http://aicor.knowledge/l2-ontology.owl#constrains": "constrains",
    "http://aicor.knowledge/l2-ontology.owl#constrainsDestination": "constrains_destination",
    "http://aicor.knowledge/l2-ontology.owl#constrainsRelation": "constrains_relation",
    "http://aicor.knowledge/l2-ontology.owl#constrainsRung": "constrains_rung",
    "http://aicor.knowledge/l2-ontology.owl#constructedBy": "constructed_by",
    "http://aicor.knowledge/l2-ontology.owl#consumedBy": "consumed_by",
    "http://aicor.knowledge/l2-ontology.owl#consumes": "consumes",
    "http://aicor.knowledge/l2-ontology.owl#contains": "contains",
    "http://aicor.knowledge/l2-ontology.owl#contextualizedBy": "contextualized_by",
    "http://aicor.knowledge/l2-ontology.owl#contributesAt": "contributes_at",
    "http://aicor.knowledge/l2-ontology.owl#controlledBy": "controlled_by",
    "http://aicor.knowledge/l2-ontology.owl#controls": "controls",
    "http://aicor.knowledge/l2-ontology.owl#corridorOf": "corridor_of",
    "http://aicor.knowledge/l2-ontology.owl#costsOver": "costs_over",
    "http://aicor.knowledge/l2-ontology.owl#coversProblemClass": "covers_problem_class",
    "http://aicor.knowledge/l2-ontology.owl#criticizesWith": "criticizes_with",
    "http://aicor.knowledge/l2-ontology.owl#crossingIs": "crossing_is",
    "http://aicor.knowledge/l2-ontology.owl#curates": "curates",
    "http://aicor.knowledge/l2-ontology.owl#declaresAssumptionSet": "declares_assumption_set",
    "http://aicor.knowledge/l2-ontology.owl#declaresBodyWorldMembership": "declares_body_world_membership",
    "http://aicor.knowledge/l2-ontology.owl#declaresExclusion": "declares_exclusion",
    "http://aicor.knowledge/l2-ontology.owl#declaresNuisanceRange": "declares_nuisance_range",
    "http://aicor.knowledge/l2-ontology.owl#declaresSystemEnvironmentSplit": "declares_system_environment_split",
    "http://aicor.knowledge/l2-ontology.owl#declaresTaskRelevantVariable": "declares_task_relevant_variable",
    "http://aicor.knowledge/l2-ontology.owl#declaresValidityInterval": "declares_validity_interval",
    "http://aicor.knowledge/l2-ontology.owl#deepens": "deepens",
    "http://aicor.knowledge/l2-ontology.owl#defeats": "defeats",
    "http://aicor.knowledge/l2-ontology.owl#definedBy": "defined_by",
    "http://aicor.knowledge/l2-ontology.owl#definedOverState": "defined_over_state",
    "http://aicor.knowledge/l2-ontology.owl#definesConcept": "defines_concept",
    "http://aicor.knowledge/l2-ontology.owl#definesConditionOn": "defines_condition_on",
    "http://aicor.knowledge/l2-ontology.owl#definesMap": "defines_map",
    "http://aicor.knowledge/l2-ontology.owl#definesRobustnessRange": "defines_robustness_range",
    "http://aicor.knowledge/l2-ontology.owl#derivedFrom": "derived_from",
    "http://aicor.knowledge/l2-ontology.owl#derivesArtifact": "derives_artifact",
    "http://aicor.knowledge/l2-ontology.owl#describedBy": "described_by",
    "http://aicor.knowledge/l2-ontology.owl#describes": "describes",
    "http://aicor.knowledge/l2-ontology.owl#describesAs": "describes_as",
    "http://aicor.knowledge/l2-ontology.owl#describesEntityVia": "describes_entity_via",
    "http://aicor.knowledge/l2-ontology.owl#designsArtifactFor": "designs_artifact_for",
    "http://aicor.knowledge/l2-ontology.owl#detectedBy": "detected_by",
    "http://aicor.knowledge/l2-ontology.owl#detects": "detects",
    "http://aicor.knowledge/l2-ontology.owl#determinedBy": "determined_by",
    "http://aicor.knowledge/l2-ontology.owl#determinesPerformanceOn": "determines_performance_on",
    "http://aicor.knowledge/l2-ontology.owl#developsScientificObject": "develops_scientific_object",
    "http://aicor.knowledge/l2-ontology.owl#differentialOf": "differential_of",
    "http://aicor.knowledge/l2-ontology.owl#directedBy": "directed_by",
    "http://aicor.knowledge/l2-ontology.owl#disambiguates": "disambiguates",
    "http://aicor.knowledge/l2-ontology.owl#dischargedBy": "discharged_by",
    "http://aicor.knowledge/l2-ontology.owl#discharges": "discharges",
    "http://aicor.knowledge/l2-ontology.owl#discoveredBy": "discovered_by",
    "http://aicor.knowledge/l2-ontology.owl#distinctFrom": "distinct_from",
    "http://aicor.knowledge/l2-ontology.owl#disturbsStability": "disturbs_stability",
    "http://aicor.knowledge/l2-ontology.owl#documents": "documents",
    "http://aicor.knowledge/l2-ontology.owl#drawnFrom": "drawn_from",
    "http://aicor.knowledge/l2-ontology.owl#driftsFrom": "drifts_from",
    "http://aicor.knowledge/l2-ontology.owl#driftsTo": "drifts_to",
    "http://aicor.knowledge/l2-ontology.owl#drivenBy": "driven_by",
    "http://aicor.knowledge/l2-ontology.owl#drives": "drives",
    "http://aicor.knowledge/l2-ontology.owl#drivesRevision": "drives_revision",
    "http://aicor.knowledge/l2-ontology.owl#edits": "edits",
    "http://aicor.knowledge/l2-ontology.owl#elaborates": "elaborates",
    "http://aicor.knowledge/l2-ontology.owl#embeddedWithinPhase": "embedded_within_phase",
    "http://aicor.knowledge/l2-ontology.owl#embeds": "embeds",
    "http://aicor.knowledge/l2-ontology.owl#emits": "emits",
    "http://aicor.knowledge/l2-ontology.owl#emitsHypothesis": "emits_hypothesis",
    "http://aicor.knowledge/l2-ontology.owl#emitsToken": "emits_token",
    "http://aicor.knowledge/l2-ontology.owl#enables": "enables",
    "http://aicor.knowledge/l2-ontology.owl#enablesActionOn": "enables_action_on",
    "http://aicor.knowledge/l2-ontology.owl#enablesLocalRepair": "enables_local_repair",
    "http://aicor.knowledge/l2-ontology.owl#enablesSolutionOf": "enables_solution_of",
    "http://aicor.knowledge/l2-ontology.owl#encodesAttribute": "encodes_attribute",
    "http://aicor.knowledge/l2-ontology.owl#entailedBy": "entailed_by",
    "http://aicor.knowledge/l2-ontology.owl#entailsTerminalState": "entails_terminal_state",
    "http://aicor.knowledge/l2-ontology.owl#entryGovernedBy": "entry_governed_by",
    "http://aicor.knowledge/l2-ontology.owl#enumeratedBy": "enumerated_by",
    "http://aicor.knowledge/l2-ontology.owl#envisionedBy": "envisioned_by",
    "http://aicor.knowledge/l2-ontology.owl#equatesWith": "equates_with",
    "http://aicor.knowledge/l2-ontology.owl#escalatesTo": "escalates_to",
    "http://aicor.knowledge/l2-ontology.owl#establishedBy": "established_by",
    "http://aicor.knowledge/l2-ontology.owl#establishesContact": "establishes_contact",
    "http://aicor.knowledge/l2-ontology.owl#establishesStability": "establishes_stability",
    "http://aicor.knowledge/l2-ontology.owl#estimates": "estimates",
    "http://aicor.knowledge/l2-ontology.owl#evaluatedAgainst": "evaluated_against",
    "http://aicor.knowledge/l2-ontology.owl#evaluatedBy": "evaluated_by",
    "http://aicor.knowledge/l2-ontology.owl#evaluatesIn": "evaluates_in",
    "http://aicor.knowledge/l2-ontology.owl#evidenceBaseFor": "evidence_base_for",
    "http://aicor.knowledge/l2-ontology.owl#evidenceDerivedFrom": "evidence_derived_from",
    "http://aicor.knowledge/l2-ontology.owl#evolvesBy": "evolves_by",
    "http://aicor.knowledge/l2-ontology.owl#examines": "examines",
    "http://aicor.knowledge/l2-ontology.owl#excludesClaimType": "excludes_claim_type",
    "http://aicor.knowledge/l2-ontology.owl#executedBy": "executed_by",
    "http://aicor.knowledge/l2-ontology.owl#executedOn": "executed_on",
    "http://aicor.knowledge/l2-ontology.owl#executes": "executes",
    "http://aicor.knowledge/l2-ontology.owl#executesInto": "executes_into",
    "http://aicor.knowledge/l2-ontology.owl#executesMotion": "executes_motion",
    "http://aicor.knowledge/l2-ontology.owl#executesPlan": "executes_plan",
    "http://aicor.knowledge/l2-ontology.owl#exemptsObjectScope": "exempts_object_scope",
    "http://aicor.knowledge/l2-ontology.owl#expandsInto": "expands_into",
    "http://aicor.knowledge/l2-ontology.owl#explainedByDiscipline": "explained_by_discipline",
    "http://aicor.knowledge/l2-ontology.owl#explains": "explains",
    "http://aicor.knowledge/l2-ontology.owl#explainsPhenomenon": "explains_phenomenon",
    "http://aicor.knowledge/l2-ontology.owl#exploitsConstraint": "exploits_constraint",
    "http://aicor.knowledge/l2-ontology.owl#exposedByLanguage": "exposed_by_language",
    "http://aicor.knowledge/l2-ontology.owl#expressedIn": "expressed_in",
    "http://aicor.knowledge/l2-ontology.owl#expressesProjection": "expresses_projection",
    "http://aicor.knowledge/l2-ontology.owl#expressesTaskAs": "expresses_task_as",
    "http://aicor.knowledge/l2-ontology.owl#failureRoutesTo": "failure_routes_to",
    "http://aicor.knowledge/l2-ontology.owl#federationMember": "federation_member",
    "http://aicor.knowledge/l2-ontology.owl#firedBy": "fired_by",
    "http://aicor.knowledge/l2-ontology.owl#fixedByDefinition": "fixed_by_definition",
    "http://aicor.knowledge/l2-ontology.owl#focusesOn": "focuses_on",
    "http://aicor.knowledge/l2-ontology.owl#forProblem": "for_problem",
    "http://aicor.knowledge/l2-ontology.owl#forPurpose": "for_purpose",
    "http://aicor.knowledge/l2-ontology.owl#forRoleSlot": "for_role_slot",
    "http://aicor.knowledge/l2-ontology.owl#forceDynamicallyPriorTo": "force_dynamically_prior_to",
    "http://aicor.knowledge/l2-ontology.owl#formalizedBy": "formalized_by",
    "http://aicor.knowledge/l2-ontology.owl#formalizes": "formalizes",
    "http://aicor.knowledge/l2-ontology.owl#frameFixes": "frame_fixes",
    "http://aicor.knowledge/l2-ontology.owl#frameOf": "frame_of",
    "http://aicor.knowledge/l2-ontology.owl#framesObject": "frames_object",
    "http://aicor.knowledge/l2-ontology.owl#fromStatus": "from_status",
    "http://aicor.knowledge/l2-ontology.owl#functionsAs": "functions_as",
    "http://aicor.knowledge/l2-ontology.owl#generatedBy": "generated_by",
    "http://aicor.knowledge/l2-ontology.owl#generates": "generates",
    "http://aicor.knowledge/l2-ontology.owl#generatesActionToken": "generates_action_token",
    "http://aicor.knowledge/l2-ontology.owl#generatesBodyMotion": "generates_body_motion",
    "http://aicor.knowledge/l2-ontology.owl#generatesInstance": "generates_instance",
    "http://aicor.knowledge/l2-ontology.owl#generatesTrace": "generates_trace",
    "http://aicor.knowledge/l2-ontology.owl#generatesWorldModel": "generates_world_model",
    "http://aicor.knowledge/l2-ontology.owl#givesInspectableStructureTo": "gives_inspectable_structure_to",
    "http://aicor.knowledge/l2-ontology.owl#goalScope": "goal_scope",
    "http://aicor.knowledge/l2-ontology.owl#governedByPhysics": "governed_by_physics",
    "http://aicor.knowledge/l2-ontology.owl#governsClaim": "governs_claim",
    "http://aicor.knowledge/l2-ontology.owl#governsConsumable": "governs_consumable",
    "http://aicor.knowledge/l2-ontology.owl#governsObjectScope": "governs_object_scope",
    "http://aicor.knowledge/l2-ontology.owl#groundedIn": "grounded_in",
    "http://aicor.knowledge/l2-ontology.owl#groundedInLayer": "grounded_in_layer",
    "http://aicor.knowledge/l2-ontology.owl#grounds": "grounds",
    "http://aicor.knowledge/l2-ontology.owl#groundsPredicate": "grounds_predicate",
    "http://aicor.knowledge/l2-ontology.owl#groundsProperty": "grounds_property",
    "http://aicor.knowledge/l2-ontology.owl#groundsTo": "grounds_to",
    "http://aicor.knowledge/l2-ontology.owl#halfOf": "half_of",
    "http://aicor.knowledge/l2-ontology.owl#hasAdmissibilityConstraint": "has_admissibility_constraint",
    "http://aicor.knowledge/l2-ontology.owl#hasAgentiveRole": "has_agentive_role",
    "http://aicor.knowledge/l2-ontology.owl#hasArrangementRole": "has_arrangement_role",
    "http://aicor.knowledge/l2-ontology.owl#hasAssessmentRegime": "has_assessment_regime",
    "http://aicor.knowledge/l2-ontology.owl#hasAssetRole": "has_asset_role",
    "http://aicor.knowledge/l2-ontology.owl#hasAuthority": "has_authority",
    "http://aicor.knowledge/l2-ontology.owl#hasBeliefState": "has_belief_state",
    "http://aicor.knowledge/l2-ontology.owl#hasBinding": "has_binding",
    "http://aicor.knowledge/l2-ontology.owl#hasBodyWorldFamily": "has_body_world_family",
    "http://aicor.knowledge/l2-ontology.owl#hasBodyWorldSystem": "has_body_world_system",
    "http://aicor.knowledge/l2-ontology.owl#hasCapability": "has_capability",
    "http://aicor.knowledge/l2-ontology.owl#hasCaseMember": "has_case_member",
    "http://aicor.knowledge/l2-ontology.owl#hasCausalStructure": "has_causal_structure",
    "http://aicor.knowledge/l2-ontology.owl#hasClaim": "has_claim",
    "http://aicor.knowledge/l2-ontology.owl#hasCodePath": "has_code_path",
    "http://aicor.knowledge/l2-ontology.owl#hasCodeTree": "has_code_tree",
    "http://aicor.knowledge/l2-ontology.owl#hasCodomain": "has_codomain",
    "http://aicor.knowledge/l2-ontology.owl#hasCommitmentRecord": "has_commitment_record",
    "http://aicor.knowledge/l2-ontology.owl#hasComponent": "has_component",
    "http://aicor.knowledge/l2-ontology.owl#hasConditionType": "has_condition_type",
    "http://aicor.knowledge/l2-ontology.owl#hasConstitutiveRole": "has_constitutive_role",
    "http://aicor.knowledge/l2-ontology.owl#hasContainerLayer": "has_container_layer",
    "http://aicor.knowledge/l2-ontology.owl#hasContinuousState": "has_continuous_state",
    "http://aicor.knowledge/l2-ontology.owl#hasCost": "has_cost",
    "http://aicor.knowledge/l2-ontology.owl#hasCoupling": "has_coupling",
    "http://aicor.knowledge/l2-ontology.owl#hasCouplingRelation": "has_coupling_relation",
    "http://aicor.knowledge/l2-ontology.owl#hasCouplingTerm": "has_coupling_term",
    "http://aicor.knowledge/l2-ontology.owl#hasCouplingType": "has_coupling_type",
    "http://aicor.knowledge/l2-ontology.owl#hasCoverageTarget": "has_coverage_target",
    "http://aicor.knowledge/l2-ontology.owl#hasCrackOrientation": "has_crack_orientation",
    "http://aicor.knowledge/l2-ontology.owl#hasCriticSet": "has_critic_set",
    "http://aicor.knowledge/l2-ontology.owl#hasDeclaredBoundary": "has_declared_boundary",
    "http://aicor.knowledge/l2-ontology.owl#hasDependency": "has_dependency",
    "http://aicor.knowledge/l2-ontology.owl#hasDependencyEdge": "has_dependency_edge",
    "http://aicor.knowledge/l2-ontology.owl#hasDependencyStructure": "has_dependency_structure",
    "http://aicor.knowledge/l2-ontology.owl#hasDestination": "has_destination",
    "http://aicor.knowledge/l2-ontology.owl#hasDifferential": "has_differential",
    "http://aicor.knowledge/l2-ontology.owl#hasDischargeAlternative": "has_discharge_alternative",
    "http://aicor.knowledge/l2-ontology.owl#hasDiscreteMode": "has_discrete_mode",
    "http://aicor.knowledge/l2-ontology.owl#hasDomain": "has_domain",
    "http://aicor.knowledge/l2-ontology.owl#hasElement": "has_element",
    "http://aicor.knowledge/l2-ontology.owl#hasEnd": "has_end",
    "http://aicor.knowledge/l2-ontology.owl#hasEndEffector": "has_end_effector",
    "http://aicor.knowledge/l2-ontology.owl#hasEpisode": "has_episode",
    "http://aicor.knowledge/l2-ontology.owl#hasEpistemicStatus": "has_epistemic_status",
    "http://aicor.knowledge/l2-ontology.owl#hasExecutiveFamily": "has_executive_family",
    "http://aicor.knowledge/l2-ontology.owl#hasExpectedObservation": "has_expected_observation",
    "http://aicor.knowledge/l2-ontology.owl#hasFDEvent": "has_fd_event",
    "http://aicor.knowledge/l2-ontology.owl#hasFDState": "has_fd_state",
    "http://aicor.knowledge/l2-ontology.owl#hasFailedTask": "has_failed_task",
    "http://aicor.knowledge/l2-ontology.owl#hasFailureId": "has_failure_id",
    "http://aicor.knowledge/l2-ontology.owl#hasFailureReason": "has_failure_reason",
    "http://aicor.knowledge/l2-ontology.owl#hasFallback": "has_fallback",
    "http://aicor.knowledge/l2-ontology.owl#hasFeedback": "has_feedback",
    "http://aicor.knowledge/l2-ontology.owl#hasField": "has_field",
    "http://aicor.knowledge/l2-ontology.owl#hasFilmThickness": "has_film_thickness",
    "http://aicor.knowledge/l2-ontology.owl#hasFirstLink": "has_first_link",
    "http://aicor.knowledge/l2-ontology.owl#hasForbiddenRegardless": "has_forbidden_regardless",
    "http://aicor.knowledge/l2-ontology.owl#hasForceProfile": "has_force_profile",
    "http://aicor.knowledge/l2-ontology.owl#hasForm": "has_form",
    "http://aicor.knowledge/l2-ontology.owl#hasFormat": "has_format",
    "http://aicor.knowledge/l2-ontology.owl#hasFrameElement": "has_frame_element",
    "http://aicor.knowledge/l2-ontology.owl#hasGoal": "has_goal",
    "http://aicor.knowledge/l2-ontology.owl#hasGoalClause": "has_goal_clause",
    "http://aicor.knowledge/l2-ontology.owl#hasHorizon": "has_horizon",
    "http://aicor.knowledge/l2-ontology.owl#hasInertiaTerm": "has_inertia_term",
    "http://aicor.knowledge/l2-ontology.owl#hasInertial": "has_inertial",
    "http://aicor.knowledge/l2-ontology.owl#hasInformationProcess": "has_information_process",
    "http://aicor.knowledge/l2-ontology.owl#hasInitialCondition": "has_initial_condition",
    "http://aicor.knowledge/l2-ontology.owl#hasIntentionalStructure": "has_intentional_structure",
    "http://aicor.knowledge/l2-ontology.owl#hasInterpretationModality": "has_interpretation_modality",
    "http://aicor.knowledge/l2-ontology.owl#hasJoint": "has_joint",
    "http://aicor.knowledge/l2-ontology.owl#hasJointElement": "has_joint_element",
    "http://aicor.knowledge/l2-ontology.owl#hasJointState": "has_joint_state",
    "http://aicor.knowledge/l2-ontology.owl#hasJointType": "has_joint_type",
    "http://aicor.knowledge/l2-ontology.owl#hasLifecycleEvents": "has_lifecycle_events",
    "http://aicor.knowledge/l2-ontology.owl#hasLink": "has_link",
    "http://aicor.knowledge/l2-ontology.owl#hasLinkElement": "has_link_element",
    "http://aicor.knowledge/l2-ontology.owl#hasMasteryCriterion": "has_mastery_criterion",
    "http://aicor.knowledge/l2-ontology.owl#hasMember": "has_member",
    "http://aicor.knowledge/l2-ontology.owl#hasModeSet": "has_mode_set",
    "http://aicor.knowledge/l2-ontology.owl#hasModelComponent": "has_model_component",
    "http://aicor.knowledge/l2-ontology.owl#hasMonitor": "has_monitor",
    "http://aicor.knowledge/l2-ontology.owl#hasMotionPhase": "has_motion_phase",
    "http://aicor.knowledge/l2-ontology.owl#hasNode": "has_node",
    "http://aicor.knowledge/l2-ontology.owl#hasNodeStructure": "has_node_structure",
    "http://aicor.knowledge/l2-ontology.owl#hasNuisanceCondition": "has_nuisance_condition",
    "http://aicor.knowledge/l2-ontology.owl#hasNuisanceKind": "has_nuisance_kind",
    "http://aicor.knowledge/l2-ontology.owl#hasNuisanceModel": "has_nuisance_model",
    "http://aicor.knowledge/l2-ontology.owl#hasObservationActionCoupling": "has_observation_action_coupling",
    "http://aicor.knowledge/l2-ontology.owl#hasObservationModel": "has_observation_model",
    "http://aicor.knowledge/l2-ontology.owl#hasOutcome": "has_outcome",
    "http://aicor.knowledge/l2-ontology.owl#hasParameter": "has_parameter",
    "http://aicor.knowledge/l2-ontology.owl#hasPart": "has_part",
    "http://aicor.knowledge/l2-ontology.owl#hasPartialDescription": "has_partial_description",
    "http://aicor.knowledge/l2-ontology.owl#hasParticipantRole": "has_participant_role",
    "http://aicor.knowledge/l2-ontology.owl#hasPayload": "has_payload",
    "http://aicor.knowledge/l2-ontology.owl#hasPerformer": "has_performer",
    "http://aicor.knowledge/l2-ontology.owl#hasPhase": "has_phase",
    "http://aicor.knowledge/l2-ontology.owl#hasPhaseGoal": "has_phase_goal",
    "http://aicor.knowledge/l2-ontology.owl#hasPhysicalCore": "has_physical_core",
    "http://aicor.knowledge/l2-ontology.owl#hasPhysicalProcess": "has_physical_process",
    "http://aicor.knowledge/l2-ontology.owl#hasPhysicsModel": "has_physics_model",
    "http://aicor.knowledge/l2-ontology.owl#hasPlaceProperty": "has_place_property",
    "http://aicor.knowledge/l2-ontology.owl#hasPlacementRegion": "has_placement_region",
    "http://aicor.knowledge/l2-ontology.owl#hasPolicy": "has_policy",
    "http://aicor.knowledge/l2-ontology.owl#hasPolicyTask": "has_policy_task",
    "http://aicor.knowledge/l2-ontology.owl#hasPortion": "has_portion",
    "http://aicor.knowledge/l2-ontology.owl#hasPose": "has_pose",
    "http://aicor.knowledge/l2-ontology.owl#hasPrecondition": "has_precondition",
    "http://aicor.knowledge/l2-ontology.owl#hasPredicate": "has_predicate",
    "http://aicor.knowledge/l2-ontology.owl#hasPredictor": "has_predictor",
    "http://aicor.knowledge/l2-ontology.owl#hasPreference": "has_preference",
    "http://aicor.knowledge/l2-ontology.owl#hasPrimaryActivity": "has_primary_activity",
    "http://aicor.knowledge/l2-ontology.owl#hasPrimaryTask": "has_primary_task",
    "http://aicor.knowledge/l2-ontology.owl#hasProblem": "has_problem",
    "http://aicor.knowledge/l2-ontology.owl#hasProblemFamily": "has_problem_family",
    "http://aicor.knowledge/l2-ontology.owl#hasProcess": "has_process",
    "http://aicor.knowledge/l2-ontology.owl#hasProcesses": "has_processes",
    "http://aicor.knowledge/l2-ontology.owl#hasProperPart": "has_proper_part",
    "http://aicor.knowledge/l2-ontology.owl#hasProperty": "has_property",
    "http://aicor.knowledge/l2-ontology.owl#hasQualityFunctional": "has_quality_functional",
    "http://aicor.knowledge/l2-ontology.owl#hasQueryVariable": "has_query_variable",
    "http://aicor.knowledge/l2-ontology.owl#hasQuestion": "has_question",
    "http://aicor.knowledge/l2-ontology.owl#hasRWMS": "has_rwms",
    "http://aicor.knowledge/l2-ontology.owl#hasRWMSFamily": "has_rwms_family",
    "http://aicor.knowledge/l2-ontology.owl#hasRecord": "has_record",
    "http://aicor.knowledge/l2-ontology.owl#hasRecording": "has_recording",
    "http://aicor.knowledge/l2-ontology.owl#hasRelationEdge": "has_relation_edge",
    "http://aicor.knowledge/l2-ontology.owl#hasRelationFamily": "has_relation_family",
    "http://aicor.knowledge/l2-ontology.owl#hasRelationalGrammar": "has_relational_grammar",
    "http://aicor.knowledge/l2-ontology.owl#hasRepresentation": "has_representation",
    "http://aicor.knowledge/l2-ontology.owl#hasRequest": "has_request",
    "http://aicor.knowledge/l2-ontology.owl#hasRequestConstraint": "has_request_constraint",
    "http://aicor.knowledge/l2-ontology.owl#hasRequestScope": "has_request_scope",
    "http://aicor.knowledge/l2-ontology.owl#hasResolution": "has_resolution",
    "http://aicor.knowledge/l2-ontology.owl#hasResolutionMode": "has_resolution_mode",
    "http://aicor.knowledge/l2-ontology.owl#hasResponsePolicy": "has_response_policy",
    "http://aicor.knowledge/l2-ontology.owl#hasRobotSubtree": "has_robot_subtree",
    "http://aicor.knowledge/l2-ontology.owl#hasRoleAssignmentLayer": "has_role_assignment_layer",
    "http://aicor.knowledge/l2-ontology.owl#hasRoleSlot": "has_role_slot",
    "http://aicor.knowledge/l2-ontology.owl#hasRung": "has_rung",
    "http://aicor.knowledge/l2-ontology.owl#hasRungFailureMode": "has_rung_failure_mode",
    "http://aicor.knowledge/l2-ontology.owl#hasScope": "has_scope",
    "http://aicor.knowledge/l2-ontology.owl#hasScopeCard": "has_scope_card",
    "http://aicor.knowledge/l2-ontology.owl#hasScopeCondition": "has_scope_condition",
    "http://aicor.knowledge/l2-ontology.owl#hasSensor": "has_sensor",
    "http://aicor.knowledge/l2-ontology.owl#hasSeverity": "has_severity",
    "http://aicor.knowledge/l2-ontology.owl#hasSolutionCardinality": "has_solution_cardinality",
    "http://aicor.knowledge/l2-ontology.owl#hasSolvabilityStructure": "has_solvability_structure",
    "http://aicor.knowledge/l2-ontology.owl#hasSpatialRelation": "has_spatial_relation",
    "http://aicor.knowledge/l2-ontology.owl#hasSpreadRadius": "has_spread_radius",
    "http://aicor.knowledge/l2-ontology.owl#hasStage": "has_stage",
    "http://aicor.knowledge/l2-ontology.owl#hasStart": "has_start",
    "http://aicor.knowledge/l2-ontology.owl#hasStateEstimate": "has_state_estimate",
    "http://aicor.knowledge/l2-ontology.owl#hasStateOfUnderstanding": "has_state_of_understanding",
    "http://aicor.knowledge/l2-ontology.owl#hasStatus": "has_status",
    "http://aicor.knowledge/l2-ontology.owl#hasStep": "has_step",
    "http://aicor.knowledge/l2-ontology.owl#hasStructuralCore": "has_structural_core",
    "http://aicor.knowledge/l2-ontology.owl#hasSubaction": "has_subaction",
    "http://aicor.knowledge/l2-ontology.owl#hasSubstrate": "has_substrate",
    "http://aicor.knowledge/l2-ontology.owl#hasSubtask": "has_subtask",
    "http://aicor.knowledge/l2-ontology.owl#hasSuccessorDistribution": "has_successor_distribution",
    "http://aicor.knowledge/l2-ontology.owl#hasSupport": "has_support",
    "http://aicor.knowledge/l2-ontology.owl#hasSystem1": "has_system1",
    "http://aicor.knowledge/l2-ontology.owl#hasSystem2": "has_system2",
    "http://aicor.knowledge/l2-ontology.owl#hasTagTable": "has_tag_table",
    "http://aicor.knowledge/l2-ontology.owl#hasTask": "has_task",
    "http://aicor.knowledge/l2-ontology.owl#hasTaskGenerator": "has_task_generator",
    "http://aicor.knowledge/l2-ontology.owl#hasTaskReading": "has_task_reading",
    "http://aicor.knowledge/l2-ontology.owl#hasTaskScope": "has_task_scope",
    "http://aicor.knowledge/l2-ontology.owl#hasTaskSource": "has_task_source",
    "http://aicor.knowledge/l2-ontology.owl#hasTaskSpecification": "has_task_specification",
    "http://aicor.knowledge/l2-ontology.owl#hasTaskStatus": "has_task_status",
    "http://aicor.knowledge/l2-ontology.owl#hasTelicRole": "has_telic_role",
    "http://aicor.knowledge/l2-ontology.owl#hasTelos": "has_telos",
    "http://aicor.knowledge/l2-ontology.owl#hasTemplate": "has_template",
    "http://aicor.knowledge/l2-ontology.owl#hasTerminalDisposition": "has_terminal_disposition",
    "http://aicor.knowledge/l2-ontology.owl#hasTerminalState": "has_terminal_state",
    "http://aicor.knowledge/l2-ontology.owl#hasTopTask": "has_top_task",
    "http://aicor.knowledge/l2-ontology.owl#hasTrace": "has_trace",
    "http://aicor.knowledge/l2-ontology.owl#hasTrajectorySpace": "has_trajectory_space",
    "http://aicor.knowledge/l2-ontology.owl#hasTranslationInterface": "has_translation_interface",
    "http://aicor.knowledge/l2-ontology.owl#hasTriple": "has_triple",
    "http://aicor.knowledge/l2-ontology.owl#hasTriplePattern": "has_triple_pattern",
    "http://aicor.knowledge/l2-ontology.owl#hasTypeConstraint": "has_type_constraint",
    "http://aicor.knowledge/l2-ontology.owl#hasValidityCondition": "has_validity_condition",
    "http://aicor.knowledge/l2-ontology.owl#hasValidityInterval": "has_validity_interval",
    "http://aicor.knowledge/l2-ontology.owl#hasValidityWindow": "has_validity_window",
    "http://aicor.knowledge/l2-ontology.owl#hasVariant": "has_variant",
    "http://aicor.knowledge/l2-ontology.owl#hasVerdict": "has_verdict",
    "http://aicor.knowledge/l2-ontology.owl#hasVerdictComponent": "has_verdict_component",
    "http://aicor.knowledge/l2-ontology.owl#hasVocabulary": "has_vocabulary",
    "http://aicor.knowledge/l2-ontology.owl#heldBy": "held_by",
    "http://aicor.knowledge/l2-ontology.owl#heldByComponent": "held_by_component",
    "http://aicor.knowledge/l2-ontology.owl#holdsInterval": "holds_interval",
    "http://aicor.knowledge/l2-ontology.owl#holdsOver": "holds_over",
    "http://aicor.knowledge/l2-ontology.owl#hostedOn": "hosted_on",
    "http://aicor.knowledge/l2-ontology.owl#hosts": "hosts",
    "http://aicor.knowledge/l2-ontology.owl#identifies": "identifies",
    "http://aicor.knowledge/l2-ontology.owl#implementationOf": "implementation_of",
    "http://aicor.knowledge/l2-ontology.owl#improvesModel": "improves_model",
    "http://aicor.knowledge/l2-ontology.owl#inVessel": "in_vessel",
    "http://aicor.knowledge/l2-ontology.owl#includes": "includes",
    "http://aicor.knowledge/l2-ontology.owl#includesEnd": "includes_end",
    "http://aicor.knowledge/l2-ontology.owl#includesStart": "includes_start",
    "http://aicor.knowledge/l2-ontology.owl#incursObligation": "incurs_obligation",
    "http://aicor.knowledge/l2-ontology.owl#indexedBy": "indexed_by",
    "http://aicor.knowledge/l2-ontology.owl#indexes": "indexes",
    "http://aicor.knowledge/l2-ontology.owl#individuates": "individuates",
    "http://aicor.knowledge/l2-ontology.owl#infers": "infers",
    "http://aicor.knowledge/l2-ontology.owl#initiatedByEvent": "initiated_by_event",
    "http://aicor.knowledge/l2-ontology.owl#initiatesStateType": "initiates_state_type",
    "http://aicor.knowledge/l2-ontology.owl#insertedInto": "inserted_into",
    "http://aicor.knowledge/l2-ontology.owl#inspectableBy": "inspectable_by",
    "http://aicor.knowledge/l2-ontology.owl#instantiates": "instantiates",
    "http://aicor.knowledge/l2-ontology.owl#instantiatesFamily": "instantiates_family",
    "http://aicor.knowledge/l2-ontology.owl#integratedInto": "integrated_into",
    "http://aicor.knowledge/l2-ontology.owl#integrates": "integrates",
    "http://aicor.knowledge/l2-ontology.owl#interactsAmong": "interacts_among",
    "http://aicor.knowledge/l2-ontology.owl#interfaceOf": "interface_of",
    "http://aicor.knowledge/l2-ontology.owl#interpretedBy": "interpreted_by",
    "http://aicor.knowledge/l2-ontology.owl#interprets": "interprets",
    "http://aicor.knowledge/l2-ontology.owl#interpretsVia": "interprets_via",
    "http://aicor.knowledge/l2-ontology.owl#intoProblemLanguage": "into_problem_language",
    "http://aicor.knowledge/l2-ontology.owl#inverseOf": "inverse_of",
    "http://aicor.knowledge/l2-ontology.owl#investigatesSolutionOf": "investigates_solution_of",
    "http://aicor.knowledge/l2-ontology.owl#involvesContact": "involves_contact",
    "http://aicor.knowledge/l2-ontology.owl#isAbout": "is_about",
    "http://aicor.knowledge/l2-ontology.owl#isCausallySufficientUnder": "is_causally_sufficient_under",
    "http://aicor.knowledge/l2-ontology.owl#isCompatibleWith": "is_compatible_with",
    "http://aicor.knowledge/l2-ontology.owl#isDualWith": "is_dual_with",
    "http://aicor.knowledge/l2-ontology.owl#isModelOf": "is_model_of",
    "http://aicor.knowledge/l2-ontology.owl#isPerformableBy": "is_performable_by",
    "http://aicor.knowledge/l2-ontology.owl#isResultOf": "is_result_of",
    "http://aicor.knowledge/l2-ontology.owl#isRunOf": "is_run_of",
    "http://aicor.knowledge/l2-ontology.owl#isTransitionOf": "is_transition_of",
    "http://aicor.knowledge/l2-ontology.owl#isWithinScopeOf": "is_within_scope_of",
    "http://aicor.knowledge/l2-ontology.owl#issues": "issues",
    "http://aicor.knowledge/l2-ontology.owl#judgesSystem": "judges_system",
    "http://aicor.knowledge/l2-ontology.owl#learnsFrom": "learns_from",
    "http://aicor.knowledge/l2-ontology.owl#licensesDisturbanceOf": "licenses_disturbance_of",
    "http://aicor.knowledge/l2-ontology.owl#linksPrimTo": "links_prim_to",
    "http://aicor.knowledge/l2-ontology.owl#listensTo": "listens_to",
    "http://aicor.knowledge/l2-ontology.owl#loads": "loads",
    "http://aicor.knowledge/l2-ontology.owl#localModelPerMode": "local_model_per_mode",
    "http://aicor.knowledge/l2-ontology.owl#logsTo": "logs_to",
    "http://aicor.knowledge/l2-ontology.owl#lowerEdge": "lower_edge",
    "http://aicor.knowledge/l2-ontology.owl#maintains": "maintains",
    "http://aicor.knowledge/l2-ontology.owl#maintainsConditionFor": "maintains_condition_for",
    "http://aicor.knowledge/l2-ontology.owl#makesAvailable": "makes_available",
    "http://aicor.knowledge/l2-ontology.owl#makesConcrete": "makes_concrete",
    "http://aicor.knowledge/l2-ontology.owl#makesFamilyRelative": "makes_family_relative",
    "http://aicor.knowledge/l2-ontology.owl#makesStatable": "makes_statable",
    "http://aicor.knowledge/l2-ontology.owl#manipulatedAtRuntimeBy": "manipulated_at_runtime_by",
    "http://aicor.knowledge/l2-ontology.owl#mapsFrom": "maps_from",
    "http://aicor.knowledge/l2-ontology.owl#mapsOutcomeTo": "maps_outcome_to",
    "http://aicor.knowledge/l2-ontology.owl#mapsRequest": "maps_request",
    "http://aicor.knowledge/l2-ontology.owl#mapsToRung": "maps_to_rung",
    "http://aicor.knowledge/l2-ontology.owl#marksTransitionTo": "marks_transition_to",
    "http://aicor.knowledge/l2-ontology.owl#matchedOver": "matched_over",
    "http://aicor.knowledge/l2-ontology.owl#maturesIntoTheorem": "matures_into_theorem",
    "http://aicor.knowledge/l2-ontology.owl#maximizes": "maximizes",
    "http://aicor.knowledge/l2-ontology.owl#mayCall": "may_call",
    "http://aicor.knowledge/l2-ontology.owl#mechanismInput": "mechanism_input",
    "http://aicor.knowledge/l2-ontology.owl#mechanismOutput": "mechanism_output",
    "http://aicor.knowledge/l2-ontology.owl#mechanismRealizedBy": "mechanism_realized_by",
    "http://aicor.knowledge/l2-ontology.owl#meets": "meets",
    "http://aicor.knowledge/l2-ontology.owl#memberOf": "member_of",
    "http://aicor.knowledge/l2-ontology.owl#minimizes": "minimizes",
    "http://aicor.knowledge/l2-ontology.owl#mirrors": "mirrors",
    "http://aicor.knowledge/l2-ontology.owl#modelOf": "model_of",
    "http://aicor.knowledge/l2-ontology.owl#modelsChannel": "models_channel",
    "http://aicor.knowledge/l2-ontology.owl#modelsStructureOf": "models_structure_of",
    "http://aicor.knowledge/l2-ontology.owl#modelsWorld": "models_world",
    "http://aicor.knowledge/l2-ontology.owl#monitorFor": "monitor_for",
    "http://aicor.knowledge/l2-ontology.owl#monitors": "monitors",
    "http://aicor.knowledge/l2-ontology.owl#monitorsPhaseOf": "monitors_phase_of",
    "http://aicor.knowledge/l2-ontology.owl#mountedOn": "mounted_on",
    "http://aicor.knowledge/l2-ontology.owl#movesBody": "moves_body",
    "http://aicor.knowledge/l2-ontology.owl#nextLink": "next_link",
    "http://aicor.knowledge/l2-ontology.owl#obligates": "obligates",
    "http://aicor.knowledge/l2-ontology.owl#occupiesInterval": "occupies_interval",
    "http://aicor.knowledge/l2-ontology.owl#occursOver": "occurs_over",
    "http://aicor.knowledge/l2-ontology.owl#ofEventType": "of_event_type",
    "http://aicor.knowledge/l2-ontology.owl#ofFluentType": "of_fluent_type",
    "http://aicor.knowledge/l2-ontology.owl#offersMethod": "offers_method",
    "http://aicor.knowledge/l2-ontology.owl#onFlatObject": "on_flat_object",
    "http://aicor.knowledge/l2-ontology.owl#operatesOn": "operates_on",
    "http://aicor.knowledge/l2-ontology.owl#operationalizedBy": "operationalized_by",
    "http://aicor.knowledge/l2-ontology.owl#operationalizes": "operationalizes",
    "http://aicor.knowledge/l2-ontology.owl#optimalDesign": "optimal_design",
    "http://aicor.knowledge/l2-ontology.owl#optimalOverFamily": "optimal_over_family",
    "http://aicor.knowledge/l2-ontology.owl#optimizes": "optimizes",
    "http://aicor.knowledge/l2-ontology.owl#organizedBy": "organized_by",
    "http://aicor.knowledge/l2-ontology.owl#organizesControlOf": "organizes_control_of",
    "http://aicor.knowledge/l2-ontology.owl#outputSupports": "output_supports",
    "http://aicor.knowledge/l2-ontology.owl#outputsVia": "outputs_via",
    "http://aicor.knowledge/l2-ontology.owl#overExecutivesOf": "over_executives_of",
    "http://aicor.knowledge/l2-ontology.owl#overInterval": "over_interval",
    "http://aicor.knowledge/l2-ontology.owl#overParameterSpace": "over_parameter_space",
    "http://aicor.knowledge/l2-ontology.owl#overProblemClass": "over_problem_class",
    "http://aicor.knowledge/l2-ontology.owl#overProblemFamily": "over_problem_family",
    "http://aicor.knowledge/l2-ontology.owl#overSystem": "over_system",
    "http://aicor.knowledge/l2-ontology.owl#packages": "packages",
    "http://aicor.knowledge/l2-ontology.owl#pairsConceptFamily": "pairs_concept_family",
    "http://aicor.knowledge/l2-ontology.owl#parameterizedBy": "parameterized_by",
    "http://aicor.knowledge/l2-ontology.owl#partOf": "part_of",
    "http://aicor.knowledge/l2-ontology.owl#partOfScientificObject": "part_of_scientific_object",
    "http://aicor.knowledge/l2-ontology.owl#partitionedInto": "partitioned_into",
    "http://aicor.knowledge/l2-ontology.owl#partitionsInto": "partitions_into",
    "http://aicor.knowledge/l2-ontology.owl#patternIn": "pattern_in",
    "http://aicor.knowledge/l2-ontology.owl#perceives": "perceives",
    "http://aicor.knowledge/l2-ontology.owl#performsDesignator": "performs_designator",
    "http://aicor.knowledge/l2-ontology.owl#physicallyTypedBy": "physically_typed_by",
    "http://aicor.knowledge/l2-ontology.owl#pipelineOf": "pipeline_of",
    "http://aicor.knowledge/l2-ontology.owl#pivotsOn": "pivots_on",
    "http://aicor.knowledge/l2-ontology.owl#posteriorSchema": "posterior_schema",
    "http://aicor.knowledge/l2-ontology.owl#predictsCompetence": "predicts_competence",
    "http://aicor.knowledge/l2-ontology.owl#predictsDegreeTrend": "predicts_degree_trend",
    "http://aicor.knowledge/l2-ontology.owl#predictsEvolutionOf": "predicts_evolution_of",
    "http://aicor.knowledge/l2-ontology.owl#predictsRequiredCapability": "predicts_required_capability",
    "http://aicor.knowledge/l2-ontology.owl#predictsTransferTo": "predicts_transfer_to",
    "http://aicor.knowledge/l2-ontology.owl#preferredConfiguration": "preferred_configuration",
    "http://aicor.knowledge/l2-ontology.owl#prescribesDestination": "prescribes_destination",
    "http://aicor.knowledge/l2-ontology.owl#prescribesPortion": "prescribes_portion",
    "http://aicor.knowledge/l2-ontology.owl#prescribesRelation": "prescribes_relation",
    "http://aicor.knowledge/l2-ontology.owl#prescribesVessel": "prescribes_vessel",
    "http://aicor.knowledge/l2-ontology.owl#presupposes": "presupposes",
    "http://aicor.knowledge/l2-ontology.owl#priorSchema": "prior_schema",
    "http://aicor.knowledge/l2-ontology.owl#producedBy": "produced_by",
    "http://aicor.knowledge/l2-ontology.owl#producedByEnvisioning": "produced_by_envisioning",
    "http://aicor.knowledge/l2-ontology.owl#producedFrom": "produced_from",
    "http://aicor.knowledge/l2-ontology.owl#produces": "produces",
    "http://aicor.knowledge/l2-ontology.owl#producesConfiguration": "produces_configuration",
    "http://aicor.knowledge/l2-ontology.owl#producesDescription": "produces_description",
    "http://aicor.knowledge/l2-ontology.owl#producesEffect": "produces_effect",
    "http://aicor.knowledge/l2-ontology.owl#producesEpisode": "produces_episode",
    "http://aicor.knowledge/l2-ontology.owl#producesInput": "produces_input",
    "http://aicor.knowledge/l2-ontology.owl#producesObservation": "produces_observation",
    "http://aicor.knowledge/l2-ontology.owl#producesRecordIn": "produces_record_in",
    "http://aicor.knowledge/l2-ontology.owl#producesSegment": "produces_segment",
    "http://aicor.knowledge/l2-ontology.owl#producesSolutionFor": "produces_solution_for",
    "http://aicor.knowledge/l2-ontology.owl#producesSynthetic": "produces_synthetic",
    "http://aicor.knowledge/l2-ontology.owl#producesTarget": "produces_target",
    "http://aicor.knowledge/l2-ontology.owl#producesTimeline": "produces_timeline",
    "http://aicor.knowledge/l2-ontology.owl#producesVerdict": "produces_verdict",
    "http://aicor.knowledge/l2-ontology.owl#prohibitsState": "prohibits_state",
    "http://aicor.knowledge/l2-ontology.owl#projectionOfBranch": "projection_of_branch",
    "http://aicor.knowledge/l2-ontology.owl#projects": "projects",
    "http://aicor.knowledge/l2-ontology.owl#projectsFrom": "projects_from",
    "http://aicor.knowledge/l2-ontology.owl#proposes": "proposes",
    "http://aicor.knowledge/l2-ontology.owl#proposesAmendmentOf": "proposes_amendment_of",
    "http://aicor.knowledge/l2-ontology.owl#provides": "provides",
    "http://aicor.knowledge/l2-ontology.owl#providesDOF": "provides_dof",
    "http://aicor.knowledge/l2-ontology.owl#providesImplementation": "provides_implementation",
    "http://aicor.knowledge/l2-ontology.owl#queries": "queries",
    "http://aicor.knowledge/l2-ontology.owl#queriesEngine": "queries_engine",
    "http://aicor.knowledge/l2-ontology.owl#queriesModel": "queries_model",
    "http://aicor.knowledge/l2-ontology.owl#queriesModelOf": "queries_model_of",
    "http://aicor.knowledge/l2-ontology.owl#queriesOver": "queries_over",
    "http://aicor.knowledge/l2-ontology.owl#raisedBy": "raised_by",
    "http://aicor.knowledge/l2-ontology.owl#rangesOver": "ranges_over",
    "http://aicor.knowledge/l2-ontology.owl#rangesOverFamily": "ranges_over_family",
    "http://aicor.knowledge/l2-ontology.owl#rangesOverProjection": "ranges_over_projection",
    "http://aicor.knowledge/l2-ontology.owl#rankedByNorm": "ranked_by_norm",
    "http://aicor.knowledge/l2-ontology.owl#reads": "reads",
    "http://aicor.knowledge/l2-ontology.owl#readsAspect": "reads_aspect",
    "http://aicor.knowledge/l2-ontology.owl#readsState": "reads_state",
    "http://aicor.knowledge/l2-ontology.owl#realizationOf": "realization_of",
    "http://aicor.knowledge/l2-ontology.owl#realizedAs": "realized_as",
    "http://aicor.knowledge/l2-ontology.owl#realizedBy": "realized_by",
    "http://aicor.knowledge/l2-ontology.owl#realizedByCategory": "realized_by_category",
    "http://aicor.knowledge/l2-ontology.owl#realizes": "realizes",
    "http://aicor.knowledge/l2-ontology.owl#realizesCommitment": "realizes_commitment",
    "http://aicor.knowledge/l2-ontology.owl#realizesDesignator": "realizes_designator",
    "http://aicor.knowledge/l2-ontology.owl#realizesIntention": "realizes_intention",
    "http://aicor.knowledge/l2-ontology.owl#realizesInteraction": "realizes_interaction",
    "http://aicor.knowledge/l2-ontology.owl#realizesModel": "realizes_model",
    "http://aicor.knowledge/l2-ontology.owl#realizesMotion": "realizes_motion",
    "http://aicor.knowledge/l2-ontology.owl#realizesVia": "realizes_via",
    "http://aicor.knowledge/l2-ontology.owl#reasonedAboutBy": "reasoned_about_by",
    "http://aicor.knowledge/l2-ontology.owl#reasonsAbout": "reasons_about",
    "http://aicor.knowledge/l2-ontology.owl#reasonsOver": "reasons_over",
    "http://aicor.knowledge/l2-ontology.owl#reasonsWith": "reasons_with",
    "http://aicor.knowledge/l2-ontology.owl#reconstructedByModel": "reconstructed_by_model",
    "http://aicor.knowledge/l2-ontology.owl#reconstructsPhase": "reconstructs_phase",
    "http://aicor.knowledge/l2-ontology.owl#records": "records",
    "http://aicor.knowledge/l2-ontology.owl#recordsBinding": "records_binding",
    "http://aicor.knowledge/l2-ontology.owl#recordsCommitment": "records_commitment",
    "http://aicor.knowledge/l2-ontology.owl#recordsGroundingAndChecks": "records_grounding_and_checks",
    "http://aicor.knowledge/l2-ontology.owl#recordsMonitorAndRepair": "records_monitor_and_repair",
    "http://aicor.knowledge/l2-ontology.owl#recordsStatus": "records_status",
    "http://aicor.knowledge/l2-ontology.owl#recordsSystem1Output": "records_system1_output",
    "http://aicor.knowledge/l2-ontology.owl#recordsTranslation": "records_translation",
    "http://aicor.knowledge/l2-ontology.owl#referenceOutcome": "reference_outcome",
    "http://aicor.knowledge/l2-ontology.owl#referenceRidesOn": "reference_rides_on",
    "http://aicor.knowledge/l2-ontology.owl#references": "references",
    "http://aicor.knowledge/l2-ontology.owl#referencesAsset": "references_asset",
    "http://aicor.knowledge/l2-ontology.owl#refinesRepresentation": "refines_representation",
    "http://aicor.knowledge/l2-ontology.owl#refutesHypothesis": "refutes_hypothesis",
    "http://aicor.knowledge/l2-ontology.owl#reifies": "reifies",
    "http://aicor.knowledge/l2-ontology.owl#relatesToDescription": "relates_to_description",
    "http://aicor.knowledge/l2-ontology.owl#relatesTorqueToAcceleration": "relates_torque_to_acceleration",
    "http://aicor.knowledge/l2-ontology.owl#relativeTo": "relative_to",
    "http://aicor.knowledge/l2-ontology.owl#relativeToProblem": "relative_to_problem",
    "http://aicor.knowledge/l2-ontology.owl#relativeToProblemFamily": "relative_to_problem_family",
    "http://aicor.knowledge/l2-ontology.owl#relativeToPurpose": "relative_to_purpose",
    "http://aicor.knowledge/l2-ontology.owl#relativeToScenario": "relative_to_scenario",
    "http://aicor.knowledge/l2-ontology.owl#renders": "renders",
    "http://aicor.knowledge/l2-ontology.owl#rendersOccasion": "renders_occasion",
    "http://aicor.knowledge/l2-ontology.owl#rendersTractable": "renders_tractable",
    "http://aicor.knowledge/l2-ontology.owl#rendersVia": "renders_via",
    "http://aicor.knowledge/l2-ontology.owl#representsForExecutive": "represents_for_executive",
    "http://aicor.knowledge/l2-ontology.owl#representsSelectedAspectsOf": "represents_selected_aspects_of",
    "http://aicor.knowledge/l2-ontology.owl#representsSelfStateOf": "represents_self_state_of",
    "http://aicor.knowledge/l2-ontology.owl#requestsObjectScope": "requests_object_scope",
    "http://aicor.knowledge/l2-ontology.owl#requiredForAccomplishing": "required_for_accomplishing",
    "http://aicor.knowledge/l2-ontology.owl#requires": "requires",
    "http://aicor.knowledge/l2-ontology.owl#requiresBeforeCommit": "requires_before_commit",
    "http://aicor.knowledge/l2-ontology.owl#requiresCounterpart": "requires_counterpart",
    "http://aicor.knowledge/l2-ontology.owl#requiresDeclaredScope": "requires_declared_scope",
    "http://aicor.knowledge/l2-ontology.owl#requiresIngredient": "requires_ingredient",
    "http://aicor.knowledge/l2-ontology.owl#requiresMasteryCriterion": "requires_mastery_criterion",
    "http://aicor.knowledge/l2-ontology.owl#requiresMethodology": "requires_methodology",
    "http://aicor.knowledge/l2-ontology.owl#requiresProperty": "requires_property",
    "http://aicor.knowledge/l2-ontology.owl#requiresRelation": "requires_relation",
    "http://aicor.knowledge/l2-ontology.owl#reservedFor": "reserved_for",
    "http://aicor.knowledge/l2-ontology.owl#resolvedAgainst": "resolved_against",
    "http://aicor.knowledge/l2-ontology.owl#resolvedAgainstBy": "resolved_against_by",
    "http://aicor.knowledge/l2-ontology.owl#resolves": "resolves",
    "http://aicor.knowledge/l2-ontology.owl#resolvesContactBy": "resolves_contact_by",
    "http://aicor.knowledge/l2-ontology.owl#resolvesDesignator": "resolves_designator",
    "http://aicor.knowledge/l2-ontology.owl#resolvesEach": "resolves_each",
    "http://aicor.knowledge/l2-ontology.owl#resolvesRequest": "resolves_request",
    "http://aicor.knowledge/l2-ontology.owl#resolvesTo": "resolves_to",
    "http://aicor.knowledge/l2-ontology.owl#resolvesToState": "resolves_to_state",
    "http://aicor.knowledge/l2-ontology.owl#resolvesVia": "resolves_via",
    "http://aicor.knowledge/l2-ontology.owl#resourceFor": "resource_for",
    "http://aicor.knowledge/l2-ontology.owl#respondsTo": "responds_to",
    "http://aicor.knowledge/l2-ontology.owl#resultObtainedIn": "result_obtained_in",
    "http://aicor.knowledge/l2-ontology.owl#revisesVia": "revises_via",
    "http://aicor.knowledge/l2-ontology.owl#ridesOn": "rides_on",
    "http://aicor.knowledge/l2-ontology.owl#rollsOut": "rolls_out",
    "http://aicor.knowledge/l2-ontology.owl#rungOf": "rung_of",
    "http://aicor.knowledge/l2-ontology.owl#runs": "runs",
    "http://aicor.knowledge/l2-ontology.owl#runsScheduler": "runs_scheduler",
    "http://aicor.knowledge/l2-ontology.owl#sampledFrom": "sampled_from",
    "http://aicor.knowledge/l2-ontology.owl#satisfiedBy": "satisfied_by",
    "http://aicor.knowledge/l2-ontology.owl#satisfiedByEvent": "satisfied_by_event",
    "http://aicor.knowledge/l2-ontology.owl#satisfies": "satisfies",
    "http://aicor.knowledge/l2-ontology.owl#satisfiesRequest": "satisfies_request",
    "http://aicor.knowledge/l2-ontology.owl#schedules": "schedules",
    "http://aicor.knowledge/l2-ontology.owl#schemaOf": "schema_of",
    "http://aicor.knowledge/l2-ontology.owl#scopedBy": "scoped_by",
    "http://aicor.knowledge/l2-ontology.owl#scopedByPurpose": "scoped_by_purpose",
    "http://aicor.knowledge/l2-ontology.owl#scopedToClass": "scoped_to_class",
    "http://aicor.knowledge/l2-ontology.owl#scopesModel": "scopes_model",
    "http://aicor.knowledge/l2-ontology.owl#scopesProjection": "scopes_projection",
    "http://aicor.knowledge/l2-ontology.owl#segmentOf": "segment_of",
    "http://aicor.knowledge/l2-ontology.owl#selectsByMinRisk": "selects_by_min_risk",
    "http://aicor.knowledge/l2-ontology.owl#serializes": "serializes",
    "http://aicor.knowledge/l2-ontology.owl#servesActivity": "serves_activity",
    "http://aicor.knowledge/l2-ontology.owl#servesPurpose": "serves_purpose",
    "http://aicor.knowledge/l2-ontology.owl#signals": "signals",
    "http://aicor.knowledge/l2-ontology.owl#signalsDescription": "signals_description",
    "http://aicor.knowledge/l2-ontology.owl#simulatableUnder": "simulatable_under",
    "http://aicor.knowledge/l2-ontology.owl#simulatesOver": "simulates_over",
    "http://aicor.knowledge/l2-ontology.owl#simulatesVia": "simulates_via",
    "http://aicor.knowledge/l2-ontology.owl#solvesFor": "solves_for",
    "http://aicor.knowledge/l2-ontology.owl#specializesVia": "specializes_via",
    "http://aicor.knowledge/l2-ontology.owl#specifiedBy": "specified_by",
    "http://aicor.knowledge/l2-ontology.owl#specifies": "specifies",
    "http://aicor.knowledge/l2-ontology.owl#specifiesProcess": "specifies_process",
    "http://aicor.knowledge/l2-ontology.owl#standardOverFamily": "standard_over_family",
    "http://aicor.knowledge/l2-ontology.owl#steers": "steers",
    "http://aicor.knowledge/l2-ontology.owl#strengthens": "strengthens",
    "http://aicor.knowledge/l2-ontology.owl#structureOf": "structure_of",
    "http://aicor.knowledge/l2-ontology.owl#structures": "structures",
    "http://aicor.knowledge/l2-ontology.owl#studiedAs": "studied_as",
    "http://aicor.knowledge/l2-ontology.owl#studiedBy": "studied_by",
    "http://aicor.knowledge/l2-ontology.owl#studiesObject": "studies_object",
    "http://aicor.knowledge/l2-ontology.owl#subjectTo": "subject_to",
    "http://aicor.knowledge/l2-ontology.owl#successConstrainedBy": "success_constrained_by",
    "http://aicor.knowledge/l2-ontology.owl#suggestsHypothesis": "suggests_hypothesis",
    "http://aicor.knowledge/l2-ontology.owl#supports": "supports",
    "http://aicor.knowledge/l2-ontology.owl#supportsNesting": "supports_nesting",
    "http://aicor.knowledge/l2-ontology.owl#supportsOperation": "supports_operation",
    "http://aicor.knowledge/l2-ontology.owl#supportsRegime": "supports_regime",
    "http://aicor.knowledge/l2-ontology.owl#surfaces": "surfaces",
    "http://aicor.knowledge/l2-ontology.owl#synchronizedWith": "synchronized_with",
    "http://aicor.knowledge/l2-ontology.owl#takesProposal": "takes_proposal",
    "http://aicor.knowledge/l2-ontology.owl#targetPart": "target_part",
    "http://aicor.knowledge/l2-ontology.owl#templateFor": "template_for",
    "http://aicor.knowledge/l2-ontology.owl#templateOf": "template_of",
    "http://aicor.knowledge/l2-ontology.owl#tests": "tests",
    "http://aicor.knowledge/l2-ontology.owl#testsHypothesis": "tests_hypothesis",
    "http://aicor.knowledge/l2-ontology.owl#toBodyMotion": "to_body_motion",
    "http://aicor.knowledge/l2-ontology.owl#toStatus": "to_status",
    "http://aicor.knowledge/l2-ontology.owl#toValue": "to_value",
    "http://aicor.knowledge/l2-ontology.owl#toolPart": "tool_part",
    "http://aicor.knowledge/l2-ontology.owl#totallyOrders": "totally_orders",
    "http://aicor.knowledge/l2-ontology.owl#traceProducedBy": "trace_produced_by",
    "http://aicor.knowledge/l2-ontology.owl#tracksReferentOf": "tracks_referent_of",
    "http://aicor.knowledge/l2-ontology.owl#trainedOnDistribution": "trained_on_distribution",
    "http://aicor.knowledge/l2-ontology.owl#transfersAcross": "transfers_across",
    "http://aicor.knowledge/l2-ontology.owl#transforms": "transforms",
    "http://aicor.knowledge/l2-ontology.owl#transformsEntity": "transforms_entity",
    "http://aicor.knowledge/l2-ontology.owl#transitionOf": "transition_of",
    "http://aicor.knowledge/l2-ontology.owl#transitionsOn": "transitions_on",
    "http://aicor.knowledge/l2-ontology.owl#translatedFrom": "translated_from",
    "http://aicor.knowledge/l2-ontology.owl#translatesEpisode": "translates_episode",
    "http://aicor.knowledge/l2-ontology.owl#triggers": "triggers",
    "http://aicor.knowledge/l2-ontology.owl#underWarrant": "under_warrant",
    "http://aicor.knowledge/l2-ontology.owl#unlocks": "unlocks",
    "http://aicor.knowledge/l2-ontology.owl#upperEdge": "upper_edge",
    "http://aicor.knowledge/l2-ontology.owl#uses": "uses",
    "http://aicor.knowledge/l2-ontology.owl#usesLossFunction": "uses_loss_function",
    "http://aicor.knowledge/l2-ontology.owl#usesModel": "uses_model",
    "http://aicor.knowledge/l2-ontology.owl#usesPredicateFrom": "uses_predicate_from",
    "http://aicor.knowledge/l2-ontology.owl#usesRegime": "uses_regime",
    "http://aicor.knowledge/l2-ontology.owl#validatedBy": "validated_by",
    "http://aicor.knowledge/l2-ontology.owl#variesNuisance": "varies_nuisance",
    "http://aicor.knowledge/l2-ontology.owl#verifiesAgainst": "verifies_against",
    "http://aicor.knowledge/l2-ontology.owl#viaChain": "via_chain",
    "http://aicor.knowledge/l2-ontology.owl#viaProjection": "via_projection",
    "http://aicor.knowledge/l2-ontology.owl#vocabularyFor": "vocabulary_for",
    "http://aicor.knowledge/l2-ontology.owl#warrantRangesOver": "warrant_ranges_over",
    "http://aicor.knowledge/l2-ontology.owl#warrantedBy": "warranted_by",
    "http://aicor.knowledge/l2-ontology.owl#warrantedByPortableResult": "warranted_by_portable_result",
    "http://aicor.knowledge/l2-ontology.owl#withinInterval": "within_interval",
    "http://aicor.knowledge/l2-ontology.owl#yields": "yields",
    "http://aicor.knowledge/l2-ontology.owl#yieldsCapability": "yields_capability",
    "http://aicor.knowledge/l2-ontology.owl#yieldsCertification": "yields_certification",
    "http://aicor.knowledge/l2-ontology.owl#yieldsDesignator": "yields_designator",
    "http://aicor.knowledge/l2-ontology.owl#yieldsTarget": "yields_target",
    "http://aicor.knowledge/l2-ontology.owl#yieldsVerdict": "yields_verdict",
    "http://aicor.knowledge/l2-ontology.owl#allParametersBound": "all_parameters_bound",
    "http://aicor.knowledge/l2-ontology.owl#assertsSameEntity": "asserts_same_entity",
    "http://aicor.knowledge/l2-ontology.owl#atCaseLevel": "at_case_level",
    "http://aicor.knowledge/l2-ontology.owl#authorityKind": "authority_kind",
    "http://aicor.knowledge/l2-ontology.owl#boundaryIsDeclared": "boundary_is_declared",
    "http://aicor.knowledge/l2-ontology.owl#boundaryTaskRelative": "boundary_task_relative",
    "http://aicor.knowledge/l2-ontology.owl#canBeReasonedAbout": "can_be_reasoned_about",
    "http://aicor.knowledge/l2-ontology.owl#canDepartFromPrediction": "can_depart_from_prediction",
    "http://aicor.knowledge/l2-ontology.owl#capabilityAxis": "capability_axis",
    "http://aicor.knowledge/l2-ontology.owl#carrierCapacity": "carrier_capacity",
    "http://aicor.knowledge/l2-ontology.owl#carriesMargin": "carries_margin",
    "http://aicor.knowledge/l2-ontology.owl#catchesFailure": "catches_failure",
    "http://aicor.knowledge/l2-ontology.owl#claimScope": "claim_scope",
    "http://aicor.knowledge/l2-ontology.owl#commitmentKind": "commitment_kind",
    "http://aicor.knowledge/l2-ontology.owl#constraintKind": "constraint_kind",
    "http://aicor.knowledge/l2-ontology.owl#createsOrDestroysObjects": "creates_or_destroys_objects",
    "http://aicor.knowledge/l2-ontology.owl#criticDimension": "critic_dimension",
    "http://aicor.knowledge/l2-ontology.owl#designerControllable": "designer_controllable",
    "http://aicor.knowledge/l2-ontology.owl#disturbanceCause": "disturbance_cause",
    "http://aicor.knowledge/l2-ontology.owl#effectDimension": "effect_dimension",
    "http://aicor.knowledge/l2-ontology.owl#enablesSuccessfulAction": "enables_successful_action",
    "http://aicor.knowledge/l2-ontology.owl#evaluatesInPerceptualSpace": "evaluates_in_perceptual_space",
    "http://aicor.knowledge/l2-ontology.owl#failureCode": "failure_code",
    "http://aicor.knowledge/l2-ontology.owl#familyDeclaredIndependentlyOfOutcome": "family_declared_independently_of_outcome",
    "http://aicor.knowledge/l2-ontology.owl#fitFromData": "fit_from_data",
    "http://aicor.knowledge/l2-ontology.owl#forbidsShortcut": "forbids_shortcut",
    "http://aicor.knowledge/l2-ontology.owl#forwardBranching": "forward_branching",
    "http://aicor.knowledge/l2-ontology.owl#freeSpace": "free_space",
    "http://aicor.knowledge/l2-ontology.owl#fromVideo": "from_video",
    "http://aicor.knowledge/l2-ontology.owl#functionCode": "function_code",
    "http://aicor.knowledge/l2-ontology.owl#generatorAgnostic": "generator_agnostic",
    "http://aicor.knowledge/l2-ontology.owl#governedBy": "governed_by",
    "http://aicor.knowledge/l2-ontology.owl#hasAmbiguity": "has_ambiguity",
    "http://aicor.knowledge/l2-ontology.owl#hasCertificationKind": "has_certification_kind",
    "http://aicor.knowledge/l2-ontology.owl#hasDimension": "has_dimension",
    "http://aicor.knowledge/l2-ontology.owl#hasFunctionalTyping": "has_functional_typing",
    "http://aicor.knowledge/l2-ontology.owl#hasInaccuracy": "has_inaccuracy",
    "http://aicor.knowledge/l2-ontology.owl#hasNonDeterministicEffects": "has_non_deterministic_effects",
    "http://aicor.knowledge/l2-ontology.owl#hasParameterCount": "has_parameter_count",
    "http://aicor.knowledge/l2-ontology.owl#hasTimestamp": "has_timestamp",
    "http://aicor.knowledge/l2-ontology.owl#holdsActiveWhileBelievedSucceedable": "holds_active_while_believed_succeedable",
    "http://aicor.knowledge/l2-ontology.owl#homogeneous": "homogeneous",
    "http://aicor.knowledge/l2-ontology.owl#interruptsPrimary": "interrupts_primary",
    "http://aicor.knowledge/l2-ontology.owl#isAbstractProcess": "is_abstract_process",
    "http://aicor.knowledge/l2-ontology.owl#isAccountable": "is_accountable",
    "http://aicor.knowledge/l2-ontology.owl#isArchitectureNeutral": "is_architecture_neutral",
    "http://aicor.knowledge/l2-ontology.owl#isAssessmentInput": "is_assessment_input",
    "http://aicor.knowledge/l2-ontology.owl#isBeliefState": "is_belief_state",
    "http://aicor.knowledge/l2-ontology.owl#isBound": "is_bound",
    "http://aicor.knowledge/l2-ontology.owl#isBounded": "is_bounded",
    "http://aicor.knowledge/l2-ontology.owl#isCausal": "is_causal",
    "http://aicor.knowledge/l2-ontology.owl#isClosedCycle": "is_closed_cycle",
    "http://aicor.knowledge/l2-ontology.owl#isClosedLoop": "is_closed_loop",
    "http://aicor.knowledge/l2-ontology.owl#isCommitmentSensitive": "is_commitment_sensitive",
    "http://aicor.knowledge/l2-ontology.owl#isConcurrent": "is_concurrent",
    "http://aicor.knowledge/l2-ontology.owl#isContextConditioned": "is_context_conditioned",
    "http://aicor.knowledge/l2-ontology.owl#isContinuouslyUpdated": "is_continuously_updated",
    "http://aicor.knowledge/l2-ontology.owl#isCriticAssigned": "is_critic_assigned",
    "http://aicor.knowledge/l2-ontology.owl#isCriticRelative": "is_critic_relative",
    "http://aicor.knowledge/l2-ontology.owl#isDataStructureRealization": "is_data_structure_realization",
    "http://aicor.knowledge/l2-ontology.owl#isDefeasible": "is_defeasible",
    "http://aicor.knowledge/l2-ontology.owl#isDerived": "is_derived",
    "http://aicor.knowledge/l2-ontology.owl#isDispositional": "is_dispositional",
    "http://aicor.knowledge/l2-ontology.owl#isDual": "is_dual",
    "http://aicor.knowledge/l2-ontology.owl#isEventDescription": "is_event_description",
    "http://aicor.knowledge/l2-ontology.owl#isExecutable": "is_executable",
    "http://aicor.knowledge/l2-ontology.owl#isExecutiveIndependent": "is_executive_independent",
    "http://aicor.knowledge/l2-ontology.owl#isExplicit": "is_explicit",
    "http://aicor.knowledge/l2-ontology.owl#isGenerator": "is_generator",
    "http://aicor.knowledge/l2-ontology.owl#isGroundTruth": "is_ground_truth",
    "http://aicor.knowledge/l2-ontology.owl#isGrounded": "is_grounded",
    "http://aicor.knowledge/l2-ontology.owl#isGroundedInSimulation": "is_grounded_in_simulation",
    "http://aicor.knowledge/l2-ontology.owl#isHardConstraint": "is_hard_constraint",
    "http://aicor.knowledge/l2-ontology.owl#isHierarchical": "is_hierarchical",
    "http://aicor.knowledge/l2-ontology.owl#isImmutable": "is_immutable",
    "http://aicor.knowledge/l2-ontology.owl#isInterruptible": "is_interruptible",
    "http://aicor.knowledge/l2-ontology.owl#isJointDistribution": "is_joint_distribution",
    "http://aicor.knowledge/l2-ontology.owl#isLazy": "is_lazy",
    "http://aicor.knowledge/l2-ontology.owl#isLowDimensional": "is_low_dimensional",
    "http://aicor.knowledge/l2-ontology.owl#isManipulableAtRuntime": "is_manipulable_at_runtime",
    "http://aicor.knowledge/l2-ontology.owl#isMetareasoning": "is_metareasoning",
    "http://aicor.knowledge/l2-ontology.owl#isMethodologicalObject": "is_methodological_object",
    "http://aicor.knowledge/l2-ontology.owl#isModelQuantity": "is_model_quantity",
    "http://aicor.knowledge/l2-ontology.owl#isMonteCarlo": "is_monte_carlo",
    "http://aicor.knowledge/l2-ontology.owl#isOpenFrontier": "is_open_frontier",
    "http://aicor.knowledge/l2-ontology.owl#isPartial": "is_partial",
    "http://aicor.knowledge/l2-ontology.owl#isPhysical": "is_physical",
    "http://aicor.knowledge/l2-ontology.owl#isProbabilistic": "is_probabilistic",
    "http://aicor.knowledge/l2-ontology.owl#isPulsed": "is_pulsed",
    "http://aicor.knowledge/l2-ontology.owl#isRWMSComponent": "is_rwms_component",
    "http://aicor.knowledge/l2-ontology.owl#isRWMSPart": "is_rwms_part",
    "http://aicor.knowledge/l2-ontology.owl#isRateBearing": "is_rate_bearing",
    "http://aicor.knowledge/l2-ontology.owl#isRawLog": "is_raw_log",
    "http://aicor.knowledge/l2-ontology.owl#isReasonableAtRuntime": "is_reasonable_at_runtime",
    "http://aicor.knowledge/l2-ontology.owl#isRevisable": "is_revisable",
    "http://aicor.knowledge/l2-ontology.owl#isScoped": "is_scoped",
    "http://aicor.knowledge/l2-ontology.owl#isShareableResearchObject": "is_shareable_research_object",
    "http://aicor.knowledge/l2-ontology.owl#isSignalled": "is_signalled",
    "http://aicor.knowledge/l2-ontology.owl#isSinglePurpose": "is_single_purpose",
    "http://aicor.knowledge/l2-ontology.owl#isSpecialization": "is_specialization",
    "http://aicor.knowledge/l2-ontology.owl#isSpecification": "is_specification",
    "http://aicor.knowledge/l2-ontology.owl#isSymbolic": "is_symbolic",
    "http://aicor.knowledge/l2-ontology.owl#isTestable": "is_testable",
    "http://aicor.knowledge/l2-ontology.owl#isTransparent": "is_transparent",
    "http://aicor.knowledge/l2-ontology.owl#isUnderdetermined": "is_underdetermined",
    "http://aicor.knowledge/l2-ontology.owl#isUniversallyQuantified": "is_universally_quantified",
    "http://aicor.knowledge/l2-ontology.owl#isWarrantedCommitment": "is_warranted_commitment",
    "http://aicor.knowledge/l2-ontology.owl#isWhiteBox": "is_white_box",
    "http://aicor.knowledge/l2-ontology.owl#lazy": "lazy",
    "http://aicor.knowledge/l2-ontology.owl#licensedBySingleEpisode": "licensed_by_single_episode",
    "http://aicor.knowledge/l2-ontology.owl#marginValue": "margin_value",
    "http://aicor.knowledge/l2-ontology.owl#matterKind": "matter_kind",
    "http://aicor.knowledge/l2-ontology.owl#maximal": "maximal",
    "http://aicor.knowledge/l2-ontology.owl#microtheoryFocus": "microtheory_focus",
    "http://aicor.knowledge/l2-ontology.owl#nearActionInterface": "near_action_interface",
    "http://aicor.knowledge/l2-ontology.owl#obligationStatus": "obligation_status",
    "http://aicor.knowledge/l2-ontology.owl#openEntryValue": "open_entry_value",
    "http://aicor.knowledge/l2-ontology.owl#ordersSubtasks": "orders_subtasks",
    "http://aicor.knowledge/l2-ontology.owl#overMereology": "over_mereology",
    "http://aicor.knowledge/l2-ontology.owl#physicsValidated": "physics_validated",
    "http://aicor.knowledge/l2-ontology.owl#predictsSuccess": "predicts_success",
    "http://aicor.knowledge/l2-ontology.owl#presupposesRationality": "presupposes_rationality",
    "http://aicor.knowledge/l2-ontology.owl#principleTag": "principle_tag",
    "http://aicor.knowledge/l2-ontology.owl#problemIsProperPart": "problem_is_proper_part",
    "http://aicor.knowledge/l2-ontology.owl#propagatesConstraintsDownward": "propagates_constraints_downward",
    "http://aicor.knowledge/l2-ontology.owl#propertyKind": "property_kind",
    "http://aicor.knowledge/l2-ontology.owl#realizedByKRR": "realized_by_krr",
    "http://aicor.knowledge/l2-ontology.owl#recordsIntentionAndActual": "records_intention_and_actual",
    "http://aicor.knowledge/l2-ontology.owl#redundant": "redundant",
    "http://aicor.knowledge/l2-ontology.owl#relationKind": "relation_kind",
    "http://aicor.knowledge/l2-ontology.owl#replansDuringExecution": "replans_during_execution",
    "http://aicor.knowledge/l2-ontology.owl#representationEdgeOpen": "representation_edge_open",
    "http://aicor.knowledge/l2-ontology.owl#requiresCompatibleDisposition": "requires_compatible_disposition",
    "http://aicor.knowledge/l2-ontology.owl#requiresHarderTool": "requires_harder_tool",
    "http://aicor.knowledge/l2-ontology.owl#requiresLimitedReprogramming": "requires_limited_reprogramming",
    "http://aicor.knowledge/l2-ontology.owl#rerunsSensorimotorLoop": "reruns_sensorimotor_loop",
    "http://aicor.knowledge/l2-ontology.owl#restorationKind": "restoration_kind",
    "http://aicor.knowledge/l2-ontology.owl#rungLevel": "rung_level",
    "http://aicor.knowledge/l2-ontology.owl#runsAsControlProcess": "runs_as_control_process",
    "http://aicor.knowledge/l2-ontology.owl#satisfiesReference": "satisfies_reference",
    "http://aicor.knowledge/l2-ontology.owl#scopeKind": "scope_kind",
    "http://aicor.knowledge/l2-ontology.owl#severityValue": "severity_value",
    "http://aicor.knowledge/l2-ontology.owl#singleValued": "single_valued",
    "http://aicor.knowledge/l2-ontology.owl#standardCode": "standard_code",
    "http://aicor.knowledge/l2-ontology.owl#statedRate": "stated_rate",
    "http://aicor.knowledge/l2-ontology.owl#statusCode": "status_code",
    "http://aicor.knowledge/l2-ontology.owl#stepReached": "step_reached",
    "http://aicor.knowledge/l2-ontology.owl#stopsWhenBelievedUnsucceedableOrIrrelevant": "stops_when_believed_unsucceedable_or_irrelevant",
    "http://aicor.knowledge/l2-ontology.owl#sufficiencyKind": "sufficiency_kind",
    "http://aicor.knowledge/l2-ontology.owl#system2": "system2",
    "http://aicor.knowledge/l2-ontology.owl#thinsAsRadiusGrows": "thins_as_radius_grows",
    "http://aicor.knowledge/l2-ontology.owl#total": "total",
    "http://aicor.knowledge/l2-ontology.owl#underspecified": "underspecified",
    "http://aicor.knowledge/l2-ontology.owl#usesImagination": "uses_imagination",
    "http://aicor.knowledge/l2-ontology.owl#vesselKind": "vessel_kind",
}
