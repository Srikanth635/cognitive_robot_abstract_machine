"""
What the EQL panel can ask of the active hypothesis: a variable per kind of fragment of
its phase structure, the vocabulary its answers name, preset questions, and how answers
about its fragments read.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import StrEnum

from krrood.entity_query_language.predicate import HasType
from llm_reasoner.knowledge.access import (
    ConceptReference,
    DescriptionEntity,
    action_core,
    core,
    gaze,
    pouring,
    prehension,
)
from typing_extensions import Any, Dict, List, Optional

from cramera.knowledge.presets import Preset
from cramera.knowledge.query_contribution import QueryContribution
from cramera.knowledge.query_domain import QueryDomain
from cramera.knowledge.query_runner import AnswerRendering
from cramera.resolution import hypothesis_knowledge
from cramera.resolution.hypothesis_knowledge import HypothesisKnowledge
from cramera.resolution.hypothesis_knowledge.ontology_backing import OntologyBacking
from cramera.resolution.paths import ResolutionFile

# %% the variables


class HypothesisVariable(StrEnum):
    """
    The query variables over the fragments of the active phase structure, one per kind
    of fragment.
    """

    PHASE = "pouring_phase"
    ROLE = "pouring_role"
    EVENT_TYPE = "pouring_event_type"
    FORCE_DYNAMIC_EVENT = "pouring_force_dynamic_event"
    TRANSITION = "pouring_transition"
    CONTROL_POINT = "pouring_control_point"
    COUPLING = "pouring_coupling"
    CONTINUOUS_PREDICTION = "pouring_continuous_prediction"
    CONDITION_REQUIREMENT = "pouring_condition_requirement"
    PARAMETER = "pouring_parameter"
    SENSORY_PREDICTION = "pouring_sensory_prediction"
    COMPARISON = "pouring_comparison"
    CORRECTION = "pouring_correction"
    GAZE_FIXATION = "pouring_gaze_fixation"

    @property
    def entity_type(self) -> type:
        """
        :return: The kind of fragment the variable ranges over.
        """
        return {
            HypothesisVariable.PHASE: core.ActionPhase,
            HypothesisVariable.ROLE: core.ManipulationRole,
            HypothesisVariable.EVENT_TYPE: core.PhaseBoundaryEventType,
            HypothesisVariable.FORCE_DYNAMIC_EVENT: core.MechanicalEventType,
            HypothesisVariable.TRANSITION: core.PhaseTransition,
            HypothesisVariable.CONTROL_POINT: core.SensorimotorControlPoint,
            HypothesisVariable.COUPLING: prehension.GripLoadCoupling,
            HypothesisVariable.CONTINUOUS_PREDICTION: core.ContinuousSignalPrediction,
            HypothesisVariable.CONDITION_REQUIREMENT: core.ConditionRequirement,
            HypothesisVariable.PARAMETER: core.Parameter,
            HypothesisVariable.SENSORY_PREDICTION: core.SensoryEventPrediction,
            HypothesisVariable.COMPARISON: core.PredictionComparison,
            HypothesisVariable.CORRECTION: core.CorrectiveAction,
            HypothesisVariable.GAZE_FIXATION: gaze.GazeFixation,
        }[self]


# %% the queries


@dataclass(frozen=True)
class HypothesisQueries(QueryContribution):
    """
    The active hypothesis, offered to the EQL panel: its fragments as variables, the
    vocabulary their values are terms of, and questions about them.
    """

    knowledge: Optional[HypothesisKnowledge]
    """
    The hypothesis knowledge asked about; ``None`` while none is active, when every
    variable ranges over nothing and no question is offered.
    """

    @classmethod
    def of_active(cls) -> HypothesisQueries:
        """
        :return: The queries of the hypothesis knowledge the views show now.
        """
        return cls(knowledge=hypothesis_knowledge.active())

    def fragments(self) -> Dict[HypothesisVariable, List[Any]]:
        """
        :return: The fragments each variable ranges over, those of a phase structure in
            the order it runs them; none without a phase structure.
        """
        view = None if self.knowledge is None else self.knowledge.phase_structure
        if view is None:
            return {variable: [] for variable in HypothesisVariable}
        structure = view.root
        requirements = {
            id(requirement): requirement
            for phase in view.phases()
            for requirement in phase.has_condition_requirement
        }
        couplings = (
            list(structure.defines_force_coupling)
            if isinstance(structure, prehension.PrehensionPhaseStructure)
            else []
        )
        return {
            HypothesisVariable.PHASE: list(view.phases()),
            HypothesisVariable.ROLE: list(structure.defines_role),
            HypothesisVariable.EVENT_TYPE: list(structure.defines_event_type),
            HypothesisVariable.FORCE_DYNAMIC_EVENT: [
                dynamics.event for dynamics in view.force_dynamics()
            ],
            HypothesisVariable.TRANSITION: list(structure.defines_transition),
            HypothesisVariable.CONTROL_POINT: list(structure.defines_control_point),
            HypothesisVariable.COUPLING: couplings,
            HypothesisVariable.CONTINUOUS_PREDICTION: list(
                structure.defines_continuous_prediction
            ),
            HypothesisVariable.CONDITION_REQUIREMENT: list(requirements.values()),
            HypothesisVariable.PARAMETER: list(structure.defines_parameter),
            HypothesisVariable.SENSORY_PREDICTION: list(structure.defines_prediction),
            HypothesisVariable.COMPARISON: list(structure.defines_comparison),
            HypothesisVariable.CORRECTION: list(structure.defines_correction),
            HypothesisVariable.GAZE_FIXATION: list(structure.defines_gaze_fixation),
        }

    def domains(self) -> List[QueryDomain]:
        fragments = self.fragments()
        return [
            QueryDomain(variable.value, variable.entity_type, fragments[variable])
            for variable in HypothesisVariable
        ]

    def names(self) -> Dict[str, Any]:
        vocabulary = (
            core.RegulationMode,
            core.CouplingRegime,
            action_core.ConditionTiming,
            core.ComparisonOutcome,
            core.ForceKind,
            HasType,
        )
        modules = (core, action_core, prehension, pouring, gaze)
        return {kind.__name__: kind for kind in vocabulary} | {
            module.__name__.rsplit(".", 1)[-1]: module for module in modules
        }

    def presets(self) -> List[Preset]:
        if self.knowledge is None or self.knowledge.phase_structure is None:
            return []
        return [
            Preset(**preset)
            for preset in json.loads(ResolutionFile.HYPOTHESIS_PRESETS.read_text())
        ]

    def answers(self) -> HypothesisAnswers:
        return HypothesisAnswers(knowledge=self.knowledge)


# %% how answers read


@dataclass(frozen=True)
class HypothesisAnswers(AnswerRendering):
    """
    Writes a fragment of the hypothesis by its name, and a term of the ontologies by its
    prefixed name; an answer about a fragment lights it up in the Hypothesis Graph.
    """

    knowledge: Optional[HypothesisKnowledge]
    """
    The hypothesis knowledge whose graph the fragments are lit up in; none lights up
    without it.
    """

    backing: OntologyBacking = field(default_factory=OntologyBacking)
    """
    The ontologies whose terms are written with their prefixes.
    """

    def renders(self, value: Any) -> bool:
        if isinstance(value, (list, tuple)):
            return bool(value) and all(self.renders(item) for item in value)
        if isinstance(value, StrEnum):
            return self.backing.namespace_of(value.value) is not None
        return isinstance(value, (DescriptionEntity, ConceptReference))

    def rendered(self, value: Any) -> Any:
        if isinstance(value, (list, tuple)):
            return [self.rendered(item) for item in value]
        if isinstance(value, StrEnum):
            return self.backing.prefixed(value.value)
        if isinstance(value, ConceptReference):
            return self.backing.prefixed(value.iri)
        return value.label or type(value).__name__

    def highlights_of(self, value: Any) -> List[str]:
        if isinstance(value, (list, tuple)):
            return [node for item in value for node in self.highlights_of(item)]
        node = (
            None
            if self.knowledge is None
            else self.knowledge.node_identifier_for(value)
        )
        return [] if node is None else [node]
