"""
The phases a manipulation is divided into.

A phase is a discrete sensorimotor building block: driven by a subgoal, parameterised in
a feedforward manner by internal models, and delimited by boundary events. It is a task
concept and never an occurrence. What is active in a phase, its parameters and its
continuous predictions, is listed here on the phase, so those concepts need not know the
phases that use them.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from llm_reasoner.knowledge_updated.microworlds.flanagan.alignments import (
    SOMA,
    subclass_of,
)
from llm_reasoner.knowledge_updated.microworlds.flanagan.core.conditions import (
    ConditionRequirement,
    Subgoal,
)
from llm_reasoner.knowledge_updated.microworlds.flanagan.core.events import (
    PhaseBoundaryEventType,
)
from llm_reasoner.knowledge_updated.microworlds.flanagan.core.models import (
    InternalModel,
)
from llm_reasoner.knowledge_updated.microworlds.flanagan.core.parameters import (
    Parameter,
)
from llm_reasoner.knowledge_updated.microworlds.flanagan.core.prediction import (
    ContinuousSignalPrediction,
)
from llm_reasoner.knowledge_updated.microworlds.flanagan.core.roles import RoleSlot
from llm_reasoner.knowledge_updated.microworlds.flanagan.core.value_partitions import (
    CouplingRegime,
    RegulationMode,
    SensoryModality,
)
from llm_reasoner.knowledge_updated.microworlds.flanagan.ontology import CORE
from llm_reasoner.knowledge_updated.ontological.entities import KnowledgeEntity
from llm_reasoner.knowledge_updated.ontological.properties import attribute, relation
from llm_reasoner.knowledge_updated.ontological.vocabulary import EvidenceStatus

# %% phases


@CORE.concept(
    alignments=(subclass_of(SOMA, "PhysicalTask"),),
    disjoint_siblings=True,
    status=EvidenceStatus.AUTHOR_HYPOTHESIS,
    claims=("FJ-003",),
)
@dataclass(eq=False)
class ActionPhase(KnowledgeEntity):
    """
    A discrete sensorimotor building block of a manipulation task.

    A task concept, not an occurrence. Every phase has exactly one subgoal, one coupling
    regime and one within-phase regulation mode. Its termination is represented
    independently by one or more boundary event types.
    """

    aims_for: Optional[Subgoal] = relation(functional=True, minimum_count=1)
    """
    The single subgoal that individuates this phase.
    """

    coupling_regime: Optional[CouplingRegime] = attribute(
        functional=True, minimum_count=1
    )
    """
    Whether the effector and the manipulandum move as one unit during the phase.
    """

    regulation_mode: Optional[RegulationMode] = attribute(
        functional=True, minimum_count=1
    )
    """
    How commands are regulated within the phase, independent of its boundary.
    """

    initiated_by: list[PhaseBoundaryEventType] = relation()
    """
    The events that begin the phase.

    The first phase of a structure has none.
    """

    terminated_by: list[PhaseBoundaryEventType] = relation(minimum_count=1)
    """
    The events that end the phase.

    A phase with no boundary is not a phase.
    """

    phase_involves: list[RoleSlot] = relation(minimum_count=1)
    """
    The participant roles taking part in the phase.
    """

    has_condition_requirement: list[ConditionRequirement] = relation()
    """
    What the phase demands of the world at entry, throughout and at exit.
    """

    monitors_modality: list[SensoryModality] = attribute()
    """
    The sensory channels tracked throughout the phase and not sampled at a boundary.

    A continuously regulated phase that monitors nothing has no control law.
    """

    governed_by: list[InternalModel] = relation()
    """
    The internal models that parameterise the phase.
    """

    active_parameters: list[Parameter] = relation()
    """
    The parameters that are in force during the phase.
    """

    continuous_predictions: list[ContinuousSignalPrediction] = relation()
    """
    The continuous sensory predictions compared throughout the phase.
    """
