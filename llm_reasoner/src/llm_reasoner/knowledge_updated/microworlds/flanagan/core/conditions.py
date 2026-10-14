"""
The world states a phase depends on and aims for.

There are deliberately no precondition and postcondition types. Being a precondition is
relational: the very same state is the postcondition of one phase and the precondition of
the next. The distinction is carried by a :class:`ConditionRequirement`, which qualifies
a phase's demand on a condition, and a :class:`Subgoal` states which condition a phase
exists to bring about.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from llm_reasoner.knowledge_updated.microworlds.flanagan.alignments import (
    DUL,
    SOMA,
    subclass_of,
)
from llm_reasoner.knowledge_updated.microworlds.flanagan.core.events import (
    MechanicalEventType,
    PhaseBoundaryEventType,
)
from llm_reasoner.knowledge_updated.microworlds.flanagan.core.roles import RoleSlot
from llm_reasoner.knowledge_updated.microworlds.flanagan.core.value_partitions import (
    ConditionMode,
    ConditionPolarity,
)
from llm_reasoner.knowledge_updated.microworlds.flanagan.ontology import CORE
from llm_reasoner.knowledge_updated.ontological.entities import KnowledgeEntity
from llm_reasoner.knowledge_updated.ontological.properties import attribute, relation
from llm_reasoner.knowledge_updated.ontological.vocabulary import EvidenceStatus

# %% conditions


@CORE.concept(
    alignments=(subclass_of(SOMA, "StateType"),),
    disjoint_siblings=True,
    status=EvidenceStatus.ENGINEERING_EXTENSION,
)
@dataclass(eq=False)
class PhaseConditionType(KnowledgeEntity):
    """
    A type of world state that is relevant to a phase.
    """

    condition_on_role: list[RoleSlot] = relation(minimum_count=1)
    """
    The roles the condition is about. Without them the resolver cannot bind it to the
    world.
    """

    violation_signalled_by: Optional[MechanicalEventType] = relation(functional=True)
    """
    The failure event that registers the condition ceasing to hold.

    It gives an invariant operational content: a condition whose violation nothing can
    detect is a hope and not a guard.
    """


# %% requirements


@CORE.concept(
    alignments=(subclass_of(DUL, "Relation"),),
    status=EvidenceStatus.ENGINEERING_EXTENSION,
)
@dataclass(eq=False)
class ConditionRequirement(KnowledgeEntity):
    """
    A phase's demand on a condition, qualified by mode, polarity and whether it is assumed.

    A separate object so that mode, polarity and status are expressible without a grid of
    properties for every combination.
    """

    on_condition: Optional[PhaseConditionType] = relation(
        functional=True, minimum_count=1
    )
    """
    The condition the phase makes a demand on.
    """

    mode: Optional[ConditionMode] = attribute(functional=True, minimum_count=1)
    """
    When, relative to the phase, the condition is demanded.
    """

    polarity: ConditionPolarity = attribute(
        functional=True, default=ConditionPolarity.MUST_HOLD
    )
    """
    Whether the condition is demanded to hold or to be absent.
    """

    is_world_assumption: bool = attribute(
        functional=True, default=False, marks_assumption=True
    )
    """
    Whether no phase establishes the condition and it is presupposed of the world.

    Assumptions are what the resolver must check against the world model, and report when
    it cannot.
    """


# %% subgoals


@CORE.concept(status=EvidenceStatus.AUTHOR_HYPOTHESIS, claims=("FJ-003",))
@dataclass(eq=False)
class Subgoal(KnowledgeEntity):
    """
    The state a phase exists to bring about, and the event that shows it was reached.

    The state itself is a condition. A subgoal only says which condition, so the outcome
    is stated once.
    """

    targets_condition: Optional[PhaseConditionType] = relation(
        functional=True, minimum_count=1
    )
    """
    The condition the phase is meant to bring about.
    """

    attainment_signalled_by: Optional[PhaseBoundaryEventType] = relation(
        functional=True, minimum_count=1
    )
    """
    The event whose registration is evidence that the subgoal is reached.

    Functional: a subgoal with two independent signatures is two subgoals. It must not be
    a failure event.
    """
