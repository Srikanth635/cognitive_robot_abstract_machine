"""
The command pathway of an inverse model.

An inverse model selects a motor command for a desired state. An internal copy of the
outgoing command, the efference copy, is what a dynamics model uses to predict the load
the command will meet.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from llm_reasoner.knowledge_updated.microworlds.flanagan.alignments import (
    DUL,
    related_to,
    subclass_of,
)
from llm_reasoner.knowledge_updated.microworlds.flanagan.core.conditions import (
    PhaseConditionType,
)
from llm_reasoner.knowledge_updated.microworlds.flanagan.core.models import (
    InverseModel,
)
from llm_reasoner.knowledge_updated.microworlds.flanagan.core.parameters import (
    MotorSetpoint,
)
from llm_reasoner.knowledge_updated.microworlds.flanagan.ontology import OBJECT_DYNAMICS
from llm_reasoner.knowledge_updated.ontological.entities import KnowledgeEntity
from llm_reasoner.knowledge_updated.ontological.properties import relation
from llm_reasoner.knowledge_updated.ontological.vocabulary import EvidenceStatus

# %% command pathway


@OBJECT_DYNAMICS.concept(
    alignments=(related_to(DUL, "Situation"),),
    status=EvidenceStatus.SUPPORTED_INTERPRETATION,
    claims=("FJ-002",),
)
@dataclass(eq=False)
class DesiredState(KnowledgeEntity):
    """
    A state that an inverse model seeks to establish.
    """

    desired_condition: list[PhaseConditionType] = relation(minimum_count=1)
    """
    The conditions whose establishment defines the desired state.
    """


@OBJECT_DYNAMICS.concept(
    alignments=(related_to(DUL, "Description"),),
    status=EvidenceStatus.SUPPORTED_INTERPRETATION,
    claims=("FJ-002",),
)
@dataclass(eq=False)
class MotorCommand(KnowledgeEntity):
    """
    A command selected by an inverse model for a desired state.
    """

    pursues_state: Optional[DesiredState] = relation(functional=True, minimum_count=1)
    """
    The desired state the command is intended to establish.
    """

    generated_by: Optional[InverseModel] = relation(functional=True, minimum_count=1)
    """
    The inverse model that selected the command.
    """

    commands_setpoint: list[MotorSetpoint] = relation(minimum_count=1)
    """
    The motor setpoints specified by the command.
    """


@OBJECT_DYNAMICS.concept(
    alignments=(subclass_of(DUL, "InformationObject"),),
    status=EvidenceStatus.SUPPORTED_INTERPRETATION,
    claims=("FJ-008",),
)
@dataclass(eq=False)
class EfferenceCopy(KnowledgeEntity):
    """
    An internal copy of an outgoing motor command.
    """

    copies_command: Optional[MotorCommand] = relation(functional=True, minimum_count=1)
    """
    The outgoing command represented by the copy.
    """
