from __future__ import annotations

from typing import Optional, List, Literal, Union, Set
from uuid import uuid4
from pydantic import BaseModel, Field, model_validator


# -----------------------------------------------------------------------------
# References (identity-only, recursion-safe)
# -----------------------------------------------------------------------------

class EntityRef(BaseModel):
    id: str = Field(..., description="Unique identifier of the referenced node")
    ontology_class: str = Field(..., description="Ontology class name")
    label: Optional[str] = Field(default=None, examples=["table","sink","kitchen_counter"])

    model_config = {"extra": "forbid"}


class RoleRef(EntityRef):
    ontology_class: Literal["Role"]


class TimeIntervalRef(EntityRef):
    ontology_class: Literal["TimeInterval"]


class AgentRef(EntityRef):
    ontology_class: Literal["Agent"]


class PhysicalAgentRef(EntityRef):
    ontology_class: Literal["PhysicalAgent"]


class ObjectRef(EntityRef):
    ontology_class: Literal["Object"]


class PhysicalObjectRef(EntityRef):
    ontology_class: Literal["PhysicalObject"]


ParticipantRef = Union[
    AgentRef,
    PhysicalAgentRef,
    ObjectRef,
    PhysicalObjectRef,
]


class TaskRef(EntityRef):
    ontology_class: Literal[
        "EventType",
        "Task",
        "PhysicalTask",
        "Manipulating",
        "PickingUp",
        "PuttingDown",
        "Cutting",
        "Pouring"
    ]


# -----------------------------------------------------------------------------
# Role binding (instance-level)
# -----------------------------------------------------------------------------

class RoleAssignment(BaseModel):
    """
    Bind a participant to an Action via a Role.

    Convention:
    - agent/theme/source/destination are RoleRef.label values (recommended).
    - related_to is optional extra linkage (can mirror source/destination target).
    """
    participant: ParticipantRef
    role: RoleRef
    related_to: Optional[EntityRef] = Field(
        default=None,
        description="Optional secondary entity (e.g., source/destination target).",
    )

    model_config = {"extra": "forbid"}


# -----------------------------------------------------------------------------
# Action-centric instance models (LLM-facing)
# -----------------------------------------------------------------------------

class PhysicalAction(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    ontology_class: Literal["PhysicalAction"] = "PhysicalAction"

    has_participant: List[RoleAssignment] = Field(..., min_length=1)
    executes_task: List[TaskRef] = Field(..., min_length=1)
    has_time_interval: Optional[List[TimeIntervalRef]] = None

    @staticmethod
    def _role_key(ra: RoleAssignment) -> str:
        s = (ra.role.label or ra.role.id or "").strip().lower()
        if s.startswith("role-"):
            s = s.replace("role-", "", 1)
        return s

    @model_validator(mode="after")
    def enforce_task_role_constraints(self) -> "PhysicalAction":
        """
        Enforce task-specific roles conditionally for PickingUp and PuttingDown.
        """
        task_classes: Set[str] = {t.ontology_class for t in self.executes_task}
        is_pickup = "PickingUp" in task_classes
        is_putdown = "PuttingDown" in task_classes

        if not (is_pickup or is_putdown):
            return self

        agents = [ra for ra in self.has_participant if self._role_key(ra) == "agent"]
        themes = [ra for ra in self.has_participant if self._role_key(ra) == "theme"]
        sources = [ra for ra in self.has_participant if self._role_key(ra) == "source"]
        dests = [ra for ra in self.has_participant if self._role_key(ra) == "destination"]

        if not any(ra.participant.ontology_class in {"Agent", "PhysicalAgent"} for ra in agents):
            raise ValueError(
                "Manipulation action requires an Agent/PhysicalAgent RoleAssignment with role 'agent'."
            )

        if len(themes) != 1:
            raise ValueError(
                "Manipulation action requires exactly one RoleAssignment with role 'theme'."
            )

        theme = themes[0]

        if is_pickup:
            if len(sources) > 1:
                raise ValueError("PickingUp allows at most one RoleAssignment with role 'source'.")
            if theme.related_to is not None and len(sources) == 1:
                if sources[0].participant.id != theme.related_to.id:
                    raise ValueError(
                        "PickingUp inconsistency: theme.related_to must match the 'source' participant id."
                    )

        if is_putdown:
            if len(dests) > 1:
                raise ValueError("PuttingDown allows at most one RoleAssignment with role 'destination'.")
            if theme.related_to is not None and len(dests) == 1:
                if dests[0].participant.id != theme.related_to.id:
                    raise ValueError(
                        "PuttingDown inconsistency: theme.related_to must match the 'destination' participant id."
                    )

        return self


class ActionSequence(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    ontology_class: Literal["ActionSequence"] = "ActionSequence"

    actions: List[PhysicalAction] = Field(..., min_length=1)


__all__ = [
    "EntityRef",
    "RoleRef",
    "TimeIntervalRef",
    "AgentRef",
    "PhysicalAgentRef",
    "ObjectRef",
    "PhysicalObjectRef",
    "ParticipantRef",
    "TaskRef",
    "RoleAssignment",
    "PhysicalAction",
    "ActionSequence",
]