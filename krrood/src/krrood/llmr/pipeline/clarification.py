"""Clarification types raised when entity grounding finds zero candidates.

Generic, framework-agnostic — no robot, pycram, or sdt references.
ArmCapacityError is intentionally absent: arm management is robot-specific
and belongs in the caller's integration layer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing_extensions import List


@dataclass
class ClarificationRequest:
    """Structured description of why clarification is needed.

    :param entity_name: The name the LLM extracted that could not be grounded.
    :param entity_role: Human-readable role (e.g. ``"object"`` or ``"target surface"``).
    :param available_names: Names of all groundable entities visible in the world
        so the caller can present valid alternatives to the user.
    :param message: Human-readable explanation of the failure.
    """

    entity_name: str
    entity_role: str
    available_names: List[str] = field(default_factory=list)
    message: str = ""


class ClarificationNeededError(Exception):
    """Raised by an ActionHandler when zero entities are found for a description.

    Carries the :class:`ClarificationRequest` as ``self.request`` so the
    ``ExecutionLoop`` can forward it to the caller without re-parsing the
    exception message.
    """

    def __init__(self, request: ClarificationRequest) -> None:
        self.request: ClarificationRequest = request
        super().__init__(request.message or f"Cannot ground entity '{request.entity_name}'")
