"""
What reading the ontology modules refuses.
"""

from __future__ import annotations

from dataclasses import dataclass

from krrood.exceptions import DataclassException

from llm_reasoner.exceptions import LLMReasonerError

# %% roles and slots


@dataclass
class SeveralRolesFillSlot(LLMReasonerError, DataclassException):
    """
    Several manipulation roles are put directly under one slot of the action core's
    frame, so no single role fills it.
    """

    slot: str
    """
    The IRI of the slot.
    """

    roles: list[str]
    """
    The IRIs of the roles put directly under it.
    """

    def error_message(self) -> str:
        return f"{self.slot} is filled by several roles: " + ", ".join(self.roles)

    def suggest_correction(self) -> str:
        return "name a slot that one manipulation role is put directly under"
