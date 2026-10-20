"""
Invalid lifecycle proposals at the knowledge-operation boundary.
"""

from __future__ import annotations

from dataclasses import dataclass

from krrood.exceptions import DataclassException
from typing_extensions import TYPE_CHECKING

if TYPE_CHECKING:
    from repository_knowledge.publication import TrustIssue

# %% lifecycle errors


@dataclass
class InvalidConceptTransition(DataclassException):
    """
    Reject a lifecycle change outside the drafting boundary's authority.
    """

    relative_path: str
    """The concept selected for the rejected change."""

    def error_message(self) -> str:
        return f"Unsupported concept lifecycle change: {self.relative_path}."

    def suggest_correction(self) -> str:
        return "Use draft or a deprecated tombstone; stable promotion needs authenticated publication."


# %% publication refusals


@dataclass
class PublicationDenied(DataclassException):
    """
    Refuse a canonical result that lacks required trust evidence.
    """

    issues: list[TrustIssue]
    """The failed trust conditions."""

    def error_message(self) -> str:
        return "Publication refused: " + ", ".join(self.issues)

    def suggest_correction(self) -> str:
        return (
            "Resolve the reported trust conditions; use source-only context meanwhile."
        )
