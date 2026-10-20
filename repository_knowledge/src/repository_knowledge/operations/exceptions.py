"""
Invalid lifecycle proposals at the knowledge-operation boundary.
"""

from __future__ import annotations

from dataclasses import dataclass

from krrood.exceptions import DataclassException

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
