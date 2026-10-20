"""
Invalid curated documents at the format-exchange boundary.
"""

from __future__ import annotations

from dataclasses import dataclass

from krrood.exceptions import DataclassException

# %% knowledge document errors


@dataclass
class InvalidKnowledgeDocument(DataclassException):
    """
    Reject a document whose framing or metadata shape cannot be interpreted.
    """

    relative_path: str
    """The malformed document path."""

    def error_message(self) -> str:
        return f"Invalid knowledge document: {self.relative_path}."

    def suggest_correction(self) -> str:
        return (
            "Use a delimited metadata mapping and the documented CRAM evidence schema."
        )
