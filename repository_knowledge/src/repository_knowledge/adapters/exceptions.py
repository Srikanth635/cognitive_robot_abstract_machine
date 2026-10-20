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


# %% independent identity errors


@dataclass
class InvalidIdentityEvidence(DataclassException):
    """
    Refuse absent, malformed or inconsistent identity-provider evidence.
    """

    resource: str
    """The provider resource or evidence field that failed validation."""

    def error_message(self) -> str:
        return f"Invalid independent identity evidence: {self.resource}."

    def suggest_correction(self) -> str:
        return "Provide authenticated repository controls and complete evidence; do not substitute producer metadata."
