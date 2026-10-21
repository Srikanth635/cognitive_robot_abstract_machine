"""
Invalid notebook trials and refused repository evidence.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from krrood.exceptions import DataclassException

# %% notebook trial errors


@dataclass
class LiveRequestsDisabled(DataclassException):
    """
    Require explicit enablement before contacting a model provider.
    """

    def error_message(self) -> str:
        return "Live provider requests are disabled."

    def suggest_correction(self) -> str:
        return (
            "Select your provider/model and explicitly enable the live notebook trial."
        )


@dataclass
class InvalidModelSettings(DataclassException):
    """
    Reject an incomplete provider selection or nonpositive request budget.
    """

    def error_message(self) -> str:
        return "The provider, model or request budget is invalid."

    def suggest_correction(self) -> str:
        return "Provide a provider/model and positive turn, timeout and output limits."


@dataclass
class NotebookQueryRefused(DataclassException):
    """
    Keep refused or malformed MCP results out of model evidence.
    """

    operation: str
    """The refused repository operation."""

    def error_message(self) -> str:
        return f"Repository evidence was refused for {self.operation}."

    def suggest_correction(self) -> str:
        return "Check source freshness and query bounds; refresh explicitly if needed."


@dataclass
class NotebookRepositoryMissing(DataclassException):
    """
    Require an explicit checkout when notebook location cannot identify one.
    """

    directory: Path
    """The directory from which repository discovery began."""

    def error_message(self) -> str:
        return f"No repository-knowledge project was found above {self.directory}."

    def suggest_correction(self) -> str:
        return "Open the notebook inside CRAM or explicitly select its repository root."
