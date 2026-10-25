"""
What storing knowledge refuses.
"""

from __future__ import annotations

from dataclasses import dataclass

from krrood.exceptions import DataclassException

from llm_reasoner.exceptions import LLMReasonerError
from llm_reasoner.knowledge.acceptance.rules import Violation


@dataclass
class InvalidMeasurement(LLMReasonerError, DataclassException):
    """
    A physical measurement breaks the restrictions, ranges or shapes of its ontology, so
    it is not stored.
    """

    violations: list[Violation]
    """
    What the measurement and the fragments it reaches fail.
    """

    def error_message(self) -> str:
        return "the measurement breaks its ontology: " + "; ".join(
            str(violation) for violation in self.violations
        )

    def suggest_correction(self) -> str:
        return "give every value the measurement's classes require, within its range"
