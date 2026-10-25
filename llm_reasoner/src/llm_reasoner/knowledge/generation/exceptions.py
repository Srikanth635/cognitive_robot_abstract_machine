"""
What generating a hypothesis refuses.
"""

from __future__ import annotations

from dataclasses import dataclass

from krrood.exceptions import DataclassException

from llm_reasoner.exceptions import LLMReasonerError


@dataclass
class MeasurementIsNotGenerated(LLMReasonerError, DataclassException):
    """
    A query asks a language model to generate a class whose values only a shape or a
    physics simulation may give.
    """

    root_type: type
    """
    The class the query asks for.
    """

    def error_message(self) -> str:
        return (
            f"{self.root_type.__name__} is a physical measurement, which a language "
            "model never generates"
        )

    def suggest_correction(self) -> str:
        return (
            "derive the value from the container's shape or measure it in a physics "
            "simulation, and read it from the measurement store"
        )
