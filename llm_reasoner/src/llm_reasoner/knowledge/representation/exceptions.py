"""
What reading a hypothesis from RDF refuses.
"""

from __future__ import annotations

from dataclasses import dataclass

from krrood.exceptions import DataclassException

from llm_reasoner.exceptions import LLMReasonerError

# %% concepts


@dataclass
class ConceptWrittenAsIndividual(LLMReasonerError, DataclassException):
    """
    A fragment's concept is written as the IRI of a term rather than as an instance of
    the term's class, as records were written before concepts became instances.
    """

    individual: str
    """
    The IRI of the fragment.
    """

    property_iri: str
    """
    The IRI of the concept's property.
    """

    term: str
    """
    The IRI written as the value.
    """

    def error_message(self) -> str:
        return (
            f"{self.individual} has {self.term} as its value of {self.property_iri}, "
            "which takes an instance of a class"
        )

    def suggest_correction(self) -> str:
        return (
            "generate the hypothesis again, which writes the concept as an anonymous "
            "instance of its class"
        )


@dataclass
class AmbiguousConceptType(LLMReasonerError, DataclassException):
    """
    A fragment's concept is an instance of several classes, or of none, so no single
    class names it.
    """

    individual: str
    """
    The IRI of the fragment.
    """

    property_iri: str
    """
    The IRI of the concept's property.
    """

    classes: list[str]
    """
    The IRIs of the classes the concept is an instance of.
    """

    def error_message(self) -> str:
        return (
            f"the value of {self.property_iri} of {self.individual} is an instance of "
            + (", ".join(self.classes) if self.classes else "no class")
            + ", not of exactly one class"
        )

    def suggest_correction(self) -> str:
        return "type the concept's anonymous instance with the one class it names"
