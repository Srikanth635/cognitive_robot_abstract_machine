"""
What a query asks to be generated: the vocabulary class, the values the query fixes, and
the instruction the hypothesis is for.
"""

from __future__ import annotations

from dataclasses import dataclass

from krrood.entity_query_language.query.match import Match
from typing_extensions import Any

from llm_reasoner.exceptions import LLMReasonerError
from llm_reasoner.knowledge.generation.prompts import written_value
from llm_reasoner.knowledge.generation.schema import GenerationSchema
from llm_reasoner.knowledge.ontologies import OntologyModuleFile


@dataclass
class UnsupportedFixedValue(LLMReasonerError):
    """
    The query fixes a field to a nested match, which cannot be stated to the model as a
    value.
    """

    field_name: str
    """
    The field the query fixes.
    """

    def __post_init__(self):
        super().__init__(
            f"field '{self.field_name}' is fixed to a nested match; fix it to a value "
            "or leave it open"
        )


@dataclass(frozen=True)
class HypothesisRequest:
    """
    A request for a hypothesis, read from an underspecified match on a vocabulary class.
    """

    root_type: type
    """
    The vocabulary class the hypothesis is rooted in.
    """

    fixed_values: dict[str, Any]
    """
    The values the query fixes, by field name; fields left open are absent.
    """

    instruction: str
    """
    The action to describe, in plain words.
    """

    @classmethod
    def from_match(cls, expression: Match, instruction: str) -> HypothesisRequest:
        """
        :param expression: The match on the vocabulary class.
        :param instruction: The action to describe; the class's name when empty.
        :return: The request the match makes.
        :raises UnsupportedFixedValue: When the match fixes a field to a nested match.
        """
        fixed = {}
        for name, value in expression.kwargs.items():
            if value is Ellipsis:
                continue
            if isinstance(value, Match):
                raise UnsupportedFixedValue(field_name=name)
            fixed[name] = value
        return cls(
            root_type=expression.type,
            fixed_values=fixed,
            instruction=instruction or expression.type.__name__,
        )

    @property
    def module_files(self) -> set[OntologyModuleFile]:
        """
        The ontology modules the requested hypothesis's vocabulary uses.
        """
        return GenerationSchema.for_root(self.root_type).module_files()

    @property
    def written_fixed_values(self) -> dict[str, str]:
        """
        The fixed values as the model is told them and the store keys them.
        """
        return {name: written_value(value) for name, value in self.fixed_values.items()}
