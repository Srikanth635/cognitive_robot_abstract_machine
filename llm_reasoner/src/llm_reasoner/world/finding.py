"""
Finding the individuals of a kind whose fields meet conditions, as a krrood query that
is also stated in English.
"""

from __future__ import annotations

import operator
from collections import Counter
from dataclasses import dataclass
from enum import StrEnum
from operator import attrgetter

from krrood.entity_query_language.core.base_expressions import SymbolicExpression
from krrood.entity_query_language.factories import an, contains, entity, variable
from krrood.entity_query_language.verbalization.pipeline import verbalize_expression
from typing_extensions import Any, Callable, List, Optional, Sequence

from llm_reasoner.world.naming import HandleRegistry
from llm_reasoner.world.sources import WorldSource
from llm_reasoner.world.values import FieldValue


class Comparison(StrEnum):
    """
    How the value at a field path is compared with a given value.
    """

    EQUAL = "equal"
    NOT_EQUAL = "not equal"
    LESS = "less"
    LESS_OR_EQUAL = "less or equal"
    GREATER = "greater"
    GREATER_OR_EQUAL = "greater or equal"
    CONTAINS = "contains"
    """
    The value at the path contains the given value, such as a name containing a word.
    """

    @property
    def operation(self) -> Callable[[Any, Any], Any]:
        """
        The operation comparing the value at the path with the given one.
        """
        return COMPARISON_OPERATIONS[self]


COMPARISON_OPERATIONS = {
    Comparison.EQUAL: operator.eq,
    Comparison.NOT_EQUAL: operator.ne,
    Comparison.LESS: operator.lt,
    Comparison.LESS_OR_EQUAL: operator.le,
    Comparison.GREATER: operator.gt,
    Comparison.GREATER_OR_EQUAL: operator.ge,
    Comparison.CONTAINS: contains,
}
"""
The operation of each comparison.
"""


@dataclass(frozen=True)
class Condition:
    """
    A condition on the value at a field path of an individual.
    """

    path: str
    """
    The fields leading to the value, separated by dots, such as ``inertial.mass``.
    """

    comparison: Comparison
    """
    How the value is compared.
    """

    value: Any
    """
    The value compared with.
    """


@dataclass(frozen=True)
class IndividualSummary:
    """
    An individual a search found.
    """

    handle: str
    """
    Its handle.
    """

    type_name: str
    """
    The name of its kind.
    """

    reached: Optional[FieldValue] = None
    """
    What a path followed from it reaches, when the search follows one.
    """


@dataclass(frozen=True)
class KindCount:
    """
    How many individuals a search found of one kind.
    """

    type_name: str
    """
    The name of the kind.
    """

    count: int
    """
    How many were found of exactly this kind.
    """


@dataclass(frozen=True)
class FoundIndividuals:
    """
    What a search found.
    """

    query_in_english: str
    """
    The search, stated in English.
    """

    total: int
    """
    How many individuals meet it.
    """

    kinds: List[KindCount]
    """
    How many of them are of each kind, most numerous first, all of them counted.
    """

    individuals: List[IndividualSummary]
    """
    The first of them, within the limit of the search.
    """


@dataclass(frozen=True)
class IndividualFinder:
    """
    Finds individuals of a kind by conditions on their fields.
    """

    source: WorldSource
    """
    The world searched.
    """

    handles: HandleRegistry
    """
    Gives the individuals found their handles.
    """

    def query(self, kind: type, conditions: Sequence[Condition]) -> SymbolicExpression:
        """
        :return: The krrood query for the individuals of a kind meeting the conditions.
        """
        found = variable(kind, domain=self.source.individuals_of(kind))
        constraints = [
            condition.comparison.operation(
                attrgetter(condition.path)(found), condition.value
            )
            for condition in conditions
        ]
        query = an(entity(found))
        return query.where(*constraints) if constraints else query

    def find(
        self, kind: type, conditions: Sequence[Condition], limit: int
    ) -> FoundIndividuals:
        """
        :return: The individuals of a kind meeting the conditions, the first ``limit`` of
            them summarised.
        """
        query = self.query(kind, conditions)
        individuals = list(query.evaluate())
        counts = Counter(type(individual).__name__ for individual in individuals)
        return FoundIndividuals(
            query_in_english=verbalize_expression(query),
            total=len(individuals),
            kinds=[
                KindCount(type_name=type_name, count=count)
                for type_name, count in sorted(
                    counts.items(), key=lambda entry: (-entry[1], entry[0])
                )
            ],
            individuals=[
                IndividualSummary(
                    handle=self.handles.handle_of(individual),
                    type_name=type(individual).__name__,
                )
                for individual in individuals[:limit]
            ],
        )
