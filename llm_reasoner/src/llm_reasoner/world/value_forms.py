"""
The forms in which the world injector reads what a field holds.

Each form recognises the values it reads, so a new form is added by adding a reader
rather than by changing the walk.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import MISSING, Field, dataclass
from itertools import islice

from krrood.symbolic_math.symbolic_math import SymbolicMathType
from typing_extensions import TYPE_CHECKING, Any, Optional, Tuple

from llm_reasoner.world.classes import PLAIN_TYPES, is_individual
from llm_reasoner.world.values import (
    CollectionValue,
    FieldValue,
    IndividualReference,
    NestedIndividual,
    NumericValue,
    PlainValue,
    WorldStateDependentValue,
)

if TYPE_CHECKING:
    from llm_reasoner.world.describer import IndividualWalk

COLLECTION_TYPES = (list, tuple, set, frozenset)
"""
Types whose values are given as their size and first items.
"""


class ValueForm(ABC):
    """
    One form in which a held value is read.
    """

    @abstractmethod
    def reads(self, held: Any) -> bool:
        """
        :return: Whether this form reads the value.
        """

    @abstractmethod
    def read(
        self, held: Any, walk: IndividualWalk, depth: int, declared: Optional[Field]
    ) -> FieldValue:
        """
        :param held: The value.
        :param walk: The walk reading it.
        :param depth: How many levels below the first individual its holder is.
        :param declared: The field holding it, when a field holds it directly.
        :return: The value in this form.
        """


@dataclass(frozen=True)
class PlainForm(ValueForm):
    """
    Numbers, texts, truth values, enumeration members and nothing, given as they are.
    """

    def reads(self, held: Any) -> bool:
        return isinstance(held, PLAIN_TYPES)

    def read(
        self, held: Any, walk: IndividualWalk, depth: int, declared: Optional[Field]
    ) -> FieldValue:
        return PlainValue(value=held, is_default=self.is_default(held, declared))

    @staticmethod
    def is_default(held: Any, declared: Optional[Field]) -> bool:
        """
        :return: Whether a plain value is the default its field declares. A default made
            by a factory is not compared, since making one could create individuals.
        """
        return (
            declared is not None
            and declared.default is not MISSING
            and held == declared.default
        )


@dataclass(frozen=True)
class SymbolicForm(ValueForm):
    """
    Symbolic values: constants as their numbers, varying ones by the state variables
    they depend on and their numbers in the current state.
    """

    def reads(self, held: Any) -> bool:
        return isinstance(held, SymbolicMathType)

    def read(
        self, held: Any, walk: IndividualWalk, depth: int, declared: Optional[Field]
    ) -> FieldValue:
        if held.is_constant():
            return NumericValue(values=self.numbers(held))
        variables = held.free_variables()
        readable = all(variable.resolve is not None for variable in variables)
        return WorldStateDependentValue(
            variables=tuple(variable.name for variable in variables),
            current=self.numbers(held) if readable else None,
        )

    @staticmethod
    def numbers(held: SymbolicMathType) -> Tuple[float, ...]:
        """
        :return: The numbers of a symbolic value in the current state, row by row.
        """
        return tuple(held.evaluate().flatten().tolist())


@dataclass(frozen=True)
class IndividualForm(ValueForm):
    """
    Individuals, described in turn within the limits of the walk and referred to by
    identifier beyond them or when described already.
    """

    def reads(self, held: Any) -> bool:
        return is_individual(held)

    def read(
        self, held: Any, walk: IndividualWalk, depth: int, declared: Optional[Field]
    ) -> FieldValue:
        if not walk.may_describe(held, depth):
            return IndividualReference(
                identifier=walk.naming.identifier_of(held),
                type_name=type(held).__name__,
            )
        return NestedIndividual(description=walk.describe(held, depth + 1))


@dataclass(frozen=True)
class CollectionForm(ValueForm):
    """
    Collections, given as their size and their first items.
    """

    def reads(self, held: Any) -> bool:
        return isinstance(held, COLLECTION_TYPES)

    def read(
        self, held: Any, walk: IndividualWalk, depth: int, declared: Optional[Field]
    ) -> FieldValue:
        return CollectionValue(
            size=len(held),
            items=tuple(
                walk.value(item, depth)
                for item in islice(held, walk.limits.items_per_collection)
            ),
        )


DEFAULT_VALUE_FORMS: Tuple[ValueForm, ...] = (
    PlainForm(),
    SymbolicForm(),
    IndividualForm(),
    CollectionForm(),
)
"""
The forms values are read in, the first that reads a value reading it.
"""
