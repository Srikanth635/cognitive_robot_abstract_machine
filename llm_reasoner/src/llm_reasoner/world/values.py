"""
What a field of a world individual holds, in forms a language model can read.
"""

from __future__ import annotations

from abc import ABC
from dataclasses import dataclass

from typing_extensions import TYPE_CHECKING, Any, Optional, Tuple

if TYPE_CHECKING:
    from llm_reasoner.world.description import IndividualDescription


class FieldValue(ABC):
    """
    What a field of an individual holds, in a form a language model can read.
    """


@dataclass(frozen=True)
class PlainValue(FieldValue):
    """
    A number, text, truth value, enumeration member or nothing.
    """

    value: Any
    """
    The value as stored.
    """

    is_default: bool
    """
    Whether the value is the default its field declares, rather than one that was set.
    """


@dataclass(frozen=True)
class NumericValue(FieldValue):
    """
    A constant symbolic value, such as a point or a matrix, as its numbers.
    """

    values: Tuple[float, ...]
    """
    The numbers, row by row.
    """


@dataclass(frozen=True)
class WorldStateDependentValue(FieldValue):
    """
    A symbolic value that depends on the state of the world, such as a pose moved by a
    joint.
    """

    variables: Tuple[str, ...]
    """
    The names of the state variables it depends on.
    """

    current: Optional[Tuple[float, ...]] = None
    """
    Its numbers in the current state of the world, row by row; ``None`` when the world
    cannot read one of its variables.
    """


@dataclass(frozen=True)
class NestedIndividual(FieldValue):
    """
    An individual held by the field, described in turn.
    """

    description: IndividualDescription
    """
    Its description.
    """


@dataclass(frozen=True)
class IndividualReference(FieldValue):
    """
    An individual named rather than described: one described elsewhere in the same
    description, or one beyond the limits of the description, to be asked about next.
    """

    identifier: str
    """
    The identifier naming it.
    """

    type_name: str
    """
    The name of its kind.
    """


@dataclass(frozen=True)
class CollectionValue(FieldValue):
    """
    A collection held by the field.
    """

    size: int
    """
    How many items it holds.
    """

    items: Tuple[FieldValue, ...]
    """
    Its first items, described.
    """


@dataclass(frozen=True)
class UndescribedValue(FieldValue):
    """
    A value of no form the world injector reads, named only by its type.
    """

    type_name: str
    """
    The name of its type.
    """
