"""
The description of a world individual: its classes, its fields with what they mean and
hold, and the individuals holding it.
"""

from __future__ import annotations

from abc import ABC
from dataclasses import dataclass, field

from typing_extensions import Any, List, Tuple

# %% values


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
    joint, and so has no numbers of its own.
    """

    variables: Tuple[str, ...]
    """
    The names of the state variables it depends on.
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
class RepeatedIndividual(FieldValue):
    """
    An individual already described elsewhere in the same description.
    """

    identifier: str
    """
    The identifier of its description.
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
    A value named only by its type: one of no readable form, or an individual beyond the
    limits of the walk.
    """

    type_name: str
    """
    The name of its type.
    """


# %% description


@dataclass(frozen=True)
class ClassSummary:
    """
    One class an individual belongs to.
    """

    name: str
    """
    The name of the class.
    """

    summary: str
    """
    The first line of its docstring, empty when it has none.
    """


@dataclass(frozen=True)
class FieldDescription:
    """
    One field of an individual.
    """

    name: str
    """
    The name of the field.
    """

    meaning: str
    """
    What the field holds, from its docstring; empty when it has none.
    """

    value: FieldValue
    """
    What it holds.
    """


@dataclass(frozen=True)
class Referrer:
    """
    An individual holding the described one in a field.
    """

    type_name: str
    """
    The name of its class.
    """

    field_name: str
    """
    The field holding the described individual.
    """

    summary: str
    """
    The first line of the docstring of its class.
    """


@dataclass(frozen=True)
class IndividualDescription:
    """
    A world individual as a language model reads it.
    """

    identifier: str
    """
    How references within the same description name it.
    """

    lineage: List[ClassSummary]
    """
    Its class followed by the classes that class specialises, most specific first.
    """

    fields: List[FieldDescription]
    """
    Its fields that hold content.
    """

    referrers: List[Referrer] = field(default_factory=list)
    """
    The individuals holding it, found for the individual a description starts from.
    """

    @property
    def type_name(self) -> str:
        """
        The name of its class.
        """
        return self.lineage[0].name
