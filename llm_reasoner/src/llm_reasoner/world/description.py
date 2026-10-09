"""
The description of a world individual: its classes, its fields with what they mean and
hold, and the individuals holding it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from typing_extensions import List

from llm_reasoner.world.values import FieldValue


@dataclass(frozen=True)
class ClassSummary:
    """
    One class an individual belongs to.
    """

    identifier: str
    """
    The module and qualified name of the class, which tell it from others of its name.
    """

    name: str
    """
    The name of the class.
    """

    summary: str
    """
    The first line of its docstring, empty when it has none of its own.
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
    How references name it.
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
