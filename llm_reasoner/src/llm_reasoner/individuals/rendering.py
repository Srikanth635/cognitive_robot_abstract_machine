"""
Writing the description of a world individual as indented text for a prompt.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from functools import singledispatchmethod

from typing_extensions import List

from llm_reasoner.exceptions import LLMReasonerError
from llm_reasoner.individuals.description import (
    CollectionValue,
    FieldDescription,
    FieldValue,
    IndividualDescription,
    NestedIndividual,
    NumericValue,
    PlainValue,
    Referrer,
    RepeatedIndividual,
    UndescribedValue,
    WorldStateDependentValue,
)

INDENT = "  "
"""
The indentation of one level of nesting.
"""

ITEM_MARK = "- "
"""
The mark before an item of a collection.
"""

LINEAGE_SEPARATOR = " < "
"""
What separates a class from the class it specialises.
"""

# %% exceptions


@dataclass
class UnwritableValue(LLMReasonerError):
    """
    A field value of a form the renderer has no way of writing.
    """

    value: FieldValue
    """
    The value.
    """

    def __post_init__(self):
        super().__init__(f"No way of writing a {type(self.value).__name__}")


# %% renderer


@dataclass(frozen=True)
class IndividualRenderer:
    """
    Writes descriptions of individuals as indented text: a heading naming the individual
    and its classes, then one line per field, nested individuals indented below the
    field holding them.
    """

    line_budget: int = 40
    """
    The most lines the text may take, the last of which then says how many were left
    out.
    """

    def render(self, description: IndividualDescription) -> str:
        """
        :return: The description as text, within the line budget.
        """
        lines = [self.heading(description)]
        lines.extend(self.referrer_line(referrer) for referrer in description.referrers)
        lines.extend(self.field_lines(description, level=1))
        return "\n".join(self.within_budget(lines))

    @staticmethod
    def referrer_line(referrer: Referrer) -> str:
        """
        :return: The line naming an individual that holds the described one, which often
            says more about what it is than its own fields.
        """
        return (
            f"{INDENT}held as {referrer.field_name} of a {referrer.type_name}"
            + IndividualRenderer.annotation(referrer.summary)
        )

    def within_budget(self, lines: List[str]) -> List[str]:
        """
        :return: The lines, the overflow replaced by a line counting it.
        """
        if len(lines) <= self.line_budget:
            return lines
        kept = self.line_budget - 1
        return lines[:kept] + [self.omission(len(lines) - kept)]

    @staticmethod
    def omission(count: int) -> str:
        """
        :return: The line standing in for lines left out.
        """
        return f"… {count} more lines"

    @staticmethod
    def heading(description: IndividualDescription) -> str:
        """
        :return: The identifier of an individual, its classes and the summary of its own
            class. An ancestor's summary is not borrowed, since a general ancestor
            describes the individual too vaguely to tell it from others.
        """
        names = LINEAGE_SEPARATOR.join(summary.name for summary in description.lineage)
        return f"{description.identifier} [{names}]" + IndividualRenderer.annotation(
            description.lineage[0].summary
        )

    @staticmethod
    def annotation(text: str) -> str:
        """
        :return: Explanatory text set off after a line, or nothing when there is none.
        """
        return f"   # {text}" if text else ""

    def field_lines(self, description: IndividualDescription, level: int) -> List[str]:
        """
        :return: The lines of every field of a description, at a level of nesting.
        """
        lines: List[str] = []
        for described in description.fields:
            lines.extend(self.lines_of(described.value, described, level))
        return lines

    @singledispatchmethod
    def lines_of(
        self, value: FieldValue, described: FieldDescription, level: int
    ) -> List[str]:
        """
        :return: The lines of a field holding a value given on one line.
        """
        return [
            f"{INDENT * level}{described.name} = {self.inline(value)}"
            + self.annotation(described.meaning)
        ]

    @lines_of.register
    def _(
        self, value: NestedIndividual, described: FieldDescription, level: int
    ) -> List[str]:
        return [
            f"{INDENT * level}{described.name}: {self.heading(value.description)}"
        ] + self.field_lines(value.description, level + 1)

    @lines_of.register
    def _(
        self, value: CollectionValue, described: FieldDescription, level: int
    ) -> List[str]:
        lines = [
            f"{INDENT * level}{described.name}: {value.size} item(s)"
            + self.annotation(described.meaning)
        ]
        for item in value.items:
            if isinstance(item, NestedIndividual):
                lines.append(
                    f"{INDENT * (level + 1)}{ITEM_MARK}{self.heading(item.description)}"
                )
                lines.extend(self.field_lines(item.description, level + 2))
            else:
                lines.append(f"{INDENT * (level + 1)}{ITEM_MARK}{self.inline(item)}")
        return lines

    @singledispatchmethod
    def inline(self, value: FieldValue) -> str:
        """
        :return: A value written on one line.
        """
        raise UnwritableValue(value=value)

    @inline.register
    def _(self, value: PlainValue) -> str:
        written = (
            f"{type(value.value).__name__}.{value.value.name}"
            if isinstance(value.value, Enum)
            else repr(value.value)
        )
        return written + (" (default)" if value.is_default else "")

    @inline.register
    def _(self, value: NumericValue) -> str:
        return "(" + ", ".join(f"{number:g}" for number in value.values) + ")"

    @inline.register
    def _(self, value: WorldStateDependentValue) -> str:
        return "depends on the world state: " + ", ".join(value.variables)

    @inline.register
    def _(self, value: RepeatedIndividual) -> str:
        return f"see {value.identifier}"

    @inline.register
    def _(self, value: UndescribedValue) -> str:
        return f"<{value.type_name}>"

    @inline.register
    def _(self, value: NestedIndividual) -> str:
        return self.heading(value.description)

    @inline.register
    def _(self, value: CollectionValue) -> str:
        return f"{value.size} item(s)"
