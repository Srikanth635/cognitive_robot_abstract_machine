"""
Following a path of fields from an individual to what it reaches.
"""

from __future__ import annotations

from dataclasses import Field, dataclass, replace

from typing_extensions import Any, Optional, Tuple

from llm_reasoner.world.classes import DocumentedClass, is_individual
from llm_reasoner.world.exceptions import UnknownField

PATH_SEPARATOR = "."
"""
What separates the fields of a path.
"""


@dataclass(frozen=True)
class PathEnd:
    """
    Where a path of fields from an individual ends.
    """

    value: Any
    """
    What the path reaches; ``None`` when a field on the way holds nothing.
    """

    holding_field: Optional[Field] = None
    """
    The field holding the value, when the path reached it through one.
    """

    missing_field: Optional[UnknownField] = None
    """
    The field the path needed but a value on the way lacks, when it stopped there.
    """

    @property
    def is_reached(self) -> bool:
        """
        Whether the path was followed to its last field.
        """
        return self.missing_field is None


@dataclass(frozen=True)
class FieldPath:
    """
    Fields leading from an individual to a value, such as ``inertial.mass``.
    """

    path: str
    """
    The field names, separated by dots.
    """

    def walk(self, individual: Any) -> PathEnd:
        """
        :return: Where the path ends from an individual: what it reaches, or the field
            it stopped at.
        """
        end = PathEnd(value=individual)
        for name in self.path.split(PATH_SEPARATOR):
            if end.value is None:
                return PathEnd(value=None)
            declared = (
                DocumentedClass.of(type(end.value)).content_field_named(end.value, name)
                if is_individual(end.value)
                else None
            )
            if declared is None:
                return replace(
                    end,
                    missing_field=UnknownField(
                        type_name=type(end.value).__name__, field_name=name
                    ),
                )
            end = PathEnd(
                value=DocumentedClass.value_of(end.value, declared),
                holding_field=declared,
            )
        return end

    def reach(self, individual: Any) -> Tuple[Any, Optional[Field]]:
        """
        :return: What the path reaches, and the field holding it; nothing when a field on
            the way holds nothing.
        :raises UnknownField: When an individual on the way lacks the next content field.
        """
        end = self.walk(individual)
        if not end.is_reached:
            raise end.missing_field
        return end.value, end.holding_field
