"""
Following a path of fields from an individual to what it reaches.
"""

from __future__ import annotations

from dataclasses import Field, dataclass

from typing_extensions import Any, Optional, Tuple

from llm_reasoner.world.classes import DocumentedClass, is_individual
from llm_reasoner.world.exceptions import UnknownField

PATH_SEPARATOR = "."
"""
What separates the fields of a path.
"""


@dataclass(frozen=True)
class FieldPath:
    """
    Fields leading from an individual to a value, such as ``inertial.mass``.
    """

    path: str
    """
    The field names, separated by dots.
    """

    def reach(self, individual: Any) -> Tuple[Any, Optional[Field]]:
        """
        :return: What the path reaches, and the field holding it.
        :raises UnknownField: When an individual on the way lacks the next content field.
        """
        reached, holding_field = individual, None
        for name in self.path.split(PATH_SEPARATOR):
            declared = (
                DocumentedClass.of(type(reached)).content_field_named(reached, name)
                if is_individual(reached)
                else None
            )
            if declared is None:
                raise UnknownField(type_name=type(reached).__name__, field_name=name)
            reached, holding_field = (
                DocumentedClass.value_of(reached, declared),
                declared,
            )
        return reached, holding_field
