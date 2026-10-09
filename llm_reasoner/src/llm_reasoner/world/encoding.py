"""
Writing the world injector's answers as JSON for a language model.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from enum import Enum

from typing_extensions import Any, Mapping

from llm_reasoner.world.classes import DocumentedClass, is_individual
from llm_reasoner.world.values import FieldValue

FORM_KEY = "form"
"""
The key naming the form of a value, so a model can tell a plain value from a reference
to an individual or from a value depending on the world state.
"""

JSON_SCALARS = (str, int, float, bool, type(None))
"""
Values JSON writes as they are.
"""

JSON_ARRAYS = (list, tuple, set, frozenset)
"""
Collections JSON writes as arrays.
"""


@dataclass(frozen=True)
class AnswerEncoder:
    """
    Turns the world injector's typed answers into values JSON can write.
    """

    def encode(self, answer: Any) -> Any:
        """
        :return: The answer as nested dictionaries, lists and scalars: a result as its
            fields, a field value tagged with its form, an enumeration member as its
            value, and anything else as its representation.
        """
        if isinstance(answer, Enum):
            return answer.value
        if isinstance(answer, JSON_SCALARS):
            return answer
        if is_individual(answer):
            return self.encode_fields(answer)
        if isinstance(answer, Mapping):
            return {str(key): self.encode(value) for key, value in answer.items()}
        if isinstance(answer, JSON_ARRAYS):
            return [self.encode(item) for item in answer]
        return repr(answer)

    def encode_fields(self, answer: Any) -> Mapping[str, Any]:
        """
        :return: The fields of a result, a field value tagged with its form first.
        """
        encoded = {
            declared.name: self.encode(DocumentedClass.value_of(answer, declared))
            for declared in fields(answer)
        }
        if isinstance(answer, FieldValue):
            return {FORM_KEY: type(answer).__name__, **encoded}
        return encoded
