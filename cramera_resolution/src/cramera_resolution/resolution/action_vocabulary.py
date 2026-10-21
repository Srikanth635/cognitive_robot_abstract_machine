"""
The action designators a query can name, and the parameters each takes, as the names a
query box offers while it is being typed.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from enum import StrEnum

from coraplex.robot_plans.actions.base import ActionDescription
from typing_extensions import Any, Dict, List

from cramera_resolution.resolution.designators import QUERY_VOCABULARY

# %% what is offered


class VocabularyKind(StrEnum):
    """
    What a name offered is of.
    """

    CLASS = "class"
    """
    An action designator.
    """

    FIELD = "field"
    """
    A parameter of an action designator.
    """


class VocabularyKey(StrEnum):
    """
    The keys of a name offered, as JSON data.
    """

    NAME = "name"
    """
    The name that is typed.
    """

    KIND = "kind"
    """
    What the name is of.
    """

    DETAIL = "detail"
    """
    A parameter's type and default; what an action designator is.
    """

    MODULE = "module"
    """
    Where the name comes from.
    """

    OWNER = "owner"
    """
    The action designator a parameter belongs to.
    """


@dataclass(frozen=True)
class VocabularyEntry:
    """
    One name a query box offers.
    """

    name: str
    """
    The name that is typed.
    """

    kind: VocabularyKind
    """
    What the name is of.
    """

    detail: str
    """
    A parameter's type and default; what an action designator is.
    """

    module: str
    """
    Where the name comes from.
    """

    owner: str = ""
    """
    The action designator a parameter belongs to, empty for the designator itself.
    """

    def payload(self) -> Dict[str, Any]:
        """
        :return: The entry as JSON data.
        """
        return {
            VocabularyKey.NAME: self.name,
            VocabularyKey.KIND: self.kind,
            VocabularyKey.DETAIL: self.detail,
            VocabularyKey.MODULE: self.module,
            VocabularyKey.OWNER: self.owner,
        }


@dataclass(frozen=True)
class ActionDesignatorVocabulary:
    """
    An action designator and the parameters it takes.
    """

    action: type[ActionDescription]
    """
    The action designator's class.
    """

    @classmethod
    def every(cls) -> List[ActionDesignatorVocabulary]:
        """
        :return: The vocabulary of each action designator a designator's query reads.
        """
        return [
            cls(action=item)
            for item in QUERY_VOCABULARY
            if issubclass(item, ActionDescription)
        ]

    def entries(self) -> List[VocabularyEntry]:
        """
        :return: The action designator, then each parameter it is constructed with.
        """
        return [
            VocabularyEntry(
                name=self.action.__name__,
                kind=VocabularyKind.CLASS,
                detail="action designator",
                module=self.action.__module__,
            ),
            *(
                VocabularyEntry(
                    name=field.name,
                    kind=VocabularyKind.FIELD,
                    detail=self.detail_of(field),
                    module=self.action.__name__,
                    owner=self.action.__name__,
                )
                for field in dataclasses.fields(self.action)
                if field.init
            ),
        ]

    @staticmethod
    def detail_of(field: dataclasses.Field) -> str:
        """
        :return: The type of a parameter, and its default when it has one.
        """
        annotation = field.type
        type_text = (
            annotation.__name__
            if isinstance(annotation, type)
            else str(annotation).replace("typing.", "")
        )
        if field.default is dataclasses.MISSING:
            return type_text
        return f"{type_text} = {field.default}"
