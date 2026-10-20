"""
Declarations that must be indexed without executing module statements.
"""

from __future__ import annotations

from enum import StrEnum
from typing_extensions import TYPE_CHECKING

from .base import Same as Parent, helper

if TYPE_CHECKING:
    from second.base import Same as Other

# %% import guard

raise RuntimeError("Source indexing must never import this module.")

# %% nested and generic declarations


class Child(Parent[int]):
    """
    Bind the base's generic parameter.
    """

    member: int = 2
    """
    A documented field.
    """

    def act(self, argument: int) -> int:
        """
        Call the imported helper.
        """
        return helper(argument)

    class Nested:
        """
        Retain its containing lexical scope.
        """


class Choice(StrEnum):
    """
    Possible fixture states.
    """

    READY = "ready"


def use_alias(argument: int) -> int:
    """
    Refer to an imported function.
    """
    return helper(argument)


def shadow_alias(helper: object) -> object:
    """
    A parameter shadows the imported callable.
    """
    return helper()


def parenthesized() -> Parent[int]:
    """
    Retain a generic return annotation.
    """
    return Child()
