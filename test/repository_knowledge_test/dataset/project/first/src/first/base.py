"""
A generic base and callable used by another module.
"""

from dataclasses import dataclass
from typing_extensions import Generic, TypeVar

# %% declarations

Value = TypeVar("Value")


@dataclass
class Same(Generic[Value]):
    """
    Keep an annotated value.
    """

    stored_value: Value
    """
    The contained value.
    """


def helper(argument: int) -> int:
    """
    Return the supplied integer.
    """
    return argument
