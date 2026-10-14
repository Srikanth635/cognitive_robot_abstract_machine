"""
Classes whose attribute documentation the documentation reader is tested against.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

# %% enumerations


class Colour(StrEnum):
    """
    A colour of paint.
    """

    RED = "red"
    """
    The colour of blood.
    """

    GREEN = "green"
    """The colour of grass."""

    BLUE = "blue"


# %% dataclasses


@dataclass
class Vehicle:
    """
    Something that moves people.
    """

    wheels: int = 4
    """
    How many wheels touch the ground.
    """

    passengers: list[str] = field(default_factory=list)
    """
    Names of the people aboard.

    Listed in boarding order.
    """

    registration: str = ""


@dataclass
class ElectricVehicle(Vehicle):
    """
    A vehicle driven by a battery.
    """

    wheels: int = 3
    """
    How many wheels touch the ground on this model.
    """

    range_in_kilometres: float = 0.0
    """
    Distance on a full charge.
    """


@dataclass
class UndocumentedVehicle:
    value: int = 0


# %% nesting


class Outer:
    """
    A class that holds another.
    """

    @dataclass
    class Inner:
        """
        A class declared inside another.
        """

        depth: int = 0
        """
        How deep the nesting goes.
        """


def build_local_class() -> type:
    """
    Return a class declared inside a function.
    """

    @dataclass
    class Local:
        """
        A class declared inside a function.
        """

        size: int = 0
        """
        How large it is.
        """

    return Local


# %% strings that document nothing


class StrayString:
    """
    A class whose strings do not follow an attribute.
    """

    def method(self) -> None:
        pass

    """
    Follows a method, so it documents no attribute.
    """

    value = 1
