from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Generic, TypeVar, List, Optional

T = TypeVar('T')


class Derivatives(IntEnum):
    """
    Enumaration of interpretation for the order of derivativeson the spatial positions
    """
    position = 0
    velocity = 1
    acceleration = 2
    jerk = 3
    snap = 4
    crackle = 5
    pop = 6

    @classmethod
    def range(cls, start: Derivatives, stop: Derivatives, step: int = 1):
        """
        Includes stop!
        """
        return [item for item in cls if start <= item <= stop][::step]


@dataclass
class DerivativeMap(Generic[T]):
    data: List[Optional[T]] = field(default_factory=lambda: [None] * len(Derivatives))

    @property
    def position(self) -> T:
        return self.data[Derivatives.position]

    @position.setter
    def position(self, value: T):
        self.data[Derivatives.position] = value

    @property
    def velocity(self) -> T:
        return self.data[Derivatives.velocity]

    @velocity.setter
    def velocity(self, value: T):
        self.data[Derivatives.velocity] = value

    @property
    def acceleration(self) -> T:
        return self.data[Derivatives.acceleration]

    @acceleration.setter
    def acceleration(self, value: T):
        self.data[Derivatives.acceleration] = value

    @property
    def jerk(self) -> T:
        return self.data[Derivatives.jerk]

    @jerk.setter
    def jerk(self, value: T):
        self.data[Derivatives.jerk] = value
