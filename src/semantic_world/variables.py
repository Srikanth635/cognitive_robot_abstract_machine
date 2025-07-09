from __future__ import annotations

from enum import Enum
from functools import cached_property

from random_events.variable import Continuous
from sortedcontainers import SortedSet


class SpatialVariables(Enum):
    """
    Enum for spatial variables used in the semantic world. Used in the context of random events.
    """
    x = Continuous("x")
    y = Continuous("y")
    z = Continuous("z")

    @classmethod
    def xy(cls):
        return SortedSet([cls.x.value, cls.y.value])
