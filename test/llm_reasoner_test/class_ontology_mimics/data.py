"""
The data structures and enumerations the designators' fields use, which the ontology
describes in full.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Side(Enum):
    """
    A side of the robot.
    """

    LEFT = "left"
    RIGHT = "right"


class Approach(Enum):
    """
    Where a hand comes from.
    """

    FRONT = "front"
    TOP = "top"


@dataclass
class Grasp:
    """
    How an object is held.
    """

    approach: Approach
    """
    Where the hand comes from.
    """

    offset: float = 0.1
    """
    How far the hand stops from the object, in metres.
    """
