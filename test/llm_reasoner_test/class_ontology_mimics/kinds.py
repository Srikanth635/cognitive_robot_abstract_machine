"""
The kinds of things designators act on.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from typing_extensions import Generic, List, TypeVarTuple, Unpack


@dataclass
class SemanticAnnotation:
    """
    What a thing of the world is.
    """


@dataclass
class Container(SemanticAnnotation):
    """
    Something that holds other things.
    """

    rim_height: float = 0.0
    """
    How high the rim is above the ground, in metres.
    """


@dataclass
class Vessel(Container):
    """
    A container that can be poured from.
    """


TSensors = TypeVarTuple("TSensors")
"""
The kinds of sensor a sensing thing carries.
"""


@dataclass
class Camera(SemanticAnnotation):
    """
    A sensor that takes pictures.
    """


@dataclass
class SensingThing(SemanticAnnotation, Generic[Unpack[TSensors]]):
    """
    Something that carries sensors of some kinds.
    """

    sensors: List[Unpack[TSensors]] = field(default_factory=list)
    """
    The sensors it carries.
    """


@dataclass
class WatchingThing(SensingThing[Camera]):
    """
    Something that carries cameras.
    """


@dataclass
class MovingSensingThing(SensingThing[Unpack[TSensors]]):
    """
    Something that moves and carries sensors of some kinds.
    """
