"""
Designators: actions a robot can perform, on a base that keeps bookkeeping fields, one
with a mixin adding a field, and one whose field names a type imported only for type
checking.
"""

from __future__ import annotations

from abc import ABC
from dataclasses import dataclass, field

from typing_extensions import Generic, List, Optional, TYPE_CHECKING, TypeVar

from .data import Grasp, Side
from .geometry import Pose
from .kinds import Container, Vessel

if TYPE_CHECKING:
    from .nowhere import Mystery

TContainer = TypeVar("TContainer", bound=Container)
"""
The kind of container an action that stores things stores them in.
"""


@dataclass
class Designator:
    """
    Something a plan performs.
    """

    plan_node: Optional[str] = field(kw_only=True, default=None)
    """
    The plan node that manages the designator.
    """


@dataclass
class ActionDescription(Designator, ABC):
    """
    An action a robot can perform.
    """


@dataclass
class HasSpeedLimit:
    """
    Something whose motion can be limited in speed.
    """

    speed_limit: Optional[float] = field(kw_only=True, default=None)
    """
    The fastest the effector may move, in metres per second.

    ``None`` leaves it to the controller.
    """


@dataclass
class PourAction(ActionDescription, HasSpeedLimit):
    """
    Pour from a held vessel into a container.

    The vessel is tilted next to the container's rim.
    """

    source: Vessel
    """
    The vessel poured from.
    """

    target: Container
    """
    The container poured into.
    """

    side: Side
    """
    The side of the robot the vessel is held on.
    """

    grasp: Grasp
    """
    How the vessel is held.
    """

    tilt: float = 1.85
    """
    How far the vessel is tilted, in radians.

    Measured from upright.
    """

    waypoints: List[Pose] = field(default_factory=list)
    """
    Where the vessel passes on its way to the container.
    """

    _attempts: int = 0
    """
    How often the pour was tried.
    """


@dataclass
class NavigateAction(ActionDescription):
    """
    Move the robot to where it should stand.
    """

    target_location: Pose
    """
    Where the robot should stand.
    """


@dataclass
class InspectAction(ActionDescription):
    """
    Look closely at something.
    """

    target_location: Pose
    """
    Where to look from.
    """

    clue: Optional[Mystery] = None
    """
    What to look for.
    """


@dataclass
class StoreAction(ActionDescription, Generic[TContainer]):
    """
    Put something away in a container of some kind.
    """

    place: TContainer
    """
    The container to put it in.
    """


@dataclass
class CarefulStoreAction(StoreAction[TContainer]):
    """
    Put something away gently, in a container of some kind.
    """


@dataclass
class PantryStoreAction(StoreAction[Vessel]):
    """
    Put something away in a vessel in the pantry.
    """


@dataclass
class _DraftAction(ActionDescription):
    """
    An action not meant to be used.
    """
