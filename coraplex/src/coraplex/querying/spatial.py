from __future__ import annotations

from dataclasses import dataclass

from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.world_entity import (
    Body,
    KinematicStructureEntity,
)
from typing_extensions import List, Optional, Tuple

# %% thresholds

NEGLIGIBLE_DISTANCE = 0.05
"""Displacement in metres below which a direction is not worth naming."""


# %% relative position


@dataclass
class RelativePosition:
    """
    Where one entity lies as seen from another.

    Designators state positions as world coordinates, which leaves a model to subtract
    vectors -- an operation it is unreliable at and which decides parameters such as
    which arm to use. Naming the relation directly removes that arithmetic from the
    model's job.
    """

    forward: float
    """Metres ahead of the observer along its own x-axis, negative for behind."""

    left: float
    """Metres to the observer's left along its own y-axis, negative for right."""

    up: float
    """Metres above the observer along its own z-axis, negative for below."""

    @property
    def horizontal_distance(self) -> float:
        """:return: Distance in the observer's ground plane, in metres."""
        return (self.forward**2 + self.left**2) ** 0.5

    @property
    def is_ahead(self) -> bool:
        """:return: Whether the target lies in front of the observer."""
        return self.forward > 0.0

    @property
    def side(self) -> Optional[str]:
        """:return: Which side the target lies on, or ``None`` when centred."""
        if abs(self.left) < NEGLIGIBLE_DISTANCE:
            return None
        return "left" if self.left > 0.0 else "right"

    def describe(self) -> str:
        """
        :return: The relation in words, naming only the axes that carry a meaningful
            displacement.
        """
        parts: List[str] = []
        if abs(self.forward) >= NEGLIGIBLE_DISTANCE:
            direction = "ahead" if self.is_ahead else "behind"
            parts.append(f"{abs(self.forward):.2f} m {direction}")
        if self.side is not None:
            parts.append(f"{abs(self.left):.2f} m to the {self.side}")
        if abs(self.up) >= NEGLIGIBLE_DISTANCE:
            direction = "above" if self.up > 0.0 else "below"
            parts.append(f"{abs(self.up):.2f} m {direction}")
        if not parts:
            return "at the same place"
        return ", ".join(parts)


def relative_position(
    world: World, observer: KinematicStructureEntity, target: Body
) -> RelativePosition:
    """
    Express a body's position in an observer's own frame.

    :param world: The world both belong to.
    :param observer: The entity the relation is seen from, usually a robot's root.
    :param target: The body being located.
    :return: The displacement along the observer's forward, left and up axes.
    """
    observer_position = world.transform(
        target.global_pose.to_position(), observer
    )
    forward, left, up = observer_position[:3].to_np().flatten().tolist()
    return RelativePosition(forward=float(forward), left=float(left), up=float(up))


def describe_relative_position(
    world: World, observer: KinematicStructureEntity, target: Body
) -> str:
    """
    :param world: The world both belong to.
    :param observer: The entity the relation is seen from.
    :param target: The body being located.
    :return: The relation in words, prefixed by the observer's name.
    """
    relation = relative_position(world, observer, target)
    return f"{relation.describe()} of {observer.name.name}"


# %% reach


def reachable_extent(relation: RelativePosition, reach: float) -> Tuple[bool, str]:
    """
    Judge whether a target is within a robot's reach without moving its base.

    :param relation: Where the target lies relative to the robot.
    :param reach: The robot's horizontal reach in metres.
    :return: Whether it is within reach, and a phrase saying so.
    """
    distance = relation.horizontal_distance
    if distance <= reach:
        return True, f"within reach at {distance:.2f} m"
    return False, f"out of reach at {distance:.2f} m, needs the base to move closer"