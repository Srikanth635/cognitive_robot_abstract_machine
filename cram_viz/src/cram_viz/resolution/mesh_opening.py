"""
Measuring from an object's mesh whether it has an opening and where it faces, by sending
rays into it: a side is open when most of the rays sent in through it cross the object
before they hit anything. Facing up means the rays a pour follows, falling straight down,
get in.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from itertools import product

import numpy as np
import trimesh
from typing_extensions import List, Optional

AXES = (0, 1, 2)
"""
The axes of the mesh's frame a ray is sent along; the last is the vertical.
"""

VERTICAL_AXIS = 2
"""
The axis rays fall along: the world's up.
"""

SIGNS = (1.0, -1.0)
"""
The two sides of an axis a ray enters from.
"""


class OpeningDirection(StrEnum):
    """
    Where a container's opening faces.
    """

    UP = "up"
    DOWN = "down"
    SIDEWAYS = "sideways"


@dataclass(frozen=True)
class RayTolerances:
    """
    How rays are sent into an object and when they count as having gone in through an
    opening. The objects the repository has are told apart the same way across a range
    of these values; the defaults sit inside it.
    """

    rays_per_side: int = 5
    """
    How many rays are sent along each of the two other axes, so how many in all is the
    square of it.
    """

    central_share: float = 0.5
    """
    The share of the object's width the rays are spread over, around its centre, leaving
    out the rim and the walls.
    """

    entry_margin: float = 0.05
    """
    How far outside the object a ray starts, as a share of the object's extent along it.
    """

    crossing_share: float = 0.5
    """
    How much of the object's extent along its direction a ray must travel before hitting
    anything to count as having gone in through an opening.
    """

    open_ray_share: float = 0.5
    """
    The share of the rays sent in through a side that must cross for the side to be an
    opening.
    """


@dataclass
class OpeningMeasurement:
    """
    Where an object's mesh has an opening, found by sending rays into it.
    """

    mesh: trimesh.Trimesh
    """
    The object's mesh, with the world's axes: turned as the object is placed, so that its
    vertical is the world's.
    """

    tolerances: RayTolerances = field(default_factory=RayTolerances)
    """
    How the rays are sent and judged.
    """

    def faces(self) -> Optional[OpeningDirection]:
        """
        :return: Where the opening faces: up when the rays falling straight down get in,
            down when the ones rising do, sideways when only horizontal rays do; the
            best of up and down when both get in; ``None`` when no side is open.
        """
        up = self.crossing_share(VERTICAL_AXIS, 1.0)
        down = self.crossing_share(VERTICAL_AXIS, -1.0)
        if max(up, down) >= self.tolerances.open_ray_share:
            return OpeningDirection.UP if up >= down else OpeningDirection.DOWN
        if any(
            self.crossing_share(axis, sign) >= self.tolerances.open_ray_share
            for axis, sign in product(AXES[:VERTICAL_AXIS], SIGNS)
        ):
            return OpeningDirection.SIDEWAYS
        return None

    def crossing_share(self, axis: int, sign: float) -> float:
        """
        :param axis: The axis of the mesh's frame the rays run along.
        :param sign: The side of the object on that axis the rays come in from.
        :return: Of the rays sent in through that side that meet the object, the share
            that travel on, before hitting the mesh, through at least
            :attr:`RayTolerances.crossing_share` of the object's extent along the ray,
            which is the stretch the ray spends inside the object's convex hull; a ray
            that hits nothing has found no floor and does not count.
        """
        lowest, highest = self.mesh.bounds
        origins = self.origins(axis, sign, highest[axis] - lowest[axis])
        inward = np.zeros(3)
        inward[axis] = -sign
        directions = np.tile(inward, (len(origins), 1))
        entered, left = self.hull_chords(origins, directions)
        met = ~np.isnan(entered)
        if not met.any():
            return 0.0
        hit = np.full(len(origins), np.inf)
        locations, ray_indices, _ = self.mesh.ray.intersects_location(
            origins, directions, multiple_hits=False
        )
        hit[ray_indices] = np.linalg.norm(locations - origins[ray_indices], axis=1)
        extent = left - entered
        crossed = (hit - entered) >= self.tolerances.crossing_share * extent
        return float(np.mean(crossed[met] & np.isfinite(hit[met])))

    def hull_chords(self, origins: np.ndarray, directions: np.ndarray):
        """
        :return: For each ray, how far along it the object's convex hull is entered and
            left; not a number for a ray that misses the hull.
        """
        entered = np.full(len(origins), np.nan)
        left = np.full(len(origins), np.nan)
        locations, ray_indices, _ = self.mesh.convex_hull.ray.intersects_location(
            origins, directions, multiple_hits=True
        )
        for ray in np.unique(ray_indices):
            distances = np.linalg.norm(
                locations[ray_indices == ray] - origins[ray], axis=1
            )
            entered[ray], left[ray] = distances.min(), distances.max()
        return entered, left

    def origins(self, axis: int, sign: float, extent: float) -> np.ndarray:
        """
        :return: The points the rays along the axis start from: a square grid around the
            object's centre, just outside the side the rays come in from.
        """
        lowest, highest = self.mesh.bounds
        centre = 0.5 * (lowest + highest)
        width = highest - lowest
        others: List[int] = [other for other in AXES if other != axis]
        offsets = (
            np.linspace(-0.5, 0.5, self.tolerances.rays_per_side)
            * self.tolerances.central_share
        )
        origins = []
        for first, second in product(offsets, offsets):
            origin = centre.copy()
            origin[axis] = (highest[axis] if sign > 0 else lowest[axis]) + sign * (
                self.tolerances.entry_margin * extent
            )
            origin[others[0]] += first * width[others[0]]
            origin[others[1]] += second * width[others[1]]
            origins.append(origin)
        return np.array(origins)
