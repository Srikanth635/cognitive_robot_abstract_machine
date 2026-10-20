"""
Measuring from an object's mesh whether it has an opening and which way it faces: rays
are sent into the object along each of its axes, and a direction is an opening when most
of the rays cross the whole object before they hit anything.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product

import numpy as np
import trimesh
from typing_extensions import List, Optional

AXES = (0, 1, 2)
"""
The axes of the mesh's own frame a ray is sent along.
"""

SIGNS = (1.0, -1.0)
"""
The two sides of an axis a ray enters from.
"""

RAYS_PER_SIDE = 5
"""
How many rays are sent along each of the two other axes, so how many in all is the
square of it.
"""

CENTRAL_SHARE = 0.5
"""
The share of the object's width the rays are spread over, around its centre, leaving out
the rim and the walls.
"""

ENTRY_MARGIN = 0.05
"""
How far outside the object a ray starts, as a share of the object's extent along it.
"""

CROSSING_SHARE = 0.75
"""
How much of the object's extent along its direction a ray must travel before hitting
anything to count as having gone in through an opening.
"""

OPEN_RAY_SHARE = 0.5
"""
The share of the rays along a direction that must cross for the direction to be an
opening.
"""


@dataclass
class OpeningMeasurement:
    """
    Where an object's mesh has an opening, found by sending rays into it.
    """

    mesh: trimesh.Trimesh
    """
    The object's mesh, in the object's own frame.
    """

    def direction(self) -> Optional[np.ndarray]:
        """
        :return: The unit vector, in the mesh's frame, pointing out of the object's
            opening; the best one when several directions are open; ``None`` when the
            object has no opening.
        """
        shares = {
            (axis, sign): self.crossing_share(axis, sign)
            for axis, sign in product(AXES, SIGNS)
        }
        (axis, sign), best = max(shares.items(), key=lambda item: item[1])
        if best < OPEN_RAY_SHARE:
            return None
        outward = np.zeros(3)
        outward[axis] = sign
        return outward

    def crossing_share(self, axis: int, sign: float) -> float:
        """
        :param axis: The axis of the mesh's frame the rays run along.
        :param sign: The side of the object on that axis the rays come in from.
        :return: The share of the rays sent in through that side that cross the object
            before hitting anything.
        """
        lowest, highest = self.mesh.bounds
        extent = highest[axis] - lowest[axis]
        origins = self.origins(axis, sign, extent)
        inward = np.zeros(3)
        inward[axis] = -sign
        travelled = np.full(len(origins), np.inf)
        locations, ray_indices, _ = self.mesh.ray.intersects_location(
            origins, np.tile(inward, (len(origins), 1)), multiple_hits=False
        )
        travelled[ray_indices] = np.linalg.norm(
            locations - origins[ray_indices], axis=1
        )
        needed = CROSSING_SHARE * extent
        return float(np.mean(travelled >= needed))

    def origins(self, axis: int, sign: float, extent: float) -> np.ndarray:
        """
        :return: The points the rays along the axis start from: a square grid around the
            object's centre, just outside the side the rays come in from.
        """
        lowest, highest = self.mesh.bounds
        centre = 0.5 * (lowest + highest)
        width = highest - lowest
        others: List[int] = [other for other in AXES if other != axis]
        offsets = np.linspace(-0.5, 0.5, RAYS_PER_SIDE) * CENTRAL_SHARE
        origins = []
        for first, second in product(offsets, offsets):
            origin = centre.copy()
            origin[axis] = (highest[axis] if sign > 0 else lowest[axis]) + sign * (
                ENTRY_MARGIN * extent
            )
            origin[others[0]] += first * width[others[0]]
            origin[others[1]] += second * width[others[1]]
            origins.append(origin)
        return np.array(origins)
