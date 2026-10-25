"""
Filling a container with particles from its basin, and reading how full the particles
leave it once they have settled.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from cram_viz.resolution.basin_geometry import ContainerBasin
from scipy import ndimage

# %% filling


@dataclass(frozen=True)
class ParticleFilling:
    """
    The particles filling a container's upright basin to a share of it, on a lattice of
    the particles' spacing, clear of the container's walls.
    """

    basin: ContainerBasin
    """
    The space the container holds a liquid in.
    """

    spacing: float
    """
    The distance, in metres, between neighbouring particles.
    """

    wall_clearance: float
    """
    The distance, in metres, particles start from the container's surface, so none
    starts inside a wall.
    """

    surface_share: float = 0.99
    """
    The share of the settled particles that lie below the liquid's surface; the few
    above it are spray.
    """

    def liquid(self, fill_fraction: float) -> np.ndarray:
        """
        :return: Whether each voxel of the basin's grid holds liquid when the container
            stands upright filled to the share of its basin.
        """
        cavity = self.basin.cavity
        heights = np.broadcast_to(self.basin.axis_coordinates[2], cavity.shape)
        level = np.quantile(heights[cavity], fill_fraction)
        return cavity & (heights <= level)

    def positions(self, fill_fraction: float) -> np.ndarray:
        """
        :return: Where the particles start, in the container's own frame, one row each,
            on a lattice :attr:`spacing` apart.
        """
        clearance = int(np.ceil(self.wall_clearance / self.basin.pitch))
        cleared = ndimage.binary_erosion(
            self.liquid(fill_fraction),
            iterations=clearance,
            mask=self.basin.cavity,
            border_value=0,
        )
        indices = np.argwhere(cleared)
        if len(indices) == 0:
            return np.zeros((0, 3))
        axes = self.basin.axis_coordinates
        lattice = np.meshgrid(
            *[
                np.arange(
                    axis[indices[:, number].min()],
                    axis[indices[:, number].max()],
                    self.spacing,
                )
                for number, axis in enumerate(axes)
            ],
            indexing="ij",
        )
        points = np.column_stack([coordinates.ravel() for coordinates in lattice])
        voxels = np.column_stack(
            [
                np.rint((points[:, number] - axis[0]) / self.basin.pitch).astype(int)
                for number, axis in enumerate(axes)
            ]
        )
        return points[cleared[voxels[:, 0], voxels[:, 1], voxels[:, 2]]]

    def settled_fill_fraction(self, particles: np.ndarray) -> float:
        """
        :param particles: The settled particles the container holds upright, in its own
            frame, one row each.
        :return: The share of the basin lying below the particles' surface.
        """
        surface = np.quantile(particles[:, 2], self.surface_share) + self.spacing / 2
        cavity = self.basin.cavity
        heights = np.broadcast_to(self.basin.axis_coordinates[2], cavity.shape)
        return float((cavity & (heights <= surface)).sum() / cavity.sum())

    def start_fill_settling_at(
        self, fill_fraction: float, started_at: float, settled_at: float
    ) -> float:
        """
        :param fill_fraction: The fill the particles are to settle to.
        :param started_at: The fill particles were started at.
        :param settled_at: The fill those particles settled to.
        :return: The fill to start the particles at so that, shrinking alike as they
            settle, they settle to the fill asked; at most full.
        """
        return min(1.0, started_at * fill_fraction / settled_at)
