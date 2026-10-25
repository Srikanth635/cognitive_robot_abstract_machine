"""
The quasi-static liquid derivation, from a container's mesh, of how much liquid the
container holds at a tilt and of the tilt at which a liquid filled to a share of it
starts to flow out. A liquid's surface stays level, so the container holds what lies
below the highest level at which the space under it is still closed off from the
outside. A granular or cohesive material keeps a slope and is not covered.

The mesh's frame is the container's own: its +z is the container's up.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property

import numpy as np
import trimesh
from scipy import ndimage

from cram_viz.resolution.exceptions import FillFractionOutOfRange

# %% tilt


@dataclass(frozen=True)
class ContainerTilt:
    """
    How a container is tilted away from upright, as the direction gravity pulls in the
    container's own frame.
    """

    tilt: float = 0.0
    """
    The angle between gravity and the container's down, in degrees.
    """

    gravity_azimuth: float = 0.0
    """
    The direction gravity's part in the container's xy plane points, in degrees from its
    +x axis toward its +y axis: the side the container leans toward.
    """

    @classmethod
    def of_rotation(cls, world_rotation: np.ndarray) -> ContainerTilt:
        """
        :param world_rotation: The rotation from the container's own frame to the world,
            whose z axis points up.
        :return: How a container so rotated is tilted; an azimuth of 0 when it stands
            upright.
        """
        gravity = np.asarray(world_rotation).T @ np.array([0.0, 0.0, -1.0])
        return cls(
            tilt=float(np.degrees(np.arccos(np.clip(-gravity[2], -1.0, 1.0)))),
            gravity_azimuth=float(
                np.degrees(np.arctan2(gravity[1], gravity[0])) % 360.0
            ),
        )

    def gravity(self) -> np.ndarray:
        """
        :return: The unit vector gravity pulls along, in the container's own frame.
        """
        tilt = np.radians(self.tilt)
        azimuth = np.radians(self.gravity_azimuth)
        return np.array(
            [
                np.sin(tilt) * np.cos(azimuth),
                np.sin(tilt) * np.sin(azimuth),
                -np.cos(tilt),
            ]
        )


@dataclass(frozen=True)
class LiquidFlowOnset:
    """
    The smallest tilt at which a liquid starts to flow out of a container, derived from
    the container's mesh.
    """

    degrees: float
    """
    The tilt away from upright, in degrees.
    """

    pitch: float
    """
    The side of the voxels the container's mesh was derived on, in metres.
    """


# %% basin


@dataclass
class ContainerBasin:
    """
    The space a container holds a liquid in, derived on a voxel grid of its mesh.
    """

    mesh: trimesh.Trimesh
    """
    The container's mesh in its own frame.
    """

    pitch: float = 0.001
    """
    The side of the voxels, in metres; well below the thickness of the container's
    walls.
    """

    padding: int = 1
    """
    How many layers of empty voxels surround the voxelized mesh, so that the outside is
    one connected space.
    """

    level_tolerance: float = 0.25
    """
    How close, as a share of :attr:`pitch`, the level a basin spills at is bracketed.
    """

    tilt_step: float = 5.0
    """
    The step, in degrees, between the tilts tried before the onset is bracketed.
    """

    tilt_tolerance: float = 0.1
    """
    How close, in degrees, the onset is bracketed.
    """

    largest_tilt: float = 180.0
    """
    The largest tilt tried, in degrees: upside down.
    """

    flow_onsets: dict[tuple[float, float], LiquidFlowOnset | None] = field(
        default_factory=dict, init=False, repr=False
    )
    """
    The onsets already derived, by fill fraction and gravity azimuth.
    """

    @cached_property
    def voxels(self) -> trimesh.voxel.VoxelGrid:
        """
        The mesh voxelized at :attr:`pitch`, with its enclosed space filled.
        """
        return self.mesh.voxelized(self.pitch).fill()

    @cached_property
    def solid(self) -> np.ndarray:
        """
        Whether each voxel of the padded grid is part of the container.
        """
        return np.pad(self.voxels.matrix, self.padding)

    @cached_property
    def border(self) -> np.ndarray:
        """
        Whether each voxel of the padded grid lies on its outer faces.
        """
        border = np.ones_like(self.solid)
        border[1:-1, 1:-1, 1:-1] = False
        return border

    @cached_property
    def axis_coordinates(self) -> list[np.ndarray]:
        """
        The coordinates of the voxel centres of the padded grid along each axis of the
        container's frame.
        """
        return [
            self.voxels.translation[axis]
            + self.voxels.scale[axis] * (np.arange(count) - self.padding)
            for axis, count in enumerate(self.solid.shape)
        ]

    @cached_property
    def cavity(self) -> np.ndarray:
        """
        Whether each voxel of the padded grid holds liquid when the container stands
        upright filled to its lowest lip: the largest space closed off from the outside
        below any horizontal layer of the grid.
        """
        empty = ~self.solid
        cavity = np.zeros_like(empty)
        for layer in range(1, empty.shape[2]):
            labels, count = ndimage.label(empty[:, :, :layer])
            sizes = np.bincount(labels.ravel(), minlength=count + 1)
            sizes[0] = 0
            for outer_face in (
                labels[0],
                labels[-1],
                labels[:, 0],
                labels[:, -1],
                labels[:, :, 0],
            ):
                sizes[outer_face] = 0
            if sizes.max() > cavity.sum():
                cavity[:] = False
                cavity[:, :, :layer] = labels == sizes.argmax()
        return cavity

    def heights(self, tilt: ContainerTilt) -> np.ndarray:
        """
        :return: The height of each voxel centre of the padded grid against gravity at
            the tilt.
        """
        x, y, z = self.axis_coordinates
        gravity = tilt.gravity()
        return -(
            gravity[0] * x[:, None, None]
            + gravity[1] * y[None, :, None]
            + gravity[2] * z[None, None, :]
        )

    def closed_space(
        self, heights: np.ndarray, level: float, seed: tuple[int, int, int]
    ) -> np.ndarray | None:
        """
        :return: The empty voxels below the level 6-connected to the seed, or ``None``
            when the seed is not among them or they reach the grid's outer faces.
        """
        labels, _ = ndimage.label(~self.solid & (heights < level))
        component = labels[seed]
        if component == 0 or component in labels[self.border]:
            return None
        return labels == component

    def basin(self, tilt: ContainerTilt) -> np.ndarray:
        """
        :return: Whether each voxel of the padded grid holds liquid when the container is
            at the tilt and filled until it spills: the closed space around the lowest
            voxel of the :attr:`cavity`, below the highest level at which it is still
            closed off.
        """
        nothing = np.zeros_like(self.solid)
        if not self.cavity.any():
            return nothing
        heights = self.heights(tilt)
        seed = np.unravel_index(
            np.argmin(np.where(self.cavity, heights, np.inf)), heights.shape
        )
        tolerance = self.level_tolerance * self.pitch
        low = heights[seed] + tolerance
        held = self.closed_space(heights, low, seed)
        if held is None:
            return nothing
        high = heights.max()
        while high - low > tolerance:
            middle = (low + high) / 2
            space = self.closed_space(heights, middle, seed)
            if space is None:
                high = middle
            else:
                low, held = middle, space
        return held

    def capacity(self, tilt: ContainerTilt) -> float:
        """
        :return: The volume of liquid the container holds at the tilt, in cubic metres.
        """
        return float(self.basin(tilt).sum()) * self.pitch**3

    def liquid_flow_onset(
        self, fill_fraction: float, gravity_azimuth: float
    ) -> LiquidFlowOnset | None:
        """
        :param fill_fraction: The liquid's volume over the container's upright capacity,
            where 1 is filled to the lowest lip.
        :param gravity_azimuth: The side the container leans toward, as
            :attr:`ContainerTilt.gravity_azimuth`.
        :return: The smallest tilt toward that side at which the container holds less
            than the liquid, or ``None`` when it holds all of it at every tilt.
        """
        if not 0 < fill_fraction <= 1:
            raise FillFractionOutOfRange(
                fill_fraction=fill_fraction, allows_empty=False
            )
        key = (fill_fraction, gravity_azimuth)
        if key not in self.flow_onsets:
            self.flow_onsets[key] = self.derive_liquid_flow_onset(
                fill_fraction, gravity_azimuth
            )
        return self.flow_onsets[key]

    def derive_liquid_flow_onset(
        self, fill_fraction: float, gravity_azimuth: float
    ) -> LiquidFlowOnset | None:
        """
        :return: The onset of :meth:`liquid_flow_onset`, found by stepping the tilt up by
            :attr:`tilt_step` until the container spills and then bisecting the last
            step.
        """
        liquid = fill_fraction * self.basin(ContainerTilt()).sum()
        if liquid == 0:
            return None
        tilts = np.append(
            np.arange(0.0, self.largest_tilt, self.tilt_step), self.largest_tilt
        )
        first_spilling = next(
            (
                index
                for index in range(1, len(tilts))
                if self.spills(ContainerTilt(tilts[index], gravity_azimuth), liquid)
            ),
            None,
        )
        if first_spilling is None:
            return None
        low, high = tilts[first_spilling - 1], tilts[first_spilling]
        while high - low > self.tilt_tolerance:
            middle = (low + high) / 2
            if self.spills(ContainerTilt(middle, gravity_azimuth), liquid):
                high = middle
            else:
                low = middle
        return LiquidFlowOnset(degrees=float(high), pitch=self.pitch)

    def spills(self, tilt: ContainerTilt, liquid: float) -> bool:
        """
        :param liquid: The liquid's volume, in voxels.
        :return: Whether the container at the tilt holds less than the liquid.
        """
        return self.basin(tilt).sum() < liquid
