"""
Containers of known shape built from flat walls, for the tests that measure a container
from its mesh.
"""

from __future__ import annotations

from dataclasses import dataclass

import trimesh

# %% walls


def wall(extents: list[float], at: list[float]) -> trimesh.Trimesh:
    """
    :return: A box of the extents centred at the position.
    """
    return trimesh.creation.box(
        extents=extents, transform=trimesh.transformations.translation_matrix(at)
    )


def turned(mesh: trimesh.Trimesh, angle: float, axis: list[float]) -> trimesh.Trimesh:
    """
    :return: A copy of the mesh turned about the axis through its origin.
    """
    turned_mesh = mesh.copy()
    turned_mesh.apply_transform(
        trimesh.transformations.rotation_matrix(angle, axis, point=[0, 0, 0])
    )
    return turned_mesh


# %% boxes


@dataclass
class WalledBox:
    """
    A cube centred at the origin whose floor and four walls are slabs; its +z is up.
    """

    side: float = 0.1
    """
    The length of the cube's sides, in metres, measured to the middle of its walls.
    """

    wall_thickness: float = 0.01
    """
    The thickness of the floor and the walls, in metres.
    """

    @property
    def inner_width(self) -> float:
        """
        The distance between the inner faces of two opposite walls, in metres.
        """
        return self.side - self.wall_thickness

    @property
    def inner_height(self) -> float:
        """
        The height of the walls above the floor's top face, in metres.
        """
        return self.side - self.wall_thickness / 2

    def open_box(self) -> trimesh.Trimesh:
        """
        :return: The floor and four walls with no top: open along +z.
        """
        half = self.side / 2
        return trimesh.util.concatenate(
            [
                wall([self.side, self.side, self.wall_thickness], [0, 0, -half]),
                wall([self.wall_thickness, self.side, self.side], [-half, 0, 0]),
                wall([self.wall_thickness, self.side, self.side], [half, 0, 0]),
                wall([self.side, self.wall_thickness, self.side], [0, -half, 0]),
                wall([self.side, self.wall_thickness, self.side], [0, half, 0]),
            ]
        )

    def closed_box(self) -> trimesh.Trimesh:
        """
        :return: The open box with a lid on top.
        """
        lid = wall([self.side, self.side, self.wall_thickness], [0, 0, self.side / 2])
        return trimesh.util.concatenate([self.open_box(), lid])


@dataclass
class LowLippedBox(WalledBox):
    """
    The walled box with its +x wall ending lower than the others, a handle loop on its
    -x side and a foot ring under its floor: a container that spills over one side
    first.
    """

    lip_drop: float = 0.03
    """
    How much lower the +x wall ends than the other walls, in metres.
    """

    handle_reach: float = 0.03
    """
    How far the handle loop stands out from the -x wall, in metres.
    """

    foot_height: float = 0.01
    """
    How far the foot ring reaches below the floor, in metres.
    """

    @property
    def lowest_lip_height(self) -> float:
        """
        The height of the +x wall above the floor's top face, in metres.
        """
        return self.inner_height - self.lip_drop

    def container(self) -> trimesh.Trimesh:
        """
        :return: The floor, the walls with the lowered one, the handle and the foot ring.
        """
        half = self.side / 2
        thickness = self.wall_thickness
        low_wall_height = self.side - self.lip_drop
        handle_x = -half - self.handle_reach
        arm_x = -half - self.handle_reach / 2
        foot_z = -half - thickness / 2 - self.foot_height / 2
        return trimesh.util.concatenate(
            [
                wall([self.side, self.side, thickness], [0, 0, -half]),
                wall([thickness, self.side, self.side], [-half, 0, 0]),
                wall(
                    [thickness, self.side, low_wall_height],
                    [half, 0, -half + low_wall_height / 2],
                ),
                wall([self.side, thickness, self.side], [0, -half, 0]),
                wall([self.side, thickness, self.side], [0, half, 0]),
                wall([self.handle_reach, thickness, thickness], [arm_x, 0, half / 2]),
                wall([self.handle_reach, thickness, thickness], [arm_x, 0, -half / 2]),
                wall([thickness, thickness, half + thickness], [handle_x, 0, 0]),
                wall([self.side, thickness, self.foot_height], [0, -half, foot_z]),
                wall([self.side, thickness, self.foot_height], [0, half, foot_z]),
                wall([thickness, self.side, self.foot_height], [-half, 0, foot_z]),
                wall([thickness, self.side, self.foot_height], [half, 0, foot_z]),
            ]
        )
