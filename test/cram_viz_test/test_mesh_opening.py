"""
Tests of measuring from a mesh whether an object has an opening and which way it faces,
by sending rays into the object along each of its axes: rays that cross the whole object
before hitting anything went in through an opening.
"""

from __future__ import annotations

import numpy as np
import pytest
import trimesh

from cram_viz.resolution.mesh_opening import OpeningMeasurement

WALL = 0.01
"""
The thickness of the walls of the test shapes, in metres.
"""

SIDE = 0.1
"""
The length of a test shape's sides, in metres.
"""

# %% shapes


def wall(extents: list[float], at: list[float]) -> trimesh.Trimesh:
    """
    :return: A box of the extents centred at the position.
    """
    return trimesh.creation.box(
        extents=extents, transform=trimesh.transformations.translation_matrix(at)
    )


def open_box() -> trimesh.Trimesh:
    """
    :return: A box with a floor and four walls and no top: open along +z.
    """
    half = SIDE / 2
    return trimesh.util.concatenate(
        [
            wall([SIDE, SIDE, WALL], [0, 0, -half]),
            wall([WALL, SIDE, SIDE], [-half, 0, 0]),
            wall([WALL, SIDE, SIDE], [half, 0, 0]),
            wall([SIDE, WALL, SIDE], [0, -half, 0]),
            wall([SIDE, WALL, SIDE], [0, half, 0]),
        ]
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


# %% measuring


class TestOpeningMeasurement:
    def test_a_box_with_no_top_opens_along_plus_z(self):
        opening = OpeningMeasurement(mesh=open_box()).direction()
        assert np.allclose(opening, [0.0, 0.0, 1.0])

    def test_the_same_box_turned_over_opens_along_minus_z(self):
        opening = OpeningMeasurement(
            mesh=turned(open_box(), np.pi, [1, 0, 0])
        ).direction()
        assert np.allclose(opening, [0.0, 0.0, -1.0])

    def test_the_same_box_turned_on_its_side_opens_along_the_axis_it_was_turned_to(
        self,
    ):
        opening = OpeningMeasurement(
            mesh=turned(open_box(), np.pi / 2, [0, 1, 0])
        ).direction()
        assert np.allclose(opening, [1.0, 0.0, 0.0])

    def test_a_closed_box_has_no_opening(self):
        closed = trimesh.util.concatenate(
            [open_box(), wall([SIDE, SIDE, WALL], [0, 0, SIDE / 2])]
        )
        assert OpeningMeasurement(mesh=closed).direction() is None

    def test_a_solid_block_has_no_opening(self):
        assert (
            OpeningMeasurement(
                mesh=trimesh.creation.box(extents=[SIDE] * 3)
            ).direction()
            is None
        )
