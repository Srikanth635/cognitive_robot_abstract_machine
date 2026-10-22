"""
Tests of measuring from a mesh whether an object has an opening and which way it faces,
by sending rays into the object along each of its axes: rays that cross the whole object
before hitting anything went in through an opening.
"""

from __future__ import annotations

import numpy as np
import pytest
import trimesh

from cramera.resolution.resolver.mesh_opening import (
    OpeningDirection,
    OpeningMeasurement,
    RayTolerances,
)

from .conftest import OBJECT_MESHES

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
    def test_a_box_with_no_top_opens_up(self):
        assert OpeningMeasurement(mesh=open_box()).faces() is OpeningDirection.UP

    def test_the_same_box_turned_over_opens_down(self):
        turned_over = turned(open_box(), np.pi, [1, 0, 0])
        assert OpeningMeasurement(mesh=turned_over).faces() is OpeningDirection.DOWN

    def test_the_same_box_turned_on_its_side_opens_sideways(self):
        on_its_side = turned(open_box(), np.pi / 2, [0, 1, 0])
        assert OpeningMeasurement(mesh=on_its_side).faces() is OpeningDirection.SIDEWAYS

    def test_a_box_leaning_less_than_it_would_take_to_spill_still_opens_up(self):
        leaning = turned(open_box(), np.radians(30), [0, 1, 0])
        assert OpeningMeasurement(mesh=leaning).faces() is OpeningDirection.UP

    def test_a_closed_box_has_no_opening(self):
        closed = trimesh.util.concatenate(
            [open_box(), wall([SIDE, SIDE, WALL], [0, 0, SIDE / 2])]
        )
        assert OpeningMeasurement(mesh=closed).faces() is None

    def test_a_solid_block_has_no_opening(self):
        block = trimesh.creation.box(extents=[SIDE] * 3)
        assert OpeningMeasurement(mesh=block).faces() is None


# %% the tolerances

OPEN_OBJECTS = (
    "bowl.stl",
    "apartment_bowl.stl",
    "jeroen_cup.stl",
    "Static_MilkPitcher.stl",
)
"""
Meshes of the repository's objects that have an opening facing up.
"""

CLOSED_OBJECTS = ("milk.stl", "breakfast_cereal.stl", "bread.stl", "big-knife.stl")
"""
Meshes of the repository's objects that have no opening.
"""

CROSSING_SHARES = (0.5, 0.55, 0.6)
"""
Crossing shares, the share of an object a ray must cross, across which the objects above
are told apart the same way.
"""

OPEN_RAY_SHARES = (0.3, 0.5, 0.7)
"""
Open-ray shares, the share of rays that must cross, across which the objects above are
told apart the same way.
"""


class TestTolerances:
    @pytest.mark.parametrize("crossing_share", CROSSING_SHARES)
    @pytest.mark.parametrize("open_ray_share", OPEN_RAY_SHARES)
    def test_the_objects_are_told_apart_across_the_tolerances_the_defaults_sit_in(
        self, crossing_share, open_ray_share
    ):
        tolerances = RayTolerances(
            crossing_share=crossing_share, open_ray_share=open_ray_share
        )

        def opening(name: str):
            mesh = trimesh.load_mesh(OBJECT_MESHES / name)
            return OpeningMeasurement(mesh=mesh, tolerances=tolerances).faces()

        assert [opening(name) for name in OPEN_OBJECTS] == [OpeningDirection.UP] * len(
            OPEN_OBJECTS
        )
        assert [opening(name) for name in CLOSED_OBJECTS] == [None] * len(
            CLOSED_OBJECTS
        )

    def test_the_defaults_lie_inside_the_range_that_was_checked(self):
        defaults = RayTolerances()
        assert min(CROSSING_SHARES) <= defaults.crossing_share <= max(CROSSING_SHARES)
        assert min(OPEN_RAY_SHARES) <= defaults.open_ray_share <= max(OPEN_RAY_SHARES)
