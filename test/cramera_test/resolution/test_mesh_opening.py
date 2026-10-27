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
from .walled_boxes import WalledBox, turned

# %% measuring


class TestOpeningMeasurement:
    def test_a_box_with_no_top_opens_up(self):
        assert (
            OpeningMeasurement(mesh=WalledBox().open_box()).faces()
            is OpeningDirection.UP
        )

    def test_the_same_box_turned_over_opens_down(self):
        turned_over = turned(WalledBox().open_box(), np.pi, [1, 0, 0])
        assert OpeningMeasurement(mesh=turned_over).faces() is OpeningDirection.DOWN

    def test_the_same_box_turned_on_its_side_opens_sideways(self):
        on_its_side = turned(WalledBox().open_box(), np.pi / 2, [0, 1, 0])
        assert OpeningMeasurement(mesh=on_its_side).faces() is OpeningDirection.SIDEWAYS

    def test_a_box_leaning_less_than_it_would_take_to_spill_still_opens_up(self):
        leaning = turned(WalledBox().open_box(), np.radians(30), [0, 1, 0])
        assert OpeningMeasurement(mesh=leaning).faces() is OpeningDirection.UP

    def test_a_closed_box_has_no_opening(self):
        closed = WalledBox().closed_box()
        assert OpeningMeasurement(mesh=closed).faces() is None

    def test_a_solid_block_has_no_opening(self):
        block = trimesh.creation.box(extents=[WalledBox().side] * 3)
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
