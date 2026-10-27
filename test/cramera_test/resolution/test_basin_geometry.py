"""
Tests of deriving from a container's mesh how much liquid it holds at a tilt and the
tilt at which a liquid filled to a given share starts to leave it.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from cramera.resolution.resolver.basin_geometry import ContainerBasin, ContainerTilt
from cramera.resolution.resolver.exceptions import FillFractionOutOfRange

from .walled_boxes import LowLippedBox, WalledBox

# %% expected values


def box_onset_tangent(inner_width: float, lip_height: float, level: float) -> float:
    """
    :return: The tangent of the tilt at which the surface of a liquid standing at the
        level in a box of the inner width, turning about the box's centre line, reaches a
        lip at the height; valid while the level is at least half the lip's height.
    """
    return 2 * (lip_height - level) / inner_width


def tangent_tolerance(basin: ContainerBasin, inner_width: float) -> float:
    """
    :return: How far a tangent derived on the basin's voxels may be off: the surface and
        the lip are each placed to within one voxel.
    """
    return 2 * (2 * basin.pitch) / inner_width


def volume_tolerance(basin: ContainerBasin, area: float, wetted_area: float) -> float:
    """
    :return: How far a volume derived on the basin's voxels may be off: one voxel layer
        over the free surface of the area and over the wetted walls.
    """
    return basin.pitch * (area + wetted_area)


# %% capacity


class TestCapacity:
    def test_an_open_box_upright_holds_its_inner_volume(self):
        box = WalledBox()
        basin = ContainerBasin(mesh=box.open_box())
        area = box.inner_width**2
        wetted_area = area + 4 * box.inner_width * box.inner_height
        assert basin.capacity(ContainerTilt()) == pytest.approx(
            area * box.inner_height,
            abs=volume_tolerance(basin, area, wetted_area),
        )

    def test_a_container_upright_is_full_at_its_lowest_lip(self):
        box = LowLippedBox()
        basin = ContainerBasin(mesh=box.container())
        area = box.inner_width**2
        wetted_area = area + 4 * box.inner_width * box.lowest_lip_height
        assert basin.capacity(ContainerTilt()) == pytest.approx(
            area * box.lowest_lip_height,
            abs=volume_tolerance(basin, area, wetted_area),
        )

    def test_a_closed_box_holds_nothing_it_could_be_filled_with(self):
        basin = ContainerBasin(mesh=WalledBox().closed_box())
        assert basin.capacity(ContainerTilt()) == 0.0


# %% flow onset of a liquid


class TestLiquidFlowOnset:
    @pytest.mark.parametrize("fill_fraction", [0.6, 0.75, 0.9])
    @pytest.mark.parametrize("gravity_azimuth", [0.0, 90.0])
    def test_an_open_box_spills_when_the_surface_reaches_the_lip(
        self, fill_fraction, gravity_azimuth
    ):
        box = WalledBox()
        basin = ContainerBasin(mesh=box.open_box())
        onset = basin.liquid_flow_onset(fill_fraction, gravity_azimuth)
        expected = box_onset_tangent(
            box.inner_width, box.inner_height, fill_fraction * box.inner_height
        )
        assert np.tan(np.radians(onset.degrees)) == pytest.approx(
            expected, abs=tangent_tolerance(basin, box.inner_width)
        )

    def test_the_onset_falls_as_the_fill_rises(self):
        basin = ContainerBasin(mesh=WalledBox().open_box())
        onsets = [
            basin.liquid_flow_onset(fill_fraction, 0.0).degrees
            for fill_fraction in (0.25, 0.5, 0.75)
        ]
        assert onsets == sorted(onsets, reverse=True)
        assert len(set(onsets)) == len(onsets)

    def test_a_container_spills_over_its_low_lip_before_over_its_handle_side(self):
        basin = ContainerBasin(mesh=LowLippedBox().container())
        toward_low_lip = basin.liquid_flow_onset(0.5, 0.0)
        toward_handle = basin.liquid_flow_onset(0.5, 180.0)
        assert toward_low_lip.degrees < toward_handle.degrees

    def test_a_container_full_to_its_lowest_lip_spills_over_it_at_once(self):
        box = LowLippedBox()
        basin = ContainerBasin(mesh=box.container())
        onset = basin.liquid_flow_onset(1.0, 0.0)
        assert np.tan(np.radians(onset.degrees)) == pytest.approx(
            0.0, abs=tangent_tolerance(basin, box.inner_width)
        )

    def test_a_container_full_to_its_lowest_lip_spills_over_a_higher_one_once_it_reaches_it(
        self,
    ):
        box = LowLippedBox()
        basin = ContainerBasin(mesh=box.container())
        onset = basin.liquid_flow_onset(1.0, 180.0)
        expected = box_onset_tangent(
            box.inner_width, box.inner_height, box.lowest_lip_height
        )
        assert np.tan(np.radians(onset.degrees)) == pytest.approx(
            expected, abs=tangent_tolerance(basin, box.inner_width)
        )

    def test_a_closed_box_never_spills(self):
        basin = ContainerBasin(mesh=WalledBox().closed_box())
        assert basin.liquid_flow_onset(0.5, 0.0) is None

    def test_the_onset_records_the_pitch_it_was_derived_at(self):
        basin = ContainerBasin(mesh=WalledBox().open_box())
        assert basin.liquid_flow_onset(0.5, 0.0).pitch == basin.pitch

    def test_the_same_onset_is_derived_once(self):
        basin = ContainerBasin(mesh=WalledBox().open_box())
        assert basin.liquid_flow_onset(0.5, 0.0) is basin.liquid_flow_onset(0.5, 0.0)

    @pytest.mark.parametrize("fill_fraction", [0.0, -0.1, 1.1])
    def test_a_fill_outside_empty_to_full_is_refused(self, fill_fraction):
        basin = ContainerBasin(mesh=WalledBox().open_box())
        with pytest.raises(FillFractionOutOfRange):
            basin.liquid_flow_onset(fill_fraction, 0.0)


# %% the tilt of a rotated container


class TestTiltOfARotation:
    def test_an_upright_container_is_not_tilted(self):
        assert ContainerTilt.of_rotation(np.eye(3)).tilt == pytest.approx(0.0)

    def test_tilting_about_its_own_y_axis_tips_it_toward_its_x_axis(self):
        tilt = ContainerTilt.of_rotation(
            Rotation.from_euler("y", 60, degrees=True).as_matrix()
        )
        assert tilt.tilt == pytest.approx(60.0)
        assert tilt.gravity_azimuth == pytest.approx(0.0)

    def test_the_heading_it_is_carried_at_leaves_the_direction_of_tilt(self):
        tilt = ContainerTilt.of_rotation(
            Rotation.from_euler("yz", [60, 130], degrees=True).as_matrix()
        )
        assert tilt.tilt == pytest.approx(60.0)
        assert tilt.gravity_azimuth == pytest.approx(0.0)

    def test_turning_it_in_the_hand_turns_the_direction_of_tilt(self):
        tilt = ContainerTilt.of_rotation(
            Rotation.from_euler("zy", [90, 60], degrees=True).as_matrix()
        )
        assert tilt.tilt == pytest.approx(60.0)
        assert tilt.gravity_azimuth == pytest.approx(270.0)
