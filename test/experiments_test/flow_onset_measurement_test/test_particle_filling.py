"""
Tests of filling a container with particles from its basin and of reading the fill the
settled particles leave it at.
"""

from __future__ import annotations

import pytest
from cram_viz.resolution.basin_geometry import ContainerBasin

from experiments.flow_onset_measurement.filling import ParticleFilling

from ...cram_viz_test.walled_boxes import WalledBox

# %% fixtures


@pytest.fixture(scope="module")
def box() -> WalledBox:
    """
    An open box of 10 cm with walls of 1 cm.
    """
    return WalledBox()


@pytest.fixture(scope="module")
def filling(box) -> ParticleFilling:
    """
    Particles 4 mm apart, starting 2 mm clear of the walls of the box.
    """
    return ParticleFilling(
        basin=ContainerBasin(mesh=box.open_box(), pitch=0.002),
        spacing=0.004,
        wall_clearance=0.002,
    )


# %% filling


class TestParticleFilling:
    def test_the_particles_start_inside_the_basin_clear_of_its_walls(
        self, filling, box
    ):
        positions = filling.positions(0.5)
        half_inner = box.inner_width / 2
        assert abs(positions[:, :2]).max() <= half_inner - filling.wall_clearance

    def test_more_fill_takes_more_particles(self, filling):
        assert len(filling.positions(0.8)) > len(filling.positions(0.4))

    def test_the_particles_reach_up_to_the_fill_asked(self, filling, box):
        positions = filling.positions(0.5)
        floor = positions[:, 2].min() - filling.wall_clearance
        assert positions[:, 2].max() - floor == pytest.approx(
            0.5 * box.inner_height, abs=2 * filling.spacing
        )

    def test_particles_settled_where_they_started_read_as_the_fill_asked(self, filling):
        assert filling.settled_fill_fraction(filling.positions(0.5)) == pytest.approx(
            0.5, abs=0.1
        )

    def test_a_liquid_settling_smaller_is_started_fuller(self, filling):
        assert filling.start_fill_settling_at(0.5, 0.5, 0.25) == pytest.approx(1.0)
        assert filling.start_fill_settling_at(0.4, 0.4, 0.3) == pytest.approx(0.4 * 0.4 / 0.3)

    def test_a_start_fill_is_never_above_full(self, filling):
        assert filling.start_fill_settling_at(0.75, 0.75, 0.3) == 1.0
