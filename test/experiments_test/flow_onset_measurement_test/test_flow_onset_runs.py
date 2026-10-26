"""
Tests of the parts of the flow onset measurement that need no simulator: when the
particles count as leaving the container, and the measurement kept from repeated runs.
"""

from __future__ import annotations

import numpy as np
import pytest
import trimesh
from llm_reasoner.knowledge.access import (
    action_core,
    core,
    flow_onset,
    pouring_frame,
)

from experiments.flow_onset_measurement.exceptions import NoSimulatedTilt
from experiments.flow_onset_measurement.runs import (
    ParticlesLeaving,
    SimulatedFlowOnset,
    SimulatedTilt,
    SimulationSettings,
)

# %% fixtures


@pytest.fixture
def leaving() -> ParticlesLeaving:
    """
    Particles leaving a unit box once three of them are outside it.
    """
    return ParticlesLeaving(
        hull=trimesh.creation.box(extents=(1.0, 1.0, 1.0)), particle_count=3
    )


def simulated(*runs: SimulatedTilt) -> SimulatedFlowOnset:
    """
    :return: The runs of a tilt of a cup toward its own x axis.
    """
    return SimulatedFlowOnset(
        container=flow_onset.ContainerModel(
            mesh_digest="c" * 64, mesh_file_name="cup.stl"
        ),
        gravity_azimuth=0.0,
        settings=SimulationSettings(engine_version="6.0.0.1"),
        runs=list(runs),
    )


# %% leaving the container


class TestParticlesLeaving:
    def test_particles_inside_the_hull_have_not_left(self, leaving):
        assert not leaving.has_left(np.zeros((10, 3)))

    def test_fewer_particles_outside_than_needed_have_not_left(self, leaving):
        particles = np.zeros((10, 3))
        particles[:2] = 2.0
        assert leaving.outside(particles) == 2
        assert not leaving.has_left(particles)

    def test_enough_particles_outside_have_left(self, leaving):
        particles = np.zeros((10, 3))
        particles[:3] = 2.0
        assert leaving.has_left(particles)


# %% the measurement kept


class TestSimulatedFlowOnset:
    def test_the_onset_ranges_over_every_run(self):
        measurement = simulated(
            SimulatedTilt(onset_tilt=71.0, settled_fill_fraction=0.48),
            SimulatedTilt(onset_tilt=73.5, settled_fill_fraction=0.5),
        ).measurement()
        assert measurement.has_measured_range.lower_bound == 71.0
        assert measurement.has_measured_range.upper_bound == 73.5
        assert (
            measurement.has_measured_range.has_unit
            is action_core.MeasurementUnit.DEGREE
        )

    def test_the_fill_is_the_one_the_particles_settled_to(self):
        measurement = simulated(
            SimulatedTilt(onset_tilt=71.0, settled_fill_fraction=0.48),
            SimulatedTilt(onset_tilt=73.5, settled_fill_fraction=0.5),
        ).measurement()
        assert measurement.at_fill_fraction == pytest.approx(0.49)
        settled = measurement.follows_setup.has_settled_fill_range
        assert (settled.lower_bound, settled.upper_bound) == (0.48, 0.5)

    def test_the_setup_states_the_settings_and_the_number_of_runs(self):
        settings = SimulationSettings(engine_version="6.0.0.1")
        measurement = simulated(
            SimulatedTilt(onset_tilt=71.0, settled_fill_fraction=0.5)
        ).measurement()
        setup = measurement.follows_setup
        assert isinstance(setup, flow_onset.PhysicsSimulationSetup)
        assert setup.run_count == 1
        assert setup.tilt_rate == settings.tilt_rate
        assert setup.particle_spacing == settings.particle_spacing
        assert setup.simulation_engine_version == settings.engine_version
        assert measurement.has_material_regime is pouring_frame.MaterialRegime.LIQUID

    def test_no_run_gives_no_measurement(self):
        with pytest.raises(NoSimulatedTilt):
            simulated().measurement()
