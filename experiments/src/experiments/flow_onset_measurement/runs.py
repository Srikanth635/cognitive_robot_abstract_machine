"""
What the simulated tilts of a container find, and the physical measurement kept from
them: when the particles count as leaving the container, the onset of every run, and the
settings that decided it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import trimesh
from llm_reasoner.knowledge.access import (
    action_core,
    core,
    flow_onset,
    pouring_frame,
)

from experiments.flow_onset_measurement.exceptions import NoSimulatedTilt

# %% leaving the container


@dataclass(frozen=True)
class ParticlesLeaving:
    """
    When the simulated substance counts as leaving a container: enough of its particles
    lie outside the container's convex hull, in the container's own frame.
    """

    hull: trimesh.Trimesh
    """
    The convex hull of the container, in its own frame.
    """

    particle_count: int = 10
    """
    How many particles have to be outside the hull; a few stray particles do not make a
    flow.
    """

    def outside(self, particles: np.ndarray) -> int:
        """
        :param particles: Positions of particles in the container's own frame, one row
            each.
        :return: How many of them lie outside the hull.
        """
        return int(np.count_nonzero(~self.hull.contains(particles)))

    def has_left(self, particles: np.ndarray) -> bool:
        """
        :param particles: Positions of particles in the container's own frame, one row
            each.
        :return: Whether enough of them lie outside the hull.
        """
        return self.outside(particles) >= self.particle_count


# %% runs


@dataclass(frozen=True)
class SimulationSettings:
    """
    The settings a tilt is simulated with, which a measurement states as its setup.
    """

    engine_version: str
    """
    The version of the physics engine.
    """

    engine: str = "Isaac Sim"
    """
    The physics engine that simulates the substance.
    """

    particle_spacing: float = 0.002
    """
    The distance, in metres, between neighbouring particles at rest.
    """

    tilt_rate: float = 5.0
    """
    The constant rate, in degrees per second, at which the container is tilted.
    """

    time_step: float = 1 / 600
    """
    The simulated time, in seconds, the engine advances per step; at 1/200 s particles
    leaked through a 3 mm wall and the liquid settled denser.
    """

    onset_particle_count: int = 10
    """
    How many particles have to be outside the container for the substance to count as
    leaving it.
    """

    fluid_viscosity: float = 0.0000017
    """
    The engine's viscosity setting of the substance; the engine's own setting for water.
    """

    fluid_cohesion: float = 0.01
    """
    The engine's cohesion setting of the substance; the engine's own setting for water.
    """

    fluid_surface_tension: float = 0.0074
    """
    The engine's surface tension setting of the substance; the engine's own setting for
    water.
    """


@dataclass(frozen=True)
class SimulatedTilt:
    """
    One simulated tilt of a filled container until the substance started to leave it.
    """

    onset_tilt: float
    """
    The tilt from upright, in degrees, at which the substance started to leave.
    """

    settled_fill_fraction: float
    """
    The fill fraction the particles took up once they had settled upright.
    """


@dataclass(frozen=True)
class SimulatedFlowOnset:
    """
    Every simulated tilt of one container, filled alike and tilted toward one side.
    """

    container: flow_onset.ContainerModel
    """
    The kind of container tilted.
    """

    gravity_azimuth: float
    """
    The direction, in degrees about the container's own up axis, toward which it was
    tilted.
    """

    settings: SimulationSettings
    """
    The settings every run was simulated with.
    """

    runs: list[SimulatedTilt] = field(default_factory=list)
    """
    Every run, in the order they were simulated.
    """

    def measurement(self) -> flow_onset.FlowOnsetTiltMeasurement:
        """
        :return: The measurement the runs make: the onset over every run, at the fill
            the particles settled to on average.
        :raises NoSimulatedTilt: When there is no run.
        """
        if not self.runs:
            raise NoSimulatedTilt(mesh_file_name=self.container.mesh_file_name)
        onsets = [run.onset_tilt for run in self.runs]
        fills = [run.settled_fill_fraction for run in self.runs]
        return flow_onset.FlowOnsetTiltMeasurement(
            measured_for_container=self.container,
            at_fill_fraction=float(np.mean(fills)),
            has_material_regime=pouring_frame.MaterialRegime.LIQUID,
            gravity_azimuth=self.gravity_azimuth,
            has_measured_range=core.MagnitudeRange(
                lower_bound=min(onsets),
                upper_bound=max(onsets),
                has_unit=action_core.MeasurementUnit.DEGREE,
            ),
            follows_setup=flow_onset.PhysicsSimulationSetup(
                simulation_engine=self.settings.engine,
                simulation_engine_version=self.settings.engine_version,
                particle_spacing=self.settings.particle_spacing,
                tilt_rate=self.settings.tilt_rate,
                time_step=self.settings.time_step,
                run_count=len(self.runs),
                onset_particle_count=self.settings.onset_particle_count,
                fluid_viscosity=self.settings.fluid_viscosity,
                fluid_cohesion=self.settings.fluid_cohesion,
                fluid_surface_tension=self.settings.fluid_surface_tension,
                has_settled_fill_range=core.MagnitudeRange(
                    lower_bound=min(fills),
                    upper_bound=max(fills),
                    has_unit=action_core.MeasurementUnit.UNITLESS,
                ),
            ),
        )
